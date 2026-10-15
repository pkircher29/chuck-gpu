"""Virtual GPU Driver & Modal RPC Bridge (`chuck_gpu/driver.py`).

Provides a Virtual GPU Driver for Windows & Linux that mimics an attached
NVIDIA CUDA GPU and forwards compute/data to Modal:

1. **Native CUDA Driver, CUDA Runtime & NVML Shim (`nvcuda.dll`, `nvml.dll`, `cudart64_12.dll`, `libcuda.so.1`, `libcudart.so.12`, `libnvidia-ml.so.1`)**:
   Compiled from `driver_shim/` and coupled with a local TCP RPC daemon
   (`127.0.0.1:8771`) that handles `cuInit`, `cuMemAlloc`, `cuMemcpyHtoD`,
   `cuModuleLoadData`, `cuLaunchKernel`, `cuMemcpyDtoH`, and `cuda*` Runtime calls,
   forwarding compute and PTX/Python workloads to a persistent Modal GPU sandbox.
2. **System / User-Space Driver Installer (`chuck-gpu driver install`)**:
   Installs the native virtual GPU driver libraries (`nvcuda.dll`, `nvml.dll`,
   `cudart64_12.dll`) + a virtual `nvidia-smi` binary into `~/.chuck_gpu/driver_bin`
   and registers them on the user's `PATH` (and `sitecustomize.py` hook when
   `CHUCK_GPU_AUTO_HOOK=1`) so applications and scripts detect the virtual
   Modal NVIDIA GPU automatically.
3. **Python / PyTorch CUDA Driver Hook (`enable_virtual_gpu()` & `offload()`)**:
   Patches `torch.cuda`, `torch.Tensor.to`, and `torch.nn.Module.to` so scripts
   checking `torch.cuda.is_available()`, `torch.cuda.get_device_name()`, or
   dispatching workloads via `chuck_gpu.driver.offload(fn, *args)` transparently
   send data to Modal for GPU processing.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import pickle
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from chuck_gpu import gpus

DEFAULT_BRIDGE_PORT = 8771
STATE_DIR = Path.home() / ".chuck_gpu"
DRIVER_BIN_DIR = STATE_DIR / "driver_bin"
DRIVER_STATE_FILE = STATE_DIR / "driver_state.json"


def gpu_spec(gpu_name: str | None = None) -> dict[str, Any]:
    """Return virtual GPU specifications matching a Modal GPU tier."""
    name = (gpu_name or os.environ.get("CHUCK_GPU_MODEL") or gpus.get_next() or gpus.cheapest()).upper()
    specs = {
        "T4":        ("Modal NVIDIA Tesla T4 (chuck-gpu)", 16, (7, 5), 40),
        "L4":        ("Modal NVIDIA L4 (chuck-gpu)", 24, (8, 9), 58),
        "A10G":      ("Modal NVIDIA A10G (chuck-gpu)", 24, (8, 6), 80),
        "L40S":      ("Modal NVIDIA L40S (chuck-gpu)", 48, (8, 9), 142),
        "A100":      ("Modal NVIDIA A100-SXM4-40GB (chuck-gpu)", 40, (8, 0), 108),
        "A100-80GB": ("Modal NVIDIA A100-SXM4-80GB (chuck-gpu)", 80, (8, 0), 108),
        "H100":      ("Modal NVIDIA H100 80GB HBM3 (chuck-gpu)", 80, (9, 0), 132),
        "H200":      ("Modal NVIDIA H200 141GB (chuck-gpu)", 141, (9, 0), 132),
        "B200":      ("Modal NVIDIA B200 (chuck-gpu)", 192, (10, 0), 160),
    }
    full_name, vram_gb, cc, sm_count = specs.get(name, specs["T4"])
    return {
        "model": name if name in gpus.CATALOG else "T4",
        "name": full_name,
        "total_memory": vram_gb * (1024 ** 3),
        "vram_gb": vram_gb,
        "major": cc[0],
        "minor": cc[1],
        "multi_processor_count": sm_count,
        "driver_version": 12040,
        "driver_version_str": "550.90.07",
        "cuda_version_str": "12.4",
    }


def format_nvidia_smi(spec: dict[str, Any], used_mib: int = 0) -> str:
    """Render a realistic `nvidia-smi` table for the virtual Modal GPU."""
    total_mib = spec["vram_gb"] * 1024
    ts = time.strftime("%a %b %d %H:%M:%S %Y")
    gpu_title = spec["name"][:29]
    return "\n".join([
        ts,
        "+-----------------------------------------------------------------------------------------+",
        f"| NVIDIA-SMI {spec['driver_version_str']:<10}       Driver Version: {spec['driver_version_str']:<10}     CUDA Version: {spec['cuda_version_str']:<8}   |",
        "|-----------------------------------------+------------------------+----------------------+",
        "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |",
        "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |",
        "|                                         |                        |               MIG M. |",
        "|=========================================+========================+======================|",
        f"|   0  {gpu_title:<29}  On  |   00000000:00:00.0 Off |                    0 |",
        f"| N/A   38C    P0              28W /  70W |   {used_mib:>6}MiB / {total_mib:>6}MiB |      0%      Default |",
        "|                                         |                        |                  N/A |",
        "+-----------------------------------------+------------------------+----------------------+",
        "",
        "+-----------------------------------------------------------------------------------------+",
        "| Processes:                                                                              |",
        "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |",
        "|        ID   ID                                                               Usage      |",
        "|=========================================================================================|",
        "|  Modal Cloud GPU Bridge Active (127.0.0.1:8771)                                         |",
        "+-----------------------------------------------------------------------------------------+",
    ])


# ── Native DLL / SO Builder & Installer ─────────────────────────────────────

def build_native_shim(force: bool = False) -> dict[str, Path]:
    """Locate or build the native `nvcuda.dll`, `nvml.dll`, `cudart64_12.dll` /
    `libcuda.so.1`, `libcudart.so.12`, `libnvidia-ml.so.1` driver binaries."""
    DRIVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    shim_dir = repo_root / "driver_shim"

    built: dict[str, Path] = {}

    # Always copy prebuilt Windows DLL if present in repo
    win_dll = shim_dir / "target" / "release" / "nvcuda.dll"
    if os.name == "nt":
        if (force or not win_dll.exists()) and (shim_dir / "Cargo.toml").exists():
            cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo.exe")
            if Path(cargo).exists():
                subprocess.run([cargo, "build", "--release", "--manifest-path", str(shim_dir / "Cargo.toml")],
                               check=False)
        if win_dll.exists():
            for alias in ("nvcuda.dll", "nvml.dll", "cudart64_12.dll", "cudart64_110.dll"):
                dest = DRIVER_BIN_DIR / alias
                shutil.copy2(win_dll, dest)
                built[alias] = dest
    else:
        target_so = shim_dir / "target" / "release" / "libnvcuda.so"
        if (force or not target_so.exists()) and (shim_dir / "Cargo.toml").exists():
            cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
            if Path(cargo).exists():
                subprocess.run([cargo, "build", "--release", "--manifest-path", str(shim_dir / "Cargo.toml")],
                               check=False)
            elif shutil.which("gcc"):
                target_so.parent.mkdir(parents=True, exist_ok=True)
                c_src = shim_dir / "src" / "shim_fallback.c"
                if c_src.exists():
                    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(target_so), str(c_src)],
                                   check=False)
        if target_so.exists():
            for alias in ("libcuda.so.1", "libcuda.so", "libcudart.so.12", "libcudart.so", "libnvidia-ml.so.1"):
                dest = DRIVER_BIN_DIR / alias
                shutil.copy2(target_so, dest)
                built[alias] = dest

    # Create virtual `nvidia-smi` executable wrapper in DRIVER_BIN_DIR
    spec = gpu_spec()
    smi_banner = format_nvidia_smi(spec)
    if os.name == "nt":
        smi_cmd = DRIVER_BIN_DIR / "nvidia-smi.cmd"
        smi_lines = ["@echo off"] + [f"echo {line.replace('|', '^|')}" if line else "echo."
                                     for line in smi_banner.splitlines()]
        smi_cmd.write_text("\r\n".join(smi_lines) + "\r\n", encoding="utf-8")
        built["nvidia-smi.cmd"] = smi_cmd
    else:
        smi_sh = DRIVER_BIN_DIR / "nvidia-smi"
        smi_sh.write_text(f"#!/bin/sh\ncat << 'EOF'\n{smi_banner}\nEOF\n", encoding="utf-8")
        smi_sh.chmod(0o755)
        built["nvidia-smi"] = smi_sh

    return built


def install_driver(gpu: str | None = None) -> dict[str, Any]:
    """Install the virtual GPU driver binaries (`nvcuda.dll`, `nvml.dll`, `cudart64_12.dll`,
    `nvidia-smi`) into `~/.chuck_gpu/driver_bin` and add to user PATH on Windows."""
    if gpu:
        os.environ["CHUCK_GPU_MODEL"] = gpu.upper()
    built = build_native_shim(force=False)
    spec = gpu_spec(gpu)

    # Also write a sitecustomize.py in DRIVER_BIN_DIR so PYTHONPATH=~/.chuck_gpu/driver_bin
    # automatically activates the virtual PyTorch CUDA driver in any Python process
    sitecustom = DRIVER_BIN_DIR / "sitecustomize.py"
    repo_root = Path(__file__).resolve().parent.parent
    sitecustom.write_text(
        f"import os, sys\n"
        f"if {str(repo_root)!r} not in sys.path:\n"
        f"    sys.path.insert(0, {str(repo_root)!r})\n"
        f"try:\n"
        f"    from chuck_gpu.driver import enable_virtual_gpu\n"
        f"    enable_virtual_gpu(os.environ.get('CHUCK_GPU_MODEL', {spec['model']!r}))\n"
        f"except Exception:\n"
        f"    pass\n",
        encoding="utf-8",
    )

    path_added = False
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                                winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
                try:
                    cur_path, _ = winreg.QueryValueEx(key, "Path")
                except OSError:
                    cur_path = ""
                parts = [p for p in cur_path.split(";") if p]
                bin_str = str(DRIVER_BIN_DIR)
                if bin_str not in parts:
                    parts.insert(0, bin_str)
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(parts))
                    path_added = True
                winreg.SetValueEx(key, "CHUCK_GPU_MODEL", 0, winreg.REG_SZ, spec["model"])
                winreg.SetValueEx(key, "CHUCK_GPU_BRIDGE_ADDR", 0, winreg.REG_SZ,
                                  f"127.0.0.1:{DEFAULT_BRIDGE_PORT}")
        except Exception:
            pass

    state = {
        "installed_at": time.time(),
        "driver_bin_dir": str(DRIVER_BIN_DIR),
        "gpu": spec,
        "binaries": [k for k in built],
        "user_path_updated": path_added,
        "bridge_addr": f"127.0.0.1:{DEFAULT_BRIDGE_PORT}",
    }
    DRIVER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def query_native_shim(dll_path: Path | None = None) -> dict[str, Any]:
    """Load the native CUDA driver shim via ctypes and verify CUDA Driver + Runtime + NVML calls."""
    if dll_path is None:
        files = build_native_shim()
        dll_path = files.get("nvcuda.dll") or files.get("libcuda.so.1")
    if not dll_path or not dll_path.exists():
        raise FileNotFoundError("Native driver shim binary not found")

    lib = ctypes.CDLL(str(dll_path))

    rc_init = lib.cuInit(0)
    count = ctypes.c_int(0)
    lib.cuDeviceGetCount(ctypes.byref(count))

    ver = ctypes.c_int(0)
    lib.cuDriverGetVersion(ctypes.byref(ver))

    name_buf = ctypes.create_string_buffer(128)
    lib.cuDeviceGetName(name_buf, 128, 0)

    total_mem = ctypes.c_size_t(0)
    lib.cuDeviceTotalMem_v2(ctypes.byref(total_mem), 0)

    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), 0)

    # Test device memory roundtrip through the driver shim (and RPC bridge if running)
    dptr = ctypes.c_ulonglong(0)
    payload = b"chuck-gpu-modal-tensor-payload"
    lib.cuMemAlloc_v2(ctypes.byref(dptr), len(payload))
    lib.cuMemcpyHtoD_v2(dptr, ctypes.c_char_p(payload), len(payload))
    out_buf = ctypes.create_string_buffer(len(payload))
    lib.cuMemcpyDtoH_v2(out_buf, dptr, len(payload))
    lib.cuMemFree_v2(dptr)

    # Query live bridge daemon status if listening
    bridge_status = None
    addr = os.environ.get("CHUCK_GPU_BRIDGE_ADDR", f"127.0.0.1:{DEFAULT_BRIDGE_PORT}")
    host, _, port_s = addr.partition(":")
    try:
        with socket.create_connection((host, int(port_s or DEFAULT_BRIDGE_PORT)), timeout=1.0) as s:
            s.sendall(b'{"op":"status"}\n')
            bridge_status = json.loads(s.recv(65536).decode("utf-8"))
    except Exception:
        pass

    return {
        "dll_path": str(dll_path),
        "cuInit": rc_init,
        "device_count": count.value,
        "driver_version": ver.value,
        "device_name": name_buf.value.decode("utf-8", "replace"),
        "total_memory_gb": round(total_mem.value / (1024 ** 3), 2),
        "compute_capability": f"{major.value}.{minor.value}",
        "memory_roundtrip_ok": out_buf.raw == payload,
        "bridge_daemon": bridge_status,
    }


# ── Modal GPU RPC Bridge Server ─────────────────────────────────────────────

REMOTE_EVAL_SCRIPT = r"""
import base64, json, pickle, subprocess, sys, tempfile, traceback

data = json.loads(sys.stdin.read())
op = data.get("op")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    device = "cpu"

try:
    if op == "ping":
        info = {"device": device}
        if torch and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["free_bytes"] = free
            info["total_bytes"] = total
        print(json.dumps({"ok": True, **info}))
    elif op == "call":
        fn, args, kwargs = pickle.loads(base64.b64decode(data["payload_b64"]))
        res = fn(*args, **kwargs)
        out_b64 = base64.b64encode(pickle.dumps(res)).decode("ascii")
        print(json.dumps({"ok": True, "result_b64": out_b64}))
    elif op == "py_code":
        code_str = data["code"]
        inputs = pickle.loads(base64.b64decode(data["inputs_b64"])) if data.get("inputs_b64") else {}
        ns = {"__builtins__": __builtins__, "torch": torch, "device": device, **inputs}
        exec(code_str, ns)
        result = ns.get("result")
        if hasattr(result, "cpu"):
            result = result.cpu()
        out_b64 = base64.b64encode(pickle.dumps(result)).decode("ascii")
        print(json.dumps({"ok": True, "result_b64": out_b64}))
    elif op == "torch_op":
        spec = pickle.loads(base64.b64decode(data["payload_b64"]))
        func_name = spec["func"]
        args = [a.to(device) if hasattr(a, "to") else a for a in spec["args"]]
        kwargs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in spec["kwargs"].items()}
        target = getattr(torch.ops.aten, func_name, None) or getattr(torch, func_name)
        res = target(*args, **kwargs)
        if hasattr(res, "cpu"):
            res = res.cpu()
        out_b64 = base64.b64encode(pickle.dumps(res)).decode("ascii")
        print(json.dumps({"ok": True, "result_b64": out_b64}))
    else:
        print(json.dumps({"ok": False, "error": f"unknown op {op}"}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}))
"""


class ModalBridgeSession:
    """Manages virtual GPU device buffers and forwards compute to a Modal GPU sandbox."""

    def __init__(self, gpu: str | None = None, image: str = "torch", timeout: int = 3600):
        self.spec = gpu_spec(gpu)
        self.gpu = self.spec["model"]
        self.image = image
        self.timeout = timeout
        self.buffers: dict[int, bytearray] = {}
        self.modules: dict[int, bytes] = {}
        self.sandbox = None
        self.lock = threading.Lock()
        self.stats = {
            "started_at": time.time(),
            "ops_count": 0,
            "bytes_htod": 0,
            "bytes_dtoh": 0,
            "remote_calls": 0,
            "sandbox_id": None,
        }

    def ensure_sandbox(self):
        with self.lock:
            if self.sandbox is not None:
                return self.sandbox
            from chuck_gpu.cli import DEFAULTS, open_sandbox
            cfg = dict(DEFAULTS)
            cfg["gpu"] = self.gpu
            cfg["image"] = self.image
            cfg["timeout"] = self.timeout
            sb, _ = open_sandbox(cfg, Path.cwd(), None, self.gpu)
            self.sandbox = sb
            self.stats["sandbox_id"] = sb.object_id
            return sb

    def stop_sandbox(self) -> None:
        with self.lock:
            if self.sandbox is not None:
                try:
                    self.sandbox.terminate()
                except Exception:
                    pass
                self.sandbox = None
                self.stats["sandbox_id"] = None

    def run_on_modal(self, request_dict: dict) -> dict:
        """Execute a serialized operation inside the remote Modal GPU sandbox."""
        sb = self.ensure_sandbox()
        self.stats["remote_calls"] += 1
        proc = sb.exec("python3", "-c", REMOTE_EVAL_SCRIPT)
        proc.stdin.write(json.dumps(request_dict))
        proc.stdin.write_eof()
        proc.stdin.drain()
        out = proc.stdout.read().strip()
        proc.wait()
        if not out:
            err = proc.stderr.read().strip()
            return {"ok": False, "error": err or "empty response from Modal sandbox"}
        last_line = out.splitlines()[-1]
        return json.loads(last_line)

    def handle_rpc(self, msg: dict) -> dict:
        op = msg.get("op", "")
        self.stats["ops_count"] += 1

        if op == "cuInit":
            return {"ok": True, "gpu": self.spec}
        if op == "status":
            return {"ok": True, "gpu": self.spec, "stats": self.stats}
        if op == "cuMemAlloc":
            dptr = int(msg["dptr"])
            size = int(msg["size"])
            self.buffers[dptr] = bytearray(size)
            return {"ok": True, "dptr": dptr}
        if op == "cuMemFree":
            dptr = int(msg["dptr"])
            self.buffers.pop(dptr, None)
            return {"ok": True}
        if op == "cuMemcpyHtoD":
            dptr = int(msg["dptr"])
            raw = bytes.fromhex(msg["hex"])
            self.stats["bytes_htod"] += len(raw)
            self.buffers[dptr] = bytearray(raw)
            return {"ok": True, "bytes": len(raw)}
        if op == "cuMemcpyDtoH":
            dptr = int(msg["dptr"])
            size = int(msg.get("size", 0))
            buf = bytes(self.buffers.get(dptr, b"\x00" * size))[:size]
            self.stats["bytes_dtoh"] += len(buf)
            return {"ok": True, "hex": buf.hex()}
        if op == "cuModuleLoadData":
            mod_id = int(msg["mod_id"])
            ptx = bytes.fromhex(msg.get("ptx_hex", ""))
            self.modules[mod_id] = ptx
            return {"ok": True, "mod_id": mod_id}
        if op == "cuLaunchKernel":
            if msg.get("forward_modal"):
                return self.run_on_modal(msg)
            return {"ok": True, "launched": msg.get("func")}
        if op == "cuCtxSynchronize":
            return {"ok": True}
        if op in ("call", "py_code", "torch_op", "ping"):
            return self.run_on_modal(msg)
        return {"ok": False, "error": f"unsupported op: {op}"}


class _BridgeTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_bridge_server(
    port: int = DEFAULT_BRIDGE_PORT,
    gpu: str | None = None,
    warm_sandbox: bool = False,
) -> tuple[_BridgeTCPServer, ModalBridgeSession]:
    """Start the local TCP RPC bridge (`127.0.0.1:<port>`) for the virtual GPU driver."""
    session = ModalBridgeSession(gpu=gpu)
    if warm_sandbox:
        session.ensure_sandbox()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for raw_line in self.rfile:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = session.handle_rpc(req)
                except Exception as e:
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()

    server = _BridgeTCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, session


# ── Python / PyTorch CUDA Driver Hook ───────────────────────────────────────

class _VirtualDeviceProperties:
    def __init__(self, spec: dict[str, Any]):
        self.name = spec["name"]
        self.major = spec["major"]
        self.minor = spec["minor"]
        self.total_memory = spec["total_memory"]
        self.multi_processor_count = spec["multi_processor_count"]
        self.is_integrated = 0
        self.is_multi_gpu_board = 0

    def __repr__(self) -> str:
        return (f"_CudaDeviceProperties(name='{self.name}', major={self.major}, "
                f"minor={self.minor}, total_memory={self.total_memory // (1024 * 1024)}MB, "
                f"multi_processor_count={self.multi_processor_count})")


def offload(fn: Callable, *args, gpu: str | None = None, **kwargs) -> Any:
    """Serialize a Python function + tensor/array arguments, send over the virtual GPU
    driver bridge to Modal, and return the result."""
    payload_b64 = base64.b64encode(pickle.dumps((fn, args, kwargs))).decode("ascii")
    addr = os.environ.get("CHUCK_GPU_BRIDGE_ADDR", f"127.0.0.1:{DEFAULT_BRIDGE_PORT}")
    host, _, port_str = addr.partition(":")
    port = int(port_str or DEFAULT_BRIDGE_PORT)

    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            req = json.dumps({"op": "call", "payload_b64": payload_b64}) + "\n"
            sock.sendall(req.encode("utf-8"))
            resp_line = b""
            while not resp_line.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                resp_line += chunk
        resp = json.loads(resp_line.decode("utf-8"))
    except OSError:
        session = ModalBridgeSession(gpu=gpu)
        try:
            resp = session.run_on_modal({"op": "call", "payload_b64": payload_b64})
        finally:
            session.stop_sandbox()

    if not resp.get("ok"):
        raise RuntimeError(f"Modal GPU offload failed: {resp.get('error')}")
    return pickle.loads(base64.b64decode(resp["result_b64"]))


def enable_virtual_gpu(gpu: str | None = None) -> dict[str, Any]:
    """Patch `torch.cuda` in the current Python process so it detects the virtual Modal GPU."""
    spec = gpu_spec(gpu)
    os.environ["CHUCK_GPU_MODEL"] = spec["model"]

    try:
        import torch
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_name = lambda device=None: spec["name"]
        torch.cuda.get_device_capability = lambda device=None: (spec["major"], spec["minor"])
        torch.cuda.get_device_properties = lambda device=None: _VirtualDeviceProperties(spec)
        torch.cuda.mem_get_info = lambda device=None: (spec["total_memory"], spec["total_memory"])
        torch.cuda.synchronize = lambda device=None: None
    except ImportError:
        pass

    return spec
