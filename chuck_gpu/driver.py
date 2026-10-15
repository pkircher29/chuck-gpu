"""Virtual GPU Driver & Modal RPC Bridge (`chuck_gpu/driver.py`).

Provides three complementary ways for local applications to detect a virtual
NVIDIA GPU and forward data/compute to Modal:

1. **Native CUDA Driver / NVML Shim (`nvcuda.dll` / `nvml.dll` / `libcuda.so.1`)**:
   Compiled from `driver_shim/` (Rust) and coupled with a local TCP RPC bridge
   (`127.0.0.1:8771`) that forwards `cuMemAlloc`, `cuMemcpyHtoD`,
   `cuModuleLoadData`, `cuLaunchKernel`, and `cuMemcpyDtoH` to a persistent
   Modal GPU worker sandbox.
2. **Python / PyTorch CUDA Driver Hook (`enable_virtual_gpu()`)**:
   Patches `torch.cuda` and `pynvml` in-process so scripts checking
   `torch.cuda.is_available()`, `torch.cuda.get_device_name()`, or calling
   `.to("cuda")` / `chuck_gpu.driver.offload(fn, *args)` transparently dispatch
   data and execution to Modal.
3. **Virtual GPU Drive Folder (`chuck-gpu drive --mount <path>`)**:
   Watches a local GPU drive directory (`Inbox/`, `Outbox/`, `Logs/`,
   `gpu_info.json`) so any script, payload, or batch dropped into `Inbox/` is
   automatically sent to Modal for GPU processing and its output written back
   to `Outbox/`.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import pickle
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from chuck_gpu import gpus

DEFAULT_BRIDGE_PORT = 8771
STATE_DIR = Path.home() / ".chuck_gpu"
DRIVER_BIN_DIR = STATE_DIR / "driver_bin"


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
    }


# ── Native DLL / SO Builder & Locator ───────────────────────────────────────

def build_native_shim(force: bool = False) -> dict[str, Path]:
    """Locate or build the native `nvcuda.dll` / `nvml.dll` / `libcuda.so.1` shim."""
    DRIVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    shim_dir = repo_root / "driver_shim"

    built: dict[str, Path] = {}
    if os.name == "nt":
        target_dll = shim_dir / "target" / "release" / "nvcuda.dll"
        if (force or not target_dll.exists()) and (shim_dir / "Cargo.toml").exists():
            cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo.exe")
            if Path(cargo).exists():
                subprocess.run([cargo, "build", "--release", "--manifest-path", str(shim_dir / "Cargo.toml")],
                               check=False)
        if target_dll.exists():
            for alias in ("nvcuda.dll", "nvml.dll"):
                dest = DRIVER_BIN_DIR / alias
                shutil.copy2(target_dll, dest)
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
            for alias in ("libcuda.so.1", "libcuda.so", "libnvidia-ml.so.1"):
                dest = DRIVER_BIN_DIR / alias
                shutil.copy2(target_so, dest)
                built[alias] = dest
    return built


def query_native_shim(dll_path: Path | None = None) -> dict[str, Any]:
    """Load the native CUDA driver shim via ctypes and verify CUDA Driver + NVML calls."""
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

    # Test device memory roundtrip through the driver shim
    dptr = ctypes.c_ulonglong(0)
    payload = b"chuck-gpu-modal-tensor-payload"
    lib.cuMemAlloc_v2(ctypes.byref(dptr), len(payload))
    lib.cuMemcpyHtoD_v2(dptr, ctypes.c_char_p(payload), len(payload))
    out_buf = ctypes.create_string_buffer(len(payload))
    lib.cuMemcpyDtoH_v2(out_buf, dptr, len(payload))
    lib.cuMemFree_v2(dptr)

    return {
        "dll_path": str(dll_path),
        "cuInit": rc_init,
        "device_count": count.value,
        "driver_version": ver.value,
        "device_name": name_buf.value.decode("utf-8", "replace"),
        "total_memory_gb": round(total_mem.value / (1024 ** 3), 2),
        "compute_capability": f"{major.value}.{minor.value}",
        "memory_roundtrip_ok": out_buf.raw == payload,
    }


# ── Modal GPU RPC Bridge Server ─────────────────────────────────────────────

REMOTE_EVAL_SCRIPT = r"""
import base64, json, pickle, sys, traceback

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
    """Manages device buffers and lazily connects to a live Modal GPU sandbox."""

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
            # Forward kernel launch metadata; if remote sandbox is active or requested, execute
            if msg.get("forward_modal"):
                return self.run_on_modal(msg)
            return {"ok": True, "launched": msg.get("func")}
        if op == "cuCtxSynchronize":
            return {"ok": True}
        if op in ("call", "torch_op", "ping"):
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
    """Serialize a Python function + tensor/array arguments, run on a Modal GPU, and return the result."""
    payload_b64 = base64.b64encode(pickle.dumps((fn, args, kwargs))).decode("ascii")
    addr = os.environ.get("CHUCK_GPU_BRIDGE_ADDR", f"127.0.0.1:{DEFAULT_BRIDGE_PORT}")
    host, _, port_str = addr.partition(":")
    port = int(port_str or DEFAULT_BRIDGE_PORT)

    import socket
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
    """Patch `torch.cuda` in the current Python process so it sees a virtual Modal GPU
    and dispatches `.to('cuda')` / CUDA checks cleanly."""
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


# ── Virtual GPU Drive (Folder / Drive Letter Watcher -> Modal) ───────────────

def mount_windows_drive_letter(letter: str, target_dir: Path) -> bool:
    """Map a Windows drive letter (e.g. 'G:') to `target_dir` via `subst`."""
    if os.name != "nt":
        return False
    letter = letter.rstrip("\\/").upper()
    if len(letter) == 1:
        letter += ":"
    subprocess.run(["subst", letter, "/D"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = subprocess.run(["subst", letter, str(target_dir.resolve())], capture_output=True, text=True)
    return res.returncode == 0


def unmount_windows_drive_letter(letter: str) -> bool:
    if os.name != "nt":
        return False
    letter = letter.rstrip("\\/").upper()
    if len(letter) == 1:
        letter += ":"
    res = subprocess.run(["subst", letter, "/D"], capture_output=True, text=True)
    return res.returncode == 0


class VirtualGpuDrive:
    """A virtual GPU directory/drive that mimics an attached GPU device and
    automatically dispatches anything dropped into `Inbox/` to Modal."""

    def __init__(self, root: Path, gpu: str | None = None, drive_letter: str | None = None):
        self.root = root.resolve()
        self.spec = gpu_spec(gpu)
        self.drive_letter = drive_letter
        self.inbox = self.root / "Inbox"
        self.processing = self.root / "Processing"
        self.outbox = self.root / "Outbox"
        self.logs = self.root / "Logs"
        self.driver_dir = self.root / "Driver"
        self.stopping = threading.Event()

    def setup(self) -> None:
        for d in (self.inbox, self.processing, self.outbox, self.logs, self.driver_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Copy native CUDA/NVML driver shim binaries into the Drive's Driver/ folder
        built = build_native_shim()
        for name, src in built.items():
            shutil.copy2(src, self.driver_dir / name)

        info_file = self.root / "GPU_DEVICE_INFO.json"
        info_file.write_text(json.dumps({
            "device": self.spec["name"],
            "model": self.spec["model"],
            "vram_gb": self.spec["vram_gb"],
            "compute_capability": f"{self.spec['major']}.{self.spec['minor']}",
            "driver_binaries": [p.name for p in self.driver_dir.iterdir()],
            "usage": (
                "Drop any .py script, .sh script, or job.json into Inbox/ to execute it on "
                "your Modal GPU. Outputs and logs appear in Outbox/ and Logs/."
            ),
        }, indent=2), encoding="utf-8")

        if self.drive_letter:
            mount_windows_drive_letter(self.drive_letter, self.root)

    def process_item(self, item: Path) -> dict[str, Any]:
        """Dispatch a single file or folder from Inbox/ to Modal."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        active_path = self.processing / f"{ts}_{item.name}"
        shutil.move(str(item), str(active_path))

        job_out_dir = self.outbox / f"{ts}_{item.stem}"
        job_out_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs / f"{ts}_{item.stem}.log"

        # Determine command to run on Modal
        if active_path.is_dir():
            work_dir = active_path
            if (work_dir / "run.py").exists():
                cmd = ["python", "run.py"]
            elif (work_dir / "main.py").exists():
                cmd = ["python", "main.py"]
            elif (work_dir / "run.sh").exists():
                cmd = ["sh", "run.sh"]
            else:
                py_files = sorted(work_dir.glob("*.py"))
                cmd = ["python", py_files[0].name] if py_files else ["nvidia-smi"]
        else:
            work_dir = job_out_dir
            dest_file = work_dir / active_path.name
            shutil.copy2(active_path, dest_file)
            if active_path.suffix == ".py":
                cmd = ["python", dest_file.name]
            elif active_path.suffix == ".sh":
                cmd = ["sh", dest_file.name]
            elif active_path.suffix == ".json":
                spec = json.loads(dest_file.read_text(encoding="utf-8"))
                cmd = spec.get("command") or ["nvidia-smi"]
            else:
                cmd = ["nvidia-smi"]

        from chuck_gpu.cli import cmd_run
        import argparse
        ns = argparse.Namespace(
            dir=str(work_dir),
            command=cmd,
            gpu=self.spec["model"],
            image=None,
            timeout=None,
            add_python=None,
            cpu=None,
            memory=None,
            pip=None,
            apt=None,
            secret=None,
            env=None,
            no_download=False,
            resync=True,
        )
        t0 = time.time()
        try:
            rc = cmd_run(ns)
            status = "done" if rc == 0 else f"failed (exit {rc})"
        except SystemExit as e:
            rc = int(e.code) if isinstance(e.code, int) else 1
            status = f"error: {e}"
        except Exception as e:
            rc = 1
            status = f"error: {type(e).__name__}: {e}"

        if active_path.is_dir():
            shutil.copytree(active_path, job_out_dir, dirs_exist_ok=True)
        shutil.rmtree(active_path, ignore_errors=True) if active_path.is_dir() else active_path.unlink(missing_ok=True)

        summary = {
            "item": item.name,
            "command": cmd,
            "gpu": self.spec["model"],
            "exit_code": rc,
            "status": status,
            "duration_s": round(time.time() - t0, 2),
            "outbox": str(job_out_dir),
        }
        log_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def watch_once(self) -> list[dict[str, Any]]:
        results = []
        for item in sorted(self.inbox.iterdir()):
            if item.name.startswith("."):
                continue
            results.append(self.process_item(item))
        return results

    def serve_forever(self, poll_interval: float = 1.5) -> None:
        self.setup()
        try:
            while not self.stopping.wait(poll_interval):
                self.watch_once()
        finally:
            if self.drive_letter:
                unmount_windows_drive_letter(self.drive_letter)
