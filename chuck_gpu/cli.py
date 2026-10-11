"""chuck-gpu - run any command on a Modal cloud GPU as if it were local.

    chuck-gpu login                    # once: connect your Modal account
    chuck-gpu run -- python train.py --epochs 10
    chuck-gpu run --gpu A100 --image torch -- python finetune.py
    chuck-gpu smi                      # nvidia-smi on the remote GPU

What happens on `run`:
  1. the current directory is synced (incrementally) to a persistent Modal
     Volume, one workspace per local project directory;
  2. a GPU sandbox starts from the chosen image with that workspace mounted
     as its working directory, and the command runs there with stdout/stderr
     streamed back live;
  3. every file the command created or changed is copied back into the local
     directory, so outputs land exactly where a local run would put them.

Project defaults can live in ./chuck-gpu.toml (see README).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import sys
import threading
import time
import tomllib
from pathlib import Path

APP_NAME = "chuck-gpu"
VOLUME_NAME = "chuck-gpu-workspaces"
MOUNT = "/ws"
STATE_DIR = Path.home() / ".chuck_gpu"

IMAGES = {
    # name: (registry image, add_python) — add_python None means the image has one
    "torch": ("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", None),
    "torch-devel": ("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", None),
    "cuda": ("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", "3.11"),
    "cuda-devel": ("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", "3.11"),
}
from chuck_gpu.gpus import GPUS  # noqa: E402  (catalog + prices live in gpus.py)

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    "*.pyc", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    ".DS_Store", "Thumbs.db", ".chuck_gpu",
]

DEFAULTS = {
    "gpu": None,  # None = the dashboard pick, else the cheapest (see gpus.py)
    "image": "torch",
    "add_python": None,
    "pip": [],
    "apt": [],
    "requirements": None,
    "exclude": [],
    "env": {},
    "secrets": [],
    "timeout": 3600,
    "cpu": None,
    "memory": None,
    "download": True,
    "max_upload_gb": 50,
}


# ── config ──────────────────────────────────────────────────────────────────

def load_config(root: Path) -> dict:
    cfg = dict(DEFAULTS)
    path = root / "chuck-gpu.toml"
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        unknown = set(data) - set(DEFAULTS)
        if unknown:
            sys.exit(f"chuck-gpu.toml: unknown keys {sorted(unknown)}")
        cfg.update(data)
    return cfg


def workspace_key(root: Path) -> str:
    resolved = str(root.resolve()).lower()
    digest = hashlib.sha1(resolved.encode()).hexdigest()[:10]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in root.resolve().name) or "root"
    return f"{safe}-{digest}"


# ── local file walk ─────────────────────────────────────────────────────────

def read_ignore(root: Path) -> list[str]:
    path = root / ".mgpuignore"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def excluded(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pat in patterns:
        pat = pat.rstrip("/")
        if "/" in pat:
            if fnmatch.fnmatch(rel, pat) or rel.startswith(pat + "/"):
                return True
        elif any(fnmatch.fnmatch(p, pat) for p in parts):
            return True
    return False


def walk(root: Path, patterns: list[str]) -> dict[str, tuple[int, int]]:
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = [d for d in dirnames
                       if not excluded(f"{rel_dir}/{d}".lstrip("/"), patterns)]
        for name in filenames:
            rel = f"{rel_dir}/{name}".lstrip("/")
            if excluded(rel, patterns):
                continue
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            files[rel] = (st.st_size, st.st_mtime_ns)
    return files


# ── manifest (what this machine already uploaded) ───────────────────────────

def manifest_path(key: str) -> Path:
    return STATE_DIR / "manifests" / f"{key}.json"


def load_manifest(key: str) -> dict:
    try:
        return json.loads(manifest_path(key).read_text())
    except (OSError, ValueError):
        return {}


def save_manifest(key: str, manifest: dict) -> None:
    p = manifest_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, p)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def log(msg: str) -> None:
    print(f"\033[36m[chuck-gpu]\033[0m {msg}", file=sys.stderr, flush=True)


# ── modal plumbing ──────────────────────────────────────────────────────────

def build_image(cfg: dict, root: Path):
    import modal
    name = cfg["image"]
    if name in IMAGES:
        ref, add_python = IMAGES[name]
    else:
        ref, add_python = name, cfg["add_python"]
    image = modal.Image.from_registry(ref, add_python=add_python) if add_python \
        else modal.Image.from_registry(ref)
    if name not in IMAGES:
        # an app image's ENTRYPOINT would wrap the sandbox's own command
        image = image.entrypoint([])
    if cfg["apt"]:
        image = image.apt_install(*cfg["apt"])
    if cfg["requirements"]:
        req = (root / cfg["requirements"]).resolve()
        if not req.exists():
            sys.exit(f"requirements file not found: {req}")
        image = image.pip_install_from_requirements(str(req))
    if cfg["pip"]:
        image = image.pip_install(*cfg["pip"])
    return image


def sync_up(vol, key: str, root: Path, cfg: dict, force: bool, job=None) -> None:
    patterns = DEFAULT_EXCLUDES + list(cfg["exclude"]) + read_ignore(root)
    local = walk(root, patterns)
    manifest = {} if force else load_manifest(key)
    changed = [rel for rel, sig in local.items() if manifest.get(rel) != list(sig)]
    total = sum(local[r][0] for r in changed)
    limit = float(cfg["max_upload_gb"]) * 1024 ** 3
    if total > limit:
        sys.exit(f"refusing to upload {human(total)} (> max_upload_gb={cfg['max_upload_gb']}); "
                 "add excludes to .mgpuignore or raise max_upload_gb in chuck-gpu.toml")
    if not changed:
        log(f"workspace up to date ({len(local)} files)")
        return
    log(f"uploading {len(changed)} changed file(s), {human(total)} "
        f"({len(local) - len(changed)} unchanged)")
    t0 = time.time()
    sent = 0
    if job:
        job.update(upload={"files": len(changed), "bytes": total, "sent": 0})
    # batches keep a crash from losing everything already sent
    batch_bytes = 512 * 1024 ** 2
    i = 0
    while i < len(changed):
        chunk, size = [], 0
        while i < len(changed) and (not chunk or size < batch_bytes):
            chunk.append(changed[i])
            size += local[changed[i]][0]
            i += 1
        with vol.batch_upload(force=True) as batch:
            for rel in chunk:
                batch.put_file(str(root / rel), f"/{key}/{rel}")
        for rel in chunk:
            manifest[rel] = list(local[rel])
        save_manifest(key, manifest)
        sent += size
        if job:
            job.update(upload={"files": len(changed), "bytes": total, "sent": sent})
    log(f"upload done in {time.time() - t0:.1f}s")


REMOTE_OUT, REMOTE_RC = "/tmp/.chuck_gpu-out", "/tmp/.chuck_gpu-rc"


def emit(text: str, out, job=None) -> None:
    try:
        out.write(text)
        out.flush()
    except UnicodeEncodeError:  # console that can't show e.g. tqdm's block characters
        out.write(text.encode(out.encoding or "ascii", "replace").decode(out.encoding or "ascii"))
        out.flush()
    if job:
        job.output(text)


def pump(stream, out, job=None, counter: list | None = None) -> None:
    """Copy a remote stream locally. Any failure is recorded in `counter[1]`
    so the caller can switch to reading the remote copy of the output."""
    try:
        for chunk in stream:
            emit(chunk, out, job)
            if counter is not None:
                counter[0] += len(chunk.encode("utf-8"))
    except Exception as e:
        if counter is not None:
            counter[1] = f"{type(e).__name__}: {e}"


def sync_down(vol, key: str, root: Path, changed: list[tuple[str, int]]) -> None:
    if not changed:
        log("no output files changed")
        return
    manifest = load_manifest(key)
    for rel, _size in changed:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".chuck_gpu-part")
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(f"/{key}/{rel}"):
                f.write(chunk)
        os.replace(tmp, dest)
        st = dest.stat()
        # the remote already has this exact file — don't re-upload it next run
        manifest[rel] = [st.st_size, st.st_mtime_ns]
    save_manifest(key, manifest)


def open_sandbox(cfg: dict, root: Path, key: str | None, gpu: str):
    import modal
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    secrets = [modal.Secret.from_name(s) for s in cfg["secrets"]]
    kwargs = dict(app=app, image=build_image(cfg, root), gpu=gpu,
                  timeout=int(cfg["timeout"]), secrets=secrets,
                  env={k: str(v) for k, v in cfg["env"].items()} or None)
    if key:
        kwargs["volumes"] = {MOUNT: vol}
        kwargs["workdir"] = f"{MOUNT}/{key}"
    if cfg["cpu"]:
        kwargs["cpu"] = float(cfg["cpu"])
    if cfg["memory"]:
        kwargs["memory"] = int(cfg["memory"])
    # keep the sandbox alive with an idle process; work happens via exec
    return modal.Sandbox.create("sleep", "infinity", **kwargs), vol


def run_logged(sb, cmd: str, job=None) -> int:
    """Run a shell command, streaming its output (stderr merged into stdout).

    The output is also teed to a file in the sandbox. Modal's live exec stream
    can drop on a long job; if it does, follow that file instead, so a
    multi-hour run never goes silent while it is still working."""
    # inner subshell: a command that calls `exit` still gets its status recorded
    wrapped = (f"( ( {cmd}\n); echo $? > {REMOTE_RC} ) 2>&1 | tee {REMOTE_OUT}; "
               f"exit \"$(cat {REMOTE_RC} 2>/dev/null || echo 1)\"")
    proc = sb.exec("sh", "-c", wrapped)
    counter = [0, None]  # bytes received, stream error
    t = threading.Thread(target=pump, args=(proc.stdout, sys.stdout, job, counter), daemon=True)
    t.start()
    warned, grace_until = False, None
    while (code := proc.poll()) is None:
        t.join(timeout=2)
        if t.is_alive():
            continue
        if counter[1] is None and not warned:
            # clean end of stream: the exit status is usually a moment behind
            grace_until = grace_until or time.time() + 10
            if time.time() < grace_until:
                time.sleep(1)
                continue
        if not warned:
            why = counter[1] or "stream ended early"
            log(f"live output dropped ({why}); following the remote log instead")
            if job:
                job.update(detail=f"live output dropped ({why[:200]}); following remote log")
            warned = True
        counter[0] = follow_remote(sb, counter[0], job)
        time.sleep(2)
    t.join(timeout=5)
    if warned:
        follow_remote(sb, counter[0], job)  # whatever arrived after the last poll
    return code


def follow_remote(sb, offset: int, job=None) -> int:
    """Print what REMOTE_OUT gained past byte `offset`; return the new offset."""
    try:
        p = sb.exec("tail", "-c", f"+{offset + 1}", REMOTE_OUT)
        data = p.stdout.read()
        p.wait()
    except Exception:
        return offset
    if data:
        emit(data, sys.stdout, job)
        offset += len(data.encode("utf-8"))
    return offset


def run_exec(sb, argv: list[str], workdir: str | None = None, job=None) -> int:
    proc = sb.exec(*argv, workdir=workdir)
    threads = [threading.Thread(target=pump, args=(proc.stdout, sys.stdout, job), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, sys.stderr, job), daemon=True)]
    for t in threads:
        t.start()
    code = proc.wait()
    for t in threads:
        t.join(timeout=5)
    return code


def capture(sb, argv: list[str]) -> str:
    proc = sb.exec(*argv)
    out = proc.stdout.read()
    proc.wait()
    return out


# ── dashboard ───────────────────────────────────────────────────────────────

def ensure_ui() -> None:
    """Start the dashboard in the background if it isn't already up."""
    import socket
    import subprocess
    from chuck_gpu.ui import load_settings
    settings = load_settings()
    if not settings["auto_ui"]:
        return
    port = int(settings["ui_port"])
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            log(f"dashboard: http://127.0.0.1:{port}")
            return
    exe = Path(sys.executable)
    if getattr(sys, "frozen", False):
        agent = exe.with_name("chuck-gpu-agent.exe")
        argv = [str(agent)] if agent.exists() else [str(exe), "ui", "--no-browser"]
    else:
        pyw = exe.with_name("pythonw.exe")
        argv = [str(pyw if pyw.exists() else exe), "-m", "chuck-gpu", "agent"]
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(STATE_DIR / "ui.log", "ab")
    subprocess.Popen(argv, stdout=log_f, stderr=log_f, stdin=subprocess.DEVNULL,
                     creationflags=flags, close_fds=True)
    log(f"dashboard started: http://127.0.0.1:{port}")


def modal_cli() -> list[str]:
    """argv prefix that runs Modal's own CLI, also from the packaged .exe."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).with_name("chuck-gpu.exe")), "_modal"]
    return [sys.executable, "-m", "modal"]


def cmd_ui(args) -> int:
    from chuck_gpu.ui import serve
    return serve(open_browser=not args.no_browser, port=args.port)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_run(args) -> int:
    root = Path(args.dir).resolve()
    cfg = load_config(root)
    if not args.command:
        sys.exit("nothing to run: chuck-gpu run [options] -- <command ...>")
    # GPU: --gpu, else the one-shot dashboard pick, else chuck-gpu.toml, else cheapest
    from chuck_gpu import gpus
    toml_gpu = cfg["gpu"]
    if args.gpu:
        gpu_source = "--gpu"
    elif (picked := gpus.take_next()):
        cfg["gpu"], gpu_source = picked, "picked in the dashboard (next job resets to cheapest)"
    elif toml_gpu:
        gpu_source = "chuck-gpu.toml"
    else:
        cfg["gpu"], gpu_source = gpus.cheapest(), "cheapest"
    for field in ("gpu", "image", "timeout", "add_python", "cpu", "memory"):
        if getattr(args, field, None):
            cfg[field] = getattr(args, field)
    cfg["pip"] = list(cfg["pip"]) + (args.pip or [])
    cfg["apt"] = list(cfg["apt"]) + (args.apt or [])
    cfg["secrets"] = list(cfg["secrets"]) + (args.secret or [])
    cfg["env"] = dict(cfg["env"])
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        cfg["env"][k] = v
    if args.no_download:
        cfg["download"] = False
    if cfg["gpu"] not in gpus.CATALOG:
        sys.exit(f"unknown GPU {cfg['gpu']!r} (choose from {', '.join(gpus.GPUS)})")
    rate = gpus.gpu_rates()[cfg["gpu"]]
    log(f"GPU: {cfg['gpu']} (${rate:.2f}/hr, {gpus.CATALOG[cfg['gpu']][1]} GB) — {gpu_source}")

    import modal
    from chuck_gpu.jobs import JobRecorder
    key = workspace_key(root)
    ensure_ui()
    job = JobRecorder(root=root, key=key, command=args.command, gpu=cfg["gpu"],
                      image=cfg["image"], timeout=int(cfg["timeout"]))
    job.update(gpu_source=gpu_source, cpu=cfg["cpu"], memory=cfg["memory"])
    try:
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        log(f"workspace {key}  ←  {root}")
        sync_up(vol, key, root, cfg, force=args.resync, job=job)

        log(f"starting {cfg['gpu']} sandbox (image: {cfg['image']}) …")
        job.update(phase="starting")
        t0 = time.time()
        sb, vol = open_sandbox(cfg, root, key, cfg["gpu"])
    except BaseException as e:
        job.update(detail=f"{type(e).__name__}: {e}"[:500])
        job.finish("stopped" if isinstance(e, KeyboardInterrupt) else "failed", None)
        raise
    job.update(sandbox_id=sb.object_id, started=t0)
    code = 1
    interrupted = False
    try:
        log(f"sandbox {sb.object_id} up in {time.time() - t0:.0f}s — running: "
            f"{shlex.join(args.command)}")
        job.update(phase="running", running_since=time.time())
        marker = "/tmp/.chuck_gpu-start"
        capture(sb, ["sh", "-c", f": > {marker}; sleep 1"])
        cmd = shlex.join(args.command)
        try:
            code = run_logged(sb, cmd, job=job)
        except Exception:
            if not job.stop_requested():
                raise
        if job.stop_requested():
            log("stopped from the dashboard — no results copied back")
            job.finish("stopped", 143)
            return 143
        log(f"command exited with code {code} after {time.time() - t0:.0f}s")

        changed = []
        if cfg["download"]:
            job.update(phase="downloading")
            listing = capture(sb, ["sh", "-c",
                                   f"find . -type f -newer {marker} -printf '%P\\t%s\\n'; sync {MOUNT} 2>/dev/null; sync"])
            # excluded paths never sync either way (e.g. big caches kept remote)
            patterns = DEFAULT_EXCLUDES + list(cfg["exclude"]) + read_ignore(root)
            for line in listing.splitlines():
                rel, _, size = line.partition("\t")
                if rel and not rel.endswith(".chuck_gpu-part") and not excluded(rel, patterns):
                    changed.append((rel, int(size or 0)))
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — stopping the remote GPU")
    except Exception as e:
        job.update(detail=f"{type(e).__name__}: {e}"[:500])
        job.finish("failed", None)
        sb.terminate()
        raise
    finally:
        sb.terminate()
        job.update(gpu_released=time.time())
    if interrupted:
        job.finish("stopped", 130)
        return 130
    if cfg["download"] and changed:
        log(f"downloading {len(changed)} new/changed file(s), "
            f"{human(sum(s for _, s in changed))}")
    if cfg["download"]:
        # volume commits when the sandbox terminates; give it a moment to land
        for attempt in range(10):
            try:
                sync_down(vol, key, root, changed)
                break
            except Exception as e:  # not yet committed
                if attempt == 9:
                    log(f"download failed: {e}")
                    job.update(detail=f"download failed: {e}"[:500])
                    job.finish("failed", code)
                    return code or 1
                time.sleep(2)
    job.update(download={"files": len(changed), "bytes": sum(s for _, s in changed)})
    job.finish("done" if code == 0 else "failed", code)
    return code


def cmd_smi(args) -> int:
    cfg = dict(DEFAULTS)
    cfg["image"] = "cuda"
    from chuck_gpu import gpus
    gpu = args.gpu or gpus.cheapest()
    log(f"starting {gpu} sandbox …")
    sb, _ = open_sandbox(cfg, Path.cwd(), None, gpu)
    try:
        return run_exec(sb, ["nvidia-smi"])
    finally:
        sb.terminate()


def cmd_ls(args) -> int:
    import modal
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    me = workspace_key(Path(args.dir).resolve())
    for e in vol.listdir("/"):
        tag = "  ← this directory" if e.path.strip("/") == me else ""
        print(f"{e.path.strip('/')}{tag}")
    return 0


def cmd_rm(args) -> int:
    import modal
    root = Path(args.dir).resolve()
    key = args.key or workspace_key(root)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    vol.remove_file(f"/{key}", recursive=True)
    manifest_path(key).unlink(missing_ok=True)
    log(f"removed remote workspace {key}")
    return 0


def require_login() -> None:
    from chuck_gpu import account
    if not account.status(lookup=False)["logged_in"]:
        sys.exit("chuck-gpu: not logged in to Modal. Run `chuck-gpu login` (or log in from "
                 "the dashboard: `chuck-gpu ui`).")


def cmd_login(args) -> int:
    from chuck_gpu import account
    if args.token_id or args.token_secret:
        if not (args.token_id and args.token_secret):
            sys.exit("give both --token-id and --token-secret")
        try:
            s = account.login_token(args.token_id, args.token_secret)
        except Exception as e:
            sys.exit(f"login failed: {e}")
    else:
        # browser flow: opens modal.com, waits for approval, stores the token
        import subprocess
        code = subprocess.call(modal_cli() + ["token", "new", "--activate"])
        if code:
            return code
        s = account.status()
    log(f"logged in to Modal as {s['workspace'] or s['username']} (profile {s['profile']})")
    return 0


def cmd_logout(args) -> int:
    from chuck_gpu import account
    before = account.status(lookup=False)
    if not before["logged_in"]:
        log("not logged in")
        return 0
    try:
        account.logout()
    except RuntimeError as e:
        sys.exit(str(e))
    log(f"logged out (removed profile {before['profile']} from {before['config_path']})")
    return 0


def cmd_whoami(args) -> int:
    from chuck_gpu import account
    s = account.status()
    if not s["logged_in"]:
        print("not logged in — run `chuck-gpu login`")
        return 1
    print(f"{s['workspace'] or s['username']}  (profile {s['profile']}"
          f"{', from environment' if s['via_env'] else ''})")
    if s["error"]:
        print(f"token problem: {s['error']}")
        return 1
    others = [p for p in s["profiles"] if p != s["profile"]]
    if others:
        print("other profiles: " + ", ".join(others) + "  (switch: modal profile activate NAME)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):  # piped output on Windows defaults to cp1252
        if stream and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    if argv[:1] == ["_modal"]:  # Modal's own CLI, bundled (used for browser login)
        from modal.cli.entry_point import entrypoint_cli
        sys.argv = ["modal"] + argv[1:]
        return entrypoint_cli() or 0
    command: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(prog="chuck-gpu", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="sync this directory, run a command on a cloud GPU, sync results back")
    r.add_argument("--gpu", choices=GPUS, help="GPU type (default: your dashboard pick, else the cheapest)")
    r.add_argument("--image", help=f"preset ({', '.join(IMAGES)}) or any registry image")
    r.add_argument("--add-python", help="python version to add to a bare registry image")
    r.add_argument("--pip", action="append", help="extra pip package (repeatable)")
    r.add_argument("--apt", action="append", help="extra apt package (repeatable)")
    r.add_argument("--env", "-e", action="append", help="KEY=VALUE (repeatable)")
    r.add_argument("--secret", action="append", help="Modal secret name (repeatable)")
    r.add_argument("--timeout", type=int, help="seconds before the GPU is stopped (default 3600)")
    r.add_argument("--cpu", type=float, help="CPU cores to reserve (default: Modal's minimum)")
    r.add_argument("--memory", type=int, help="memory to reserve, MiB")
    r.add_argument("--dir", default=".", help="project directory to sync (default .)")
    r.add_argument("--resync", action="store_true", help="re-upload everything")
    r.add_argument("--no-download", action="store_true", help="don't copy outputs back")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("smi", help="run nvidia-smi on a cloud GPU")
    m.add_argument("--gpu", choices=GPUS)
    m.set_defaults(func=cmd_smi)

    u = sub.add_parser("ui", help="live dashboard: current job, spend, remaining credit")
    u.add_argument("--no-browser", action="store_true")
    u.add_argument("--port", type=int)
    u.set_defaults(func=cmd_ui)

    ls = sub.add_parser("ls", help="list remote workspaces")
    ls.add_argument("--dir", default=".")
    ls.set_defaults(func=cmd_ls)

    rm = sub.add_parser("rm", help="delete this directory's remote workspace")
    rm.add_argument("--dir", default=".")
    rm.add_argument("--key", help="workspace name from `chuck-gpu ls`")
    rm.set_defaults(func=cmd_rm)

    li = sub.add_parser("login", help="connect your Modal account (opens the browser, or use a token)")
    li.add_argument("--token-id", help="ak-… from modal.com/settings/tokens")
    li.add_argument("--token-secret", help="as-… (the matching secret)")
    li.set_defaults(func=cmd_login)

    lo = sub.add_parser("logout", help="remove the active Modal token from this machine")
    lo.set_defaults(func=cmd_logout)

    wh = sub.add_parser("whoami", help="show which Modal account chuck-gpu uses")
    wh.set_defaults(func=cmd_whoami)

    ag = sub.add_parser("agent", help="run in the system tray: dashboard always on, job alerts")
    ag.set_defaults(func=lambda a: __import__("chuck-gpu.agent", fromlist=["main"]).main())

    args = p.parse_args(argv)
    args.command = command
    if args.cmd in ("run", "smi", "ls", "rm"):
        require_login()
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
