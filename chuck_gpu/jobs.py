"""Job records shared by `chuck-gpu run` (writer) and `chuck-gpu ui` (reader).

Each run gets ~/.chuck_gpu/jobs/<job_id>.json (state, rewritten atomically) and
<job_id>.log (the command's combined output, appended as it streams).
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path

JOBS_DIR = Path.home() / ".chuck_gpu" / "jobs"
TERMINAL = {"done", "failed", "stopped", "lost"}
HOST = socket.gethostname()
# A job run on another host (e.g. inside a Docker container sharing
# ~/.chuck_gpu) can't be pid-checked from here; it beats instead.
HEARTBEAT_S = 60
HEARTBEAT_LOST_S = 600

# tqdm:  " 42%|████▏     | 420/1000 [01:02<01:26,  6.71it/s]"
TQDM = re.compile(r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:(\d+):)?(\d+):(\d+)<(?:(\d+):)?(\d+):(\d+)")
# "epoch 3/10", "step 1200/50000", "[3/10]"
FRACTION = re.compile(r"(?i)(?:epoch|step|iter(?:ation)?|batch)\s*[:=]?\s*(\d+)\s*/\s*(\d+)")
PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s?%")


def _hms(h, m, s) -> int:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


class ProgressParser:
    """Turns streamed output into {percent, eta_s, label} — best effort."""

    def __init__(self):
        self.buf = ""
        self.state: dict | None = None

    def feed(self, text: str) -> dict | None:
        self.buf = (self.buf + text)[-4000:]
        # progress bars redraw with \r; only the newest segment matters
        lines = re.split(r"[\r\n]+", self.buf)
        for line in reversed(lines[-40:]):
            m = TQDM.search(line)
            if m:
                pct = int(m.group(1))
                eta = _hms(m.group(7), m.group(8), m.group(9))
                label = line[:m.start()].strip().rstrip(":") or "progress"
                self.state = {"percent": pct, "eta_s": eta, "label": label[-60:],
                              "done": int(m.group(2)), "total": int(m.group(3)), "at": time.time()}
                return self.state
            m = FRACTION.search(line)
            if m and int(m.group(2)) > 0:
                done, total = int(m.group(1)), int(m.group(2))
                self.state = {"percent": min(100, round(100 * done / total, 1)), "eta_s": None,
                              "label": line[max(0, m.start() - 20):m.end()].strip()[-60:],
                              "done": done, "total": total, "at": time.time()}
                return self.state
        return None


class JobRecorder:
    def __init__(self, *, root: Path, key: str, command: list[str], gpu: str, image: str,
                 timeout: int):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = JOBS_DIR / f"{self.id}.json"
        self.log_path = JOBS_DIR / f"{self.id}.log"
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.progress = ProgressParser()
        self._last_write = 0.0
        self.data = {
            "id": self.id, "pid": os.getpid(), "host": HOST, "beat": time.time(),
            "dir": str(root), "workspace": key,
            "command": command, "gpu": gpu, "image": image, "timeout": timeout,
            "phase": "uploading", "detail": "", "created": time.time(),
            "started": None, "ended": None, "exit_code": None, "sandbox_id": None,
            "progress": None, "upload": None, "download": None,
        }
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.write()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self) -> None:
        while not self._done.wait(HEARTBEAT_S):
            self.update(beat=time.time())

    def write(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            # Windows refuses to replace a file another process has open,
            # and the dashboard reads these constantly (seen from a Docker
            # container writing to a bind-mounted ~/.chuck_gpu). Retry
            # briefly; a missed update is harmless, the next one carries it.
            for attempt in range(10):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            self._last_write = time.time()

    def update(self, **kw) -> None:
        self.data.update(kw)
        self.write()

    def output(self, text: str) -> None:
        self._log.write(text)
        self._log.flush()
        state = self.progress.feed(text)
        if state is not None:
            self.data["progress"] = state
            if time.time() - self._last_write > 1:
                self.write()

    def stop_requested(self) -> bool:
        return (JOBS_DIR / f"{self.id}.stop").exists()

    def finish(self, phase: str, exit_code: int | None) -> None:
        stop_marker = JOBS_DIR / f"{self.id}.stop"
        if stop_marker.exists():
            phase = "stopped"
            self.data["detail"] = "stopped from the dashboard"
            stop_marker.unlink(missing_ok=True)
        self._done.set()
        self.update(phase=phase, exit_code=exit_code, ended=time.time())
        self._log.close()


def load_jobs(limit: int = 50) -> list[dict]:
    if not JOBS_DIR.is_dir():
        return []
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            jobs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return jobs


def save_job(job: dict) -> None:
    p = JOBS_DIR / f"{job['id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job))
    os.replace(tmp, p)


def tail_log(job_id: str, max_bytes: int = 6000) -> str:
    p = JOBS_DIR / f"{job_id}.log"
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    # collapse progress-bar redraws to their latest frame per line (after
    # normalising CRLF, or every line would collapse to the empty string)
    raw = raw.replace("\r\n", "\n")
    lines = [seg.rstrip("\r").split("\r")[-1] for seg in raw.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines[-40:])


def job_alive(job: dict) -> bool:
    """Whether the `chuck-gpu run` that owns this job is still running."""
    if job.get("host", HOST) != HOST:
        return time.time() - (job.get("beat") or job.get("created") or 0) < HEARTBEAT_LOST_S
    return pid_alive(job.get("pid"))


def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
