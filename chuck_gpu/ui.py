"""`chuck-gpu ui` — local dashboard: what's running, what it costs, what's left.

Spend comes from three places, most precise first:
  * per-sandbox metered usage (GPU / CPU / memory nanoseconds) × your
    workspace's live billing rates — the cost of each chuck-gpu job;
  * the workspace billing summary for this month — everything Modal billed,
    including other apps, and how much plan credit has been used;
  * `monthly_credit` from ~/.chuck_gpu/config.toml — Modal doesn't expose the plan
    allowance, so remaining credit = allowance − credits applied this month.
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import tomllib
import webbrowser
from datetime import datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chuck_gpu import account, gpus
from chuck_gpu.jobs import JOBS_DIR, TERMINAL, load_jobs, pid_alive, save_job, tail_log

SETTINGS_PATH = Path.home() / ".chuck_gpu" / "config.toml"
SETTINGS_DEFAULTS = {"monthly_credit": 30.0, "ui_port": 8770, "auto_ui": True,
                     "start_with_windows": True}
STATIC = Path(__file__).parent / "static"

GPU_RATE_KEYS = {g: key for g, (key, *_) in gpus.CATALOG.items()}
# Sandbox defaults Modal bills at minimum when nothing larger is requested
DEFAULT_CORES, DEFAULT_MEM_GIB = 0.125, 0.125
NS_PER_HOUR = 3_600 * 1e9


def load_settings() -> dict:
    s = dict(SETTINGS_DEFAULTS)
    try:
        with open(SETTINGS_PATH, "rb") as f:
            s.update(tomllib.load(f))
    except OSError:
        pass
    except tomllib.TOMLDecodeError as e:
        print(f"[chuck-gpu ui] ignoring bad {SETTINGS_PATH}: {e}", file=sys.stderr)
    return s


def save_settings(updates: dict) -> dict:
    s = load_settings()
    s.update(updates)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in s.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            lines.append(f"{k} = {json.dumps(str(v))}")
    SETTINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return s


class Modal:
    """Billing + usage lookups, cached; all calls are best effort."""

    def __init__(self):
        self.rates: dict[str, float] = {}
        self.rates_at = 0.0
        self.summary: dict | None = None
        self.summary_at = 0.0
        self.summary_error: str | None = None
        self.usage: dict[str, dict] = {}  # sandbox_id -> {cost, at, ...}
        self.lock = threading.Lock()
        # The usage RPC isn't exposed on Sandbox objects; run it on the modal
        # client's own event loop (a private loop hangs on the shared client).
        from modal._utils.async_utils import synchronizer
        self._usage_rpc = synchronizer.create_blocking(_usage_rpc)

    def reset(self) -> None:
        """Forget everything cached from the previous account."""
        self.rates, self.rates_at = {}, 0.0
        self.summary, self.summary_at, self.summary_error = None, 0.0, None
        self.usage = {}

    def refresh_rates(self) -> None:
        if time.time() - self.rates_at < 3600 and self.rates:
            return
        import modal
        r = modal.Workspace.from_context().billing.rates()
        self.rates = {k: float(r[k]) for k in r}
        self.rates_at = time.time()
        try:
            gpus.save_rates(self.rates)  # so `chuck-gpu run` knows today's cheapest GPU
        except OSError:
            pass

    def refresh_summary(self) -> None:
        if time.time() - self.summary_at < 60:
            return
        import modal
        try:
            s = modal.Workspace.from_context().billing.summary()
            self.summary = {
                "start": s.start.isoformat(), "end": s.end.isoformat(),
                "metered": float(s.metered_cost), "billed": float(s.billed_cost),
                "breakdown": {k: float(v) for k, v in s.metered_cost_breakdown.items()},
                "adjustments": {k: float(v) for k, v in s.adjustments.items()},
            }
            self.summary_error = None
        except Exception as e:
            self.summary_error = f"{type(e).__name__}: {e}"[:300]
        self.summary_at = time.time()

    def gpu_rate_hr(self, gpu: str) -> float:
        return self.rates.get(GPU_RATE_KEYS.get(gpu, ""), 0.0)

    def nominal_rate_hr(self, gpu: str, cpu: float | None = None,
                        memory_mib: int | None = None) -> float:
        """$/hr of a sandbox with these reservations (Modal bills at least these)."""
        cores = float(cpu) if cpu else DEFAULT_CORES
        mem_gib = memory_mib / 1024 if memory_mib else DEFAULT_MEM_GIB
        return (self.gpu_rate_hr(gpu)
                + cores * self.rates.get("cpu_hour_cost_sandbox", 0.1419)
                + mem_gib * self.rates.get("mem_gib_hour_cost_sandbox", 0.024))

    def sandbox_cost(self, sandbox_id: str, gpu_hint: str) -> dict | None:
        try:
            r = self._usage_rpc(sandbox_id)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200]}
        gpu = r.gpu_type or gpu_hint
        gpu_cost = r.gpu_nanosecs / NS_PER_HOUR * self.gpu_rate_hr(gpu)
        cpu_cost = r.cpu_core_nanosecs / NS_PER_HOUR * self.rates.get("cpu_hour_cost_sandbox", 0)
        mem_cost = r.mem_gib_nanosecs / NS_PER_HOUR * self.rates.get("mem_gib_hour_cost_sandbox", 0)
        return {"cost": gpu_cost + cpu_cost + mem_cost, "gpu_cost": gpu_cost,
                "cpu_cost": cpu_cost, "mem_cost": mem_cost,
                "gpu_s": r.gpu_nanosecs / 1e9, "at": time.time()}

    def stop(self, sandbox_id: str) -> None:
        import modal
        modal.Sandbox.from_id(sandbox_id).terminate()

    def list_sandboxes(self) -> list[dict]:
        """All live sandboxes on this workspace (not just chuck-gpu jobs)."""
        import modal
        try:
            sbs = list(modal.Sandbox.list())
        except Exception:
            return []
        out = []
        for sb in sbs:
            rc = sb.returncode
            if rc is not None:
                continue  # finished, skip
            tags = {}
            try:
                tags = sb.get_tags() or {}
            except Exception:
                pass
            out.append({"id": sb.object_id, "tags": tags,
                         "usage": self.sandbox_usage(sb.object_id)})
        return out

    def sandbox_usage(self, sandbox_id: str) -> dict | None:
        """Metered CPU-core and memory time. Modal does not publish live GPU %."""
        try:
            r = self._usage_rpc(sandbox_id)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:160]}
        return {
            "gpu_type": r.gpu_type or None,
            "gpu_s": round(r.gpu_nanosecs / 1e9, 1),
            "cpu_core_s": round(r.cpu_core_nanosecs / 1e9, 1),
            "mem_gib_s": round(r.mem_gib_nanosecs / 1e9, 1),
        }

    def kill_sandbox(self, sandbox_id: str) -> None:
        import modal
        modal.Sandbox.from_id(sandbox_id).terminate()


async def _usage_rpc(sandbox_id: str):
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    return await client._stub.SandboxGetResourceUsage(
        api_pb2.SandboxGetResourceUsageRequest(sandbox_id=sandbox_id))


M: Modal | None = None
BROWSER_LOGIN = account.BrowserLogin()
_history_cache: dict = {}


def _command_key(job: dict) -> str:
    return job["workspace"] + "\0" + json.dumps(job["command"])


def _past_durations(jobs: list[dict], job: dict) -> list[float]:
    key = _command_key(job)
    out = []
    for j in jobs:
        if j is job or j.get("phase") != "done" or _command_key(j) != key:
            continue
        if j.get("running_since") and j.get("ended"):
            out.append(j["ended"] - j["running_since"])
    return out


def estimate(job: dict, jobs: list[dict], cost_now: float, rate_s: float) -> dict:
    """Remaining time and projected total cost for a live job."""
    now = time.time()
    running_for = now - job["running_since"] if job.get("running_since") else 0.0
    remaining, basis = None, None
    prog = job.get("progress")
    if job["phase"] == "running" and prog and now - prog.get("at", 0) < 120:
        if prog.get("eta_s") is not None:
            remaining = max(0.0, prog["eta_s"] - (now - prog["at"]))
            basis = f"progress bar “{prog.get('label') or 'progress'}”"
        elif prog.get("percent") and running_for > 10:
            pct = prog["percent"] / 100
            remaining = max(0.0, running_for / pct - running_for)
            basis = f"{prog['percent']:.0f}% done"
    if remaining is None:
        past = _past_durations(jobs, job)
        if past:
            med = statistics.median(past)
            remaining = max(0.0, med - running_for)
            basis = f"median of {len(past)} previous run{'s' * (len(past) > 1)}"
    max_total = cost_now + max(0.0, job["timeout"] - (now - (job.get("started") or now))) * rate_s
    if remaining is None:
        return {"remaining_s": None, "total": None, "basis": "no progress output or history yet",
                "max_total": max_total}
    return {"remaining_s": remaining, "total": cost_now + remaining * rate_s,
            "basis": basis, "max_total": max_total}


def local_host_metrics() -> dict:
    """This machine, not the cloud GPU. psutil is optional."""
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not installed"}
    if not getattr(local_host_metrics, "_primed", False):
        psutil.cpu_percent(interval=None)
        psutil.Process().cpu_percent(interval=None)
        local_host_metrics._primed = True
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(str(Path.home().anchor or "C:\\"))
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "app_cpu_pct": proc.cpu_percent(interval=None),
        "app_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "mem_used_gb": round(vm.used / (1024 ** 3), 2),
        "mem_total_gb": round(vm.total / (1024 ** 3), 2),
        "mem_pct": vm.percent,
        "disk_used_gb": round(du.used / (1024 ** 3), 1),
        "disk_total_gb": round(du.total / (1024 ** 3), 1),
        "disk_pct": du.percent,
    }


def build_state() -> dict:
    assert M is not None
    settings = load_settings()
    if BROWSER_LOGIN.state == "done":  # finished in the background since the last poll
        M.reset()
        BROWSER_LOGIN.state = "idle"
    acct = account.status()
    base = {"now": time.time(), "account": acct, "login": BROWSER_LOGIN.info(),
            "monthly_credit": float(settings["monthly_credit"]),
            "settings_path": str(SETTINGS_PATH),
            "local": local_host_metrics()}
    if not acct["logged_in"]:
        return base
    try:
        M.refresh_rates()
    except Exception as e:
        return {**base, "error": f"could not reach Modal: {e}"}
    M.refresh_summary()

    jobs = load_jobs(60)
    now = time.time()
    out_jobs = []
    live_rate_hr = 0.0
    for job in jobs:
        # a run whose CLI died leaves an idle sandbox billing until timeout — stop it
        if job["phase"] not in TERMINAL and not pid_alive(job.get("pid")):
            if job.get("sandbox_id") and not job.get("gpu_released"):
                try:
                    M.stop(job["sandbox_id"])
                except Exception:
                    pass
            job.update(phase="lost", ended=job.get("ended") or now,
                       detail="chuck-gpu process exited without finishing; remote sandbox stopped")
            save_job(job)

        live = job["phase"] not in TERMINAL
        cost = None
        sid = job.get("sandbox_id")
        if job.get("cost_final") is not None:
            cost = job["cost_final"]
        elif sid:
            cached = M.usage.get(sid)
            stale = not cached or now - cached.get("at", 0) > (4 if live else 15)
            if stale:
                fresh = M.sandbox_cost(sid, job["gpu"])
                if fresh and "cost" in fresh:
                    prev = M.usage.get(sid)
                    fresh["prev"] = prev and {"cost": prev.get("cost"), "at": prev.get("at")}
                    M.usage[sid] = fresh
                    # usage settles shortly after the sandbox stops; then freeze it
                    if not live and job.get("ended") and now - job["ended"] > 120:
                        job["cost_final"] = fresh["cost"]
                        save_job(job)
            u = M.usage.get(sid)
            if u and "cost" in u:
                cost = u["cost"]
        nominal_hr = M.nominal_rate_hr(job["gpu"], job.get("cpu"), job.get("memory"))
        if job.get("started"):
            # Modal bills at least the reserved GPU/CPU/memory for as long as the
            # sandbox runs; its per-sandbox usage figure can lag far behind on
            # long jobs, so never show less than that floor
            end = job.get("gpu_released") or job.get("ended") or now
            floor = max(0.0, end - job["started"]) * nominal_hr / 3600
            cost = floor if cost is None else max(cost, floor)

        row = {k: job.get(k) for k in ("id", "dir", "workspace", "command", "gpu", "image",
                                       "phase", "detail", "created", "started", "ended",
                                       "running_since", "exit_code", "sandbox_id", "progress",
                                       "upload", "download", "timeout", "cpu", "memory",
                                       "gpu_source")}
        if row["progress"] and now - row["progress"].get("at", 0) > 300:
            row["progress"] = None  # no progress output for 5 min: don't show a stale bar
        row["cost"] = cost
        u = M.usage.get(sid) if sid else None
        if u and "cost" in u and cost is not None and u["cost"] >= cost * 0.95:
            row["breakdown"] = {"gpu": u["gpu_cost"], "cpu": u["cpu_cost"], "mem": u["mem_cost"]}
        elif cost and job.get("started"):  # the reservation floor is what's shown: split that
            hrs = max(0.0, (job.get("gpu_released") or job.get("ended") or now) - job["started"]) / 3600
            cores = float(job.get("cpu") or DEFAULT_CORES)
            mem_gib = (job.get("memory") or DEFAULT_MEM_GIB * 1024) / 1024
            row["breakdown"] = {
                "gpu": hrs * M.gpu_rate_hr(job["gpu"]),
                "cpu": hrs * cores * M.rates.get("cpu_hour_cost_sandbox", 0),
                "mem": hrs * mem_gib * M.rates.get("mem_gib_hour_cost_sandbox", 0)}
        row["rate_hr"] = nominal_hr
        if live:
            rate_s = nominal_hr / 3600
            if u and u.get("prev") and u["prev"].get("cost") is not None:
                dt = u["at"] - u["prev"]["at"]
                if dt > 1:
                    rate_s = max(rate_s, (u["cost"] - u["prev"]["cost"]) / dt)
            row["rate_hr"] = rate_s * 3600
            if job.get("started"):
                live_rate_hr += row["rate_hr"]
            row["estimate"] = estimate(job, jobs, cost or 0.0, rate_s)
            row["log"] = tail_log(job["id"])
        out_jobs.append(row)

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0,
                                               microsecond=0).timestamp()
    mgpu_today = sum(j["cost"] or 0 for j in out_jobs if (j["created"] or 0) >= today)
    mgpu_month = sum(j["cost"] or 0 for j in out_jobs if (j["created"] or 0) >= month)

    credit = None
    s = M.summary
    if s:
        used = -s["adjustments"].get("Credits", 0.0)
        allowance = float(settings["monthly_credit"])
        credit = {"allowance": allowance, "used": used, "remaining": max(0.0, allowance - used),
                  "billed": s["billed"], "metered": s["metered"],
                  "as_of": M.summary_at, "period_end": s["end"]}
    return {
        **base,
        "now": now, "jobs": out_jobs, "live_rate_hr": live_rate_hr,
        "mgpu_today": mgpu_today, "mgpu_month": mgpu_month,
        "summary": s, "summary_error": M.summary_error, "credit": credit,
        "rates": {g: M.gpu_rate_hr(g) for g in GPU_RATE_KEYS},
        "gpus": gpus.catalog(M.rates),
        "next_gpu": gpus.get_next(),
        "default_gpu": gpus.cheapest(M.rates),
        "cpu_rate_hr": M.rates.get("cpu_hour_cost_sandbox"),
        "mem_rate_hr": M.rates.get("mem_gib_hour_cost_sandbox"),
        "sandboxes": M.list_sandboxes(),
        "egress_rate_hr": M.rates.get("egress_gib_cost"),
        "volume_storage_rate": M.rates.get("volume_storage_gib_month_cost"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=lambda o: float(o) if isinstance(o, Decimal) else str(o)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local_host(self) -> bool:
        # refuse DNS-rebinding: only answer requests addressed to this machine
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host in ("127.0.0.1", "localhost", "::1"):
            return True
        self.send_error(403)
        return False

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        data = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
        return data if isinstance(data, dict) else {}

    def do_GET(self):
        if not self._local_host():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with M.lock:
                try:
                    self._json(build_state())
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        else:
            self.send_error(404)

    def do_POST(self):
        # only from this page (same-origin header check)
        if not self._local_host():
            return
        parts = self.path.strip("/").split("/")
        origin = self.headers.get("Origin", "")
        if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
            return self.send_error(403)
        if parts[:2] == ["api", "account"] and len(parts) == 3:
            return self._account(parts[2])
        if self.path == "/api/next-gpu":
            try:
                gpus.set_next(self._body().get("gpu") or None)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"ok": True, "next_gpu": gpus.get_next()})
        if self.path == "/api/settings":
            try:
                credit = float(self._body().get("monthly_credit"))
                if not 0 <= credit < 1e6:
                    raise ValueError
            except (TypeError, ValueError):
                return self._json({"error": "monthly credit must be a dollar amount"}, 400)
            save_settings({"monthly_credit": credit})
            return self._json({"ok": True})
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            job = next((j for j in load_jobs(200) if j["id"] == parts[2]), None)
            if not job or not job.get("sandbox_id"):
                return self._json({"error": "no running sandbox for that job"}, 404)
            try:
                M.stop(job["sandbox_id"])
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            # the running `chuck-gpu run` owns the job file; leave it a marker instead
            (JOBS_DIR / f"{job['id']}.stop").touch()
            return self._json({"ok": True})
        if self.path == "/api/sandbox-kill":
            sid = self._body().get("id", "")
            if not sid:
                return self._json({"error": "no sandbox id"}, 400)
            try:
                M.kill_sandbox(sid)
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            return self._json({"ok": True})
        self.send_error(404)

    def _account(self, action: str):
        try:
            body = self._body()
            with M.lock:
                if action == "login-browser":
                    return self._json(BROWSER_LOGIN.start())
                if action == "login-cancel":
                    BROWSER_LOGIN.cancel()
                    return self._json({"ok": True})
                if action == "login-token":
                    result = account.login_token(str(body.get("token_id", "")),
                                                 str(body.get("token_secret", "")))
                elif action == "logout":
                    result = account.logout()
                elif action == "switch":
                    result = account.switch(str(body.get("profile", "")))
                else:
                    return self.send_error(404)
                M.reset()
                return self._json(result)
        except Exception as e:
            msg = str(e) or type(e).__name__
            if type(e).__name__ == "AuthError":
                msg = f"Modal rejected that token ({msg})"
            return self._json({"error": msg[:400]}, 400)


def start_server(port: int | None = None) -> ThreadingHTTPServer:
    """Bind the dashboard (raises OSError if the port is taken); caller serves it."""
    global M
    port = port or int(load_settings()["ui_port"])
    M = M or Modal()
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(open_browser: bool = True, port: int | None = None) -> int:
    port = port or int(load_settings()["ui_port"])
    try:
        httpd = start_server(port)
    except OSError as e:
        print(f"[chuck-gpu ui] port {port} busy ({e}) — is the dashboard already running?",
              file=sys.stderr)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        return 1
    url = f"http://127.0.0.1:{port}"
    print(f"[chuck-gpu ui] {url}", file=sys.stderr, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
