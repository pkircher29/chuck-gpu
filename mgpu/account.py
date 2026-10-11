"""Which Modal account mgpu uses: status, log in, log out, switch.

Credentials stay where the Modal CLI keeps them (~/.modal.toml, or
$MODAL_CONFIG_PATH, or $MODAL_TOKEN_ID/$MODAL_TOKEN_SECRET); mgpu never
stores them anywhere else.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

DEFAULT_SERVER_URL = "https://api.modal.com"
_lookup_cache: dict[str, tuple[float, dict]] = {}


def _mc():
    import modal.config as mc
    return mc


def _blocking(fn):
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(fn)


def reload() -> None:
    """Make this process pick up a changed ~/.modal.toml (modal caches it at import)."""
    mc = _mc()
    mc._user_config = mc._read_user_config()
    mc._profile = os.environ.get("MODAL_PROFILE") or mc._config_active_profile()
    from modal.client import _Client
    _Client.set_env_client(None)
    _lookup_cache.clear()


def _credentials() -> tuple[str | None, str | None, str | None, str]:
    """(profile, token_id, token_secret, server_url) that modal would use now."""
    mc = _mc()
    user = mc._read_user_config()
    profile = os.environ.get("MODAL_PROFILE") or next(
        (k for k, v in user.items() if v.get("active") is True), "default")
    prof = user.get(profile, {})
    tid = os.environ.get("MODAL_TOKEN_ID") or prof.get("token_id")
    sec = os.environ.get("MODAL_TOKEN_SECRET") or prof.get("token_secret")
    url = os.environ.get("MODAL_SERVER_URL") or prof.get("server_url") or DEFAULT_SERVER_URL
    return (profile if profile in user else None), tid, sec, url


def status(lookup: bool = True) -> dict:
    mc = _mc()
    user = mc._read_user_config()
    profile, tid, sec, url = _credentials()
    out = {
        "logged_in": bool(tid and sec),
        "profile": profile,
        "profiles": sorted(user),
        "via_env": bool(os.environ.get("MODAL_TOKEN_ID")),
        "config_path": mc.user_config_path,
        "workspace": None, "username": None, "error": None,
    }
    if out["logged_in"] and lookup:
        cached = _lookup_cache.get(tid)
        if cached and time.time() - cached[0] < 600:
            out.update(cached[1])
        else:
            try:
                r = _blocking(mc._lookup_workspace)(url, tid, sec)
                info = {"workspace": r.workspace_name, "username": r.username}
                _lookup_cache[tid] = (time.time(), info)
                out.update(info)
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def login_token(token_id: str, token_secret: str) -> dict:
    """Verify a token pair, store it as a profile named after the workspace, activate it."""
    token_id, token_secret = token_id.strip(), token_secret.strip()
    if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
        raise ValueError("a Modal token ID starts with ak- and its secret with as-")
    from modal.token_flow import _set_token
    _blocking(_set_token)(token_id, token_secret, activate=True, verify=True)
    reload()
    return status()


def logout() -> dict:
    """Remove the active profile's token from the Modal config file."""
    if os.environ.get("MODAL_TOKEN_ID"):
        raise RuntimeError("logged in through MODAL_TOKEN_ID/MODAL_TOKEN_SECRET environment "
                           "variables; unset them to log out")
    mc = _mc()
    profile, *_ = _credentials()
    user = mc._read_user_config()
    if profile and profile in user:
        del user[profile]
        mc._write_user_config(user)
    reload()
    return status(lookup=False)


def switch(profile: str) -> dict:
    mc = _mc()
    if profile not in mc._read_user_config():
        raise ValueError(f"no Modal profile named {profile!r}")
    mc.config_set_active_profile(profile)
    reload()
    return status()


class BrowserLogin:
    """`modal token new` in the background: it opens modal.com in the browser,
    waits for the user to approve, then writes the token."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.state = "idle"  # idle | waiting | done | failed
        self.error: str | None = None
        self.lock = threading.Lock()

    def start(self) -> dict:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return self.info()
            env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_COLOR="1", TERM="dumb",
                       COLUMNS="200")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "modal", "token", "new", "--activate"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=env, creationflags=flags, text=True, encoding="utf-8", errors="replace")
            self.url, self.error, self.state = None, None, "waiting"
            threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
            return self.info()

    def _watch(self, proc: subprocess.Popen) -> None:
        tail = []
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-8:]
            m = re.search(r"https://\S+", line)
            if m and not self.url and "token" in m.group(0):
                self.url = m.group(0)
        code = proc.wait()
        if code == 0:
            reload()
            self.state = "done"
        else:
            self.state = "failed"
            self.error = "\n".join(t for t in tail if t.strip())[-500:] or f"exit {code}"

    def cancel(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
            self.state = "idle"

    def info(self) -> dict:
        return {"state": self.state, "url": self.url, "error": self.error}
