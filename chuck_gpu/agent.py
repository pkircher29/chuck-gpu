"""`chuck-gpu agent` / chuck-gpu-agent.exe — the always-on background half of chuck-gpu.

Lives in the system tray and keeps the dashboard (http://127.0.0.1:8770)
serving, so it is there the moment you want it. It also keeps watching jobs
when no browser is open: a run whose terminal was closed gets its cloud GPU
stopped instead of billing until the timeout, and you get a notification when
a job finishes.

Tray menu: Open dashboard (also a click on the icon), live status, Start with
Windows, Quit.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "chuck-gpu"
MUTEX_NAME = "Local\\chuck-gpu-agent"
LOG = Path.home() / ".chuck_gpu" / "agent.log"


# ── icon ────────────────────────────────────────────────────────────────────

def make_icon(size: int = 64, state: str = "idle"):
    """A GPU-chip glyph; the corner dot shows idle (none) / running / problem."""
    from PIL import Image, ImageDraw
    s = size / 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    blue = (47, 111, 222, 255)
    d.rounded_rectangle([4 * s, 10 * s, 60 * s, 54 * s], radius=9 * s, fill=blue)
    d.rounded_rectangle([18 * s, 21 * s, 46 * s, 43 * s], radius=4 * s, fill=(255, 255, 255, 255))
    for i in range(4):  # pins
        x = (15 + i * 11) * s
        d.rectangle([x, 4 * s, x + 4 * s, 10 * s], fill=blue)
        d.rectangle([x, 54 * s, x + 4 * s, 60 * s], fill=blue)
    if state != "idle":
        color = (31, 196, 98, 255) if state == "running" else (229, 72, 72, 255)
        d.ellipse([40 * s, 36 * s, 63 * s, 59 * s], fill=color, outline=(255, 255, 255, 255),
                  width=max(1, int(3 * s)))
    return img


def save_ico(path: str) -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    make_icon(256).save(path, sizes=[(n, n) for n in sizes])


# ── start with Windows ──────────────────────────────────────────────────────

def launch_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    pyw = Path(sys.executable).with_name("pythonw.exe")
    return f'"{pyw if pyw.exists() else sys.executable}" -m chuck_gpu agent'


def autostart_enabled() -> bool:
    if os.name != "nt":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, RUN_NAME)
            return True
    except OSError:
        return False


def set_autostart(on: bool) -> None:
    if os.name != "nt":
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, launch_command())
        else:
            try:
                winreg.DeleteValue(k, RUN_NAME)
            except OSError:
                pass


# ── agent ───────────────────────────────────────────────────────────────────

def _single_instance() -> bool:
    if os.name != "nt":
        return True
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW(None, False, MUTEX_NAME)
    return k32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def _money(v) -> str:
    return "—" if v is None else f"${v:,.2f}"


class Agent:
    def __init__(self):
        from chuck_gpu import ui
        self.ui = ui
        self.settings = ui.load_settings()
        self.port = int(self.settings["ui_port"])
        self.url = f"http://127.0.0.1:{self.port}"
        self.status = "starting…"
        self.state = "idle"
        self.seen_live: dict[str, str] = {}  # job id -> command, while live
        self.httpd = None
        self.icon = None
        self.stopping = threading.Event()

    def start_server(self) -> None:
        try:
            self.httpd = self.ui.start_server(self.port)
        except OSError:
            # an older `chuck-gpu ui` already holds the port; the dashboard still works
            self.status = f"dashboard already running on :{self.port}"
            return
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def poll(self) -> None:
        """Keep job bookkeeping alive and mirror it into the tray."""
        while not self.stopping.wait(5):
            try:
                if self.ui.M is None:
                    self.ui.M = self.ui.Modal()
                with self.ui.M.lock:
                    s = self.ui.build_state()
                self.update(s)
            except Exception as e:
                self.status, self.state = f"error: {e}"[:120], "problem"
                self.refresh()

    def update(self, s: dict) -> None:
        acct = s.get("account") or {}
        jobs = s.get("jobs") or []
        live = [j for j in jobs if "estimate" in j]
        # notify about jobs that were live last time and have now ended
        ended = {j["id"]: j for j in jobs if j["id"] in self.seen_live and "estimate" not in j}
        for jid, j in ended.items():
            cmd = self.seen_live.pop(jid)
            result = "finished" if j["phase"] == "done" else j["phase"]
            if j["phase"] == "failed" and j.get("exit_code") is not None:
                result = f"failed (exit {j['exit_code']})"
            self.notify(f"Job {result}", f"{cmd[:80]}\n{_money(j.get('cost'))} on {j.get('gpu')}")
        for j in live:
            self.seen_live.setdefault(j["id"], " ".join(j.get("command") or []))

        if not acct.get("logged_in"):
            self.state, self.status = "problem", "not logged in — open the dashboard"
        elif s.get("error"):
            self.state, self.status = "problem", s["error"][:120]
        elif live:
            j = live[0]
            cmd = " ".join(j.get("command") or [])
            prog = j.get("progress") or {}
            pct = f" {prog['percent']:.0f}%" if prog.get("percent") is not None else ""
            more = f" (+{len(live) - 1} more)" if len(live) > 1 else ""
            self.state = "running"
            self.status = (f"{j['phase']}{pct}: {cmd[:40]}{more} · {_money(j.get('cost'))}"
                           f" · {_money(s.get('live_rate_hr'))}/hr")
        else:
            credit = (s.get("credit") or {}).get("remaining")
            self.state = "idle"
            self.status = f"idle · {_money(credit)} credit left" if credit is not None else "idle"
        self.refresh()

    def refresh(self) -> None:
        if not self.icon:
            return
        self.icon.icon = make_icon(64, self.state)
        self.icon.title = ("chuck-gpu — " + self.status)[:127]  # Windows tooltip limit
        self.icon.update_menu()

    def notify(self, title: str, msg: str) -> None:
        try:
            self.icon.notify(msg, f"chuck-gpu: {title}")
        except Exception:
            pass

    def open_dashboard(self, *_):
        webbrowser.open(self.url)

    def toggle_autostart(self, *_):
        on = not autostart_enabled()
        set_autostart(on)
        self.ui.save_settings({"start_with_windows": on})

    def quit(self, *_):
        self.stopping.set()
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        self.icon.stop()

    def run(self) -> int:
        import pystray
        from pystray import Menu, MenuItem as Item

        if bool(self.settings.get("start_with_windows", True)) != autostart_enabled():
            try:
                set_autostart(bool(self.settings.get("start_with_windows", True)))
            except OSError:
                pass
        self.start_server()
        from chuck_gpu import gpus

        def gpu_item(g: dict):
            name, cheap = g["name"], g["cheapest"]
            label = f"{name}   {g['vram_gb']} GB   ${g['rate_hr']:.2f}/hr" + ("   (cheapest)" if cheap else "")
            picked = lambda _: (gpus.get_next() or None) in ((None, name) if cheap else (name,))  # noqa: E731
            return Item(label, lambda *_: (gpus.set_next(None if cheap else name), self.icon.update_menu()),
                        checked=picked, radio=True)

        def gpu_menu():
            rates = self.ui.M.rates if self.ui.M and self.ui.M.rates else None
            yield Item("Next job only, then back to cheapest", None, enabled=False)
            yield from (gpu_item(g) for g in gpus.catalog(rates))

        menu = Menu(
            Item("Open dashboard", self.open_dashboard, default=True),
            Item(lambda _: self.status, None, enabled=False),
            Menu.SEPARATOR,
            Item(lambda _: f"GPU for next job: {gpus.get_next() or 'cheapest'}", Menu(gpu_menu)),
            Menu.SEPARATOR,
            Item("Start with Windows", self.toggle_autostart,
                 checked=lambda _: autostart_enabled()),
            Item("Quit chuck-gpu", self.quit),
        )
        self.icon = pystray.Icon("chuck-gpu", make_icon(64), "chuck-gpu — starting…", menu)
        threading.Thread(target=self.poll, daemon=True).start()
        self.icon.run()
        return 0


def main(argv: list[str] | None = None) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or getattr(sys, "frozen", False):  # windowed: no console
        sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
    if not _single_instance():
        # already running: just bring up the dashboard
        from chuck_gpu.ui import load_settings
        webbrowser.open(f"http://127.0.0.1:{int(load_settings()['ui_port'])}")
        return 0
    print(f"[chuck-gpu agent] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return Agent().run()


if __name__ == "__main__":
    sys.exit(main())
