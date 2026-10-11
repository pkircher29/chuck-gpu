# PyInstaller spec: one folder, two executables sharing it.
#   mgpu.exe        console — the CLI (run, login, ui, …)
#   mgpu-agent.exe  windowed — tray icon + always-on dashboard
# Build with packaging\build.ps1.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [("../mgpu/static", "mgpu/static")], [], []
for pkg in ("modal", "modal_proto", "synchronicity", "grpclib", "certifi", "rich", "click"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
hidden += collect_submodules("mgpu") + ["toml", "pystray._win32", "PIL.ImageDraw"]

common = dict(pathex=[".."], binaries=binaries, datas=datas, hiddenimports=hidden,
              excludes=["tkinter", "matplotlib", "numpy", "pytest"])
a_cli = Analysis(["entry_cli.py"], **common)
a_agent = Analysis(["entry_agent.py"], **common)

exe_cli = EXE(PYZ(a_cli.pure), a_cli.scripts, [], exclude_binaries=True, name="mgpu",
              console=True, icon="mgpu.ico")
exe_agent = EXE(PYZ(a_agent.pure), a_agent.scripts, [], exclude_binaries=True, name="mgpu-agent",
                console=False, icon="mgpu.ico")
COLLECT(exe_cli, exe_agent,
        a_cli.binaries, a_cli.datas, a_agent.binaries, a_agent.datas,
        name="mgpu")
