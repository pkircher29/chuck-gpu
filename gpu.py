#!/usr/bin/env python3
"""
Chuck GPU — Modal-backed remote NVIDIA GPU compute.

  gpu ui                        Web dashboard (auto-opens browser)
  gpu run script.py [args...]   Upload & run on GPU
  gpu python [-c cmd|script]    Remote Python on GPU
  gpu shell                     Bash on a GPU box
  gpu status                    List running sandboxes
  gpu kill [id]                 Kill sandbox(es)

  --gpu T4|A10G|L40S|H100       GPU type (default T4)
  --timeout SECONDS             Sandbox timeout (default 3600)
  --token TOKEN                 Dashboard auth token (or set CHUCK_GPU_TOKEN env)

The dashboard shows live sandboxes, streaming output, live billing from your
Modal account, and a running total of money spent — persisted across sessions.

Setup:
  pip install modal aiohttp
  modal token new               # if not already authenticated
  python gpu.py ui              # opens dashboard at http://localhost:8777
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import secrets
import sys
import time
import webbrowser
from pathlib import Path

import modal

# ---- constants -------------------------------------------------------------

APP_NAME = "chuck-gpu"
VOLUME_NAME = "chuck-gpu-cache"
SPEND_FILE = Path.home() / ".chuck-gpu" / "spend.json"
CONFIG_FILE = Path.home() / ".chuck-gpu" / "config.json"

# All Modal GPUs, sorted cheapest first. VRAM in GB.
# Rates are fallbacks; live rates fetched from billing API at runtime.
GPU_INFO = {
    "T4":          {"vram": 16,  "rate": 0.59, "label": "Nvidia T4"},
    "L4":          {"vram": 24,  "rate": 0.80, "label": "Nvidia L4"},
    "A10G":        {"vram": 24,  "rate": 1.10, "label": "Nvidia A10G"},
    "L40S":        {"vram": 48,  "rate": 1.95, "label": "Nvidia L40S"},
    "A100-40GB":   {"vram": 40,  "rate": 2.10, "label": "Nvidia A100 (40 GB)"},
    "A100-80GB":   {"vram": 80,  "rate": 2.50, "label": "Nvidia A100 (80 GB)"},
    "rtx6000":     {"vram": 48,  "rate": 3.03, "label": "Nvidia RTX PRO 6000"},
    "H100":        {"vram": 80,  "rate": 3.95, "label": "Nvidia H100"},
    "H200":        {"vram": 141, "rate": 4.54, "label": "Nvidia H200"},
    "B200":        {"vram": 192, "rate": 6.25, "label": "Nvidia B200"},
    "mi355x":      {"vram": 288, "rate": 6.75, "label": "AMD MI355X"},
    "B300":        {"vram": 288, "rate": 7.10, "label": "Nvidia B300"},
}
# Modal's internal rate key names
_GPU_RATE_KEYS = {
    "T4": "gpu_hour_cost_t4", "L4": "gpu_hour_cost_l4",
    "A10G": "gpu_hour_cost_a10g", "L40S": "gpu_hour_cost_l40s",
    "A100-40GB": "gpu_hour_cost_a100_40gb", "A100-80GB": "gpu_hour_cost_a100_80gb",
    "rtx6000": "gpu_hour_cost_rtx6000", "H100": "gpu_hour_cost_h100",
    "H200": "gpu_hour_cost_h200", "B200": "gpu_hour_cost_b200",
    "mi355x": "gpu_hour_cost_mi355x", "B300": "gpu_hour_cost_b300",
}
# Cheapest GPU = default
DEFAULT_GPU = "T4"


def _get_app() -> modal.App:
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def _get_token() -> str:
    """Get the dashboard auth token from env, config, or generate one."""
    tok = os.environ.get("CHUCK_GPU_TOKEN")
    if tok:
        return tok
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            tok = cfg.get("token")
            if tok:
                return tok
        except Exception:
            pass
    tok = secrets.token_urlsafe(24)
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"token": tok}, indent=2))
    return tok


def _get_billing() -> dict:
    """Fetch live billing summary, GPU rates, and remaining credit from Modal."""
    try:
        ws = modal.Workspace.from_context()
        s = ws.billing.summary()
        metered = float(s.metered_cost)
        credits = abs(float(s.adjustments.get("Credits", 0)))
        billed = float(s.billed_cost)
        # Modal free tier: $30/mo credits. The billing summary shows
        # credits_applied = how much of that $30 was used this month.
        # billed_cost > 0 means credits ran out and you're paying cash.
        # Remaining credit = max(0, 30 - credits_used).
        FREE_CREDIT_POOL = 30.0
        if billed > 0:
            remaining_credit = 0.0
        else:
            remaining_credit = max(0, FREE_CREDIT_POOL - credits)
        live_rates = {}
        try:
            rates = ws.billing.rates()
            raw = getattr(rates, "_rates", {})
            for gpu_id, rate_key in _GPU_RATE_KEYS.items():
                val = raw.get(rate_key)
                if val is not None:
                    live_rates[gpu_id] = float(val)
                    GPU_INFO[gpu_id]["rate"] = float(val)
        except Exception:
            pass
        return {
            "metered_this_month": round(metered, 4),
            "credits_applied": round(credits, 4),
            "billed_this_month": round(billed, 4),
            "remaining_credit": round(remaining_credit, 4),
            "period_start": s.start.isoformat() if s.start else None,
            "period_end": s.end.isoformat() if s.end else None,
            "live_rates": live_rates,
        }
    except Exception as e:
        return {"error": str(e)}


def _gpu_list() -> list[dict]:
    """Return GPU info sorted cheapest first, for the dropdown."""
    out = []
    for gpu_id, info in GPU_INFO.items():
        out.append({
            "id": gpu_id,
            "label": info["label"],
            "vram": info["vram"],
            "rate": info["rate"],
        })
    out.sort(key=lambda g: g["rate"])
    return out


BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ca-certificates", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install("torch", "torchaudio", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install("numpy<2", "scipy", "soundfile", "librosa", "pyyaml", "tqdm", "onnx")
)

# ---- import detection ------------------------------------------------------

_PIP_MAP = {
    "cv2": "opencv-python", "PIL": "Pillow",
    "sklearn": "scikit-learn", "skimage": "scikit-image",
    "transformers": "transformers", "datasets": "datasets",
    "accelerate": "accelerate", "diffusers": "diffusers",
    "torchvision": "torchvision", "torchaudio": "torchaudio",
    "matplotlib": "matplotlib", "pandas": "pandas", "seaborn": "seaborn",
    "huggingface_hub": "huggingface_hub", "speechbrain": "speechbrain",
    "audiomentations": "audiomentations", "torch_audiomentations": "torch-audiomentations",
    "webrtcvad": "webrtcvad", "piper_phonemize": "piper-phonemize",
    "aiohttp": "aiohttp",
}
_BUILTIN = set(sys.stdlib_module_names) | {"modal", "__future__"}


def detect_imports(script_path: str) -> list[str]:
    try:
        tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    base = {"torch", "torchaudio", "numpy", "scipy", "soundfile", "librosa", "pyyaml", "tqdm", "onnx"}
    pkgs = []
    for mod in sorted(mods):
        if mod in _BUILTIN:
            continue
        pip = _PIP_MAP.get(mod, mod)
        if pip is None or pip in base:
            continue
        pkgs.append(pip)
    return list(dict.fromkeys(pkgs))


# ---- spend tracking --------------------------------------------------------

def _load_spend() -> dict:
    if SPEND_FILE.exists():
        try:
            return json.loads(SPEND_FILE.read_text())
        except Exception:
            pass
    return {"total_spent": 0.0, "sessions": []}


def _save_spend(data: dict) -> None:
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps(data, indent=2))


def _record_session(sb_id: str, gpu: str, started: float, ended: float, cost: float) -> None:
    data = _load_spend()
    if any(s["id"] == sb_id for s in data["sessions"]):
        return
    data["sessions"].append({
        "id": sb_id, "gpu": gpu, "rate_hr": GPU_INFO.get(gpu, {"rate": 0.59})["rate"],
        "started": started, "ended": ended, "seconds": ended - started, "cost": cost,
    })
    data["total_spent"] = sum(s["cost"] for s in data["sessions"])
    _save_spend(data)


# ---- sandbox helpers -------------------------------------------------------

def _get_vol() -> modal.Volume:
    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _list_sandboxes() -> list[dict]:
    out = []
    now = time.time()
    for sb in modal.Sandbox.list():
        tags = sb.get_tags() or {}
        gpu = tags.get("gpu_type", "T4")
        start_ts = float(tags.get("start_ts", str(now)))
        elapsed = max(0, now - start_ts)
        rate = GPU_INFO.get(gpu, {"rate": 0.59})["rate"]
        running_cost = elapsed / 3600 * rate
        rc = sb.returncode
        out.append({
            "id": sb.object_id, "gpu": gpu,
            "elapsed_secs": round(elapsed),
            "running_cost": round(running_cost, 4),
            "status": "running" if rc is None else f"exited({rc})",
            "rate_hr": rate,
        })
        if rc is not None:
            _record_session(sb.object_id, gpu, start_ts, now, running_cost)
    return out


# ---- CLI commands ----------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    script_path = str(Path(args.script).resolve())
    if not Path(script_path).exists():
        print(f"gpu: file not found: {script_path}", file=sys.stderr)
        return 1
    extra = detect_imports(script_path)
    if extra:
        print(f"gpu: auto-installing: {', '.join(extra)}")
    image = BASE_IMAGE.add_local_file(script_path, f"/root/{Path(script_path).name}")
    if extra:
        image = image.pip_install(*extra)
    vol = _get_vol()
    remote = f"/root/{Path(script_path).name}"
    cmd = ["python", remote] + args.script_args
    sb = modal.Sandbox.create(
        *cmd, app=_get_app(), image=image, gpu=args.gpu, volumes={"/root/.cache/pip": vol},
        timeout=args.timeout, workdir="/root",
        tags={"gpu_type": args.gpu, "start_ts": str(time.time())},
    )
    print(f"gpu: sandbox {sb.object_id} on {args.gpu} (${GPU_INFO.get(args.gpu, {'rate': 0.59})['rate']}/hr)")
    for chunk in sb.stdout:
        sys.stdout.write(chunk); sys.stdout.flush()
    for chunk in sb.stderr:
        sys.stderr.write(chunk); sys.stderr.flush()
    sb.wait()
    rc = sb.returncode
    if rc:
        print(f"\ngpu: exit code {rc}", file=sys.stderr)
    return rc or 0


def cmd_python(args: argparse.Namespace) -> int:
    vol = _get_vol()
    cmd = ["python"]
    if args.command:
        cmd += ["-c", args.command]
    elif args.script:
        sp = str(Path(args.script).resolve())
        if not Path(sp).exists():
            print(f"gpu: file not found: {sp}", file=sys.stderr); return 1
        cmd += [f"/root/{Path(args.script).name}"] + args.script_args
    else:
        cmd += ["-i"]
    extra = detect_imports(str(Path(args.script).resolve())) if args.script else []
    image = BASE_IMAGE
    if args.script:
        image = image.add_local_file(str(Path(args.script).resolve()), f"/root/{Path(args.script).name}")
    if extra:
        image = image.pip_install(*extra)
    sb = modal.Sandbox.create(
        *cmd, app=_get_app(), image=image, gpu=args.gpu, volumes={"/root/.cache/pip": vol},
        timeout=args.timeout, tags={"gpu_type": args.gpu, "start_ts": str(time.time())},
    )
    for chunk in sb.stdout:
        sys.stdout.write(chunk); sys.stdout.flush()
    for chunk in sb.stderr:
        sys.stderr.write(chunk); sys.stderr.flush()
    sb.wait()
    return sb.returncode or 0


def cmd_shell(args: argparse.Namespace) -> int:
    vol = _get_vol()
    sb = modal.Sandbox.create(
        "bash", app=_get_app(), image=BASE_IMAGE, gpu=args.gpu, volumes={"/root/.cache/pip": vol},
        timeout=3600, pty=True, tags={"gpu_type": args.gpu, "start_ts": str(time.time())},
    )
    print(f"gpu: shell on {args.gpu}, sandbox {sb.object_id}")
    sb.wait()
    return sb.returncode or 0


def cmd_status(args: argparse.Namespace) -> int:
    sbs = _list_sandboxes()
    if not sbs:
        print("No running sandboxes.")
    else:
        for s in sbs:
            print(f"  {s['id'][:12]}  {s['gpu']}  {s['elapsed_secs']}s  ${s['running_cost']:.4f}  {s['status']}")
    spend = _load_spend()
    print(f"  Total spent: ${spend['total_spent']:.4f}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    sbs = list(modal.Sandbox.list())
    targets = [s for s in sbs if s.object_id.startswith(args.sandbox_id)] if args.sandbox_id else sbs
    if not targets:
        print("No matching sandboxes."); return 0
    for s in targets:
        s.terminate()
        print(f"  killed {s.object_id}")
    return 0


# ---- web dashboard ---------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chuck GPU</title>
<style>
:root{--bg:#0a0e14;--sf:#11161e;--sf2:#161b22;--bd:#1e2530;--bd2:#2a3340;
--tx:#e6edf3;--t2:#7d8590;--t3:#484f58;--ac:#388bfd;--ac2:#1f6feb;
--gn:#3fb950;--rd:#f85149;--or:#d29922;--pp:#bc8cff;--mn:ui-monospace,'SF Mono',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.5}
.c{max-width:1000px;margin:0 auto;padding:24px 20px}
.header{display:flex;align-items:center;gap:12px;margin-bottom:24px}
.logo{width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,var(--ac2),var(--pp));display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:#fff}
h1{font-size:22px;font-weight:700;letter-spacing:-.3px}h1 span{color:var(--ac)}
.sub{color:var(--t2);font-size:13px;margin-bottom:20px}
.cd{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:18px;margin-bottom:14px}
.ct{font-size:12px;color:var(--t2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px;font-weight:600}
.sr{display:flex;gap:28px;flex-wrap:wrap}.st{display:flex;flex-direction:column;gap:4px}
.sv{font-size:26px;font-weight:700;font-family:var(--mn);letter-spacing:-.5px}.sl{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.sv.g{color:var(--gn)}.sv.o{color:var(--or)}.sv.a{color:var(--ac)}.sv.p{color:var(--pp)}.sv.r{color:var(--rd)}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;color:var(--t3);text-transform:uppercase;padding:10px 14px;border-bottom:1px solid var(--bd);font-weight:600;letter-spacing:.4px}
td{padding:11px 14px;border-bottom:1px solid var(--bd);font-family:var(--mn);font-size:13px}
tr:last-child td{border:none}
td.g{color:var(--gn)}td.r{color:var(--rd)}td.o{color:var(--or)}
.btn{background:var(--ac2);color:#fff;border:none;border-radius:8px;padding:9px 18px;cursor:pointer;font-size:13px;font-weight:600;transition:all .15s}
.btn:hover{background:var(--ac);transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.btn.d{background:var(--rd)}.btn.d:hover{background:#da3633}
.btn.s{background:var(--sf2);border:1px solid var(--bd2);color:var(--tx)}
.btn.s:hover{background:var(--bd2)}
.fr{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input,select{background:var(--bg);border:1px solid var(--bd2);border-radius:8px;padding:9px 14px;color:var(--tx);font-size:13px;transition:border .15s}
input:focus,select:focus{outline:none;border-color:var(--ac)}
input{flex:1;min-width:220px}
select{cursor:pointer;min-width:200px}
.lb{background:#060a10;border:1px solid var(--bd);border-radius:8px;padding:14px;font-family:var(--mn);font-size:12px;max-height:320px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;line-height:1.6}
.lb::-webkit-scrollbar{width:6px}.lb::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
.empty{color:var(--t3);font-style:italic;padding:24px;text-align:center;font-size:13px}
.gb{display:inline-block;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:600;letter-spacing:.3px}
.pulse{animation:p 2s infinite}@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}
.hr{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--bd);font-family:var(--mn);font-size:12px}
.hr:last-child{border:none}
.login{max-width:380px;margin:100px auto}
.login .cd{padding:24px}
.login input{width:100%;margin-bottom:14px;font-size:15px;padding:11px 14px}
.login .btn{width:100%;padding:11px;font-size:14px}
.hidden{display:none}
.footer{text-align:center;color:var(--t3);font-size:11px;padding:20px 0 10px}
.est-box{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:12px 16px;margin-top:12px}
.est-row{display:flex;justify-content:space-between;align-items:center}
.est-big{font-family:var(--mn);font-size:18px;font-weight:700;color:var(--or)}
.est-small{color:var(--t2);font-size:12px}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;background:var(--bd2);color:var(--t2);margin-left:6px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px}
.dot.g{background:var(--gn)}.dot.r{background:var(--rd)}.dot.o{background:var(--or)}
</style></head><body>

<div id="login-view" class="c login">
<div class="header" style="justify-content:center">
<div class="logo">CG</div>
<div><h1>Chuck <span>GPU</span></h1></div>
</div>
<p class="sub" style="text-align:center">Modal-backed remote NVIDIA GPU compute</p>
<div class="cd">
<label class="sl" style="display:block;margin-bottom:8px">Access Token</label>
<input type="password" id="token-input" placeholder="Enter your token" onkeydown="if(event.key==='Enter')doLogin()" />
<button class="btn" onclick="doLogin()">Unlock Dashboard</button>
<p class="sub" id="login-err" style="margin-top:10px;color:var(--rd);text-align:center"></p>
</div>
<p class="sub" style="text-align:center;font-size:11px">First run? Check terminal for your token, or set CHUCK_GPU_TOKEN env var.</p>
</div>

<div id="app-view" class="c hidden">
<div class="header">
<div class="logo">CG</div>
<div><h1>Chuck <span>GPU</span></h1><p class="sub" style="margin:0">Modal-backed remote NVIDIA GPU compute</p></div>
</div>

<div class="cd"><div class="ct">Modal Account</div><div class="sr">
<div class="st"><div class="sv g" id="bill-remaining">$0.00</div><div class="sl">Remaining Credit</div></div>
<div class="st"><div class="sv o" id="bill-metered">$0.00</div><div class="sl">Metered This Month</div></div>
<div class="st"><div class="sv p" id="bill-credits">$0.00</div><div class="sl">Credits Applied</div></div>
<div class="st"><div class="sv a" id="bill-billed">$0.00</div><div class="sl">Billed This Month</div></div>
<div class="st"><div class="sv" id="bill-period" style="font-size:14px">&mdash;</div><div class="sl">Billing Period</div></div>
</div></div>

<div class="cd"><div class="ct">GPU Bridge Spend</div><div class="sr">
<div class="st"><div class="sv g" id="ts">$0.00</div><div class="sl">Total Spent</div></div>
<div class="st"><div class="sv o" id="rc">$0.00</div><div class="sl">Active Now</div></div>
<div class="st"><div class="sv a" id="sc">0</div><div class="sl">Sessions</div></div>
</div></div>

<div class="cd"><div class="ct">Launch a Job</div>
<div class="fr">
<input type="text" id="sp" placeholder="/path/to/your/script.py" />
<select id="gs" onchange="updEstimate()"></select>
<button class="btn" onclick="lj()">Run on GPU</button>
</div>
<div class="est-box">
<div class="est-row">
<div><span class="est-big" id="est-cost">$0.59/hr</span><span class="tag" id="est-gpu-name">Nvidia T4</span><span class="tag" id="est-gpu-vram">16 GB VRAM</span></div>
<div class="est-small" id="est-breakdown">1hr = $0.59 &middot; 10min = $0.10 &middot; 1min = $0.0098</div>
</div>
</div>
<p class="sub" style="margin-top:10px" id="lm"></p>
</div>

<div class="cd"><div class="ct">Active Sandboxes</div><div id="stb"><div class="empty">No active sandboxes</div></div></div>
<div class="cd"><div class="ct">Output</div><div class="lb" id="log"><span class="empty">No output yet. Launch a job to see streaming output here.</span></div></div>
<div class="cd"><div class="ct">Recent Sessions</div><div id="hl"><div class="empty">No completed sessions yet</div></div></div>

<div class="footer">Chuck GPU &middot; Modal-backed remote GPU compute &middot; <a href="https://github.com/pkircher29/chuck-gpu" style="color:var(--t3)">GitHub</a></div>
</div>

<script>
let _token='',_gpus=[];
function setToken(t){_token=t;localStorage.setItem('chuck_gpu_token',t)}
function getToken(){return _token||localStorage.getItem('chuck_gpu_token')||''}
async function api(path,opts){
const h=opts&&opts.headers?opts.headers:{};
h['X-GPU-Token']=getToken();
const r=await fetch(path,{...opts,headers:h});
if(r.status===401){showLogin();throw new Error('auth')}
return r;
}
function showLogin(){document.getElementById('app-view').classList.add('hidden');document.getElementById('login-view').classList.remove('hidden')}
function showApp(){document.getElementById('login-view').classList.add('hidden');document.getElementById('app-view').classList.remove('hidden')}
async function doLogin(){
const t=document.getElementById('token-input').value.trim();
if(!t)return;
setToken(t);
try{
const r=await fetch('/api/status',{headers:{'X-GPU-Token':t}});
if(r.status===401){document.getElementById('login-err').textContent='Invalid token';return}
showApp();poll();setInterval(poll,2000);
}catch(e){document.getElementById('login-err').textContent='Connection failed'}
}
function buildDropdown(gpus){
const sel=document.getElementById('gs');
sel.innerHTML=gpus.map(g=>'<option value="'+g.id+'">'+g.label+' &mdash; $'+g.rate.toFixed(2)+'/hr &mdash; '+g.vram+' GB</option>').join('');
}
async function poll(){try{
const r=await api('/api/status');const d=await r.json();
document.getElementById('ts').textContent='$'+d.total_spent.toFixed(4);
document.getElementById('rc').textContent='$'+d.running_cost.toFixed(4);
document.getElementById('sc').textContent=d.session_count;
if(d.billing&&!d.billing.error){
document.getElementById('bill-metered').textContent='$'+d.billing.metered_this_month.toFixed(2);
document.getElementById('bill-credits').textContent='$'+d.billing.credits_applied.toFixed(2);
document.getElementById('bill-billed').textContent='$'+d.billing.billed_this_month.toFixed(2);
document.getElementById('bill-remaining').textContent='$'+d.billing.remaining_credit.toFixed(2);
if(d.billing.period_start){const s=new Date(d.billing.period_start).toLocaleDateString();const e=new Date(d.billing.period_end).toLocaleDateString();document.getElementById('bill-period').textContent=s+' \u2014 '+e}
}
if(d.gpu_list&&d.gpu_list.length){_gpus=d.gpu_list;buildDropdown(_gpus)}
updEstimate();
const t=document.getElementById('stb');
if(!d.sandboxes.length){t.innerHTML='<div class="empty">No active sandboxes</div>'}
else{let h='<table><tr><th>ID</th><th>GPU</th><th>Elapsed</th><th>Cost</th><th>Status</th><th></th></tr>';
for(const s of d.sandboxes){const c=s.status==='running'?'g':'r';
h+='<tr><td>'+s.id.slice(0,12)+'\u2026</td><td>'+s.gpu+'</td><td>'+ft(s.elapsed_secs)+'</td><td class="o">$'+s.running_cost.toFixed(4)+'</td><td class="'+c+'"><span class="dot '+c+'"></span>'+s.status+'</td><td><button class="btn d" style="padding:5px 12px;font-size:11px" onclick="ks(\''+s.id+'\')">Kill</button></td></tr>'}
t.innerHTML=h+'</table>'}
const hl=document.getElementById('hl');
if(!d.history.length){hl.innerHTML='<div class="empty">No completed sessions yet</div>'}
else{hl.innerHTML=d.history.slice(-10).reverse().map(s=>'<div class="hr"><span>'+s.gpu+' &middot; '+ft(s.seconds)+' &middot; '+new Date(s.started*1000).toLocaleString()+'</span><span class="o">$'+s.cost.toFixed(4)+'</span></div>').join('')}
if(d.logs){document.getElementById('log').textContent=d.logs}
}catch(e){if(e.message==='auth')return;console.error(e)}}
async function lj(){const p=document.getElementById('sp').value.trim();const g=document.getElementById('gs').value;
if(!p){document.getElementById('lm').textContent='Enter a script path';return}
document.getElementById('lm').textContent='Launching on '+g+'\u2026';
try{const r=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p,gpu:g})});
const d=await r.json();document.getElementById('lm').textContent=d.ok?'Launched: '+d.id:'Error: '+d.error}
catch(e){document.getElementById('lm').textContent='Error: '+e.message}
setTimeout(poll,500)}
async function ks(id){await api('/api/kill',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});setTimeout(poll,500)}
function ft(s){if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m '+(s%60)+'s';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m'}
function updEstimate(){
const g=document.getElementById('gs').value;
const gpu=_gpus.find(x=>x.id===g);
if(!gpu)return;
document.getElementById('est-cost').textContent='$'+gpu.rate.toFixed(2)+'/hr';
document.getElementById('est-gpu-name').textContent=gpu.label;
document.getElementById('est-gpu-vram').textContent=gpu.vram+' GB VRAM';
const bd=document.getElementById('est-breakdown');
bd.textContent='1hr = $'+gpu.rate.toFixed(2)+' \u00b7 10min = $'+(gpu.rate/6).toFixed(2)+' \u00b7 1min = $'+(gpu.rate/60).toFixed(4);
}
(async()=>{
const t=getToken();
if(t){try{const r=await fetch('/api/status',{headers:{'X-GPU-Token':t}});if(r.status===200){showApp();poll();setInterval(poll,2000);return}}catch{}}
showLogin();
})();
</script></body></html>"""

_log_lines: list[str] = []


def _check_token(request) -> bool:
    return request.headers.get("X-GPU-Token", "") == _get_token()


def cmd_ui(args: argparse.Namespace) -> int:
    from aiohttp import web

    token = _get_token()

    async def index(_req):
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def api_status(req):
        if not _check_token(req):
            return web.json_response({"error": "unauthorized"}, status=401)
        sbs = _list_sandboxes()
        spend = _load_spend()
        running = sum(s["running_cost"] for s in sbs if s["status"] == "running")
        billing = _get_billing()
        return web.json_response({
            "total_spent": round(spend["total_spent"] + running, 4),
            "running_cost": round(running, 4),
            "session_count": len(spend["sessions"]),
            "sandboxes": sbs,
            "history": spend["sessions"],
            "logs": "\n".join(_log_lines[-200:]),
            "gpu_list": _gpu_list(),
            "billing": billing,
        })

    async def api_run(req):
        if not _check_token(req):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            data = await req.json()
            path = data["path"]
            gpu = data.get("gpu", DEFAULT_GPU)
            sp = str(Path(path).resolve())
            if not Path(sp).exists():
                return web.json_response({"ok": False, "error": f"file not found: {sp}"})
            extra = detect_imports(sp)
            image = BASE_IMAGE.add_local_file(sp, f"/root/{Path(sp).name}")
            if extra:
                image = image.pip_install(*extra)
            vol = _get_vol()
            remote = f"/root/{Path(sp).name}"
            sb = modal.Sandbox.create(
                "python", remote, app=_get_app(), image=image, gpu=gpu,
                volumes={"/root/.cache/pip": vol}, timeout=3600, workdir="/root",
                tags={"gpu_type": gpu, "start_ts": str(time.time())},
            )
            _log_lines.append(f"[{time.strftime('%H:%M:%S')}] launched {sb.object_id} on {gpu}")
            asyncio.get_event_loop().create_task(_capture_logs(sb))
            return web.json_response({"ok": True, "id": sb.object_id})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

    async def api_kill(req):
        if not _check_token(req):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            data = await req.json()
            sid = data.get("id", "")
            for sb in modal.Sandbox.list():
                if sb.object_id.startswith(sid):
                    sb.terminate()
                    _log_lines.append(f"[{time.strftime('%H:%M:%S')}] killed {sb.object_id}")
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

    async def _capture_logs(sb):
        try:
            for chunk in sb.stdout:
                _log_lines.append(chunk.rstrip("\n"))
            for chunk in sb.stderr:
                _log_lines.append(chunk.rstrip("\n"))
        except Exception:
            pass

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/run", api_run)
    app.router.add_post("/api/kill", api_kill)
    port = args.port
    url = f"http://localhost:{port}"
    print(f"chuck-gpu: dashboard at {url}")
    print(f"chuck-gpu: token: {token}")
    webbrowser.open(url)
    web.run_app(app, host="localhost", port=port, print=lambda *a: None)
    return 0


# ---- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(prog="gpu", description="Chuck GPU — Modal-backed remote GPU compute.")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Upload & run a Python script on GPU")
    pr.add_argument("script"); pr.add_argument("script_args", nargs=argparse.REMAINDER)
    pr.add_argument("--gpu", default=DEFAULT_GPU); pr.add_argument("--timeout", type=int, default=3600)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("shell", help="Bash on GPU")
    ps.add_argument("--gpu", default=DEFAULT_GPU); ps.set_defaults(func=cmd_shell)

    pp = sub.add_parser("python", help="Remote Python on GPU")
    pp.add_argument("-c", "--command"); pp.add_argument("script", nargs="?")
    pp.add_argument("script_args", nargs=argparse.REMAINDER)
    pp.add_argument("--gpu", default=DEFAULT_GPU); pp.add_argument("--timeout", type=int, default=3600)
    pp.set_defaults(func=cmd_python)

    pst = sub.add_parser("status", help="List running sandboxes"); pst.set_defaults(func=cmd_status)

    pk = sub.add_parser("kill", help="Kill sandboxes")
    pk.add_argument("sandbox_id", nargs="?"); pk.set_defaults(func=cmd_kill)

    pu = sub.add_parser("ui", help="Web dashboard")
    pu.add_argument("--port", type=int, default=8777); pu.set_defaults(func=cmd_ui)

    args = p.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ngpu: interrupted", file=sys.stderr); return 130

if __name__ == "__main__":
    sys.exit(main())