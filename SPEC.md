# chuck-gpu — Technical Specification

## Overview

chuck-gpu turns a [Modal](https://modal.com) account into a local-feeling cloud GPU.
Any program that runs as a command — training scripts, inference servers, data
pipelines, CUDA tools — can be dispatched to a remote NVIDIA GPU with file
sync, live output streaming, and automatic result download, as if the GPU were
in the machine running the command.

**Not for games.** chuck-gpu sends jobs (whole commands), not individual CUDA calls.
A GPU in Device Manager is not possible over the internet — every kernel launch
would pay network latency. Anything you run as one command works.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Local PC (Windows/Linux/macOS)                 │
│                                                 │
│  chuck-gpu CLI ──→ sync up ──→ Modal Sandbox ──→     │
│  chuck-gpu agent (tray)     persistent Volume        │
│  chuck-gpu ui (dashboard)   workspace per folder     │
│         ↑               ↓                       │
│         └── live output ←── stream + tee        │
│                 ↓                               │
│           sync down (changed files)             │
└─────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| CLI | `cli.py` | `chuck-gpu run`, `chuck-gpu driver`, `chuck-gpu drive`, `chuck-gpu smi`, `chuck-gpu ui`, `chuck-gpu login`, etc. |
| Driver & Bridge | `driver.py` + `driver_shim/` | Native `nvcuda.dll`/`nvml.dll`/`libcuda.so.1` shim, Modal RPC bridge (:8771), PyTorch hook, Virtual GPU Drive |
| Agent | `agent.py` | System tray icon, always-on dashboard, job watchdog, notifications |
| Dashboard | `ui.py` + `static/index.html` | HTTP server (:8770), billing, spend, GPU picker, sandbox manager |
| GPU catalog | `gpus.py` | 12 GPUs, live rates, cheapest-by-default logic, one-shot picks |
| Job records | `jobs.py` | Per-job JSON state + log files, progress parser (tqdm/epoch/step) |
| Account | `account.py` | Modal auth: browser login, token login, multi-profile, logout |

### Modal integration

- **Volume** (`chuck-gpu-workspaces`): persistent per-project file storage. One
  workspace per local folder (SHA-1 of resolved path). Incremental sync via
  manifest (`~/.chuck_gpu/manifests/<key>.json`).
- **Sandbox**: `modal.Sandbox.create("sleep", "infinity", gpu=..., image=...)`
  with workspace mounted at `/ws`. Commands run via `sb.exec()`; output
  streamed live and teed to a remote file for drop recovery.
- **Billing**: `Workspace.from_context().billing.summary()` for monthly
  metered/credits/billed; `.rates()` for live per-GPU, CPU, memory rates.
  Remaining credit = configurable monthly allowance − credits applied.
- **Sandbox listing**: `modal.Sandbox.list()` enumerates all live sandboxes
  on the workspace — not just chuck-gpu jobs — for the dashboard's sandbox panel.

## GPU Catalog

12 GPUs, sorted cheapest-first, defaulting to the cheapest (T4):

| GPU | VRAM | Architecture | $/hr | Use |
|---|---|---|---|---|
| T4 | 16 GB | Turing | 0.59 | inference, small training, testing |
| L4 | 24 GB | Ada | 0.80 | inference, light training (~2× T4) |
| A10G | 24 GB | Ampere | 1.10 | training mid-size models |
| L40S | 48 GB | Ada | 1.95 | large inference, fine-tuning |
| A100 | 40 GB | Ampere | 2.10 | serious training |
| A100-80GB | 80 GB | Ampere | 2.50 | big models, long contexts |
| rtx6000 | 48 GB | Ada | 3.03 | pro workloads |
| H100 | 80 GB | Hopper | 3.95 | fastest widely available |
| H200 | 141 GB | Hopper | 4.54 | H100 with more memory |
| B200 | 192 GB | Blackwell | 6.25 | largest models |
| MI355X | 288 GB | CDNA 4 | 6.75 | AMD flagship, 288 GB HBM3 |
| B300 | 288 GB | Blackwell | 7.10 | next-gen largest models |

Rates fetched live from `billing.rates()._rates` dict; fallbacks hardcoded.

## Dashboard (`chuck-gpu ui` → http://127.0.0.1:8770)

### Cards

- **Remaining credit**: monthly allowance − credits used, with progress meter
  and "hours of T4 left" estimate. Configurable allowance (default $30).
- **Spending now**: live $/hr across all running chuck-gpu jobs.
- **This month**: all Modal usage (every app), plus chuck-gpu's own today/month.
- **Active GPUs**: count of live sandboxes on the workspace.

### Sections

- **GPU picker**: dropdown + spec cards (price, VRAM, arch, use-case).
  Defaults to cheapest; one-shot pick for next run only.
- **Active Sandboxes**: all live sandboxes (chuck-gpu + other), with ID, tags,
  and Kill button. POST `/api/sandbox-kill`.
- **Working on**: live job cards with phase steps (upload → start → run →
  copy back), progress bar, cost breakdown (GPU/CPU/Memory split bar),
  estimated total, time left, live ticking cost, log tail, Stop button.
- **Recent jobs**: history table with command, folder, GPU, runtime, cost,
  result pill.
- **GPU table**: all 12 GPUs with live rates, clickable to pick for next run.
- **Footer**: plan credit config, cost methodology explanation.

### Theme

Light/dark via `prefers-color-scheme` + manual toggle (stored in
`localStorage["chuck-gpu-theme"]`). Toggle button in header.

## Packaging

### Build

```
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

PyInstaller spec (`packaging/chuck-gpu.spec`) produces two executables in one
folder (`dist/chuck-gpu/`):

| EXE | Type | Role |
|---|---|---|
| `chuck-gpu.exe` | console | CLI: run, smi, ui, login, etc. |
| `chuck-gpu-agent.exe` | windowed | Tray agent: always-on dashboard, watchdog, notifications |

Output: `dist/chuck-gpu/` (folder) + `dist/chuck-gpu-windows.zip` (portable archive).

### Install (per-user, no admin)

```
powershell -ExecutionPolicy Bypass -File dist\chuck-gpu\install.ps1
```

- Copies to `%LOCALAPPDATA%\Programs\chuck-gpu`
- Adds to user PATH
- Starts `chuck-gpu-agent.exe` (tray icon)
- Registers for Windows startup (`HKCU\...\Run\chuck-gpu`)
- Uninstall: `install.ps1 -Uninstall` (preserves `~/.modal.toml` and `~/.chuck_gpu/`)

### Tray agent (`chuck-gpu-agent.exe` / `chuck-gpu agent`)

- Single-instance (Windows mutex `Local\chuck-gpu-agent`)
- Keeps dashboard serving at :8770
- Polls job state every 5s; stops orphaned sandboxes (closed terminal)
- Tray icon shows running/idle/error state with live cost
- Notifications on job completion
- Right-click: Open dashboard, GPU picker, Start with Windows, Quit
- Auto-generated icon (GPU chip glyph, state-colored corner dot)

## File Layout

```
~/.modal.toml                    Modal credentials (shared with modal CLI)
~/.chuck_gpu/
  config.toml                    monthly_credit, ui_port, auto_ui
  jobs/<id>.json                 per-job state (atomic writes)
  jobs/<id>.log                  per-job output log
  jobs/<id>.stop                 stop marker (dashboard kill)
  manifests/<key>.json           upload manifest per workspace
  rates.json                     cached billing rates
  next_gpu.json                  one-shot GPU pick
  agent.log                      tray agent log
```

## API Endpoints

| Method | Path | Role |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/api/state` | Full state: account, jobs, billing, GPUs, sandboxes |
| POST | `/api/next-gpu` | Set one-shot GPU pick |
| POST | `/api/settings` | Update monthly_credit |
| POST | `/api/sandbox-kill` | Kill any sandbox by ID |
| POST | `/api/jobs/<id>/stop` | Stop a specific chuck-gpu job |
| POST | `/api/account/login-browser` | Start Modal OAuth browser flow |
| POST | `/api/account/login-token` | Login with token ID + secret |
| POST | `/api/account/login-cancel` | Cancel browser login |
| POST | `/api/account/logout` | Remove active profile |
| POST | `/api/account/switch` | Switch Modal profile |

DNS-rebinding protection: only answers requests addressed to 127.0.0.1 or
localhost. Same-origin check on POST.

## Security

- Credentials stay in `~/.modal.toml` (Modal's own config); chuck-gpu never
  stores or transmits them elsewhere.
- Dashboard binds to 127.0.0.1 only; refuses non-local Host headers.
- No credentials, tokens, or personal data in the git repo or builds.
- All billing data fetched live from the user's own Modal account.

## Dependencies

- `modal>=1.6` — Modal SDK (sandboxes, volumes, billing, auth)
- `pystray>=0.19` — system tray icon (Windows)
- `pillow>=10` — icon generation
- Python 3.11+ (uses `tomllib`)

## License

MIT
