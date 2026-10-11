# Chuck GPU

Modal-backed remote NVIDIA GPU compute. Point any Python script at it and it runs on a real GPU in the cloud — no local CUDA, no Docker, no setup beyond a Modal token.

## What it does

- **`gpu run script.py`** — uploads your script, auto-detects pip dependencies from imports, runs on a remote NVIDIA GPU, streams output back
- **`gpu ui`** — web dashboard with live sandbox monitoring, streaming output, billing info, and cost tracking
- **`gpu shell`** — bash on a GPU box
- **`gpu python -c "..."`** — one-liner Python on GPU

## Quick start

```bash
pip install modal aiohttp
modal token new          # authenticate with Modal
python gpu.py ui         # opens dashboard at http://localhost:8777
```

The first run prints an access token. Enter it in the dashboard login screen. Set `CHUCK_GPU_TOKEN` env var to use your own.

## CLI

```
gpu run script.py [args...]    Upload & run on GPU
gpu python [-c cmd|script]     Remote Python on GPU
gpu shell                      Bash on a GPU box
gpu status                     List running sandboxes + spend
gpu kill [id]                  Kill sandbox(es)
gpu ui [--port 8777]           Web dashboard

--gpu T4|A10G|L40S|H100        GPU type (default T4)
--timeout SECONDS              Sandbox timeout (default 3600)
```

## Dashboard

- **Login screen** — token-based auth, stored in `~/.chuck-gpu/config.json`
- **Modal Account** — live billing: metered cost, credits applied, billed amount, billing period
- **Spend** — total spent through Chuck GPU, active-now cost, session count
- **Launch a Job** — file picker, GPU selector with live cost estimate ($/hr, 1hr, 10min, 1min)
- **Active Sandboxes** — live elapsed time, running cost, kill button
- **Output** — streaming logs from active jobs
- **Recent Sessions** — history with cost per session

## GPU pricing

| GPU | VRAM | $/hr |
|-----|------|------|
| T4 | 16 GB | $0.59 |
| A10G | 24 GB | $1.10 |
| L40S | 48 GB | $1.95 |
| H100 | 80 GB | $3.95 |

Rates are fetched live from Modal's billing API.

## How it works

1. Your script is added to a Modal Docker image (Python 3.11 + CUDA torch)
2. Imports are AST-scanned to auto-detect pip dependencies
3. A Modal Sandbox spins up on your chosen GPU with the image
4. stdout/stderr stream back to your terminal or the dashboard
5. Cost is tracked per session and persisted in `~/.chuck-gpu/spend.json`

## Requirements

- Python 3.11+
- `modal` (`pip install modal`)
- `aiohttp` (`pip install aiohttp`) — for the dashboard
- A Modal account (free tier includes credits)

## License

MIT