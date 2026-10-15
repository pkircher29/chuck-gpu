# chuck-gpu — your Modal account as a CUDA GPU

Run any command on a Modal cloud GPU from your own project folder, as if the
GPU were in this PC:

```
cd my-project
chuck-gpu run -- python train.py --epochs 10
```

1. The folder is synced to a persistent Modal Volume (`chuck-gpu-workspaces`), one
   workspace per local folder. Only new/changed files upload — the first run
   of a big dataset is slow, later runs are seconds.
2. A GPU sandbox starts (≈1 s warm; first use of an image pulls it, ≈1–2 min),
   with the workspace as its working directory. Output streams back live.
3. Every file the command created or changed is copied back into your folder.
   The exit code is the command's own exit code, so it works in scripts.

Ctrl+C stops the remote GPU immediately. You pay only while the sandbox runs.

## Install

Needs Python 3.11+ and a [Modal](https://modal.com) account (free to sign
up; new accounts get monthly credit).

```
pip install git+https://github.com/pkircher29/chuck-gpu
chuck-gpu login         # once: connect your Modal account
chuck-gpu smi           # check: nvidia-smi on a cloud T4
```

### Windows app (runs in the background)

Build it (or download `chuck-gpu-windows.zip` from Releases), unzip, and run
`install.ps1`:

```
powershell -ExecutionPolicy Bypass -File packaging\build.ps1     # → dist\chuck-gpu-windows.zip
powershell -ExecutionPolicy Bypass -File dist\chuck-gpu\install.ps1  # per-user, no admin
```

That installs to `%LOCALAPPDATA%\Programs\chuck-gpu` and adds it to your PATH:

- **`chuck-gpu.exe`**: the command line, same commands as above (no Python needed).
- **`chuck-gpu-agent.exe`**: sits in the system tray and starts with Windows. It
  keeps the dashboard up at http://127.0.0.1:8770, shows the current job and
  spend in its tooltip, pops a notification when a job finishes, and stops the
  cloud GPU of a job whose terminal was closed, even with no browser open.
  Click the icon for the dashboard; right-click for **Start with Windows** and
  **Quit**.

Uninstall: `install.ps1 -Uninstall` (keeps your Modal login and job history).
Without the packaged app, `chuck-gpu agent` runs the same tray agent from Python.

## Logging in

Everything runs on — and is billed to — **your own** Modal account. Connect
it either way:

- **Dashboard:** run `chuck-gpu ui`. With no account connected it opens on a login
  screen: **Log in with Modal** (approve in the browser tab it opens), or paste
  a token ID + secret from [modal.com/settings/tokens](https://modal.com/settings/tokens).
  The header then shows the connected workspace, with **Log out**, **Add
  account**, and a switcher when you have several.
- **Terminal:** `chuck-gpu login` (browser approval), or
  `chuck-gpu login --token-id ak-… --token-secret as-…`. `chuck-gpu whoami` shows the
  account in use; `chuck-gpu logout` removes it from this machine.

The token is kept where the Modal CLI keeps it (`~/.modal.toml`, or
`$MODAL_CONFIG_PATH`), so an existing `modal setup` login works as-is, and
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` environment variables are honoured.
chuck-gpu stores no credentials of its own and sends them only to Modal.

## Commands

| | |
|---|---|
| `chuck-gpu run [opts] -- <cmd>` | sync up, run on GPU, sync results down |
| `chuck-gpu driver <serve\|exec\|test\|build>` | virtual CUDA/NVML GPU driver shim (`nvcuda.dll`/`libcuda.so.1`) + Modal RPC bridge |
| `chuck-gpu drive [--letter G:] [--gpu A100]` | mount a virtual GPU drive folder/letter that sends dropped data to Modal |
| `chuck-gpu smi [--gpu A100]` | `nvidia-smi` on a cloud GPU (quick check, cheapest by default) |
| `chuck-gpu ls` | list remote workspaces |
| `chuck-gpu rm [--key NAME]` | delete this folder's remote workspace |
| `chuck-gpu ui` | dashboard: login, live job, spend, remaining credit |
| `chuck-gpu login` / `logout` / `whoami` | manage the connected Modal account |

## Virtual GPU Driver & Drive

`chuck-gpu` includes a virtual GPU driver shim (`chuck_gpu/driver.py` + `driver_shim/`) so local programs and scripts can detect a virtual NVIDIA GPU and send data/compute to Modal:

1. **Native CUDA Driver & NVML Shim (`nvcuda.dll` / `nvml.dll` / `libcuda.so.1`)**:
   ```
   chuck-gpu driver build                  # compile native CUDA/NVML shim
   chuck-gpu driver test                   # verify CUDA Driver API & memory roundtrip
   chuck-gpu driver serve --gpu A100       # start local Modal RPC bridge on 127.0.0.1:8771
   chuck-gpu driver exec -- <command ...>  # run a binary/script with the virtual GPU driver injected
   ```
2. **Python / PyTorch Virtual GPU Hook & Offload**:
   ```python
   from chuck_gpu.driver import enable_virtual_gpu, offload

   enable_virtual_gpu("A100")  # patches torch.cuda so is_available() == True
   result = offload(my_gpu_fn, tensor_input, gpu="A100")
   ```
3. **Virtual GPU Drive (`chuck-gpu drive`)**:
   ```
   chuck-gpu drive --letter G: --gpu T4
   ```
   Creates and mounts a virtual GPU drive (`Inbox/`, `Processing/`, `Outbox/`, `Logs/`, `Driver/`, `GPU_DEVICE_INFO.json`). Any script or payload dropped into `Inbox/` is automatically sent to Modal for GPU processing and outputs are written back to `Outbox/`.

**Which GPU:** every job runs on the **cheapest** GPU (by your workspace's
live prices, normally the T4) unless you deliberately choose another. Pick one
in the dashboard's *GPU for the next job* dropdown (or the GPU table, or the
tray menu): it applies to the **next `chuck-gpu run` only**, then resets to the
cheapest. `--gpu NAME` or `gpu = "NAME"` in a project's `chuck-gpu.toml` also
choose one. Order: `--gpu` › dashboard pick › `chuck-gpu.toml` › cheapest.

`run` options: `--gpu T4|L4|A10G|L40S|A100|A100-80GB|H100|H200|B200|B300|MI355X|rtx6000`,
`--image torch|torch-devel|cuda|cuda-devel|<any registry image>`,
`--pip PKG`, `--apt PKG`, `-e KEY=VAL`, `--secret MODAL_SECRET`,
`--timeout SECONDS` (default 3600), `--cpu CORES`, `--memory MIB`,
`--resync`, `--no-download`, `--dir PATH`.

Registry images that set their own `ENTRYPOINT` (app images) work: chuck-gpu
clears it so your command runs as-is.

## Per-project defaults: `chuck-gpu.toml`

```toml
gpu = "A10G"
image = "torch"              # PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
requirements = "requirements.txt"
pip = ["transformers"]
apt = ["ffmpeg"]
setup = ["git clone https://github.com/me/tool /opt/tool"]   # extra image build steps
exclude = ["datasets/raw", "*.mp4"]
env = { HF_HOME = "/ws/.hf" }
timeout = 7200
```

`.mgpuignore` (one glob per line) also excludes files. `.git`, `.venv`,
`node_modules`, `__pycache__` are always skipped.

Excluded paths are skipped in **both** directions: a job can download a big
dataset or cache into an excluded folder of its workspace, and it stays on the
Modal volume for the next run instead of being copied back to your PC.

Also settable in `chuck-gpu.toml`: `cpu = 8`, `memory = 16384`, `add_python`,
`secrets = ["my-modal-secret"]`, `download = false`, `max_upload_gb`.

## Notes / limits

- This sends **jobs**, not individual CUDA calls — a GPU in Device Manager is
  not possible over the internet (every kernel launch would pay network
  latency). Anything you'd run as one command works.
- Local deletions are not mirrored to the remote workspace (`chuck-gpu rm` resets it).
- The upload manifest lives in `~/.chuck_gpu/manifests/`; `--resync` forces a full
  re-upload if the remote workspace was changed by hand.
- Output is streamed live and also saved on the GPU machine; if Modal's live
  stream drops during a long job, chuck-gpu keeps following that copy, so the
  terminal and dashboard never go silent while the job is still working.
  stderr is merged into stdout.

## Dashboard: `chuck-gpu ui`  →  http://127.0.0.1:8770

Starts automatically (in the background) the first time you `chuck-gpu run`.

- **Remaining credit** — monthly allowance minus the credit Modal has applied
  this month (from Modal's billing summary, refreshed every minute, minus the
  running job). Modal doesn't expose the allowance itself: enter your plan's
  monthly credit at the bottom of the dashboard (default $30, the Starter
  plan); it's saved as `monthly_credit` in `~/.chuck_gpu/config.toml`.
- **Spending now** — live $/hr of everything chuck-gpu has on a GPU.
- **This month** — all Modal usage this month (every app, not just chuck-gpu),
  plus chuck-gpu's own spend today / this month.
- **Working on** — phase (uploading → starting → running → copying back),
  progress bar, live cost ticking, **estimated total cost** and time left,
  log tail, and a **Stop GPU** button.
- **Active Sandboxes** — every live sandbox on your Modal workspace, not just
  chuck-gpu jobs. Each row shows the sandbox ID, tags, and a **Kill** button to
  terminate it immediately.
- **Recent jobs** — run time, final metered cost, result.
- **Theme toggle** — light/dark, remembered in localStorage.

Cost of each job = the sandbox's metered GPU + CPU + memory time × your
workspace's live rates. The estimate comes from, in order: a tqdm-style
progress bar in the output (`42%|…| 420/1000 [01:02<01:26]`), lines like
`epoch 3/10` / `step 120/5000`, the median of previous runs of the same
command in the same folder, or — with none of those — the cost if it runs to
the timeout.

If an `chuck-gpu run` terminal is closed mid-job, the dashboard notices and stops
the orphaned sandbox so it doesn't bill until the timeout.

`~/.chuck_gpu/config.toml`:

```toml
monthly_credit = 30.0
ui_port = 8770
auto_ui = true
```
