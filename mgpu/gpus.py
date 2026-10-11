"""GPU catalog, live prices, and which GPU the next `mgpu run` uses.

The next job runs on the cheapest GPU unless you deliberately pick another:
in the dashboard / tray (a one-shot pick, used by the next run and then
cleared), with `--gpu`, or with `gpu = …` in a project's mgpu.toml.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

STATE_DIR = Path.home() / ".mgpu"
RATES_CACHE = STATE_DIR / "rates.json"
NEXT_GPU = STATE_DIR / "next_gpu.json"

# name → (Modal billing-rate key, memory GB, architecture, rough use)
CATALOG = {
    "T4":        ("gpu_hour_cost_t4",        16,  "Turing",    "inference, small training, testing"),
    "L4":        ("gpu_hour_cost_l4",        24,  "Ada",       "inference, light training (~2× T4)"),
    "A10G":      ("gpu_hour_cost_a10g",      24,  "Ampere",    "training mid-size models"),
    "L40S":      ("gpu_hour_cost_l40s",      48,  "Ada",       "large inference, fine-tuning"),
    "A100":      ("gpu_hour_cost_a100_40gb", 40,  "Ampere",    "serious training"),
    "A100-80GB": ("gpu_hour_cost_a100_80gb", 80,  "Ampere",    "big models, long contexts"),
    "H100":      ("gpu_hour_cost_h100",      80,  "Hopper",    "fastest widely available"),
    "H200":      ("gpu_hour_cost_h200",      141, "Hopper",    "H100 with more memory"),
    "B200":      ("gpu_hour_cost_b200",      192, "Blackwell", "largest models"),
    "B300":      ("gpu_hour_cost_b300",      288, "Blackwell", "next-gen largest models"),
    "MI355X":    ("gpu_hour_cost_mi355x",    288, "CDNA 4",    "AMD flagship, 288 GB HBM3"),
    "rtx6000":   ("gpu_hour_cost_rtx6000",    48, "Ada",       "pro workloads, RTX A6000 successor"),
}
GPUS = list(CATALOG)
# used only until the first live lookup (Modal list prices, $/hr)
FALLBACK_RATES = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.10,
                  "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54, "B200": 6.25,
                  "B300": 7.10, "MI355X": 6.75, "rtx6000": 3.03}


def save_rates(rates: dict[str, float]) -> None:
    """Called by the dashboard after each live rate lookup."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RATES_CACHE.write_text(json.dumps({"at": time.time(), "rates": rates}))


def gpu_rates(rates: dict[str, float] | None = None) -> dict[str, float]:
    """GPU name → $/hr, from `rates` (Modal's billing keys), the cache, or list prices."""
    if rates is None:
        try:
            rates = json.loads(RATES_CACHE.read_text())["rates"]
        except (OSError, ValueError, KeyError):
            rates = {}
    return {g: float(rates.get(key) or FALLBACK_RATES[g]) for g, (key, *_) in CATALOG.items()}


def cheapest(rates: dict[str, float] | None = None) -> str:
    r = gpu_rates(rates)
    return min(GPUS, key=lambda g: (r[g], GPUS.index(g)))


def catalog(rates: dict[str, float] | None = None) -> list[dict]:
    r = gpu_rates(rates)
    low = cheapest(rates)
    return [{"name": g, "rate_hr": r[g], "vram_gb": vram, "arch": arch, "use": use,
             "cheapest": g == low}
            for g, (_key, vram, arch, use) in CATALOG.items()]


def get_next() -> str | None:
    """The GPU picked for the next run, or None (= cheapest)."""
    try:
        g = json.loads(NEXT_GPU.read_text()).get("gpu")
    except (OSError, ValueError):
        return None
    return g if g in CATALOG else None


def set_next(gpu: str | None) -> None:
    if gpu and gpu not in CATALOG:
        raise ValueError(f"unknown GPU {gpu!r}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if gpu:
        NEXT_GPU.write_text(json.dumps({"gpu": gpu, "at": time.time()}))
    else:
        NEXT_GPU.unlink(missing_ok=True)


def take_next() -> str | None:
    """Use up the one-shot pick: return it and reset to cheapest."""
    g = get_next()
    if g:
        set_next(None)
    return g
