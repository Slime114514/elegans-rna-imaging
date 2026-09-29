# -*- coding: utf-8 -*-
"""
Long-term pipeline runner — runs Modules 1-2 in order.

Each module is independently runnable.  Outputs are skipped when present
unless OVERWRITE_EXISTING is True.

Supports hotkey (q) and stop-file graceful termination with checkpoint/resume.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_config():
    module_path = Path(__file__).resolve().with_name("longterm_00_config_utils.py")
    spec = importlib.util.spec_from_file_location("lt_config", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()
SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "name": "01_compute_data",
        "script": "longterm_01_compute_data.py",
        "output": cfg.COMPUTED_DATA_DIR / "computed_data_manifest.txt",
    },
    {
        "name": "02_plot_figure",
        "script": "longterm_02_plot_figure.py",
        "output": cfg.OUTPUT_DIR / f"{cfg.OUTPUT_BASENAME}.png",
    },
]


def run_step(step):
    cfg.raise_if_stop_requested(f"run_all before {step['name']}")
    script_path = SCRIPT_DIR / step["script"]
    cfg.log("=" * 70)
    cfg.log(f"[run_all] Step: {step['name']}")
    cfg.log(f"  Script: {script_path}")
    cfg.log(f"  Expected output: {step['output']}")

    if Path(step["output"]).exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log("  Output exists and OVERWRITE_EXISTING=False; skipping.")
        return "skipped"

    subprocess.run([sys.executable, str(script_path)], check=True)
    cfg.raise_if_stop_requested(f"run_all after {step['name']}")
    return "ran"


def main():
    cfg.ensure_dirs()
    cfg.log("[run_all] Starting long-term modular pipeline.")
    results = []
    for step in STEPS:
        status = run_step(step)
        results.append({"step": step["name"], "status": status, "output": str(step["output"])})

    cfg.log("=" * 70)
    cfg.log("[run_all] Summary:")
    for rec in results:
        cfg.log(f"    {rec['step']}: {rec['status']}")
    cfg.log(f"  Output root: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[run_all] Stopped gracefully: {exc}")
        cfg.log("[run_all] Completed outputs are kept. Clear the stop request before resuming.")
