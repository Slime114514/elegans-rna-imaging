# -*- coding: utf-8 -*-
"""
Optional pipeline runner.

This script calls Modules 1-7 in order. Each module remains independently runnable.

Inputs:
    The inputs required by each stage, starting from raw CSV folders in INPUT_GROUPS.

Outputs:
    The stage outputs documented in 01_build_binned_data.py through
    07_response_timing_and_strength_heatmaps.py.

Depends on:
    00_config_utils.py and the seven stage modules.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_config():
    module_path = Path(__file__).resolve().with_name("00_config_utils.py")
    spec = importlib.util.spec_from_file_location("pipeline_config_utils", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()


SCRIPT_DIR = Path(__file__).resolve().parent


STEPS = [
    {
        "name": "01_build_binned_data",
        "script": "01_build_binned_data.py",
        "output": cfg.compressed_path(cfg.BINNED_DATA_DIR / "binned_data_manifest"),
    },
    {
        "name": "02_compute_delta_values",
        "script": "02_compute_delta_values.py",
        "output": cfg.DELTA_VALUES_DIR / "delta_values_summary.csv",
    },
    {
        "name": "03_group_bin_response_tests",
        "script": "03_group_bin_response_tests.py",
        "output": cfg.compressed_path(cfg.GROUP_TESTS_DIR / "group_bin_response_tests"),
    },
    {
        "name": "04_robustness_filter_significant_events",
        "script": "04_robustness_filter_significant_events.py",
        "output": cfg.compressed_path(cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust"),
    },
    {
        "name": "05_plot_robust_bin_curves",
        "script": "05_plot_robust_bin_curves.py",
        "output": cfg.compressed_path(cfg.BIN_CURVES_DIR / "robust_bin_curve_manifest"),
    },
    {
        "name": "06_expanded_region_robustness",
        "script": "06_expanded_region_robustness.py",
        "output": cfg.compressed_path(cfg.EXPANDED_REGIONS_DIR / "expanded_region_event_summary"),
    },
    {
        "name": "07_response_timing_and_strength_heatmaps",
        "script": "07_response_timing_and_strength_heatmaps.py",
        "output": cfg.compressed_path(cfg.RESPONSE_HEATMAP_DIR / "all_response_heatmap_manifest"),
    },
    {
        "name": "08_strength_distribution_tests",
        "script": "08_strength_distribution_tests.py",
        "output": cfg.STRENGTH_DISTRIBUTION_DIR / "strength_distribution_tests.csv",
    },
]


def step_output_is_current(step):
    output_path = Path(step["output"])
    if not output_path.exists():
        return False
    if step["name"] == "07_response_timing_and_strength_heatmaps":
        composite_png = cfg.RESPONSE_HEATMAP_DIR / f"{cfg.RESPONSE_COMPOSITE_OUTPUT_BASENAME}.png"
        if cfg.RESPONSE_COMPOSITE_SAVE_PNG and not composite_png.exists():
            cfg.log("Existing Module 7 manifest predates the composite figure; rerunning Module 7.")
            return False
        return True
    if step["name"] != "04_robustness_filter_significant_events":
        return True
    try:
        robust_df = cfg.read_dataframe_compressed(output_path)
    except Exception as exc:
        cfg.log(f"Existing Module 4 output could not be inspected; rerunning. Reason: {exc}")
        return False
    if "robustness_passed" not in robust_df.columns:
        cfg.log("Existing Module 4 output is missing robustness_passed; rerunning Module 4.")
        return False
    return True


def run_step(step):
    cfg.raise_if_stop_requested(f"run_all before step {step['name']}")
    script_path = SCRIPT_DIR / step["script"]
    cfg.log("=" * 80)
    cfg.log(f"[run_all_pipeline] Step: {step['name']}")
    cfg.log(f"Script: {script_path}")
    cfg.log(f"Expected output: {step['output']}")

    if step_output_is_current(step) and not cfg.OVERWRITE_EXISTING:
        cfg.log("Output exists and OVERWRITE_EXISTING=False; skipping.")
        return "skipped"

    subprocess.run([sys.executable, str(script_path)], check=True)
    cfg.raise_if_stop_requested(f"run_all after step {step['name']}")
    return "ran"


def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="run_all_pipeline")
    cfg.log("[run_all_pipeline] Starting modular pipeline.")
    results = []
    for step in STEPS:
        cfg.raise_if_stop_requested(f"run_all before step {step['name']}")
        status = run_step(step)
        results.append({"step": step["name"], "status": status, "output": str(step["output"])})

    cfg.log("=" * 80)
    cfg.log("[run_all_pipeline] Summary:")
    for rec in results:
        cfg.log(f"    {rec['step']}: {rec['status']}")
    cfg.log(f"Output root: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[run_all_pipeline] Stopped gracefully: {exc}")
        cfg.log("[run_all_pipeline] Completed outputs are kept. Clear the stop request before resuming.")

