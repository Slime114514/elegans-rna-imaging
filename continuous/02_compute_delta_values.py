# -*- coding: utf-8 -*-
"""
Module 2: compute instance-level deltaF and deltaF/F0 for each s-r bin over time.

Inputs:
    OUTPUT_DIR/01_binned_data/<gene__condition>/per_instance_bin_stats.pkl.gz

Outputs:
    OUTPUT_DIR/02_delta_values/<gene__condition>/per_instance_bin_delta_values.pkl.gz
    OUTPUT_DIR/02_delta_values/<gene__condition>/per_instance_bin_baseline_values.pkl.gz
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    OUTPUT_DIR/02_delta_values/delta_values_summary.csv

Depends on:
    01_build_binned_data.py output.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_config():
    module_path = Path(__file__).resolve().with_name("00_config_utils.py")
    spec = importlib.util.spec_from_file_location("pipeline_config_utils", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()


INSTANCE_BIN_KEYS = [
    "gene_id",
    "gene_label",
    "condition",
    "condition_label",
    "instance",
    "instance_uid",
    "s_bin",
    "r_bin",
]


def compute_delta_values(binned_df: pd.DataFrame):
    metric = cfg.PRIMARY_METRIC_FOR_DELTA
    if metric not in {"bin_mean_h", "bin_signal_sum_h"}:
        raise ValueError(
            f"PRIMARY_METRIC_FOR_DELTA must be 'bin_mean_h' or 'bin_signal_sum_h', got {metric!r}"
        )
    if metric not in binned_df.columns:
        raise ValueError(f"Input binned table is missing metric column: {metric}")

    front_cols = [
        "gene_id",
        "gene_label",
        "condition",
        "condition_label",
        "instance",
        "instance_uid",
        "time_min",
        "csv_uid",
        "file",
        "s_bin",
        "r_bin",
        "s_start",
        "s_end",
        "r_start",
        "r_end",
        "s_center",
        "r_center",
    ]
    needed_cols = [c for c in front_cols + [metric] if c in binned_df.columns]
    # Keep Module 2 memory bounded: carry only columns needed by later stages.
    work = binned_df.loc[:, needed_cols].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")
    work = work.sort_values(INSTANCE_BIN_KEYS + ["time_min"])

    valid_mask = work[metric].notna() & work["time_min"].notna()
    valid_for_baseline = work.loc[valid_mask, INSTANCE_BIN_KEYS + ["time_min", metric]]
    if valid_for_baseline.empty:
        baseline_values = pd.DataFrame(columns=INSTANCE_BIN_KEYS + ["baseline_time_min", "F0", "F0_valid_for_dff0"])
        delta_values = work.copy()
        delta_values["F_t"] = np.nan
        delta_values["baseline_time_min"] = np.nan
        delta_values["F0"] = np.nan
        delta_values["F0_valid_for_dff0"] = False
        delta_values["deltaF"] = np.nan
        delta_values["dff0"] = np.nan
        return delta_values, baseline_values

    baseline_values = (
        valid_for_baseline.groupby(INSTANCE_BIN_KEYS, sort=False)
        .first()
        .reset_index()[INSTANCE_BIN_KEYS + ["time_min", metric]]
        .rename(columns={"time_min": "baseline_time_min", metric: "F0"})
    )
    baseline_values["F0_valid_for_dff0"] = baseline_values["F0"] >= cfg.MIN_F0_FOR_DFF0

    delta_values = work.merge(baseline_values, on=INSTANCE_BIN_KEYS, how="inner")
    delta_values["F_t"] = delta_values[metric].astype(float)
    delta_values["deltaF"] = delta_values["F_t"] - delta_values["F0"]
    delta_values["dff0"] = np.where(
        delta_values["F0_valid_for_dff0"] & np.isfinite(delta_values["deltaF"]),
        delta_values["deltaF"] / delta_values["F0"],
        np.nan,
    )

    output_front_cols = front_cols + ["baseline_time_min", "F0", "F0_valid_for_dff0", "F_t", "deltaF", "dff0"]
    ordered_cols = [c for c in output_front_cols if c in delta_values.columns]
    delta_values = delta_values[ordered_cols].sort_values(
        ["gene_id", "condition", "instance_uid", "r_bin", "s_bin", "time_min"]
    )
    baseline_values = baseline_values.sort_values(["gene_id", "condition", "instance_uid", "r_bin", "s_bin"])
    return delta_values, baseline_values


def summarize_delta_values(delta_df: pd.DataFrame, baseline_df: pd.DataFrame):
    if baseline_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "gene_label",
                "condition",
                "condition_label",
                "n_instance_bins_with_deltaF",
                "n_instance_bins_f0_too_low_for_dff0",
                "n_delta_rows",
                "n_dff0_non_na_rows",
            ]
        )

    baseline_summary = (
        baseline_df.groupby(["gene_id", "gene_label", "condition", "condition_label"], as_index=False)
        .agg(
            n_instance_bins_with_deltaF=("F0", "size"),
            n_instance_bins_f0_too_low_for_dff0=("F0_valid_for_dff0", lambda x: int((~x.astype(bool)).sum())),
        )
    )
    row_summary = (
        delta_df.groupby(["gene_id", "gene_label", "condition", "condition_label"], as_index=False)
        .agg(
            n_delta_rows=("deltaF", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            n_dff0_non_na_rows=("dff0", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        )
    )
    return baseline_summary.merge(
        row_summary,
        on=["gene_id", "gene_label", "condition", "condition_label"],
        how="left",
    )



def process_one_group(group: dict):
    cfg.raise_if_stop_requested(f"module 2 before group {group['gene_id']} | {group['condition']}")
    group_name = cfg.group_folder_name(group)
    input_path = cfg.compressed_path(cfg.BINNED_DATA_DIR / group_name / "per_instance_bin_stats")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Module 1 output for {group_name}: {input_path}")

    group_output_dir = cfg.DELTA_VALUES_DIR / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)
    delta_base = group_output_dir / "per_instance_bin_delta_values"
    baseline_base = group_output_dir / "per_instance_bin_baseline_values"
    delta_path = cfg.compressed_path(delta_base)
    baseline_path = cfg.compressed_path(baseline_base)

    if delta_path.exists() and baseline_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"    Existing group delta output found; skipping {group_name}")
        delta_df = cfg.read_dataframe_compressed(delta_path)
        baseline_df = cfg.read_dataframe_compressed(baseline_path)
    else:
        cfg.log(f"    Reading binned stats: {input_path}")
        binned_df = cfg.read_dataframe_compressed(input_path)
        cfg.log(f"    {group_name}: binned rows loaded = {len(binned_df)}")
        delta_df, baseline_df = compute_delta_values(binned_df)
        cfg.save_dataframe_compressed(delta_df, delta_base, f"{group_name} delta values")
        cfg.save_dataframe_compressed(baseline_df, baseline_base, f"{group_name} baseline values")

        if cfg.SAVE_DELTA_VALUES_CSV:
            delta_df.to_csv(group_output_dir / "per_instance_bin_delta_values.csv", index=False, encoding="utf-8-sig")
        if cfg.SAVE_BASELINE_VALUES_CSV:
            baseline_df.to_csv(
                group_output_dir / "per_instance_bin_baseline_values.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary_df = summarize_delta_values(delta_df, baseline_df)
    return {
        "gene_id": group["gene_id"],
        "gene_label": group["gene_label"],
        "condition": group["condition"],
        "condition_label": group["condition_label"],
        "group_name": group_name,
        "delta_values_path": str(delta_path),
        "baseline_values_path": str(baseline_path),
        "n_delta_rows": int(len(delta_df)),
        "n_baseline_rows": int(len(baseline_df)),
        "n_dff0_non_na_rows": int(pd.to_numeric(delta_df["dff0"], errors="coerce").notna().sum())
        if "dff0" in delta_df.columns
        else 0,
        "n_instance_bins_f0_too_low_for_dff0": int((~baseline_df["F0_valid_for_dff0"].astype(bool)).sum())
        if "F0_valid_for_dff0" in baseline_df.columns
        else 0,
    }, summary_df



def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="02_compute_delta_values")

    manifest_base = cfg.DELTA_VALUES_DIR / "delta_values_manifest"
    manifest_path = cfg.compressed_path(manifest_base)
    summary_path = cfg.DELTA_VALUES_DIR / "delta_values_summary.csv"

    if manifest_path.exists() and summary_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 2 output already exists; skipping: {manifest_path}")
        return

    if manifest_path.exists() and not summary_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log("Module 2 manifest exists but summary is missing; rebuilding summary from manifest.")
        manifest_df = cfg.read_dataframe_compressed(manifest_path)
        summary_cols = [
            "gene_id",
            "gene_label",
            "condition",
            "condition_label",
            "n_baseline_rows",
            "n_instance_bins_f0_too_low_for_dff0",
            "n_delta_rows",
            "n_dff0_non_na_rows",
        ]
        summary_df = manifest_df[[c for c in summary_cols if c in manifest_df.columns]].copy()
        if "n_baseline_rows" in summary_df.columns:
            summary_df = summary_df.rename(columns={"n_baseline_rows": "n_instance_bins_with_deltaF"})
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        cfg.log(f"    Rebuilt missing summary: {summary_path}")
        return

    cfg.log("[Module 2] Computing deltaF and deltaF/F0 group by group.")
    manifest_records = []
    summary_parts = []
    for group_index, group in enumerate(cfg.INPUT_GROUPS, start=1):
        cfg.raise_if_stop_requested(f"module 2 before group {group_index}/{len(cfg.INPUT_GROUPS)}")
        cfg.log("=" * 80)
        cfg.log(f"[Module 2 group {group_index}/{len(cfg.INPUT_GROUPS)}] {group['gene_id']} | {group['condition']}")
        manifest_record, summary_part = process_one_group(group)
        manifest_records.append(manifest_record)
        summary_parts.append(summary_part)
        cfg.raise_if_stop_requested(f"module 2 after group {group_index}/{len(cfg.INPUT_GROUPS)}")

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(cfg.DELTA_VALUES_DIR / "delta_values_manifest.csv", index=False, encoding="utf-8-sig")
    cfg.save_dataframe_compressed(manifest_df, manifest_base, "delta values manifest")

    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    summary_df.to_csv(cfg.DELTA_VALUES_DIR / "delta_values_summary.csv", index=False, encoding="utf-8-sig")

    cfg.log("[Module 2] Summary:")
    cfg.log(f"    Delta rows: {int(manifest_df['n_delta_rows'].sum()) if not manifest_df.empty else 0}")
    cfg.log(f"    Instance-bin baselines: {int(manifest_df['n_baseline_rows'].sum()) if not manifest_df.empty else 0}")
    cfg.log(
        "    F0 too low for dff0: "
        f"{int(manifest_df['n_instance_bins_f0_too_low_for_dff0'].sum()) if not manifest_df.empty else 0}"
    )
    cfg.log(f"    Output: {cfg.DELTA_VALUES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 2] Stopped gracefully: {exc}")
        cfg.log("[Module 2] Completed outputs are kept. Clear the stop request before resuming.")



