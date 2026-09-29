# -*- coding: utf-8 -*-
"""
Module 6: expanded spatial-temporal region analysis for robust significant events.

Inputs:
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    and per-group delta chunks listed in that manifest.
    OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.pkl.gz

Outputs:
    OUTPUT_DIR/06_expanded_regions/expanded_region_instance_metrics.pkl.gz
    OUTPUT_DIR/06_expanded_regions/expanded_region_instance_metrics.csv
    OUTPUT_DIR/06_expanded_regions/expanded_region_event_summary.pkl.gz
    OUTPUT_DIR/06_expanded_regions/expanded_region_event_summary.csv

Depends on:
    02_compute_delta_values.py and 04_robustness_filter_significant_events.py outputs.
"""

import importlib.util
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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

_EXPANDED_DELTA_DF = None


def init_expanded_worker(delta_df):
    global _EXPANDED_DELTA_DF
    _EXPANDED_DELTA_DF = delta_df


def top_contributions(delta_values):
    arr = np.asarray(delta_values, dtype=float)
    positives = np.sort(arr[arr > 0])[::-1]
    positive_sum = float(np.sum(positives))
    if positive_sum <= 0 or len(positives) == 0:
        return np.nan, np.nan
    top1 = float(positives[0] / positive_sum)
    top2 = float(np.sum(positives[:2]) / positive_sum)
    return top1, top2


def leave_one_out_delta(instance_df: pd.DataFrame):
    valid = instance_df.dropna(subset=["region_delta_at_event"]).copy()
    if len(valid) <= 1:
        return np.nan, False
    means = []
    for instance_uid in valid["instance_uid"]:
        remaining = valid.loc[valid["instance_uid"] != instance_uid, "region_delta_at_event"].to_numpy(dtype=float)
        if len(remaining):
            means.append(float(np.mean(remaining)))
    means = np.asarray(means, dtype=float)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return np.nan, False
    return float(np.min(means)), bool(np.all(means > 0))


def aggregate_region_f(region_rows: pd.DataFrame):
    metric = "F_t"
    if cfg.PRIMARY_METRIC_FOR_DELTA == "bin_mean_h":
        agg = (
            region_rows.groupby(["instance_uid", "instance", "time_min"], as_index=False)
            .agg(region_F=(metric, "mean"), n_region_bins_observed=(metric, "count"))
            .sort_values(["instance_uid", "time_min"])
        )
        agg["region_aggregation"] = "mean_bin_F_t"
    else:
        agg = (
            region_rows.groupby(["instance_uid", "instance", "time_min"], as_index=False)
            .agg(region_F=(metric, "sum"), n_region_bins_observed=(metric, "count"))
            .sort_values(["instance_uid", "time_min"])
        )
        agg["region_aggregation"] = "sum_bin_F_t"
    return agg


def compute_one_event(event: dict):
    event_region_df = event.get("_region_delta_df")
    if event_region_df is not None:
        delta_df = event_region_df
        event = {k: v for k, v in event.items() if k != "_region_delta_df"}
    else:
        delta_df = _EXPANDED_DELTA_DF
    event_id = int(event.get("event_id", -1))
    gene_id = event["gene_id"]
    condition = event["condition"]
    center_s = int(event["s_bin"])
    center_r = int(event["r_bin"])
    center_time = float(event["time_min"])

    s_min = max(0, center_s - cfg.EXPANSION_BIN_RADIUS)
    s_max = min(cfg.N_S_BINS - 1, center_s + cfg.EXPANSION_BIN_RADIUS)
    r_min = max(0, center_r - cfg.EXPANSION_BIN_RADIUS)
    r_max = min(cfg.N_R_BINS - 1, center_r + cfg.EXPANSION_BIN_RADIUS)
    t_min = center_time - cfg.EXPANSION_TIME_RADIUS_MIN
    t_max = center_time + cfg.EXPANSION_TIME_RADIUS_MIN

    mask = (
        (delta_df["gene_id"] == gene_id)
        & (delta_df["condition"] == condition)
        & (delta_df["s_bin"].astype(int) >= s_min)
        & (delta_df["s_bin"].astype(int) <= s_max)
        & (delta_df["r_bin"].astype(int) >= r_min)
        & (delta_df["r_bin"].astype(int) <= r_max)
    )
    region_rows = delta_df.loc[mask].copy()
    if region_rows.empty:
        return [], build_event_summary(event, pd.DataFrame(), s_min, s_max, r_min, r_max, t_min, t_max)

    region_rows["F_t"] = pd.to_numeric(region_rows["F_t"], errors="coerce")
    region_rows["time_min"] = pd.to_numeric(region_rows["time_min"], errors="coerce")
    region_rows = region_rows.dropna(subset=["F_t", "time_min"])
    if region_rows.empty:
        return [], build_event_summary(event, pd.DataFrame(), s_min, s_max, r_min, r_max, t_min, t_max)

    region_ts = aggregate_region_f(region_rows)
    baseline = (
        region_ts.dropna(subset=["region_F", "time_min"])
        .sort_values(["instance_uid", "time_min"])
        .groupby(["instance_uid", "instance"], as_index=False)
        .first()[["instance_uid", "instance", "time_min", "region_F"]]
        .rename(columns={"time_min": "baseline_region_time_min", "region_F": "baseline_region_F"})
    )
    region_ts = region_ts.merge(baseline, on=["instance_uid", "instance"], how="inner")
    region_ts["region_delta"] = region_ts["region_F"] - region_ts["baseline_region_F"]
    region_ts["region_dff0"] = np.where(
        region_ts["baseline_region_F"] >= cfg.MIN_F0_FOR_DFF0,
        region_ts["region_delta"] / region_ts["baseline_region_F"],
        np.nan,
    )

    window_ts = region_ts.loc[(region_ts["time_min"] >= t_min) & (region_ts["time_min"] <= t_max)].copy()
    instance_records = []
    for (instance_uid, instance), g_inst_all in region_ts.groupby(["instance_uid", "instance"], sort=False):
        g_inst_window = window_ts.loc[window_ts["instance_uid"] == instance_uid].sort_values("time_min")
        event_row = g_inst_all.loc[np.isclose(g_inst_all["time_min"].to_numpy(dtype=float), center_time)]
        if event_row.empty:
            event_region_F = np.nan
            region_delta_at_event = np.nan
            region_dff0_at_event = np.nan
            n_region_bins_at_event = 0
        else:
            event_first = event_row.iloc[0]
            event_region_F = float(event_first["region_F"])
            region_delta_at_event = float(event_first["region_delta"])
            region_dff0_at_event = float(event_first["region_dff0"]) if pd.notna(event_first["region_dff0"]) else np.nan
            n_region_bins_at_event = int(event_first["n_region_bins_observed"])

        if g_inst_window.empty:
            peak_delta = np.nan
            peak_time = np.nan
            auc = np.nan
            auc_positive = np.nan
            n_window_timepoints = 0
        else:
            deltas = pd.to_numeric(g_inst_window["region_delta"], errors="coerce").to_numpy(dtype=float)
            times = pd.to_numeric(g_inst_window["time_min"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(deltas) & np.isfinite(times)
            deltas = deltas[valid]
            times = times[valid]
            n_window_timepoints = int(len(times))
            if len(deltas):
                peak_idx = int(np.nanargmax(deltas))
                peak_delta = float(deltas[peak_idx])
                peak_time = float(times[peak_idx])
            else:
                peak_delta = np.nan
                peak_time = np.nan
            if len(times) >= 2:
                order = np.argsort(times)
                times_sorted = times[order]
                deltas_sorted = deltas[order]
                auc = float(np.trapz(deltas_sorted, times_sorted))
                auc_positive = float(np.trapz(np.maximum(deltas_sorted, 0.0), times_sorted))
            elif len(times) == 1:
                auc = 0.0
                auc_positive = 0.0
            else:
                auc = np.nan
                auc_positive = np.nan

        base_row = baseline.loc[baseline["instance_uid"] == instance_uid].iloc[0]
        instance_records.append(
            {
                "event_id": event_id,
                "gene_id": gene_id,
                "gene_label": event.get("gene_label", gene_id),
                "condition": condition,
                "condition_label": event.get("condition_label", condition),
                "center_s_bin": center_s,
                "center_r_bin": center_r,
                "center_time_min": center_time,
                "region_s_bin_min": s_min,
                "region_s_bin_max": s_max,
                "region_r_bin_min": r_min,
                "region_r_bin_max": r_max,
                "region_time_min": t_min,
                "region_time_max": t_max,
                "instance_uid": instance_uid,
                "instance": instance,
                "baseline_region_time_min": float(base_row["baseline_region_time_min"]),
                "baseline_region_F": float(base_row["baseline_region_F"]),
                "event_region_F": event_region_F,
                "region_delta_at_event": region_delta_at_event,
                "region_dff0_at_event": region_dff0_at_event,
                "n_region_bins_at_event": n_region_bins_at_event,
                "region_peak_delta": peak_delta,
                "region_peak_time": peak_time,
                "region_AUC": auc,
                "region_AUC_positive": auc_positive,
                "region_positive_delta": bool(np.isfinite(region_delta_at_event) and region_delta_at_event > 0),
                "n_window_timepoints": n_window_timepoints,
                "region_aggregation": g_inst_all["region_aggregation"].iloc[0],
            }
        )

    instance_df = pd.DataFrame(instance_records)
    summary = build_event_summary(event, instance_df, s_min, s_max, r_min, r_max, t_min, t_max)
    return instance_records, summary


def build_event_summary(event, instance_df, s_min, s_max, r_min, r_max, t_min, t_max):
    out = dict(event)
    out.update(
        {
            "event_id": int(event.get("event_id", -1)),
            "center_s_bin": int(event.get("s_bin")),
            "center_r_bin": int(event.get("r_bin")),
            "center_time_min": float(event.get("time_min")),
            "region_s_bin_min": int(s_min),
            "region_s_bin_max": int(s_max),
            "region_r_bin_min": int(r_min),
            "region_r_bin_max": int(r_max),
            "region_time_min": float(t_min),
            "region_time_max": float(t_max),
            "expansion_bin_radius": cfg.EXPANSION_BIN_RADIUS,
            "expansion_time_radius_min": cfg.EXPANSION_TIME_RADIUS_MIN,
        }
    )
    if instance_df.empty:
        out.update(
            {
                "region_n_instances": 0,
                "region_mean_delta": np.nan,
                "region_median_delta": np.nan,
                "region_peak_delta_mean": np.nan,
                "region_AUC_mean": np.nan,
                "region_AUC_positive_mean": np.nan,
                "region_positive_ratio": np.nan,
                "region_top1_contribution": np.nan,
                "region_top2_contribution": np.nan,
                "region_leave_one_out_min_delta": np.nan,
                "region_leave_one_out_all_positive": False,
            }
        )
        return out

    delta = pd.to_numeric(instance_df["region_delta_at_event"], errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(delta))
    n_positive = int(np.sum(delta > 0)) if n else 0
    top1, top2 = top_contributions(delta)
    loo_min, loo_all_positive = leave_one_out_delta(instance_df)

    out.update(
        {
            "region_n_instances": n,
            "region_mean_delta": float(np.mean(delta)) if n else np.nan,
            "region_median_delta": float(np.median(delta)) if n else np.nan,
            "region_peak_delta_mean": float(pd.to_numeric(instance_df["region_peak_delta"], errors="coerce").mean()),
            "region_AUC_mean": float(pd.to_numeric(instance_df["region_AUC"], errors="coerce").mean()),
            "region_AUC_positive_mean": float(
                pd.to_numeric(instance_df["region_AUC_positive"], errors="coerce").mean()
            ),
            "region_positive_ratio": n_positive / n if n else np.nan,
            "region_top1_contribution": top1,
            "region_top2_contribution": top2,
            "region_leave_one_out_min_delta": loo_min,
            "region_leave_one_out_all_positive": loo_all_positive,
        }
    )
    return out




def _event_region_subset(delta_df: pd.DataFrame, event: dict):
    center_s = int(event["s_bin"])
    center_r = int(event["r_bin"])
    s_min = max(0, center_s - cfg.EXPANSION_BIN_RADIUS)
    s_max = min(cfg.N_S_BINS - 1, center_s + cfg.EXPANSION_BIN_RADIUS)
    r_min = max(0, center_r - cfg.EXPANSION_BIN_RADIUS)
    r_max = min(cfg.N_R_BINS - 1, center_r + cfg.EXPANSION_BIN_RADIUS)
    needed = [
        "gene_id", "gene_label", "condition", "condition_label", "instance", "instance_uid",
        "time_min", "s_bin", "r_bin", "F_t",
    ]
    work = delta_df.loc[
        (delta_df["s_bin"].astype(int) >= s_min)
        & (delta_df["s_bin"].astype(int) <= s_max)
        & (delta_df["r_bin"].astype(int) >= r_min)
        & (delta_df["r_bin"].astype(int) <= r_max),
        [c for c in needed if c in delta_df.columns],
    ].copy()
    return work


def _prepare_expanded_event(delta_df: pd.DataFrame, event: dict):
    prepared = dict(event)
    prepared["_region_delta_df"] = _event_region_subset(delta_df, event)
    return prepared


def run_expanded_analysis(delta_df: pd.DataFrame, robust_df: pd.DataFrame):
    if robust_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    robust_df = robust_df.copy().reset_index(drop=True)
    if "event_id" not in robust_df.columns:
        robust_df.insert(0, "event_id", np.arange(1, len(robust_df) + 1, dtype=int))
    event_records = robust_df.to_dict(orient="records")

    workers = min(int(cfg.EXPANDED_REGION_WORKERS), len(event_records))
    instance_records = []
    summary_records = []
    if not (cfg.USE_MULTIPROCESSING and workers > 1 and len(event_records) > 1):
        cfg.log("    Expanded-region analysis sequentially.")
        init_expanded_worker(delta_df)
        for i, event in enumerate(event_records, start=1):
            cfg.raise_if_stop_requested(f"module 6 before expanded event {i}/{len(event_records)}")
            inst_rows, summary = compute_one_event(event)
            instance_records.extend(inst_rows)
            summary_records.append(summary)
            if i == 1 or i == len(event_records) or (
                cfg.PROGRESS_EVERY_N_EVENTS and i % cfg.PROGRESS_EVERY_N_EVENTS == 0
            ):
                cfg.log(f"        Expanded events finished: {i}/{len(event_records)}")
        return pd.DataFrame(instance_records), pd.DataFrame(summary_records)

    max_pending = min(int(getattr(cfg, "EXPANDED_REGION_MAX_PENDING_EVENTS", workers)), len(event_records))
    cfg.log(f"    Parallel expanded-region analysis enabled with small per-event region slices: workers={workers}, max_pending={max_pending}")
    next_i = 0
    finished = 0
    pending = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while next_i < len(event_records) and len(pending) < max_pending:
            cfg.raise_if_stop_requested(f"module 6 before submitting expanded event {next_i + 1}/{len(event_records)}")
            prepared = _prepare_expanded_event(delta_df, event_records[next_i])
            future = executor.submit(compute_one_event, prepared)
            pending[future] = next_i + 1
            next_i += 1

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                inst_rows, summary = future.result()
                instance_records.extend(inst_rows)
                summary_records.append(summary)
                finished += 1
                if finished == 1 or finished == len(event_records) or (
                    cfg.PROGRESS_EVERY_N_EVENTS and finished % cfg.PROGRESS_EVERY_N_EVENTS == 0
                ):
                    cfg.log(f"        Expanded events finished: {finished}/{len(event_records)}")
                stop_requested = cfg.should_stop_requested()
                if (not stop_requested) and next_i < len(event_records):
                    cfg.raise_if_stop_requested(f"module 6 before submitting expanded event {next_i + 1}/{len(event_records)}")
                    prepared = _prepare_expanded_event(delta_df, event_records[next_i])
                    new_future = executor.submit(compute_one_event, prepared)
                    pending[new_future] = next_i + 1
                    next_i += 1
                elif stop_requested and not pending:
                    cfg.raise_if_stop_requested(f"module 6 after expanded event {finished}/{len(event_records)}")

    return pd.DataFrame(instance_records), pd.DataFrame(summary_records)


def load_delta_manifest():
    manifest_path = cfg.compressed_path(cfg.DELTA_VALUES_DIR / "delta_values_manifest")
    if manifest_path.exists():
        return cfg.read_dataframe_compressed(manifest_path)

    legacy_delta_path = cfg.compressed_path(cfg.DELTA_VALUES_DIR / "per_instance_bin_delta_values")
    if legacy_delta_path.exists():
        return pd.DataFrame(
            [
                {
                    "gene_id": "ALL",
                    "condition": "ALL",
                    "group_name": "ALL",
                    "delta_values_path": str(legacy_delta_path),
                }
            ]
        )
    raise FileNotFoundError(
        "Missing Module 2 delta manifest. Run the patched 02_compute_delta_values.py first."
    )


def run_expanded_analysis_from_manifest(delta_manifest: pd.DataFrame, robust_df: pd.DataFrame):
    if robust_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "ALL" in set(delta_manifest.get("gene_id", [])):
        delta_df = cfg.read_dataframe_compressed(Path(delta_manifest.iloc[0]["delta_values_path"]))
        delta_df, _ = cfg.filter_late_baseline_bins(delta_df)
        return run_expanded_analysis(delta_df, robust_df)

    instance_parts = []
    summary_parts = []
    for (gene_id, condition), robust_g in robust_df.groupby(["gene_id", "condition"], sort=False):
        cfg.raise_if_stop_requested(f"module 6 before group {gene_id} | {condition}")
        match = delta_manifest.loc[
            (delta_manifest["gene_id"] == gene_id) & (delta_manifest["condition"] == condition)
        ]
        if match.empty:
            raise FileNotFoundError(f"No delta chunk found for {gene_id} | {condition}")
        delta_path = Path(match.iloc[0]["delta_values_path"])
        cfg.log(f"    Reading delta chunk for expanded regions: {delta_path}")
        delta_df = cfg.read_dataframe_compressed(delta_path)
        delta_df, _ = cfg.filter_late_baseline_bins(delta_df)
        inst_g, summary_g = run_expanded_analysis(delta_df, robust_g)
        instance_parts.append(inst_g)
        summary_parts.append(summary_g)

    instance_df = pd.concat(instance_parts, ignore_index=True) if instance_parts else pd.DataFrame()
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    return instance_df, summary_df


def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="06_expanded_region_robustness")

    instance_base = cfg.EXPANDED_REGIONS_DIR / "expanded_region_instance_metrics"
    instance_path = cfg.compressed_path(instance_base)
    if instance_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 6 output already exists; skipping: {instance_path}")
        return

    robust_path = cfg.compressed_path(cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust")
    cfg.log("[Module 6] Reading Module 2 delta manifest.")
    delta_manifest = load_delta_manifest()
    cfg.log(f"[Module 6] Reading robust significant events: {robust_path}")
    robust_df = cfg.read_dataframe_compressed(robust_path)

    cfg.log("[Module 6] Running expanded spatial-temporal region analysis.")
    instance_df, summary_df = run_expanded_analysis_from_manifest(delta_manifest, robust_df)

    cfg.save_dataframe_compressed(instance_df, instance_base, "expanded region instance metrics")
    instance_df.to_csv(
        cfg.EXPANDED_REGIONS_DIR / "expanded_region_instance_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_base = cfg.EXPANDED_REGIONS_DIR / "expanded_region_event_summary"
    cfg.save_dataframe_compressed(summary_df, summary_base, "expanded region event summary")
    summary_df.to_csv(
        cfg.EXPANDED_REGIONS_DIR / "expanded_region_event_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg.log("[Module 6] Summary:")
    cfg.log(f"    Expanded events summarized: {len(summary_df)}")
    cfg.log(f"    Instance-event metric rows: {len(instance_df)}")
    cfg.log(f"    Output: {cfg.EXPANDED_REGIONS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 6] Stopped gracefully: {exc}")
        cfg.log("[Module 6] Completed outputs are kept. Clear the stop request before resuming.")
