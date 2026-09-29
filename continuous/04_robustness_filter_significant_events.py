# -*- coding: utf-8 -*-
"""
Module 4: robustness diagnostics for initially significant bin x time events.

Inputs:
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    and per-group delta chunks listed in that manifest.
    OUTPUT_DIR/03_group_tests/significant_bin_timepoints_initial.pkl.gz

Outputs:
    OUTPUT_DIR/04_robust_events/significant_event_robustness_all.pkl.gz
    OUTPUT_DIR/04_robust_events/significant_event_robustness_all.csv
    OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.pkl.gz
    OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.csv
    OUTPUT_DIR/04_robust_events/robust_significant_time_maps.xlsx
    OUTPUT_DIR/04_robust_events/robust_significant_time_maps_manifest.csv

Depends on:
    02_compute_delta_values.py and 03_group_bin_response_tests.py outputs.
"""

import importlib.util
import math
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

try:
    from scipy.stats import t as t_dist
except Exception:
    t_dist = None

_ROBUST_DELTA_DF = None


def init_robust_worker(delta_df):
    global _ROBUST_DELTA_DF
    _ROBUST_DELTA_DF = delta_df


def top_contributions(delta_values):
    arr = np.asarray(delta_values, dtype=float)
    positives = np.sort(arr[arr > 0])[::-1]
    positive_sum = float(np.sum(positives))
    if positive_sum <= 0 or len(positives) == 0:
        return np.nan, np.nan, positive_sum
    top1 = float(positives[0] / positive_sum)
    top2 = float(np.sum(positives[:2]) / positive_sum)
    return top1, top2, positive_sum


def one_sample_greater_test(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    mean = float(np.mean(arr)) if n else np.nan
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    sem = float(sd / math.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan

    if n < cfg.MIN_N_FOR_GROUP_TEST:
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "t_stat": np.nan,
            "p_raw": np.nan,
            "zero_variance": False,
            "significant": False,
            "status": "n_below_min",
        }

    if not np.isfinite(sd) or sd == 0.0:
        if mean > 0:
            return {
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": 0.0,
                "t_stat": np.inf,
                "p_raw": 0.0,
                "zero_variance": True,
                "significant": True,
                "status": "significant",
            }
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "sem": 0.0,
            "t_stat": 0.0,
            "p_raw": 1.0,
            "zero_variance": True,
            "significant": False,
            "status": "mean_not_positive",
        }

    t_stat = mean / sem
    if t_dist is not None:
        p_raw = float(t_dist.sf(t_stat, df=n - 1))
    else:
        p_raw = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
    p_raw = min(max(p_raw, 0.0), 1.0)

    significant = bool(mean > 0 and p_raw < cfg.ALPHA)
    if significant:
        status = "significant"
    elif mean <= 0:
        status = "mean_not_positive"
    else:
        status = "p_not_below_alpha"

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "t_stat": float(t_stat),
        "p_raw": p_raw,
        "zero_variance": False,
        "significant": significant,
        "status": status,
    }


def top_removal_summary(instance_df: pd.DataFrame):
    """
    Sort instances by deltaF descending, then remove the largest remaining
    responder one at a time. After each removal, test whether the remaining
    instance-level deltaF values are still significantly greater than zero.
    """
    ordered = instance_df.loc[:, ["instance_uid", "deltaF"]].copy() if not instance_df.empty else pd.DataFrame()
    if not ordered.empty:
        ordered["deltaF"] = pd.to_numeric(ordered["deltaF"], errors="coerce")
        ordered = ordered.dropna(subset=["deltaF"]).sort_values(
            ["deltaF", "instance_uid"], ascending=[False, True]
        )

    n_total = int(len(ordered))
    ordered_instances = ";".join(ordered["instance_uid"].astype(str).tolist()) if n_total else ""
    default = {
        "top_removal_ordered_instances": ordered_instances,
        "top_removal_first_removed_instance": "",
        "top_removal_last_tested_n_removed": 0,
        "top_removal_last_tested_remaining_n": n_total,
        "top_removal_top1_remaining_n": np.nan,
        "top_removal_top1_mean_deltaF": np.nan,
        "top_removal_top1_t_stat": np.nan,
        "top_removal_top1_p_raw": np.nan,
        "top_removal_top1_significant": False,
        "top_removal_top1_status": "not_enough_instances_to_remove",
        "top_removal_first_failure_n_removed": np.nan,
        "top_removal_first_failure_reason": "not_enough_instances_to_remove",
        "top_removal_first_failure_removed_instances": "",
        "top_removal_first_failure_remaining_n": np.nan,
        "top_removal_first_failure_mean_deltaF": np.nan,
        "top_removal_first_failure_p_raw": np.nan,
        "top_removal_first_nonsignificant_n_removed": np.nan,
        "top_removal_first_nonsignificant_removed_instances": "",
        "top_removal_first_nonsignificant_remaining_n": np.nan,
        "top_removal_first_nonsignificant_mean_deltaF": np.nan,
        "top_removal_first_nonsignificant_p_raw": np.nan,
        "top_removal_all_tested_steps_significant": False,
    }
    if n_total <= 1:
        return default

    removed_instances = []
    first_failure = None
    first_nonsignificant = None
    last_step = None
    top1_step = None
    all_testable_steps_significant = True
    any_testable_step = False

    for n_removed in range(1, n_total):
        removed_instances.append(str(ordered.iloc[n_removed - 1]["instance_uid"]))
        remaining = ordered.iloc[n_removed:]["deltaF"].to_numpy(dtype=float)
        test = one_sample_greater_test(remaining)
        step = {
            "n_removed": int(n_removed),
            "removed_instances": ";".join(removed_instances),
            "remaining_n": int(test["n"]),
            "mean_deltaF": test["mean"],
            "t_stat": test["t_stat"],
            "p_raw": test["p_raw"],
            "significant": bool(test["significant"]),
            "status": str(test["status"]),
        }
        last_step = step
        if n_removed == 1:
            top1_step = step

        if test["n"] >= cfg.MIN_N_FOR_GROUP_TEST:
            any_testable_step = True
            if not test["significant"]:
                all_testable_steps_significant = False
                if first_nonsignificant is None:
                    first_nonsignificant = step
        else:
            if first_failure is None:
                first_failure = step
            break

        if not test["significant"]:
            if first_failure is None:
                first_failure = step
            break

    if first_failure is None:
        first_failure = last_step

    if top1_step is not None:
        default.update(
            {
                "top_removal_first_removed_instance": top1_step["removed_instances"],
                "top_removal_top1_remaining_n": top1_step["remaining_n"],
                "top_removal_top1_mean_deltaF": top1_step["mean_deltaF"],
                "top_removal_top1_t_stat": top1_step["t_stat"],
                "top_removal_top1_p_raw": top1_step["p_raw"],
                "top_removal_top1_significant": top1_step["significant"],
                "top_removal_top1_status": top1_step["status"],
            }
        )

    if last_step is not None:
        default.update(
            {
                "top_removal_last_tested_n_removed": last_step["n_removed"],
                "top_removal_last_tested_remaining_n": last_step["remaining_n"],
            }
        )

    if first_failure is not None:
        default.update(
            {
                "top_removal_first_failure_n_removed": first_failure["n_removed"],
                "top_removal_first_failure_reason": first_failure["status"],
                "top_removal_first_failure_removed_instances": first_failure["removed_instances"],
                "top_removal_first_failure_remaining_n": first_failure["remaining_n"],
                "top_removal_first_failure_mean_deltaF": first_failure["mean_deltaF"],
                "top_removal_first_failure_p_raw": first_failure["p_raw"],
            }
        )

    if first_nonsignificant is not None:
        default.update(
            {
                "top_removal_first_nonsignificant_n_removed": first_nonsignificant["n_removed"],
                "top_removal_first_nonsignificant_removed_instances": first_nonsignificant[
                    "removed_instances"
                ],
                "top_removal_first_nonsignificant_remaining_n": first_nonsignificant["remaining_n"],
                "top_removal_first_nonsignificant_mean_deltaF": first_nonsignificant["mean_deltaF"],
                "top_removal_first_nonsignificant_p_raw": first_nonsignificant["p_raw"],
            }
        )

    default["top_removal_all_tested_steps_significant"] = bool(
        any_testable_step and all_testable_steps_significant
    )
    return default


def event_instance_values(event: dict):
    df = _ROBUST_DELTA_DF
    mask = (
        (df["gene_id"] == event["gene_id"])
        & (df["condition"] == event["condition"])
        & (df["s_bin"].astype(int) == int(event["s_bin"]))
        & (df["r_bin"].astype(int) == int(event["r_bin"]))
        & (pd.to_numeric(df["time_min"], errors="coerce") == float(event["time_min"]))
    )
    subset = df.loc[mask, ["instance_uid", "deltaF", "dff0"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["instance_uid", "deltaF", "dff0"])
    subset["deltaF"] = pd.to_numeric(subset["deltaF"], errors="coerce")
    subset["dff0"] = pd.to_numeric(subset["dff0"], errors="coerce")
    inst = (
        subset.groupby("instance_uid", as_index=False)
        .agg(deltaF=("deltaF", "mean"), dff0=("dff0", "mean"))
        .dropna(subset=["deltaF"])
    )
    return inst


def evaluate_event(event: dict):
    inst = event_instance_values(event)
    delta = inst["deltaF"].to_numpy(dtype=float) if not inst.empty else np.asarray([], dtype=float)
    dff0 = pd.to_numeric(inst["dff0"], errors="coerce").dropna().to_numpy(dtype=float) if not inst.empty else np.asarray([], dtype=float)
    n = int(len(delta))
    mean_delta = float(np.mean(delta)) if n else np.nan
    median_delta = float(np.median(delta)) if n else np.nan
    mean_dff0 = float(np.mean(dff0)) if len(dff0) else np.nan
    median_dff0 = float(np.median(dff0)) if len(dff0) else np.nan
    n_positive = int(np.sum(delta > 0)) if n else 0
    positive_ratio = n_positive / n if n else np.nan

    top1, top2, positive_sum = top_contributions(delta)
    top_removal = top_removal_summary(inst)

    passes_positive_ratio = bool(np.isfinite(positive_ratio) and positive_ratio >= cfg.POSITIVE_RATIO_THRESHOLD)
    passes_top2 = bool(np.isfinite(top2) and top2 <= cfg.MAX_TOP2_CONTRIBUTION)
    passes_loo = bool(top_removal["top_removal_top1_significant"]) if cfg.REQUIRE_LEAVE_ONE_OUT_POSITIVE else True
    passes_robustness = passes_positive_ratio and passes_top2 and passes_loo

    out = dict(event)
    out.update(
        {
            "robust_n_instances": n,
            "robust_mean_deltaF": mean_delta,
            "robust_median_deltaF": median_delta,
            "robust_mean_dff0": mean_dff0,
            "robust_median_dff0": median_dff0,
            "robust_n_positive_deltaF": n_positive,
            "robust_positive_ratio": positive_ratio,
            "mean_deltaF": mean_delta,
            "median_deltaF": median_delta,
            "mean_dff0": mean_dff0,
            "median_dff0": median_dff0,
            "positive_ratio": positive_ratio,
            "positive_delta_sum": positive_sum,
            "top1_contribution": top1,
            "top2_contribution": top2,
            **top_removal,
            "leave_one_out_min_mean_deltaF": top_removal["top_removal_top1_mean_deltaF"],
            "leave_one_out_max_mean_deltaF": top_removal["top_removal_top1_mean_deltaF"],
            "leave_one_out_all_positive": top_removal["top_removal_top1_significant"],
            "leave_one_out_flipped_by_instance": top_removal["top_removal_first_failure_removed_instances"],
            "passes_positive_ratio": passes_positive_ratio,
            "passes_top2_contribution": passes_top2,
            "passes_leave_one_out": passes_loo,
            "passes_robustness": passes_robustness,
            "robustness_passed": passes_robustness,
            "positive_ratio_threshold": cfg.POSITIVE_RATIO_THRESHOLD,
            "max_top2_contribution": cfg.MAX_TOP2_CONTRIBUTION,
            "require_leave_one_out_positive": cfg.REQUIRE_LEAVE_ONE_OUT_POSITIVE,
            "leave_one_out_definition": "top_deltaF_removal_one_sample_greater",
        }
    )
    return out



def evaluate_event_from_instances(event: dict, inst: pd.DataFrame):
    delta = inst["deltaF"].to_numpy(dtype=float) if not inst.empty else np.asarray([], dtype=float)
    dff0 = pd.to_numeric(inst["dff0"], errors="coerce").dropna().to_numpy(dtype=float) if not inst.empty else np.asarray([], dtype=float)
    n = int(len(delta))
    mean_delta = float(np.mean(delta)) if n else np.nan
    median_delta = float(np.median(delta)) if n else np.nan
    mean_dff0 = float(np.mean(dff0)) if len(dff0) else np.nan
    median_dff0 = float(np.median(dff0)) if len(dff0) else np.nan
    n_positive = int(np.sum(delta > 0)) if n else 0
    positive_ratio = n_positive / n if n else np.nan

    top1, top2, positive_sum = top_contributions(delta)
    top_removal = top_removal_summary(inst)

    passes_positive_ratio = bool(np.isfinite(positive_ratio) and positive_ratio >= cfg.POSITIVE_RATIO_THRESHOLD)
    passes_top2 = bool(np.isfinite(top2) and top2 <= cfg.MAX_TOP2_CONTRIBUTION)
    passes_loo = bool(top_removal["top_removal_top1_significant"]) if cfg.REQUIRE_LEAVE_ONE_OUT_POSITIVE else True
    passes_robustness = passes_positive_ratio and passes_top2 and passes_loo

    out = dict(event)
    out.update(
        {
            "robust_n_instances": n,
            "robust_mean_deltaF": mean_delta,
            "robust_median_deltaF": median_delta,
            "robust_mean_dff0": mean_dff0,
            "robust_median_dff0": median_dff0,
            "robust_n_positive_deltaF": n_positive,
            "robust_positive_ratio": positive_ratio,
            "mean_deltaF": mean_delta,
            "median_deltaF": median_delta,
            "mean_dff0": mean_dff0,
            "median_dff0": median_dff0,
            "positive_ratio": positive_ratio,
            "positive_delta_sum": positive_sum,
            "top1_contribution": top1,
            "top2_contribution": top2,
            **top_removal,
            "leave_one_out_min_mean_deltaF": top_removal["top_removal_top1_mean_deltaF"],
            "leave_one_out_max_mean_deltaF": top_removal["top_removal_top1_mean_deltaF"],
            "leave_one_out_all_positive": top_removal["top_removal_top1_significant"],
            "leave_one_out_flipped_by_instance": top_removal["top_removal_first_failure_removed_instances"],
            "passes_positive_ratio": passes_positive_ratio,
            "passes_top2_contribution": passes_top2,
            "passes_leave_one_out": passes_loo,
            "passes_robustness": passes_robustness,
            "robustness_passed": passes_robustness,
            "positive_ratio_threshold": cfg.POSITIVE_RATIO_THRESHOLD,
            "max_top2_contribution": cfg.MAX_TOP2_CONTRIBUTION,
            "require_leave_one_out_positive": cfg.REQUIRE_LEAVE_ONE_OUT_POSITIVE,
            "leave_one_out_definition": "top_deltaF_removal_one_sample_greater",
        }
    )
    return out


ROBUST_OUTPUT_REQUIRED_COLUMNS = [
    "gene_id",
    "gene_label",
    "condition",
    "condition_label",
    "s_bin",
    "r_bin",
    "s_start",
    "s_end",
    "r_start",
    "r_end",
    "s_center",
    "r_center",
    "time_min",
    "mean_deltaF",
    "median_deltaF",
    "mean_dff0",
    "median_dff0",
    "p_raw",
    "p_adj",
    "significant",
    "positive_ratio",
    "top1_contribution",
    "top2_contribution",
    "leave_one_out_min_mean_deltaF",
    "leave_one_out_all_positive",
    "robustness_passed",
]


def ensure_robust_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ROBUST_OUTPUT_REQUIRED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out


def run_group_events(group_delta: pd.DataFrame, group_events: pd.DataFrame):
    """
    Robustness diagnostics for one gene x condition.

    This intentionally avoids ProcessPoolExecutor on Windows. Passing a whole
    per-group delta DataFrame into worker initializers requires pickling tens of
    millions of rows and can fail with truncated pickle/Invalid argument errors.
    Instead, perform one vectorized merge to extract only the significant
    event rows, then summarize the much smaller event x instance table.
    """
    if group_events.empty:
        return []

    events = group_events.copy().reset_index(drop=True)
    if "event_id" not in events.columns:
        events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=int))

    event_keys = events[["event_id", "s_bin", "r_bin", "time_min"]].copy()
    event_keys["s_bin"] = pd.to_numeric(event_keys["s_bin"], errors="coerce").astype("Int64")
    event_keys["r_bin"] = pd.to_numeric(event_keys["r_bin"], errors="coerce").astype("Int64")
    event_keys["time_min"] = pd.to_numeric(event_keys["time_min"], errors="coerce")
    event_keys = event_keys.dropna(subset=["s_bin", "r_bin", "time_min"]).copy()
    event_keys["s_bin"] = event_keys["s_bin"].astype(np.int32)
    event_keys["r_bin"] = event_keys["r_bin"].astype(np.int32)

    cfg.log(f"    Vectorized robustness checks for {len(events)} events")
    cfg.log("        Extracting event x instance delta values with one merge...")

    needed_cols = ["instance_uid", "s_bin", "r_bin", "time_min", "deltaF", "dff0", "baseline_time_min"]
    work = group_delta.loc[:, [c for c in needed_cols if c in group_delta.columns]].copy()
    work["s_bin"] = pd.to_numeric(work["s_bin"], errors="coerce").astype("Int64")
    work["r_bin"] = pd.to_numeric(work["r_bin"], errors="coerce").astype("Int64")
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")
    work["deltaF"] = pd.to_numeric(work["deltaF"], errors="coerce")
    work["dff0"] = pd.to_numeric(work["dff0"], errors="coerce")
    if "baseline_time_min" in work.columns:
        work["baseline_time_min"] = pd.to_numeric(work["baseline_time_min"], errors="coerce")
    work, _ = cfg.filter_late_baseline_bins(work)
    work = work.dropna(subset=["s_bin", "r_bin", "time_min"]).copy()
    work["s_bin"] = work["s_bin"].astype(np.int32)
    work["r_bin"] = work["r_bin"].astype(np.int32)

    matched = work.merge(event_keys, on=["s_bin", "r_bin", "time_min"], how="inner")
    if matched.empty:
        inst_by_event = pd.DataFrame(columns=["event_id", "instance_uid", "deltaF", "dff0"])
    else:
        inst_by_event = (
            matched.groupby(["event_id", "instance_uid"], as_index=False)
            .agg(deltaF=("deltaF", "mean"), dff0=("dff0", "mean"))
            .dropna(subset=["deltaF"])
        )

    instance_groups = {
        int(event_id): g.drop(columns=["event_id"]).reset_index(drop=True)
        for event_id, g in inst_by_event.groupby("event_id", sort=False)
    }

    empty_inst = pd.DataFrame(columns=["instance_uid", "deltaF", "dff0"])
    events_by_id = {int(row["event_id"]): row for row in events.to_dict(orient="records")}
    results = []
    for i, event_id in enumerate(events_by_id.keys(), start=1):
        cfg.raise_if_stop_requested(f"module 4 before robustness event {i}/{len(events_by_id)}")
        inst = instance_groups.get(int(event_id), empty_inst)
        results.append(evaluate_event_from_instances(events_by_id[int(event_id)], inst))
        if i == 1 or i == len(events_by_id) or (
            cfg.PROGRESS_EVERY_N_EVENTS and i % cfg.PROGRESS_EVERY_N_EVENTS == 0
        ):
            cfg.log(f"        Robustness events finished: {i}/{len(events_by_id)}")

    return results


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


def run_robustness(delta_manifest: pd.DataFrame, significant_df: pd.DataFrame):
    if significant_df.empty:
        return pd.DataFrame()

    significant_df = significant_df.copy().reset_index(drop=True)
    if "event_id" not in significant_df.columns:
        significant_df.insert(0, "event_id", np.arange(1, len(significant_df) + 1, dtype=int))

    all_results = []
    for (gene_id, condition), events_g in significant_df.groupby(["gene_id", "condition"], sort=False):
        cfg.raise_if_stop_requested(f"module 4 before group {gene_id} | {condition}")
        cfg.log(f"    Robustness group: {gene_id} | {condition} ({len(events_g)} events)")
        if "ALL" in set(delta_manifest.get("gene_id", [])):
            delta_path = Path(delta_manifest.iloc[0]["delta_values_path"])
        else:
            match = delta_manifest.loc[
                (delta_manifest["gene_id"] == gene_id) & (delta_manifest["condition"] == condition)
            ]
            if match.empty:
                raise FileNotFoundError(f"No delta chunk found for {gene_id} | {condition}")
            delta_path = Path(match.iloc[0]["delta_values_path"])
        cfg.log(f"        Reading delta chunk: {delta_path}")
        delta_g = cfg.read_dataframe_compressed(delta_path)
        all_results.extend(run_group_events(delta_g, events_g))

    return pd.DataFrame(all_results)


def _excel_sheet_name(name: str) -> str:
    invalid = set("[]:*?/\\")
    clean = "".join("_" if ch in invalid else ch for ch in str(name))
    clean = clean.strip() or "sheet"
    return clean[:31]


def _robust_bin_time_table_for_group(robust_group: pd.DataFrame) -> pd.DataFrame:
    records = []
    if robust_group.empty:
        return pd.DataFrame(columns=["r_bin", "s_bin", "significant_times", "n_significant_timepoints"])

    work = robust_group.copy()
    work["s_bin"] = pd.to_numeric(work["s_bin"], errors="coerce")
    work["r_bin"] = pd.to_numeric(work["r_bin"], errors="coerce")
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")
    work = work.dropna(subset=["s_bin", "r_bin", "time_min"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["r_bin", "s_bin", "significant_times", "n_significant_timepoints"])

    work["s_bin"] = work["s_bin"].astype(int)
    work["r_bin"] = work["r_bin"].astype(int)
    for (r_bin, s_bin), g in work.groupby(["r_bin", "s_bin"], sort=True):
        times = sorted(set(pd.to_numeric(g["time_min"], errors="coerce").dropna().to_numpy(dtype=float)))
        records.append(
            {
                "r_bin": int(r_bin),
                "s_bin": int(s_bin),
                "significant_times": cfg.INTERVAL_SEPARATOR.join(cfg.format_time_value(t) for t in times),
                "n_significant_timepoints": int(len(times)),
            }
        )
    return pd.DataFrame(records)


def save_robust_bin_time_maps_excel(robust_df: pd.DataFrame):
    """Write one Excel workbook with s-bin x r-bin maps of robust significant times."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    output_path = cfg.ROBUST_EVENTS_DIR / "robust_significant_time_maps.xlsx"
    cfg.log(f"    Writing robust significant time maps: {output_path}")

    wb = Workbook()
    wb.remove(wb.active)

    red_fill = PatternFill(fill_type="solid", fgColor="FFFF6666")
    header_fill = PatternFill(fill_type="solid", fgColor="FFD9EAF7")
    header_font = Font(bold=True)
    cell_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    summary_rows = []
    used_sheet_names = set()

    for group in cfg.INPUT_GROUPS:
        gene_id = group["gene_id"]
        condition = group["condition"]
        gene_label = group.get("gene_label", gene_id)
        condition_label = group.get("condition_label", condition)
        sheet_base = _excel_sheet_name(f"{gene_id}_{condition}")
        sheet_name = sheet_base
        suffix = 2
        while sheet_name in used_sheet_names:
            suffix_text = f"_{suffix}"
            sheet_name = f"{sheet_base[:31 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        used_sheet_names.add(sheet_name)

        if robust_df.empty:
            robust_group = pd.DataFrame()
        else:
            robust_group = robust_df.loc[
                (robust_df["gene_id"] == gene_id) & (robust_df["condition"] == condition)
            ].copy()

        bin_time_table = _robust_bin_time_table_for_group(robust_group)
        time_lookup = {
            (int(row.r_bin), int(row.s_bin)): str(row.significant_times)
            for row in bin_time_table.itertuples(index=False)
        }

        ws = wb.create_sheet(title=sheet_name)
        ws.cell(row=1, column=1, value=f"{gene_label} | {condition_label}")
        ws.cell(row=1, column=1).font = Font(bold=True)
        ws.cell(
            row=2,
            column=1,
            value="Rows: r_bin; columns: s_bin; red cells passed Module 4 robustness filters.",
        )
        ws.cell(row=3, column=1, value="r_bin / s_bin")
        ws.cell(row=3, column=1).fill = header_fill
        ws.cell(row=3, column=1).font = header_font
        ws.cell(row=3, column=1).alignment = cell_alignment

        for s_bin in range(cfg.N_S_BINS):
            cell = ws.cell(row=3, column=s_bin + 2, value=s_bin)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = cell_alignment

        for r_bin in range(cfg.N_R_BINS):
            row_idx = r_bin + 4
            row_header = ws.cell(row=row_idx, column=1, value=r_bin)
            row_header.fill = header_fill
            row_header.font = header_font
            row_header.alignment = cell_alignment

        for (r_bin, s_bin), value in time_lookup.items():
            if 0 <= s_bin < cfg.N_S_BINS and 0 <= r_bin < cfg.N_R_BINS and value:
                cell = ws.cell(row=r_bin + 4, column=s_bin + 2, value=value)
                cell.fill = red_fill
                cell.alignment = cell_alignment

        ws.freeze_panes = "B4"
        ws.column_dimensions["A"].width = 12
        for col_idx in range(2, cfg.N_S_BINS + 2):
            ws.column_dimensions[get_column_letter(col_idx)].width = 8
        for row_idx in range(4, cfg.N_R_BINS + 4):
            ws.row_dimensions[row_idx].height = 24

        summary_rows.append(
            {
                "gene_id": gene_id,
                "condition": condition,
                "sheet_name": sheet_name,
                "n_robust_events": int(len(robust_group)),
                "n_robust_bins": int(len(bin_time_table)),
            }
        )

    summary_ws = wb.create_sheet(title="summary", index=0)
    summary_headers = ["gene_id", "condition", "sheet_name", "n_robust_events", "n_robust_bins"]
    for col_idx, header in enumerate(summary_headers, start=1):
        cell = summary_ws.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
    for row_idx, row in enumerate(summary_rows, start=2):
        for col_idx, header in enumerate(summary_headers, start=1):
            summary_ws.cell(row=row_idx, column=col_idx, value=row[header])
    summary_ws.freeze_panes = "A2"
    for col_idx in range(1, len(summary_headers) + 1):
        summary_ws.column_dimensions[get_column_letter(col_idx)].width = 20

    wb.save(output_path)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        cfg.ROBUST_EVENTS_DIR / "robust_significant_time_maps_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return output_path, summary_df


def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="04_robustness_filter_significant_events")

    robust_all_base = cfg.ROBUST_EVENTS_DIR / "significant_event_robustness_all"
    robust_path = cfg.compressed_path(robust_all_base)
    robust_base = cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust"
    robust_events_path = cfg.compressed_path(robust_base)
    excel_map_path = cfg.ROBUST_EVENTS_DIR / "robust_significant_time_maps.xlsx"
    if robust_path.exists() and robust_events_path.exists() and not cfg.OVERWRITE_EXISTING:
        existing_robust_df = cfg.read_dataframe_compressed(robust_events_path)
        missing_columns = [column for column in ROBUST_OUTPUT_REQUIRED_COLUMNS if column not in existing_robust_df.columns]
        if missing_columns:
            cfg.log("Module 4 output exists but is missing current required fields; recomputing Module 4.")
            cfg.log(f"    Missing fields: {', '.join(missing_columns)}")
        elif excel_map_path.exists():
            cfg.log(f"Module 4 output already exists; skipping: {robust_path}")
            return
        else:
            cfg.log("Module 4 robust outputs already exist; generating missing Excel time map only.")
            map_path, map_manifest = save_robust_bin_time_maps_excel(existing_robust_df)
            cfg.log("[Module 4] Summary:")
            cfg.log(f"    Robust events loaded: {len(existing_robust_df)}")
            cfg.log(f"    Robust bins in Excel maps: {int(map_manifest['n_robust_bins'].sum()) if not map_manifest.empty else 0}")
            cfg.log(f"    Excel map: {map_path}")
            cfg.log(f"    Output: {cfg.ROBUST_EVENTS_DIR}")
            return

    sig_path = cfg.compressed_path(cfg.GROUP_TESTS_DIR / "significant_bin_timepoints_initial")

    cfg.log("[Module 4] Reading Module 2 delta manifest.")
    delta_manifest = load_delta_manifest()
    cfg.log(f"[Module 4] Reading initial significant events: {sig_path}")
    significant_df = cfg.read_dataframe_compressed(sig_path)

    cfg.log("[Module 4] Running robustness diagnostics.")
    robustness_df = ensure_robust_output_columns(run_robustness(delta_manifest, significant_df))
    robust_df = (
        robustness_df.loc[robustness_df["passes_robustness"].astype(bool)].copy()
        if not robustness_df.empty
        else ensure_robust_output_columns(pd.DataFrame())
    )

    cfg.save_dataframe_compressed(robustness_df, robust_all_base, "all significant-event robustness diagnostics")
    robustness_df.to_csv(
        cfg.ROBUST_EVENTS_DIR / "significant_event_robustness_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg.save_dataframe_compressed(robust_df, robust_base, "robust significant bin timepoints")
    robust_df.to_csv(
        cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust.csv",
        index=False,
        encoding="utf-8-sig",
    )
    map_path, map_manifest = save_robust_bin_time_maps_excel(robust_df)

    cfg.log("[Module 4] Summary:")
    cfg.log(f"    Initial significant events checked: {len(robustness_df)}")
    cfg.log(f"    Robust events retained: {len(robust_df)}")
    cfg.log(f"    Robust bins in Excel maps: {int(map_manifest['n_robust_bins'].sum()) if not map_manifest.empty else 0}")
    cfg.log(f"    Excel map: {map_path}")
    cfg.log(f"    Output: {cfg.ROBUST_EVENTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 4] Stopped gracefully: {exc}")
        cfg.log("[Module 4] Completed outputs are kept. Clear the stop request before resuming.")
