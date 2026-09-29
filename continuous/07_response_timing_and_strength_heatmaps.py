# -*- coding: utf-8 -*-
"""
Module 7: composite response figures and supporting tables.

Inputs:
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    Per-group Module 2 delta chunks listed in that manifest.
    OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.pkl.gz
    Fallback: OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.csv

Outputs:
    OUTPUT_DIR/07_response_heatmaps/full_worm_instance_time_total_deltaF.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/full_worm_time_summary.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/full_worm_time_vs_first_time_tests.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/axial_instance_s_bin_cumulative_deltaF.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/axial_s_bin_summary.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/<gene>__<condition>/*heatmap*.pkl.gz / csv
    OUTPUT_DIR/07_response_heatmaps/response_timing_composite.png / pdf / svg
    OUTPUT_DIR/07_response_heatmaps/response_strength_composite.png / pdf / svg
    OUTPUT_DIR/07_response_heatmaps/response_strength_composite_dff0.png / pdf / svg
    OUTPUT_DIR/07_response_heatmaps/all_response_heatmap_manifest.pkl.gz / csv

Depends on:
    Module 2 for Panels A-D and Module 4 robust events for the heat maps.

Figure layout:
    Main timing figure: 4 rows x 2 columns, Col 0 = A-D line plots, Col 1 = E-H
    onset heat maps. Standalone strength figures: 4 rows x 1 column with their own
    A-D lettering and the same panel width as the main onset column. Every
    heat-map column shares one horizontal colour bar drawn in a band above the
    grid, so no per-panel colour bar is drawn. Pixels added by the display-only
    visual dilation are faded to RESPONSE_HEATMAP_VISUAL_DILATION_ALPHA, while
    measured bins stay opaque.

Statistical rule:
    Panels A-B use instance-level whole-worm total deltaF. Each time point is
    paired by instance_uid against the same condition's first available time,
    followed by multiplicity correction within gene x condition. The heat-map
    panels perform no new statistical test and only summarize Module 4 robust
    events.
"""

import gc
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.ioff()


def _load_config():
    module_path = Path(__file__).resolve().with_name("00_config_utils.py")
    spec = importlib.util.spec_from_file_location("pipeline_config_utils", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()
plt.rcParams["font.family"] = cfg.FONT_FAMILY
ROBUST_EVENTS_PATH = cfg.compressed_path(cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust")
ROBUST_EVENTS_CSV_PATH = cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust.csv"
DELTA_MANIFEST_PATH = cfg.compressed_path(cfg.DELTA_VALUES_DIR / "delta_values_manifest")
DELTA_MANIFEST_CSV_PATH = cfg.DELTA_VALUES_DIR / "delta_values_manifest.csv"
REQUIRED_ROBUST_COLUMNS = [
    "gene_id", "gene_label", "condition", "condition_label", "s_bin", "r_bin",
    "s_start", "s_end", "r_start", "r_end", "s_center", "r_center", "time_min",
    "mean_deltaF", "median_deltaF", "mean_dff0", "median_dff0", "p_raw", "p_adj",
    "significant", "positive_ratio", "top1_contribution", "top2_contribution",
    "leave_one_out_min_mean_deltaF", "leave_one_out_all_positive", "robustness_passed",
]
DELTA_AGGREGATION_COLUMNS = ["instance_uid", "time_min", "s_bin", "deltaF"]
GROUP_COLUMNS = ["gene_id", "gene_label", "condition", "condition_label"]


def _bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def read_robust_events():
    if ROBUST_EVENTS_PATH.exists():
        cfg.log(f"[Module 7] Reading Module 4 robust events: {ROBUST_EVENTS_PATH}")
        robust_df = cfg.read_dataframe_compressed(ROBUST_EVENTS_PATH)
    elif ROBUST_EVENTS_CSV_PATH.exists():
        cfg.log(f"[Module 7] Reading Module 4 CSV fallback: {ROBUST_EVENTS_CSV_PATH}")
        robust_df = pd.read_csv(ROBUST_EVENTS_CSV_PATH)
    else:
        raise FileNotFoundError("Missing Module 4 robust-event output. Run Module 4 first.")
    missing = [c for c in REQUIRED_ROBUST_COLUMNS if c not in robust_df.columns]
    if missing:
        raise ValueError("Module 4 robust output is missing fields: " + ", ".join(missing))
    for c in ["s_bin", "r_bin", "s_start", "s_end", "r_start", "r_end", "s_center", "r_center", "time_min", "mean_deltaF"]:
        robust_df[c] = pd.to_numeric(robust_df[c], errors="coerce")
    robust_df = robust_df.loc[_bool_series(robust_df["robustness_passed"])].copy()
    robust_df = robust_df.dropna(subset=["gene_id", "condition", "s_bin", "r_bin", "time_min", "mean_deltaF"])
    robust_df["s_bin"] = robust_df["s_bin"].astype(int)
    robust_df["r_bin"] = robust_df["r_bin"].astype(int)
    in_bounds = robust_df["s_bin"].between(0, cfg.N_S_BINS - 1) & robust_df["r_bin"].between(0, cfg.N_R_BINS - 1)
    return robust_df.loc[in_bounds].copy()


def read_delta_manifest():
    if DELTA_MANIFEST_PATH.exists():
        cfg.log(f"[Module 7] Reading Module 2 delta manifest: {DELTA_MANIFEST_PATH}")
        manifest = cfg.read_dataframe_compressed(DELTA_MANIFEST_PATH)
    elif DELTA_MANIFEST_CSV_PATH.exists():
        manifest = pd.read_csv(DELTA_MANIFEST_CSV_PATH)
    else:
        raise FileNotFoundError("Missing Module 2 delta manifest. Run Module 2 first.")
    missing = [c for c in ["gene_id", "condition", "delta_values_path"] if c not in manifest.columns]
    if missing:
        raise ValueError("Module 2 manifest is missing fields: " + ", ".join(missing))
    return manifest


def configured_groups(robust_df, delta_manifest):
    groups, seen = [], set()
    for group in cfg.INPUT_GROUPS:
        rec = {"gene_id": group["gene_id"], "gene_label": group.get("gene_label", group["gene_id"]), "condition": group["condition"], "condition_label": group.get("condition_label", group["condition"])}
        key = (rec["gene_id"], rec["condition"])
        if key not in seen:
            groups.append(rec); seen.add(key)
    for source in [delta_manifest, robust_df]:
        if source is None or source.empty:
            continue
        for row in source[["gene_id", "gene_label", "condition", "condition_label"]].drop_duplicates().itertuples(index=False):
            key = (row.gene_id, row.condition)
            if key not in seen:
                groups.append({"gene_id": row.gene_id, "gene_label": row.gene_label, "condition": row.condition, "condition_label": row.condition_label})
                seen.add(key)
    return groups


def save_table(df, base, label, index=False, index_label=None):
    cfg.save_dataframe_compressed(df, base, label)
    df.to_csv(base.with_suffix(".csv"), index=index, index_label=index_label, encoding="utf-8-sig")


def aggregate_module2(delta_manifest):
    full_parts, axial_parts = [], []
    for index, row in enumerate(delta_manifest.itertuples(index=False), start=1):
        cfg.raise_if_stop_requested(f"module 7 before Module 2 chunk {index}/{len(delta_manifest)}")
        delta_path = Path(row.delta_values_path)
        cfg.log(f"[Module 7 delta chunk {index}/{len(delta_manifest)}] Reading: {delta_path}")
        delta_df = cfg.read_dataframe_compressed(delta_path)
        delta_df, _ = cfg.filter_late_baseline_bins(delta_df)
        missing = [c for c in DELTA_AGGREGATION_COLUMNS if c not in delta_df.columns]
        if missing:
            raise ValueError(f"Delta chunk {delta_path} is missing fields: {', '.join(missing)}")
        work = delta_df.loc[:, DELTA_AGGREGATION_COLUMNS].copy()
        del delta_df
        for c in ["time_min", "s_bin", "deltaF"]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.dropna(subset=["instance_uid", "time_min", "s_bin", "deltaF"])
        work["s_bin"] = work["s_bin"].astype(np.int32)
        full = work.groupby(["instance_uid", "time_min"], as_index=False, sort=False).agg(total_deltaF=("deltaF", "sum"), n_s_r_bins_summed=("deltaF", "size"))
        axial = work.groupby(["instance_uid", "s_bin"], as_index=False, sort=False).agg(cumulative_deltaF=("deltaF", "sum"), n_bin_time_rows_summed=("deltaF", "size"))
        metadata = {column: getattr(row, column) for column in GROUP_COLUMNS}
        for column in reversed(GROUP_COLUMNS):
            full.insert(0, column, metadata[column])
            axial.insert(0, column, metadata[column])
        full_parts.append(full); axial_parts.append(axial)
        cfg.log(f"    Aggregated rows: full-worm={len(full)}, axial={len(axial)}")
        del work, full, axial
        gc.collect()
    return (pd.concat(full_parts, ignore_index=True) if full_parts else pd.DataFrame(), pd.concat(axial_parts, ignore_index=True) if axial_parts else pd.DataFrame())


def summarize_values(df, group_cols, value_col, prefix):
    records = []
    for key, group in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ci = cfg.compute_mean_ci(group[value_col])
        record = dict(zip(group_cols, key))
        record.update({f"{prefix}_n_instances": ci["n"], f"{prefix}_mean": ci["mean"], f"{prefix}_sd": ci["sd"], f"{prefix}_sem": ci["sem"], f"{prefix}_ci_low": ci["ci_low"], f"{prefix}_ci_high": ci["ci_high"]})
        records.append(record)
    return pd.DataFrame(records)


def paired_two_sided_test(differences):
    arr = pd.to_numeric(pd.Series(differences), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr)); mean_diff = float(np.mean(arr)) if n else np.nan
    sd_diff = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    sem_diff = float(sd_diff / math.sqrt(n)) if n > 1 and np.isfinite(sd_diff) else np.nan
    if n < cfg.MIN_N_FOR_GROUP_TEST:
        return n, mean_diff, sd_diff, sem_diff, np.nan, np.nan, False, "n_below_min"
    if not np.isfinite(sd_diff) or sd_diff == 0.0:
        if mean_diff == 0.0:
            return n, mean_diff, sd_diff, 0.0, 0.0, 1.0, True, "zero_difference"
        return n, mean_diff, sd_diff, 0.0, np.inf if mean_diff > 0 else -np.inf, 0.0, True, "constant_nonzero_difference"
    t_stat = mean_diff / sem_diff
    try:
        from scipy.stats import t as t_dist
        p_raw = float(2.0 * t_dist.sf(abs(t_stat), df=n - 1))
    except Exception:
        p_raw = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return n, mean_diff, sd_diff, sem_diff, float(t_stat), min(max(p_raw, 0.0), 1.0), False, "tested"


def test_total_deltaF_vs_zero(full_df):
    """Two-sided one-sample t-test: total_deltaF vs 0 per (gene, condition, time).

    Holm-corrected within each gene-condition group across time points.
    """
    records = []
    for key, group in full_df.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        group_records = []
        for time_min in sorted(pd.to_numeric(group["time_min"], errors="coerce").dropna().unique()):
            values = group.loc[group["time_min"] == time_min, "total_deltaF"]
            arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
            n = int(len(arr))
            mean_val = float(np.mean(arr)) if n else np.nan
            sd_val = float(np.std(arr, ddof=1)) if n > 1 else np.nan
            sem_val = sd_val / math.sqrt(n) if n > 1 and np.isfinite(sd_val) else np.nan
            if n < cfg.MIN_N_FOR_GROUP_TEST:
                t_stat, p_raw, status = np.nan, np.nan, "n_below_min"
            elif not np.isfinite(sd_val) or sd_val == 0.0:
                if abs(mean_val) < 1e-12:
                    t_stat, p_raw, status = 0.0, 1.0, "zero_variance"
                else:
                    t_stat, p_raw, status = (np.inf if mean_val > 0 else -np.inf), 0.0, "zero_variance"
            else:
                try:
                    from scipy.stats import t as t_dist
                    res = stats.ttest_1samp(arr, 0)
                except ImportError:
                    res = stats.ttest_1samp(arr, 0)
                t_stat = float(res.statistic)
                p_raw = float(res.pvalue)
                status = "tested"
            record = dict(zip(GROUP_COLUMNS, key))
            record.update({
                "time_min": float(time_min),
                "n_instances": n,
                "mean_total_deltaF": mean_val,
                "sd_total_deltaF": sd_val,
                "sem_total_deltaF": sem_val,
                "t_stat": t_stat,
                "p_raw": p_raw,
                "p_adj": np.nan,
                "significant_vs_zero": False,
                "test_status": status,
            })
            group_records.append(record)
        tests = pd.DataFrame(group_records)
        if not tests.empty:
            tests["p_adj"] = cfg.adjust_p_values(tests["p_raw"].to_numpy(dtype=float), method=cfg.RESPONSE_TIME_TEST_P_ADJUST_METHOD)
            tests["significant_vs_zero"] = tests["p_adj"] < cfg.ALPHA
        records.extend(tests.to_dict(orient="records"))
    return pd.DataFrame(records)



def two_sample_t_test(values_a, values_b):
    """Welch two-sample, two-sided t test on instance-level values."""
    a = pd.to_numeric(pd.Series(values_a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(values_b), errors="coerce").dropna().to_numpy(dtype=float)
    n_a, n_b = int(len(a)), int(len(b))
    mean_a = float(np.mean(a)) if n_a else np.nan
    mean_b = float(np.mean(b)) if n_b else np.nan
    sd_a = float(np.std(a, ddof=1)) if n_a > 1 else np.nan
    sd_b = float(np.std(b, ddof=1)) if n_b > 1 else np.nan
    if n_a < cfg.MIN_N_FOR_GROUP_TEST or n_b < cfg.MIN_N_FOR_GROUP_TEST:
        return n_a, n_b, mean_a, mean_b, sd_a, sd_b, np.nan, np.nan, np.nan, False, "n_below_min"
    var_a = 0.0 if not np.isfinite(sd_a) else sd_a ** 2
    var_b = 0.0 if not np.isfinite(sd_b) else sd_b ** 2
    se2 = var_a / n_a + var_b / n_b
    mean_diff = mean_b - mean_a
    if se2 <= 0 or not np.isfinite(se2):
        if abs(mean_diff) < 1e-12:
            return n_a, n_b, mean_a, mean_b, sd_a, sd_b, 0.0, 1.0, np.nan, True, "zero_variance_equal_means"
        return n_a, n_b, mean_a, mean_b, sd_a, sd_b, np.inf if mean_diff > 0 else -np.inf, 0.0, np.nan, True, "zero_variance_unequal_means"
    se = math.sqrt(se2)
    t_stat = mean_diff / se
    denom = 0.0
    if n_a > 1 and var_a > 0:
        denom += (var_a / n_a) ** 2 / (n_a - 1)
    if n_b > 1 and var_b > 0:
        denom += (var_b / n_b) ** 2 / (n_b - 1)
    df = (se2 ** 2 / denom) if denom > 0 else np.nan
    try:
        from scipy.stats import t as t_dist
        p_raw = float(2.0 * t_dist.sf(abs(t_stat), df=df)) if np.isfinite(df) else float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    except Exception:
        p_raw = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return n_a, n_b, mean_a, mean_b, sd_a, sd_b, float(t_stat), min(max(p_raw, 0.0), 1.0), float(df) if np.isfinite(df) else np.nan, False, "welch_t_test"


def ordered_conditions_for_gene(gene_id):
    ordered = []
    for group in cfg.INPUT_GROUPS:
        if group["gene_id"] == gene_id and group["condition"] not in [item[0] for item in ordered]:
            ordered.append((group["condition"], group.get("condition_label", group["condition"])))
    return ordered


def test_axial_condition_differences(axial_df, genes):
    records = []
    for gene in genes[:2]:
        gene_id = gene["gene_id"]
        conditions = ordered_conditions_for_gene(gene_id)
        if len(conditions) < 2:
            continue
        condition_a, label_a = conditions[0]
        condition_b, label_b = conditions[1]
        gene_df = axial_df.loc[axial_df["gene_id"] == gene_id].copy()
        gene_records = []
        for s_bin in range(cfg.N_S_BINS):
            a_values = gene_df.loc[(gene_df["condition"] == condition_a) & (gene_df["s_bin"] == s_bin), "cumulative_deltaF"]
            b_values = gene_df.loc[(gene_df["condition"] == condition_b) & (gene_df["s_bin"] == s_bin), "cumulative_deltaF"]
            n_a, n_b, mean_a, mean_b, sd_a, sd_b, t_stat, p_raw, df, zero_variance, status = two_sample_t_test(a_values, b_values)
            gene_records.append({
                "gene_id": gene_id,
                "gene_label": gene.get("gene_label", gene_id),
                "s_bin": int(s_bin),
                "s_start": cfg.S_EDGES[s_bin],
                "s_end": cfg.S_EDGES[s_bin + 1],
                "s_center": cfg.S_CENTERS[s_bin],
                "condition_a": condition_a,
                "condition_a_label": label_a,
                "condition_b": condition_b,
                "condition_b_label": label_b,
                "n_condition_a": n_a,
                "n_condition_b": n_b,
                "mean_condition_a": mean_a,
                "mean_condition_b": mean_b,
                "mean_difference_b_minus_a": mean_b - mean_a if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan,
                "sd_condition_a": sd_a,
                "sd_condition_b": sd_b,
                "t_stat": t_stat,
                "df": df,
                "p_raw": p_raw,
                "p_adj": np.nan,
                "significant_condition_difference": False,
                "alpha": cfg.ALPHA,
                "p_adjust_method": cfg.RESPONSE_TIME_TEST_P_ADJUST_METHOD,
                "test_method": "welch_two_sample_two_sided_t_test",
                "zero_variance": bool(zero_variance),
                "test_status": status,
            })
        tests = pd.DataFrame(gene_records)
        tests["p_adj"] = cfg.adjust_p_values(tests["p_raw"].to_numpy(dtype=float), method=cfg.RESPONSE_TIME_TEST_P_ADJUST_METHOD)
        tests["significant_condition_difference"] = tests["p_adj"] < cfg.ALPHA
        records.extend(tests.to_dict(orient="records"))
    return pd.DataFrame(records)


def test_axial_bins_vs_global_mean(axial_df):
    """Per (gene, condition, s_bin): paired one-sided Wilcoxon signed-rank.

    H1: the s-bin's cumulative_deltaF is higher than the worm's own
    global mean cumulative_deltaF across all s-bins.  Holm-corrected
    within each (gene, condition) across s-bins.
    """
    work = axial_df.copy()
    work["cumulative_deltaF"] = pd.to_numeric(work["cumulative_deltaF"], errors="coerce")
    work["s_bin"] = pd.to_numeric(work["s_bin"], errors="coerce")
    work = work.dropna(subset=["cumulative_deltaF", "s_bin"]).copy()
    work["s_bin"] = work["s_bin"].astype(int)

    # worm-level global mean (across all s_bins)
    global_mean = work.groupby(["gene_id", "condition", "condition_label", "instance_uid"],
                               as_index=False)["cumulative_deltaF"].mean()
    global_mean = global_mean.rename(columns={"cumulative_deltaF": "global_mean"})
    merged = work.merge(global_mean, on=["gene_id", "condition", "condition_label", "instance_uid"])

    records = []
    group_cols = GROUP_COLUMNS + ["s_bin"]
    for key, g in merged.groupby(group_cols, sort=True):
        gene_id, gene_label, condition, condition_label, s_bin = key
        diffs = (g["cumulative_deltaF"] - g["global_mean"]).to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs)]
        n = len(diffs)
        if n >= 3 and np.any(diffs != 0):
            try:
                stat, p_raw = stats.wilcoxon(diffs, alternative="greater",
                                              zero_method="wilcox", mode="auto")
            except ValueError:
                stat, p_raw = np.nan, np.nan
        else:
            stat, p_raw = np.nan, np.nan
        records.append({
            "gene_id": gene_id, "gene_label": gene_label,
            "condition": condition, "condition_label": condition_label,
            "s_bin": int(s_bin),
            "s_start": cfg.S_EDGES[s_bin], "s_end": cfg.S_EDGES[s_bin + 1],
            "s_center": cfg.S_CENTERS[s_bin],
            "n_pairs": n,
            "mean_bin_cumulative": float(np.mean(g["cumulative_deltaF"])) if n else np.nan,
            "mean_global": float(np.mean(g["global_mean"])) if n else np.nan,
            "mean_difference": float(np.mean(diffs)) if n else np.nan,
            "median_difference": float(np.median(diffs)) if n else np.nan,
            "wilcoxon_statistic": stat,
            "p_raw": p_raw,
            "p_adj": np.nan,
            "significant_bin_vs_global": False,
            "test": "one-sided paired Wilcoxon signed-rank",
            "test_H1": "bin_cumulative_deltaF > worm_global_mean",
        })
    per_bin_df = pd.DataFrame(records)
    if not per_bin_df.empty:
        for (gene_id, condition), idx in per_bin_df.groupby(["gene_id", "condition"]).groups.items():
            per_bin_df.loc[idx, "p_adj"] = cfg.adjust_p_values(
                per_bin_df.loc[idx, "p_raw"].to_numpy(dtype=float))
        per_bin_df["significant_bin_vs_global"] = per_bin_df["p_adj"] < cfg.ALPHA
    return per_bin_df


def full_grid_long(group, value_col, matrix):
    r_bins = np.repeat(np.arange(cfg.N_R_BINS, dtype=int), cfg.N_S_BINS)
    s_bins = np.tile(np.arange(cfg.N_S_BINS, dtype=int), cfg.N_R_BINS)
    values = matrix.reshape(-1)
    return pd.DataFrame({"gene_id": group["gene_id"], "gene_label": group["gene_label"], "condition": group["condition"], "condition_label": group["condition_label"], "s_bin": s_bins, "r_bin": r_bins, "s_start": cfg.S_EDGES[s_bins], "s_end": cfg.S_EDGES[s_bins + 1], "r_start": cfg.R_EDGES[r_bins], "r_end": cfg.R_EDGES[r_bins + 1], "s_center": cfg.S_CENTERS[s_bins], "r_center": cfg.R_CENTERS[r_bins], value_col: values, "has_robust_response": np.isfinite(values)})


def matrix_dataframe(matrix):
    return pd.DataFrame(matrix, index=pd.Index(np.arange(cfg.N_R_BINS), name="r_bin"), columns=pd.Index(np.arange(cfg.N_S_BINS), name="s_bin"))


def summarize_group_events(group_events):
    onset = np.full((cfg.N_R_BINS, cfg.N_S_BINS), np.nan)
    strength = np.full((cfg.N_R_BINS, cfg.N_S_BINS), np.nan)
    dff0_strength = np.full((cfg.N_R_BINS, cfg.N_S_BINS), np.nan)
    if group_events.empty:
        empty_cols = ["r_bin", "s_bin", "onset_time_min", "total_response_strength", "n_robust_timepoints"]
        return pd.DataFrame(columns=empty_cols), onset, strength, dff0_strength
    summary = group_events.groupby(["r_bin", "s_bin"], as_index=False).agg(
        onset_time_min=("time_min", "min"),
        total_response_strength=("mean_deltaF", "sum"),
        total_response_strength_dff0=("mean_dff0", "sum"),
        n_robust_timepoints=("time_min", "size"),
    )
    r_idx, s_idx = summary["r_bin"].to_numpy(dtype=int), summary["s_bin"].to_numpy(dtype=int)
    onset[r_idx, s_idx] = summary["onset_time_min"].to_numpy(dtype=float)
    strength[r_idx, s_idx] = summary["total_response_strength"].to_numpy(dtype=float)
    dff0_strength[r_idx, s_idx] = pd.to_numeric(
        summary["total_response_strength_dff0"], errors="coerce"
    ).to_numpy(dtype=float)
    return summary, onset, strength, dff0_strength


def visually_dilate_response_matrix(matrix):
    display = np.asarray(matrix, dtype=float).copy()
    if not cfg.RESPONSE_HEATMAP_VISUAL_DILATION_ENABLED:
        return display
    s_radius = max(0, int(cfg.RESPONSE_HEATMAP_VISUAL_DILATION_S_RADIUS_BINS)); r_radius = max(0, int(cfg.RESPONSE_HEATMAP_VISUAL_DILATION_R_RADIUS_BINS))
    if s_radius == 0 and r_radius == 0:
        return display
    source = np.asarray(matrix, dtype=float); response_mask = np.isfinite(source)
    if not response_mask.any():
        return display
    offsets = []
    for dr in range(-r_radius, r_radius + 1):
        for ds in range(-s_radius, s_radius + 1):
            if dr == 0 and ds == 0:
                continue
            r_term = (dr / r_radius) ** 2 if r_radius else (0.0 if dr == 0 else np.inf)
            s_term = (ds / s_radius) ** 2 if s_radius else (0.0 if ds == 0 else np.inf)
            if r_term + s_term <= 1.0 + 1e-12:
                offsets.append((r_term + s_term, dr, ds))
    n_r, n_s = source.shape
    for _, dr, ds in sorted(offsets):
        sr0, sr1, ss0, ss1 = max(0, -dr), min(n_r, n_r - dr), max(0, -ds), min(n_s, n_s - ds)
        if sr0 >= sr1 or ss0 >= ss1:
            continue
        destination = display[sr0 + dr:sr1 + dr, ss0 + ds:ss1 + ds]
        source_values = source[sr0:sr1, ss0:ss1]; source_mask = response_mask[sr0:sr1, ss0:ss1]
        fill = source_mask & ~np.isfinite(destination)
        destination[fill] = source_values[fill]
    return display


def color_limits(matrices, configured_vmin, configured_vmax):
    parts = [m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()]
    finite = np.concatenate(parts) if parts else np.asarray([])
    vmin = float(configured_vmin) if configured_vmin is not None else (float(np.min(finite)) if len(finite) else 0.0)
    vmax = float(configured_vmax) if configured_vmax is not None else (float(np.max(finite)) if len(finite) else 1.0)
    if vmax <= vmin:
        pad = max(abs(vmin) * 0.01, 0.5); vmin, vmax = vmin - pad, vmax + pad
    return vmin, vmax


def _log_strength_tick_values(vmax_raw, offset=None, metric=None):
    """Generate log-spaced ticks with clean round-number labels.

    The bottom tick is always 0.  The top tick is the nearest round number
    ≤ vmax_raw (thousands if ≥ 1000, otherwise hundreds).
    """
    if offset is None:
        offset = float(cfg.STRENGTH_LOG_OFFSET)
    offset = float(offset)
    top = float(vmax_raw)
    if top >= 1000:
        top_rounded = (int(top) // 1000) * 1000
    else:
        top_rounded = (int(top) // 100) * 100
    if top_rounded <= 0:
        top_rounded = int(top)

    # nice intermediate ticks
    candidates = [100, 300, 500, 1000, 2000, 3000, 10000, 20000]
    if metric == "dff0_strength_matrix":
        candidates = [10, 25, 50] + candidates
    ticks_orig = [t for t in candidates if t < top_rounded * 0.95]
    if top_rounded not in ticks_orig:
        ticks_orig.append(top_rounded)

    ticks_log = np.log10(np.array(ticks_orig, dtype=float) + offset)
    labels = [str(t) for t in ticks_orig]
    return ticks_log, labels


def thin_colorbar_ticks(values, labels, value_range, width_inches, fontsize):
    """Drop interior ticks whose labels would collide on a horizontal colour bar."""
    lo, hi = float(value_range[0]), float(value_range[1])
    span = hi - lo if hi > lo else 1.0
    keep_values, keep_labels = [], []
    for value, label in zip(reversed(list(values)), reversed([str(x) for x in labels])):
        if keep_values:
            gap = (keep_values[-1] - float(value)) / span * width_inches
            needed = 0.5 * 0.0078 * fontsize * (len(label) + len(keep_labels[-1])) + 0.03
            if gap < needed:
                continue
        keep_values.append(float(value)); keep_labels.append(label)
    return list(reversed(keep_values)), list(reversed(keep_labels))


def check_dff0_validity(robust_df):
    """Flag robust events whose mean_dff0 is NaN (divide-by-zero when F0≈0).

    Returns two DataFrames:
      - per_event:  every robust event with its dff0 status
      - per_group:  summary counts per gene-condition group
    """
    if robust_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = robust_df.loc[:, ["gene_id", "gene_label", "condition", "condition_label",
                               "s_bin", "r_bin", "time_min", "mean_dff0"]].copy()
    work["dff0_valid"] = pd.to_numeric(work["mean_dff0"], errors="coerce").notna()
    work["dff0_nan_reason"] = np.where(work["dff0_valid"], "", "F0_too_small_or_zero")
    per_event = work.sort_values(["gene_id", "condition", "time_min", "r_bin", "s_bin"])
    per_group = (
        per_event.groupby(["gene_id", "gene_label", "condition", "condition_label"],
                          dropna=False)
        .agg(n_robust_events=("dff0_valid", "size"),
             n_dff0_valid=("dff0_valid", "sum"),
             n_dff0_nan=("dff0_valid", lambda x: int((~x.astype(bool)).sum())))
        .assign(dff0_nan_fraction=lambda df: df["n_dff0_nan"] / df["n_robust_events"])
        .reset_index()
    )
    cfg.log("[dff0 check] Robust events with NaN dff0 (excluded from dff0 composite):")
    for _, row in per_group.iterrows():
        cfg.log(f"    {row['gene_id']} | {row['condition']}: "
                f"{int(row['n_dff0_nan'])}/{int(row['n_robust_events'])} NaN "
                f"({row['dff0_nan_fraction']:.1%})")
    return per_event, per_group


def process_heatmap_group(group, robust_df):
    group_dir = cfg.RESPONSE_HEATMAP_DIR / cfg.safe_filename(f"{group['gene_id']}__{group['condition']}")
    group_dir.mkdir(parents=True, exist_ok=True)
    events = robust_df.loc[(robust_df["gene_id"] == group["gene_id"]) & (robust_df["condition"] == group["condition"])].copy()
    summary, onset, strength, dff0_strength = summarize_group_events(events)
    save_table(full_grid_long(group, "onset_time_min", onset), group_dir / "onset_heatmap_long", "onset heatmap long table")
    save_table(matrix_dataframe(onset), group_dir / "onset_heatmap_matrix", "onset heatmap matrix", index=True, index_label="r_bin")
    save_table(full_grid_long(group, "total_response_strength", strength), group_dir / "total_response_strength_heatmap_long", "strength heatmap long table")
    save_table(matrix_dataframe(strength), group_dir / "total_response_strength_heatmap_matrix", "strength heatmap matrix", index=True, index_label="r_bin")
    save_table(full_grid_long(group, "total_response_strength_dff0", dff0_strength), group_dir / "total_response_strength_dff0_heatmap_long", "dff0 strength heatmap long table")
    save_table(matrix_dataframe(dff0_strength), group_dir / "total_response_strength_dff0_heatmap_matrix", "dff0 strength heatmap matrix", index=True, index_label="r_bin")
    return {"group": group, "group_dir": group_dir, "n_robust_events": len(events), "n_response_bins": len(summary), "onset_matrix": onset, "strength_matrix": strength, "dff0_strength_matrix": dff0_strength}


def condition_color(condition, index):
    return cfg.RESPONSE_COMPOSITE_CONDITION_COLORS.get(condition, ["#4C78A8", "#D9534F", "#59A14F", "#B279A2"][index % 4])


def panel_grid_fraction():
    """Fraction of figure height used by the panel grid (the rest is the colour-bar band)."""
    band = float(getattr(cfg, "RESPONSE_COMPOSITE_COLORBAR_BAND", 0.0))
    return min(max(1.0 - band, 0.5), 1.0)


def column_cells(column_widths):
    """[(x0, width), ...] for each column, as figure fractions."""
    cells, x0 = [], 0.0
    for width in column_widths:
        cells.append((x0, float(width)))
        x0 += float(width)
    return cells


def panel_rect(row, col, column_widths, row_count=4):
    grid = panel_grid_fraction()
    x0, width = column_cells(column_widths)[col]
    return x0, grid - (row + 1) * grid / row_count, width, grid / row_count


def add_panel_axis(fig, row, col, rect_fraction, label, column_widths, row_count=4):
    x0, y0, width, height = panel_rect(row, col, column_widths, row_count)
    left, bottom, ax_width, ax_height = rect_fraction
    ax = fig.add_axes([x0 + left * width, y0 + bottom * height, ax_width * width, ax_height * height])
    fig.text(x0 + cfg.RESPONSE_COMPOSITE_PANEL_LABEL_X_FRACTION * width, y0 + (1.0 - cfg.RESPONSE_COMPOSITE_PANEL_LABEL_Y_FRACTION) * height, label, ha="left", va="top", fontsize=cfg.RESPONSE_COMPOSITE_PANEL_LABEL_FONTSIZE, fontweight="bold")
    return ax


def italic_gene_label(label):
    return str(label)


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=cfg.RESPONSE_COMPOSITE_PANEL_TITLE_FONTSIZE, pad=4)
    ax.set_xlabel(xlabel, fontsize=cfg.RESPONSE_COMPOSITE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=cfg.RESPONSE_COMPOSITE_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=cfg.RESPONSE_COMPOSITE_TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(cfg.RESPONSE_HEATMAP_SPINE_LINEWIDTH)


def deltaf_display_scale(prefix):
    if prefix in {"total_deltaF", "cumulative_deltaF"}:
        scale = float(getattr(cfg, "RESPONSE_COMPOSITE_DELTAF_DISPLAY_SCALE", 1.0))
        return scale if np.isfinite(scale) and scale > 0 else 1.0
    return 1.0


def scaled_deltaf_ylabel(base_label):
    scale = float(getattr(cfg, "RESPONSE_COMPOSITE_DELTAF_DISPLAY_SCALE", 1.0))
    label = str(getattr(cfg, "RESPONSE_COMPOSITE_DELTAF_DISPLAY_SCALE_LABEL", ""))
    if np.isfinite(scale) and scale != 1.0 and label:
        return f"{base_label} (x{label})"
    return base_label


def plot_mean_ci(ax, summary, x_col, prefix, condition, label, color):
    data = summary.loc[summary["condition"] == condition].sort_values(x_col)
    if data.empty:
        return
    x = data[x_col].to_numpy(dtype=float); mean = data[f"{prefix}_mean"].to_numpy(dtype=float)
    low = data[f"{prefix}_ci_low"].to_numpy(dtype=float); high = data[f"{prefix}_ci_high"].to_numpy(dtype=float)
    scale = deltaf_display_scale(prefix)
    mean = mean / scale
    low = low / scale
    high = high / scale
    ax.plot(x, mean, color=color, linewidth=cfg.RESPONSE_COMPOSITE_LINEWIDTH, marker="o", markersize=cfg.RESPONSE_COMPOSITE_MARKER_SIZE, label=label)
    finite = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
    if finite.any():
        ax.fill_between(x[finite], low[finite], high[finite], color=color, alpha=cfg.RESPONSE_COMPOSITE_CI_ALPHA, linewidth=0)


def annotate_axial_significance(ax, gene_summary, axial_tests, y_scale=1.0):
    if axial_tests is None or axial_tests.empty or gene_summary.empty:
        return
    significant = axial_tests.loc[axial_tests["significant_condition_difference"].astype(bool)].copy()
    if significant.empty:
        return
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    star_ys = []
    for row in significant.itertuples(index=False):
        local = gene_summary.loc[gene_summary["s_bin"] == row.s_bin]
        local_high = pd.to_numeric(local.get("cumulative_deltaF_ci_high", pd.Series(dtype=float)), errors="coerce")
        local_mean = pd.to_numeric(local.get("cumulative_deltaF_mean", pd.Series(dtype=float)), errors="coerce")
        values = pd.concat([local_high, local_mean], ignore_index=True).dropna()
        values = values / y_scale
        y_base = float(values.max()) if not values.empty else ymax
        y = y_base + 0.06 * yrange
        star_ys.append(y)
        ax.text(row.s_center, y, "*", ha="center", va="bottom", fontsize=cfg.RESPONSE_COMPOSITE_PANEL_LABEL_FONTSIZE, fontweight="bold", color="black")
    if star_ys:
        ax.set_ylim(ymin, max(ymax, max(star_ys) + 0.08 * yrange))


def add_line_panel(ax, gene, summary, x_col, prefix, title_suffix, xlabel, ylabel, axial_tests=None):
    data = summary.loc[summary["gene_id"] == gene["gene_id"]]
    conditions = []
    for group in cfg.INPUT_GROUPS:
        if group["gene_id"] == gene["gene_id"] and group["condition"] not in [x[0] for x in conditions]:
            conditions.append((group["condition"], group.get("condition_label", group["condition"])))
    for index, (condition, label) in enumerate(conditions):
        plot_mean_ci(ax, data, x_col, prefix, condition, label, condition_color(condition, index))
    if axial_tests is not None:
        annotate_axial_significance(ax, data, axial_tests, y_scale=deltaf_display_scale(prefix))
    style_axis(ax, italic_gene_label(gene["gene_label"]), xlabel, scaled_deltaf_ylabel(ylabel))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=max(6, cfg.RESPONSE_COMPOSITE_TICK_FONTSIZE - 1), frameon=False)


def heatmap_log_scale_enabled(metric, display):
    return bool(
        cfg.STRENGTH_HEATMAP_LOG_SCALE
        and metric in ("strength_matrix", "dff0_strength_matrix")
        and np.any(np.isfinite(display))
    )


def strength_log_offset(metric):
    return float(cfg.STRENGTH_DFF0_LOG_OFFSET if metric == "dff0_strength_matrix" else cfg.STRENGTH_LOG_OFFSET)


def heatmap_colorbar_label(metric, log_scale):
    if metric == "onset_matrix":
        return "Onset time (min)"
    label_text = "Cumulative mean ΔF / F$_0$" if metric == "dff0_strength_matrix" else "Cumulative mean ΔF"
    return f"{label_text}  [log]" if log_scale else label_text


def log_strength_values(values, offset):
    """log10(offset + x) transform used for the strength colour scale; invalid entries stay NaN."""
    raw = np.asarray(values, dtype=float)
    out = np.full_like(raw, np.nan, dtype=float)
    valid = np.isfinite(raw) & (raw >= 0)
    out[valid] = np.log10(np.maximum(raw[valid], 0.0) + offset)
    return out


def dilation_alpha():
    """Transparency for dilation-only pixels (1.0 = opaque); measured bins are drawn opaque."""
    if not cfg.RESPONSE_HEATMAP_VISUAL_DILATION_ENABLED:
        return 1.0
    alpha = float(getattr(cfg, "RESPONSE_HEATMAP_VISUAL_DILATION_ALPHA", 1.0))
    return min(max(alpha, 0.0), 1.0)


def add_heatmap_panel(fig, row, col, label, data, metric, cmap_name, cmap_colors, vmin, vmax, column_widths, row_count=4):
    """Draw one heat-map panel without a colour bar; the column bar is drawn by add_column_colorbar.

    Pixels added by the display-only visual dilation are faded to
    RESPONSE_HEATMAP_VISUAL_DILATION_ALPHA; the measured bins are re-drawn opaque on top.
    """
    ax = add_panel_axis(fig, row, col, cfg.RESPONSE_COMPOSITE_HEATMAP_AX_RECT, label, column_widths, row_count)
    cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors); cmap.set_bad(cfg.NO_RESPONSE_COLOR)

    measured = np.asarray(data[metric], dtype=float)
    display = visually_dilate_response_matrix(measured)
    display_vmin = vmin
    display_vmax = vmax
    log_scale = heatmap_log_scale_enabled(metric, display)

    if log_scale:
        offset = strength_log_offset(metric)
        measured = log_strength_values(measured, offset)
        display = log_strength_values(display, offset)
        display_vmin = np.log10(max(max(vmin, 0.0), 0.0) + offset) if np.isfinite(vmin) else np.log10(offset)
        display_vmax = np.log10(max(max(vmax, 0.0), 0.0) + offset) if np.isfinite(vmax) else np.log10(offset) + 1.0
        if display_vmax <= display_vmin:
            display_vmax = display_vmin + 1.0

    image_kwargs = {
        "origin": "lower",
        "extent": [cfg.S_RANGE[0], cfg.S_RANGE[1], cfg.R_RANGE[0], cfg.R_RANGE[1]],
        "interpolation": cfg.RESPONSE_HEATMAP_INTERPOLATION,
        "aspect": cfg.RESPONSE_HEATMAP_ASPECT,
        "cmap": cmap,
        "vmin": display_vmin,
        "vmax": display_vmax,
    }
    alpha = dilation_alpha()
    if alpha < 1.0:
        # Base layer carries measured + dilation-only pixels at reduced opacity.
        ax.imshow(np.ma.masked_invalid(display), zorder=1, alpha=alpha, **image_kwargs)
        # Overlay re-draws the measured bins opaque; its empty pixels must stay
        # transparent (the normal bad colour is opaque white and would hide the base).
        overlay_cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors)
        overlay_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        image = ax.imshow(np.ma.masked_invalid(measured), zorder=2, **dict(image_kwargs, cmap=overlay_cmap))
    else:
        image = ax.imshow(np.ma.masked_invalid(display), zorder=1, **image_kwargs)
    group = data["group"]
    style_axis(
        ax,
        f"{italic_gene_label(group['gene_label'])} {group['condition_label']}",
        "Axial position",
        "Radial position",
    )
    return image, log_scale


def add_column_colorbar(fig, col, image, label_text, tick_values=None, tick_labels=None, column_widths=None, row_count=4):
    """Horizontal colour bar spanning the top of one heat-map column."""
    x0, _, column_width, _ = panel_rect(0, col, column_widths, row_count)
    panel_left, _, panel_width, _ = cfg.RESPONSE_COMPOSITE_HEATMAP_AX_RECT
    bar_bottom = (1.0 - float(cfg.RESPONSE_COMPOSITE_COLORBAR_BAND)) + float(cfg.RESPONSE_COMPOSITE_COLORBAR_BAR_PAD)
    cax = fig.add_axes([
        x0 + panel_left * column_width,
        bar_bottom,
        panel_width * column_width,
        float(cfg.RESPONSE_COMPOSITE_COLORBAR_BAR_HEIGHT),
    ])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=cfg.RESPONSE_COMPOSITE_COLORBAR_FONTSIZE, pad=1, length=2)
    if tick_values is not None:
        bar_width_inches = panel_width * column_width * float(fig.get_size_inches()[0])
        tick_values, tick_labels = thin_colorbar_ticks(tick_values, tick_labels, image.get_clim(),
                                                       bar_width_inches, cfg.RESPONSE_COMPOSITE_COLORBAR_FONTSIZE)
        colorbar.set_ticks(tick_values)
        colorbar.set_ticklabels(tick_labels)
    colorbar.set_label(label_text, fontsize=cfg.RESPONSE_COMPOSITE_COLORBAR_FONTSIZE, labelpad=2)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("bottom")
    return colorbar


def draw_onset_column(fig, heatmaps, column_widths, labels, col=0, row_count=4):
    """Draw the onset heat-map column (E-H in the main figure) with one shared colour bar above it."""
    onset_vmin, onset_vmax = color_limits([x["onset_matrix"] for x in heatmaps[:4]], cfg.ONSET_VMIN, cfg.ONSET_VMAX)
    image = None
    for row, data in enumerate(heatmaps[:4]):
        image, _ = add_heatmap_panel(fig, row, col, labels[row], data, "onset_matrix",
                                     cfg.ONSET_CMAP_NAME, cfg.ONSET_CMAP_COLORS, onset_vmin, onset_vmax,
                                     column_widths, row_count)
    add_column_colorbar(fig, col, image, heatmap_colorbar_label("onset_matrix", False),
                        None, None, column_widths, row_count)
    return onset_vmin, onset_vmax


def draw_strength_column(fig, heatmaps, strength_metric, column_widths, labels, col=0, row_count=4):
    """Draw a strength heat-map column (standalone figure) with one shared colour bar above it."""
    strength_vmin, strength_vmax = color_limits([x[strength_metric] for x in heatmaps[:4]], cfg.STRENGTH_VMIN, cfg.STRENGTH_VMAX)
    image = None
    strength_log_scale = False
    for row, data in enumerate(heatmaps[:4]):
        image, strength_log_scale = add_heatmap_panel(fig, row, col, labels[row], data, strength_metric,
                                                      cfg.STRENGTH_CMAP_NAME, cfg.STRENGTH_CMAP_COLORS,
                                                      strength_vmin, strength_vmax, column_widths, row_count)
    tick_values, tick_labels = (None, None)
    if strength_log_scale:
        tick_values, tick_labels = _log_strength_tick_values(strength_vmax, offset=strength_log_offset(strength_metric), metric=strength_metric)
    add_column_colorbar(fig, col, image, heatmap_colorbar_label(strength_metric, strength_log_scale),
                        tick_values, tick_labels, column_widths, row_count)
    return strength_vmin, strength_vmax


def save_composite(fig, basename=None):
    if basename is None:
        basename = cfg.RESPONSE_COMPOSITE_OUTPUT_BASENAME
    base = cfg.RESPONSE_HEATMAP_DIR / basename
    files = {"png": "", "pdf": "", "svg": ""}
    for ext, enabled in [("png", cfg.RESPONSE_COMPOSITE_SAVE_PNG), ("pdf", cfg.RESPONSE_COMPOSITE_SAVE_PDF), ("svg", cfg.RESPONSE_COMPOSITE_SAVE_SVG)]:
        if enabled:
            output = base.with_suffix(f".{ext}"); fig.savefig(output, dpi=cfg.RESPONSE_COMPOSITE_DPI if ext == "png" else None); files[ext] = str(output)
    return files


def unique_genes(groups):
    genes, seen = [], set()
    for group in groups:
        if group["gene_id"] not in seen:
            genes.append({"gene_id": group["gene_id"], "gene_label": group["gene_label"]}); seen.add(group["gene_id"])
    return genes


def make_composite(full_summary, axial_summary, heatmaps, genes, axial_condition_tests=None):
    """Main figure, 4 rows x 2 cols: Col 0 = A-D line plots, Col 1 = E-H onset heat maps.

    The onset column carries one shared horizontal colour bar drawn above the grid.
    """
    if len(genes) < 2 or len(heatmaps) < 4:
        raise ValueError("Composite figure requires at least two genes and four configured gene-condition groups.")
    fig = plt.figure(figsize=cfg.RESPONSE_COMPOSITE_FIGSIZE, dpi=cfg.RESPONSE_COMPOSITE_DPI)
    labels = list("ABCDEFGH")
    widths = cfg.RESPONSE_COMPOSITE_COLUMN_WIDTHS

    # Col 0 — A-D: whole-worm and axial line curves
    ax = add_panel_axis(fig, 0, 0, cfg.RESPONSE_COMPOSITE_LINE_AX_RECT, labels[0], widths)
    add_line_panel(ax, genes[0], full_summary, "time_min", "total_deltaF",
                   "Whole-worm response", "time (min)", "Mean total ΔF")
    ax = add_panel_axis(fig, 1, 0, cfg.RESPONSE_COMPOSITE_LINE_AX_RECT, labels[1], widths)
    add_line_panel(ax, genes[1], full_summary, "time_min", "total_deltaF",
                   "Whole-worm response", "time (min)", "Mean total ΔF")
    panel_c_tests = axial_condition_tests.loc[axial_condition_tests["gene_id"] == genes[0]["gene_id"]].copy() if axial_condition_tests is not None and not axial_condition_tests.empty else pd.DataFrame()
    panel_d_tests = axial_condition_tests.loc[axial_condition_tests["gene_id"] == genes[1]["gene_id"]].copy() if axial_condition_tests is not None and not axial_condition_tests.empty else pd.DataFrame()
    ax = add_panel_axis(fig, 2, 0, cfg.RESPONSE_COMPOSITE_LINE_AX_RECT, labels[2], widths)
    add_line_panel(ax, genes[0], axial_summary, "s_center", "cumulative_deltaF",
                   "Axial cumulative response", "Axial position",
                   "Mean cumulative ΔF", axial_tests=panel_c_tests)
    ax = add_panel_axis(fig, 3, 0, cfg.RESPONSE_COMPOSITE_LINE_AX_RECT, labels[3], widths)
    add_line_panel(ax, genes[1], axial_summary, "s_center", "cumulative_deltaF",
                   "Axial cumulative response", "Axial position",
                   "Mean cumulative ΔF", axial_tests=panel_d_tests)

    # Col 1 → E-H onset, one shared colour bar on top of the column
    onset_vmin, onset_vmax = draw_onset_column(fig, heatmaps, widths, labels[4:8], col=1)

    files = save_composite(fig, basename=cfg.RESPONSE_COMPOSITE_OUTPUT_BASENAME); plt.close(fig)
    return files, onset_vmin, onset_vmax


def make_strength_composite(heatmaps, strength_metric, basename):
    """Standalone 4 x 1 strength figure: A-D heat-map panels with one shared colour bar above."""
    if len(heatmaps) < 4:
        raise ValueError("Strength composite requires at least four gene-condition groups.")
    fig = plt.figure(figsize=cfg.RESPONSE_STRENGTH_FIGSIZE, dpi=cfg.RESPONSE_COMPOSITE_DPI)
    labels = list("ABCD")
    widths = cfg.RESPONSE_STRENGTH_COLUMN_WIDTHS

    strength_vmin, strength_vmax = draw_strength_column(fig, heatmaps, strength_metric, widths, labels, col=0)

    files = save_composite(fig, basename=basename); plt.close(fig)
    return files, strength_vmin, strength_vmax


def output_is_complete():
    manifest = cfg.compressed_path(cfg.RESPONSE_HEATMAP_DIR / "all_response_heatmap_manifest")
    pngs = [
        cfg.RESPONSE_HEATMAP_DIR / f"{cfg.RESPONSE_COMPOSITE_OUTPUT_BASENAME}.png",
        cfg.RESPONSE_HEATMAP_DIR / f"{cfg.RESPONSE_STRENGTH_COMPOSITE_OUTPUT_BASENAME}.png",
        cfg.RESPONSE_HEATMAP_DIR / f"{cfg.RESPONSE_STRENGTH_DFF0_COMPOSITE_OUTPUT_BASENAME}.png",
    ]
    panel_c = cfg.RESPONSE_HEATMAP_DIR / "panel_C_axial_s_bin_condition_tests.csv"
    panel_d = cfg.RESPONSE_HEATMAP_DIR / "panel_D_axial_s_bin_condition_tests.csv"
    png_ok = (not cfg.RESPONSE_COMPOSITE_SAVE_PNG) or all(path.exists() for path in pngs)
    return manifest.exists() and panel_c.exists() and panel_d.exists() and png_ok


def main():
    cfg.ensure_stage_dirs(); cfg.write_global_settings(module_name="07_response_timing_and_strength_heatmaps")
    if output_is_complete() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 7 output already exists; skipping: {cfg.RESPONSE_HEATMAP_DIR}"); return
    cfg.log("[Module 7] Building A-L composite response figure.")
    robust_df = read_robust_events(); delta_manifest = read_delta_manifest(); groups = configured_groups(robust_df, delta_manifest); genes = unique_genes(groups)
    cfg.log("[Module 7] Aggregating Module 2 instance-level deltaF for Panels A-D.")
    full_df, axial_df = aggregate_module2(delta_manifest)
    full_summary = summarize_values(full_df, GROUP_COLUMNS + ["time_min"], "total_deltaF", "total_deltaF")
    axial_summary = summarize_values(axial_df, GROUP_COLUMNS + ["s_bin"], "cumulative_deltaF", "cumulative_deltaF")
    if not axial_summary.empty:
        bins = axial_summary["s_bin"].astype(int).to_numpy(); axial_summary["s_start"] = cfg.S_EDGES[bins]; axial_summary["s_end"] = cfg.S_EDGES[bins + 1]; axial_summary["s_center"] = cfg.S_CENTERS[bins]
    tests = test_total_deltaF_vs_zero(full_df)
    axial_condition_tests = test_axial_condition_differences(axial_df, genes)
    panel_c_tests = axial_condition_tests.loc[axial_condition_tests["gene_id"] == genes[0]["gene_id"]].copy() if len(genes) > 0 and not axial_condition_tests.empty else pd.DataFrame()
    panel_d_tests = axial_condition_tests.loc[axial_condition_tests["gene_id"] == genes[1]["gene_id"]].copy() if len(genes) > 1 and not axial_condition_tests.empty else pd.DataFrame()
    save_table(full_df, cfg.RESPONSE_HEATMAP_DIR / "full_worm_instance_time_total_deltaF", "full-worm instance-time totals")
    save_table(full_summary, cfg.RESPONSE_HEATMAP_DIR / "full_worm_time_summary", "full-worm time summary")
    save_table(tests, cfg.RESPONSE_HEATMAP_DIR / "full_worm_total_deltaF_vs_zero_tests", "total deltaF vs zero tests")
    save_table(axial_df, cfg.RESPONSE_HEATMAP_DIR / "axial_instance_s_bin_cumulative_deltaF", "axial instance-s-bin totals")
    save_table(axial_summary, cfg.RESPONSE_HEATMAP_DIR / "axial_s_bin_summary", "axial s-bin summary")
    save_table(panel_c_tests, cfg.RESPONSE_HEATMAP_DIR / "panel_C_axial_s_bin_condition_tests", "Panel C axial condition tests")
    save_table(panel_d_tests, cfg.RESPONSE_HEATMAP_DIR / "panel_D_axial_s_bin_condition_tests", "Panel D axial condition tests")
    cfg.log("[Module 7] Testing within-group axial bins vs global mean.")
    axial_bin_vs_global = test_axial_bins_vs_global_mean(axial_df)
    save_table(axial_bin_vs_global, cfg.RESPONSE_HEATMAP_DIR / "axial_bin_vs_global_mean_tests", "axial bin vs global mean tests")
    cfg.log("[Module 7] Preparing Module 4 robust-event heatmaps for the heat-map columns.")
    heatmaps = [process_heatmap_group(group, robust_df) for group in groups]
    cfg.log("[Module 7] Drawing the timing composite (A-D curves + E-H onset).")
    files, onset_vmin, onset_vmax = make_composite(full_summary, axial_summary, heatmaps, genes, axial_condition_tests=axial_condition_tests)
    cfg.log("[Module 7] Checking dff0 validity for robust events before the strength figures.")
    dff0_events, dff0_groups = check_dff0_validity(robust_df)
    save_table(dff0_events, cfg.RESPONSE_HEATMAP_DIR / "dff0_validity_per_event", "dff0 validity per event")
    save_table(dff0_groups, cfg.RESPONSE_HEATMAP_DIR / "dff0_validity_per_group", "dff0 validity per group")
    cfg.log("[Module 7] Drawing the standalone strength figures (deltaF and dff0).")
    strength_files, strength_vmin, strength_vmax = make_strength_composite(heatmaps, "strength_matrix", cfg.RESPONSE_STRENGTH_COMPOSITE_OUTPUT_BASENAME)
    dff0_files, dff0_strength_vmin, dff0_strength_vmax = make_strength_composite(heatmaps, "dff0_strength_matrix", cfg.RESPONSE_STRENGTH_DFF0_COMPOSITE_OUTPUT_BASENAME)
    manifest = []
    for index, data in enumerate(heatmaps):
        group = data["group"]
        onset = data["onset_matrix"][np.isfinite(data["onset_matrix"])]
        strength = data["strength_matrix"][np.isfinite(data["strength_matrix"])]
        dff0_str = data["dff0_strength_matrix"][np.isfinite(data["dff0_strength_matrix"])]
        manifest.append({**group, "heatmap_column_index": index + 1, "onset_panel": chr(ord("E") + index) if index < 4 else "", "strength_panel": chr(ord("A") + index) if index < 4 else "", "n_robust_events": data["n_robust_events"], "n_response_bins": data["n_response_bins"], "earliest_onset": float(np.min(onset)) if len(onset) else np.nan, "latest_onset": float(np.max(onset)) if len(onset) else np.nan, "min_total_response_strength": float(np.min(strength)) if len(strength) else np.nan, "max_total_response_strength": float(np.max(strength)) if len(strength) else np.nan, "min_dff0_strength": float(np.min(dff0_str)) if len(dff0_str) else np.nan, "max_dff0_strength": float(np.max(dff0_str)) if len(dff0_str) else np.nan, "whether_any_response": bool(data["n_response_bins"]), "onset_plot_vmin": onset_vmin, "onset_plot_vmax": onset_vmax, "strength_plot_vmin": strength_vmin, "strength_plot_vmax": strength_vmax, "dff0_strength_plot_vmin": dff0_strength_vmin, "dff0_strength_plot_vmax": dff0_strength_vmax, "composite_figure_png": files["png"], "composite_figure_pdf": files["pdf"], "composite_figure_svg": files["svg"], "strength_figure_png": strength_files["png"], "strength_figure_pdf": strength_files["pdf"], "strength_figure_svg": strength_files["svg"], "strength_dff0_figure_png": dff0_files["png"], "strength_dff0_figure_pdf": dff0_files["pdf"], "strength_dff0_figure_svg": dff0_files["svg"], "group_output_dir": str(data["group_dir"])})
    save_table(pd.DataFrame(manifest), cfg.RESPONSE_HEATMAP_DIR / "all_response_heatmap_manifest", "response heatmap manifest")
    cfg.log("[Module 7] Summary:"); cfg.log(f"    Full-worm rows: {len(full_df)}"); cfg.log(f"    Axial rows: {len(axial_df)}"); cfg.log(f"    Timing composite PNG: {files['png']}"); cfg.log(f"    Strength PNG: {strength_files['png']}"); cfg.log(f"    Strength dff0 PNG: {dff0_files['png']}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 7] Stopped gracefully: {exc}")


