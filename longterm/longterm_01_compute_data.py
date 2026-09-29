# -*- coding: utf-8 -*-
"""
Long-term Module 1: load CSV data, compute all statistics needed for panels A-L.

Inputs:
    Long-term pixel CSV folders under DAY_FOLDERS.

Outputs:
    COMPUTED_DATA_DIR/instance_summary.pkl.gz
    COMPUTED_DATA_DIR/welch_ttest.pkl.gz
    COMPUTED_DATA_DIR/ap_bin_stats.pkl.gz
    COMPUTED_DATA_DIR/ap_bin_sig_ranges.pkl.gz
    COMPUTED_DATA_DIR/heatmap_data.pkl.gz
    COMPUTED_DATA_DIR/computed_data_manifest.txt

Depends on:
    longterm_00_config_utils.py only.
"""

import importlib.util
import pickle
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _load_config():
    module_path = Path(__file__).resolve().with_name("longterm_00_config_utils.py")
    spec = importlib.util.spec_from_file_location("lt_config", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()


# ============================================================
# Worker-level helpers (module scope — pickle-safe on Windows)
# ============================================================

def _get_top5_threshold_and_contribution(h_values):
    if h_values.size == 0:
        return np.nan, np.nan
    h_positive = np.maximum(h_values, 0)
    h_positive = h_positive[np.isfinite(h_positive)]
    if h_positive.size == 0:
        return np.nan, np.nan
    n = h_positive.size
    k = max(1, int(np.ceil(cfg.TOP_FRACTION * n)))
    total_signal = np.sum(h_positive)
    if total_signal <= 0:
        return np.nan, np.nan
    top_k_values = np.partition(h_positive, n - k)[n - k:]
    contribution = np.sum(top_k_values) / total_signal
    threshold = np.min(top_k_values)
    return threshold, contribution


def _process_single_csv(csv_path_str, day):
    """Worker: read one CSV, compute all per-file metrics and local accumulators.

    Returns a dict with everything the main process needs to aggregate.
    Uses module-level ``cfg`` (re-imported in each worker on Windows).
    """
    csv_path = Path(csv_path_str)
    col_map = cfg.get_column_mapping(csv_path, required_cols=["h_net", "is_signal", "s", "r_3d"])
    h_col   = col_map["h_net"]
    sig_col = col_map["is_signal"]
    s_col   = col_map["s"]
    r_col   = col_map["r_3d"]

    # --- per-file accumulators (returned to main process) ---
    total_h_net = 0.0
    total_pixel_count = 0
    signal_pixel_count = 0
    signal_h_sum = 0.0
    signal_h_count = 0
    r3d_valid_candidate_pixels = 0
    r3d_gt1_discarded_pixels = 0

    ap_bin_h_sum = np.zeros(cfg.AP_N_BINS, dtype=float)
    ap_bin_signal_count = np.zeros(cfg.AP_N_BINS, dtype=np.int64)
    h_values_chunks = []

    # --- local heatmap accumulators (only for target days) ---
    is_target = day in cfg.TARGET_HEATMAP_DAYS
    if is_target:
        _local_mean_h_sum = np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=float)
        _local_mean_signal_count = np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64)
        _local_top_total = np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64)
        _local_top_count = np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64)

    # --- first pass: chunked reading ---
    for chunk in pd.read_csv(csv_path, usecols=[h_col, sig_col, s_col, r_col],
                             chunksize=cfg.CHUNKSIZE):
        if cfg.should_stop_requested():
            raise cfg.PipelineStopRequested(f"stop inside worker: {csv_path.name}")
        h = pd.to_numeric(chunk[h_col], errors="coerce").to_numpy(dtype=float)
        is_signal = pd.to_numeric(chunk[sig_col], errors="coerce").fillna(0).to_numpy(dtype=float) == 1
        s = pd.to_numeric(chunk[s_col], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(chunk[r_col], errors="coerce").to_numpy(dtype=float)

        total_pixel_count += len(chunk)
        signal_pixel_count += int(np.sum(is_signal))
        valid_h = np.isfinite(h)
        total_h_net += float(np.nansum(h))

        valid_r3d_candidate = (
            valid_h & np.isfinite(s) & np.isfinite(r)
            & (s >= cfg.S_MIN) & (s < cfg.S_MAX) & (r >= cfg.R_MIN)
        )
        r3d_valid_candidate_pixels += int(np.sum(valid_r3d_candidate))
        r3d_gt1_discarded_pixels += int(np.sum(valid_r3d_candidate & (r > 1.0)))

        if np.any(valid_h):
            h_values_chunks.append(h[valid_h])

        signal_valid = is_signal & valid_h
        if np.any(signal_valid):
            signal_h_sum += float(np.sum(h[signal_valid]))
            signal_h_count += int(np.sum(signal_valid))

        valid_ap = signal_valid & np.isfinite(s) & (s >= 0) & (s < 1)
        if np.any(valid_ap):
            ap_idx = np.floor(s[valid_ap] / cfg.AP_BIN_WIDTH).astype(int)
            ap_idx = np.clip(ap_idx, 0, cfg.AP_N_BINS - 1)
            np.add.at(ap_bin_h_sum, ap_idx, h[valid_ap])
            np.add.at(ap_bin_signal_count, ap_idx, 1)

        if is_target:
            valid_sr = (
                valid_h & np.isfinite(s) & np.isfinite(r)
                & (s >= cfg.S_MIN) & (s < cfg.S_MAX)
                & (r >= cfg.R_MIN) & (r < cfg.R_MAX)
            )
            if np.any(valid_sr):
                s_v = s[valid_sr]; r_v = r[valid_sr]; h_v = h[valid_sr]
                sig_v = is_signal[valid_sr]
                s_idx = np.floor((s_v - cfg.S_MIN) / cfg.S_BIN_WIDTH).astype(int)
                r_idx = np.floor((r_v - cfg.R_MIN) / cfg.R_BIN_WIDTH).astype(int)
                s_idx = np.clip(s_idx, 0, cfg.N_S_BINS - 1)
                r_idx = np.clip(r_idx, 0, cfg.N_R_BINS - 1)
                np.add.at(_local_mean_h_sum, (r_idx, s_idx), h_v)
                if np.any(sig_v):
                    np.add.at(_local_mean_signal_count,
                              (r_idx[sig_v], s_idx[sig_v]), 1)

    h_values = np.concatenate(h_values_chunks) if h_values_chunks else np.array([], dtype=float)
    threshold, top5_contribution = _get_top5_threshold_and_contribution(h_values)
    signal_mean_h_net = signal_h_sum / signal_h_count if signal_h_count > 0 else np.nan
    signal_ratio = signal_pixel_count / total_pixel_count if total_pixel_count > 0 else np.nan
    ap_curve = np.full(cfg.AP_N_BINS, np.nan, dtype=float)
    valid_bins = ap_bin_signal_count > 0
    ap_curve[valid_bins] = ap_bin_h_sum[valid_bins] / ap_bin_signal_count[valid_bins]

    record = {
        "day": day,
        "file": csv_path.name,
        "total_h_net": total_h_net,
        "signal_mean_h_net": signal_mean_h_net,
        "signal_ratio": signal_ratio,
        "top5_contribution": top5_contribution,
        "top5_threshold": threshold,
        "n_pixels": int(h_values.size),
        "r3d_gt1_discarded_pixels": r3d_gt1_discarded_pixels,
        "r3d_valid_candidate_pixels": r3d_valid_candidate_pixels,
        "r3d_gt1_discarded_fraction": (
            r3d_gt1_discarded_pixels / r3d_valid_candidate_pixels
            if r3d_valid_candidate_pixels > 0 else np.nan
        ),
    }

    # --- second pass: top-5% accumulation (only if threshold is valid) ---
    if is_target and np.isfinite(threshold):
        col_map2 = cfg.get_column_mapping(csv_path, required_cols=["h_net", "s", "r_3d"])
        h_c2 = col_map2["h_net"]; s_c2 = col_map2["s"]; r_c2 = col_map2["r_3d"]
        for chunk in pd.read_csv(csv_path, usecols=[h_c2, s_c2, r_c2],
                                 chunksize=cfg.CHUNKSIZE):
            h2 = pd.to_numeric(chunk[h_c2], errors="coerce").to_numpy(dtype=float)
            s2 = pd.to_numeric(chunk[s_c2], errors="coerce").to_numpy(dtype=float)
            r2 = pd.to_numeric(chunk[r_c2], errors="coerce").to_numpy(dtype=float)
            h2_pos = np.maximum(h2, 0)
            valid2 = (
                np.isfinite(h2_pos) & np.isfinite(s2) & np.isfinite(r2)
                & (s2 >= cfg.S_MIN) & (s2 < cfg.S_MAX)
                & (r2 >= cfg.R_MIN) & (r2 < cfg.R_MAX)
            )
            if not np.any(valid2):
                continue
            s2v = s2[valid2]; r2v = r2[valid2]; h2v = h2_pos[valid2]
            si = np.floor((s2v - cfg.S_MIN) / cfg.S_BIN_WIDTH).astype(int)
            ri = np.floor((r2v - cfg.R_MIN) / cfg.R_BIN_WIDTH).astype(int)
            si = np.clip(si, 0, cfg.N_S_BINS - 1)
            ri = np.clip(ri, 0, cfg.N_R_BINS - 1)
            np.add.at(_local_top_total, (ri, si), 1)
            top_v = h2v >= threshold
            if np.any(top_v):
                np.add.at(_local_top_count, (ri[top_v], si[top_v]), 1)

    return {
        "record": record,
        "ap_curve": ap_curve,
        "mean_h_sum": _local_mean_h_sum if is_target else None,
        "mean_signal_count": _local_mean_signal_count if is_target else None,
        "top_total": _local_top_total if is_target else None,
        "top_count": _local_top_count if is_target else None,
    }


# ============================================================
# Main-process aggregation
# ============================================================

def load_all_data():
    records = []
    ap_bin_by_day = {d: [] for d in sorted(cfg.DAY_FOLDERS)}

    accumulators = {
        "top_total_count":   {d: np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64) for d in cfg.TARGET_HEATMAP_DAYS},
        "top_count":         {d: np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64) for d in cfg.TARGET_HEATMAP_DAYS},
        "mean_h_sum":        {d: np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=float)   for d in cfg.TARGET_HEATMAP_DAYS},
        "mean_signal_count": {d: np.zeros((cfg.N_R_BINS, cfg.N_S_BINS), dtype=np.int64) for d in cfg.TARGET_HEATMAP_DAYS},
    }

    # Build task list
    all_tasks = []
    for day, folder in sorted(cfg.DAY_FOLDERS.items()):
        folder = Path(folder)
        csv_files = sorted(folder.glob(cfg.CSV_GLOB))
        if len(csv_files) == 0:
            warnings.warn(f"Day {day}: no CSV found in {folder}")
            continue
        cfg.log(f"[Read] Day {day}: {len(csv_files)} CSV files")
        for csv_path in csv_files:
            all_tasks.append((str(csv_path), day))

    n_total = len(all_tasks)
    if n_total == 0:
        raise RuntimeError("No CSV files found in any day folder.")

    use_parallel = cfg.USE_MULTIPROCESSING and n_total > 1 and cfg.MAX_WORKERS > 1
    workers = min(cfg.MAX_WORKERS, n_total) if use_parallel else 1

    if use_parallel:
        cfg.log(f"[Parallel] CSV processing enabled: workers={workers}, tasks={n_total}")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_csv, p, d): (p, d)
                       for p, d in all_tasks}
            for done_i, future in enumerate(as_completed(futures), start=1):
                cfg.raise_if_stop_requested(f"module 1 after CSV {done_i}/{n_total}")
                _, day = futures[future]
                try:
                    res = future.result()
                except cfg.PipelineStopRequested:
                    cfg.log("    [stop] Worker aborted; propagating stop request.")
                    raise
                except Exception as exc:
                    warnings.warn(f"Worker failed: {all_tasks[list(futures.values()).index((futures[future][0], day))][0]} — {exc}")
                    continue
                records.append(res["record"])
                ap_bin_by_day[day].append(res["ap_curve"])
                if res["mean_h_sum"] is not None:
                    accumulators["mean_h_sum"][day]        += res["mean_h_sum"]
                    accumulators["mean_signal_count"][day]  += res["mean_signal_count"]
                    accumulators["top_total_count"][day]    += res["top_total"]
                    accumulators["top_count"][day]          += res["top_count"]
                if (cfg.PROGRESS_EVERY_N_FILES
                        and (done_i % cfg.PROGRESS_EVERY_N_FILES == 0
                             or done_i == 1 or done_i == n_total)):
                    cfg.log(f"    Finished CSV {done_i}/{n_total}")
    else:
        cfg.log("[Serial] CSV processing sequentially.")
        for i, (csv_path_str, day) in enumerate(all_tasks, start=1):
            cfg.raise_if_stop_requested(f"module 1 before CSV {i}/{n_total}")
            try:
                res = _process_single_csv(csv_path_str, day)
            except Exception as exc:
                warnings.warn(f"Skipped file {csv_path_str}: {exc}")
                continue
            records.append(res["record"])
            ap_bin_by_day[day].append(res["ap_curve"])
            if res["mean_h_sum"] is not None:
                accumulators["mean_h_sum"][day]        += res["mean_h_sum"]
                accumulators["mean_signal_count"][day]  += res["mean_signal_count"]
                accumulators["top_total_count"][day]    += res["top_total"]
                accumulators["top_count"][day]          += res["top_count"]
            if (cfg.PROGRESS_EVERY_N_FILES
                    and (i % cfg.PROGRESS_EVERY_N_FILES == 0
                         or i == 1 or i == n_total)):
                cfg.log(f"    Finished CSV {i}/{n_total}")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No CSV files were successfully loaded.")

    # Relative metrics
    day1 = df[df["day"] == 1]
    if day1.empty:
        raise ValueError("No Day 1 data found; cannot compute relative metrics.")
    day1_total_mean = day1["total_h_net"].dropna().mean()
    day1_signal_mean = day1["signal_mean_h_net"].dropna().mean()
    if not np.isfinite(day1_total_mean) or day1_total_mean == 0:
        raise ValueError("Invalid Day 1 mean total_h_net.")
    if not np.isfinite(day1_signal_mean) or day1_signal_mean == 0:
        raise ValueError("Invalid Day 1 mean signal_mean_h_net.")
    df["relative_total_h_net"] = df["total_h_net"] / day1_total_mean
    df["relative_signal_mean_h_net"] = df["signal_mean_h_net"] / day1_signal_mean

    ap_summary = _summarize_ap_bins(ap_bin_by_day)
    ap_bin_stats_df, ap_bin_sig_ranges_df = _test_ap_bins_vs_global_mean(
        df, ap_bin_by_day, alpha=0.05, use_fdr=True)

    top5_prob_maps = {}
    for d in cfg.TARGET_HEATMAP_DAYS:
        total = accumulators["top_total_count"][d]
        top = accumulators["top_count"][d]
        prob = np.full_like(total, np.nan, dtype=float)
        valid = total > 0
        prob[valid] = top[valid] / total[valid]
        top5_prob_maps[d] = prob

    mean_hnet_maps = {}
    for d in cfg.TARGET_HEATMAP_DAYS:
        h_sum = accumulators["mean_h_sum"][d]
        sig_count = accumulators["mean_signal_count"][d]
        mean_map = np.full_like(h_sum, np.nan, dtype=float)
        valid = sig_count > 0
        mean_map[valid] = h_sum[valid] / sig_count[valid]
        mean_hnet_maps[d] = mean_map

    diff_map = mean_hnet_maps[16] - mean_hnet_maps[1]

    return df, ap_summary, top5_prob_maps, mean_hnet_maps, diff_map, ap_bin_stats_df, ap_bin_sig_ranges_df


def _summarize_ap_bins(ap_bin_by_day):
    result = {}
    for day, curves in ap_bin_by_day.items():
        if len(curves) == 0:
            continue
        arr = np.vstack(curves)
        valid = np.isfinite(arr)
        n = valid.sum(axis=0)
        sum_vals = np.nansum(arr, axis=0)
        mean = np.full(cfg.AP_N_BINS, np.nan, dtype=float)
        mean[n > 0] = sum_vals[n > 0] / n[n > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            std = np.nanstd(arr, axis=0, ddof=1)
        ci = np.zeros(cfg.AP_N_BINS, dtype=float)
        valid_ci = n > 1
        ci[valid_ci] = stats.t.ppf(0.975, df=n[valid_ci] - 1) * std[valid_ci] / np.sqrt(n[valid_ci])
        ci[~np.isfinite(ci)] = 0
        result[day] = {"mean": mean, "ci95": ci, "n": n}
    return result


def _test_ap_bins_vs_global_mean(df, ap_bin_by_day, alpha=0.05, use_fdr=True):
    per_bin_rows = []
    for day, curves in ap_bin_by_day.items():
        if len(curves) == 0:
            continue
        ap_arr = np.vstack(curves)
        day_df = df[df["day"] == day].reset_index(drop=True)
        if len(day_df) != ap_arr.shape[0]:
            warnings.warn(f"Day {day}: df rows ({len(day_df)}) and AP curves ({ap_arr.shape[0]}) mismatch.")
        n_use = min(len(day_df), ap_arr.shape[0])
        ap_arr = ap_arr[:n_use, :]
        global_mean = day_df["signal_mean_h_net"].to_numpy(dtype=float)[:n_use]
        for bin_idx in range(cfg.AP_N_BINS):
            bin_values = ap_arr[:, bin_idx]
            valid = np.isfinite(bin_values) & np.isfinite(global_mean)
            diffs = bin_values[valid] - global_mean[valid]
            n_pairs = len(diffs)
            if n_pairs >= 3 and np.any(diffs != 0):
                try:
                    stat, p_value = stats.wilcoxon(diffs, alternative="greater",
                                                   zero_method="wilcox", mode="auto")
                except ValueError:
                    stat, p_value = np.nan, np.nan
            else:
                stat, p_value = np.nan, np.nan
            per_bin_rows.append({
                "day": day, "bin_index": bin_idx,
                "s_start": bin_idx * cfg.AP_BIN_WIDTH,
                "s_end": (bin_idx + 1) * cfg.AP_BIN_WIDTH,
                "s_center": (bin_idx + 0.5) * cfg.AP_BIN_WIDTH,
                "n_pairs": n_pairs,
                "mean_bin_h_net": np.nanmean(bin_values[valid]) if n_pairs > 0 else np.nan,
                "mean_global_signal_h_net": np.nanmean(global_mean[valid]) if n_pairs > 0 else np.nan,
                "mean_difference": np.nanmean(diffs) if n_pairs > 0 else np.nan,
                "median_difference": np.nanmedian(diffs) if n_pairs > 0 else np.nan,
                "wilcoxon_statistic": stat, "p_value": p_value,
            })
    per_bin_df = pd.DataFrame(per_bin_rows)
    if per_bin_df.empty:
        return per_bin_df, pd.DataFrame()
    per_bin_df["q_value"] = np.nan
    for day in sorted(per_bin_df["day"].dropna().unique()):
        idx = per_bin_df["day"] == day
        per_bin_df.loc[idx, "q_value"] = cfg.adjust_pvalues(per_bin_df.loc[idx, "p_value"].to_numpy(), method="bh")
    per_bin_df["significant"] = (per_bin_df["q_value"] < alpha) if use_fdr else (per_bin_df["p_value"] < alpha)
    range_df = _collapse_significant_ap_bins(per_bin_df)
    return per_bin_df, range_df


def _collapse_significant_ap_bins(per_bin_df):
    range_rows = []
    for day, sub in per_bin_df.sort_values(["day", "bin_index"]).groupby("day"):
        sub = sub.reset_index(drop=True)
        sig = sub["significant"].fillna(False).to_numpy(dtype=bool)
        start = None
        for i in range(len(sig) + 1):
            cur = sig[i] if i < len(sig) else False
            if cur and start is None:
                start = i
            if (not cur) and (start is not None):
                end = i - 1
                block = sub.iloc[start:end + 1]
                range_rows.append({
                    "day": day, "s_start": float(block["s_start"].iloc[0]),
                    "s_end": float(block["s_end"].iloc[-1]), "n_bins": int(len(block)),
                    "min_p_value": float(np.nanmin(block["p_value"])),
                    "min_q_value": float(np.nanmin(block["q_value"])),
                    "mean_difference": float(np.nanmean(block["mean_difference"])),
                    "median_difference": float(np.nanmedian(block["median_difference"])),
                    "mean_n_pairs": float(np.nanmean(block["n_pairs"])),
                })
                start = None
    return pd.DataFrame(range_rows)


def welch_ttest_vs_day1(df, value_col, day_col="day", baseline_day=1):
    results = []
    days_all = sorted(df[day_col].dropna().unique())
    values_by_day, variance_by_day = {}, {}
    for day in days_all:
        values = df.loc[df[day_col] == day, value_col].dropna().to_numpy(dtype=float)
        values_by_day[day] = values
        variance_by_day[day] = float(np.var(values, ddof=1)) if len(values) >= 2 else np.nan
    baseline = values_by_day.get(baseline_day, np.array([], dtype=float))
    day1_variance = variance_by_day.get(baseline_day, np.nan)
    valid4global = [values_by_day[d] for d in days_all if len(values_by_day[d]) >= 2]
    global_var_stat = np.nan; global_var_p = np.nan
    if len(valid4global) >= 2:
        try:
            global_var_stat, global_var_p = stats.levene(*valid4global, center="median")
        except ValueError:
            pass
    valid_var_items = sorted([(d, v) for d, v in variance_by_day.items() if np.isfinite(v)],
                             key=lambda x: x[1], reverse=True)
    variance_order_desc = " > ".join([f"Day {d} ({v:.6g})" for d, v in valid_var_items])
    variance_rank_desc = {d: r + 1 for r, (d, v) in enumerate(valid_var_items)}
    for day in days_all:
        if day == baseline_day:
            continue
        current = values_by_day[day]
        t_stat = np.nan; p_value = np.nan
        if len(baseline) >= 2 and len(current) >= 2:
            t_stat, p_value = stats.ttest_ind(current, baseline, equal_var=False, nan_policy="omit")
        var_stat = np.nan; var_p_value = np.nan
        if len(baseline) >= 2 and len(current) >= 2:
            try:
                var_stat, var_p_value = stats.levene(current, baseline, center="median")
            except ValueError:
                pass
        cv = variance_by_day.get(day, np.nan)
        if np.isfinite(cv) and np.isfinite(day1_variance) and day1_variance != 0:
            ratio = cv / day1_variance
            larger = f"Day {day}" if cv > day1_variance else (f"Day {baseline_day}" if cv < day1_variance else "equal")
        else:
            ratio = np.nan; larger = "NA"
        results.append({
            "metric": value_col, "day": day, "n_day1": len(baseline), "n_current": len(current),
            "t_statistic": t_stat, "p_value": p_value, "stars": cfg.p_to_stars(p_value),
            "day1_variance": day1_variance, "current_day_variance": cv,
            "variance_ratio_current_vs_day1": ratio, "larger_variance_day_vs_day1": larger,
            "variance_test_method": "Brown-Forsythe / Levene center=median",
            "variance_statistic_vs_day1": var_stat, "variance_p_value_vs_day1": var_p_value,
            "variance_stars_vs_day1": cfg.variance_p_to_stars(var_p_value),
            "variance_significant_vs_day1_p_lt_0p05": bool(var_p_value < 0.05) if np.isfinite(var_p_value) else False,
            "global_variance_statistic_all_days": global_var_stat,
            "global_variance_p_value_all_days": global_var_p,
            "global_variance_stars_all_days": cfg.variance_p_to_stars(global_var_p),
            "current_day_variance_rank_desc": variance_rank_desc.get(day, np.nan),
            "all_days_variance_order_desc": variance_order_desc,
        })
    out = pd.DataFrame(results)
    if not out.empty:
        out["p_value_holm_within_metric"] = cfg.adjust_pvalues(out["p_value"].to_numpy())
        out["stars_holm_within_metric"] = out["p_value_holm_within_metric"].apply(cfg.p_to_stars)
        out["significant_holm_within_metric_p_lt_0p05"] = out["p_value_holm_within_metric"] < 0.05
        out["variance_p_value_holm_within_metric"] = cfg.adjust_pvalues(out["variance_p_value_vs_day1"].to_numpy())
        out["variance_stars_holm_within_metric"] = out["variance_p_value_holm_within_metric"].apply(cfg.variance_p_to_stars)
        out["variance_significant_holm_within_metric_p_lt_0p05"] = out["variance_p_value_holm_within_metric"] < 0.05
        out = out.sort_values(by=["metric", "current_day_variance"], ascending=[True, False]).reset_index(drop=True)
    return out


# =========================
# Main
# =========================

def main():
    cfg.ensure_dirs()
    manifest_path = cfg.COMPUTED_DATA_DIR / "computed_data_manifest.txt"

    if manifest_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log("[Module 1] Computed data already exists; skipping.")
        return

    cfg.log("[Module 1] Loading all CSV data and computing statistics.")
    df, ap_summary, top5_prob_maps, mean_hnet_maps, diff_map, ap_bin_stats_df, ap_bin_sig_ranges_df = load_all_data()

    p_values = {
        "relative_total_h_net": welch_ttest_vs_day1(df, "relative_total_h_net"),
        "relative_signal_mean_h_net": welch_ttest_vs_day1(df, "relative_signal_mean_h_net"),
        "signal_ratio": welch_ttest_vs_day1(df, "signal_ratio"),
        "top5_contribution": welch_ttest_vs_day1(df, "top5_contribution"),
    }
    p_all = pd.concat(p_values.values(), ignore_index=True)
    p_all["p_value_holm_across_ABCH"] = cfg.adjust_pvalues(p_all["p_value"].to_numpy())
    p_all["stars_holm_across_ABCH"] = p_all["p_value_holm_across_ABCH"].apply(cfg.p_to_stars)
    p_all["significant_holm_across_ABCH_p_lt_0p05"] = p_all["p_value_holm_across_ABCH"] < 0.05
    p_all["variance_p_value_holm_across_ABCH"] = cfg.adjust_pvalues(p_all["variance_p_value_vs_day1"].to_numpy())
    p_all["variance_stars_holm_across_ABCH"] = p_all["variance_p_value_holm_across_ABCH"].apply(cfg.variance_p_to_stars)
    p_all["variance_significant_holm_across_ABCH_p_lt_0p05"] = p_all["variance_p_value_holm_across_ABCH"] < 0.05

    cfg.save_compressed(df, cfg.COMPUTED_DATA_DIR / "instance_summary", "instance summary")
    cfg.save_compressed(p_all, cfg.COMPUTED_DATA_DIR / "welch_ttest", "Welch t-test vs Day 1")
    cfg.save_compressed(ap_bin_stats_df, cfg.COMPUTED_DATA_DIR / "ap_bin_stats", "AP bin stats")
    cfg.save_compressed(ap_bin_sig_ranges_df, cfg.COMPUTED_DATA_DIR / "ap_bin_sig_ranges", "AP bin sig ranges")

    import gzip as gzip_module
    gzip_path = cfg.COMPUTED_DATA_DIR / "heatmap_data.pkl.gz"
    with gzip_module.open(gzip_path, "wb") as f:
        pickle.dump({
            "ap_summary": ap_summary,
            "top5_prob_maps": top5_prob_maps,
            "mean_hnet_maps": mean_hnet_maps,
            "diff_map": diff_map,
        }, f)
    cfg.log(f"    Saved heatmap matrices: {gzip_path}")

    manifest_path.write_text("done\n", encoding="utf-8")
    cfg.log(f"[Module 1] Instances: {len(df)}")
    cfg.log(f"[Module 1] Output: {cfg.COMPUTED_DATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 1] Stopped gracefully: {exc}")
        cfg.log("[Module 1] Completed outputs are kept. Clear the stop request before resuming.")
