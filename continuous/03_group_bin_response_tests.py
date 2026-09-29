# -*- coding: utf-8 -*-
"""
Module 3: group-level response tests for each gene x condition x bin x time.

Inputs:
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    and per-group delta chunks listed in that manifest.

Outputs:
    OUTPUT_DIR/03_group_tests/group_bin_response_tests.pkl.gz
    OUTPUT_DIR/03_group_tests/group_bin_response_tests.csv
    OUTPUT_DIR/03_group_tests/significant_bin_timepoints_initial.pkl.gz
    OUTPUT_DIR/03_group_tests/significant_bin_timepoints_initial.csv

Depends on:
    02_compute_delta_values.py output.
"""

import importlib.util
import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
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


GROUP_ID_COLS = ["gene_id", "gene_label", "condition", "condition_label"]
BIN_COLS = ["s_bin", "r_bin"]
GEOMETRY_COLS = ["s_start", "s_end", "r_start", "r_end", "s_center", "r_center"]


def one_sample_greater_test(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n < cfg.MIN_N_FOR_GROUP_TEST:
        return np.nan, np.nan, False

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        if mean > 0:
            return np.inf, 0.0, True
        return 0.0, 1.0, True

    t_stat = mean / (sd / math.sqrt(n))
    if t_dist is not None:
        p_raw = float(t_dist.sf(t_stat, df=n - 1))
    else:
        # Fallback normal approximation if scipy is unavailable.
        p_raw = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
    p_raw = min(max(p_raw, 0.0), 1.0)
    return float(t_stat), p_raw, False


def summarize_one_bin_time(key, g: pd.DataFrame):
    gene_id, gene_label, condition, condition_label, s_bin, r_bin, time_min = key

    inst = (
        g.groupby("instance_uid", as_index=False)
        .agg(deltaF=("deltaF", "mean"), dff0=("dff0", "mean"))
        .dropna(subset=["deltaF"])
    )
    delta = inst["deltaF"].to_numpy(dtype=float)
    dff0 = pd.to_numeric(inst["dff0"], errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(delta))

    mean_delta = float(np.mean(delta)) if n else np.nan
    median_delta = float(np.median(delta)) if n else np.nan
    sd_delta = float(np.std(delta, ddof=1)) if n > 1 else np.nan
    sem_delta = sd_delta / math.sqrt(n) if n > 1 and np.isfinite(sd_delta) else np.nan
    n_positive = int(np.sum(delta > 0)) if n else 0
    positive_ratio = n_positive / n if n else np.nan
    mean_dff0 = float(np.mean(dff0)) if len(dff0) else np.nan
    median_dff0 = float(np.median(dff0)) if len(dff0) else np.nan

    all_rows_are_baseline = False
    if "baseline_time_min" in g.columns and n:
        all_rows_are_baseline = bool(
            np.allclose(
                pd.to_numeric(g["time_min"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(g["baseline_time_min"], errors="coerce").to_numpy(dtype=float),
                equal_nan=False,
            )
        )

    if all_rows_are_baseline:
        t_stat, p_raw, zero_variance = 0.0, 1.0, True
    else:
        t_stat, p_raw, zero_variance = one_sample_greater_test(delta)

    first = g.iloc[0]
    return {
        "gene_id": gene_id,
        "gene_label": gene_label,
        "condition": condition,
        "condition_label": condition_label,
        "s_bin": int(s_bin),
        "r_bin": int(r_bin),
        "s_start": first.get("s_start", np.nan),
        "s_end": first.get("s_end", np.nan),
        "r_start": first.get("r_start", np.nan),
        "r_end": first.get("r_end", np.nan),
        "s_center": first.get("s_center", np.nan),
        "r_center": first.get("r_center", np.nan),
        "time_min": time_min,
        "n_instances": n,
        "mean_deltaF": mean_delta,
        "median_deltaF": median_delta,
        "sd_deltaF": sd_delta,
        "sem_deltaF": sem_delta,
        "mean_dff0": mean_dff0,
        "median_dff0": median_dff0,
        "n_positive_deltaF": n_positive,
        "positive_ratio": positive_ratio,
        "t_stat": t_stat,
        "p_raw": p_raw,
        "p_adj": np.nan,
        "significant": False,
        "zero_variance": bool(zero_variance),
        "baseline_only_timepoint": bool(all_rows_are_baseline),
    }


def apply_p_adjustment(test_df: pd.DataFrame):
    if test_df.empty:
        test_df["p_adj"] = []
        test_df["significant"] = []
        return test_df

    scope = cfg.P_ADJUST_SCOPE.lower()
    test_df = test_df.copy()
    if scope == "within_bin":
        adjusted_parts = []
        for _, g in test_df.groupby(["gene_id", "condition", "s_bin", "r_bin"], sort=False):
            g = g.copy()
            g["p_adj"] = cfg.adjust_p_values(g["p_raw"].to_numpy(dtype=float))
            adjusted_parts.append(g)
        test_df = pd.concat(adjusted_parts, ignore_index=True)
    elif scope == "all_bin_time":
        adjusted_parts = []
        for _, g in test_df.groupby(["gene_id", "condition"], sort=False):
            g = g.copy()
            g["p_adj"] = cfg.adjust_p_values(g["p_raw"].to_numpy(dtype=float))
            adjusted_parts.append(g)
        test_df = pd.concat(adjusted_parts, ignore_index=True)
    else:
        raise ValueError(f"Unsupported P_ADJUST_SCOPE: {cfg.P_ADJUST_SCOPE}")

    test_df["significant"] = (
        pd.to_numeric(test_df["p_adj"], errors="coerce") < cfg.ALPHA
    ) & (~test_df["baseline_only_timepoint"].astype(bool))
    return test_df


def _sort_test_df(test_df: pd.DataFrame):
    if test_df.empty:
        return test_df
    sort_cols = [c for c in ["gene_id", "condition", "r_bin", "s_bin", "time_min"] if c in test_df.columns]
    return test_df.sort_values(sort_cols) if sort_cols else test_df



def _prepare_group_df_for_tests(group_df: pd.DataFrame):
    needed_cols = GROUP_ID_COLS + BIN_COLS + GEOMETRY_COLS + [
        "instance_uid",
        "time_min",
        "baseline_time_min",
        "deltaF",
        "dff0",
    ]
    cols = [c for c in needed_cols if c in group_df.columns]
    work = group_df.loc[:, cols].copy()
    work["deltaF"] = pd.to_numeric(work["deltaF"], errors="coerce")
    work["dff0"] = pd.to_numeric(work["dff0"], errors="coerce")
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")
    work["s_bin"] = pd.to_numeric(work["s_bin"], errors="coerce")
    work["r_bin"] = pd.to_numeric(work["r_bin"], errors="coerce")
    if "baseline_time_min" in work.columns:
        work["baseline_time_min"] = pd.to_numeric(work["baseline_time_min"], errors="coerce")
    work, _ = cfg.filter_late_baseline_bins(work)
    work = work.dropna(subset=["time_min", "s_bin", "r_bin"])
    work["s_bin"] = work["s_bin"].astype(np.int32)
    work["r_bin"] = work["r_bin"].astype(np.int32)
    return work



def process_gene_condition_tests(group_df: pd.DataFrame, adjust_p: bool = True):
    if group_df.empty:
        return pd.DataFrame()

    work = _prepare_group_df_for_tests(group_df)
    if work.empty:
        return pd.DataFrame()

    records = []
    group_cols = GROUP_ID_COLS + BIN_COLS + ["time_min"]
    for key, g in work.groupby(group_cols, sort=True):
        records.append(summarize_one_bin_time(key, g))

    test_df = pd.DataFrame(records)
    if adjust_p:
        test_df = apply_p_adjustment(test_df)
    return _sort_test_df(test_df)



def process_bin_chunk_tests(chunk_df: pd.DataFrame):
    return process_gene_condition_tests(chunk_df, adjust_p=False)



def _bin_key_chunks(work: pd.DataFrame, chunk_size: int):
    unique_keys = np.sort(work["_bin_key"].dropna().unique())
    return [unique_keys[i : i + chunk_size] for i in range(0, len(unique_keys), chunk_size)]



def _run_one_gene_condition_tests(group_df: pd.DataFrame):
    if group_df.empty:
        return pd.DataFrame()

    label = f"{group_df['gene_id'].iloc[0]} | {group_df['condition'].iloc[0]}"
    work = _prepare_group_df_for_tests(group_df)
    if work.empty:
        return pd.DataFrame()

    work["_bin_key"] = work["r_bin"].astype(np.int64) * int(cfg.N_S_BINS) + work["s_bin"].astype(np.int64)
    chunk_size = int(getattr(cfg, "GROUP_TEST_BIN_CHUNK_SIZE", 1000))
    key_chunks = _bin_key_chunks(work, chunk_size)
    cfg.log(f"        Testing {label}: {len(key_chunks)} bin chunks, chunk_size={chunk_size}")

    if not (cfg.USE_MULTIPROCESSING and cfg.GROUP_TEST_WORKERS > 1 and len(key_chunks) > 1):
        cfg.log("        Bin-chunk parallelism disabled; processing sequentially.")
        raw_parts = []
        for chunk_i, keys in enumerate(key_chunks, start=1):
            cfg.raise_if_stop_requested(f"module 3 {label} before bin chunk {chunk_i}/{len(key_chunks)}")
            chunk_df = work.loc[work["_bin_key"].isin(keys)].drop(columns=["_bin_key"])
            raw_parts.append(process_bin_chunk_tests(chunk_df))
            if chunk_i == 1 or chunk_i == len(key_chunks) or chunk_i % 20 == 0:
                cfg.log(f"            Finished bin chunk {chunk_i}/{len(key_chunks)}")
    else:
        workers = min(int(cfg.GROUP_TEST_WORKERS), len(key_chunks))
        max_pending = min(int(getattr(cfg, "GROUP_TEST_MAX_PENDING_CHUNKS", workers * 2)), len(key_chunks))
        cfg.log(f"        Parallel bin-chunk tests enabled: workers={workers}, max_pending={max_pending}")
        raw_parts = []
        next_chunk = 0
        finished = 0
        pending = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while next_chunk < len(key_chunks) and len(pending) < max_pending:
                cfg.raise_if_stop_requested(f"module 3 {label} before submitting bin chunk {next_chunk + 1}/{len(key_chunks)}")
                keys = key_chunks[next_chunk]
                chunk_df = work.loc[work["_bin_key"].isin(keys)].drop(columns=["_bin_key"]).copy()
                future = executor.submit(process_bin_chunk_tests, chunk_df)
                pending[future] = next_chunk + 1
                next_chunk += 1

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    chunk_i = pending.pop(future)
                    raw_parts.append(future.result())
                    finished += 1
                    stop_requested = cfg.should_stop_requested()
                    if finished == 1 or finished == len(key_chunks) or finished % 20 == 0:
                        cfg.log(f"            Finished bin chunk {finished}/{len(key_chunks)}")

                    if stop_requested and not pending:
                        cfg.raise_if_stop_requested(f"module 3 {label} after bin chunk {finished}/{len(key_chunks)}")
                    if (not stop_requested) and next_chunk < len(key_chunks):
                        cfg.raise_if_stop_requested(f"module 3 {label} before submitting bin chunk {next_chunk + 1}/{len(key_chunks)}")
                        keys = key_chunks[next_chunk]
                        chunk_df = work.loc[work["_bin_key"].isin(keys)].drop(columns=["_bin_key"]).copy()
                        new_future = executor.submit(process_bin_chunk_tests, chunk_df)
                        pending[new_future] = next_chunk + 1
                        next_chunk += 1

    cfg.raise_if_stop_requested(f"module 3 {label} before p-value adjustment")
    if not raw_parts:
        return pd.DataFrame()
    raw_df = pd.concat(raw_parts, ignore_index=True)
    adjusted_df = apply_p_adjustment(raw_df)
    return _sort_test_df(adjusted_df)



def run_tests(delta_df: pd.DataFrame):
    if delta_df.empty:
        return pd.DataFrame()

    parts = []
    grouped = delta_df.groupby(["gene_id", "condition"], sort=False)
    for i, (_, group_df) in enumerate(grouped, start=1):
        cfg.log(f"        Gene-condition block {i}/{grouped.ngroups}")
        parts.append(_run_one_gene_condition_tests(group_df))

    if not parts:
        return pd.DataFrame()
    return _sort_test_df(pd.concat(parts, ignore_index=True))



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


def run_tests_from_manifest(delta_manifest: pd.DataFrame):
    tests_parts = []
    sig_parts = []
    for i, row in delta_manifest.iterrows():
        cfg.raise_if_stop_requested(f"module 3 before delta chunk {i + 1}/{len(delta_manifest)}")
        delta_path = Path(row["delta_values_path"])
        group_name = str(row.get("group_name", f"group_{i + 1}"))
        group_output_dir = cfg.GROUP_TESTS_DIR / cfg.safe_filename(group_name)
        group_output_dir.mkdir(parents=True, exist_ok=True)
        group_tests_base = group_output_dir / "group_bin_response_tests"
        group_sig_base = group_output_dir / "significant_bin_timepoints_initial"
        group_tests_path = cfg.compressed_path(group_tests_base)
        group_sig_path = cfg.compressed_path(group_sig_base)

        cfg.log("=" * 80)
        cfg.log(f"[Module 3 chunk {i + 1}/{len(delta_manifest)}] {group_name}")

        if group_tests_path.exists() and group_sig_path.exists() and not cfg.OVERWRITE_EXISTING:
            cfg.log(f"    Existing group test output found; skipping {group_name}")
            tests_df = cfg.read_dataframe_compressed(group_tests_path)
            sig_df = cfg.read_dataframe_compressed(group_sig_path)
            tests_parts.append(tests_df)
            sig_parts.append(sig_df)
            continue

        cfg.log(f"    Reading delta chunk: {delta_path}")
        delta_df = cfg.read_dataframe_compressed(delta_path)
        tests_df = run_tests(delta_df)

        cfg.save_dataframe_compressed(tests_df, group_tests_base, f"{group_name} tests")
        tests_df.to_csv(group_output_dir / "group_bin_response_tests.csv", index=False, encoding="utf-8-sig")

        sig_df = tests_df.loc[tests_df["significant"].astype(bool)].copy() if not tests_df.empty else pd.DataFrame()
        if not sig_df.empty:
            sig_df = sig_df.sort_values(["gene_id", "condition", "p_adj", "time_min", "r_bin", "s_bin"])
        cfg.save_dataframe_compressed(
            sig_df,
            group_sig_base,
            f"{group_name} initial significant events",
        )
        sig_df.to_csv(
            group_output_dir / "significant_bin_timepoints_initial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        tests_parts.append(tests_df)
        sig_parts.append(sig_df)

    tests_all = pd.concat(tests_parts, ignore_index=True) if tests_parts else pd.DataFrame()
    sig_all = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    if not tests_all.empty:
        tests_all = tests_all.sort_values(["gene_id", "condition", "r_bin", "s_bin", "time_min"])
    if not sig_all.empty:
        sig_all = sig_all.sort_values(["gene_id", "condition", "p_adj", "time_min", "r_bin", "s_bin"])
    return tests_all, sig_all



def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="03_group_bin_response_tests")

    tests_base = cfg.GROUP_TESTS_DIR / "group_bin_response_tests"
    tests_path = cfg.compressed_path(tests_base)
    if tests_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 3 output already exists; skipping: {tests_path}")
        return

    cfg.log("[Module 3] Reading Module 2 delta manifest.")
    delta_manifest = load_delta_manifest()
    cfg.log(f"    Delta chunks: {len(delta_manifest)}")

    cfg.log("[Module 3] Running one-tailed one-sample tests on instance-level deltaF.")
    tests_df, sig_df = run_tests_from_manifest(delta_manifest)

    cfg.save_dataframe_compressed(tests_df, tests_base, "group bin response tests")
    tests_df.to_csv(cfg.GROUP_TESTS_DIR / "group_bin_response_tests.csv", index=False, encoding="utf-8-sig")

    sig_base = cfg.GROUP_TESTS_DIR / "significant_bin_timepoints_initial"
    cfg.save_dataframe_compressed(sig_df, sig_base, "initial significant bin timepoints")
    sig_df.to_csv(
        cfg.GROUP_TESTS_DIR / "significant_bin_timepoints_initial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cfg.log("[Module 3] Summary:")
    cfg.log(f"    Tested bin-time rows: {len(tests_df)}")
    cfg.log(f"    Initial significant bin-time rows: {len(sig_df)}")
    cfg.log(f"    Output: {cfg.GROUP_TESTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 3] Stopped gracefully: {exc}")
        cfg.log("[Module 3] Completed group outputs are kept. Clear the stop request before resuming.")




