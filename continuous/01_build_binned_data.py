# -*- coding: utf-8 -*-
"""
Module 1: read raw pixel CSV files, create unique IDs, and bin pixels in s-r space.

Inputs:
    Raw CSV folders from 00_config_utils.INPUT_GROUPS.
    File pattern from 00_config_utils.CSV_PATTERN.

Outputs:
    OUTPUT_DIR/01_binned_data/<gene__condition>/recognized_csv_files.csv
    OUTPUT_DIR/01_binned_data/<gene__condition>/skipped_files.csv
    OUTPUT_DIR/01_binned_data/<gene__condition>/csv_binning_errors.csv
    OUTPUT_DIR/01_binned_data/<gene__condition>/per_instance_bin_stats.pkl.gz
    OUTPUT_DIR/01_binned_data/binned_data_manifest.csv
    OUTPUT_DIR/01_binned_data/binned_data_manifest.pkl.gz

Depends on:
    00_config_utils.py only.
"""

import importlib.util
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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


EMPTY_BIN_COLUMNS = [
    "s_bin",
    "r_bin",
    "s_start",
    "s_end",
    "r_start",
    "r_end",
    "s_center",
    "r_center",
    "bin_mean_h",
    "bin_var_h",
    "n_pixels",
    "bin_sum_h",
    "bin_signal_sum_h",
    "n_signal_pixels",
]


def get_r_column(df: pd.DataFrame, csv_name: str):
    if cfg.R_COL in df.columns:
        return cfg.R_COL
    if cfg.AUTO_FALLBACK_TO_R3D and cfg.R3D_COL in df.columns:
        warnings.warn(f"{csv_name}: column '{cfg.R_COL}' not found; using '{cfg.R3D_COL}' instead.")
        return cfg.R3D_COL
    raise ValueError(f"{csv_name} missing required column: {cfg.R_COL}")


def compute_binned_stats_for_csv(csv_path: Path):
    """
    Read one pixel CSV and compute per-CSV s-r bin statistics using numpy.bincount.
    Pixels are used only for spatial aggregation.
    """
    wanted_cols = {cfg.S_COL, cfg.R_COL, cfg.R3D_COL, cfg.H_COL, cfg.IS_SIGNAL_COL}
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted_cols)

    r_col_actual = get_r_column(df, csv_path.name)
    missing_cols = [c for c in [cfg.S_COL, r_col_actual, cfg.H_COL, cfg.IS_SIGNAL_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path.name} missing required columns: {missing_cols}")

    s = pd.to_numeric(df[cfg.S_COL], errors="coerce").to_numpy(dtype=float, copy=False)
    r = pd.to_numeric(df[r_col_actual], errors="coerce").to_numpy(dtype=float, copy=False)
    h = pd.to_numeric(df[cfg.H_COL], errors="coerce").to_numpy(dtype=float, copy=False)
    is_signal = (
        pd.to_numeric(df[cfg.IS_SIGNAL_COL], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8, copy=False)
    )

    valid = np.isfinite(s) & np.isfinite(r) & np.isfinite(h)
    if cfg.FILTER_TO_SR_RANGE:
        valid &= (
            (s >= cfg.S_RANGE[0])
            & (s <= cfg.S_RANGE[1])
            & (r >= cfg.R_RANGE[0])
            & (r <= cfg.R_RANGE[1])
        )
    if not np.any(valid):
        return pd.DataFrame(columns=EMPTY_BIN_COLUMNS)

    s = s[valid]
    r = r[valid]
    h = h[valid]
    is_signal = is_signal[valid]

    if cfg.CLIP_NEGATIVE_H_TO_ZERO:
        h = np.clip(h, a_min=0.0, a_max=None)

    s_idx = cfg.assign_bin_indices(s, cfg.S_RANGE, cfg.N_S_BINS, cfg.S_BIN_SIZE)
    r_idx = cfg.assign_bin_indices(r, cfg.R_RANGE, cfg.N_R_BINS, cfg.R_BIN_SIZE)
    in_range = (s_idx >= 0) & (s_idx < cfg.N_S_BINS) & (r_idx >= 0) & (r_idx < cfg.N_R_BINS)
    if not np.any(in_range):
        return pd.DataFrame(columns=EMPTY_BIN_COLUMNS)

    s_idx = s_idx[in_range].astype(np.int64, copy=False)
    r_idx = r_idx[in_range].astype(np.int64, copy=False)
    h = h[in_range].astype(float, copy=False)
    is_signal = is_signal[in_range]

    flat_idx = r_idx * cfg.N_S_BINS + s_idx
    n_total_bins = cfg.N_S_BINS * cfg.N_R_BINS

    count = np.bincount(flat_idx, minlength=n_total_bins)
    nonzero = count > 0
    if not np.any(nonzero):
        return pd.DataFrame(columns=EMPTY_BIN_COLUMNS)

    sum_h = np.bincount(flat_idx, weights=h, minlength=n_total_bins)
    sum_h2 = np.bincount(flat_idx, weights=h * h, minlength=n_total_bins)
    signal_mask = is_signal == 1
    if np.any(signal_mask):
        signal_sum_h = np.bincount(
            flat_idx,
            weights=np.where(signal_mask, h, 0.0),
            minlength=n_total_bins,
        )
        signal_count = np.bincount(
            flat_idx,
            weights=signal_mask.astype(np.int8),
            minlength=n_total_bins,
        )
    else:
        signal_sum_h = np.zeros(n_total_bins, dtype=float)
        signal_count = np.zeros(n_total_bins, dtype=float)

    flat_nonzero = np.flatnonzero(nonzero)
    counts = count[flat_nonzero].astype(np.int64, copy=False)
    sums = sum_h[flat_nonzero]
    sums2 = sum_h2[flat_nonzero]
    mean_h = sums / counts

    var_h = np.zeros_like(mean_h, dtype=float)
    more_than_one = counts > 1
    var_h[more_than_one] = (
        sums2[more_than_one] - (sums[more_than_one] ** 2) / counts[more_than_one]
    ) / (counts[more_than_one] - 1)
    var_h = np.maximum(var_h, 0.0)

    r_bins = flat_nonzero // cfg.N_S_BINS
    s_bins = flat_nonzero % cfg.N_S_BINS

    binned = pd.DataFrame(
        {
            "s_bin": s_bins.astype(np.int32),
            "r_bin": r_bins.astype(np.int32),
            "bin_mean_h": mean_h.astype(float),
            "bin_var_h": var_h.astype(float),
            "n_pixels": counts.astype(np.int32),
            "bin_sum_h": sums.astype(float),
            "bin_signal_sum_h": signal_sum_h[flat_nonzero].astype(float),
            "n_signal_pixels": signal_count[flat_nonzero].astype(np.int32),
        }
    )
    binned["s_start"] = cfg.S_RANGE[0] + binned["s_bin"] * cfg.S_BIN_SIZE
    binned["s_end"] = cfg.S_RANGE[0] + (binned["s_bin"] + 1) * cfg.S_BIN_SIZE
    binned["r_start"] = cfg.R_RANGE[0] + binned["r_bin"] * cfg.R_BIN_SIZE
    binned["r_end"] = cfg.R_RANGE[0] + (binned["r_bin"] + 1) * cfg.R_BIN_SIZE
    binned["s_center"] = binned["s_start"] + cfg.S_BIN_SIZE / 2.0
    binned["r_center"] = binned["r_start"] + cfg.R_BIN_SIZE / 2.0

    return binned[EMPTY_BIN_COLUMNS]


def process_one_csv_for_binning(task: dict):
    group = task["group"]
    csv_path = Path(task["csv_path"])
    file_i = int(task["file_i"])
    folder = str(task["folder"])

    try:
        parsed = cfg.parse_filename(csv_path.name)
        if parsed is None:
            return {
                "status": "skipped",
                "recognized": None,
                "skipped": {
                    "gene_id": group["gene_id"],
                    "condition": group["condition"],
                    "file": csv_path.name,
                    "folder": folder,
                    "reason": "filename_pattern_mismatch",
                },
                "error": None,
                "traceback": None,
                "binned": None,
            }

        instance_raw, time_min = parsed
        instance_uid = f"{group['gene_id']}|{group['condition']}|instance-{instance_raw}"
        csv_uid = f"{group['gene_id']}|{group['condition']}|instance-{instance_raw}|t-{time_min:03d}"

        recognized = {
            "gene_id": group["gene_id"],
            "gene_label": group["gene_label"],
            "condition": group["condition"],
            "condition_label": group["condition_label"],
            "instance": instance_raw,
            "instance_uid": instance_uid,
            "time_min": time_min,
            "csv_uid": csv_uid,
            "file": csv_path.name,
            "folder": folder,
            "full_path": str(csv_path),
            "file_order_within_group": file_i,
        }

        binned = compute_binned_stats_for_csv(csv_path)
        if not binned.empty:
            binned.insert(0, "gene_id", group["gene_id"])
            binned.insert(1, "gene_label", group["gene_label"])
            binned.insert(2, "condition", group["condition"])
            binned.insert(3, "condition_label", group["condition_label"])
            binned.insert(4, "instance", instance_raw)
            binned.insert(5, "instance_uid", instance_uid)
            binned.insert(6, "time_min", time_min)
            binned.insert(7, "csv_uid", csv_uid)
            binned.insert(8, "file", csv_path.name)

        return {
            "status": "ok",
            "recognized": recognized,
            "skipped": None,
            "error": None,
            "traceback": None,
            "binned": binned,
        }
    except Exception as exc:
        return {
            "status": "error",
            "recognized": None,
            "skipped": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "file": csv_path.name,
            "folder": folder,
            "binned": None,
        }


def _empty_table(columns):
    return pd.DataFrame(columns=columns)


def collect_group_binned_data(group: dict, group_output_dir: Path):
    folder = Path(group["folder"])
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    files = sorted(folder.rglob(cfg.CSV_PATTERN) if cfg.RECURSIVE_SEARCH else folder.glob(cfg.CSV_PATTERN))
    group_label = f"{group['gene_label']} | {group['condition_label']}"
    cfg.log(f"    Folder: {folder}")
    cfg.log(f"    Matched files: {len(files)}")

    recognized_records = []
    skipped_records = []
    error_records = []
    binned_tables = []

    tasks = [
        {
            "group": group,
            "csv_path": str(csv_path),
            "file_i": i,
            "folder": str(folder),
        }
        for i, csv_path in enumerate(files, start=1)
    ]

    if cfg.USE_MULTIPROCESSING and len(tasks) > 1 and cfg.CSV_READ_WORKERS > 1:
        workers = min(cfg.CSV_READ_WORKERS, len(tasks))
        cfg.log(f"    Parallel CSV binning enabled: workers={workers}")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(process_one_csv_for_binning, task): task for task in tasks}
            for done_i, future in enumerate(as_completed(future_to_task), start=1):
                cfg.raise_if_stop_requested(f"module 1 CSV binning after file {done_i}/{len(tasks)}")
                task = future_to_task[future]
                if (
                    done_i == 1
                    or done_i == len(tasks)
                    or (cfg.PROGRESS_EVERY_N_FILES and done_i % cfg.PROGRESS_EVERY_N_FILES == 0)
                ):
                    cfg.log(f"        Finished file {done_i}/{len(tasks)}: {Path(task['csv_path']).name}")
                result = future.result()
                _collect_worker_result(result, recognized_records, skipped_records, error_records, binned_tables)
    else:
        cfg.log("    Parallel CSV binning disabled; processing sequentially.")
        for i, task in enumerate(tasks, start=1):
            cfg.raise_if_stop_requested(f"module 1 before CSV file {i}/{len(tasks)}")
            if (
                i == 1
                or i == len(tasks)
                or (cfg.PROGRESS_EVERY_N_FILES and i % cfg.PROGRESS_EVERY_N_FILES == 0)
            ):
                cfg.log(f"        Processing file {i}/{len(tasks)}: {Path(task['csv_path']).name}")
            result = process_one_csv_for_binning(task)
            _collect_worker_result(result, recognized_records, skipped_records, error_records, binned_tables)

    recognized_df = pd.DataFrame(recognized_records)
    if not recognized_df.empty:
        recognized_df = recognized_df.sort_values("file_order_within_group")
    else:
        recognized_df = _empty_table(
            [
                "gene_id",
                "gene_label",
                "condition",
                "condition_label",
                "instance",
                "instance_uid",
                "time_min",
                "csv_uid",
                "file",
                "folder",
                "full_path",
                "file_order_within_group",
            ]
        )

    skipped_df = pd.DataFrame(skipped_records)
    if skipped_df.empty:
        skipped_df = _empty_table(["gene_id", "condition", "file", "folder", "reason"])

    error_df = pd.DataFrame(error_records)
    if error_df.empty:
        error_df = _empty_table(["file", "folder", "error", "traceback"])

    if binned_tables:
        per_instance_bin_df = pd.concat(binned_tables, ignore_index=True)
        per_instance_bin_df = per_instance_bin_df.sort_values(["time_min", "instance_uid", "r_bin", "s_bin"])
    else:
        per_instance_bin_df = pd.DataFrame(
            columns=[
                "gene_id",
                "gene_label",
                "condition",
                "condition_label",
                "instance",
                "instance_uid",
                "time_min",
                "csv_uid",
                "file",
            ]
            + EMPTY_BIN_COLUMNS
        )

    recognized_df.to_csv(group_output_dir / "recognized_csv_files.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(group_output_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(group_output_dir / "csv_binning_errors.csv", index=False, encoding="utf-8-sig")

    if cfg.STRICT_FILENAME_CHECK and (len(skipped_df) > 0 or len(error_df) > 0):
        first_problem = "skipped file" if len(skipped_df) > 0 else str(error_df.iloc[0]["error"])
        raise RuntimeError(
            f"{group_label}: strict filename/error checking is enabled and a problem was found: {first_problem}. "
            f"See CSV check tables in {group_output_dir}."
        )

    cfg.log(
        f"    {group_label}: recognized files={len(recognized_df)}, "
        f"skipped={len(skipped_df)}, errors={len(error_df)}, binned rows={len(per_instance_bin_df)}"
    )
    return per_instance_bin_df, recognized_df, skipped_df, error_df


def _collect_worker_result(result, recognized_records, skipped_records, error_records, binned_tables):
    status = result.get("status")
    if status == "ok":
        if result.get("recognized") is not None:
            recognized_records.append(result["recognized"])
        binned = result.get("binned")
        if binned is not None and not binned.empty:
            binned_tables.append(binned)
    elif status == "skipped":
        skipped_records.append(result["skipped"])
    elif status == "error":
        error_records.append(
            {
                "file": result.get("file"),
                "folder": result.get("folder"),
                "error": result.get("error"),
                "traceback": result.get("traceback"),
            }
        )
    else:
        error_records.append(
            {
                "file": "",
                "folder": "",
                "error": f"Unknown worker status: {status}",
                "traceback": "",
            }
        )


def process_one_group(group: dict, group_index: int, total_groups: int):
    cfg.raise_if_stop_requested(f"module 1 before group {group_index}/{total_groups}")
    group_label = f"{group['gene_label']} | {group['condition_label']}"
    group_output_dir = cfg.BINNED_DATA_DIR / cfg.group_folder_name(group)
    group_output_dir.mkdir(parents=True, exist_ok=True)
    output_base = group_output_dir / "per_instance_bin_stats"
    output_path = cfg.compressed_path(output_base)

    cfg.log("=" * 80)
    cfg.log(f"[Module 1 group {group_index}/{total_groups}] {group_label}")
    cfg.log(f"Output: {group_output_dir}")

    if output_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"    Existing output found; skipping because OVERWRITE_EXISTING=False: {output_path.name}")
        per_instance_bin_df = cfg.read_dataframe_compressed(output_path)
        recognized_path = group_output_dir / "recognized_csv_files.csv"
        recognized_df = pd.read_csv(recognized_path) if recognized_path.exists() else pd.DataFrame()
        skipped_path = group_output_dir / "skipped_files.csv"
        skipped_df = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()
        error_path = group_output_dir / "csv_binning_errors.csv"
        error_df = pd.read_csv(error_path) if error_path.exists() else pd.DataFrame()
    else:
        per_instance_bin_df, recognized_df, skipped_df, error_df = collect_group_binned_data(group, group_output_dir)
        cfg.save_dataframe_compressed(per_instance_bin_df, output_base, "per-instance bin stats")
        if cfg.SAVE_PER_INSTANCE_BIN_STATS_CSV:
            per_instance_bin_df.to_csv(
                group_output_dir / "per_instance_bin_stats.csv",
                index=False,
                encoding="utf-8-sig",
            )

    time_min = np.nan
    time_max = np.nan
    if "time_min" in recognized_df.columns and not recognized_df.empty:
        times = pd.to_numeric(recognized_df["time_min"], errors="coerce").dropna()
        if len(times) > 0:
            time_min = float(times.min())
            time_max = float(times.max())

    return {
        "gene_id": group["gene_id"],
        "gene_label": group["gene_label"],
        "condition": group["condition"],
        "condition_label": group["condition_label"],
        "group_output_dir": str(group_output_dir),
        "per_instance_bin_stats_pkl_gz": str(output_path),
        "n_recognized_files": int(len(recognized_df)),
        "n_skipped_files": int(len(skipped_df)),
        "n_binning_errors": int(len(error_df)),
        "n_instances": int(per_instance_bin_df["instance_uid"].nunique()) if "instance_uid" in per_instance_bin_df else 0,
        "time_min": time_min,
        "time_max": time_max,
        "n_nonempty_bin_rows": int(len(per_instance_bin_df)),
    }


def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="01_build_binned_data")
    manifest_path = cfg.BINNED_DATA_DIR / "binned_data_manifest.csv"
    manifest_binary = cfg.compressed_path(cfg.BINNED_DATA_DIR / "binned_data_manifest")

    if manifest_binary.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 1 output already exists; skipping: {manifest_binary}")
        return

    cfg.log("[Module 1] Building per-instance s-r binned data.")
    manifest_records = []
    total_groups = len(cfg.INPUT_GROUPS)
    for group_index, group in enumerate(cfg.INPUT_GROUPS, start=1):
        cfg.raise_if_stop_requested(f"module 1 before group {group_index}/{total_groups}")
        manifest_records.append(process_one_group(group, group_index, total_groups))
        cfg.raise_if_stop_requested(f"module 1 after group {group_index}/{total_groups}")

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    cfg.save_dataframe_compressed(manifest_df, cfg.BINNED_DATA_DIR / "binned_data_manifest", "binned data manifest")

    cfg.log("[Module 1] Summary:")
    cfg.log(f"    Groups processed: {len(manifest_df)}")
    cfg.log(f"    Recognized CSV files: {int(manifest_df['n_recognized_files'].sum()) if not manifest_df.empty else 0}")
    cfg.log(f"    Non-empty bin rows: {int(manifest_df['n_nonempty_bin_rows'].sum()) if not manifest_df.empty else 0}")
    cfg.log(f"    Output: {cfg.BINNED_DATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 1] Stopped gracefully: {exc}")
        cfg.log("[Module 1] Completed outputs are kept. Clear the stop request before resuming.")
