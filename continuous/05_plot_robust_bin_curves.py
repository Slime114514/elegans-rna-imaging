# -*- coding: utf-8 -*-
"""
Module 5: plot deltaF and deltaF/F0 curves for robust significant bins.

Inputs:
    OUTPUT_DIR/02_delta_values/delta_values_manifest.pkl.gz
    and per-group delta chunks listed in that manifest.
    OUTPUT_DIR/04_robust_events/significant_bin_timepoints_robust.pkl.gz

Outputs:
    OUTPUT_DIR/05_bin_curves/robust_bin_curves/<prefix>_deltaF_curve.png
    OUTPUT_DIR/05_bin_curves/robust_bin_curves/<prefix>_dff0_curve.png
    OUTPUT_DIR/05_bin_curves/robust_bin_curves/<prefix>_curve_data.csv
    OUTPUT_DIR/05_bin_curves/robust_bin_curves/<prefix>_curve_summary.csv
    OUTPUT_DIR/05_bin_curves/robust_bin_curve_manifest.csv
    OUTPUT_DIR/05_bin_curves/robust_bin_curve_manifest.pkl.gz

Depends on:
    02_compute_delta_values.py and 04_robustness_filter_significant_events.py outputs.
"""

import importlib.util
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.ioff()


def _load_config():
    module_path = Path(__file__).resolve().with_name("00_config_utils.py")
    spec = importlib.util.spec_from_file_location("pipeline_config_utils", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()
plt.rcParams["font.family"] = cfg.FONT_FAMILY
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

_PLOT_DELTA_DF = None
_PLOT_SIG_DF = None


def init_plot_worker(delta_df, sig_df):
    global _PLOT_DELTA_DF, _PLOT_SIG_DF
    _PLOT_DELTA_DF = delta_df
    _PLOT_SIG_DF = sig_df


def save_figure(fig, output_base: Path):
    if cfg.SAVE_PNG:
        fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", dpi=cfg.CURVE_DPI)
    if cfg.SAVE_PDF:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if cfg.SAVE_SVG:
        fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def curve_summary(bin_df: pd.DataFrame):
    records = []
    for time_min, g in bin_df.groupby("time_min", sort=True):
        delta_ci = cfg.compute_mean_ci(g["deltaF"])
        dff0_ci = cfg.compute_mean_ci(g["dff0"])
        records.append(
            {
                "time_min": time_min,
                "deltaF_n": delta_ci["n"],
                "deltaF_mean": delta_ci["mean"],
                "deltaF_sd": delta_ci["sd"],
                "deltaF_sem": delta_ci["sem"],
                "deltaF_ci_low": delta_ci["ci_low"],
                "deltaF_ci_high": delta_ci["ci_high"],
                "dff0_n": dff0_ci["n"],
                "dff0_mean": dff0_ci["mean"],
                "dff0_sd": dff0_ci["sd"],
                "dff0_sem": dff0_ci["sem"],
                "dff0_ci_low": dff0_ci["ci_low"],
                "dff0_ci_high": dff0_ci["ci_high"],
            }
        )
    return pd.DataFrame(records).sort_values("time_min")


def add_significance_annotations(ax, significant_times, all_times, y_reference_by_time=None):
    significant_times = pd.to_numeric(pd.Series(significant_times), errors="coerce").dropna().to_numpy(dtype=float)
    if len(significant_times) == 0:
        return "0"

    all_times_arr = pd.to_numeric(pd.Series(all_times), errors="coerce").dropna().to_numpy(dtype=float)
    all_times_arr = np.asarray(sorted(set(t for t in all_times_arr if np.isfinite(t))), dtype=float)
    intervals = cfg.significant_intervals_from_times(significant_times, all_times_arr)
    interval_string = cfg.format_intervals(intervals)
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = 0.0, 1.0
    yrange = ymax - ymin
    if yrange <= 0:
        yrange = 1.0

    y_reference_by_time = y_reference_by_time or {}
    y_offset = 0.035 * yrange
    text_offset = 0.025 * yrange
    annotation_ys = []

    def expanded_interval_bounds(start, end):
        if len(all_times_arr) == 0:
            return float(start), float(end)
        start = float(start)
        end = float(end)
        before = all_times_arr[all_times_arr < start - 1e-9]
        after = all_times_arr[all_times_arr > end + 1e-9]
        left = float(before[-1]) if len(before) else start
        right = float(after[0]) if len(after) else end
        return left, right

    def local_reference_y(start, end):
        left, right = expanded_interval_bounds(start, end)
        values = []
        for time_value, y_value in y_reference_by_time.items():
            try:
                t = float(time_value)
                y = float(y_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(y) and left - 1e-9 <= t <= right + 1e-9:
                values.append(y)
        if values:
            return float(np.max(values))
        return float(ymax)

    for start, end in intervals:
        y_base = local_reference_y(start, end)
        y_mark = y_base + y_offset
        y_text = y_mark + text_offset
        annotation_ys.extend([y_mark, y_text])
        if abs(float(start) - float(end)) < 1e-9:
            ax.scatter(
                [start],
                [y_mark],
                marker=cfg.CURVE_TRIANGLE_MARKER,
                s=cfg.CURVE_TRIANGLE_SIZE,
                color=cfg.CURVE_SIGNIF_COLOR,
                zorder=5,
            )
        else:
            ax.hlines(
                y_mark,
                xmin=start,
                xmax=end,
                colors=cfg.CURVE_SIGNIF_COLOR,
                linewidth=cfg.CURVE_SIG_LINEWIDTH,
                zorder=5,
            )
            ax.text(
                (float(start) + float(end)) / 2.0,
                y_text,
                cfg.CURVE_SIG_TEXT,
                ha="center",
                va="bottom",
                fontsize=cfg.CURVE_SIG_TEXT_FONTSIZE,
                color=cfg.CURVE_SIGNIF_COLOR,
            )

    upper_needed = max(annotation_ys) + 0.04 * yrange if annotation_ys else ymax + 0.08 * yrange
    lower_needed = ymin - 0.04 * yrange
    ax.set_ylim(lower_needed, max(ymax, upper_needed))
    return interval_string


def plot_metric_curve(bin_df, summary_df, significant_times, metric, output_base, title, y_label, color):
    fig, ax = plt.subplots(figsize=cfg.CURVE_FIGSIZE, dpi=cfg.CURVE_DPI)

    for _, g in bin_df.groupby("instance_uid", sort=False):
        g = g.sort_values("time_min")
        ax.plot(
            g["time_min"],
            g[metric],
            color=cfg.CURVE_INDIVIDUAL_COLOR,
            alpha=cfg.CURVE_INDIVIDUAL_ALPHA,
            linewidth=cfg.CURVE_INDIVIDUAL_LINEWIDTH,
            zorder=1,
        )

    prefix = "deltaF" if metric == "deltaF" else "dff0"
    mean_col = f"{prefix}_mean"
    low_col = f"{prefix}_ci_low"
    high_col = f"{prefix}_ci_high"
    n_col = f"{prefix}_n"
    valid_summary = summary_df.dropna(subset=[mean_col]).copy()
    ax.plot(
        valid_summary["time_min"],
        valid_summary[mean_col],
        color=color,
        linewidth=cfg.CURVE_MEAN_LINEWIDTH,
        label="Mean",
        zorder=3,
    )
    ci_ok = valid_summary.dropna(subset=[low_col, high_col])
    if not ci_ok.empty:
        ax.fill_between(
            ci_ok["time_min"].to_numpy(dtype=float),
            ci_ok[low_col].to_numpy(dtype=float),
            ci_ok[high_col].to_numpy(dtype=float),
            color=color,
            alpha=cfg.CURVE_CI_ALPHA,
            linewidth=0,
            label="95% CI",
            zorder=2,
        )

    ref_parts = []
    metric_values = bin_df[["time_min", metric]].dropna().copy()
    if not metric_values.empty:
        ref_parts.append(metric_values.groupby("time_min", as_index=False)[metric].max().rename(columns={metric: "y_ref"}))
    ci_ref = valid_summary[["time_min", high_col]].dropna().copy()
    if not ci_ref.empty:
        ref_parts.append(ci_ref.rename(columns={high_col: "y_ref"}))
    if ref_parts:
        y_reference = (
            pd.concat(ref_parts, ignore_index=True)
            .groupby("time_min", as_index=False)["y_ref"]
            .max()
        )
        y_reference_by_time = dict(zip(y_reference["time_min"].astype(float), y_reference["y_ref"].astype(float)))
    else:
        y_reference_by_time = {}

    interval_string = add_significance_annotations(
        ax,
        significant_times,
        summary_df["time_min"].to_numpy(dtype=float),
        y_reference_by_time=y_reference_by_time,
    )
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=cfg.CURVE_TITLE_FONTSIZE)
    ax.set_xlabel("time_min", fontsize=cfg.CURVE_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=cfg.CURVE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=cfg.CURVE_TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=cfg.CURVE_LEGEND_FONTSIZE, frameon=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)
    return interval_string, int(valid_summary[n_col].max()) if n_col in valid_summary and not valid_summary.empty else 0


def plot_one_bin(task: dict):
    task_out = {k: v for k, v in task.items() if k not in {"bin_df", "significant_times"}}
    prepared_bin_df = task.get("bin_df")
    prepared_significant_times = task.get("significant_times")
    delta_df = _PLOT_DELTA_DF
    sig_df = _PLOT_SIG_DF
    gene_id = task["gene_id"]
    gene_label = task["gene_label"]
    condition = task["condition"]
    condition_label = task["condition_label"]
    s_bin = int(task["s_bin"])
    r_bin = int(task["r_bin"])
    curve_dir = Path(task["curve_dir"])

    if prepared_bin_df is not None:
        bin_df = prepared_bin_df.copy()
    else:
        mask = (
            (delta_df["gene_id"] == gene_id)
            & (delta_df["condition"] == condition)
            & (delta_df["s_bin"].astype(int) == s_bin)
            & (delta_df["r_bin"].astype(int) == r_bin)
        )
        bin_df = delta_df.loc[mask].copy()
    if bin_df.empty:
        return {
            **task_out,
            "status": "skipped_empty_bin",
            "deltaF_curve_png": "",
            "dff0_curve_png": "",
            "curve_data_csv": "",
            "curve_summary_csv": "",
        }

    bin_df["time_min"] = pd.to_numeric(bin_df["time_min"], errors="coerce")
    bin_df["deltaF"] = pd.to_numeric(bin_df["deltaF"], errors="coerce")
    bin_df["dff0"] = pd.to_numeric(bin_df["dff0"], errors="coerce")
    bin_df = bin_df.sort_values(["instance_uid", "time_min"])
    summary_df = curve_summary(bin_df)

    if prepared_significant_times is not None:
        significant_times = np.asarray(prepared_significant_times, dtype=float)
    else:
        sig_mask = (
            (sig_df["gene_id"] == gene_id)
            & (sig_df["condition"] == condition)
            & (sig_df["s_bin"].astype(int) == s_bin)
            & (sig_df["r_bin"].astype(int) == r_bin)
        )
        significant_times = pd.to_numeric(sig_df.loc[sig_mask, "time_min"], errors="coerce").dropna().to_numpy(dtype=float)

    prefix = cfg.safe_filename(f"{gene_id}_{condition}_s{s_bin:04d}_r{r_bin:04d}")
    data_path = curve_dir / f"{prefix}_curve_data.csv"
    summary_path = curve_dir / f"{prefix}_curve_summary.csv"

    data_cols = [
        "gene_id",
        "gene_label",
        "condition",
        "condition_label",
        "instance",
        "instance_uid",
        "time_min",
        "s_bin",
        "r_bin",
        "s_start",
        "s_end",
        "r_start",
        "r_end",
        "s_center",
        "r_center",
        "F_t",
        "F0",
        "deltaF",
        "dff0",
    ]
    bin_df[[c for c in data_cols if c in bin_df.columns]].to_csv(data_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    n_instances = bin_df["instance_uid"].nunique()
    title = (
        f"{gene_label} | {condition_label}\n"
        f"s bin {s_bin}, r bin {r_bin}, n={n_instances}"
    )
    delta_base = curve_dir / f"{prefix}_deltaF_curve"
    dff0_base = curve_dir / f"{prefix}_dff0_curve"
    interval_string, n_for_delta = plot_metric_curve(
        bin_df,
        summary_df,
        significant_times,
        "deltaF",
        delta_base,
        title,
        "ΔF",
        cfg.CURVE_DELTA_COLOR,
    )
    _, n_for_dff0 = plot_metric_curve(
        bin_df,
        summary_df,
        significant_times,
        "dff0",
        dff0_base,
        title,
        "ΔF/F₀",
        cfg.CURVE_DFF0_COLOR,
    )

    return {
        **task_out,
        "status": "ok",
        "prefix": prefix,
        "n_instances": int(bin_df["instance_uid"].nunique()),
        "n_for_deltaF_max": n_for_delta,
        "n_for_dff0_max": n_for_dff0,
        "significant_times": ";".join(cfg.format_time_value(t) for t in sorted(set(significant_times))),
        "significant_intervals": interval_string,
        "deltaF_curve_png": str(delta_base.with_suffix(".png")) if cfg.SAVE_PNG else "",
        "dff0_curve_png": str(dff0_base.with_suffix(".png")) if cfg.SAVE_PNG else "",
        "curve_data_csv": str(data_path),
        "curve_summary_csv": str(summary_path),
    }


def build_plot_tasks(robust_df: pd.DataFrame, curve_dir: Path):
    if robust_df.empty:
        return []
    unique_bins = (
        robust_df.sort_values(["gene_id", "condition", "p_adj", "time_min", "r_bin", "s_bin"])
        .drop_duplicates(["gene_id", "condition", "s_bin", "r_bin"], keep="first")
        .copy()
    )
    if cfg.MAX_ROBUST_BIN_CURVES_TO_DRAW is not None:
        unique_bins = unique_bins.head(int(cfg.MAX_ROBUST_BIN_CURVES_TO_DRAW))

    tasks = []
    for _, row in unique_bins.iterrows():
        tasks.append(
            {
                "gene_id": row["gene_id"],
                "gene_label": row.get("gene_label", row["gene_id"]),
                "condition": row["condition"],
                "condition_label": row.get("condition_label", row["condition"]),
                "s_bin": int(row["s_bin"]),
                "r_bin": int(row["r_bin"]),
                "curve_dir": str(curve_dir),
            }
        )
    return tasks




def _prepare_plot_tasks_with_data(delta_df: pd.DataFrame, robust_df: pd.DataFrame, curve_dir: Path):
    tasks = build_plot_tasks(robust_df, curve_dir)
    if not tasks:
        return []

    key_df = pd.DataFrame([{"s_bin": int(t["s_bin"]), "r_bin": int(t["r_bin"])} for t in tasks]).drop_duplicates()
    work_cols = [
        "gene_id", "gene_label", "condition", "condition_label", "instance", "instance_uid",
        "time_min", "s_bin", "r_bin", "s_start", "s_end", "r_start", "r_end", "s_center", "r_center",
        "F_t", "F0", "deltaF", "dff0", "baseline_time_min",
    ]
    work = delta_df.loc[:, [c for c in work_cols if c in delta_df.columns]].copy()
    work["s_bin"] = pd.to_numeric(work["s_bin"], errors="coerce").astype("Int64")
    work["r_bin"] = pd.to_numeric(work["r_bin"], errors="coerce").astype("Int64")
    if "baseline_time_min" in work.columns:
        work["baseline_time_min"] = pd.to_numeric(work["baseline_time_min"], errors="coerce")
    work, _ = cfg.filter_late_baseline_bins(work)
    work = work.dropna(subset=["s_bin", "r_bin"]).copy()
    work["s_bin"] = work["s_bin"].astype(np.int32)
    work["r_bin"] = work["r_bin"].astype(np.int32)
    selected = work.merge(key_df, on=["s_bin", "r_bin"], how="inner")
    bin_groups = {
        (int(s_bin), int(r_bin)): g.drop(columns=[]).reset_index(drop=True)
        for (s_bin, r_bin), g in selected.groupby(["s_bin", "r_bin"], sort=False)
    }

    sig_map = {}
    for (s_bin, r_bin), g in robust_df.groupby(["s_bin", "r_bin"], sort=False):
        sig_map[(int(s_bin), int(r_bin))] = pd.to_numeric(g["time_min"], errors="coerce").dropna().to_numpy(dtype=float)

    prepared = []
    empty = pd.DataFrame(columns=work.columns)
    for task in tasks:
        key = (int(task["s_bin"]), int(task["r_bin"]))
        item = dict(task)
        item["bin_df"] = bin_groups.get(key, empty)
        item["significant_times"] = sig_map.get(key, np.asarray([], dtype=float))
        prepared.append(item)
    return prepared


def run_plotting(delta_df: pd.DataFrame, robust_df: pd.DataFrame, curve_dir: Path):
    prepared_tasks = _prepare_plot_tasks_with_data(delta_df, robust_df, curve_dir)
    if not prepared_tasks:
        return pd.DataFrame()

    workers = min(int(cfg.PLOT_CURVE_WORKERS), len(prepared_tasks))
    if not (cfg.USE_MULTIPROCESSING and workers > 1):
        cfg.log("    Curve plotting sequentially.")
        records = []
        for i, task in enumerate(prepared_tasks, start=1):
            cfg.raise_if_stop_requested(f"module 5 before plotting bin {i}/{len(prepared_tasks)}")
            records.append(plot_one_bin(task))
            cfg.log(f"        Plotted bin curve {i}/{len(prepared_tasks)}")
        return pd.DataFrame(records)

    max_pending = min(int(getattr(cfg, "PLOT_CURVE_MAX_PENDING_TASKS", workers)), len(prepared_tasks))
    cfg.log(f"    Parallel curve plotting enabled with small per-bin tasks: workers={workers}, max_pending={max_pending}")
    records = []
    next_i = 0
    finished = 0
    pending = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while next_i < len(prepared_tasks) and len(pending) < max_pending:
            cfg.raise_if_stop_requested(f"module 5 before submitting plot task {next_i + 1}/{len(prepared_tasks)}")
            future = executor.submit(plot_one_bin, prepared_tasks[next_i])
            pending[future] = next_i + 1
            next_i += 1

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                records.append(future.result())
                finished += 1
                cfg.log(f"        Plotted bin curve {finished}/{len(prepared_tasks)}")
                stop_requested = cfg.should_stop_requested()
                if (not stop_requested) and next_i < len(prepared_tasks):
                    cfg.raise_if_stop_requested(f"module 5 before submitting plot task {next_i + 1}/{len(prepared_tasks)}")
                    new_future = executor.submit(plot_one_bin, prepared_tasks[next_i])
                    pending[new_future] = next_i + 1
                    next_i += 1
                elif stop_requested and not pending:
                    cfg.raise_if_stop_requested(f"module 5 after plotting bin {finished}/{len(prepared_tasks)}")
    return pd.DataFrame(records)


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


def run_plotting_from_manifest(delta_manifest: pd.DataFrame, robust_df: pd.DataFrame, curve_dir: Path):
    if robust_df.empty:
        return pd.DataFrame()

    manifests = []
    if "ALL" in set(delta_manifest.get("gene_id", [])):
        delta_df = cfg.read_dataframe_compressed(Path(delta_manifest.iloc[0]["delta_values_path"]))
        return run_plotting(delta_df, robust_df, curve_dir)

    for (gene_id, condition), robust_g in robust_df.groupby(["gene_id", "condition"], sort=False):
        cfg.raise_if_stop_requested(f"module 5 before group {gene_id} | {condition}")
        match = delta_manifest.loc[
            (delta_manifest["gene_id"] == gene_id) & (delta_manifest["condition"] == condition)
        ]
        if match.empty:
            raise FileNotFoundError(f"No delta chunk found for {gene_id} | {condition}")
        delta_path = Path(match.iloc[0]["delta_values_path"])
        cfg.log(f"    Reading delta chunk for plotting: {delta_path}")
        delta_df = cfg.read_dataframe_compressed(delta_path)
        manifest_g = run_plotting(delta_df, robust_g, curve_dir)
        manifests.append(manifest_g)

    return pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()


def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="05_plot_robust_bin_curves")
    curve_dir = cfg.BIN_CURVES_DIR / "robust_bin_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    manifest_base = cfg.BIN_CURVES_DIR / "robust_bin_curve_manifest"
    manifest_path = cfg.compressed_path(manifest_base)
    if manifest_path.exists() and not cfg.OVERWRITE_EXISTING:
        cfg.log(f"Module 5 output already exists; skipping: {manifest_path}")
        return

    robust_path = cfg.compressed_path(cfg.ROBUST_EVENTS_DIR / "significant_bin_timepoints_robust")
    cfg.log("[Module 5] Reading Module 2 delta manifest.")
    delta_manifest = load_delta_manifest()
    cfg.log(f"[Module 5] Reading robust significant events: {robust_path}")
    robust_df = cfg.read_dataframe_compressed(robust_path)

    cfg.log("[Module 5] Plotting robust bin curves.")
    manifest_df = run_plotting_from_manifest(delta_manifest, robust_df, curve_dir)
    manifest_df.to_csv(cfg.BIN_CURVES_DIR / "robust_bin_curve_manifest.csv", index=False, encoding="utf-8-sig")
    cfg.save_dataframe_compressed(manifest_df, manifest_base, "robust bin curve manifest")

    cfg.log("[Module 5] Summary:")
    cfg.log(f"    Unique robust bins plotted: {len(manifest_df)}")
    cfg.log(f"    Output: {curve_dir}")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 5] Stopped gracefully: {exc}")
        cfg.log("[Module 5] Completed outputs are kept. Clear the stop request before resuming.")
