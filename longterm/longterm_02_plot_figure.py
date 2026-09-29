# -*- coding: utf-8 -*-
"""
Long-term Module 2: read computed statistics and plot the 3×4 combined figure.

Inputs:
    COMPUTED_DATA_DIR/instance_summary.pkl.gz
    COMPUTED_DATA_DIR/welch_ttest.pkl.gz
    COMPUTED_DATA_DIR/ap_bin_stats.pkl.gz
    COMPUTED_DATA_DIR/ap_bin_sig_ranges.pkl.gz
    COMPUTED_DATA_DIR/heatmap_data.pkl.gz

Outputs:
    OUTPUT_DIR/lifespan_3x4_combined.png / pdf / svg
    OUTPUT_DIR/*.csv (intermediate tables)

Depends on:
    longterm_01_compute_data.py output.
"""

import importlib.util
import pickle
import gzip as gzip_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.transforms import Bbox


def _load_config():
    module_path = Path(__file__).resolve().with_name("longterm_00_config_utils.py")
    spec = importlib.util.spec_from_file_location("lt_config", module_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = _load_config()
plt.ioff()


# =========================
# Colormaps
# =========================

def get_blue_yellow_red_cmap():
    cmap = LinearSegmentedColormap.from_list("blue_yellow_red", [
        "#08306B", "#2171B5", "#6BAED6", "#FEE08B", "#FDAE61", "#D73027",
    ])
    cmap.set_bad(color="#E0E0E0")
    return cmap


def get_difference_cmap():
    cmap = LinearSegmentedColormap.from_list("blue_cream_red", [
        "#2C7BB6", "#FFFFDD", "#D7191C",
    ])
    cmap.set_bad(color="#E0E0E0")
    return cmap


def get_ap_day_colors(days):
    day_cmap = LinearSegmentedColormap.from_list("green_blue_gradient", [
        "#A8D8B9", "#88C9A9", "#76C1B6", "#7FB3D5", "#6895C7", "#5A7FB8",
    ])
    palette = day_cmap(np.linspace(0, 1, len(days)))
    return {day: palette[i] for i, day in enumerate(days)}


# =========================
# Plotting helpers
# =========================

def add_panel_label(ax, label):
    ax.text(-0.3, 1.15, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=cfg.PANEL_LABEL_SIZE, fontweight="bold")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=cfg.TICK_LABEL_SIZE, width=0.8, length=3)
    ax.xaxis.label.set_size(cfg.AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(cfg.AXIS_LABEL_SIZE)
    ax.title.set_size(cfg.TITLE_SIZE)
    clean_axes(ax)


def make_square(ax):
    ax.set_box_aspect(1)


def inset_panel(ax, l=0.14, r=0.12, b=0.15, t=0.08):
    """Shrink axes inside the subplot cell so labels stay within the cell.
    Returns the original cell bounding box for debugging."""
    orig = ax.get_position()
    ax.set_position([orig.x0 + l * orig.width,  orig.y0 + b * orig.height,
                     orig.width * (1 - l - r), orig.height * (1 - b - t)])
    return orig  # original cell box


def add_manual_colorbar(fig, axes_group, mappable, label, side="right",
                        width=0.010, pad=0.008, height_frac=0.78):
    bboxes = [ax.get_position() for ax in axes_group]
    bbox = Bbox.union(bboxes)
    cbar_h = bbox.height * height_frac
    cbar_y = bbox.y0 + (bbox.height - cbar_h) / 2
    cbar_x = bbox.x1 + pad if side == "right" else bbox.x0 - pad - width
    cax = fig.add_axes([cbar_x, cbar_y, width, cbar_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=cfg.CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=cfg.CBAR_TICK_SIZE, width=0.7, length=2.5, rotation=45)
    return cbar


def summarize_by_day(df, value_col):
    rows = []
    for day in sorted(df["day"].unique()):
        values = df.loc[df["day"] == day, value_col].dropna().to_numpy()
        mean, ci_val, n_val = cfg.mean_and_95ci(values)
        rows.append({"day": day, "mean": mean, "ci95": ci_val, "n": n_val})
    return pd.DataFrame(rows)


# =========================
# Panel functions
# =========================

def plot_day_metric(ax, df, value_col, ylabel, pval_df=None):
    summary = summarize_by_day(df, value_col)
    days = summary["day"].to_numpy()
    means = summary["mean"].to_numpy()
    cis = summary["ci95"].to_numpy()
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    for day in days:
        values = df.loc[df["day"] == day, value_col].dropna().to_numpy()
        jitter = rng.uniform(-cfg.JITTER_WIDTH, cfg.JITTER_WIDTH, size=len(values))
        ax.scatter(np.full(len(values), day) + jitter, values,
                   s=cfg.POINT_SIZE, color=cfg.POINT_COLOR, alpha=cfg.POINT_ALPHA,
                   edgecolors="none", zorder=2)

    ax.fill_between(days, means - cis, means + cis, color=cfg.CI_COLOR,
                    alpha=cfg.CI_ALPHA, linewidth=0, zorder=1)
    ax.plot(days, means, color=cfg.MAIN_COLOR, linewidth=cfg.LINE_WIDTH,
            marker="o", markersize=2, zorder=3)
    ax.axhline(1, color='black', linestyle="--", linewidth=cfg.LINE_WIDTH)

    if pval_df is not None and not pval_df.empty:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin if ymax > ymin else 1
        max_star_y = ymax
        for _, row in pval_df.iterrows():
            day = row["day"]
            stars = row.get("stars_holm_within_metric", row["stars"])
            if stars == "n.s.":
                continue
            day_values = df.loc[df["day"] == day, value_col].dropna().to_numpy()
            if len(day_values) == 0:
                continue
            y_base = np.nanmax(day_values)
            text_y = y_base + 0.05 * yrange
            ax.text(day, text_y, stars, ha="center", va="bottom",
                    fontsize=7, color="black")
            if text_y > max_star_y:
                max_star_y = text_y
        if max_star_y > ymax:
            ax.set_ylim(ymin, max_star_y + 0.05 * yrange)
    ax.set_xlabel("Days of adulthood")
    ax.set_ylabel(ylabel)
    ax.set_xticks(days)
    style_axis(ax)
    make_square(ax)
    inset_panel(ax)


def plot_ap_curves(ax, ap_summary):
    days = sorted(ap_summary.keys())
    bin_centers = np.arange(cfg.AP_N_BINS) * cfg.AP_BIN_WIDTH + cfg.AP_BIN_WIDTH / 2
    day_colors = get_ap_day_colors(days)
    important_days = {1, 10, 16}
    for day in days:
        info = ap_summary[day]
        color = day_colors[day]
        lw = cfg.LINE_WIDTH_IMPORTANT if day in important_days else cfg.LINE_WIDTH
        ax.plot(bin_centers, info["mean"], color=color, linewidth=lw,
                alpha=1.0, label=f"Day {day}")
        ax.fill_between(bin_centers, info["mean"] - info["ci95"],
                        info["mean"] + info["ci95"], color=color,
                        alpha=0.01, linewidth=0)
    ax.set_xlabel("Axial position")
    ax.set_ylabel("Mean fluorescence intensity")
    ax.set_xlim(0, 1)
    ax.set_ylim(-100, 12000)
    style_axis(ax)
    make_square(ax)
    inset_panel(ax)
    ax.legend(frameon=False, fontsize=cfg.LEGEND_SIZE, loc="center left",
              bbox_to_anchor=(1.02, 0.5), ncol=1, borderaxespad=0.0,
              handlelength=1.4, handletextpad=0.4)


def plot_top5_probability_heatmap(ax, prob, title):
    cmap = get_blue_yellow_red_cmap()
    im = ax.imshow(prob, origin="lower", extent=[cfg.S_MIN, cfg.S_MAX, cfg.R_MIN, cfg.R_MAX],
                   aspect="equal", cmap=cmap, vmin=cfg.TOP5_PROB_VMIN,
                   vmax=cfg.TOP5_PROB_VMAX, interpolation="nearest")
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel("Axial position"); ax.set_ylabel("Radial position")
    ax.set_xlim(cfg.S_MIN, cfg.S_MAX); ax.set_ylim(cfg.R_MIN, cfg.R_MAX)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    style_axis(ax); make_square(ax)
    orig_pos = inset_panel(ax)
    pos = ax.get_position()
    cax = ax.figure.add_axes([pos.x1 + 0.012, pos.y0, 0.01, pos.height])
    ax.figure.colorbar(im, cax=cax, label="Prob.")
    cax.tick_params(labelsize=cfg.CBAR_TICK_SIZE, width=0.7, length=2.5)
    return im


def plot_mean_hnet_heatmap(ax, mean_map, title):
    cmap = get_blue_yellow_red_cmap()
    im = ax.imshow(mean_map, origin="lower", extent=[cfg.S_MIN, cfg.S_MAX, cfg.R_MIN, cfg.R_MAX],
                   aspect="equal", cmap=cmap, vmin=cfg.MEAN_HNET_VMIN,
                   vmax=cfg.MEAN_HNET_VMAX, interpolation="nearest")
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel("Axial position"); ax.set_ylabel("Radial position")
    ax.set_xlim(cfg.S_MIN, cfg.S_MAX); ax.set_ylim(cfg.R_MIN, cfg.R_MAX)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    style_axis(ax); make_square(ax)
    orig_pos = inset_panel(ax)
    pos = ax.get_position()
    cax = ax.figure.add_axes([pos.x1 + 0.012, pos.y0, 0.01, pos.height])
    ax.figure.colorbar(im, cax=cax, label="Mean h_net")
    cax.tick_params(labelsize=cfg.CBAR_TICK_SIZE, width=0.7, length=2.5)
    return im


def plot_difference_heatmap(ax, diff_map, title):
    cmap = get_difference_cmap()
    valid = diff_map[np.isfinite(diff_map)]
    if cfg.DIFF_VABS is None:
        if valid.size == 0:
            vabs = 1.0
        else:
            vabs = np.nanpercentile(np.abs(valid), cfg.DIFF_PERCENTILE_IF_AUTO)
            if not np.isfinite(vabs) or vabs == 0:
                vabs = 1.0
    else:
        vabs = cfg.DIFF_VABS
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(diff_map, origin="lower", extent=[cfg.S_MIN, cfg.S_MAX, cfg.R_MIN, cfg.R_MAX],
                   aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=cfg.TITLE_SIZE)
    ax.set_xlabel("Axial position"); ax.set_ylabel("Radial position")
    ax.set_xlim(cfg.S_MIN, cfg.S_MAX); ax.set_ylim(cfg.R_MIN, cfg.R_MAX)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    style_axis(ax); make_square(ax)
    orig_pos = inset_panel(ax)
    pos = ax.get_position()
    cax = ax.figure.add_axes([pos.x1 + 0.012, pos.y0, 0.01, pos.height])
    ax.figure.colorbar(im, cax=cax, label=r"$\Delta$ h_net")
    cax.tick_params(labelsize=cfg.CBAR_TICK_SIZE, width=0.7, length=2.5)
    return im


# =========================
# Save tables
# =========================

def save_intermediate_outputs(df, p_all, top5_prob_maps, mean_hnet_maps, diff_map,
                              ap_bin_stats_df=None, ap_bin_sig_ranges_df=None):
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bn = cfg.OUTPUT_BASENAME
    df.to_csv(cfg.OUTPUT_DIR / f"{bn}_instance_summary.csv", index=False, encoding="utf-8-sig")
    p_all.to_csv(cfg.OUTPUT_DIR / f"{bn}_welch_ttest_vs_day1.csv", index=False, encoding="utf-8-sig")
    for day, mat in top5_prob_maps.items():
        pd.DataFrame(mat).to_csv(cfg.OUTPUT_DIR / f"{bn}_day{day}_top5_probability_matrix.csv",
                                 index=False, header=False, encoding="utf-8-sig")
    for day, mat in mean_hnet_maps.items():
        pd.DataFrame(mat).to_csv(cfg.OUTPUT_DIR / f"{bn}_day{day}_mean_hnet_matrix.csv",
                                 index=False, header=False, encoding="utf-8-sig")
    pd.DataFrame(diff_map).to_csv(cfg.OUTPUT_DIR / f"{bn}_day16_minus_day1_matrix.csv",
                                  index=False, header=False, encoding="utf-8-sig")
    if ap_bin_stats_df is not None:
        ap_bin_stats_df.to_csv(cfg.OUTPUT_DIR / f"{bn}_ap_bin_vs_global_mean_stats.csv",
                               index=False, encoding="utf-8-sig")
    if ap_bin_sig_ranges_df is not None:
        ap_bin_sig_ranges_df.to_csv(cfg.OUTPUT_DIR / f"{bn}_ap_bin_p_lt_0p05_ranges.csv",
                                    index=False, encoding="utf-8-sig")


# =========================
# Combined figure
# =========================

def plot_combined_figure(df, ap_summary, top5_prob_maps, mean_hnet_maps, diff_map, p_values):
    """4-row × 3-col: Col 0 = A-D line plots, Col 1 = E-H top5, Col 2 = I-L mean hnet."""
    fig, axes = plt.subplots(4, 3, figsize=cfg.FIGSIZE, dpi=cfg.FIG_DPI)
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.07, top=0.96,
                        wspace=0.18, hspace=0.15)

    # ---- Col 0: A-D line plots ----
    plot_day_metric(axes[0, 0], df, value_col="relative_total_h_net",
                    ylabel="Total fluorescence",
                    pval_df=p_values["relative_total_h_net"])
    plot_day_metric(axes[1, 0], df, value_col="relative_signal_mean_h_net",
                    ylabel="Mean fluorescence",
                    pval_df=p_values["relative_signal_mean_h_net"])
    plot_day_metric(axes[2, 0], df, value_col="signal_ratio",
                    ylabel="Signal-positive area ratio",
                    pval_df=p_values["signal_ratio"])
    plot_ap_curves(axes[3, 0], ap_summary)

    # ---- Col 1: E-H top-5% probability + contribution ----
    im_top5 = None
    for ax, day in zip(axes[0:3, 1], cfg.TARGET_HEATMAP_DAYS):
        im_top5 = plot_top5_probability_heatmap(ax, top5_prob_maps[day], f"Day {day}")
    plot_day_metric(axes[3, 1], df, value_col="top5_contribution",
                    ylabel="Top 5% contribution",
                    pval_df=p_values["top5_contribution"])
    axes[3, 1].set_ylim(0, min(1.05, max(0.2, np.nanmax(df["top5_contribution"].dropna()) * 1.25)))

    # ---- Col 2: I-L mean h_net maps + diff ----
    im_mean = None
    for ax, day in zip(axes[0:3, 2], cfg.TARGET_HEATMAP_DAYS):
        im_mean = plot_mean_hnet_heatmap(ax, mean_hnet_maps[day], f"Day {day}")
    im_diff = plot_difference_heatmap(axes[3, 2], diff_map, "Day 16 \u2212 Day 1")

    # ---- Panel labels (col-major: ABCD in col 0, EFGH in col 1, IJKL in col 2) ----
    labels = list("ABCDEFGHIJKL")
    idx = 0
    for col in range(3):
        for row in range(4):
            add_panel_label(axes[row, col], labels[idx])
            make_square(axes[row, col])
            idx += 1

    return fig


# =========================
# Main
# =========================

def main():
    cfg.ensure_dirs()

    # Read computed data
    cfg.log("[Module 2] Reading computed data.")
    df     = cfg.read_compressed(cfg.COMPUTED_DATA_DIR / "instance_summary")
    p_all  = cfg.read_compressed(cfg.COMPUTED_DATA_DIR / "welch_ttest")
    ap_bin_stats_df   = cfg.read_compressed(cfg.COMPUTED_DATA_DIR / "ap_bin_stats")
    ap_bin_sig_ranges = cfg.read_compressed(cfg.COMPUTED_DATA_DIR / "ap_bin_sig_ranges")

    gzip_path = cfg.COMPUTED_DATA_DIR / "heatmap_data.pkl.gz"
    with gzip_module.open(gzip_path, "rb") as f:
        hdata = pickle.load(f)
    ap_summary     = hdata["ap_summary"]
    top5_prob_maps  = hdata["top5_prob_maps"]
    mean_hnet_maps  = hdata["mean_hnet_maps"]
    diff_map        = hdata["diff_map"]

    # Reconstruct p_values dict for plotting
    p_values = {}
    for metric in ["relative_total_h_net", "relative_signal_mean_h_net", "signal_ratio", "top5_contribution"]:
        p_values[metric] = p_all[p_all["metric"] == metric].copy()

    if cfg.SAVE_INTERMEDIATE_TABLES:
        cfg.log("[Module 2] Saving intermediate CSV tables.")
        save_intermediate_outputs(df, p_all, top5_prob_maps, mean_hnet_maps, diff_map,
                                  ap_bin_stats_df, ap_bin_sig_ranges)

    cfg.log("[Module 2] Drawing the 3×4 combined figure.")
    fig = plot_combined_figure(df, ap_summary, top5_prob_maps, mean_hnet_maps, diff_map, p_values)

    for ext in ["png", "svg", "pdf"]:
        out_path = cfg.OUTPUT_DIR / f"{cfg.OUTPUT_BASENAME}.{ext}"
        fig.savefig(out_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
        cfg.log(f"[Saved] {out_path}")
    plt.close(fig)
    cfg.log("[Module 2] Done.")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 2] Stopped gracefully: {exc}")
