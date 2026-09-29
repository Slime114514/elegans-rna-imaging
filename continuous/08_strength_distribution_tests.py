# -*- coding: utf-8 -*-
"""
Module 8: distribution + correlation analyses across robust bins.

Row 1 (A-D):
  A-B  onset-time distribution (anesthesia vs stress), KS + MW tests
  C-D  rank-rank correlation: onset rank vs dff0 strength rank

Row 2 (E-H):
  E-F  deltaF cumulative-strength distribution (anesthesia vs stress)
  G-H  dff0  cumulative-strength distribution (anesthesia vs stress)

Inputs:
    Module 7 per-group long tables under 07_response_heatmaps/
      <gene>__<condition>/onset_heatmap_long.csv
      <gene>__<condition>/total_response_strength_heatmap_long.csv
      <gene>__<condition>/total_response_strength_dff0_heatmap_long.csv

Outputs:
    08_strength_distribution/strength_distribution_composite.png / pdf / svg
    08_strength_distribution/onset_distribution_tests.csv
    08_strength_distribution/strength_distribution_tests.csv
    08_strength_distribution/rank_rank_correlation.csv
"""

import importlib.util
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
plt.rcParams["svg.fonttype"] = "none"

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


OUTPUT_BASENAME = "strength_distribution_composite"
ANESTHESIA_COLOR = cfg.RESPONSE_COMPOSITE_CONDITION_COLORS.get("anesthesia", "#7F7F7F")
STRESS_COLOR     = cfg.RESPONSE_COMPOSITE_CONDITION_COLORS.get("stress", "#7FB3D5")

FIG_WIDTH  = cfg.STRENGTH_DIST_FIGSIZE[0]
FIG_HEIGHT = cfg.STRENGTH_DIST_FIGSIZE[1]
DPI        = cfg.STRENGTH_DIST_DPI
HIST_BINS  = cfg.STRENGTH_DIST_HIST_BINS
HIST_ALPHA = cfg.STRENGTH_DIST_HIST_ALPHA
HIST_LW    = cfg.STRENGTH_DIST_HIST_LINEWIDTH
SCATTER_S  = cfg.STRENGTH_DIST_SCATTER_SIZE
SCATTER_ALPHA = cfg.STRENGTH_DIST_SCATTER_ALPHA
LINE_WIDTH     = cfg.STRENGTH_DIST_REF_LINEWIDTH
PANEL_LABEL_SIZE  = cfg.STRENGTH_DIST_PANEL_LABEL_SIZE
TITLE_SIZE  = cfg.STRENGTH_DIST_TITLE_SIZE
AXIS_SIZE   = cfg.STRENGTH_DIST_AXIS_SIZE
TICK_SIZE   = cfg.STRENGTH_DIST_TICK_SIZE
TEST_SIZE   = cfg.STRENGTH_DIST_TEST_SIZE
LEGEND_SIZE = cfg.STRENGTH_DIST_LEGEND_SIZE
LABELS = list("ABCDEFGH")


# ── helpers ──────────────────────────────────────────────────────────────

def _load_long(table_name, value_col):
    df = pd.read_csv(table_name)
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    return values.to_numpy(dtype=float)


def _test_distribution(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = int(len(a)), int(len(b))
    r = {
        "n_a": na, "n_b": nb,
        "mean_a": np.mean(a) if na else np.nan, "mean_b": np.mean(b) if nb else np.nan,
        "median_a": np.median(a) if na else np.nan, "median_b": np.median(b) if nb else np.nan,
        "std_a": np.std(a, ddof=1) if na > 1 else np.nan, "std_b": np.std(b, ddof=1) if nb > 1 else np.nan,
        "median_diff_b_minus_a": np.nan,
        "direction": "no_test",
        "ks_stat": np.nan, "ks_p": np.nan,
        "mw_stat": np.nan, "mw_p": np.nan,
    }
    if na and nb:
        r["median_diff_b_minus_a"] = float(np.median(b) - np.median(a))
    if na < 3 or nb < 3 or scipy_stats is None:
        return r
    ks = scipy_stats.ks_2samp(a, b)
    mw = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    r.update(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
             mw_stat=float(mw.statistic), mw_p=float(mw.pvalue))

    med_diff = r["median_diff_b_minus_a"]
    ks_sig = r["ks_p"] < 0.05
    mw_sig = r["mw_p"] < 0.05
    if ks_sig or mw_sig:
        r["direction"] = "stress_higher" if med_diff > 0 else "stress_lower"
    else:
        r["direction"] = "no_significant_difference"
    return r


def _p_stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def _spearman_rank_corr(x, y):
    """Spearman rank correlation with p-value. Returns (rho, p)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or scipy_stats is None:
        return np.nan, np.nan
    r = scipy_stats.spearmanr(x, y)
    return float(r.correlation), float(r.pvalue)


# ── row-1 panels ─────────────────────────────────────────────────────────

def _plot_onset_histogram(ax, vals_a, vals_b, label_a, label_b, test, gene_label):
    all_vals = np.concatenate([vals_a, vals_b])
    if len(all_vals) == 0:
        return
    bins = np.histogram_bin_edges(all_vals, bins=HIST_BINS)
    ax.hist(vals_a, bins=bins, color=ANESTHESIA_COLOR, alpha=HIST_ALPHA,
            edgecolor=ANESTHESIA_COLOR, linewidth=HIST_LW, density=True)
    ax.hist(vals_b, bins=bins, color=STRESS_COLOR, alpha=HIST_ALPHA,
            edgecolor=STRESS_COLOR, linewidth=HIST_LW, density=True)
    _add_test_annotation(ax, test)
    if np.isfinite(test.get("median_a", np.nan)):
        _add_median_line(ax, test["median_a"], ANESTHESIA_COLOR)
    if np.isfinite(test.get("median_b", np.nan)):
        _add_median_line(ax, test["median_b"], STRESS_COLOR)
    ax.set_title(f"{gene_label} — onset time", fontsize=TITLE_SIZE, pad=4)
    _style_hist_ax(ax, "Onset time (min)")

    # legend：single entry per condition, place top-right inside
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ANESTHESIA_COLOR, lw=3, alpha=0.7, label=label_a),
        Line2D([0], [0], color=STRESS_COLOR, lw=3, alpha=0.7, label=label_b),
    ]
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False, loc="upper right")


def _plot_rank_rank(ax, onset, dff0_str, condition_label, color, annotate_y):
    """Rank-rank scatter of onset time vs dff0 strength.

    Adds one condition's scatter + Spearman-ρ annotation.
    Title / axis labels / panel label are set by the caller.
    """
    valid = np.isfinite(onset) & np.isfinite(dff0_str)
    onset_v = onset[valid]; dff0_v = dff0_str[valid]
    if len(onset_v) < 2:
        return

    rank_onset = np.argsort(np.argsort(onset_v)) / (len(onset_v) - 1)
    rank_dff0  = np.argsort(np.argsort(dff0_v))  / (len(dff0_v)  - 1)

    rho, p = _spearman_rank_corr(onset_v, dff0_v)

    ax.scatter(rank_onset, rank_dff0, s=SCATTER_S, color=color,
               alpha=SCATTER_ALPHA, edgecolors="none", zorder=1, label=condition_label)
    ax.text(0.03, annotate_y, f"{condition_label}  ρ={rho:.3f}  p={p:.4f}",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=TEST_SIZE, color=color)
    return rho, p


# ── row-2 panels ─────────────────────────────────────────────────────────

def _plot_strength_histogram(ax, vals_a, vals_b, label_a, label_b, test, gene_label, metric_label):
    all_vals = np.concatenate([vals_a, vals_b])
    if len(all_vals) == 0:
        return
    bins = np.histogram_bin_edges(all_vals, bins=HIST_BINS)
    ax.hist(vals_a, bins=bins, color=ANESTHESIA_COLOR, alpha=HIST_ALPHA,
            edgecolor=ANESTHESIA_COLOR, linewidth=HIST_LW, density=True)
    ax.hist(vals_b, bins=bins, color=STRESS_COLOR, alpha=HIST_ALPHA,
            edgecolor=STRESS_COLOR, linewidth=HIST_LW, density=True)
    _add_test_annotation(ax, test)
    if np.isfinite(test.get("median_a", np.nan)):
        _add_median_line(ax, test["median_a"], ANESTHESIA_COLOR)
    if np.isfinite(test.get("median_b", np.nan)):
        _add_median_line(ax, test["median_b"], STRESS_COLOR)
    metric_display = "ΔF" if metric_label == "deltaF" else "ΔF/F₀"
    ax.set_title(f"{gene_label} — cumulative {metric_display}", fontsize=TITLE_SIZE, pad=4)
    _style_hist_ax(ax, f"Cumulative {metric_display}")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ANESTHESIA_COLOR, lw=3, alpha=0.7, label=label_a),
        Line2D([0], [0], color=STRESS_COLOR, lw=3, alpha=0.7, label=label_b),
    ]
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False, loc="upper right")


# ── shared helpers ───────────────────────────────────────────────────────

def _add_median_line(ax, median, color):
    ax.axvline(median, color=color, linewidth=1.0, linestyle="--", alpha=0.7)


def _add_test_annotation(ax, test):
    ks_s = _p_stars(test["ks_p"])
    mw_s = _p_stars(test["mw_p"])
    text = (f"KS p={test['ks_p']:.4f}{ks_s}\n"
            f"MW p={test['mw_p']:.4f}{mw_s}")
    ax.text(0.03, 0.93, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=TEST_SIZE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC",
                      alpha=0.82))


def _style_hist_ax(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=AXIS_SIZE)
    ax.set_ylabel("Density", fontsize=AXIS_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    cfg.ensure_stage_dirs()
    cfg.write_global_settings(module_name="08_strength_distribution_tests")
    cfg.STRENGTH_DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)

    # build ordered group list: skn-1 anesthesia, skn-1 stress,
    #                          daf-16 anesthesia, daf-16 stress
    ordered = []
    for gene_id in ["skn-1", "daf-16"]:
        for cond in ["anesthesia", "stress"]:
            for g in cfg.INPUT_GROUPS:
                if g["gene_id"] == gene_id and g["condition"] == cond:
                    ordered.append(g)
                    break
    if len(ordered) < 4:
        raise RuntimeError("Need 4 groups.")
    cond_label_a = ordered[0]["condition_label"]
    cond_label_b = ordered[1]["condition_label"]

    # ── load data ─────────────────────────────────────────────────────
    data = {}
    for g in ordered:
        key = (g["gene_id"], g["condition"])
        gdir = cfg.RESPONSE_HEATMAP_DIR / cfg.safe_filename(
            f"{g['gene_id']}__{g['condition']}")
        onset   = _load_long(gdir / "onset_heatmap_long.csv", "onset_time_min")
        deltaF  = _load_long(gdir / "total_response_strength_heatmap_long.csv",
                             "total_response_strength")
        dff0    = _load_long(gdir / "total_response_strength_dff0_heatmap_long.csv",
                             "total_response_strength_dff0")
        data[key] = {"onset": onset, "deltaF": deltaF, "dff0": dff0}

    genes = [("skn-1", ordered[0]["gene_label"]),
             ("daf-16", ordered[2]["gene_label"])]

    onset_records   = []
    strength_records = []
    corr_records    = []

    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    # ── row 1 ─────────────────────────────────────────────────────────
    for col, (gene_id, gene_label) in enumerate(genes):
        g_a = (gene_id, "anesthesia")
        g_b = (gene_id, "stress")

        # A / B — onset histogram
        ax_on = axes[0, col]
        vals_a = data[g_a]["onset"]; vals_b = data[g_b]["onset"]
        test_on = _test_distribution(vals_a, vals_b)
        test_on.update(gene_id=gene_id, gene_label=gene_label, metric="onset_time",
                       condition_a="anesthesia", condition_b="stress")
        onset_records.append(test_on)
        _plot_onset_histogram(ax_on, vals_a, vals_b, cond_label_a, cond_label_b,
                              test_on, gene_label)

        # C / D — rank-rank
        ax_cd = axes[0, 2 + col]
        ax_cd.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=0.8,
                   linestyle="--", zorder=0)
        ax_cd.set_title(gene_label, fontsize=TITLE_SIZE, pad=4)
        ax_cd.set_xlabel("Onset-time rank", fontsize=AXIS_SIZE)
        ax_cd.set_ylabel("dF/F₀ strength rank", fontsize=AXIS_SIZE)
        ax_cd.tick_params(labelsize=TICK_SIZE)
        ax_cd.spines["top"].set_visible(False)
        ax_cd.spines["right"].set_visible(False)

        colors = {"anesthesia": ANESTHESIA_COLOR, "stress": STRESS_COLOR}
        cond_labels = {"anesthesia": cond_label_a, "stress": cond_label_b}
        for y_pos, cond_key in enumerate(["anesthesia", "stress"], start=1):
            g_key = (gene_id, cond_key)
            rho, p = _plot_rank_rank(ax_cd, data[g_key]["onset"],
                                     data[g_key]["dff0"],
                                     cond_labels[cond_key], colors[cond_key],
                                     annotate_y=0.94 - 0.08 * y_pos)
            corr_records.append(dict(gene_id=gene_id, gene_label=gene_label,
                                     condition=cond_key,
                                     n_bins=int(np.sum(
                                         np.isfinite(data[g_key]["onset"]) &
                                         np.isfinite(data[g_key]["dff0"]))),
                                     spearman_rho=rho, spearman_p=p))

    # ── row 2 ─────────────────────────────────────────────────────────
    for col, (gene_id, gene_label) in enumerate(genes):
        g_a = (gene_id, "anesthesia")
        g_b = (gene_id, "stress")

        for mcol, metric in enumerate(["deltaF", "dff0"]):
            ax_str = axes[1, col + 2 * mcol]
            vals_a = data[g_a][metric]; vals_b = data[g_b][metric]
            test_str = _test_distribution(vals_a, vals_b)
            test_str.update(gene_id=gene_id, gene_label=gene_label, metric=metric,
                            condition_a="anesthesia", condition_b="stress")
            strength_records.append(test_str)
            _plot_strength_histogram(ax_str, vals_a, vals_b,
                                     cond_label_a, cond_label_b,
                                     test_str, gene_label, metric)

    # ── panel labels ──────────────────────────────────────────────────
    for i, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, LABELS[i], transform=ax.transAxes,
                ha="left", va="top", fontsize=PANEL_LABEL_SIZE,
                fontweight="bold")

    fig.tight_layout(pad=cfg.STRENGTH_DIST_TIGHT_PAD, w_pad=cfg.STRENGTH_DIST_TIGHT_WPAD, h_pad=cfg.STRENGTH_DIST_TIGHT_HPAD)

    # ── save ──────────────────────────────────────────────────────────
    base = cfg.STRENGTH_DISTRIBUTION_DIR / OUTPUT_BASENAME
    for ext, do_save in [("png", True), ("pdf", True), ("svg", True)]:
        if do_save:
            fig.savefig(base.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)

    pd.DataFrame(onset_records).to_csv(
        cfg.STRENGTH_DISTRIBUTION_DIR / "onset_distribution_tests.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(strength_records).to_csv(
        cfg.STRENGTH_DISTRIBUTION_DIR / "strength_distribution_tests.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(corr_records).to_csv(
        cfg.STRENGTH_DISTRIBUTION_DIR / "rank_rank_correlation.csv",
        index=False, encoding="utf-8-sig")

    for label, df in [("onset", onset_records), ("strength", strength_records),
                       ("correlation", corr_records)]:
        cfg.log(f"[Module 8] {label} tests: {len(df)} rows")
    cfg.log(f"[Module 8] Figure: {base}.png / .pdf / .svg")


if __name__ == "__main__":
    try:
        main()
    except cfg.PipelineStopRequested as exc:
        cfg.log(f"[Module 8] Stopped gracefully: {exc}")
