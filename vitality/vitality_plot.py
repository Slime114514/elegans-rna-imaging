import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ============================================================
# User paths
# ============================================================

BODY_LENGTH_FILE = r"path/to/your/body_length.xlsx"
EGG_SHAPE_FILE = r"path/to/your/egg_shape.xlsx"
BROOD_SIZE_FILE = r"path/to/your/brood_size.xlsx"
LONGEVITY_FILE = r"path/to/your/longevity.xlsx"
OUTPUT_PATH = r"path/to/your/output/comprehensive_analysis.png"

# If None, the statistics workbook will be saved next to OUTPUT_PATH
# as: comprehensive_analysis_analysis_tables.xlsx
STATS_OUTPUT_PATH = None

# Multiple-comparison correction for repeated body-length comparisons.
# Supported: "holm", "bonferroni", "fdr_bh", "none"
BODY_LENGTH_P_ADJUST_METHOD = "holm"


# ============================================================
# Unified style parameters
# ============================================================

DPI = 600
FIGSIZE = (4.7, 7.0)

FONT_FAMILY = "Arial"
PANEL_LABEL_SIZE = 9
AXIS_LABEL_SIZE = 7
TICK_LABEL_SIZE = 6
LEGEND_FONT_SIZE = 6
ANNOTATION_FONT_SIZE = 6
PVAL_FONT_SIZE = 6

N2_COLOR = "#7F7F7F"
EXP_COLOR = "#7FB3D5"
N2_LIGHT = "#D9D9D9"
EXP_LIGHT = "#D6EAF8"
REF_COLOR = "#BBBBBB"
ZERO_LINE_COLOR = "#555555"

SCATTER_ALPHA = 0.55
VIOLIN_ALPHA = 0.70
CI_ALPHA = 0.25
ELLIPSE_ALPHA = 0.14
LINE_WIDTH = 1.0
MEAN_LINE_WIDTH = 1.0
REF_LINE_WIDTH = 0.7
SPINE_LINE_WIDTH = 0.8

SCATTER_SIZE = 3
BROOD_SCATTER_SIZE = 3
EGG_SCATTER_SIZE = 3
MEAN_MARKER_SIZE = 9

RNG = np.random.default_rng(2026)

# Outer-panel / inner-plot-area layout.
# Each outer panel is the full subplot cell; each inner axis is the actual plotting area.
INNER_BOUNDS = [0.17, 0.18, 0.76, 0.72]
INNER_BOUNDS_WIDE = [0.10, 0.16, 0.86, 0.76]

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# General helpers
# ============================================================

def resolve_output_path(path_like):
    path = Path(path_like)
    supported_suffixes = {
        ".eps", ".jpeg", ".jpg", ".pdf", ".pgf", ".png", ".ps",
        ".raw", ".rgba", ".svg", ".svgz", ".tif", ".tiff", ".webp"
    }
    if path.exists() and path.is_dir():
        path = path / "comprehensive_analysis.png"
    elif path.suffix == "":
        path = path.with_suffix(".png")
    elif path.suffix.lower() not in supported_suffixes:
        print(f"[Warning] Unsupported output suffix: {path.suffix}; switching to .png automatically")
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_outer_panel(fig, gridspec_item, label, inner_bounds=INNER_BOUNDS):
    """
    Create an outer subplot cell and an inner plotting area.
    Panel label is placed in the outer cell, not in the plotting area.
    """
    outer_ax = fig.add_subplot(gridspec_item)
    outer_ax.set_xticks([])
    outer_ax.set_yticks([])
    for spine in outer_ax.spines.values():
        spine.set_visible(False)

    outer_ax.text(
        0.03, 0.97, label,
        transform=outer_ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color="black",
        zorder=20
    )

    inner_ax = outer_ax.inset_axes(inner_bounds)
    return outer_ax, inner_ax


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=SPINE_LINE_WIDTH, length=3, direction="out")
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(SPINE_LINE_WIDTH)


def p_to_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bootstrap_ci(data, n_boot=1000, ci=95):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = RNG.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def resolve_stats_output_path(stats_path_like, figure_output_path):
    if stats_path_like is None:
        return figure_output_path.with_name(f"{figure_output_path.stem}_analysis_tables.xlsx")
    stats_path = Path(stats_path_like)
    if stats_path.exists() and stats_path.is_dir():
        stats_path = stats_path / f"{figure_output_path.stem}_analysis_tables.xlsx"
    elif stats_path.suffix == "":
        stats_path = stats_path.with_suffix(".xlsx")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    return stats_path


def describe_numeric(data, prefix=""):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    ci_low, ci_high = bootstrap_ci(data) if n > 0 else (np.nan, np.nan)
    return {
        f"{prefix}n": n,
        f"{prefix}mean": np.mean(data) if n > 0 else np.nan,
        f"{prefix}sd": np.std(data, ddof=1) if n > 1 else np.nan,
        f"{prefix}sem": stats.sem(data, nan_policy="omit") if n > 1 else np.nan,
        f"{prefix}median": np.median(data) if n > 0 else np.nan,
        f"{prefix}min": np.min(data) if n > 0 else np.nan,
        f"{prefix}max": np.max(data) if n > 0 else np.nan,
        f"{prefix}bootstrap_ci95_low": ci_low,
        f"{prefix}bootstrap_ci95_high": ci_high,
    }


def adjust_p_values(p_values, method="holm"):
    """
    Adjust a family of p-values without adding statsmodels as a dependency.
    Supported methods: holm, bonferroni, fdr_bh, none.
    """
    method = (method or "none").lower()
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]
    m = len(valid_p)
    if m == 0:
        return adjusted

    if method in ["none", "raw", "uncorrected"]:
        adjusted[valid_mask] = valid_p
        return adjusted

    if method == "bonferroni":
        adjusted[valid_mask] = np.minimum(valid_p * m, 1.0)
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]

    if method == "holm":
        # Holm-Bonferroni adjusted p-values, monotone in sorted-p order.
        adjusted_sorted = (m - np.arange(m)) * ranked_p
        adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    elif method in ["fdr_bh", "bh", "benjamini-hochberg"]:
        # Benjamini-Hochberg FDR adjusted p-values.
        adjusted_sorted = ranked_p * m / (np.arange(m) + 1)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    else:
        raise ValueError(f"Unsupported p-value adjustment method: {method}")

    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = adjusted_sorted
    adjusted[valid_mask] = adjusted_valid
    return adjusted


def welch_ttest_with_df(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    denom = ((vx / nx) ** 2 / (nx - 1)) + ((vy / ny) ** 2 / (ny - 1))
    df = ((vx / nx + vy / ny) ** 2 / denom) if denom > 0 else np.nan
    return t_stat, df, p_value


def format_p_value(p):
    if not np.isfinite(p):
        return "NA"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def make_all_results_long(tables):
    rows = []
    for sheet_name, df in tables.items():
        if df is None or df.empty or sheet_name == "all_results_long":
            continue
        temp = df.copy()
        temp.insert(0, "source_sheet", sheet_name)
        rows.append(temp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def save_analysis_tables(tables, stats_output_path):
    """Save all calculated summaries and test results into one Excel workbook."""
    tables = {name: df for name, df in tables.items() if df is not None and not df.empty}
    if not tables:
        print("[Warning] No statistics tables to save.")
        return

    tables["all_results_long"] = make_all_results_long(tables)

    try:
        with pd.ExcelWriter(stats_output_path, engine="openpyxl") as writer:
            for sheet_name, df in tables.items():
                safe_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        print(f"Statistics tables output: {stats_output_path}")
    except Exception as e:
        # Fallback: save separate CSV files if Excel writing fails.
        fallback_dir = stats_output_path.with_suffix("")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        for sheet_name, df in tables.items():
            df.to_csv(fallback_dir / f"{sheet_name}.csv", index=False, encoding="utf-8-sig")
        print(f"[Warning] Failed to save Excel result tables: {e}")
        print(f"Now saving CSV result tables to folder instead: {fallback_dir}")


# ============================================================
# Panel E: Kaplan-Meier survival
# ============================================================

def read_excel_and_process_data(file_path):
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    processed_data = {}

    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        times = df.iloc[:, 0].values
        deaths = df.iloc[:, 1].values

        mask = ~(pd.isna(times) | pd.isna(deaths))
        times = times[mask]
        deaths = deaths[mask]

        individual_times = []
        individual_events = []
        for t, d in zip(times, deaths):
            for _ in range(int(d)):
                individual_times.append(t)
                individual_events.append(1)

        initial_n = int(sum(deaths))
        processed_data[sheet_name] = {
            "times": np.array(individual_times),
            "events": np.array(individual_events),
            "initial_n": initial_n,
            "original_df": df
        }

    return processed_data


def plot_km_curves_from_excel(ax, file_path):
    data_dict = read_excel_and_process_data(file_path)
    kmf = KaplanMeierFitter()

    stats_text = ""
    summary_rows = []
    test_rows = []

    for idx, (strain_name, data) in enumerate(data_dict.items()):
        times = data["times"]
        events = data["events"]
        n_events = int(sum(events))
        is_n2 = strain_name.upper() == "N2"
        color = N2_COLOR if is_n2 else EXP_COLOR
        linestyle = "-" if is_n2 else "--"

        kmf.fit(times, event_observed=events, label=f"{strain_name} (n={n_events})")
        kmf.plot_survival_function(
            ax=ax,
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            ci_show=True,
            ci_alpha=CI_ALPHA
        )

        median_survival = kmf.median_survival_time_
        median_survival_for_table = median_survival if np.isfinite(median_survival) else np.nan

        print(f"\n{strain_name} strain statistics:")
        print(f"  - Total observed individuals: {data['initial_n']}")
        print(f"  - Number of events: {sum(events)}")
        print(f"  - Final survivors: {data['initial_n'] - sum(events)}")
        print(f"  - 50% survival time: {median_survival if np.isfinite(median_survival) else 'not reached'}")

        summary_rows.append({
            "panel": "E",
            "assay": "lifespan",
            "strain": strain_name,
            "initial_n": data["initial_n"],
            "event_n": int(sum(events)),
            "censored_n": int(data["initial_n"] - sum(events)),
            "median_survival_days": median_survival_for_table,
            "min_observed_days": np.min(times) if len(times) else np.nan,
            "max_observed_days": np.max(times) if len(times) else np.nan,
        })

    if len(data_dict) > 1:
        strain_names = list(data_dict.keys())
        for i in range(len(strain_names)):
            for j in range(i + 1, len(strain_names)):
                name1, name2 = strain_names[i], strain_names[j]
                times1, events1 = data_dict[name1]["times"], data_dict[name1]["events"]
                times2, events2 = data_dict[name2]["times"], data_dict[name2]["events"]

                logrank_results = logrank_test(times1, times2, events1, events2)
                logrank_p = logrank_results.p_value

                all_times = np.concatenate([times1, times2])
                all_events = np.concatenate([events1, events2])
                all_groups = np.concatenate([np.zeros(len(times1)), np.ones(len(times2))])
                event_times = np.unique(all_times[all_events == 1])

                W_stat = 0
                Var_W = 0
                for t in event_times:
                    at_risk_total = np.sum(all_times >= t)
                    at_risk_g1 = np.sum((all_times >= t) & (all_groups == 0))
                    at_risk_g2 = np.sum((all_times >= t) & (all_groups == 1))
                    events_at_t_mask = (all_times == t) & (all_events == 1)
                    events_g1_at_t = np.sum(events_at_t_mask & (all_groups == 0))
                    events_g2_at_t = np.sum(events_at_t_mask & (all_groups == 1))
                    total_events_at_t = events_g1_at_t + events_g2_at_t

                    if at_risk_total > 0:
                        exp_g1 = at_risk_g1 * total_events_at_t / at_risk_total
                        W_stat += (events_g1_at_t - exp_g1) * at_risk_g1
                        if at_risk_total > 1:
                            var_factor = (
                                at_risk_g1 * at_risk_g2 * total_events_at_t *
                                (at_risk_total - total_events_at_t)
                            ) / (at_risk_total ** 2 * (at_risk_total - 1))
                            Var_W += var_factor * (at_risk_g1 ** 2)

                if Var_W > 0:
                    Z_stat = W_stat / np.sqrt(Var_W)
                    wilcoxon_p = 2 * (1 - norm.cdf(abs(Z_stat)))
                else:
                    Z_stat = np.nan
                    wilcoxon_p = 1.0

                print(f"{name1} vs {name2}:")
                print(f"  - Log-rank test p-value = {logrank_p:.4f}")
                print(f"  - Wilcoxon test p-value = {wilcoxon_p:.4f}")

                logrank_p_str = format_p_value(logrank_p)
                wilcoxon_p_str = format_p_value(wilcoxon_p)
                stats_text += f"{name1} vs {name2}\nLog-rank p = {logrank_p_str}\nWilcoxon p = {wilcoxon_p_str}\n\n"

                test_rows.append({
                    "panel": "E",
                    "assay": "lifespan",
                    "comparison": f"{name1} vs {name2}",
                    "test": "Log-rank test",
                    "test_statistic": getattr(logrank_results, "test_statistic", np.nan),
                    "p_raw": logrank_p,
                    "significance": p_to_label(logrank_p),
                })
                test_rows.append({
                    "panel": "E",
                    "assay": "lifespan",
                    "comparison": f"{name1} vs {name2}",
                    "test": "Wilcoxon/Breslow-style weighted survival test",
                    "W_statistic": W_stat,
                    "Var_W": Var_W,
                    "Z_statistic": Z_stat,
                    "p_raw": wilcoxon_p,
                    "significance": p_to_label(wilcoxon_p),
                })

    if stats_text:
        ax.text(
            0.02, 0.04, stats_text.strip(),
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EEEEEE", edgecolor="none", alpha=0.75)
        )

    max_time = max([max(data["times"]) for data in data_dict.values()])
    ax.set_xlim(0, max_time + 1)
    ax.set_xticks(range(0, int(max_time + 2), max(1, int(max_time // 12) or 1)))
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.2, 0.2))

    ax.set_xlabel("Days of adulthood")
    ax.set_ylabel("Survival ratio")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False, loc="upper right")
    style_axis(ax)

    return {
        "lifespan_summary": pd.DataFrame(summary_rows),
        "lifespan_tests": pd.DataFrame(test_rows),
    }


# ============================================================
# Panels A-B: body length
# ============================================================

def plot_body_length_comparison(ax1, ax2, file_path, strain_ref="N2"):
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    if strain_ref not in sheets:
        raise ValueError("Please ensure that N2 exists as a sheet")
    strain_other = [s for s in sheets if s != strain_ref][0]

    df_ref = pd.read_excel(file_path, sheet_name=strain_ref)
    df_mut = pd.read_excel(file_path, sheet_name=strain_other)
    timepoints = df_ref.columns.tolist()

    data_ref = {t: pd.to_numeric(df_ref[t], errors="coerce").dropna().values for t in timepoints}
    data_mut = {t: pd.to_numeric(df_mut[t], errors="coerce").dropna().values for t in timepoints}

    means_ref, ci_ref = [], []
    means_mut, ci_mut = [], []
    p_values, t_values, df_values = [], [], []
    delta_mean, delta_ci = [], []
    summary_rows, test_rows = [], []

    for t in timepoints:
        x = data_ref[t]
        y = data_mut[t]
        m1, m2 = np.mean(x), np.mean(y)
        ci1 = bootstrap_ci(x)
        ci2 = bootstrap_ci(y)
        t_stat, df_welch, p = welch_ttest_with_df(x, y)

        means_ref.append(m1)
        means_mut.append(m2)
        ci_ref.append(ci1)
        ci_mut.append(ci2)
        t_values.append(t_stat)
        df_values.append(df_welch)
        p_values.append(p)

        boot_deltas = []
        for _ in range(1000):
            xb = RNG.choice(x, size=len(x), replace=True)
            yb = RNG.choice(y, size=len(y), replace=True)
            boot_deltas.append(np.mean(yb) - np.mean(xb))
        current_delta_mean = m2 - m1
        current_delta_ci = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))
        delta_mean.append(current_delta_mean)
        delta_ci.append(current_delta_ci)

        row_ref = {
            "panel": "A",
            "assay": "body_length",
            "timepoint": t,
            "strain": strain_ref,
            "unit": "um",
        }
        row_ref.update(describe_numeric(x))
        summary_rows.append(row_ref)

        row_mut = {
            "panel": "A",
            "assay": "body_length",
            "timepoint": t,
            "strain": strain_other,
            "unit": "um",
        }
        row_mut.update(describe_numeric(y))
        summary_rows.append(row_mut)

        test_rows.append({
            "panel": "A/B",
            "assay": "body_length",
            "timepoint": t,
            "comparison": f"{strain_other} vs {strain_ref}",
            "test": "Welch two-sample t-test",
            "p_adjust_family": "body_length_timepoints",
            "p_adjust_method": BODY_LENGTH_P_ADJUST_METHOD,
            "n_ref": len(x),
            "n_other": len(y),
            "mean_ref": m1,
            "mean_other": m2,
            "delta_mean_other_minus_ref": current_delta_mean,
            "delta_bootstrap_ci95_low": current_delta_ci[0],
            "delta_bootstrap_ci95_high": current_delta_ci[1],
            "t_statistic": t_stat,
            "df_welch": df_welch,
            "p_raw": p,
        })

    p_adj_values = adjust_p_values(p_values, method=BODY_LENGTH_P_ADJUST_METHOD)
    for row, p_adj in zip(test_rows, p_adj_values):
        row["p_adj"] = p_adj
        row["significance_raw"] = p_to_label(row["p_raw"]) if np.isfinite(row["p_raw"]) else "NA"
        row["significance_adj"] = p_to_label(p_adj) if np.isfinite(p_adj) else "NA"

    x_pos = np.arange(len(timepoints))
    for i, t in enumerate(timepoints):
        jitter = RNG.normal(0, 0.045, size=len(data_ref[t]))
        ax1.scatter(np.full(len(data_ref[t]), i) + jitter, data_ref[t], color=N2_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors="none")
        jitter = RNG.normal(0, 0.045, size=len(data_mut[t]))
        ax1.scatter(np.full(len(data_mut[t]), i) + jitter, data_mut[t], color=EXP_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors="none")

    ref_lower = [ci[0] for ci in ci_ref]
    ref_upper = [ci[1] for ci in ci_ref]
    mut_lower = [ci[0] for ci in ci_mut]
    mut_upper = [ci[1] for ci in ci_mut]

    ax1.plot(x_pos, means_ref, color=N2_COLOR, linewidth=MEAN_LINE_WIDTH, label=strain_ref)
    ax1.plot(x_pos, means_mut, color=EXP_COLOR, linewidth=MEAN_LINE_WIDTH, label=strain_other)
    ax1.fill_between(x_pos, ref_lower, ref_upper, color=N2_COLOR, alpha=CI_ALPHA, linewidth=0)
    ax1.fill_between(x_pos, mut_lower, mut_upper, color=EXP_COLOR, alpha=CI_ALPHA, linewidth=0)

    y_max = max(max(means_ref), max(means_mut), max([max(data_ref[t]) for t in timepoints]), max([max(data_mut[t]) for t in timepoints]))
    for i, (p_raw, p_adj) in enumerate(zip(p_values, p_adj_values)):
        print(f"{timepoints[i]}: raw p={p_raw}, {BODY_LENGTH_P_ADJUST_METHOD}-adjusted p={p_adj}")
        ax1.text(i, y_max * 1.04, p_to_label(p_adj), ha="center", va="center", fontsize=PVAL_FONT_SIZE)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(timepoints, rotation=0)
    ax1.set_ylabel("Body length (μm)")
    ax1.set_xlabel("Time post hatching")
    ax1.legend(frameon=False, loc="lower right", fontsize=LEGEND_FONT_SIZE)
    ax1.set_ylim(top=y_max * 1.12)
    style_axis(ax1)

    delta_lower = [ci[0] for ci in delta_ci]
    delta_upper = [ci[1] for ci in delta_ci]
    ax2.plot(x_pos, delta_mean, color=EXP_COLOR, linewidth=MEAN_LINE_WIDTH)
    ax2.fill_between(x_pos, delta_lower, delta_upper, color=EXP_COLOR, alpha=CI_ALPHA, linewidth=0)
    ax2.axhline(0, linestyle="--", color=ZERO_LINE_COLOR, linewidth=REF_LINE_WIDTH)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(timepoints)
    ax2.set_ylabel("Δ length (μm)")
    ax2.set_xlabel("Time post hatching")
    style_axis(ax2)

    return {
        "body_length_summary": pd.DataFrame(summary_rows),
        "body_length_tests": pd.DataFrame(test_rows),
    }


# ============================================================
# Panel D: brood size
# ============================================================

def plot_brood_size(ax, file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    group1_name, group2_name = sheet_names[0], sheet_names[1]

    df1 = pd.read_excel(xls, sheet_name=group1_name, usecols=[0], header=None)
    df2 = pd.read_excel(xls, sheet_name=group2_name, usecols=[0], header=None)
    group1 = pd.to_numeric(df1.iloc[:, 0], errors="coerce").dropna().values
    group2 = pd.to_numeric(df2.iloc[:, 0], errors="coerce").dropna().values

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
    median1 = np.median(group1)
    median2 = np.median(group2)
    effect_size = median2 - median1

    parts = ax.violinplot([group1, group2], positions=[0, 1], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], [N2_LIGHT, EXP_LIGHT]):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(VIOLIN_ALPHA)

    x1_jitter = RNG.normal(0, 0.055, size=len(group1))
    x2_jitter = RNG.normal(1, 0.055, size=len(group2))
    ax.scatter(x1_jitter, group1, color=N2_COLOR, alpha=SCATTER_ALPHA, s=BROOD_SCATTER_SIZE, edgecolors="none")
    ax.scatter(x2_jitter, group2, color=EXP_COLOR, alpha=SCATTER_ALPHA, s=BROOD_SCATTER_SIZE, edgecolors="none")
    ax.hlines(median1, -0.2, 0.2, linewidth=MEAN_LINE_WIDTH, color=N2_COLOR)
    ax.hlines(median2, 0.8, 1.2, linewidth=MEAN_LINE_WIDTH, color=EXP_COLOR)

    y_max = max(group1.max(), group2.max())
    y_min = min(group1.min(), group2.min())
    y_range = y_max - y_min
    line_y = y_max + y_range * 0.08
    ax.plot([0, 0, 1, 1], [line_y, line_y + y_range * 0.04, line_y + y_range * 0.04, line_y], lw=REF_LINE_WIDTH, color="black")
    ax.text(0.5, line_y + y_range * 0.06, f"p = {p_value:.3g}", ha="center", fontsize=ANNOTATION_FONT_SIZE)
    ax.text(0.5, y_min - y_range * 0.14, f"Δmedian = {effect_size:.1f}", ha="center", fontsize=ANNOTATION_FONT_SIZE)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{group1_name}\n(n={len(group1)})", f"{group2_name}\n(n={len(group2)})"])
    ax.set_ylabel("Brood size (eggs per worm)")
    ax.set_ylim(y_min - y_range * 0.22, y_max + y_range * 0.22)
    style_axis(ax)

    summary_rows = []
    row1 = {"panel": "D", "assay": "brood_size", "strain": group1_name, "unit": "eggs_per_worm"}
    row1.update(describe_numeric(group1))
    row1["q1"] = np.percentile(group1, 25)
    row1["q3"] = np.percentile(group1, 75)
    row2 = {"panel": "D", "assay": "brood_size", "strain": group2_name, "unit": "eggs_per_worm"}
    row2.update(describe_numeric(group2))
    row2["q1"] = np.percentile(group2, 25)
    row2["q3"] = np.percentile(group2, 75)
    summary_rows.extend([row1, row2])

    test_rows = [{
        "panel": "D",
        "assay": "brood_size",
        "comparison": f"{group2_name} vs {group1_name}",
        "test": "Student two-sample t-test; equal_var=True",
        "n_group1": len(group1),
        "n_group2": len(group2),
        "mean_group1": np.mean(group1),
        "mean_group2": np.mean(group2),
        "median_group1": median1,
        "median_group2": median2,
        "delta_median_group2_minus_group1": effect_size,
        "t_statistic": t_stat,
        "p_raw": p_value,
        "significance": p_to_label(p_value),
    }]

    return {
        "brood_size_summary": pd.DataFrame(summary_rows),
        "brood_size_test": pd.DataFrame(test_rows),
    }


# ============================================================
# Panel C: egg shape
# ============================================================

def confidence_ellipse(x, y, ax, n_std=1.96, edgecolor="black", facecolor="none", alpha=1.0, linewidth=1.0, linestyle="-"):
    cov = np.cov(x, y)
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or np.linalg.det(cov) <= 0:
        return None
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=theta, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(ellipse)
    return ellipse


def plot_egg_shape(ax, file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    group1_name, group2_name = sheet_names[0], sheet_names[1]

    df1 = pd.read_excel(xls, sheet_name=group1_name)
    df2 = pd.read_excel(xls, sheet_name=group2_name)
    long1 = pd.to_numeric(df1["Major"], errors="coerce").dropna()
    short1 = pd.to_numeric(df1["Minor"], errors="coerce").dropna()
    long2 = pd.to_numeric(df2["Major"], errors="coerce").dropna()
    short2 = pd.to_numeric(df2["Minor"], errors="coerce").dropna()

    ax.scatter(short1, long1, color=N2_COLOR, alpha=SCATTER_ALPHA, s=EGG_SCATTER_SIZE, label=f"{group1_name} (n={len(long1)})", edgecolors="none")
    ax.scatter(short2, long2, color=EXP_COLOR, alpha=SCATTER_ALPHA, s=EGG_SCATTER_SIZE, label=f"{group2_name} (n={len(long2)})", edgecolors="none")
    ax.scatter(np.mean(short1), np.mean(long1), color=N2_COLOR, s=MEAN_MARKER_SIZE, marker="x", zorder=3, linewidth=1.2)
    ax.scatter(np.mean(short2), np.mean(long2), color=EXP_COLOR, s=MEAN_MARKER_SIZE, marker="x", zorder=3, linewidth=1.2)

    confidence_ellipse(short1, long1, ax, edgecolor=N2_COLOR, facecolor=N2_COLOR, alpha=ELLIPSE_ALPHA, linewidth=1.0)
    confidence_ellipse(short2, long2, ax, edgecolor=EXP_COLOR, facecolor=EXP_COLOR, alpha=ELLIPSE_ALPHA, linewidth=1.0)

    min_val = min(short1.min(), short2.min(), long1.min(), long2.min())
    max_val = max(short1.max(), short2.max(), long1.max(), long2.max())
    ax.plot([0, max_val], [0, max_val], linestyle="--", color=REF_COLOR, linewidth=REF_LINE_WIDTH)
    ax.set_xlabel("Short axis length (μm)")
    ax.set_ylabel("Long axis length (μm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, max_val + 5, 5))
    ax.set_yticks(np.arange(0, max_val + 5, 5))
    ax.set_xlim(min_val - 2, max_val + 2)
    ax.set_ylim(min_val - 2, max_val + 2)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="lower right")
    style_axis(ax)

    t_major, p_major = stats.ttest_ind(long1, long2, equal_var=False)
    t_minor, p_minor = stats.ttest_ind(short1, short2, equal_var=False)
    print(f"Major axis p = {p_major:.3g}\nMinor axis p = {p_minor:.3g}")

    summary_rows = []
    for group_name, long_data, short_data in [
        (group1_name, long1, short1),
        (group2_name, long2, short2),
    ]:
        long_data = np.asarray(long_data, dtype=float)
        short_data = np.asarray(short_data, dtype=float)
        axis_ratio = long_data / short_data
        row = {
            "panel": "C",
            "assay": "egg_shape",
            "strain": group_name,
            "n_major": len(long_data),
            "n_minor": len(short_data),
            "mean_major_um": np.mean(long_data),
            "sd_major_um": np.std(long_data, ddof=1) if len(long_data) > 1 else np.nan,
            "sem_major_um": stats.sem(long_data) if len(long_data) > 1 else np.nan,
            "median_major_um": np.median(long_data),
            "mean_minor_um": np.mean(short_data),
            "sd_minor_um": np.std(short_data, ddof=1) if len(short_data) > 1 else np.nan,
            "sem_minor_um": stats.sem(short_data) if len(short_data) > 1 else np.nan,
            "median_minor_um": np.median(short_data),
            "mean_major_minor_ratio": np.mean(axis_ratio),
            "sd_major_minor_ratio": np.std(axis_ratio, ddof=1) if len(axis_ratio) > 1 else np.nan,
        }
        summary_rows.append(row)

    test_rows = [
        {
            "panel": "C",
            "assay": "egg_shape",
            "metric": "Major",
            "comparison": f"{group2_name} vs {group1_name}",
            "test": "Student two-sample t-test; equal_var=True",
            "n_group1": len(long1),
            "n_group2": len(long2),
            "mean_group1": np.mean(long1),
            "mean_group2": np.mean(long2),
            "delta_mean_group2_minus_group1": np.mean(long2) - np.mean(long1),
            "t_statistic": t_major,
            "p_raw": p_major,
            "significance": p_to_label(p_major),
        },
        {
            "panel": "C",
            "assay": "egg_shape",
            "metric": "Minor",
            "comparison": f"{group2_name} vs {group1_name}",
            "test": "Student two-sample t-test; equal_var=True",
            "n_group1": len(short1),
            "n_group2": len(short2),
            "mean_group1": np.mean(short1),
            "mean_group2": np.mean(short2),
            "delta_mean_group2_minus_group1": np.mean(short2) - np.mean(short1),
            "t_statistic": t_minor,
            "p_raw": p_minor,
            "significance": p_to_label(p_minor),
        },
    ]

    return {
        "egg_shape_summary": pd.DataFrame(summary_rows),
        "egg_shape_tests": pd.DataFrame(test_rows),
    }


# ============================================================
# Main composition
# ============================================================

def main():
    print("=== Integrated vitality figure ===")
    output_path = resolve_output_path(OUTPUT_PATH)
    stats_output_path = resolve_stats_output_path(STATS_OUTPUT_PATH, output_path)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.08], width_ratios=[1, 1], hspace=0, wspace=0)

    _, ax_a = make_outer_panel(fig, gs[0, 0], "A")
    _, ax_b = make_outer_panel(fig, gs[0, 1], "B")
    _, ax_c = make_outer_panel(fig, gs[1, 0], "C")
    _, ax_d = make_outer_panel(fig, gs[1, 1], "D")
    _, ax_e = make_outer_panel(fig, gs[2, :], "E", inner_bounds=INNER_BOUNDS_WIDE)

    analysis_tables = {}
    analysis_tables.update(plot_body_length_comparison(ax_a, ax_b, BODY_LENGTH_FILE))
    analysis_tables.update(plot_egg_shape(ax_c, EGG_SHAPE_FILE))
    analysis_tables.update(plot_brood_size(ax_d, BROOD_SIZE_FILE))
    analysis_tables.update(plot_km_curves_from_excel(ax_e, LONGEVITY_FILE))

    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.035, top=0.995, wspace=0, hspace=0)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    save_analysis_tables(analysis_tables, stats_output_path)

    plt.show()

    print("mission complete")
    print(f"PNG output: {output_path}")
    print(f"PDF output: {pdf_path}")
    print(f"SVG output: {svg_path}")
    print(f"Statistics tables output: {stats_output_path}")


if __name__ == "__main__":
    main()
