# -*- coding: utf-8 -*-
"""
Module 0: shared configuration and utility functions.

Inputs:
    Raw pixel CSV folders are configured in INPUT_GROUPS.

Outputs:
    OUTPUT_DIR/global_script_settings.csv

Depends on:
    No previous pipeline module.
"""

import os
import sys

# Force a non-interactive matplotlib backend before pyplot is imported anywhere.
os.environ["MPLBACKEND"] = "Agg"

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.font_manager as _fm
for _font_file in [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\arialbi.ttf",
]:
    if os.path.isfile(_font_file):
        _fm.fontManager.addfont(_font_file)
del _fm, _font_file

import matplotlib.pyplot as _plt
_plt.rcParams["pdf.fonttype"] = 42
_plt.rcParams["ps.fonttype"] = 42
_plt.rcParams["svg.fonttype"] = "none"
del _plt


# ============================================================
# User-adjustable settings
# ============================================================

# ---------- Input folders ----------
INPUT_GROUPS = [
    {
        "gene_id": "skn-1",
        "gene_label": "skn-1",
        "condition": "anesthesia",
        "condition_label": "Anesthesia only",
        "folder": r"path/to/your/data/skn-1_h_net",
    },
    {
        "gene_id": "skn-1",
        "gene_label": "skn-1",
        "condition": "stress",
        "condition_label": "Stress",
        "folder": r"path/to/your/data/skn-1_tBHP_h_net",
    },
    {
        "gene_id": "daf-16",
        "gene_label": "daf-16",
        "condition": "anesthesia",
        "condition_label": "Anesthesia only",
        "folder": r"path/to/your/data/daf-16_h_net",
    },
    {
        "gene_id": "daf-16",
        "gene_label": "daf-16",
        "condition": "stress",
        "condition_label": "Stress",
        "folder": r"path/to/your/data/daf-16_tBHP_h_net",
    },
]

# ---------- Output and file searching ----------
OUTPUT_DIR = Path(r"path/to/your/output/continuous")
CSV_PATTERN = "*_pixels_h_net.csv"
RECURSIVE_SEARCH = False
STRICT_FILENAME_CHECK = True

# ---------- Input columns ----------
S_COL = "s"
R_COL = "r"
H_COL = "h_net"
IS_SIGNAL_COL = "is_signal"
AUTO_FALLBACK_TO_R3D = False
R3D_COL = "r_3d"

# ---------- s-r binning ----------
S_BIN_SIZE = 0.001
R_BIN_SIZE = 0.01
S_RANGE = (0.0, 1.0)
R_RANGE = (0.0, 1.0)
FILTER_TO_SR_RANGE = True
CLIP_NEGATIVE_H_TO_ZERO = True

# ---------- Delta and group-level statistics ----------
PRIMARY_METRIC_FOR_DELTA = "bin_signal_sum_h"  # "bin_mean_h" or "bin_signal_sum_h"
MIN_F0_FOR_DFF0 = 1e-12
MAX_BASELINE_TIME_MIN = 16  # Exclude bins whose baseline time >= this threshold (None = no filtering)
MIN_N_FOR_GROUP_TEST = 3
ALPHA = 0.05
P_ADJUST_METHOD = "holm"
P_ADJUST_SCOPE = "within_bin"  # "within_bin" or "all_bin_time"

# ---------- Robustness filters ----------
POSITIVE_RATIO_THRESHOLD = 0.50
MAX_TOP2_CONTRIBUTION = 0.80
REQUIRE_LEAVE_ONE_OUT_POSITIVE = True

# ---------- Expanded local region analysis ----------
EXPANSION_BIN_RADIUS = 2
EXPANSION_TIME_RADIUS_MIN = 5

# ---------- Parallel processing ----------
USE_MULTIPROCESSING = True
MAX_WORKERS = 16
CSV_READ_WORKERS = MAX_WORKERS
GROUP_TEST_WORKERS = max(1, min(12, MAX_WORKERS))
GROUP_TEST_BIN_CHUNK_SIZE = 200
GROUP_TEST_MAX_PENDING_CHUNKS = GROUP_TEST_WORKERS
ROBUSTNESS_WORKERS = max(1, min(8, MAX_WORKERS))
PLOT_CURVE_WORKERS = max(1, min(4, MAX_WORKERS))
PLOT_CURVE_MAX_PENDING_TASKS = PLOT_CURVE_WORKERS
EXPANDED_REGION_WORKERS = max(1, min(4, MAX_WORKERS))
EXPANDED_REGION_MAX_PENDING_EVENTS = EXPANDED_REGION_WORKERS
PROGRESS_EVERY_N_FILES = 10
PROGRESS_EVERY_N_EVENTS = 20

# ---------- Saving ----------
SAVE_CSV_CHECK_TABLES = True
SAVE_COMPRESSED_BINARY = True
COMPRESSED_BINARY_SUFFIX = ".pkl.gz"
OVERWRITE_EXISTING = False
SAVE_PER_INSTANCE_BIN_STATS_CSV = False
SAVE_DELTA_VALUES_CSV = False
SAVE_BASELINE_VALUES_CSV = True

# ---------- Plotting ----------
MAX_ROBUST_BIN_CURVES_TO_DRAW = None
SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = False
CURVE_FIGSIZE = (3.5, 2.4)
CURVE_DPI = 600
CURVE_TITLE_FONTSIZE = 8
CURVE_LABEL_FONTSIZE = 8
CURVE_TICK_FONTSIZE = 8
CURVE_LEGEND_FONTSIZE = 6
CURVE_INDIVIDUAL_ALPHA = 0.28
CURVE_INDIVIDUAL_LINEWIDTH = 0.8
CURVE_MEAN_LINEWIDTH = 1.0
CURVE_CI_ALPHA = 0.25
CURVE_DELTA_COLOR = "#7FB3D5"
CURVE_DFF0_COLOR = "#88C9A9"
CURVE_INDIVIDUAL_COLOR = "#777777"
CURVE_SIGNIF_COLOR = "#222222"
CURVE_TRIANGLE_MARKER = "v"
CURVE_TRIANGLE_SIZE = 1
CURVE_SIG_LINEWIDTH = 0.7
CURVE_SIG_TEXT = "*"
CURVE_SIG_TEXT_FONTSIZE = 6
FONT_FAMILY = "Arial"

# ---------- Response timing and strength heatmaps ----------
RESPONSE_HEATMAP_OUTPUT_SUBDIR = "07_response_heatmaps"

NO_RESPONSE_COLOR = "#FFFFFF"

ONSET_CMAP_NAME = "onset_red_to_blue"
ONSET_CMAP_COLORS = [
            "#D73027",
            "#FDAE61",
            "#FEE08B",
            "#6BAED6",
            "#2171B5",
            "#08306B",
        ]
ONSET_VMIN = None
ONSET_VMAX = None

STRENGTH_CMAP_NAME = "strength_blue_to_red"
STRENGTH_CMAP_COLORS = ["#08306B", "#2171B5", "#6BAED6", "#FEE08B", "#FDAE61", "#D73027"]
STRENGTH_VMIN = None
STRENGTH_VMAX = None
STRENGTH_HEATMAP_LOG_SCALE = True  # Apply log10(offset+x) to strength heatmap colour scale
STRENGTH_LOG_OFFSET = 100.0  # Offset for log transform; larger = gentler compression of low values
STRENGTH_DFF0_LOG_OFFSET = 5.0  # Separate offset for the dff0 strength heatmap

RESPONSE_HEATMAP_FIGSIZE = (4.0, 3.2)
RESPONSE_HEATMAP_DPI = 600
RESPONSE_HEATMAP_TITLE_FONTSIZE = 8
RESPONSE_HEATMAP_LABEL_FONTSIZE = 8
RESPONSE_HEATMAP_TICK_FONTSIZE = 8
RESPONSE_HEATMAP_COLORBAR_FONTSIZE = 8
RESPONSE_HEATMAP_SPINE_LINEWIDTH = 0.8

RESPONSE_HEATMAP_SAVE_PNG = True
RESPONSE_HEATMAP_SAVE_PDF = True
RESPONSE_HEATMAP_SAVE_SVG = True

RESPONSE_HEATMAP_INTERPOLATION = "nearest"
RESPONSE_HEATMAP_ASPECT = "auto"

# Display-only dilation. Saved long tables and matrices remain unchanged.
RESPONSE_HEATMAP_VISUAL_DILATION_ENABLED = True
RESPONSE_HEATMAP_VISUAL_DILATION_S_RADIUS_BINS = 2
RESPONSE_HEATMAP_VISUAL_DILATION_R_RADIUS_BINS = 0
# Alpha for pixels added by the visual dilation (1.0 = opaque, 0.5 = 50% transparent).
# Measured bins are always drawn opaque on top, so only the dilated add-on is faded.
RESPONSE_HEATMAP_VISUAL_DILATION_ALPHA = 0.75

# ---------- Module 7 composite figures ----------
# Main figure: A-D line panels (1/3 width) + E-H onset heat maps (2/3 width).
RESPONSE_COMPOSITE_OUTPUT_BASENAME = "response_timing_composite"
RESPONSE_COMPOSITE_FIGSIZE = (7.0, 9.4)
RESPONSE_COMPOSITE_COLUMN_WIDTHS = (1.0 / 3.0, 2.0 / 3.0)
# Standalone strength figures: one 4-panel column, same panel width as the main onset column.
RESPONSE_STRENGTH_COMPOSITE_OUTPUT_BASENAME = "response_strength_composite"
RESPONSE_STRENGTH_DFF0_COMPOSITE_OUTPUT_BASENAME = "response_strength_composite_dff0"
RESPONSE_STRENGTH_FIGSIZE = (14.0 / 3.0, 9.4)  # 2/3 of the main figure width
RESPONSE_STRENGTH_COLUMN_WIDTHS = (1.0,)
RESPONSE_COMPOSITE_DPI = 600
RESPONSE_COMPOSITE_SAVE_PNG = True
RESPONSE_COMPOSITE_SAVE_PDF = True
RESPONSE_COMPOSITE_SAVE_SVG = True
RESPONSE_COMPOSITE_PANEL_LABEL_FONTSIZE = 12
RESPONSE_COMPOSITE_PANEL_TITLE_FONTSIZE = 8
RESPONSE_COMPOSITE_AXIS_LABEL_FONTSIZE = 8
RESPONSE_COMPOSITE_TICK_FONTSIZE = 8
RESPONSE_COMPOSITE_COLORBAR_FONTSIZE = 8
RESPONSE_COMPOSITE_PANEL_LABEL_X_FRACTION = 0.025
RESPONSE_COMPOSITE_PANEL_LABEL_Y_FRACTION = 0.035
RESPONSE_COMPOSITE_LINE_AX_RECT = (0.24, 0.15, 0.75, 0.73)
RESPONSE_COMPOSITE_HEATMAP_AX_RECT = (0.12, 0.15, 0.78, 0.73)
# One shared horizontal colour bar per heat-map column, drawn in the top band.
RESPONSE_COMPOSITE_COLORBAR_BAND = 0.062
RESPONSE_COMPOSITE_COLORBAR_BAR_HEIGHT = 0.012
RESPONSE_COMPOSITE_COLORBAR_BAR_PAD = 0.026
RESPONSE_COMPOSITE_CONDITION_COLORS = {
    "anesthesia": "#7F7F7F",
    "stress": "#7FB3D5",
}
RESPONSE_COMPOSITE_LINEWIDTH = 1.0
RESPONSE_COMPOSITE_CI_ALPHA = 0.25
RESPONSE_COMPOSITE_MARKER_SIZE = 0.4
RESPONSE_COMPOSITE_DELTAF_DISPLAY_SCALE = 1_000_000.0
RESPONSE_COMPOSITE_DELTAF_DISPLAY_SCALE_LABEL = "10^6"
RESPONSE_TIME_TEST_P_ADJUST_METHOD = "holm"

# ---------- Module 8: strength distribution figure ----------
STRENGTH_DIST_FIGSIZE = (15.0, 7.2)
STRENGTH_DIST_DPI = 600
STRENGTH_DIST_HIST_BINS = 50
STRENGTH_DIST_HIST_ALPHA = 0.50
STRENGTH_DIST_HIST_LINEWIDTH = 0.5
STRENGTH_DIST_SCATTER_SIZE = 3
STRENGTH_DIST_SCATTER_ALPHA = 0.35
STRENGTH_DIST_REF_LINEWIDTH = 1.8
STRENGTH_DIST_PANEL_LABEL_SIZE = 13
STRENGTH_DIST_TITLE_SIZE = 8
STRENGTH_DIST_AXIS_SIZE = 8
STRENGTH_DIST_TICK_SIZE = 8
STRENGTH_DIST_TEST_SIZE = 6
STRENGTH_DIST_LEGEND_SIZE = 6
STRENGTH_DIST_TIGHT_PAD = 1.5
STRENGTH_DIST_TIGHT_WPAD = 2.5
STRENGTH_DIST_TIGHT_HPAD = 2.5

# ---------- Stage directories ----------
BINNED_DATA_DIR = OUTPUT_DIR / "01_binned_data"
DELTA_VALUES_DIR = OUTPUT_DIR / "02_delta_values"
GROUP_TESTS_DIR = OUTPUT_DIR / "03_group_tests"
ROBUST_EVENTS_DIR = OUTPUT_DIR / "04_robust_events"
BIN_CURVES_DIR = OUTPUT_DIR / "05_bin_curves"
EXPANDED_REGIONS_DIR = OUTPUT_DIR / "06_expanded_regions"
RESPONSE_HEATMAP_DIR = OUTPUT_DIR / RESPONSE_HEATMAP_OUTPUT_SUBDIR
STRENGTH_DISTRIBUTION_DIR = OUTPUT_DIR / "08_strength_distribution"

INTERVAL_SEPARATOR = ";"
PRINT_PROGRESS = True
TIMESTAMP_TERMINAL_OUTPUT = True
TERMINAL_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
STOP_REQUEST_FILE = OUTPUT_DIR / "_STOP_PIPELINE_REQUESTED.txt"
STOP_CHECK_ENABLED = True
STOP_HOTKEY_ENABLED = True
STOP_HOTKEY = "q"
STOP_HOTKEY_HINT = "Press q in the running terminal to stop."


# ============================================================
# Derived bin constants
# ============================================================

def get_n_bins(value_range, bin_size):
    return int(round((value_range[1] - value_range[0]) / bin_size))


N_S_BINS = get_n_bins(S_RANGE, S_BIN_SIZE)
N_R_BINS = get_n_bins(R_RANGE, R_BIN_SIZE)
S_EDGES = S_RANGE[0] + np.arange(N_S_BINS + 1) * S_BIN_SIZE
R_EDGES = R_RANGE[0] + np.arange(N_R_BINS + 1) * R_BIN_SIZE
S_CENTERS = S_EDGES[:-1] + S_BIN_SIZE / 2
R_CENTERS = R_EDGES[:-1] + R_BIN_SIZE / 2


# ============================================================
# Utility functions
# ============================================================

class _TimestampedStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._line_start = True
        self._pipeline_timestamped = True

    def write(self, text):
        if text is None:
            return 0
        text = str(text)
        if text == "":
            return 0
        written = 0
        for chunk in text.splitlines(keepends=True):
            if self._line_start:
                timestamp = datetime.now().strftime(TERMINAL_TIMESTAMP_FORMAT)
                self._wrapped.write(f"[{timestamp}] ")
            self._wrapped.write(chunk)
            self._line_start = chunk.endswith("\n") or chunk.endswith("\r")
            written += len(chunk)
        return written

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _install_timestamped_terminal_output():
    if not TIMESTAMP_TERMINAL_OUTPUT:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if getattr(stream, "_pipeline_timestamped", False):
            continue
        setattr(sys, stream_name, _TimestampedStream(stream))


_install_timestamped_terminal_output()


def log(msg):
    if PRINT_PROGRESS:
        print(msg, flush=True)


class PipelineStopRequested(Exception):
    pass


def should_stop_requested() -> bool:
    return bool(STOP_CHECK_ENABLED and Path(STOP_REQUEST_FILE).exists())


def _poll_stop_hotkey() -> bool:
    if not STOP_HOTKEY_ENABLED:
        return False
    if not getattr(sys.stdin, "isatty", lambda: False)():
        return False
    try:
        if os.name == "nt":
            import msvcrt

            if not msvcrt.kbhit():
                return False
            char = msvcrt.getwch()
        else:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return False
            char = sys.stdin.read(1)
    except Exception:
        return False

    if str(char).lower() == str(STOP_HOTKEY).lower():
        path = write_stop_request(reason=f"hotkey {STOP_HOTKEY!r} pressed")
        log(f"[stop] Hotkey {STOP_HOTKEY!r} detected. Graceful stop requested: {path}")
        return True
    return False


def raise_if_stop_requested(context: str = ""):
    hotkey_requested = _poll_stop_hotkey()
    if hotkey_requested or should_stop_requested():
        suffix = f" during {context}" if context else ""
        raise PipelineStopRequested(
            f"Stop requested{suffix}. Remove {STOP_REQUEST_FILE} before resuming."
        )


def write_stop_request(reason: str = "manual stop request"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(STOP_REQUEST_FILE).write_text(
        f"requested_at={datetime.now().isoformat(timespec='seconds')}\nreason={reason}\n",
        encoding="utf-8",
    )
    return Path(STOP_REQUEST_FILE)


def clear_stop_request():
    path = Path(STOP_REQUEST_FILE)
    if path.exists():
        path.unlink()
        return True
    return False


def safe_filename(text: str, max_len: int = 180) -> str:
    text = str(text)
    text = text.replace("|", "__")
    text = re.sub(r"[\\/:*?\"<>\s]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def parse_filename(filename: str):
    """
    Parse instance and time from a pixel CSV filename.

    Compatible examples:
        skn-1_instance-0_t-021_pixels_h_net.csv
        skn-1_instance-0_t-021_pixels.csv
    """
    pattern = re.compile(
        r".*?_instance-(?P<instance>[^_]+)_t-(?P<time>\d+)_pixels(?:_h_net)?\.csv$",
        re.IGNORECASE,
    )
    match = pattern.match(filename)
    if not match:
        return None
    instance = match.group("instance")
    time_min = int(match.group("time"))
    return instance, time_min


def assign_bin_indices(values: np.ndarray, value_range, n_bins: int, bin_size: float):
    """
    Assign values to 0-based bin indices.
    Values equal to range upper bound are assigned to the last bin.
    """
    idx = np.floor((values - value_range[0]) / bin_size).astype(int)
    idx[idx == n_bins] = n_bins - 1
    return idx


def holm_bonferroni_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return adjusted

    ordered_idx = valid_idx[np.argsort(p[valid_idx])]
    m = len(ordered_idx)
    prev_adj = 0.0
    for rank, idx in enumerate(ordered_idx, start=1):
        adj = (m - rank + 1) * p[idx]
        adj = max(adj, prev_adj)
        adjusted[idx] = min(adj, 1.0)
        prev_adj = adjusted[idx]
    return adjusted


def adjust_p_values(p_values, method=None):
    method = (method or P_ADJUST_METHOD).lower()
    if method == "holm":
        return holm_bonferroni_adjust(p_values)
    if method in ["none", "raw", "no"]:
        return np.asarray(p_values, dtype=float)
    raise ValueError(f"Unsupported P_ADJUST_METHOD: {method}")


def compressed_path(path_base: Path) -> Path:
    path = Path(path_base)
    if str(path).endswith(COMPRESSED_BINARY_SUFFIX):
        return path
    return path.with_name(path.name + COMPRESSED_BINARY_SUFFIX)


def save_dataframe_compressed(df: pd.DataFrame, path_base: Path, label: str = "table"):
    if not SAVE_COMPRESSED_BINARY:
        return None
    path = compressed_path(path_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path, compression="gzip")
    log(f"    Saved {label}: {path}")
    return path


def read_dataframe_compressed(path_base: Path) -> pd.DataFrame:
    path = Path(path_base)
    if not path.exists():
        path = compressed_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Compressed dataframe not found: {path}")
    return pd.read_pickle(path, compression="gzip")


def compute_mean_ci(values, confidence: float = 0.95):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": n,
            "mean": mean,
            "sd": np.nan,
            "sem": np.nan,
            "ci_low": mean,
            "ci_high": mean,
        }
    sd = float(np.std(arr, ddof=1))
    sem = sd / math.sqrt(n)
    try:
        from scipy.stats import t as t_dist

        crit = float(t_dist.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:
        crit = 1.96
    half_width = crit * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def filter_late_baseline_bins(df: pd.DataFrame):
    """
    Exclude rows whose baseline_time_min >= MAX_BASELINE_TIME_MIN.

    Returns (filtered_df, n_removed).  If the column is missing or
    MAX_BASELINE_TIME_MIN is None the original DataFrame is returned unchanged.
    """
    threshold = MAX_BASELINE_TIME_MIN
    if threshold is None:
        return df, 0
    if "baseline_time_min" not in df.columns:
        return df, 0
    baseline = pd.to_numeric(df["baseline_time_min"], errors="coerce")
    mask = baseline < float(threshold)
    n_before = len(df)
    filtered = df.loc[mask].copy() if not mask.all() else df
    n_removed = n_before - len(filtered)
    if n_removed > 0:
        log(f"    Excluded {n_removed} late-baseline rows (baseline_time_min >= {threshold})")
    return filtered, n_removed


def format_time_value(t):
    if pd.isna(t):
        return "NA"
    t = float(t)
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return f"{t:g}"


def significant_intervals_from_times(sig_times, all_times):
    sig_times = np.asarray(sig_times, dtype=float)
    sig_times = sig_times[np.isfinite(sig_times)]
    if len(sig_times) == 0:
        return []

    all_times = np.asarray(all_times, dtype=float)
    all_times = np.sort(np.unique(all_times[np.isfinite(all_times)]))
    if len(all_times) >= 2:
        diffs = np.diff(all_times)
        diffs = diffs[diffs > 0]
        step = float(np.min(diffs)) if len(diffs) > 0 else 1.0
    else:
        step = 1.0

    sig_times = np.sort(np.unique(sig_times))
    intervals = []
    start = sig_times[0]
    prev = sig_times[0]
    for t in sig_times[1:]:
        if t - prev <= step * 1.5:
            prev = t
        else:
            intervals.append((start, prev))
            start = t
            prev = t
    intervals.append((start, prev))
    return intervals


def format_intervals(intervals):
    if not intervals:
        return "0"
    parts = []
    for start, end in intervals:
        if abs(float(start) - float(end)) < 1e-9:
            parts.append(format_time_value(start))
        else:
            parts.append(f"{format_time_value(start)}-{format_time_value(end)}")
    return INTERVAL_SEPARATOR.join(parts)


def group_folder_name(group: dict) -> str:
    return safe_filename(f"{group['gene_id']}__{group['condition']}")


def ensure_stage_dirs():
    for path in [
        OUTPUT_DIR,
        BINNED_DATA_DIR,
        DELTA_VALUES_DIR,
        GROUP_TESTS_DIR,
        ROBUST_EVENTS_DIR,
        BIN_CURVES_DIR,
        EXPANDED_REGIONS_DIR,
        RESPONSE_HEATMAP_DIR,
        STRENGTH_DISTRIBUTION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _setting_value_to_string(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return value


def write_global_settings(module_name: str = None):
    ensure_stage_dirs()
    rows = []
    for name, value in sorted(globals().items()):
        if not name.isupper():
            continue
        if name in {"NP", "PD"}:
            continue
        rows.append(
            {
                "parameter": name,
                "value": _setting_value_to_string(value),
                "module": module_name or "",
                "written_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    for i, group in enumerate(INPUT_GROUPS, start=1):
        for key, value in group.items():
            rows.append(
                {
                    "parameter": f"INPUT_GROUPS[{i}].{key}",
                    "value": value,
                    "module": module_name or "",
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
    settings_df = pd.DataFrame(rows)
    settings_df.to_csv(OUTPUT_DIR / "global_script_settings.csv", index=False, encoding="utf-8-sig")
    return settings_df


def main():
    write_global_settings(module_name="00_config_utils")
    log(f"Wrote settings to: {OUTPUT_DIR / 'global_script_settings.csv'}")


if __name__ == "__main__":
    main()










