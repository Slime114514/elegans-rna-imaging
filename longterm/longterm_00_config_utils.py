# -*- coding: utf-8 -*-
"""
Long-term analysis shared configuration and utility functions.

Inputs:
    Raw pixel CSV folders are configured in DAY_FOLDERS.
"""

import os
import sys
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# User-adjustable settings
# ============================================================

DAY_FOLDERS = {
    1:  r"path/to/your/data/col-10_day1_output_h_net",
    4:  r"path/to/your/data/col-10_day4_output_h_net",
    7:  r"path/to/your/data/col-10_day7_output_h_net",
    10: r"path/to/your/data/col-10_day10_output_h_net",
    13: r"path/to/your/data/col-10_day13_output_h_net",
    16: r"path/to/your/data/col-10_day16_output_h_net",
}

OUTPUT_DIR = Path(r"path/to/your/output/longterm")
OUTPUT_BASENAME = "lifespan_3x4_combined"

CSV_GLOB = "*.csv"
CHUNKSIZE = 300_000
RANDOM_SEED = 20260509

# ---------- Parallel processing ----------
USE_MULTIPROCESSING = True
MAX_WORKERS = 16
PROGRESS_EVERY_N_FILES = 10

# ---------- 1D A-P profile binning for panel D ----------
AP_BIN_WIDTH = 0.001
AP_N_BINS = int(1 / AP_BIN_WIDTH)

# ---------- 2D s-r heatmap binning ----------
S_BIN_WIDTH = 0.001
R_BIN_WIDTH = 0.01
S_MIN, S_MAX = 0.0, 1.0
R_MIN, R_MAX = 0.0, 1.0
N_S_BINS = int((S_MAX - S_MIN) / S_BIN_WIDTH)
N_R_BINS = int((R_MAX - R_MIN) / R_BIN_WIDTH)

TARGET_HEATMAP_DAYS = [1, 10, 16]
TOP_FRACTION = 0.05

# ---------- Heatmap display ranges ----------
TOP5_PROB_VMIN = 0.0
TOP5_PROB_VMAX = 1.0
MEAN_HNET_VMIN = 0
MEAN_HNET_VMAX = 20000
DIFF_VABS = 6000
DIFF_PERCENTILE_IF_AUTO = 98

# ---------- Saving ----------
SAVE_INTERMEDIATE_TABLES = True
OVERWRITE_EXISTING = False

# ---------- Stage directories ----------
COMPUTED_DATA_DIR  = OUTPUT_DIR / "01_computed_data"
PLOT_OUTPUT_DIR    = OUTPUT_DIR / "02_figures"

# ---------- Stop mechanism ----------
STOP_REQUEST_FILE = OUTPUT_DIR / "_STOP_PIPELINE_REQUESTED.txt"
STOP_CHECK_ENABLED = True
STOP_HOTKEY_ENABLED = True
STOP_HOTKEY = "q"


# ============================================================
# Plotting style
# ============================================================

MAIN_COLOR = "#7FB3D5"
POINT_COLOR = "#A9CCE3"
CI_COLOR = "#D6EAF8"

POINT_SIZE = 3
POINT_ALPHA = 0.75
LINE_WIDTH = 1.0
LINE_WIDTH_IMPORTANT = 1.8
CI_ALPHA = 0.25
JITTER_WIDTH = 0.12

PANEL_LABEL_SIZE = 9
TITLE_SIZE = 7
AXIS_LABEL_SIZE = 7
TICK_LABEL_SIZE = 6
LEGEND_SIZE = 6
CBAR_LABEL_SIZE = 6
CBAR_TICK_SIZE = 6
FIGSIZE = (7.0, 8.0)
FIG_DPI = 600

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": AXIS_LABEL_SIZE,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Stop mechanism
# ============================================================

class PipelineStopRequested(Exception):
    pass


def should_stop_requested():
    return bool(STOP_CHECK_ENABLED and Path(STOP_REQUEST_FILE).exists())


def _poll_stop_hotkey():
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
        print(f"[stop] Hotkey {STOP_HOTKEY!r} detected. Graceful stop requested: {path}", flush=True)
        return True
    return False


def raise_if_stop_requested(context=""):
    hotkey_requested = _poll_stop_hotkey()
    if hotkey_requested or should_stop_requested():
        suffix = f" during {context}" if context else ""
        raise PipelineStopRequested(
            f"Stop requested{suffix}. Remove {STOP_REQUEST_FILE} before resuming."
        )


def write_stop_request(reason="manual stop request"):
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


# ============================================================
# Utility functions
# ============================================================

def log(msg):
    print(msg, flush=True)


def normalize_colname(name):
    return str(name).strip().strip('"').strip("'")


def get_column_mapping(csv_path, required_cols):
    header = pd.read_csv(csv_path, nrows=0)
    original_cols = list(header.columns)
    normalized_map = {normalize_colname(c): c for c in original_cols}
    missing = [c for c in required_cols if c not in normalized_map]
    if missing:
        raise ValueError(
            f"{csv_path} missing required columns: {missing}\n"
            f"Current columns: {original_cols}"
        )
    return {c: normalized_map[c] for c in required_cols}


def mean_and_95ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return np.mean(arr), 0.0, 1
    mean = np.mean(arr)
    sem = stats.sem(arr, nan_policy="omit")
    ci = stats.t.ppf(0.975, df=n - 1) * sem
    return mean, ci, n


def p_to_stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def variance_p_to_stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def adjust_pvalues(p_values, method="holm"):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    if method == "holm":
        adjusted_ranked = (m - np.arange(m)) * ranked
        adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
        adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    elif method == "bh":
        adjusted_ranked = ranked * m / np.arange(1, m + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
        adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")
    q_valid = np.full_like(pv, np.nan, dtype=float)
    q_valid[order] = adjusted_ranked
    q[valid] = q_valid
    return q


def bh_adjust(p_values):
    return adjust_pvalues(p_values, method="bh")


def ensure_dirs():
    for path in [OUTPUT_DIR, COMPUTED_DATA_DIR, PLOT_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_compressed_path(stem):
    path = Path(stem)
    return path.with_name(path.name + ".pkl.gz")


def save_compressed(df, stem, desc=""):
    path = get_compressed_path(stem)
    df.to_pickle(path, compression="gzip")
    if desc:
        log(f"    Saved {desc}: {path}")


def read_compressed(stem):
    path = get_compressed_path(stem)
    if not path.exists():
        alt = Path(stem)
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Compressed data not found: {path}")
    return pd.read_pickle(path, compression="gzip")
