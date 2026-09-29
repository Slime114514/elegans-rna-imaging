import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
from tifffile import imread

try:
    from scipy import stats
except ImportError:
    stats = None


# ============================================================
# User paths
# ============================================================

NEG_FOLDER = r"path/to/your/negative_control_pixels"
EXP_FOLDER = r"path/to/your/experimental_pixels"
OUTPUT_PATH = r"path/to/your/output/negative_control_3x3_tstack.png"

# Target timepoints for each group
NEG_TARGET_TIME = "t-000"
EXP_TARGET_TIME = "t-010"


# ============================================================
# Global style parameters
# ============================================================

DPI = 600
FIGSIZE = (7.0, 7.0)

FONT_FAMILY = "Arial"
LABEL_FONT_SIZE = 7
TICK_FONT_SIZE = 6
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 9

NEG_COLOR = "#7F7F7F"
EXP_COLOR = "#7FB3D5"
REF_COLOR = "#BBBBBB"

LINE_WIDTH_MEAN = 1.0
LINE_WIDTH_REF = 0.7
LINE_WIDTH_MARK = 0.7
CI_ALPHA = 0.25

INDIVIDUAL_ALPHA = 0.25
INDIVIDUAL_LINEWIDTH = 0.45

POINT_SIZE = 3
POINT_ALPHA = 0.85
MEAN_POINT_SIZE = 9
NEG_FILL_ALPHA = 0.16
EXP_FILL_ALPHA = 0.20

# Image-panel style
MASK_COLOR = np.array([1.0, 0.0, 0.6], dtype=np.float32)  # magenta
MASK_ALPHA = 0.38

# Display stretching. BF and CF are separated because their dynamic ranges differ.
BF_LOW_PCTL = 0.5
BF_HIGH_PCTL = 99.9
CF_LOW_PCTL = 0.5
CF_HIGH_PCTL = 99.95
ZOOM_LOW_PCTL = 0.5
ZOOM_HIGH_PCTL = 99.95

# Optional mask thickening. 0 means native single-pixel mask. 1 means 3x3 thickening.
MASK_DILATE_RADIUS = 1

# CF zoom rectangle in E panel, using original image coordinates.
ZOOM_RECT_X1 = 648
ZOOM_RECT_Y1 = 580
ZOOM_RECT_X2 = 727
ZOOM_RECT_Y2 = 657
ZOOM_RECT_COLOR = "red"
ZOOM_RECT_LINEWIDTH = 1.0

# Inset location in E: [left, bottom, width, height] in axes fraction.
ZOOM_INSET_BOUNDS = [0.0, 0.52, 0.48, 0.48]
ZOOM_BORDER_COLOR = "white"
ZOOM_BORDER_LINEWIDTH = 0.7

# CSV reading
CHUNK_SIZE = 500_000

# Parallel processing
USE_MULTIPROCESSING = True
MAX_WORKERS = 16

# G panel parameters
BIN_WIDTH = 256
H_MIN = 0
H_MAX = 65535
G_USE_LOG_X = True
G_USE_LOG_Y = True
PLOT_EPS = 1e-8
SHOW_NEG_P99_IN_G = True

# H panel parameters
N_GRID = 1001
X_MARK = 0.05

# I panel parameters
CHI2_95_DF2 = 5.991464547
I_USE_LOG_Y = True


# ============================================================
# General helpers
# ============================================================

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


def resolve_output_path(path_like):
    path = Path(path_like)
    supported_suffixes = {
        ".eps", ".jpeg", ".jpg", ".pdf", ".pgf", ".png", ".ps",
        ".raw", ".rgba", ".svg", ".svgz", ".tif", ".tiff", ".webp"
    }

    if path.exists() and path.is_dir():
        path = path / "negative_control_3x3.png"
    elif path.suffix == "":
        path = path.with_suffix(".png")
    elif path.suffix.lower() not in supported_suffixes:
        print(f"[Warning] Unsupported output suffix: {path.suffix}; switching to .png automatically")
        path = path.with_suffix(".png")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_column(columns, target_name):
    normalized = {str(c).strip().lower(): c for c in columns}
    key = target_name.strip().lower()
    if key not in normalized:
        raise ValueError(f"Column not found: {target_name}. Current column names: {list(columns)}")
    return normalized[key]


def update_online_mean_ci(n, mean, m2, new_values):
    if mean is None:
        mean = np.zeros_like(new_values, dtype=np.float64)
        m2 = np.zeros_like(new_values, dtype=np.float64)

    n += 1
    delta = new_values - mean
    mean += delta / n
    delta2 = new_values - mean
    m2 += delta * delta2
    return n, mean, m2


def _parallel_group_process(csv_tasks, worker_fn, label=""):
    """Process CSV tasks via worker_fn, returning (results, skipped, total).

    csv_tasks can be a list of plain Paths or a list of tuples (Path, ...).
    """
    if not USE_MULTIPROCESSING or len(csv_tasks) <= 1 or MAX_WORKERS < 2:
        return None, None, None

    # Normalise: extract csv_path for logging, keep task for submission
    tasks = []
    for t in csv_tasks:
        csv_path = t[0] if isinstance(t, (tuple, list)) else t
        tasks.append((csv_path, t))

    workers = min(MAX_WORKERS, len(tasks))
    results = []
    skipped = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, t): csv_path for csv_path, t in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            csv_path = futures[future]
            try:
                res = future.result()
            except Exception as exc:
                skipped.append((str(csv_path.name) if hasattr(csv_path, 'name') else str(csv_path), str(exc)))
                continue
            if res is None:
                skipped.append((str(csv_path.name) if hasattr(csv_path, 'name') else str(csv_path), "no valid data"))
                continue
            results.append(res)
            if i == 1 or i == len(tasks) or i % 20 == 0:
                print(f"  [{label}] Finished {i}/{len(tasks)}", flush=True)
    return results, skipped, len(tasks)


def _worker_distribution(args):
    csv_path, bin_edges, group_name = args
    try:
        proportion, valid_pixel_count, h_values = calculate_h_distribution(csv_path, bin_edges)
    except Exception:
        return None
    if proportion is None:
        return None
    print(f"[G-{group_name}] {csv_path.name} | pixels={valid_pixel_count}")
    return {"proportion": proportion, "valid_pixel_count": valid_pixel_count, "h_values": h_values}


def _worker_cumulative(args):
    csv_path, x_grid, group_name = args
    try:
        h_values = read_t_h_values(csv_path)
    except Exception:
        return None
    if h_values is None or len(h_values) == 0:
        return None
    x, y = compute_cumulative_curve(h_values)
    if x is None or y is None:
        return None
    y_interp = np.interp(x_grid, x, y)
    top_contrib = float(np.interp(X_MARK, x, y))
    return {"y_interp": y_interp, "curve": (x, y), "n_pixels": len(h_values),
            "top_contrib": top_contrib, "file": csv_path.name}


def _worker_scatter(csv_path):
    sr, mh, n_total, n_sig = calculate_signal_ratio_and_mean_h(csv_path)
    return {"file": csv_path.name, "signal_ratio": sr, "mean_h_signal": mh,
            "n_total_rows": n_total, "n_signal_rows": n_sig}


def style_stat_axis(ax):
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=0.8, length=3)
    ax.xaxis.label.set_size(LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(LABEL_FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def make_inner_plot_axes(outer_ax):
    """
    Create an inner plotting area inside the outer subplot cell.
    This keeps titles, tick labels, axis labels, legends, and panel labels
    inside the subplot cell rather than spilling outside.
    """
    outer_ax.set_xticks([])
    outer_ax.set_yticks([])
    for spine in outer_ax.spines.values():
        spine.set_visible(False)
    # [left, bottom, width, height] in outer-axes fraction
    inner_ax = outer_ax.inset_axes([0.18, 0.18, 0.76, 0.72])
    return inner_ax


def add_panel_label(ax, label, panel_type="image"):
    """
    Panel labels share the same upper-left anchor in the subplot cell.
    For image panels the label sits on top of the image.
    For plot panels the label sits in the outer panel area, not inside the
    inner plotting region.
    """
    if panel_type == "image":
        ax.text(
            0.03, 0.97, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            color="white",
            zorder=20
        )
    else:
        ax.text(
            0.03, 0.97, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            color="black",
            zorder=20
        )


# ============================================================
# A-F image-panel functions
# ============================================================


def normalize_channel_name(channel):
    ch = str(channel).strip().lower()
    if ch == "bf":
        return "BF"
    if ch in ("cf", "cf488"):
        return "CF"
    if ch in ("zoom", "cf488_zoom"):
        return "zoom"
    raise ValueError(f"Unknown channel name: {channel}")


def parse_tif_name(tif_path):
    name = Path(tif_path).stem
    m = re.match(
        r"(.+?)_t-(\d+).*_(CF|BF|zoom)$",
        name,
        flags=re.IGNORECASE
    )
    if not m:
        raise ValueError(
            f"File name does not match the expected pattern: {Path(tif_path).name}\n"
            f"Expected something like: example_t-010_BF.tif or example_t-010_CF.tif"
        )
    instance_name = m.group(1)
    t_value = int(m.group(2))
    channel = normalize_channel_name(m.group(3))
    return instance_name, t_value, channel


def list_tif_files(folder):
    folder = Path(folder)
    files = []
    for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        files.extend(folder.glob(pattern))
    unique_files = {}
    for p in files:
        unique_files[str(p.resolve()).lower()] = p
    return sorted(unique_files.values())


def auto_find_example_instance_and_t(folder, target_time="t-010"):
    folder = Path(folder)
    target_t = _parse_t_value(target_time)
    tif_files = list_tif_files(folder)
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No tif/tiff files found in {folder}.")

    grouped = {}
    for tif_path in tif_files:
        try:
            instance_name, t_value, channel = parse_tif_name(tif_path)
        except ValueError:
            continue
        key = (instance_name, t_value)
        grouped.setdefault(key, set()).add(channel)

    complete_candidates = []
    preferred_candidates = []
    for (instance_name, t_value), channels in grouped.items():
        if "BF" in channels and "CF" in channels:
            complete_candidates.append((instance_name, t_value))
            if t_value == target_t:
                preferred_candidates.append((instance_name, t_value))

    if len(complete_candidates) == 0:
        print("\n[Debug info] Recognizable tif files in the current folder:")
        for tif_path in tif_files[:20]:
            try:
                inst, t_val, ch = parse_tif_name(tif_path)
                print(f"  {tif_path.name} -> instance={inst}, t=t-{t_val:03d}, channel={ch}")
            except ValueError:
                print(f"  Unrecognized: {tif_path.name}")
        raise FileNotFoundError(
            f"No tif file found in {folder} that contains both BF and CF for the same instance at the same t."
        )

    if len(preferred_candidates) > 0:
        candidates = sorted(preferred_candidates, key=lambda x: (x[0], x[1]))
    else:
        candidates = sorted(complete_candidates, key=lambda x: (x[0], x[1]))

    if len(candidates) > 1:
        print(f"\n[Note] Multiple usable instances found in {folder}; using the first one by default:")
        for i, (inst, t_val) in enumerate(candidates[:10], start=1):
            print(f"  {i}. instance = {inst}, t = t-{t_val:03d}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more candidates")

    selected_instance, selected_t = candidates[0]
    print(f"\n[Auto-selected] {folder}")
    print(f"  instance = {selected_instance}")
    print(f"  t = t-{selected_t:03d}")
    return selected_instance, selected_t


def read_tif_preserve_rgb(tif_path):
    """
    Read a TIFF image while preserving RGB/RGBA data when present.

    Return:
    - arr: either 2D grayscale-like array or H,W,3 RGB array
    - original_shape information printed for debugging

    This avoids forcing exported RGB/RGBA TIFFs into grayscale.
    """
    name = Path(tif_path).name
    arr = np.asarray(imread(tif_path))
    print(f"[Debug] {name} original shape = {arr.shape}, dtype = {arr.dtype}")

    arr = np.squeeze(arr)
    print(f"[Debug] {name} squeezed shape = {arr.shape}")

    # Already 2D grayscale/intensity image.
    if arr.ndim == 2:
        return arr

    # H,W,3 or H,W,4: keep RGB, drop alpha if present.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
        print(f"[Debug] {name} preserved as H,W,RGB: {rgb.shape}")
        return rgb

    # 3,H,W or 4,H,W: move channels to last dimension and keep RGB.
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        rgb = np.moveaxis(arr[:3, ...], 0, -1)
        print(f"[Debug] {name} preserved as C,H,W -> H,W,RGB: {rgb.shape}")
        return rgb

    # If still 3D, assume a stack-like object and take the middle plane.
    # This fallback is only for true stacks, not RGB images.
    if arr.ndim == 3:
        mid = (arr.shape[0] - 1) // 2
        plane = arr[mid, :, :]
        print(f"[Warning] {name} is 3D stack after squeeze; using middle plane index {mid}: {plane.shape}")
        return plane

    raise ValueError(f"{name} cannot be parsed as a 2D image or an RGB image, shape = {arr.shape}")


def rgb_to_unit_float(rgb):
    """
    Convert an RGB image to a Matplotlib-compatible float RGB image in [0,1]
    without changing its RGB channel relationship.
    """
    arr = np.asarray(rgb)

    if np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        if max_value <= 0:
            max_value = 255
        out = arr.astype(np.float32) / max_value
    else:
        out = arr.astype(np.float32)
        if np.nanmax(out) > 1.0:
            # Most exported display RGB images are 0-255 floats if not integer.
            out = out / np.nanmax(out)

    return np.clip(out, 0, 1)


def make_display_image(img, channel, low_pctl, high_pctl):
    """
    Prepare image for display.

    - If img is RGB/RGBA-derived, preserve RGB instead of converting to gray.
    - If img is 2D:
        BF is shown as grayscale.
        CF / zoom are shown as pure green pseudocolor.
    """
    channel = normalize_channel_name(channel)

    if img.ndim == 3 and img.shape[-1] == 3:
        return rgb_to_unit_float(img)

    if img.ndim != 2:
        raise ValueError(f"Display image must be 2D or H,W,3 RGB; current shape = {img.shape}")

    gray = normalize_to_gray(img, low_pctl, high_pctl)
    if channel in ("CF", "zoom"):
        return gray_to_green_rgb(gray)
    return gray


def normalize_to_gray(img, low_pctl, high_pctl):
    img = img.astype(np.float32)
    lo = np.percentile(img, low_pctl)
    hi = np.percentile(img, high_pctl)
    if hi <= lo:
        hi = lo + 1.0
    img = (img - lo) / (hi - lo)
    return np.clip(img, 0, 1)


def gray_to_green_rgb(gray_img):
    zeros = np.zeros_like(gray_img)
    return np.stack([zeros, gray_img, zeros], axis=-1)


def find_matching_tif(folder, instance_name, t_value, channel):
    folder = Path(folder)
    target_channel = normalize_channel_name(channel)
    matches = []

    for tif_path in list_tif_files(folder):
        try:
            inst, t_val, ch = parse_tif_name(tif_path)
        except ValueError:
            continue
        if inst == instance_name and t_val == t_value and ch == target_channel:
            matches.append(tif_path)

    unique_matches = {}
    for p in matches:
        unique_matches[str(p.resolve()).lower()] = p
    matches = sorted(unique_matches.values())

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No matching tif found: instance={instance_name}, t=t-{t_value:03d}, channel={target_channel}"
        )
    if len(matches) > 1:
        print("[Warning] Multiple distinct matching files found:")
        for p in matches:
            print(f"  - {p.name}")
        print(f"Using the first one: {matches[0].name}")
    return matches[0]


def find_optional_matching_tif(folder, instance_name, t_value, channel):
    try:
        return find_matching_tif(folder, instance_name, t_value, channel)
    except FileNotFoundError:
        return None


def _parse_t_value(target_time):
    """Extract integer t value from a time string like 't-010'."""
    m = re.match(r"t-(\d+)", str(target_time), re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse timepoint: {target_time}; expected format t-XXX")


def find_matching_csv(folder, instance_name, target_time=None):
    folder = Path(folder)
    all_csv = sorted(folder.glob("*.csv"))
    if target_time is not None:
        t_str = str(target_time)
        all_csv = [p for p in all_csv if t_str in p.stem]
    contains = [p for p in all_csv if instance_name in p.stem]
    if len(contains) == 0:
        raise FileNotFoundError(f"No csv file whose name contains '{instance_name}' was found in {folder}.")

    startswith = [p for p in contains if p.stem.startswith(instance_name)]
    if len(startswith) > 0:
        if len(startswith) > 1:
            print(f"[Warning] Instance {instance_name} matches multiple csv files; using: {startswith[0].name}")
        return startswith[0]

    if len(contains) > 1:
        print(f"[Warning] Instance {instance_name} matches multiple csv files; using: {contains[0].name}")
    return contains[0]


def read_signal_pixels(csv_path):
    header = pd.read_csv(csv_path, nrows=0)
    col_map = {str(c).strip().lower(): c for c in header.columns}
    for col in ["x", "y", "is_signal"]:
        if col not in col_map:
            raise ValueError(f"{csv_path.name} is missing column: {col}")

    x_col = col_map["x"]
    y_col = col_map["y"]
    signal_col = col_map["is_signal"]
    out = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=[x_col, y_col, signal_col],
        chunksize=CHUNK_SIZE
    ):
        sub = chunk.copy()
        sub[signal_col] = pd.to_numeric(sub[signal_col], errors="coerce")
        sub = sub.loc[sub[signal_col] == 1].copy()
        if len(sub) == 0:
            continue

        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna(subset=[x_col, y_col])
        if len(sub) > 0:
            out.append(sub[[x_col, y_col]].rename(columns={x_col: "x", y_col: "y"}))

    if len(out) == 0:
        return pd.DataFrame(columns=["x", "y"])
    return pd.concat(out, ignore_index=True)


def coords_to_pixel_indices(signal_df, image_shape):
    h, w = image_shape[:2]
    if len(signal_df) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    x = signal_df["x"].to_numpy()
    y = signal_df["y"].to_numpy()
    x_round = np.rint(x).astype(int)
    y_round = np.rint(y).astype(int)

    direct_ok = (
        np.all(np.abs(x - x_round) < 1e-6) and
        np.all(np.abs(y - y_round) < 1e-6) and
        np.all((x_round >= 0) & (x_round < w)) and
        np.all((y_round >= 0) & (y_round < h))
    )
    if direct_ok:
        return y_round, x_round

    unique_x = np.sort(signal_df["x"].unique())
    unique_y = np.sort(signal_df["y"].unique())

    if len(unique_x) == w:
        x_to_idx = {xv: i for i, xv in enumerate(unique_x)}
    else:
        x_idx = np.linspace(0, w - 1, max(len(unique_x), 1))
        x_to_idx = {xv: int(round(idx)) for xv, idx in zip(unique_x, x_idx)}

    if len(unique_y) == h:
        y_to_idx = {yv: i for i, yv in enumerate(unique_y)}
    else:
        y_idx = np.linspace(0, h - 1, max(len(unique_y), 1))
        y_to_idx = {yv: int(round(idx)) for yv, idx in zip(unique_y, y_idx)}

    cols = signal_df["x"].map(x_to_idx).to_numpy(dtype=int)
    rows = signal_df["y"].map(y_to_idx).to_numpy(dtype=int)
    return np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)


def dilate_boolean_mask(mask, radius):
    if radius <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    ys, xs = np.where(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            xx = np.clip(xs + dx, 0, w - 1)
            out[yy, xx] = True
    return out


def make_signal_mask_rgba(signal_df, image_shape):
    h, w = image_shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    if len(signal_df) == 0:
        return rgba

    rows, cols = coords_to_pixel_indices(signal_df, image_shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[rows, cols] = True
    mask = dilate_boolean_mask(mask, MASK_DILATE_RADIUS)

    rgba[mask, 0] = MASK_COLOR[0]
    rgba[mask, 1] = MASK_COLOR[1]
    rgba[mask, 2] = MASK_COLOR[2]
    rgba[mask, 3] = MASK_ALPHA
    return rgba


def prepare_group_images(folder, instance_name, t_value, target_time=None):
    folder = Path(folder)
    bf_path = find_matching_tif(folder, instance_name, t_value, "BF")
    cf_path = find_matching_tif(folder, instance_name, t_value, "CF")
    zoom_path = find_optional_matching_tif(folder, instance_name, t_value, "zoom")
    csv_path = find_matching_csv(folder, instance_name, target_time=target_time)

    # Read TIFFs while preserving RGB/RGBA display images.
    bf_img = read_tif_preserve_rgb(bf_path)
    cf_img = read_tif_preserve_rgb(cf_path)

    if bf_img.ndim not in (2, 3) or cf_img.ndim not in (2, 3):
        raise ValueError(
            f"Cannot parse image: {bf_path.name} shape={bf_img.shape} / "
            f"{cf_path.name} shape={cf_img.shape}"
        )

    signal_df = read_signal_pixels(csv_path)

    print(f"[prepare] {folder.name} | instance={instance_name} | t=t-{t_value:03d}")
    print(f"  CF shape: {cf_img.shape} | CSV: {csv_path.name} | signal pixels: {len(signal_df)}")
    if len(signal_df) > 0:
        print(f"  signal x range: [{signal_df['x'].min():.0f}, {signal_df['x'].max():.0f}]"
              f"  y range: [{signal_df['y'].min():.0f}, {signal_df['y'].max():.0f}]")

    # Display images:
    # - RGB/RGBA TIFFs keep their RGB colors.
    # - 2D BF stays grayscale.
    # - 2D CF becomes pure green pseudocolor.
    bf_show = make_display_image(bf_img, "BF", BF_LOW_PCTL, BF_HIGH_PCTL)
    cf_show = make_display_image(cf_img, "CF", CF_LOW_PCTL, CF_HIGH_PCTL)

    # Mask size follows the CF image height/width, whether CF is 2D or RGB.
    mask_rgba = make_signal_mask_rgba(signal_df, cf_img.shape[:2])

    zoom_show = None
    if zoom_path is not None:
        zoom_img = read_tif_preserve_rgb(zoom_path)
        zoom_show = make_display_image(zoom_img, "zoom", ZOOM_LOW_PCTL, ZOOM_HIGH_PCTL)

    return {
        "bf": bf_show,
        "cf": cf_show,
        "mask_rgba": mask_rgba,
        "zoom": zoom_show,
        "bf_path": bf_path,
        "cf_path": cf_path,
        "zoom_path": zoom_path,
        "csv_path": csv_path,
        "n_signal_pixels": len(signal_df)
    }


def add_zoom_inset(ax, zoom_img):
    if zoom_img is None:
        print("[Note] zoom.tif not found; skipping the zoom inset for panel E.")
        return
    inset_ax = ax.inset_axes(ZOOM_INSET_BOUNDS)
    inset_ax.imshow(zoom_img, interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ZOOM_BORDER_COLOR)
        spine.set_linewidth(ZOOM_BORDER_LINEWIDTH)


def add_zoom_rectangle(ax):
    rect = Rectangle(
        (ZOOM_RECT_X1, ZOOM_RECT_Y1),
        ZOOM_RECT_X2 - ZOOM_RECT_X1,
        ZOOM_RECT_Y2 - ZOOM_RECT_Y1,
        fill=False,
        edgecolor=ZOOM_RECT_COLOR,
        linewidth=ZOOM_RECT_LINEWIDTH
    )
    ax.add_patch(rect)


# ============================================================
# G panel: pixel intensity distribution
# ============================================================


def calculate_h_distribution(csv_path, bin_edges):
    header = pd.read_csv(csv_path, nrows=0)
    h_col = find_column(header.columns, "h")

    counts_total = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    valid_pixel_count = 0
    h_values_list = []

    for chunk in pd.read_csv(csv_path, usecols=[h_col], chunksize=CHUNK_SIZE):
        sub = chunk[h_col]
        sub = pd.to_numeric(sub, errors="coerce")
        sub = sub[(sub >= H_MIN) & (sub <= H_MAX)].dropna()
        if len(sub) == 0:
            continue

        arr = sub.to_numpy()
        counts, _ = np.histogram(arr, bins=bin_edges)
        counts_total += counts
        valid_pixel_count += len(arr)
        h_values_list.append(arr)

    if valid_pixel_count == 0:
        return None, valid_pixel_count, None

    proportion = counts_total / valid_pixel_count
    h_values = np.concatenate(h_values_list) if h_values_list else None
    return proportion, valid_pixel_count, h_values


def process_distribution_group(folder, group_name, bin_edges, target_time=None):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if target_time is not None:
        t_str = str(target_time)
        csv_files = [p for p in csv_files if t_str in p.stem]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No matching .csv files found in {folder}.")

    skipped_files = []

    # Build tasks with explicit args (closures can't pickle on Windows)
    tasks = [(p, bin_edges, group_name) for p in csv_files]
    results, skipped, total = _parallel_group_process(tasks, _worker_distribution, f"G-{group_name}")
    if results is not None:
        # Aggregate parallel results
        skipped_files.extend(skipped)
        proportions_parts = []
        all_h_values = []
        for r in results:
            proportions_parts.append(r["proportion"])
            if r["h_values"] is not None and len(r["h_values"]) > 0:
                all_h_values.append(r["h_values"])
        if not proportions_parts:
            raise RuntimeError(f"G panel: {group_name} did not successfully process any files.")
        stacked = np.stack(proportions_parts)
        n = len(proportions_parts)
        mean = np.mean(stacked, axis=0)
        if n > 1:
            variance = np.var(stacked, axis=0, ddof=1)
            sem = np.sqrt(variance) / np.sqrt(n)
            ci95 = 1.96 * sem
        else:
            ci95 = np.zeros_like(mean)
        all_h_values = np.concatenate(all_h_values) if all_h_values else np.array([])
    else:
        # Serial fallback
        n = 0
        mean = None
        m2 = None
        all_h_values = []
        for csv_path in csv_files:
            try:
                proportion, valid_pixel_count, h_values = calculate_h_distribution(csv_path, bin_edges)
                if proportion is None:
                    skipped_files.append((csv_path.name, "no valid h pixels"))
                    continue
                n, mean, m2 = update_online_mean_ci(n, mean, m2, proportion)
                if h_values is not None and len(h_values) > 0:
                    all_h_values.append(h_values)
                print(f"[G-{group_name}] {csv_path.name} | pixels={valid_pixel_count}")
            except Exception as e:
                skipped_files.append((csv_path.name, str(e)))
                print(f"[G-skipped] {csv_path.name}: {e}")
        if n == 0:
            raise RuntimeError(f"G panel: {group_name} did not successfully process any files.")
        if n > 1:
            variance = m2 / (n - 1)
            sem = np.sqrt(variance) / np.sqrt(n)
            ci95 = 1.96 * sem
        else:
            ci95 = np.zeros_like(mean)
        all_h_values = np.concatenate(all_h_values) if all_h_values else np.array([])

    return {"n": n, "mean": mean, "ci95": ci95,
            "all_h_values": all_h_values, "skipped_files": skipped_files}




def calculate_signal_above_threshold_for_file(csv_path, threshold):
    """
    For one CSV file, calculate how many is_signal == 1 pixels
    have h greater than the negative-control 99th percentile.

    Important:
    - Uses column "h" only.
    - Does not use h_raw or h_net.
    - Denominator is the number of valid is_signal == 1 pixels in the file.
    """
    header = pd.read_csv(csv_path, nrows=0)
    h_col = find_column(header.columns, "h")
    is_signal_col = find_column(header.columns, "is_signal")

    n_total_rows = 0
    n_signal_rows = 0
    n_valid_signal_h = 0
    n_signal_h_above_threshold = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=[h_col, is_signal_col],
        chunksize=CHUNK_SIZE
    ):
        sub = chunk.copy()
        n_total_rows += len(sub)

        sub[is_signal_col] = pd.to_numeric(sub[is_signal_col], errors="coerce")
        signal_sub = sub.loc[sub[is_signal_col] == 1].copy()
        n_signal_rows += len(signal_sub)

        if len(signal_sub) == 0:
            continue

        signal_h = pd.to_numeric(signal_sub[h_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(signal_h) == 0:
            continue

        n_valid_signal_h += len(signal_h)
        n_signal_h_above_threshold += int(np.sum(signal_h > threshold))

    fraction_above = (
        n_signal_h_above_threshold / n_valid_signal_h
        if n_valid_signal_h > 0 else np.nan
    )

    return {
        "file": csv_path.name,
        "n_total_rows": n_total_rows,
        "n_signal_rows": n_signal_rows,
        "n_valid_signal_h": n_valid_signal_h,
        "n_signal_h_above_threshold": n_signal_h_above_threshold,
        "fraction_signal_h_above_threshold": fraction_above,
        "percent_signal_h_above_threshold": fraction_above * 100 if np.isfinite(fraction_above) else np.nan
    }


def export_g_distribution_data_csv(neg_result, exp_result, x_values, bin_edges, neg_p99, exp_folder, output_csv_path, exp_target_time=None):
    """
    Export an additional CSV for G panel data without changing any existing outputs.

    The CSV includes:
    1. Mean pixel-fraction distribution and 95% CI for every intensity bin in G;
    2. The negative-control 99th percentile threshold;
    3. Per-instance and pooled proportions of experimental is_signal == 1 pixels
       with h greater than the negative-control 99th percentile threshold.
    """
    rows = []

    for group_label, result in [("Control", neg_result), ("daf-16", exp_result)]:
        mean = result["mean"]
        ci95 = result["ci95"]
        lower = np.maximum(mean - ci95, 0)
        upper = mean + ci95

        for i in range(len(x_values)):
            rows.append({
                "section": "g_bin_distribution",
                "group": group_label,
                "file": "GROUP_MEAN",
                "target_t": "",
                "bin_index": i,
                "bin_left": bin_edges[i],
                "bin_right": bin_edges[i + 1],
                "bin_center": x_values[i],
                "mean_pixel_fraction": mean[i],
                "ci95": ci95[i],
                "ci95_low": lower[i],
                "ci95_high": upper[i],
                "negative_control_h_99th_percentile": neg_p99,
                "threshold_source": "pooled Control t-000 h values used in G panel",
                "n_files": result["n"],
                "n_pooled_h_values_for_p99": len(neg_result["all_h_values"]),
                "n_total_rows": "",
                "n_signal_rows": "",
                "n_valid_signal_h": "",
                "n_signal_h_above_threshold": "",
                "fraction_signal_h_above_threshold": "",
                "percent_signal_h_above_threshold": "",
                "skip_reason": ""
            })

    rows.append({
        "section": "negative_p99_summary",
        "group": "Control",
        "file": "ALL_CONTROL_H_VALUES_POOLED",
        "target_t": "",
        "bin_index": "",
        "bin_left": "",
        "bin_right": "",
        "bin_center": "",
        "mean_pixel_fraction": "",
        "ci95": "",
        "ci95_low": "",
        "ci95_high": "",
        "negative_control_h_99th_percentile": neg_p99,
        "threshold_source": "pooled Control t-000 h values used in G panel",
        "n_files": neg_result["n"],
        "n_pooled_h_values_for_p99": len(neg_result["all_h_values"]),
        "n_total_rows": "",
        "n_signal_rows": "",
        "n_valid_signal_h": "",
        "n_signal_h_above_threshold": "",
        "fraction_signal_h_above_threshold": "",
        "percent_signal_h_above_threshold": "",
        "skip_reason": ""
    })

    exp_folder = Path(exp_folder)
    exp_csv_files = sorted(exp_folder.glob("*.csv"))
    if exp_target_time is not None:
        t_str = str(exp_target_time)
        exp_csv_files = [p for p in exp_csv_files if t_str in p.stem]
    exp_threshold_records = []
    skipped_files = []

    for csv_path in exp_csv_files:
        try:
            rec = calculate_signal_above_threshold_for_file(csv_path, neg_p99)
            exp_threshold_records.append(rec)
            rows.append({
                "section": "experimental_signal_positive_above_negative_p99_per_instance",
                "group": "daf-16",
                "file": rec["file"],
                "target_t": "",
                "bin_index": "",
                "bin_left": "",
                "bin_right": "",
                "bin_center": "",
                "mean_pixel_fraction": "",
                "ci95": "",
                "ci95_low": "",
                "ci95_high": "",
                "negative_control_h_99th_percentile": neg_p99,
                "threshold_source": "pooled Control t-000 h values used in G panel",
                "n_files": "",
                "n_pooled_h_values_for_p99": len(neg_result["all_h_values"]),
                "n_total_rows": rec["n_total_rows"],
                "n_signal_rows": rec["n_signal_rows"],
                "n_valid_signal_h": rec["n_valid_signal_h"],
                "n_signal_h_above_threshold": rec["n_signal_h_above_threshold"],
                "fraction_signal_h_above_threshold": rec["fraction_signal_h_above_threshold"],
                "percent_signal_h_above_threshold": rec["percent_signal_h_above_threshold"],
                "skip_reason": ""
            })
            print(
                f"[G-threshold-daf-16] {csv_path.name} | "
                f"signal h > Control p99: {rec['n_signal_h_above_threshold']}/"
                f"{rec['n_valid_signal_h']} "
                f"({rec['percent_signal_h_above_threshold']:.3f}%)"
            )
        except Exception as e:
            skipped_files.append((csv_path.name, str(e)))
            print(f"[G-threshold-daf-16 skipped] {csv_path.name}: {e}")

    total_signal_h = int(sum(r["n_valid_signal_h"] for r in exp_threshold_records))
    total_signal_above = int(sum(r["n_signal_h_above_threshold"] for r in exp_threshold_records))
    pooled_fraction = total_signal_above / total_signal_h if total_signal_h > 0 else np.nan

    rows.append({
        "section": "experimental_signal_positive_above_negative_p99_pooled_summary",
        "group": "daf-16",
        "file": "ALL_EXPERIMENTAL_SIGNAL_POSITIVE_H_VALUES_POOLED",
        "target_t": "",
        "bin_index": "",
        "bin_left": "",
        "bin_right": "",
        "bin_center": "",
        "mean_pixel_fraction": "",
        "ci95": "",
        "ci95_low": "",
        "ci95_high": "",
        "negative_control_h_99th_percentile": neg_p99,
        "threshold_source": "pooled Control t-000 h values used in G panel",
        "n_files": len(exp_threshold_records),
        "n_pooled_h_values_for_p99": len(neg_result["all_h_values"]),
        "n_total_rows": "",
        "n_signal_rows": "",
        "n_valid_signal_h": total_signal_h,
        "n_signal_h_above_threshold": total_signal_above,
        "fraction_signal_h_above_threshold": pooled_fraction,
        "percent_signal_h_above_threshold": pooled_fraction * 100 if np.isfinite(pooled_fraction) else np.nan,
        "skip_reason": ""
    })

    for name, reason in skipped_files:
        rows.append({
            "section": "experimental_signal_positive_above_negative_p99_skipped_file",
            "group": "daf-16",
            "file": name,
            "target_t": "",
            "bin_index": "",
            "bin_left": "",
            "bin_right": "",
            "bin_center": "",
            "mean_pixel_fraction": "",
            "ci95": "",
            "ci95_low": "",
            "ci95_high": "",
            "negative_control_h_99th_percentile": neg_p99,
            "threshold_source": "pooled Control t-000 h values used in G panel",
            "n_files": "",
            "n_pooled_h_values_for_p99": len(neg_result["all_h_values"]),
            "n_total_rows": "",
            "n_signal_rows": "",
            "n_valid_signal_h": "",
            "n_signal_h_above_threshold": "",
            "fraction_signal_h_above_threshold": "",
            "percent_signal_h_above_threshold": "",
            "skip_reason": reason
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[G] G panel pixel distribution and negative-control 99% threshold statistics saved: {output_csv_path}")


def draw_g_panel(ax, neg_result, exp_result, x_values, neg_p99=None):
    def plot_group(result, color, label):
        mean = result["mean"]
        ci95 = result["ci95"]
        lower = np.maximum(mean - ci95, 0)
        upper = mean + ci95

        if G_USE_LOG_Y:
            mean_plot = np.maximum(mean, PLOT_EPS)
            lower_plot = np.maximum(lower, PLOT_EPS)
            upper_plot = np.maximum(upper, PLOT_EPS)
        else:
            mean_plot = mean
            lower_plot = lower
            upper_plot = upper

        ax.plot(x_values, mean_plot, color=color, linewidth=LINE_WIDTH_MEAN, label=f"{label}, n={result['n']}")
        ax.fill_between(x_values, lower_plot, upper_plot, color=color, alpha=CI_ALPHA, linewidth=0)

    plot_group(neg_result, NEG_COLOR, "Control")
    plot_group(exp_result, EXP_COLOR, "daf-16")

    if SHOW_NEG_P99_IN_G and neg_p99 is not None:
        # A log x axis cannot display a non-positive threshold.
        if not G_USE_LOG_X or neg_p99 > 0:
            ax.axvline(neg_p99, color=NEG_COLOR, linestyle="--", linewidth=LINE_WIDTH_MARK)

    combined_mean = neg_result["mean"] + exp_result["mean"]
    nonzero_bins = np.where(combined_mean > 0)[0]
    if len(nonzero_bins) > 0:
        x_pad = (x_values[1] - x_values[0]) * 2 if len(x_values) > 1 else 0
        x_lo = max(H_MIN, x_values[nonzero_bins[0]] - x_pad)
        x_hi = min(H_MAX, x_values[nonzero_bins[-1]] + x_pad)
    else:
        x_lo, x_hi = H_MIN, H_MAX

    if G_USE_LOG_X:
        ax.set_xscale("log")
        if x_lo <= 0:
            # Log axes cannot start at 0; use half of the first bin center.
            x_lo = max(x_values[0] * 0.5, PLOT_EPS)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Fluorescence intensity")
    ax.set_ylabel("Pixel fraction")
    if G_USE_LOG_Y:
        ax.set_yscale("log")
        ax.set_ylim(PLOT_EPS, None)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper right")
    style_stat_axis(ax)
    add_panel_label(ax, "G", panel_type="plot")


# ============================================================
# H panel: cumulative intensity contribution
# ============================================================


def read_t_h_values(csv_path):
    header = pd.read_csv(csv_path, nrows=0)
    h_col = find_column(header.columns, "h")
    h_values_list = []

    for chunk in pd.read_csv(csv_path, usecols=[h_col], chunksize=CHUNK_SIZE):
        sub = chunk[h_col]
        sub = pd.to_numeric(sub, errors="coerce")
        sub = sub[sub >= 0].dropna()
        if len(sub) > 0:
            h_values_list.append(sub.to_numpy())

    if not h_values_list:
        return None
    h_values = np.concatenate(h_values_list)
    if len(h_values) == 0:
        return None
    return h_values


def compute_cumulative_curve(h_values):
    if h_values is None or len(h_values) == 0:
        return None, None
    total_h = h_values.sum()
    if total_h <= 0:
        return None, None
    h_sorted = np.sort(h_values)[::-1]
    cum_h = np.cumsum(h_sorted)
    n = len(h_sorted)
    x = np.arange(1, n + 1) / n
    y = cum_h / total_h
    x = np.insert(x, 0, 0.0)
    y = np.insert(y, 0, 0.0)
    return x, y


def process_cumulative_group(folder, group_name, x_grid, target_time=None):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if target_time is not None:
        t_str = str(target_time)
        csv_files = [p for p in csv_files if t_str in p.stem]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No matching .csv files found in {folder}.")

    skipped_files = []

    tasks = [(p, x_grid, group_name) for p in csv_files]
    results, skipped, total = _parallel_group_process(tasks, _worker_cumulative, f"H-{group_name}")
    if results is not None:
        skipped_files.extend(skipped)
        top_fraction_records = []
        curves = []
        y_parts = []
        for r in results:
            y_parts.append(r["y_interp"])
            curves.append(r["curve"])
            top_fraction_records.append({
                "file": r["file"], "group": group_name,
                "n_pixels": r["n_pixels"], "top_pixel_fraction": X_MARK,
                "contribution_fraction": r["top_contrib"],
                "contribution_percent": r["top_contrib"] * 100,
            })
            print(f"[H-{group_name}] {r['file']} | "
                  f"pixels={r['n_pixels']} | top {X_MARK:.0%} contribution={r['top_contrib']:.4f}")
        if not y_parts:
            raise RuntimeError(f"H panel: {group_name} did not successfully process any files.")
        stacked = np.stack(y_parts)
        n = len(y_parts)
        mean = np.mean(stacked, axis=0)
        if n > 1:
            variance = np.var(stacked, axis=0, ddof=1)
            sem = np.sqrt(variance) / np.sqrt(n)
            ci95 = 1.96 * sem
        else:
            ci95 = np.zeros_like(mean)
        lower = np.maximum(mean - ci95, 0)
        upper = np.minimum(mean + ci95, 1)
        return {"n": n, "mean": mean, "ci95": ci95, "lower": lower, "upper": upper,
                "curves": curves, "skipped_files": skipped_files,
                "top_fraction_records": top_fraction_records}
    else:
        # serial fallback (unchanged)
        n = 0; mean = None; m2 = None
        curves = []; skipped_files = []; top_fraction_records = []
        for csv_path in csv_files:
            try:
                h_values = read_t_h_values(csv_path)
                if h_values is None or len(h_values) == 0:
                    skipped_files.append((csv_path.name, "no valid h>=0 pixels"))
                    continue
                x, y = compute_cumulative_curve(h_values)
                if x is None or y is None:
                    skipped_files.append((csv_path.name, "total h <= 0; cannot compute cumulative intensity fraction"))
                    continue
                y_interp = np.interp(x_grid, x, y)
                curves.append((x, y))
                n, mean, m2 = update_online_mean_ci(n, mean, m2, y_interp)
                top = float(np.interp(X_MARK, x, y))
                top_fraction_records.append({
                    "file": csv_path.name, "group": group_name,
                    "n_pixels": len(h_values), "top_pixel_fraction": X_MARK,
                    "contribution_fraction": top, "contribution_percent": top * 100})
                print(f"[H-{group_name}] {csv_path.name} | "
                      f"pixels={len(h_values)} | top {X_MARK:.0%} contribution={top:.4f}")
            except Exception as e:
                skipped_files.append((csv_path.name, str(e)))
                print(f"[H-skipped] {csv_path.name}: {e}")
        if n == 0:
            raise RuntimeError(f"H panel: {group_name} did not successfully process any files.")
        if n > 1:
            variance = m2 / (n - 1)
            sem = np.sqrt(variance) / np.sqrt(n)
            ci95 = 1.96 * sem
        else:
            ci95 = np.zeros_like(mean)
        lower = np.maximum(mean - ci95, 0)
        upper = np.minimum(mean + ci95, 1)
        return {"n": n, "mean": mean, "ci95": ci95, "lower": lower, "upper": upper,
                "curves": curves, "skipped_files": skipped_files,
                "top_fraction_records": top_fraction_records}


def summarize_1d_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan
        }

    mean = float(np.mean(values))
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))

    if n > 1:
        sd = float(np.std(values, ddof=1))
        sem = sd / np.sqrt(n)
        ci95_low = mean - 1.96 * sem
        ci95_high = mean + 1.96 * sem
    else:
        sd = np.nan
        sem = np.nan
        ci95_low = np.nan
        ci95_high = np.nan

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "median": median,
        "q1": q1,
        "q3": q3
    }


def welch_ttest_df(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n1 = len(a)
    n2 = len(b)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1_sq = np.var(a, ddof=1)
    s2_sq = np.var(b, ddof=1)
    v1 = s1_sq / n1
    v2 = s2_sq / n2
    denom = (v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)
    if denom == 0:
        return np.nan
    return (v1 + v2) ** 2 / denom


def cohens_d(values_control, values_exp):
    control = np.asarray(values_control, dtype=float)
    exp = np.asarray(values_exp, dtype=float)
    control = control[np.isfinite(control)]
    exp = exp[np.isfinite(exp)]

    n1 = len(control)
    n2 = len(exp)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1_sq = np.var(control, ddof=1)
    s2_sq = np.var(exp, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(exp) - np.mean(control)) / pooled_sd


def export_h_top_fraction_stats(neg_result, exp_result, output_csv_path):
    """
    Export per-instance top X_MARK pixel intensity contribution values and
    group-level statistics for H panel.

    This output is independent of the plotted figure: no p values are drawn on H.
    """
    neg_records = neg_result.get("top_fraction_records", [])
    exp_records = exp_result.get("top_fraction_records", [])

    neg_values = np.array([r["contribution_fraction"] for r in neg_records], dtype=float)
    exp_values = np.array([r["contribution_fraction"] for r in exp_records], dtype=float)

    neg_summary = summarize_1d_values(neg_values)
    exp_summary = summarize_1d_values(exp_values)

    rows = []
    common_empty = {
        "test": "",
        "statistic": "",
        "df": "",
        "p_value": "",
        "effect_size_cohens_d": "",
        "mean_difference_exp_minus_control": ""
    }

    for r in neg_records + exp_records:
        row = {
            "section": "per_instance",
            "group": r["group"],
            "file": r["file"],
            "target_t": "",
            "n_pixels": r["n_pixels"],
            "top_pixel_fraction": r["top_pixel_fraction"],
            "contribution_fraction": r["contribution_fraction"],
            "contribution_percent": r["contribution_percent"],
            "n": "",
            "mean": "",
            "sd": "",
            "sem": "",
            "ci95_low": "",
            "ci95_high": "",
            "median": "",
            "q1": "",
            "q3": "",
        }
        row.update(common_empty)
        rows.append(row)

    for group_name, summary in [("Control", neg_summary), ("daf-16", exp_summary)]:
        row = {
            "section": "summary",
            "group": group_name,
            "file": "",
            "target_t": "",
            "n_pixels": "",
            "top_pixel_fraction": X_MARK,
            "contribution_fraction": summary["mean"],
            "contribution_percent": summary["mean"] * 100 if np.isfinite(summary["mean"]) else np.nan,
            "n": summary["n"],
            "mean": summary["mean"],
            "sd": summary["sd"],
            "sem": summary["sem"],
            "ci95_low": summary["ci95_low"],
            "ci95_high": summary["ci95_high"],
            "median": summary["median"],
            "q1": summary["q1"],
            "q3": summary["q3"],
        }
        row.update(common_empty)
        rows.append(row)

    if len(neg_values) >= 2 and len(exp_values) >= 2:
        mean_diff = float(np.mean(exp_values) - np.mean(neg_values))
        d_value = cohens_d(neg_values, exp_values)

        if stats is not None:
            welch = stats.ttest_ind(exp_values, neg_values, equal_var=False, nan_policy="omit")
            welch_stat = float(welch.statistic)
            welch_p = float(welch.pvalue)
            try:
                mw = stats.mannwhitneyu(exp_values, neg_values, alternative="two-sided")
                mw_stat = float(mw.statistic)
                mw_p = float(mw.pvalue)
            except Exception as e:
                print(f"[Warning] Mann-Whitney U test failed: {e}")
                mw_stat = np.nan
                mw_p = np.nan
        else:
            print("[Warning] scipy is not installed; p-values from statistical tests in the H top-fraction CSV will be recorded as NaN.")
            welch_stat = np.nan
            welch_p = np.nan
            mw_stat = np.nan
            mw_p = np.nan

        rows.append({
            "section": "test",
            "group": "daf-16_vs_Control",
            "file": "",
            "target_t": "",
            "n_pixels": "",
            "top_pixel_fraction": X_MARK,
            "contribution_fraction": "",
            "contribution_percent": "",
            "n_control": len(neg_values),
            "n_experimental": len(exp_values),
            "mean_control": float(np.mean(neg_values)),
            "mean_experimental": float(np.mean(exp_values)),
            "mean_difference_exp_minus_control": mean_diff,
            "test": "Welch_t_test",
            "statistic": welch_stat,
            "df": welch_ttest_df(exp_values, neg_values),
            "p_value": welch_p,
            "effect_size_cohens_d": d_value
        })

        rows.append({
            "section": "test",
            "group": "daf-16_vs_Control",
            "file": "",
            "target_t": "",
            "n_pixels": "",
            "top_pixel_fraction": X_MARK,
            "contribution_fraction": "",
            "contribution_percent": "",
            "n_control": len(neg_values),
            "n_experimental": len(exp_values),
            "mean_control": float(np.mean(neg_values)),
            "mean_experimental": float(np.mean(exp_values)),
            "mean_difference_exp_minus_control": mean_diff,
            "test": "Mann_Whitney_U_test",
            "statistic": mw_stat,
            "df": "",
            "p_value": mw_p,
            "effect_size_cohens_d": ""
        })
    else:
        print("[Warning] H top-fraction statistical test failed: at least one group has fewer than 2 valid instances.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[H] top {X_MARK:.0%} pixel contribution statistics saved: {output_csv_path}")


def draw_h_panel(ax, neg_result, exp_result, x_grid):
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=LINE_WIDTH_REF, color=REF_COLOR)

    for x, y in neg_result["curves"]:
        ax.plot(x, y, color=NEG_COLOR, alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LINEWIDTH, rasterized=True)
    for x, y in exp_result["curves"]:
        ax.plot(x, y, color=EXP_COLOR, alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LINEWIDTH, rasterized=True)

    ax.plot(x_grid, neg_result["mean"], color=NEG_COLOR, linewidth=LINE_WIDTH_MEAN, label=f"Control, n={neg_result['n']}")
    ax.fill_between(x_grid, neg_result["lower"], neg_result["upper"], color=NEG_COLOR, alpha=CI_ALPHA, linewidth=0)

    ax.plot(x_grid, exp_result["mean"], color=EXP_COLOR, linewidth=LINE_WIDTH_MEAN, label=f"daf-16, n={exp_result['n']}")
    ax.fill_between(x_grid, exp_result["lower"], exp_result["upper"], color=EXP_COLOR, alpha=CI_ALPHA, linewidth=0)

    y_mark = np.interp(X_MARK, x_grid, exp_result["mean"])
    ax.axvline(x=X_MARK, ymin=0, ymax=y_mark, color=EXP_COLOR, linestyle="--", linewidth=LINE_WIDTH_MARK)
    ax.axhline(y=y_mark, xmin=0, xmax=X_MARK, color=EXP_COLOR, linestyle="--", linewidth=LINE_WIDTH_MARK)

    current_xticks = list(ax.get_xticks())
    current_yticks = [0, 0.25, 0.5, 0.75, 1.0]
    new_xticks = sorted(set(current_xticks + [X_MARK]))
    new_yticks = sorted(set(current_yticks + [y_mark]))
    ax.set_xticks(new_xticks)
    ax.set_yticks(new_yticks)
    ax.set_xticklabels([f"{tick:.2f}" if np.isclose(tick, X_MARK) else f"{tick:g}" for tick in new_xticks])
    ax.set_yticklabels([f"{tick:.2f}" if np.isclose(tick, y_mark) else f"{tick:g}" for tick in new_yticks])

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Pixel fraction")
    ax.set_ylabel("Cumulated intensity fraction")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="lower right")
    style_stat_axis(ax)
    add_panel_label(ax, "H", panel_type="plot")


# ============================================================
# I panel: signal-positive fraction vs signal mean h
# ============================================================


def calculate_signal_ratio_and_mean_h(csv_path):
    header = pd.read_csv(csv_path, nrows=0)
    h_col = find_column(header.columns, "h")
    is_signal_col = find_column(header.columns, "is_signal")

    n_total_rows = 0
    n_signal_rows = 0
    signal_h_sum = 0.0
    signal_h_count = 0

    for chunk in pd.read_csv(csv_path, usecols=[h_col, is_signal_col], chunksize=CHUNK_SIZE):
        sub = chunk.copy()
        n_total_rows += len(sub)

        sub[is_signal_col] = pd.to_numeric(sub[is_signal_col], errors="coerce")
        signal_sub = sub.loc[sub[is_signal_col] == 1]
        n_signal_rows += len(signal_sub)

        if len(signal_sub) > 0:
            signal_h = pd.to_numeric(signal_sub[h_col], errors="coerce").dropna()
            if len(signal_h) > 0:
                signal_h_sum += signal_h.sum()
                signal_h_count += len(signal_h)

    if n_total_rows == 0:
        raise ValueError(f"{csv_path.name} contains no data.")

    signal_ratio = n_signal_rows / n_total_rows
    mean_h_signal = 0.0 if signal_h_count == 0 else signal_h_sum / signal_h_count
    return signal_ratio, mean_h_signal, n_total_rows, n_signal_rows


def process_scatter_group(folder, group_name, target_time=None):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if target_time is not None:
        t_str = str(target_time)
        csv_files = [p for p in csv_files if t_str in p.stem]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No matching .csv files found in {folder}.")

    results, skipped, total = _parallel_group_process(csv_files, _worker_scatter, f"I-{group_name}")
    if results is not None:
        records = []
        for r in results:
            records.append(dict(group=group_name, **r))
            print(f"[I-{group_name}] {r['file']} | "
                  f"signal_ratio={r['signal_ratio']:.4f} | mean_h_signal={r['mean_h_signal']:.2f}")
        return pd.DataFrame(records), skipped
    else:
        # serial fallback
        records = []
        skipped_files = []
        for csv_path in csv_files:
            try:
                signal_ratio, mean_h_signal, n_total_rows, n_signal_rows = calculate_signal_ratio_and_mean_h(csv_path)
                records.append({
                    "file": csv_path.name, "group": group_name,
                    "signal_ratio": signal_ratio, "mean_h_signal": mean_h_signal,
                    "n_total_rows": n_total_rows, "n_signal_rows": n_signal_rows})
                print(f"[I-{group_name}] {csv_path.name} | "
                      f"signal_ratio={signal_ratio:.4f} | mean_h_signal={mean_h_signal:.2f}")
            except Exception as e:
                skipped_files.append((csv_path.name, str(e)))
                print(f"[I-skipped] {csv_path.name}: {e}")
        if len(records) == 0:
            raise RuntimeError(f"I panel: {group_name} did not successfully process any files.")
        return pd.DataFrame(records), skipped_files


def confidence_ellipse(x, y, ax, edgecolor, facecolor, alpha=0.2, linewidth=1.0, log_y=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if log_y:
        # Only positive values exist on a log axis, and the ellipse is fitted
        # in log space so it keeps its shape after the axis transform.
        positive = y > 0
        x = x[positive]
        y = np.log10(y[positive])

    if len(x) < 2:
        return None
    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return None
    if not np.isfinite(cov).all():
        return None
    if np.linalg.det(cov) <= 0:
        return None

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2 * math.sqrt(CHI2_95_DF2 * eigvals[0])
    height = 2 * math.sqrt(CHI2_95_DF2 * eigvals[1])

    if log_y:
        # An ellipse in log space is not an ellipse in data space, so trace
        # its boundary as a polygon and map the y coordinates back.
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        major = eigvecs[:, 0]
        minor = eigvecs[:, 1]
        dx = (width / 2) * np.cos(theta) * major[0] + (height / 2) * np.sin(theta) * minor[0]
        dy_log = (width / 2) * np.cos(theta) * major[1] + (height / 2) * np.sin(theta) * minor[1]
        boundary = np.column_stack([mean_x + dx, 10.0 ** (mean_y + dy_log)])
        ellipse = Polygon(
            boundary,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth
        )
        ax.add_patch(ellipse)
        return ellipse

    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth
    )
    ax.add_patch(ellipse)
    return ellipse


def draw_i_panel(ax, neg_df, exp_df):
    neg_ratio = neg_df["signal_ratio"].to_numpy(dtype=float)
    neg_h = neg_df["mean_h_signal"].to_numpy(dtype=float)
    exp_ratio = exp_df["signal_ratio"].to_numpy(dtype=float)
    exp_h = exp_df["mean_h_signal"].to_numpy(dtype=float)

    if I_USE_LOG_Y:
        # Non-positive mean intensities cannot be placed on a log axis.
        neg_visible = neg_h > 0
        exp_visible = exp_h > 0
    else:
        neg_visible = np.ones(len(neg_h), dtype=bool)
        exp_visible = np.ones(len(exp_h), dtype=bool)

    neg_ellipse = confidence_ellipse(
        neg_ratio, neg_h,
        ax=ax,
        edgecolor=NEG_COLOR,
        facecolor=NEG_COLOR,
        alpha=NEG_FILL_ALPHA,
        linewidth=1.0,
        log_y=I_USE_LOG_Y
    )
    exp_ellipse = confidence_ellipse(
        exp_ratio, exp_h,
        ax=ax,
        edgecolor=EXP_COLOR,
        facecolor=EXP_COLOR,
        alpha=EXP_FILL_ALPHA,
        linewidth=1.0,
        log_y=I_USE_LOG_Y
    )

    ax.scatter(
        neg_ratio[neg_visible],
        neg_h[neg_visible],
        s=POINT_SIZE,
        color=NEG_COLOR,
        alpha=POINT_ALPHA,
        edgecolors="none",
        label=f"Control, n={len(neg_df)}"
    )
    ax.scatter(
        exp_ratio[exp_visible],
        exp_h[exp_visible],
        s=POINT_SIZE,
        color=EXP_COLOR,
        alpha=POINT_ALPHA,
        edgecolors="none",
        label=f"daf-16, n={len(exp_df)}"
    )

    def group_center_y(values, visible):
        values = values[visible]
        if len(values) == 0:
            return np.nan
        if I_USE_LOG_Y:
            # Geometric mean, matching the log-space confidence ellipse.
            return float(10.0 ** np.mean(np.log10(values)))
        return float(np.mean(values))

    ax.scatter(
        [float(np.mean(neg_ratio))],
        [group_center_y(neg_h, neg_visible)],
        s=MEAN_POINT_SIZE,
        marker="x",
        color=NEG_COLOR,
        linewidth=1.2,
        zorder=5
    )
    ax.scatter(
        [float(np.mean(exp_ratio))],
        [group_center_y(exp_h, exp_visible)],
        s=MEAN_POINT_SIZE,
        marker="x",
        color=EXP_COLOR,
        linewidth=1.2,
        zorder=5
    )

    ax.set_xlabel("Signal-positive pixel fraction")
    ax.set_ylabel("Mean fluorescence intensity")
    ax.set_xlim(0, 1)
    if I_USE_LOG_Y:
        ax.set_yscale("log")
        y_limit_parts = [neg_h[neg_visible], exp_h[exp_visible]]
        for ellipse in (neg_ellipse, exp_ellipse):
            if ellipse is not None:
                y_limit_parts.append(ellipse.get_xy()[:, 1])
        y_limit_values = np.concatenate(y_limit_parts)
        y_limit_values = y_limit_values[y_limit_values > 0]
        if len(y_limit_values) > 0:
            y_lo = float(np.min(y_limit_values))
            y_hi = float(np.max(y_limit_values))
            pad = 0.05 * (math.log10(y_hi) - math.log10(y_lo)) if y_hi > y_lo else 0.05
            ax.set_ylim(10 ** (math.log10(y_lo) - pad), 10 ** (math.log10(y_hi) + pad))
    else:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper right")
    style_stat_axis(ax)
    add_panel_label(ax, "I", panel_type="plot")


# ============================================================
# Main composition
# ============================================================


def main():
    print("=== Integrated 3x3 negative-control figure (t-stack) ===")

    neg_folder = Path(NEG_FOLDER)
    exp_folder = Path(EXP_FOLDER)
    output_path = resolve_output_path(OUTPUT_PATH)

    neg_target_t = _parse_t_value(NEG_TARGET_TIME)
    exp_target_t = _parse_t_value(EXP_TARGET_TIME)

    # A-F data
    neg_instance, neg_t = auto_find_example_instance_and_t(neg_folder, target_time=NEG_TARGET_TIME)
    exp_instance, exp_t = auto_find_example_instance_and_t(exp_folder, target_time=EXP_TARGET_TIME)
    neg_img = prepare_group_images(neg_folder, neg_instance, neg_t, target_time=NEG_TARGET_TIME)
    exp_img = prepare_group_images(exp_folder, exp_instance, exp_t, target_time=EXP_TARGET_TIME)

    print("\n[Info] Negative-control group representative images:")
    print(f"  instance: {neg_instance}")
    print(f"  t: t-{neg_t:03d}")
    print(f"  BF: {neg_img['bf_path'].name}")
    print(f"  CF: {neg_img['cf_path'].name}")
    print(f"  CSV: {neg_img['csv_path'].name}")
    print(f"  signal pixels: {neg_img['n_signal_pixels']}")

    print("\n[Info] Experimental group representative images:")
    print(f"  instance: {exp_instance}")
    print(f"  t: t-{exp_t:03d}")
    print(f"  BF: {exp_img['bf_path'].name}")
    print(f"  CF: {exp_img['cf_path'].name}")
    print(f"  zoom: {exp_img['zoom_path'].name if exp_img['zoom_path'] is not None else 'not found'}")
    print(f"  CSV: {exp_img['csv_path'].name}")
    print(f"  signal pixels: {exp_img['n_signal_pixels']}")

    # G data
    bin_edges = np.arange(H_MIN, H_MAX + BIN_WIDTH + 1, BIN_WIDTH)
    if bin_edges[-1] < H_MAX + 1:
        bin_edges = np.append(bin_edges, H_MAX + 1)
    x_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    neg_g = process_distribution_group(neg_folder, "Control", bin_edges, target_time=NEG_TARGET_TIME)
    exp_g = process_distribution_group(exp_folder, "daf-16", bin_edges, target_time=EXP_TARGET_TIME)
    neg_p99 = np.percentile(neg_g["all_h_values"], 99) if len(neg_g["all_h_values"]) > 0 else None
    if neg_p99 is not None:
        print(f"\n[G] Negative-control group 99th percentile of h = {neg_p99:.3f}")

    # Additional G-panel data export. Existing image and intermediate outputs are unchanged.
    g_distribution_csv = output_path.with_name(
        f"{output_path.stem}_G_distribution_threshold_data.csv"
    )
    export_g_distribution_data_csv(
        neg_result=neg_g,
        exp_result=exp_g,
        x_values=x_values,
        bin_edges=bin_edges,
        neg_p99=neg_p99,
        exp_folder=exp_folder,
        output_csv_path=g_distribution_csv,
        exp_target_time=EXP_TARGET_TIME
    )

    # H data
    x_grid = np.linspace(0, 1, N_GRID)
    neg_h = process_cumulative_group(neg_folder, "Control", x_grid, target_time=NEG_TARGET_TIME)
    exp_h = process_cumulative_group(exp_folder, "daf-16", x_grid, target_time=EXP_TARGET_TIME)

    # Additional H-panel top X_MARK contribution export.
    h_top_fraction_csv = output_path.with_name(
        f"{output_path.stem}_H_top{int(X_MARK * 100)}pct_contribution_stats.csv"
    )
    export_h_top_fraction_stats(
        neg_result=neg_h,
        exp_result=exp_h,
        output_csv_path=h_top_fraction_csv
    )

    # I data
    neg_i, neg_i_skipped = process_scatter_group(neg_folder, "Control", target_time=NEG_TARGET_TIME)
    exp_i, exp_i_skipped = process_scatter_group(exp_folder, "daf-16", target_time=EXP_TARGET_TIME)

    # Figure composition
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE, gridspec_kw={"wspace": 0, "hspace": 0})

    # A
    axes[0, 0].imshow(neg_img["bf"], cmap="gray", interpolation="nearest")
    add_panel_label(axes[0, 0], "A", panel_type="image")

    # B
    axes[0, 1].imshow(neg_img["cf"], interpolation="nearest")
    add_panel_label(axes[0, 1], "B", panel_type="image")

    # C
    axes[0, 2].imshow(neg_img["cf"], interpolation="nearest")
    axes[0, 2].imshow(neg_img["mask_rgba"], interpolation="nearest")
    add_panel_label(axes[0, 2], "C", panel_type="image")

    # D
    axes[1, 0].imshow(exp_img["bf"], cmap="gray", interpolation="nearest")
    add_panel_label(axes[1, 0], "D", panel_type="image")

    # E
    axes[1, 1].imshow(exp_img["cf"], interpolation="nearest")
    add_zoom_rectangle(axes[1, 1])
    add_zoom_inset(axes[1, 1], exp_img["zoom"])
    add_panel_label(axes[1, 1], "E", panel_type="image")

    # F
    axes[1, 2].imshow(exp_img["cf"], interpolation="nearest")
    axes[1, 2].imshow(exp_img["mask_rgba"], interpolation="nearest")
    add_panel_label(axes[1, 2], "F", panel_type="image")

    # Image panel formatting
    for ax in axes[:2, :].flat:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # G-H-I: each bottom-row subplot cell is split into an outer panel area
    # and an inner plotting area so labels stay inside the subplot cell.
    add_panel_label(axes[2, 0], "G", panel_type="plot")
    add_panel_label(axes[2, 1], "H", panel_type="plot")
    add_panel_label(axes[2, 2], "I", panel_type="plot")

    g_ax = make_inner_plot_axes(axes[2, 0])
    h_ax = make_inner_plot_axes(axes[2, 1])
    i_ax = make_inner_plot_axes(axes[2, 2])

    draw_g_panel(g_ax, neg_g, exp_g, x_values, neg_p99=neg_p99)
    draw_h_panel(h_ax, neg_h, exp_h, x_grid)
    draw_i_panel(i_ax, neg_i, exp_i)

    # Remove duplicated plot-panel labels that draw_* functions place on the
    # inner plotting area; the outer panel already carries the label.
    for inner_ax in (g_ax, h_ax, i_ax):
        for txt in list(inner_ax.texts):
            txt.remove()

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.05, top=0.995, wspace=0.0, hspace=0.0)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()

    print("\nDone.")
    print(f"PNG output: {output_path}")
    print(f"PDF output: {pdf_path}")
    print(f"SVG output: {svg_path}")

    if neg_i_skipped:
        print("\n[I] Skipped files in the negative-control group:")
        for name, reason in neg_i_skipped:
            print(f"  - {name}: {reason}")
    if exp_i_skipped:
        print("\n[I] Skipped files in the experimental group:")
        for name, reason in exp_i_skipped:
            print(f"  - {name}: {reason}")

    # Write all skipped files to a log
    skip_log_path = output_path.with_suffix(".skipped.txt")
    all_skipped = (neg_g.get("skipped_files", []) + exp_g.get("skipped_files", [])
                   + neg_h.get("skipped_files", []) + exp_h.get("skipped_files", [])
                   + [(n, r) for n, r in neg_i_skipped] + [(n, r) for n, r in exp_i_skipped])
    with open(skip_log_path, "w", encoding="utf-8") as f:
        f.write(f"Skipped-file log for {output_path}\n")
        f.write(f"{'='*60}\n")
        for name, reason in all_skipped:
            f.write(f"{name}\t{reason}\n")
    print(f"\nSkipped-file log: {skip_log_path}")


if __name__ == "__main__":
    main()
