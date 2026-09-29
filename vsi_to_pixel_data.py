# signal_mask and no_signal_mask segmentation complete
import os
import re
import time
import traceback
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import napari
import javabridge
import bioformats
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.draw import polygon2mask
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects

try:
    from qtpy.QtWidgets import QApplication
except Exception:
    QApplication = None

STOP_FLAG = False
JVM_STARTED = False

Z_STEP_PIXELS = 2.9651


# =========================
# JVM lifecycle
# =========================
def ensure_jvm_started():
    global JVM_STARTED
    if not JVM_STARTED:
        javabridge.start_vm(class_path=bioformats.JARS)
        JVM_STARTED = True
        print("[Java] Java VM started.")


def shutdown_jvm():
    global JVM_STARTED
    if JVM_STARTED:
        try:
            javabridge.kill_vm()
            print("[Java] Java VM stopped.")
        finally:
            JVM_STARTED = False


def debug_print(message: str):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {message}", flush=True)


def pump_gui_events():
    if QApplication is not None:
        try:
            QApplication.processEvents()
        except Exception:
            pass


# =========================
# Geometry utils
# =========================
def get_middle_index(n: int) -> int:
    return n // 2 - 1 if n % 2 == 0 else n // 2


def compute_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    if len(points) == 1:
        return np.array([0.0], dtype=np.float32)
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(dists)]).astype(np.float32)


def chaikin_smooth(points: np.ndarray, n_iter: int = 2) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return pts.copy()
    out = pts.copy()
    for _ in range(max(0, int(n_iter))):
        if len(out) < 3:
            break
        new_pts = [out[0]]
        for i in range(len(out) - 1):
            p = out[i]
            q = out[i + 1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        new_pts.append(out[-1])
        out = np.asarray(new_pts, dtype=np.float32)
    return out


def densify_polyline(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()
    if len(points) == 1:
        return points.copy()

    arc = compute_arc_length(points)
    total = float(arc[-1])
    if total < 1e-6:
        return points[[0]].copy()

    n_samples = max(int(np.ceil(total / max(step, 1e-6))) + 1, len(points))
    s_new = np.linspace(0, total, n_samples, dtype=np.float32)
    y_new = np.interp(s_new, arc, points[:, 0])
    x_new = np.interp(s_new, arc, points[:, 1])
    return np.column_stack([y_new, x_new]).astype(np.float32)


def ensure_2d_polygon(vertices) -> Optional[np.ndarray]:
    arr = np.asarray(vertices, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    if arr.shape[1] > 2:
        arr = arr[:, -2:]
    return arr.astype(np.float32)


def smooth_closed_polygon(vertices: np.ndarray, n_iter: int = 2) -> np.ndarray:
    poly = ensure_2d_polygon(vertices)
    if poly is None or len(poly) < 3:
        return np.asarray(vertices, dtype=np.float32)

    out = poly.copy()
    for _ in range(max(0, int(n_iter))):
        n = len(out)
        if n < 3:
            break
        new_pts = []
        for i in range(n):
            p = out[i]
            q = out[(i + 1) % n]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        out = np.asarray(new_pts, dtype=np.float32)
    return out.astype(np.float32)


def _compute_endpoint_tangent(points: np.ndarray, at_start: bool = True) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if at_start:
        vec = pts[min(1, len(pts)-1)] - pts[0]
    else:
        vec = pts[-1] - pts[max(0, len(pts)-2)]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def project_points_to_polyline(points: np.ndarray, centerline: np.ndarray, arc: np.ndarray,
                               chunk_size: int = 2048) -> Dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    cl = np.asarray(centerline, dtype=np.float32)
    arc = np.asarray(arc, dtype=np.float32)

    n = len(pts)
    if n == 0 or len(cl) < 2:
        z = np.full(n, np.nan, dtype=np.float32)
        zi = np.zeros(n, dtype=np.int32)
        zb = np.zeros(n, dtype=bool)
        return {
            "s": z.copy(),
            "arc_proj": z.copy(),
            "signed_dist": z.copy(),
            "nearest_index": zi,
            "is_before_start": zb.copy(),
            "is_after_end": zb.copy(),
            "is_within_axis": zb.copy(),
        }

    seg_start = cl[:-1].astype(np.float32)
    seg_end = cl[1:].astype(np.float32)
    seg_vec = (seg_end - seg_start).astype(np.float32)
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1).astype(np.float32)
    seg_len2[seg_len2 < 1e-6] = 1e-6
    seg_len = np.sqrt(seg_len2).astype(np.float32)

    start_tangent = _compute_endpoint_tangent(cl, at_start=True)
    end_tangent = _compute_endpoint_tangent(cl, at_start=False)

    arc_proj = np.zeros(n, dtype=np.float32)
    signed_dist = np.zeros(n, dtype=np.float32)
    nearest_index = np.zeros(n, dtype=np.int32)
    is_before_start = np.zeros(n, dtype=bool)
    is_after_end = np.zeros(n, dtype=bool)

    start_point = cl[0]
    end_point = cl[-1]

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        chunk = pts[i0:i1].astype(np.float32)

        v = chunk[:, None, :] - seg_start[None, :, :]
        t = np.sum(v * seg_vec[None, :, :], axis=2) / seg_len2[None, :]
        t_clip = np.clip(t, 0.0, 1.0).astype(np.float32)
        proj = seg_start[None, :, :] + t_clip[:, :, None] * seg_vec[None, :, :]
        diff = chunk[:, None, :] - proj
        dist2 = np.sum(diff * diff, axis=2)
        best_seg = np.argmin(dist2, axis=1)

        local_t = t_clip[np.arange(i1 - i0), best_seg]
        local_proj = proj[np.arange(i1 - i0), best_seg]
        local_vec = seg_vec[best_seg]
        local_len = seg_len[best_seg]
        local_diff = chunk - local_proj

        tangent = local_vec / np.maximum(local_len[:, None], 1e-6)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1).astype(np.float32)
        local_signed = np.sum(local_diff * normal, axis=1).astype(np.float32)
        local_arc = (arc[best_seg] + local_t * local_len).astype(np.float32)

        arc_proj[i0:i1] = local_arc
        signed_dist[i0:i1] = local_signed
        nearest_index[i0:i1] = np.clip(np.round(local_arc).astype(np.int32), 0, len(cl) - 1)

        vec_from_start = chunk - start_point[None, :]
        vec_from_end = chunk - end_point[None, :]
        before_start = (best_seg == 0) & (np.sum(vec_from_start * start_tangent[None, :], axis=1) < 0)
        after_end = (best_seg == len(seg_vec) - 1) & (np.sum(vec_from_end * end_tangent[None, :], axis=1) > 0)
        is_before_start[i0:i1] = before_start
        is_after_end[i0:i1] = after_end

    total = float(arc[-1]) if len(arc) > 0 else 0.0
    s = np.full(n, np.nan, dtype=np.float32)
    if total > 1e-6:
        s = (arc_proj / total).astype(np.float32)

    is_within_axis = (~is_before_start) & (~is_after_end)
    s[~is_within_axis] = np.nan

    return {
        "s": s,
        "arc_proj": arc_proj.astype(np.float32),
        "signed_dist": signed_dist.astype(np.float32),
        "nearest_index": nearest_index.astype(np.int32),
        "is_before_start": is_before_start,
        "is_after_end": is_after_end,
        "is_within_axis": is_within_axis,
    }


def compute_relative_pixel_coordinates(worm_mask: np.ndarray, centerline: np.ndarray, arc: np.ndarray) -> Dict[str, np.ndarray]:
    worm_mask = worm_mask.astype(bool)
    ys, xs = np.nonzero(worm_mask)
    points = np.column_stack([ys, xs]).astype(np.float32)
    proj = project_points_to_polyline(points, centerline, arc)

    n_center = len(centerline)
    half_width = np.full(n_center, np.nan, dtype=np.float32)
    valid = proj["is_within_axis"]
    if np.any(valid):
        for j in range(n_center):
            m = valid & (proj["nearest_index"] == j)
            if np.any(m):
                half_width[j] = float(np.max(np.abs(proj["signed_dist"][m])))

    valid_hw = np.isfinite(half_width) & (half_width > 1e-6)
    if np.any(valid_hw):
        idx = np.arange(n_center, dtype=np.float32)
        half_width = np.interp(idx, idx[valid_hw], half_width[valid_hw]).astype(np.float32)
    else:
        half_width[:] = np.nan

    r = np.full(len(points), np.nan, dtype=np.float32)
    if len(points) > 0:
        hw = half_width[np.clip(proj["nearest_index"], 0, n_center - 1)] if n_center > 0 else np.full(len(points), np.nan, dtype=np.float32)
        ok = valid & np.isfinite(hw) & (hw > 1e-6)
        r[ok] = (np.abs(proj["signed_dist"][ok]) / hw[ok]).astype(np.float32)

    return {
        "y": ys.astype(np.int32),
        "x": xs.astype(np.int32),
        "s": proj["s"].astype(np.float32),
        "r": r.astype(np.float32),
        "arc_proj": proj["arc_proj"].astype(np.float32),
        "signed_dist": proj["signed_dist"].astype(np.float32),
        "nearest_index": proj["nearest_index"].astype(np.int32),
        "is_before_start": proj["is_before_start"],
        "is_after_end": proj["is_after_end"],
        "is_within_axis": proj["is_within_axis"],
        "half_width_by_centerline": half_width.astype(np.float32),
    }


def build_geometry_dataframe(rec: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    centerline = np.asarray(rec["centerline"], dtype=np.float32)
    arc = np.asarray(rec["arc"], dtype=np.float32)
    s = np.asarray(rec["s"], dtype=np.float32)
    for i, pt in enumerate(centerline):
        rows.append({
            "geometry_type": "centerline",
            "order": int(i),
            "y": float(pt[0]),
            "x": float(pt[1]),
            "arc_length": float(arc[i]) if i < len(arc) else np.nan,
            "s": float(s[i]) if i < len(s) else np.nan,
        })
    polygon = rec.get("polygon")
    if polygon is not None:
        polygon = np.asarray(polygon, dtype=np.float32)
        for i, pt in enumerate(polygon):
            rows.append({
                "geometry_type": "polygon",
                "order": int(i),
                "y": float(pt[0]),
                "x": float(pt[1]),
                "arc_length": np.nan,
                "s": np.nan,
            })
    return pd.DataFrame(rows)


# =========================
# Channel detection utils
# =========================
def _pixel_type_to_str(pixel_type: int) -> str:
    pixel_types = {
        0: "INT8", 1: "UINT8", 2: "INT16", 3: "UINT16",
        4: "INT32", 5: "UINT32", 6: "FLOAT", 7: "DOUBLE"
    }
    return pixel_types.get(pixel_type, "UNKNOWN")


def _normalize_channel_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _classify_channel_by_name(name: Optional[str]) -> Optional[str]:
    n = _normalize_channel_name(name)
    if any(k in n for k in ["cf488", "af488", "488", "fitc", "gfp"]):
        return "FL"
    if any(k in n for k in ["dic", "brightfield", "bright field", "bf", "transmitted"]):
        return "BF"
    return None


def _get_channel_names_from_metadata_store(reader, series_index: int, size_c: int) -> List[str]:
    names = []
    try:
        store = reader.rdr.getMetadataStore()
        for c in range(size_c):
            try:
                jname = store.getChannelName(series_index, c)
                name = javabridge.to_string(jname) if jname is not None else ""
            except Exception:
                name = ""
            names.append(name)
    except Exception as e:
        print(f"[metadata] Failed to read MetadataStore: {e}")
    return names


def _get_channel_names_from_series_metadata(reader, series_index: int, size_c: int) -> List[str]:
    names = [""] * size_c
    try:
        md = reader.rdr.getSeriesMetadata()
        if md is None:
            return names
        iterator = md.keySet().iterator()
        all_items = {}
        while iterator.hasNext():
            key = iterator.next()
            key_str = javabridge.to_string(key)
            val = md.get(key)
            val_str = javabridge.to_string(val) if val is not None else ""
            all_items[key_str] = val_str

        for key_str, val_str in all_items.items():
            m = re.search(r"Channel name\s*#(\d+)", key_str, flags=re.I)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < size_c:
                    names[idx] = val_str
    except Exception as e:
        print(f"[metadata] Failed to read raw reader metadata: {e}")
    return names


def _channel_stats(img: np.ndarray) -> dict:
    img = img.astype(np.float32)
    p50 = float(np.percentile(img, 50))
    p90 = float(np.percentile(img, 90))
    p99 = float(np.percentile(img, 99))
    p999 = float(np.percentile(img, 99.9))
    mean = float(np.mean(img))
    std = float(np.std(img))
    tail_ratio = (p999 + 1e-6) / (p90 + 1e-6)
    contrast_ratio = (p99 + 1e-6) / (p50 + 1e-6)
    score_fl = 1.5 * tail_ratio + 1.0 * contrast_ratio + 0.25 * (std / (mean + 1e-6))
    return {
        "mean": mean,
        "std": std,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "p999": p999,
        "tail_ratio": tail_ratio,
        "contrast_ratio": contrast_ratio,
        "score_fl": score_fl,
    }


def _infer_bf_fl_by_image_statistics(reader, series_index: int, size_z: int, size_c: int, size_t: int) -> Tuple[int, int]:
    z_idx = get_middle_index(size_z)
    t_idx = 0 if size_t <= 1 else get_middle_index(size_t)
    stats_all = []
    print("[metadata] Could not identify channels from channel names; falling back to image statistics ...")
    for c in range(size_c):
        img = reader.read(series=series_index, z=z_idx, c=c, t=t_idx, rescale=False)
        stats = _channel_stats(img)
        stats_all.append(stats)
        print(
            f"[fallback] Channel {c} stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"p99={stats['p99']:.3f}, p99.9={stats['p999']:.3f}, "
            f"tail_ratio={stats['tail_ratio']:.3f}, contrast_ratio={stats['contrast_ratio']:.3f}, "
            f"score_fl={stats['score_fl']:.3f}"
        )
    scores = [s["score_fl"] for s in stats_all]
    fl_channel = int(np.argmax(scores))
    bf_channel = 1 - fl_channel if size_c == 2 else int(np.argmin(scores))
    print(f"[fallback] Inferred from image statistics: CF488=channel {fl_channel}, BF/DIC=channel {bf_channel}")
    return bf_channel, fl_channel


def _read_channel_stack(reader, series_index: int, channel_index: int, size_z: int, size_t: int) -> np.ndarray:
    if size_t > 1:
        z_idx = get_middle_index(size_z)
        stack = [reader.read(series=series_index, z=z_idx, c=channel_index, t=t, rescale=False) for t in range(size_t)]
        return np.stack(stack, axis=0)
    stack = [reader.read(series=series_index, z=z, c=channel_index, t=0, rescale=False) for z in range(size_z)]
    return np.stack(stack, axis=0)


# =========================
# VSI reading
# =========================
def read_vsi_with_bioformats(filepath: str):
    """
    Supports two layout types:
    1) 3-series: Series 0 = FL, Series 1 = BF/DIC, Series 2 = thumbnail
    2) 2-series: Series 0 = main image, C=2 (order of DIC and CF488 not fixed), Series 1 = thumbnail

    Returns:
        bf_stack, fl_stack, stack_axis, meta
        - stack_axis: 'z' or 't'
        - For a z-stack, the stack shape is (Z, Y, X)
        - For a t-stack, the stack shape is (T, Y, X), sampled at the middle z by default
    """
    print(f"Reading VSI file with Bio-Formats: {filepath}")
    print("=" * 60)

    ensure_jvm_started()

    try:
        rdr = bioformats.ImageReader(filepath, perform_init=True)
        series_count = rdr.rdr.getSeriesCount()
        print(f"Total image series: {series_count}")

        bf_stack = None
        fl_stack = None
        stack_axis = None
        meta = {}

        if series_count == 3:
            for s in range(series_count):
                rdr.rdr.setSeries(s)
                size_x = rdr.rdr.getSizeX()
                size_y = rdr.rdr.getSizeY()
                size_z = rdr.rdr.getSizeZ()
                size_c = rdr.rdr.getSizeC()
                size_t = rdr.rdr.getSizeT()
                pixel_type_str = _pixel_type_to_str(rdr.rdr.getPixelType())
                print(f"\n--- Series {s} ---")
                print(f"  Dimensions: X={size_x}, Y={size_y}, Z={size_z}, C={size_c}, T={size_t}")
                print(f"  Pixel type: {pixel_type_str}")

                this_axis = 't' if size_t > 1 else 'z'
                if s == 0 and size_x == 1152:
                    print(f"  -> FL data detected (Series {s}), reading...")
                    fl_stack = _read_channel_stack(rdr, s, 0, size_z, size_t)
                    stack_axis = this_axis
                    meta['size_z'] = size_z
                    meta['size_t'] = size_t
                    print(f"  FL stack shape: {fl_stack.shape} | interpreted axes: {stack_axis.upper()}YX")
                elif s == 1 and size_x == 1152:
                    print(f"  -> BF/DIC data detected (Series {s}), reading...")
                    bf_stack = _read_channel_stack(rdr, s, 0, size_z, size_t)
                    stack_axis = this_axis
                    meta['size_z'] = size_z
                    meta['size_t'] = size_t
                    print(f"  BF stack shape: {bf_stack.shape} | interpreted axes: {stack_axis.upper()}YX")
                elif s == 2:
                    print("  -> Thumbnail detected, skipping...")

        elif series_count == 2:
            s = 0
            rdr.rdr.setSeries(s)
            size_x = rdr.rdr.getSizeX()
            size_y = rdr.rdr.getSizeY()
            size_z = rdr.rdr.getSizeZ()
            size_c = rdr.rdr.getSizeC()
            size_t = rdr.rdr.getSizeT()
            pixel_type_str = _pixel_type_to_str(rdr.rdr.getPixelType())
            print(f"\n--- Series {s} ---")
            print(f"  Dimensions: X={size_x}, Y={size_y}, Z={size_z}, C={size_c}, T={size_t}")
            print(f"  Pixel type: {pixel_type_str}")
            print("  -> Two-channel main image detected; trying to remap DIC / CF488 from metadata ...")

            if size_c < 2:
                raise ValueError(f"Main image of the 2-series file has size_c={size_c}; cannot map DIC/CF488.")

            store_names = _get_channel_names_from_metadata_store(rdr, s, size_c)
            print(f"[metadata] MetadataStore channel info: {store_names}")
            raw_names = _get_channel_names_from_series_metadata(rdr, s, size_c)
            print(f"[metadata] Raw reader metadata channel info: {raw_names}")

            names = []
            for i in range(size_c):
                candidate = store_names[i] if i < len(store_names) and store_names[i] else ""
                if not candidate and i < len(raw_names):
                    candidate = raw_names[i]
                names.append(candidate)

            bf_channel = None
            fl_channel = None
            for i, name in enumerate(names):
                cls = _classify_channel_by_name(name)
                print(f"  Channel {i}: name='{name}' -> classified as {cls}")
                if cls == 'BF' and bf_channel is None:
                    bf_channel = i
                elif cls == 'FL' and fl_channel is None:
                    fl_channel = i

            if bf_channel is None or fl_channel is None:
                bf_channel, fl_channel = _infer_bf_fl_by_image_statistics(rdr, s, size_z, size_c, size_t)

            stack_axis = 't' if size_t > 1 else 'z'
            meta['size_z'] = size_z
            meta['size_t'] = size_t
            meta['bf_channel'] = bf_channel
            meta['fl_channel'] = fl_channel

            print(f"  -> BF/DIC mapped to channel {bf_channel}, reading...")
            bf_stack = _read_channel_stack(rdr, s, bf_channel, size_z, size_t)
            print(f"  BF stack shape: {bf_stack.shape} | interpreted axes: {stack_axis.upper()}YX")
            print(f"  -> CF488 mapped to channel {fl_channel}, reading...")
            fl_stack = _read_channel_stack(rdr, s, fl_channel, size_z, size_t)
            print(f"  FL stack shape: {fl_stack.shape} | interpreted axes: {stack_axis.upper()}YX")
            print("\n--- Series 1 ---")
            print("  -> Thumbnail detected, skipping...")
        else:
            raise ValueError(f"Unsupported series layout: series_count={series_count}")

        if bf_stack is None or fl_stack is None:
            raise ValueError("Could not read BF and FL data completely.")

        return bf_stack, fl_stack, stack_axis, meta

    except Exception as e:
        print(f"❌ Error while reading: {e}")
        traceback.print_exc()
        return None, None, None, None


# =========================
# Manual annotation session
# =========================
class ManualAnnotationSession:
    def __init__(self, viewer, output_dir: str, base_name: str, frame_tag: str = "", fl_analysis_stack: Optional[np.ndarray] = None, analysis_mode: str = "t", analysis_display_index: int = 0):
        self.viewer = viewer
        self.output_dir = output_dir
        self.base_name = base_name
        self.frame_tag = frame_tag
        self.fl_analysis_stack = fl_analysis_stack
        self.analysis_mode = analysis_mode
        self.analysis_display_index = analysis_display_index
        self.last_signal_result = None

        self.mode = "polygon_pending"  # polygon_pending, polygon_centerline
        self.current_points: List[np.ndarray] = []
        self.current_polygon: Optional[np.ndarray] = None
        self.records = []

        self.points_layer = viewer.add_points(
            name="centerline_points",
            ndim=2,
            size=10,
            face_color="red",
            edge_color="white",
            edge_width=0.5,
        )

        self.current_line_layer = viewer.add_shapes(
            name="current_centerline",
            shape_type="path",
            edge_color="yellow",
            edge_width=3,
            face_color="transparent",
        )

        self.saved_line_layer = viewer.add_shapes(
            name="saved_centerlines",
            shape_type="path",
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )

        self.current_polygon_layer = viewer.add_shapes(
            name="current_polygon",
            shape_type="polygon",
            edge_color="magenta",
            edge_width=2,
            face_color="transparent",
            opacity=0.9,
        )
        self.current_polygon_layer.editable = True

        self.saved_polygon_layer = viewer.add_shapes(
            name="saved_polygons",
            shape_type="polygon",
            edge_color="magenta",
            edge_width=1.5,
            face_color="transparent",
            opacity=0.5,
        )
        self.saved_polygon_layer.editable = False

        if self.fl_analysis_stack is not None:
            if self.analysis_mode == "z":
                idx = int(np.clip(self.analysis_display_index, 0, self.fl_analysis_stack.shape[0] - 1))
                mask_shape = tuple(int(v) for v in self.fl_analysis_stack[idx].shape[-2:])
            else:
                mask_shape = tuple(int(v) for v in self.fl_analysis_stack[0].shape[-2:])
        else:
            try:
                mask_shape = tuple(int(v) for v in viewer.layers["CF488"].data.shape[-2:])
            except Exception:
                mask_shape = tuple(int(v) for v in viewer.layers[0].data.shape[-2:])


        self.no_signal_layer = viewer.add_image(
            np.zeros(mask_shape, dtype=np.float32),
            name="no_signal_mask",
            opacity=0.001,
            colormap="yellow",
        )

        self.signal_layer = viewer.add_image(
            np.zeros(mask_shape, dtype=np.float32),
            name="signal_mask",
            opacity=0.001,
            colormap="red",
        )

        self._install_mouse_callbacks()
        self.print_status()

    def _install_mouse_callbacks(self):
        @self.viewer.mouse_drag_callbacks.append
        def _record_point(viewer, event):
            if self.mode != "polygon_centerline":
                return
            if event.type != "mouse_press":
                return
            if event.position is None:
                return

            coord = np.array(event.position[:2], dtype=np.float32)

            if event.button == 1:
                self.current_points.append(coord)
                if len(self.current_points) == 1:
                    print(f"[centerline] Pharyngeal pump point recorded: {coord}")
                else:
                    print(f"[centerline] Intermediate point #{len(self.current_points)-1} recorded: {coord}")
                self.refresh_current_display()
                self.print_status()
            elif event.button in (2, 3):
                self.current_points.append(coord)
                print(f"[centerline] Anus endpoint recorded: {coord}")
                self.refresh_current_display()
                print("[centerline] Current centerline is complete and displayed. Please check it; press n to confirm this instance when satisfied.")
                self.print_status()

    def refresh_current_display(self):
        if len(self.current_points) > 0:
            self.points_layer.data = np.asarray(self.current_points, dtype=np.float32)
        else:
            self.points_layer.data = np.empty((0, 2), dtype=np.float32)

        self.current_line_layer.data = []
        if len(self.current_points) >= 2:
            self.current_line_layer.add(np.asarray(self.current_points, dtype=np.float32), shape_type="path")

    def clear_current_annotation(self):
        self.current_points = []
        self.points_layer.data = np.empty((0, 2), dtype=np.float32)
        self.current_line_layer.data = []

    def clear_current_polygon(self):
        self.current_polygon = None
        self.current_polygon_layer.data = []
        self.current_polygon_layer.selected_data = set()

    def start_polygon_mode(self):
        self.mode = "polygon_pending"
        self.clear_current_polygon()
        self.clear_current_annotation()
        self.viewer.layers.selection.active = self.current_polygon_layer
        self.current_polygon_layer.mode = 'add_polygon'
        print("\n>>> Entered mode w: first draw the instance polygon, then draw the centerline")
        print("Automatically switched to the current_polygon layer and entered add_polygon mode.")
        print("Draw a polygon (click to add points, double-click to finish), then press y to lock this instance.")
        self.print_status()

    def lock_latest_polygon(self):
        if len(self.current_polygon_layer.data) == 0:
            print("[instance] No polygon detected yet; draw a polygon on the current_polygon layer first.")
            self.print_status()
            return

        poly = ensure_2d_polygon(self.current_polygon_layer.data[-1])
        if poly is None or len(poly) < 3:
            print("[instance] Current polygon is invalid; please redraw it.")
            self.print_status()
            return

        smoothed_poly = smooth_closed_polygon(poly, n_iter=2)
        self.current_polygon = smoothed_poly.copy()
        self.current_polygon_layer.data = [smoothed_poly]
        self.mode = "polygon_centerline"
        self.current_polygon_layer.mode = 'select'
        print(f"[instance] Polygon locked and smoothed; original vertex count: {len(poly)}, smoothed vertex count: {len(self.current_polygon)}")
        print("Now draw the centerline directly on the whole window: left-click for the pharyngeal pump / waypoints, right-click for the anus.")
        self.print_status()

    def undo_last_point(self):
        if len(self.current_points) == 0:
            print("[undo] No centerline points to undo.")
        else:
            removed = self.current_points.pop()
            print(f"[undo] Removed the last point: {removed}")
            self.refresh_current_display()
        self.print_status()

    def reset_current_work(self):
        if self.mode == "polygon_pending":
            self.clear_current_polygon()
            print("[reset] Current unconfirmed polygon cleared. Confirmed polygons are unaffected.")
        elif self.mode == "polygon_centerline":
            self.clear_current_annotation()
            self.clear_current_polygon()
            self.mode = "polygon_pending"
            self.viewer.layers.selection.active = self.current_polygon_layer
            self.current_polygon_layer.mode = 'add_polygon'
            print("[reset] Current unconfirmed polygon and centerline cleared. Confirmed instances are unaffected.")
        else:
            self.clear_current_annotation()
            self.clear_current_polygon()
            self.mode = "polygon_pending"
            self.viewer.layers.selection.active = self.current_polygon_layer
            self.current_polygon_layer.mode = 'add_polygon'
            print("[reset] Current unconfirmed work cleared.")
        self.print_status()

    def print_status(self):
        state_map = {
            "polygon_pending": "waiting for polygon to be completed",
            "polygon_centerline": "polygon locked, waiting for centerline drawing",
        }

        print("\n" + "-" * 60)
        print(f"Current state: {state_map.get(self.mode, self.mode)}")
        print(f"Centerline points currently staged: {len(self.current_points)} | confirmed instances: {len(self.records)}")

        print("Key bindings:")
        print("  h : print this help again")
        print("  w : enter / return to polygon drawing state")

        if self.mode == "polygon_pending":
            print("  Default flow: draw a polygon on the current_polygon layer (click to add points, double-click to finish), then press y")
            print("  y : lock the latest polygon and switch to centerline drawing")

        if self.mode == "polygon_centerline":
            print("  Left mouse button : click the pharyngeal pump / waypoints in order")
            print("  Right mouse button : click the anus and finish the current line")
            print("  u : undo the last point")

        print("  q : run signal segmentation and refresh the signal / no_signal masks")
        print("  m : clear current unconfirmed work (confirmed instances are unaffected)")
        print("  n : confirm the current instance and automatically smooth / resample the centerline")
        print("  o : save all confirmed instances of the current file / frame and close the viewer")
        print("  c : terminate batch processing and close the viewer")
        print("-" * 60)

    def cancel_all(self):
        global STOP_FLAG
        STOP_FLAG = True
        print("[terminate] User terminated batch processing; remaining VSI files will be skipped.")
        self.viewer.close()


    def _get_display_plane_index(self) -> int:
        if self.fl_analysis_stack is None or self.fl_analysis_stack.ndim != 3:
            return 0
        return int(np.clip(self.analysis_display_index, 0, self.fl_analysis_stack.shape[0] - 1))

    def _ensure_signal_results_for_save(self):
        if self.fl_analysis_stack is None or self.fl_analysis_stack.ndim != 3:
            return

        is_z_mode = (self.analysis_mode == 'z')
        for rec in self.records:
            if rec.get("signal_result") is not None:
                continue
            poly = rec.get("polygon")
            if poly is None:
                continue
            exclusion_mask = self._compose_other_instances_mask(exclude_instance_id=rec["instance_id"], include_current_polygon=False)
            result = analyze_stack_with_polygon(
                self.fl_analysis_stack,
                np.asarray(poly, dtype=np.float32),
                is_z_mode=is_z_mode,
                display_index=self._get_display_plane_index(),
                background_exclusion_mask=exclusion_mask,
                verbose=False,
            )
            rec["signal_result"] = {
                "display_signal_mask": result["display_signal_mask"].copy(),
                "display_no_signal_mask": result["display_no_signal_mask"].copy(),
                "worm_mask": result["worm_mask"].copy(),
                "display_bg_mean": float(result["display_bg_mean"]),
                "display_bg_std": float(result["display_bg_std"]),
                "display_signal_thresh": float(result["display_signal_thresh"]),
                "per_plane_signal_masks": [r["signal_mask"].copy() for r in result["per_plane"]],
                "per_plane_bg_mean": np.asarray([r["bg_mean"] for r in result["per_plane"]], dtype=np.float32),
            }

    def _build_instance_pixel_dataframe(self, rec: Dict[str, Any]) -> pd.DataFrame:
        if self.fl_analysis_stack is None or self.fl_analysis_stack.ndim != 3:
            raise ValueError("Missing fl_analysis_stack; cannot export pixel-level data.")

        signal_result = rec.get("signal_result")
        if signal_result is None:
            raise ValueError(f'Instance {rec["instance_id"]} has no signal_result; run q first or let it be computed automatically before saving.')

        worm_mask = signal_result["worm_mask"].astype(bool)
        signal_mask_display = signal_result["display_signal_mask"].astype(bool)
        bg_mean = float(signal_result.get("display_bg_mean", np.nan))
        per_plane_signal_masks = signal_result.get("per_plane_signal_masks")
        per_plane_bg_mean = signal_result.get("per_plane_bg_mean")

        rel = compute_relative_pixel_coordinates(
            worm_mask=worm_mask,
            centerline=np.asarray(rec["centerline"], dtype=np.float32),
            arc=np.asarray(rec["arc"], dtype=np.float32),
        )

        ys = rel["y"]
        xs = rel["x"]
        d_xy = np.abs(rel["signed_dist"].astype(np.float32))
        s = rel["s"].astype(np.float32)
        r = rel["r"].astype(np.float32)
        s[~rel["is_within_axis"]] = np.nan
        r[~rel["is_within_axis"]] = np.nan

        if self.analysis_mode != "z":
            display_index = self._get_display_plane_index()
            raw_plane = self.fl_analysis_stack[display_index].astype(np.float32)
            h_raw = raw_plane[ys, xs].astype(np.float32)
            h = (h_raw - bg_mean).astype(np.float32) if np.isfinite(bg_mean) else np.full_like(h_raw, np.nan, dtype=np.float32)
            in_signal = signal_mask_display[ys, xs].astype(np.uint8)

            df = pd.DataFrame({
                "x": xs.astype(np.int32),
                "y": ys.astype(np.int32),
                "s": s,
                "r": r,
                "h_raw": h_raw,
                "h": h,
                "is_signal": in_signal,
            })
            return df

        # z mode: export pixels from all z planes and compute the pseudo-3D radial distance
        z_ref = self._get_display_plane_index()
        half_width = rel["half_width_by_centerline"].astype(np.float32)
        nearest_index = np.clip(rel["nearest_index"].astype(np.int32), 0, max(len(half_width) - 1, 0))
        local_radius = half_width[nearest_index] if len(half_width) > 0 else np.full(len(xs), np.nan, dtype=np.float32)

        rows = []
        n_planes = int(self.fl_analysis_stack.shape[0])
        for z_idx in range(n_planes):
            raw_plane = self.fl_analysis_stack[z_idx].astype(np.float32)
            h_raw = raw_plane[ys, xs].astype(np.float32)

            bg_mean_plane = np.nan
            if per_plane_bg_mean is not None and z_idx < len(per_plane_bg_mean):
                bg_mean_plane = float(per_plane_bg_mean[z_idx])
            elif np.isfinite(bg_mean):
                bg_mean_plane = bg_mean
            h = (h_raw - bg_mean_plane).astype(np.float32) if np.isfinite(bg_mean_plane) else np.full_like(h_raw, np.nan, dtype=np.float32)

            if per_plane_signal_masks is not None and z_idx < len(per_plane_signal_masks):
                in_signal = per_plane_signal_masks[z_idx][ys, xs].astype(np.uint8)
            else:
                in_signal = signal_mask_display[ys, xs].astype(np.uint8)

            d_z = np.full(len(xs), abs(z_idx - z_ref) * Z_STEP_PIXELS, dtype=np.float32)
            r_3d = np.full(len(xs), np.nan, dtype=np.float32)
            ok = rel["is_within_axis"] & np.isfinite(local_radius) & (local_radius > 1e-6)
            if np.any(ok):
                r_3d[ok] = np.sqrt(d_xy[ok] ** 2 + d_z[ok] ** 2) / local_radius[ok]

            df_plane = pd.DataFrame({
                "z": np.full(len(xs), z_idx, dtype=np.int32),
                "x": xs.astype(np.int32),
                "y": ys.astype(np.int32),
                "s": s,
                "r": r,
                "d_xy": d_xy,
                "d_z": d_z,
                "r_3d": r_3d,
                "h_raw": h_raw,
                "h": h,
                "is_signal": in_signal,
            })
            rows.append(df_plane)

        if not rows:
            return pd.DataFrame(columns=["z", "x", "y", "s", "r", "d_xy", "d_z", "r_3d", "h_raw", "h", "is_signal"])
        return pd.concat(rows, ignore_index=True)

    def save_all(self):
        print(f"[save] Starting save")
        if len(self.records) == 0:
            print("[save] No confirmed instances; no files were written.")
            self.viewer.close()
            return

        self._ensure_signal_results_for_save()

        prefix = self.base_name if not self.frame_tag else f"{self.base_name}_{self.frame_tag}"
        for rec in self.records:
            idx = rec["instance_id"]

            df_geom = build_geometry_dataframe(rec)
            df_geom.to_csv(
                os.path.join(self.output_dir, f"{prefix}_instance-{idx}_geometry.csv"),
                index=False
            )

            df_pixels = self._build_instance_pixel_dataframe(rec)
            df_pixels.to_csv(
                os.path.join(self.output_dir, f"{prefix}_instance-{idx}_pixels.csv"),
                index=False
            )

        print(f"[save] Saved {len(self.records)} instances to: {self.output_dir}")
        self.viewer.close()

    def confirm_current_instance(self):
        if len(self.current_points) < 2:
            print("[confirmation failed] Not enough centerline points; at least the pharyngeal pump point and the anus point are required.")
            self.print_status()
            return

        raw_points = np.asarray(self.current_points, dtype=np.float32)
        smoothed_points = chaikin_smooth(raw_points, n_iter=2)
        dense_points = densify_polyline(smoothed_points, step=1.0)
        arc = compute_arc_length(dense_points)
        if len(arc) == 0:
            s_norm = np.array([], dtype=np.float32)
        elif arc[-1] < 1e-6:
            s_norm = np.zeros_like(arc)
        else:
            s_norm = arc / arc[-1]

        record = {
            "instance_id": len(self.records),
            "mode": self.mode,
            "polygon": None if self.current_polygon is None else self.current_polygon.copy(),
            "raw_points": raw_points.copy(),
            "smoothed_points": smoothed_points.copy(),
            "centerline": dense_points.copy(),
            "arc": arc.copy(),
            "s": s_norm.copy(),
            "signal_result": None if self.last_signal_result is None else {
                "display_signal_mask": self.last_signal_result["display_signal_mask"].copy(),
                "display_no_signal_mask": self.last_signal_result["display_no_signal_mask"].copy(),
                "worm_mask": self.last_signal_result["worm_mask"].copy(),
                "display_bg_mean": float(self.last_signal_result["display_bg_mean"]),
                "display_bg_std": float(self.last_signal_result["display_bg_std"]),
                "display_signal_thresh": float(self.last_signal_result["display_signal_thresh"]),
                "per_plane_signal_masks": [r["signal_mask"].copy() for r in self.last_signal_result["per_plane"]],
                "per_plane_bg_mean": np.asarray([r["bg_mean"] for r in self.last_signal_result["per_plane"]], dtype=np.float32),
            },
        }
        self.records.append(record)

        self.saved_line_layer.add(dense_points, shape_type="path")
        if self.current_polygon is not None:
            self.saved_polygon_layer.add(self.current_polygon, shape_type="polygon")

        print(
            f"[confirm] Recorded instance {record['instance_id']}; "
            f"raw points={len(raw_points)}, smoothed nodes={len(smoothed_points)}, resampled points={len(dense_points)}"
        )

        prev_mode = self.mode
        self.clear_current_annotation()
        self.clear_current_polygon()
        self.last_signal_result = None
        self._refresh_aggregate_mask_layers()

        self.mode = "polygon_pending"
        self.viewer.layers.selection.active = self.current_polygon_layer
        self.current_polygon_layer.mode = 'add_polygon'
        print("[next step] To annotate another worm, draw a new polygon and press y when done.")

        self.print_status()

    def _compose_other_instances_mask(self, exclude_instance_id: Optional[int] = None,
                                      include_current_polygon: bool = False) -> np.ndarray:
        mask_shape = self.signal_layer.data.shape
        combined = np.zeros(mask_shape, dtype=bool)

        for rec in self.records:
            if exclude_instance_id is not None and rec.get("instance_id") == exclude_instance_id:
                continue
            poly = rec.get("polygon")
            if poly is None:
                continue
            combined |= polygon_to_mask(mask_shape, poly)

        if include_current_polygon and self.current_polygon is not None:
            combined |= polygon_to_mask(mask_shape, self.current_polygon)

        return combined

    def _refresh_aggregate_mask_layers(self):
        agg_signal = np.zeros(self.signal_layer.data.shape, dtype=bool)
        agg_no_signal = np.zeros(self.no_signal_layer.data.shape, dtype=bool)

        for rec in self.records:
            sr = rec.get("signal_result")
            if not sr:
                continue
            agg_signal |= sr["display_signal_mask"].astype(bool)
            agg_no_signal |= sr["display_no_signal_mask"].astype(bool)

        if self.current_polygon is not None and self.last_signal_result is not None:
            agg_signal |= self.last_signal_result["display_signal_mask"].astype(bool)
            agg_no_signal |= self.last_signal_result["display_no_signal_mask"].astype(bool)

        self.signal_layer.data = agg_signal.astype(np.float32)
        self.no_signal_layer.data = agg_no_signal.astype(np.float32)

    def trigger_hotspot_search(self):
        debug_print("[signal segmentation] q key triggered; running signal / no_signal segmentation for all instances ...")

        if self.fl_analysis_stack is None or self.fl_analysis_stack.ndim != 3:
            debug_print("[signal segmentation] No FL stack available for analysis.")
            self.print_status()
            return

        tasks: List[Dict] = []
        for rec in self.records:
            poly = rec.get("polygon")
            if poly is None:
                continue
            tasks.append({
                "kind": "record",
                "instance_id": rec["instance_id"],
                "polygon": np.asarray(poly, dtype=np.float32).copy(),
                "label": f"confirmed instance #{rec['instance_id']}",
            })

        if self.current_polygon is not None:
            tasks.append({
                "kind": "current",
                "instance_id": None,
                "polygon": self.current_polygon.copy(),
                "label": "current locked instance",
            })

        if not tasks:
            debug_print("[signal segmentation] No instances to analyze. Complete at least one instance first, or draw a polygon and press y.")
            self.print_status()
            return

        debug_print(
            f"[signal segmentation] Instances to analyze: {len(tasks)} | FL stack shape: {self.fl_analysis_stack.shape} | "
            f"analysis_mode={self.analysis_mode} | display_index={self.analysis_display_index}"
        )
        pump_gui_events()

        is_z_mode = (self.analysis_mode == 'z')
        try:
            for task in tasks:
                if task["kind"] == "record":
                    exclusion_mask = self._compose_other_instances_mask(exclude_instance_id=task["instance_id"], include_current_polygon=False)
                else:
                    exclusion_mask = self._compose_other_instances_mask(exclude_instance_id=None, include_current_polygon=False)

                debug_print(
                    f"[signal segmentation] Analyzing {task['label']} | polygon vertices={len(task['polygon'])} | "
                    f"other_instance_px={int(exclusion_mask.sum())}"
                )
                result = analyze_stack_with_polygon(
                    self.fl_analysis_stack,
                    task["polygon"],
                    is_z_mode=is_z_mode,
                    display_index=self.analysis_display_index,
                    background_exclusion_mask=exclusion_mask,
                    verbose=True,
                )

                if task["kind"] == "record":
                    rec = self.records[task["instance_id"]]
                    rec["signal_result"] = {
                        "display_signal_mask": result["display_signal_mask"].copy(),
                        "display_no_signal_mask": result["display_no_signal_mask"].copy(),
                        "worm_mask": result["worm_mask"].copy(),
                        "display_bg_mean": float(result["display_bg_mean"]),
                        "display_bg_std": float(result["display_bg_std"]),
                        "display_signal_thresh": float(result["display_signal_thresh"]),
                    }
                else:
                    self.last_signal_result = result

            self._refresh_aggregate_mask_layers()
            pump_gui_events()
            debug_print("[signal segmentation] All instances analyzed; merged display layers refreshed; per-instance results are still stored separately.")
        except Exception as e:
            debug_print(f"[signal segmentation] Error during execution: {e}")
            traceback.print_exc()

        self.print_status()
# =========================
# Signal / no-signal analysis
# =========================
def polygon_to_mask(image_shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    poly = ensure_2d_polygon(polygon)
    if poly is None:
        return np.zeros(image_shape, dtype=bool)
    return polygon2mask(image_shape, poly).astype(bool)


def make_ring(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    se = disk(radius)
    dil = binary_dilation(mask, structure=se)
    return np.logical_and(dil, ~mask)


def make_annulus(mask: np.ndarray, inner_radius: int = 2, outer_radius: int = 12) -> np.ndarray:
    mask = mask.astype(bool)
    if outer_radius <= 0 or outer_radius <= inner_radius:
        return np.zeros_like(mask, dtype=bool)
    outer = binary_dilation(mask, structure=disk(outer_radius))
    inner = binary_dilation(mask, structure=disk(max(0, inner_radius))) if inner_radius > 0 else mask.copy()
    return outer & (~inner)


def estimate_local_background(denoise: np.ndarray, worm_mask: np.ndarray,
                              exclusion_mask: Optional[np.ndarray] = None,
                              inner_radius: int = 2, outer_radius: int = 12) -> Tuple[float, float, np.ndarray]:
    local_ring = make_annulus(worm_mask, inner_radius=inner_radius, outer_radius=outer_radius) & (~worm_mask)
    if exclusion_mask is not None:
        local_ring &= (~exclusion_mask.astype(bool))

    if np.any(local_ring):
        vals = denoise[local_ring]
        return float(np.mean(vals)), float(np.std(vals)), local_ring

    outside = ~worm_mask
    if exclusion_mask is not None:
        outside &= (~exclusion_mask.astype(bool))

    if np.any(outside):
        vals = denoise[outside]
        return float(np.mean(vals)), float(np.std(vals)), outside

    return float(np.mean(denoise)), float(np.std(denoise)), np.zeros_like(worm_mask, dtype=bool)


def analyze_single_plane(fl_plane: np.ndarray, worm_mask: np.ndarray,
                         sigma_small: float = 1.0,
                         diffuse_thresh_k: float = 3.3,
                         bg_inner_radius: int = 2, bg_outer_radius: int = 12,
                         background_exclusion_mask: Optional[np.ndarray] = None,
                         plane_index: Optional[int] = None,
                         verbose: bool = False) -> dict:
    prefix = f"[signal segmentation][plane {plane_index}]" if plane_index is not None else "[signal segmentation][plane]"
    flf = fl_plane.astype(np.float32)
    worm_mask = worm_mask.astype(bool)
    worm_area = int(worm_mask.sum())
    if verbose:
        debug_print(f"{prefix} Analysis started | plane shape={flf.shape} | worm_area={worm_area}")
    if worm_area == 0:
        z = np.zeros_like(worm_mask, dtype=bool)
        if verbose:
            debug_print(f"{prefix} worm_mask is empty; returning an empty result.")
        return {
            'signal_mask': z.copy(), 'no_signal_mask': z.copy(),
            'bg_mean': 0.0, 'bg_std': 0.0, 'signal_thresh': 0.0,
            'background_region': z.copy(),
        }

    denoise = gaussian(flf, sigma=sigma_small, preserve_range=True)
    if verbose:
        debug_print(f"{prefix} Gaussian done | sigma_small={sigma_small}")
    pump_gui_events()

    bg_mean, bg_std, bg_region = estimate_local_background(
        denoise,
        worm_mask,
        exclusion_mask=background_exclusion_mask,
        inner_radius=bg_inner_radius,
        outer_radius=bg_outer_radius,
    )
    if verbose:
        debug_print(
            f"{prefix} Background estimation done | bg_mean={bg_mean:.3f}, bg_std={bg_std:.3f} | "
            f"bg_px={int(bg_region.sum())} | inner={bg_inner_radius} | outer={bg_outer_radius}"
        )

    signal_img = denoise - bg_mean
    signal_thresh = max(float(bg_std * diffuse_thresh_k), 0.0)
    signal_mask = (signal_img > signal_thresh) & worm_mask
    signal_mask = remove_small_objects(signal_mask, min_size=4)
    no_signal_mask = worm_mask & (~signal_mask)

    if verbose:
        debug_print(
            f"{prefix} Output done | signal={int(signal_mask.sum())} px, "
            f"no_signal={int(no_signal_mask.sum())} px | threshold={signal_thresh:.3f}"
        )
    pump_gui_events()

    return {
        'signal_mask': signal_mask.astype(bool),
        'no_signal_mask': no_signal_mask.astype(bool),
        'bg_mean': bg_mean,
        'bg_std': bg_std,
        'signal_thresh': signal_thresh,
        'background_region': bg_region.astype(bool),
    }


def analyze_stack_with_polygon(fl_stack: np.ndarray, polygon: np.ndarray, is_z_mode: bool, display_index: int,
                               background_exclusion_mask: Optional[np.ndarray] = None,
                               verbose: bool = False) -> dict:
    if fl_stack.ndim != 3:
        raise ValueError(f'fl_stack must be a 3D array, but ndim={fl_stack.ndim}')
    mode_name = 'z' if is_z_mode else 't'
    debug_print(f"[signal segmentation] analyze_stack_with_polygon started | mode={mode_name} | fl_stack shape={fl_stack.shape} | display_index={display_index}")
    worm_mask = polygon_to_mask(fl_stack.shape[1:], polygon)
    debug_print(f"[signal segmentation] worm_mask generated | area={int(worm_mask.sum())} px | image_shape={fl_stack.shape[1:]}")
    pump_gui_events()

    results = []
    n_planes = fl_stack.shape[0]
    for i in range(n_planes):
        debug_print(f"[signal segmentation] -> processing plane {i+1}/{n_planes}")
        plane_result = analyze_single_plane(
            fl_stack[i],
            worm_mask,
            background_exclusion_mask=background_exclusion_mask,
            plane_index=i,
            verbose=verbose,
        )
        results.append(plane_result)

    disp_idx = int(np.clip(display_index, 0, fl_stack.shape[0] - 1))
    disp = results[disp_idx]
    signal_sum = int(sum(r['signal_mask'].sum() for r in results))
    debug_print(f"[signal segmentation] {mode_name}-stack analysis complete; {fl_stack.shape[0]} planes total; display plane index={disp_idx}.")
    debug_print(f"[signal segmentation] Current display plane: signal={int(disp['signal_mask'].sum())} px, no_signal={int(disp['no_signal_mask'].sum())} px")
    debug_print(f"[signal segmentation] Whole-stack total: signal={signal_sum} px")
    return {
        'worm_mask': worm_mask,
        'per_plane': results,
        'display_index': disp_idx,
        'display_signal_mask': disp['signal_mask'],
        'display_no_signal_mask': disp['no_signal_mask'],
        'display_bg_mean': float(disp['bg_mean']),
        'display_bg_std': float(disp['bg_std']),
        'display_signal_thresh': float(disp['signal_thresh']),
    }

# =========================
# Display / processing
# =========================
def launch_annotation_viewer(bf: np.ndarray, fl: np.ndarray, output_dir: str, base_name: str, frame_tag: str = "", fl_analysis_stack: Optional[np.ndarray] = None, analysis_mode: str = "t", analysis_display_index: int = 0):
    print(f"[display check] Mean pixel intensity of the displayed CF488 frame: {float(np.mean(fl)):.6f}")
    viewer = napari.Viewer()
    viewer.add_image(bf, name="BF", contrast_limits=[np.percentile(bf, 1), np.percentile(bf, 99.99)])
    viewer.add_image(
        fl,
        name="CF488",
        contrast_limits=[np.percentile(fl, 1), np.percentile(fl, 99.99)],
        colormap="green",
        opacity=0.1,
    )

    session = ManualAnnotationSession(viewer, output_dir, base_name, frame_tag=frame_tag, fl_analysis_stack=fl_analysis_stack, analysis_mode=analysis_mode, analysis_display_index=analysis_display_index)
    session.start_polygon_mode()

    @viewer.bind_key("q")
    def _signal(v):
        session.trigger_hotspot_search()

    @viewer.bind_key("w")
    def _mode_polygon(v):
        session.start_polygon_mode()

    @viewer.bind_key("y")
    def _lock_polygon(v):
        session.lock_latest_polygon()

    @viewer.bind_key("u")
    def _undo(v):
        session.undo_last_point()

    @viewer.bind_key("m")
    def _reset(v):
        session.reset_current_work()

    @viewer.bind_key("n")
    def _confirm(v):
        session.confirm_current_instance()

    @viewer.bind_key("o")
    def _save(v):
        session.save_all()

    @viewer.bind_key("c")
    def _cancel(v):
        session.cancel_all()

    @viewer.bind_key("h")
    def _help(v):
        session.print_status()

    napari.run()


def process_vsi(file_path: str, output_dir: str, user_mode: str, experiment_name: str):
    bf_stack, fl_stack, stack_axis, meta = read_vsi_with_bioformats(file_path)
    if bf_stack is None or fl_stack is None:
        print(f"Cannot read file: {file_path}")
        return

    raw_base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_name = f"{experiment_name}_{raw_base_name}"

    if user_mode == 'z':
        if stack_axis == 'z':
            bf = bf_stack[get_middle_index(bf_stack.shape[0])]
            fl = fl_stack[get_middle_index(fl_stack.shape[0])]
            print(f"[mode] z mode: displaying the middle z (index={get_middle_index(bf_stack.shape[0])}).")
        else:
            bf = bf_stack[get_middle_index(bf_stack.shape[0])]
            fl = fl_stack[get_middle_index(fl_stack.shape[0])]
            print(f"[mode] z mode: read result is a T stack; displaying the middle t (index={get_middle_index(bf_stack.shape[0])}) and processing it as a single frame.")
        launch_annotation_viewer(bf, fl, output_dir, base_name, frame_tag="", fl_analysis_stack=fl_stack, analysis_mode="z", analysis_display_index=get_middle_index(fl_stack.shape[0]))

    elif user_mode == 't':
        if stack_axis == 't':
            n_frames = bf_stack.shape[0]
            print(f"[mode] t mode: {n_frames} time points; annotating frame by frame.")
            for t in range(n_frames):
                if STOP_FLAG:
                    break
                print("\n" + "=" * 60)
                print(f"[T frame] Processing t={t}/{n_frames - 1}")
                launch_annotation_viewer(bf_stack[t], fl_stack[t], output_dir, base_name, frame_tag=f"t-{t:03d}", fl_analysis_stack=fl_stack[t:t+1], analysis_mode="t", analysis_display_index=0)
        else:
            n_frames = bf_stack.shape[0]
            print(f"[mode] t mode: read result is a Z stack; processing plane by plane for every z. {n_frames} planes total.")
            for t in range(n_frames):
                if STOP_FLAG:
                    break
                print("\n" + "=" * 60)
                print(f"[plane] Processing index={t}/{n_frames - 1}")
                launch_annotation_viewer(bf_stack[t], fl_stack[t], output_dir, base_name, frame_tag=f"t-{t:03d}", fl_analysis_stack=fl_stack[t:t+1], analysis_mode="t", analysis_display_index=0)
    else:
        raise ValueError(f"Unknown mode: {user_mode}")


def ask_processing_mode() -> str:
    print("\nPlease select a processing mode:")
    print("  z : process a z stack (display the middle z, then manually annotate instances and centerlines)")
    print("  t : process a t stack (manually annotate instances and centerlines for every t frame)")
    while True:
        mode = input("Enter mode (z/t): ").strip().lower()
        if mode in {"z", "t"}:
            return mode
        print("Invalid input; please enter z or t.")


def ask_experiment_name() -> str:
    while True:
        name = input("Enter the current experiment name: ").strip()
        if name:
            safe_name = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_")
            if safe_name:
                print(f"[experiment name] Current experiment name: {safe_name}")
                return safe_name
        print("Experiment name cannot be empty.")

def main():
    input_dir = r"path/to/your/vsi_sample_folder"
    output_dir = r"path/to/your/output_folder"

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".vsi")]
    files.sort()

    experiment_name = ask_experiment_name()
    user_mode = ask_processing_mode()
    ensure_jvm_started()
    try:
        for f in files:
            if STOP_FLAG:
                print("Batch processing stopped.")
                break
            path = os.path.join(input_dir, f)
            print("\n" + "#" * 70)
            print("Processing:", f)
            process_vsi(path, output_dir, user_mode, experiment_name)
    finally:
        shutdown_jvm()


if __name__ == "__main__":
    main()