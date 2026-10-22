"""Evaluate floorplan/v0 predictions against the test-set ground truth (eval-v0: the metrics defined below).

Usage (run with the eval venv: numpy, scipy, shapely, opencv-python-headless):

  .venv-eval/bin/python -I tools/fp_eval.py --pred-dir PRED_DIR --split dev|test|all \
        [--out report.json] [--md report.md] [--testdata DIR]
  .venv-eval/bin/python -I tools/fp_eval.py --pred P.json --gt G.json [--out report.json] [--md report.md]
  .venv-eval/bin/python -I tools/fp_eval.py --pred-from-gt OUT_DIR --split dev|test|all

PRED_DIR holds one `<key>_<NN>.json` per sample (a `<dir>/<key>_<NN>.json` layout is also accepted).
A missing prediction is scored as an empty prediction (all misses). An unreadable prediction is
scored as empty too and flagged as invalid. `--pred-from-gt` writes P = G copies (gt_fidelity
stripped) for sanity checks: evaluating them must give 1.0 on every applicable metric.

Wall IoU uses exact areas (shapely union of the wall footprints, clipped to the image) -- the
continuous limit of rasterising; walls without a polygon are centerline x thickness rectangles
(flat caps). Centerline P/R samples every ~1 px with exact point-to-segment distances.

Every metric is reported as {"value": v, "n": n}. value is null when n == 0 (not applicable).
Per-dataset and overall values are micro-averages: the raw counts of all samples are summed first.
"""
import argparse
import copy
import json
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("fp_eval.py needs opencv-python-headless (use .venv-eval/bin/python)")
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    sys.exit("fp_eval.py needs scipy (use .venv-eval/bin/python)")
try:
    import shapely
    from shapely.geometry import LineString, Point, Polygon
except ImportError:  # pragma: no cover
    sys.exit("fp_eval.py needs shapely (use .venv-eval/bin/python)")

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import fpgt  # noqa: E402  (our own validator; -I does not add the script dir)

DATASETS = ["cubicasa5k", "floorplancad", "cvc-fp", "sesyd", "swiss-dwellings", "synthetic", "wikimedia-commons"]
DOOR_CLASSES = ["single", "double", "sliding", "pocket", "folding", "revolving", "garage", "none"]
ROOM_CLASSES = ["kitchen", "bedroom", "bath", "living", "corridor", "storage", "balcony", "outdoor", "other"]
ANGLE_GATE_DEG = 20.0
ROOM_IOU_GATE = 0.5
BIG = 1e12
EPS = 1e-9

# Order of the metrics in the report. (key, counter-derived; see finalize()).
METRIC_ORDER = [
    "wall_iou", "wall_cl_p@τ1", "wall_cl_r@τ1", "wall_cl_f1@τ1", "wall_cl_p@τ2", "wall_cl_r@τ2", "wall_cl_f1@τ2",
    "wall_kind_acc",
    "door_p", "door_r", "door_f1", "window_p", "window_r", "window_f1",
    "door_as_window", "window_as_door", "door_width_err_median", "window_width_err_median",
    "door_type_acc",
    "hinge_side_acc", "swing_side_acc", "door_dir_acc", "door_dir_recall", "double_leaf_count_acc",
    "swing_into_acc", "hinge_offset_median",
    "room_p@0.5", "room_r@0.5", "room_f1@0.5", "room_sq", "room_pq", "room_type_acc",
]
# Metrics that are not "higher is better, 1.0 = perfect".
DIAGNOSTIC = {"door_as_window", "window_as_door", "door_width_err_median", "window_width_err_median",
              "hinge_offset_median"}


# ----------------------------------------------------------------------------------------- parsing

def _pt(p):
    """Return a finite float [x, y] array, or None."""
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return None
    try:
        v = np.array([float(p[0]), float(p[1])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(v)):
        return None
    return v


def _poly(pts):
    if not isinstance(pts, list) or len(pts) < 3:
        return None
    out = [_pt(p) for p in pts]
    if any(p is None for p in out):
        return None
    return np.array(out, dtype=float)


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _list(doc, key):
    v = doc.get(key) if isinstance(doc, dict) else None
    return v if isinstance(v, list) else []


def parse_walls(doc, default_thickness):
    walls = []
    for w in _list(doc, "walls"):
        if not isinstance(w, dict):
            continue
        cl = w.get("centerline")
        if not isinstance(cl, list) or len(cl) != 2:
            continue
        a, b = _pt(cl[0]), _pt(cl[1])
        if a is None or b is None:
            continue
        t = _num(w.get("thickness_px"))
        if t is None or t <= 0:
            t = default_thickness
        kind = w.get("kind") if w.get("kind") in ("exterior", "interior", "unknown") else "unknown"
        walls.append({"id": w.get("id"), "a": a, "b": b, "t": t, "kind": kind, "polygon": _poly(w.get("polygon"))})
    return walls


def parse_leaves(raw):
    leaves = []
    for lf in raw if isinstance(raw, list) else []:
        if not isinstance(lf, dict):
            continue
        h, c, o = _pt(lf.get("hinge")), _pt(lf.get("closed_tip")), _pt(lf.get("open_tip"))
        if h is None or c is None or o is None:
            continue
        leaves.append({"hinge": h, "closed_tip": c, "open_tip": o})
    return leaves


def parse_openings(doc):
    """Return {"door": [...], "window": [...]}. category 'opening' is a door with type 'none'."""
    out = {"door": [], "window": []}
    for o in _list(doc, "openings"):
        if not isinstance(o, dict):
            continue
        cat = o.get("category")
        if cat not in ("door", "window", "opening"):
            continue
        jambs = o.get("jambs")
        if not isinstance(jambs, list) or len(jambs) != 2:
            continue
        a, b = _pt(jambs[0]), _pt(jambs[1])
        if a is None or b is None:
            continue
        d = b - a
        length = float(np.hypot(*d))
        width = _num(o.get("width_px"))
        if width is None or width <= 0:
            width = length
        typ = o.get("type") if isinstance(o.get("type"), str) else "unknown"
        if cat == "opening":
            typ = "none"
        out["door" if cat in ("door", "opening") else "window"].append({
            "id": o.get("id"), "category": cat, "type": typ, "a": a, "b": b, "mid": (a + b) / 2,
            "dir": d / length if length > EPS else None, "width": width,
            "leaves": parse_leaves(o.get("leaves")), "swing_scorable": o.get("swing_scorable"),
            "swing_into": o.get("swing_into") if isinstance(o.get("swing_into"), str) else None,
        })
    return out


def _valid_polygon(arr):
    try:
        g = Polygon(arr)
        if not g.is_valid:
            g = shapely.make_valid(g)
            polys = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
            g = shapely.union_all(polys) if polys else Polygon()
        return g
    except Exception:
        return Polygon()


def parse_rooms(doc):
    rooms = []
    for r in _list(doc, "rooms"):
        if not isinstance(r, dict):
            continue
        arr = _poly(r.get("polygon"))
        if arr is None:
            continue
        rooms.append({"id": r.get("id"), "poly": _valid_polygon(arr),
                      "type": r.get("type") if isinstance(r.get("type"), str) else "unknown"})
    return rooms


def image_size(gt):
    img = gt.get("image") if isinstance(gt.get("image"), dict) else {}
    w, h = img.get("width"), img.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError("GT image.width/height missing")
    return w, h


def fidelity(gt):
    """gt_fidelity of G; for a GT without it (single-pair mode) infer it from what is present."""
    fid = gt.get("gt_fidelity")
    if isinstance(fid, dict):
        return fid
    ops = _list(gt, "openings")
    rooms = _list(gt, "rooms")
    return {
        "walls": "full" if _list(gt, "walls") else "none",
        "door_swing": "full" if any(isinstance(o, dict) and o.get("leaves") for o in ops) else "none",
        "door_type": True, "windows": True, "rooms": bool(rooms),
        "room_types": any(isinstance(r, dict) and r.get("type") not in (None, "unknown") for r in rooms),
    }


def validate_pred(pred, gt):
    """Schema check of a prediction (fpgt.validate minus the GT-only gt_fidelity checks)."""
    if not isinstance(pred, dict):
        return ["prediction is not a JSON object"]
    try:
        errs = [e for e in fpgt.validate(pred) if not e.startswith("gt_fidelity")]
    except Exception as exc:  # fpgt assumes well-formed lists/dicts
        errs = [f"validator crashed: {type(exc).__name__}: {exc}"]
    img = pred.get("image") if isinstance(pred.get("image"), dict) else {}
    gimg = gt.get("image", {})
    if (img.get("width"), img.get("height")) != (gimg.get("width"), gimg.get("height")):
        errs.append(f"image size {img.get('width')}x{img.get('height')} != GT {gimg.get('width')}x{gimg.get('height')}")
    return errs


# ------------------------------------------------------------------------------------------ geometry

def seg_distances(points, A, B, chunk=2_000_000):
    """Distances (K, M) from points (K,2) to segments A[j]-B[j], yielded per row chunk."""
    K, M = len(points), len(A)
    D = B - A
    L2 = np.maximum((D ** 2).sum(1), EPS)
    step = max(1, chunk // max(M, 1))
    for s in range(0, K, step):
        P = points[s:s + step]
        AP = P[:, None, :] - A[None, :, :]
        tt = np.clip((AP * D[None]).sum(2) / L2[None], 0.0, 1.0)
        diff = AP - tt[..., None] * D[None]
        yield s, np.sqrt((diff ** 2).sum(2))


def sample_centerlines(walls):
    """Midpoints of ceil(L) equal sub-intervals per wall, each weighted by its length (sum = total length)."""
    pts, wts, idx = [], [], []
    for i, w in enumerate(walls):
        L = float(np.hypot(*(w["b"] - w["a"])))
        if L <= EPS:
            continue
        n = min(max(1, int(math.ceil(L))), 200_000)
        f = (np.arange(n) + 0.5) / n
        pts.append(w["a"][None] + f[:, None] * (w["b"] - w["a"])[None])
        wts.append(np.full(n, L / n))
        idx.append(np.full(n, i))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(wts), np.concatenate(idx)


def near_distances(points, A, B, cutoff, chunk=4096, budget=2_000_000):
    """Exact point-to-segment distances, restricted to the pairs that can be <= cutoff.

    Points are sorted into grid cells and processed in spatially compact chunks; per chunk only the
    segments whose bbox (grown by cutoff) overlaps the chunk bbox are evaluated. Yields
    (rows, J, d): row indices into points, candidate segment indices, distances (len(rows), len(J)).
    Points never yielded have no segment within cutoff."""
    K = len(points)
    if K == 0 or len(A) == 0:
        return
    slo = np.minimum(A, B) - cutoff
    shi = np.maximum(A, B) + cutoff
    cell = max(16.0, 2.0 * cutoff)
    q = np.floor((points - points.min(0)) / cell).astype(np.int64)
    order = np.lexsort((q[:, 1], q[:, 0]))
    for s in range(0, K, chunk):
        rows = order[s:s + chunk]
        P = points[rows]
        plo, phi = P.min(0), P.max(0)
        J = np.nonzero((slo[:, 0] <= phi[0]) & (shi[:, 0] >= plo[0]) &
                       (slo[:, 1] <= phi[1]) & (shi[:, 1] >= plo[1]))[0]
        if len(J) == 0:
            continue
        step = max(1, budget // len(J))
        for t in range(0, len(rows), step):
            r = rows[t:t + step]
            for _, d in seg_distances(points[r], A[J], B[J], chunk=budget):
                yield r, J, d


def min_dist(points, walls, cutoff):
    """Distance to the nearest wall centerline; exact when <= cutoff, otherwise some value > cutoff
    (inf when no segment is within cutoff)."""
    out = np.full(len(points), np.inf)
    if len(points) == 0 or not walls:
        return out
    A = np.array([w["a"] for w in walls])
    B = np.array([w["b"] for w in walls])
    for r, _, d in near_distances(points, A, B, cutoff):
        out[r] = d.min(1)
    return out


def clip_segment(a, b, lo, hi):
    """Liang-Barsky clip of segment a-b to the box [lo, hi]. Returns (a', b') or None; unclipped
    endpoints are returned unchanged (bit-exact) so P = G stays exact."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(2):
        for p, q in ((-d[k], a[k] - lo[k]), (d[k], hi[k] - a[k])):
            if abs(p) <= EPS:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
    return (a if t0 == 0.0 else a + t0 * d), (b if t1 == 1.0 else a + t1 * d)


def wall_area_geom(walls, w, h):
    """Union of the wall footprints, intersected with the image, as an exact shapely geometry.
    polygon if present, otherwise the centerline buffered by t/2 with FLAT caps (a wall is a
    centerline x thickness rectangle; the GT polygons of floorplancad / swiss-dwellings / cvc-fp are
    exactly that, so round caps would cap a perfect centerline prediction at IoU ~0.88-0.93)."""
    from shapely.geometry import box
    diag = math.hypot(w, h)
    img = box(0, 0, w, h)
    geoms = []
    for wl in walls:
        if wl["polygon"] is not None:
            ring = np.clip(wl["polygon"], -4.0 * diag, 5.0 * diag)  # shapely breaks on ~1e300 coordinates
            g = _valid_polygon(ring)
        else:
            t = min(wl["t"], 4.0 * diag)
            m = t / 2.0 + 1.0
            seg = clip_segment(np.clip(wl["a"], -1e15, 1e15), np.clip(wl["b"], -1e15, 1e15),
                               np.array([-m, -m]), np.array([w + m, h + m]))
            if seg is None or np.hypot(*(seg[1] - seg[0])) <= EPS:
                continue
            g = LineString(seg).buffer(t / 2.0, cap_style="flat")
        if not g.is_empty:
            geoms.append(g)
    if not geoms:
        return Polygon()
    return shapely.union_all(geoms).intersection(img)


def wall_mask(walls, w, h):
    """Legacy integer raster (cv2 rounds vertices and includes the boundary: ~+1 px per dimension).
    Only used as a fallback if the exact geometry path throws."""
    mask = np.zeros((h, w), np.uint8)
    lim = 4.0 * max(w, h)
    for wl in walls:
        if wl["polygon"] is not None:
            rings = [wl["polygon"]]
        else:
            try:
                a, b = np.clip(wl["a"], -lim, lim), np.clip(wl["b"], -lim, lim)
                g = LineString([a, b]) if np.hypot(*(b - a)) > EPS else Point(a)
                g = g.buffer(min(wl["t"], lim) / 2.0)
            except Exception:
                continue
            if g.is_empty:
                continue
            rings = [np.array(p.exterior.coords) for p in getattr(g, "geoms", [g])]
        for ring in rings:
            r = np.round(np.clip(ring, -lim, lim) * 16).astype(np.int32)
            cv2.fillPoly(mask, [r], 1, lineType=cv2.LINE_8, shift=4)
    return mask


def hungarian(cost, feasible):
    """One-to-one assignment minimising cost; returns [(row, col)] restricted to feasible pairs."""
    if cost.size == 0 or not feasible.any():
        return []
    C = np.where(feasible, cost, BIG)
    rows, cols = linear_sum_assignment(C)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if feasible[r, c]]


def match_openings(G, P, tau2):
    """Section 2 gates: centre distance, jamb-direction angle, width ratio. Returns [(gi, pi)]."""
    if not G or not P:
        return []
    gm = np.array([g["mid"] for g in G])
    pm = np.array([p["mid"] for p in P])
    cost = np.sqrt(((gm[:, None] - pm[None]) ** 2).sum(2))
    feas = np.zeros(cost.shape, bool)
    cos_gate = math.cos(math.radians(ANGLE_GATE_DEG)) - EPS
    for i, g in enumerate(G):
        gate = max(tau2, 0.5 * g["width"]) + EPS
        for j, p in enumerate(P):
            if cost[i, j] > gate or g["dir"] is None or p["dir"] is None:
                continue
            if abs(float(g["dir"] @ p["dir"])) < cos_gate:
                continue
            ratio = p["width"] / g["width"]
            if 0.5 - EPS <= ratio <= 2.0 + EPS:
                feas[i, j] = True
    return hungarian(cost, feas)


def _sign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


# ---------------------------------------------------------------------------------------- counters

def new_counters():
    c = {k: 0.0 for k in (
        "n_samples", "wall_samples", "wall_ng", "wall_np", "wall_inter", "wall_union",
        "cl_g_len", "cl_p_len", "cl_g_in1", "cl_p_in1", "cl_g_in2", "cl_p_in2", "wkind_ok", "wkind_n",
        "door_samples", "door_tp", "door_np", "door_ng", "window_samples", "window_tp", "window_np", "window_ng",
        "d2w_hit", "d2w_n", "w2d_hit", "w2d_n",
        "dtype_ok", "dtype_n",
        "dir_samples", "hinge_ok", "hinge_n", "swing_sum", "swing_n", "dir_ok", "dir_n", "dirrec_ok", "dirrec_n",
        "leafcnt_ok", "leafcnt_n", "sinto_ok", "sinto_n",
        "room_samples", "room_tp", "room_np", "room_ng", "room_iou_sum", "rtype_ok", "rtype_n",
    )}
    c.update({"door_width_err": [], "window_width_err": [], "hinge_offset": [],
              "dtype_conf": {}, "rtype_conf": {}})
    return c


def add_counters(acc, c):
    for k, v in c.items():
        if isinstance(v, list):
            acc[k].extend(v)
        elif isinstance(v, dict):
            for g, row in v.items():
                dst = acc[k].setdefault(g, {})
                for p, n in row.items():
                    dst[p] = dst.get(p, 0) + n
        else:
            acc[k] += v
    return acc


def _conf(conf, g, p):
    row = conf.setdefault(g, {})
    row[p] = row.get(p, 0) + 1


def _m(num, den, n=None):
    return {"value": (num / den) if den > 0 else None, "n": int(round(den if n is None else n))}


def _prf(tp, np_, ng, prefix, suffix=""):
    p = tp / np_ if np_ > 0 else (0.0 if ng > 0 else None)
    r = tp / ng if ng > 0 else None
    if p is None or r is None:
        f = None
    else:
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {f"{prefix}_p{suffix}": {"value": p, "n": int(np_), "tp": int(tp)},
            f"{prefix}_r{suffix}": {"value": r, "n": int(ng), "tp": int(tp)},
            f"{prefix}_f1{suffix}": {"value": f, "n": int(ng), "n_pred": int(np_), "tp": int(tp)}}


def _len_prf(p_in, p_len, g_in, g_len, ng, np_, suffix):
    if g_len <= 0:
        return {f"wall_cl_p{suffix}": _m(0, 0), f"wall_cl_r{suffix}": _m(0, 0), f"wall_cl_f1{suffix}": _m(0, 0)}
    p = p_in / p_len if p_len > 0 else 0.0
    r = g_in / g_len
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {f"wall_cl_p{suffix}": {"value": p, "n": int(np_), "len_px": round(p_len, 1)},
            f"wall_cl_r{suffix}": {"value": r, "n": int(ng), "len_px": round(g_len, 1)},
            f"wall_cl_f1{suffix}": {"value": f, "n": int(ng), "n_pred": int(np_)}}


def finalize(c):
    """Turn summed raw counts into metrics {name: {"value", "n", ...}}."""
    m = {}
    if c["wall_samples"] > 0:
        m["wall_iou"] = {"value": c["wall_inter"] / c["wall_union"] if c["wall_union"] > 0 else 0.0,
                         "n": int(c["wall_ng"]), "n_pred": int(c["wall_np"])}
        m.update(_len_prf(c["cl_p_in1"], c["cl_p_len"], c["cl_g_in1"], c["cl_g_len"], c["wall_ng"], c["wall_np"], "@τ1"))
        m.update(_len_prf(c["cl_p_in2"], c["cl_p_len"], c["cl_g_in2"], c["cl_g_len"], c["wall_ng"], c["wall_np"], "@τ2"))
    else:
        for k in ("wall_iou", "wall_cl_p@τ1", "wall_cl_r@τ1", "wall_cl_f1@τ1",
                  "wall_cl_p@τ2", "wall_cl_r@τ2", "wall_cl_f1@τ2"):
            m[k] = _m(0, 0)
    m["wall_kind_acc"] = {"value": c["wkind_ok"] / c["wkind_n"] if c["wkind_n"] > 0 else None,
                          "n": int(round(c["wkind_n"])), "unit": "px of P centerline"}
    if c["door_samples"] > 0:
        m.update(_prf(c["door_tp"], c["door_np"], c["door_ng"], "door"))
    else:
        m.update({k: _m(0, 0) for k in ("door_p", "door_r", "door_f1")})
    if c["window_samples"] > 0:
        m.update(_prf(c["window_tp"], c["window_np"], c["window_ng"], "window"))
    else:
        m.update({k: _m(0, 0) for k in ("window_p", "window_r", "window_f1")})
    m["door_as_window"] = _m(c["d2w_hit"], c["d2w_n"])
    m["window_as_door"] = _m(c["w2d_hit"], c["w2d_n"])
    for k in ("door", "window"):
        errs = c[f"{k}_width_err"]
        m[f"{k}_width_err_median"] = {"value": float(np.median(errs)) if errs else None, "n": len(errs)}
    m["door_type_acc"] = _m(c["dtype_ok"], c["dtype_n"])
    m["door_type_acc"]["confusion"] = c["dtype_conf"]
    m["hinge_side_acc"] = _m(c["hinge_ok"], c["hinge_n"])
    m["swing_side_acc"] = _m(c["swing_sum"], c["swing_n"])
    m["door_dir_acc"] = _m(c["dir_ok"], c["dir_n"])
    m["door_dir_recall"] = _m(c["dirrec_ok"], c["dirrec_n"])
    m["double_leaf_count_acc"] = _m(c["leafcnt_ok"], c["leafcnt_n"])
    m["swing_into_acc"] = _m(c["sinto_ok"], c["sinto_n"])
    ho = c["hinge_offset"]
    m["hinge_offset_median"] = {"value": float(np.median(ho)) if ho else None, "n": len(ho), "unit": "width_G"}
    if c["room_samples"] > 0:
        rp = _prf(c["room_tp"], c["room_np"], c["room_ng"], "room", "@0.5")
        m.update(rp)
        f1 = rp["room_f1@0.5"]["value"]
        sq = c["room_iou_sum"] / c["room_tp"] if c["room_tp"] > 0 else 0.0
        m["room_sq"] = {"value": sq, "n": int(c["room_tp"])}
        m["room_pq"] = {"value": sq * f1 if f1 is not None else None, "n": int(c["room_ng"])}
    else:
        m.update({k: _m(0, 0) for k in ("room_p@0.5", "room_r@0.5", "room_f1@0.5", "room_sq", "room_pq")})
    m["room_type_acc"] = _m(c["rtype_ok"], c["rtype_n"])
    m["room_type_acc"]["confusion"] = c["rtype_conf"]
    return {k: m[k] for k in METRIC_ORDER}


# -------------------------------------------------------------------------------------- evaluation

def tolerances(gt):
    w, h = image_size(gt)
    th = [w_["t"] for w_ in parse_walls(gt, 0.0) if w_["t"] > 0]
    t = float(np.median(th)) if th else 0.006 * math.hypot(w, h)
    return {"t": t, "tau1": max(2.0, 0.5 * t), "tau2": max(4.0, 1.0 * t)}


def eval_walls(c, Gw, Pw, w, h, tau):
    c["wall_samples"] += 1
    c["wall_ng"] += len(Gw)
    c["wall_np"] += len(Pw)
    # Wall IoU on exact areas (px^2) inside the image: the continuous limit of rasterising
    try:
        ug, up = wall_area_geom(Gw, w, h), wall_area_geom(Pw, w, h)
        c["wall_inter"] += float(ug.intersection(up).area)
        c["wall_union"] += float(ug.union(up).area)
    except Exception:  # GEOS failure on a pathological P: fall back to the integer raster
        mg, mp = wall_mask(Gw, w, h), wall_mask(Pw, w, h)
        c["wall_inter"] += float(np.count_nonzero(mg & mp))
        c["wall_union"] += float(np.count_nonzero(mg | mp))
    gp, gw, _ = sample_centerlines(Gw)
    # P centerline parts farther than τ2 from the bounding box of all G centerlines can never be
    # within τ of G: clip them away for the distance work and count their length as unmatched.
    # (Exact for every threshold <= τ2, and keeps huge / off-image predictions cheap.)
    ends = np.array([p for g in Gw for p in (g["a"], g["b"])])
    lo, hi = ends.min(0) - (tau["tau2"] + 1.0), ends.max(0) + (tau["tau2"] + 1.0)
    Pc, p_out_len = [], 0.0
    for p in Pw:
        L = float(np.hypot(*(p["b"] - p["a"])))
        seg = clip_segment(np.clip(p["a"], -1e15, 1e15), np.clip(p["b"], -1e15, 1e15), lo, hi)
        if seg is None:
            p_out_len += L
            continue
        Lc = float(np.hypot(*(seg[1] - seg[0])))
        p_out_len += max(0.0, L - Lc)
        Pc.append({**p, "a": seg[0], "b": seg[1]})
    Pw = Pc
    pp, pw, pidx = sample_centerlines(Pw)
    cut = tau["tau2"] + 1.0
    dg = min_dist(gp, Pw, cut)
    dp = min_dist(pp, Gw, cut)
    c["cl_g_len"] += float(gw.sum())
    c["cl_p_len"] += float(pw.sum()) + p_out_len
    for key, tv in (("1", tau["tau1"]), ("2", tau["tau2"])):
        c[f"cl_g_in{key}"] += float(gw[dg <= tv + EPS].sum())
        c[f"cl_p_in{key}"] += float(pw[dp <= tv + EPS].sum())
    # Optional wall kind accuracy: each P sample takes the G wall within τ2 with the lowest
    # distance + direction-misalignment cost (the tie-break keeps junction samples on the right wall).
    if len(pp) and Gw and any(g["kind"] != "unknown" for g in Gw):
        A = np.array([g["a"] for g in Gw])
        B = np.array([g["b"] for g in Gw])
        gd = B - A
        gd = gd / np.maximum(np.hypot(gd[:, 0], gd[:, 1]), EPS)[:, None]
        pd = np.array([p["b"] - p["a"] for p in Pw])
        pd = pd / np.maximum(np.hypot(pd[:, 0], pd[:, 1]), EPS)[:, None]
        mis = 1.0 - np.abs(pd @ gd.T)  # (Mp, Mg)
        gk = np.array([g["kind"] for g in Gw])
        pkind = np.array([p["kind"] for p in Pw])
        for r, J, d in near_distances(pp, A, B, cut):
            ii = pidx[r]
            cost = np.where(d <= tau["tau2"] + EPS, d + tau["tau2"] * mis[ii][:, J], np.inf)
            best = cost.min(1, keepdims=True)
            # exact ties = overlapping collinear G walls (CubiCasa has a few with different kinds):
            # any known kind in the tie set is accepted
            gkJ = gk[J]
            tie = np.isfinite(best) & (cost <= best + 1e-6) & (gkJ[None] != "unknown")
            use = tie.any(1)
            pk = pkind[ii]
            hit = (tie & (gkJ[None] == pk[:, None])).any(1)
            wts = pw[r]
            c["wkind_n"] += float(wts[use].sum())
            c["wkind_ok"] += float(wts[use & hit].sum())


def match_rooms(Gr, Pr):
    if not Gr or not Pr:
        return []
    iou = np.zeros((len(Gr), len(Pr)))
    for i, g in enumerate(Gr):
        if g["poly"].is_empty:
            continue
        gb = g["poly"].bounds
        for j, p in enumerate(Pr):
            if p["poly"].is_empty:
                continue
            pb = p["poly"].bounds
            if pb[0] > gb[2] or pb[2] < gb[0] or pb[1] > gb[3] or pb[3] < gb[1]:
                continue
            try:
                inter = g["poly"].intersection(p["poly"]).area
                union = g["poly"].union(p["poly"]).area
            except Exception:
                continue
            iou[i, j] = inter / union if union > 0 else 0.0
    pairs = hungarian(1.0 - iou, iou >= ROOM_IOU_GATE - EPS)
    return [(i, j, float(iou[i, j])) for i, j in pairs]


def leaf_frame(g):
    a, b = g["a"], g["b"]
    u = (b - a) / np.hypot(*(b - a))
    return (a + b) / 2, u, np.array([-u[1], u[0]])


def scorable_direction(g, w, h):
    """Section 4 item set: G doors with leaves, swing_scorable != false, hinges inside the image."""
    if g["category"] != "door" or not g["leaves"] or g["swing_scorable"] is False or g["dir"] is None:
        return False
    return all(0 <= lf["hinge"][0] <= w and 0 <= lf["hinge"][1] <= h for lf in g["leaves"])


def eval_direction(c, g, p, frame):
    """Score one scorable G door against its matched P door (p may be None = unmatched)."""
    m, u, n = frame
    hs = lambda lf: _sign(float((lf["hinge"] - m) @ u))
    ss = lambda lf: _sign(float((lf["open_tip"] - lf["hinge"]) @ n))
    c["dirrec_n"] += 1
    if p is None:
        return
    gl, pl = g["leaves"], p["leaves"]
    if len(gl) == 1:
        c["hinge_n"] += 1
        c["swing_n"] += 1
        c["dir_n"] += 1
        if not pl:
            return
        lp = pl[0]
        hok = hs(lp) == hs(gl[0])
        sok = ss(lp) == ss(gl[0])
        c["hinge_ok"] += hok
        c["swing_sum"] += sok
        c["dir_ok"] += hok and sok
        c["dirrec_ok"] += hok and sok
        c["hinge_offset"].append(float(np.hypot(*(lp["hinge"] - gl[0]["hinge"]))) / g["width"])
        return
    # double (or more) leaves: pair P leaves with G leaves by hinge side
    c["swing_n"] += 1
    c["dir_n"] += 1
    c["leafcnt_n"] += 1
    used = set()
    oks = []
    for lg in gl:
        hit = None
        for j, lp in enumerate(pl):
            if j not in used and hs(lp) == hs(lg) != 0:
                hit = j
                break
        if hit is None:
            oks.append(False)
            continue
        used.add(hit)
        oks.append(ss(pl[hit]) == ss(lg))
        c["hinge_offset"].append(float(np.hypot(*(pl[hit]["hinge"] - lg["hinge"]))) / g["width"])
    lc = len(pl) == 2 and hs(pl[0]) * hs(pl[1]) == -1
    c["leafcnt_ok"] += lc
    c["swing_sum"] += sum(oks) / len(oks)
    c["dir_ok"] += lc and all(oks)
    c["dirrec_ok"] += lc and all(oks)


def evaluate_pair(pred, gt):
    """Score one prediction (dict, or None = empty) against one GT. Returns (counters, info)."""
    w, h = image_size(gt)
    fid = fidelity(gt)
    tau = tolerances(gt)
    c = new_counters()
    c["n_samples"] = 1
    pred = pred if isinstance(pred, dict) else {}
    scored = {}

    # walls
    Gw = parse_walls(gt, tau["t"])
    Pw = parse_walls(pred, tau["t"])
    scored["walls"] = fid.get("walls") in ("full", "geometry_only") and bool(Gw)
    if scored["walls"]:
        eval_walls(c, Gw, Pw, w, h, tau)

    # openings
    Go, Po = parse_openings(gt), parse_openings(pred)
    pairs = {k: match_openings(Go[k], Po[k], tau["tau2"]) for k in ("door", "window")}
    scored["doors"] = bool(Go["door"])
    scored["windows"] = bool(fid.get("windows")) and bool(Go["window"])
    for k in ("door", "window"):
        if not scored[k + "s"]:
            continue
        c[f"{k}_samples"] += 1
        c[f"{k}_tp"] += len(pairs[k])
        c[f"{k}_np"] += len(Po[k])
        c[f"{k}_ng"] += len(Go[k])
        c[f"{k}_width_err"] += [abs(Po[k][j]["width"] - Go[k][i]["width"]) / Go[k][i]["width"] for i, j in pairs[k]]
    # door <-> window confusion: unmatched G of one category against unmatched P of the other
    for gk, pk, key, on in (("door", "window", "d2w", scored["doors"]), ("window", "door", "w2d", scored["windows"])):
        if not on:
            continue
        gi = sorted(set(range(len(Go[gk]))) - {i for i, _ in pairs[gk]})
        pj = sorted(set(range(len(Po[pk]))) - {j for _, j in pairs[pk]})
        cross = match_openings([Go[gk][i] for i in gi], [Po[pk][j] for j in pj], tau["tau2"])
        c[f"{key}_n"] += len(gi)
        c[f"{key}_hit"] += len(cross)
    door_match = dict(pairs["door"])  # G door index -> P door index

    # door type
    scored["door_type"] = bool(fid.get("door_type")) and scored["doors"]
    if scored["door_type"]:
        for i, j in pairs["door"]:
            gtyp = Go["door"][i]["type"]
            if gtyp == "unknown" or gtyp not in DOOR_CLASSES:
                continue
            ptyp = Po["door"][j]["type"]
            c["dtype_n"] += 1
            c["dtype_ok"] += ptyp == gtyp
            _conf(c["dtype_conf"], gtyp, ptyp)

    # rooms (matching is also needed for swing_into)
    Gr, Pr = parse_rooms(gt), parse_rooms(pred)
    rpairs = match_rooms(Gr, Pr)
    room_map = {Gr[i]["id"]: Pr[j]["id"] for i, j, _ in rpairs}
    scored["rooms"] = bool(fid.get("rooms")) and bool(Gr)
    if scored["rooms"]:
        c["room_samples"] += 1
        c["room_tp"] += len(rpairs)
        c["room_np"] += len(Pr)
        c["room_ng"] += len(Gr)
        c["room_iou_sum"] += sum(v for _, _, v in rpairs)
    scored["room_types"] = bool(fid.get("room_types")) and scored["rooms"]
    if scored["room_types"]:
        for i, j, _ in rpairs:
            gtyp = Gr[i]["type"]
            if gtyp not in ROOM_CLASSES:
                continue
            ptyp = Pr[j]["type"]
            c["rtype_n"] += 1
            c["rtype_ok"] += ptyp == gtyp
            _conf(c["rtype_conf"], gtyp, ptyp)

    # door direction
    scored["door_direction"] = fid.get("door_swing") == "full" and scored["doors"]
    n_dir_items = 0
    if scored["door_direction"]:
        c["dir_samples"] += 1
        for i, g in enumerate(Go["door"]):
            if not scorable_direction(g, w, h):
                continue
            n_dir_items += 1
            p = Po["door"][door_match[i]] if i in door_match else None
            eval_direction(c, g, p, leaf_frame(g))
            if p is not None and g["swing_into"] is not None:
                c["sinto_n"] += 1
                if g["swing_into"] == "outside":
                    c["sinto_ok"] += p["swing_into"] == "outside"
                else:
                    pr = room_map.get(g["swing_into"])
                    c["sinto_ok"] += pr is not None and p["swing_into"] == pr
    scored["door_direction"] = scored["door_direction"] and n_dir_items > 0
    return c, {"tau": {k: round(v, 3) for k, v in tau.items()}, "scored": scored}


# ----------------------------------------------------------------------------------- dataset runs

def load_manifest(testdata):
    m = json.load(open(os.path.join(testdata, "all_manifest.json")))
    return {os.path.splitext(e["file"])[0]: e for e in m}


def load_split(testdata, split, manifest):
    if split == "all":
        return sorted(manifest)
    s = json.load(open(os.path.join(testdata, "splits.json")))
    return list(s[split])


def find_pred(pred_dir, sid):
    name = os.path.basename(sid) + ".json"
    for cand in (os.path.join(pred_dir, name), os.path.join(pred_dir, sid + ".json")):
        if os.path.isfile(cand):
            return cand
    return None


def _clean(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def score_sample(sid, pred, gt, pred_status):
    """pred_status: dict(has_pred, pred_valid, pred_errors). Never raises on a bad prediction."""
    try:
        return evaluate_pair(pred, gt)
    except Exception as exc:
        pred_status["pred_valid"] = False
        pred_status["eval_error"] = f"{type(exc).__name__}: {exc}"
    # Only the part of the prediction that breaks the evaluator is dropped (scored as empty), so a
    # single pathological wall cannot wipe out the door / room scores of the same sample.
    pred = pred if isinstance(pred, dict) else {}
    groups = ("walls", "openings", "rooms")
    for drop in [(g,) for g in groups] + [tuple(x for x in groups if x != g) for g in groups]:
        try:
            c, info = evaluate_pair({k: v for k, v in pred.items() if k not in drop}, gt)
        except Exception:
            continue
        pred_status["eval_error"] += f" -> scored with P {'/'.join(drop)} dropped"
        return c, info
    pred_status["eval_error"] += " -> scored as an empty prediction"
    return evaluate_pair(None, gt)


def run_split(pred_dir, split, testdata):
    manifest = load_manifest(testdata)
    ids = load_split(testdata, split, manifest)
    per_sample, per_ds = [], {}
    overall = new_counters()
    rob = {}
    for sid in ids:
        ent = manifest[sid]
        ds = ent["dataset"]
        gt = json.load(open(os.path.join(testdata, ent["gt_file"])))
        path = find_pred(pred_dir, sid)
        status = {"has_pred": path is not None, "pred_valid": False, "pred_errors": []}
        pred = None
        if path is not None:
            try:
                with open(path) as f:
                    pred = json.load(f)
                errs = validate_pred(pred, gt)
                status["pred_valid"] = not errs
                status["pred_errors"] = errs[:20]
            except Exception as exc:
                status["pred_errors"] = [f"cannot read prediction: {type(exc).__name__}: {exc}"]
                pred = None
        else:
            status["pred_errors"] = ["missing prediction (scored as empty)"]
        c, info = score_sample(sid, pred, gt, status)
        add_counters(per_ds.setdefault(ds, new_counters()), c)
        add_counters(overall, c)
        r = rob.setdefault(ds, {"n_samples": 0, "n_missing": 0, "n_invalid": 0, "n_eval_errors": 0})
        r["n_samples"] += 1
        r["n_missing"] += not status["has_pred"]
        r["n_invalid"] += status["has_pred"] and not status["pred_valid"]
        r["n_eval_errors"] += "eval_error" in status
        per_sample.append({"id": sid, "dataset": ds, **status, **info, "metrics": finalize(c)})
    datasets = [d for d in DATASETS if d in per_ds] + sorted(d for d in per_ds if d not in DATASETS)
    rob_all = {k: sum(r[k] for r in rob.values()) for k in ("n_samples", "n_missing", "n_invalid", "n_eval_errors")}
    report = {
        "eval": "floorplan eval-v0", "split": split, "pred_dir": os.path.abspath(pred_dir),
        "notes": ["floorplancad wall GT is incomplete: read wall precision there as a lower bound"] * ("floorplancad" in per_ds)
                 + ["per-dataset and overall values are micro-averages of the raw counts",
                    "wikimedia-commons samples have no GT: robustness only (prediction present, valid, no crash)"],
        "per_dataset": {d: {"robustness": rob[d], "metrics": finalize(per_ds[d])} for d in datasets},
        "overall": {"robustness": rob_all, "metrics": finalize(overall)},
        "per_sample": per_sample,
    }
    return _clean(report)


# --------------------------------------------------------------------------------------- markdown

MD_COLS = [("wall_iou", "wall IoU"), ("wall_cl_f1@τ1", "CL F1@τ1"), ("wall_cl_f1@τ2", "CL F1@τ2"),
           ("door_f1", "door F1"), ("window_f1", "win F1"), ("door_type_acc", "door type"),
           ("hinge_side_acc", "hinge side"), ("swing_side_acc", "swing side"), ("door_dir_acc", "dir acc"),
           ("door_dir_recall", "dir recall"), ("swing_into_acc", "swing_into"),
           ("room_f1@0.5", "room F1"), ("room_pq", "room PQ"), ("room_type_acc", "room type")]


def _cell(m):
    if m is None or m.get("value") is None:
        return "–"
    return f"{m['value']:.3f} ({m['n']})"


def to_markdown(report):
    lines = [f"### floorplan eval-v0 — split `{report.get('split')}`", "",
             "value (n). n = items behind the number (G items for recall/F1, matched items for accuracies). "
             "– = not applicable.", ""]
    head = ["dataset", "samples (miss/inval)"] + [h for _, h in MD_COLS]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = list(report["per_dataset"].items()) + [("**overall**", report["overall"])]
    for name, block in rows:
        r = block.get("robustness", {})
        cells = [name, f"{r.get('n_samples', 1)} ({r.get('n_missing', 0)}/{r.get('n_invalid', 0)})"]
        cells += [_cell(block["metrics"].get(k)) for k, _ in MD_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    if "floorplancad" in report["per_dataset"]:
        lines += ["", "FloorPlanCAD wall GT is incomplete: its wall precision (and F1) is a lower bound."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- CLI

def pred_from_gt(out_dir, split, testdata):
    manifest = load_manifest(testdata)
    os.makedirs(out_dir, exist_ok=True)
    ids = load_split(testdata, split, manifest)
    for sid in ids:
        gt = json.load(open(os.path.join(testdata, manifest[sid]["gt_file"])))
        p = copy.deepcopy(gt)
        p.pop("gt_fidelity", None)
        with open(os.path.join(out_dir, os.path.basename(sid) + ".json"), "w") as f:
            json.dump(p, f)
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir")
    ap.add_argument("--split", choices=["dev", "test", "all"])
    ap.add_argument("--pred")
    ap.add_argument("--gt")
    ap.add_argument("--pred-from-gt", metavar="OUT_DIR")
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--testdata", default=ROOT)
    a = ap.parse_args(argv)
    try:  # metric names contain "τ": never crash on a non-UTF-8 terminal / locale
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if a.pred_from_gt:
        ids = pred_from_gt(a.pred_from_gt, a.split or "all", a.testdata)
        print(f"wrote {len(ids)} P=G predictions to {a.pred_from_gt}")
        return 0
    if a.pred or a.gt:
        if not (a.pred and a.gt):
            ap.error("single-pair mode needs both --pred and --gt")
        gt = json.load(open(a.gt))
        status = {"has_pred": os.path.isfile(a.pred), "pred_valid": False, "pred_errors": []}
        pred = None
        if status["has_pred"]:
            try:
                pred = json.load(open(a.pred))
                errs = validate_pred(pred, gt)
                status.update(pred_valid=not errs, pred_errors=errs[:20])
            except Exception as exc:
                status["pred_errors"] = [f"cannot read prediction: {exc}"]
        c, info = score_sample(a.pred, pred, gt, status)
        rob = {"n_samples": 1, "n_missing": int(not status["has_pred"]),
               "n_invalid": int(status["has_pred"] and not status["pred_valid"]),
               "n_eval_errors": int("eval_error" in status)}
        metrics = finalize(c)
        report = _clean({"eval": "floorplan eval-v0", "split": "pair", "pred": a.pred, "gt": a.gt, **status, **info,
                         "per_dataset": {}, "overall": {"robustness": rob, "metrics": metrics}})
    else:
        if not (a.pred_dir and a.split):
            ap.error("need --pred-dir and --split (or --pred/--gt, or --pred-from-gt)")
        report = run_split(a.pred_dir, a.split, a.testdata)
    md = to_markdown(report)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    if a.pred or a.gt:
        print(json.dumps({k: v["value"] for k, v in report["overall"]["metrics"].items()}, indent=1, ensure_ascii=False))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
