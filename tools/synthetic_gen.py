#!/usr/bin/env python3
"""synthetic_gen.py - deterministic synthetic floor-plan generator with exact floorplan/v0 ground truth.

Run with the project venv (needs shapely + opencv + numpy):
  .venv/bin/python -I tools/synthetic_gen.py --all --out synthetic
  .venv/bin/python -I tools/synthetic_gen.py --index 7 --out /tmp/x

Every sample is fully determined by its row in TABLE (seed + knobs). All geometry is built in
millimetres (x right, y down), rendered with OpenCV, then optionally degraded. Every ground-truth
point is pushed through exactly the same mm->px map and the same rotation matrix as the pixels.

Conventions written into each GT file (see "conventions" key):
  * jambs: the two opening ends on the wall CENTERLINE (a = jambs[0], b = jambs[1]).
  * leaves[].hinge / closed_tip: the jamb corners on the wall FACE on the swing side (that is where
    the leaf and arc are drawn); open_tip = hinge + width * swing normal (leaf drawn fully open, 90 deg).
  * handedness: schema convention (US letters, viewer stands on the side the door swings AWAY from,
    facing the door): hinge on viewer's left = "LH", on the right = "RH". Because the viewer is always
    on the non-swing side, LHR/RHR never occur. Only single-leaf doors get a handedness.
  * swing_into: room id the leaf sweeps into, or "outside".
  * connects: [room on non-swing side, room on swing side] for swing doors; otherwise both rooms.
  * room polygon: clear (wall-face) polygon; polygon_centerline: wall-centerline polygon.
  * polygon winding: positive shoelace area in image coordinates (x right, y down), i.e.
    mathematically counter-clockwise in the image frame (this appears clockwise on screen).
  * pocket door: slide.track_side = jamb ('a'/'b') whose side holds the wall pocket,
    slide.direction = unit vector the panel moves to open. Bypass sliding: track_side/direction null.
"""
import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

GRID = 50.0
MIN_DIM = 1900.0
KEY = "synthetic"

# ----------------------------------------------------------------------------------------------
# Per-sample knobs. Rows 0-11 are clean renders (JPEG q95) with clutter growing; rows 12-23 add
# degradations whose strength grows with the "deg" level (see DEGRADE).
#   rooms, l_shape, chamfers(diagonal walls), corridor, wall style, text, dims, furniture,
#   extras, long side px, degrade level
# ----------------------------------------------------------------------------------------------
TABLE = [
    (3, False, 0, False, "solid",   False, False, False, [],                                   1000, 0),
    (4, False, 0, False, "hollow",  False, False, False, [],                                   1200, 0),
    (5, True,  0, False, "hatched", True,  False, False, [],                                   1400, 0),
    (5, False, 1, True,  "solid",   True,  True,  False, [],                                   1600, 0),
    (6, True,  0, True,  "hollow",  True,  True,  False, ["scalebar"],                         1800, 0),
    (6, False, 0, False, "gray",    False, False, True,  [],                                   1500, 0),
    (7, True,  0, True,  "hatched", True,  False, True,  ["tiles"],                            2000, 0),
    (7, False, 2, True,  "solid",   True,  True,  True,  ["scalebar", "north"],                2200, 0),
    (8, True,  1, True,  "hollow",  True,  True,  True,  ["stairs", "area"],                   2400, 0),
    (8, False, 0, True,  "hatched", True,  True,  True,  ["tiles", "axisgrid"],                2600, 0),
    (9, True,  0, True,  "solid",   True,  True,  True,  ["color", "area", "title"],           2800, 0),
    (9, False, 1, True,  "hollow",  True,  True,  True,  ["graphpaper", "stairs", "scalebar"], 3200, 0),
    (3, False, 0, False, "solid",   True,  False, False, [],                                   1000, 1),
    (4, True,  0, False, "hollow",  True,  True,  False, [],                                   1300, 2),
    (5, False, 1, False, "hatched", True,  True,  True,  [],                                   1600, 3),
    (6, True,  0, True,  "gray",    True,  True,  True,  ["tiles"],                            1800, 4),
    (6, False, 0, True,  "solid",   True,  True,  True,  ["color"],                            2000, 5),
    (7, True,  1, True,  "hollow",  True,  True,  True,  ["stairs", "title"],                  2200, 6),
    (7, False, 2, True,  "hatched", True,  True,  True,  ["axisgrid", "area"],                 2400, 7),
    (8, True,  0, True,  "solid",   True,  True,  True,  ["graphpaper", "tiles"],              2000, 8),
    (8, False, 0, True,  "hollow",  True,  True,  True,  ["scalebar", "north", "title"],       1600, 9),
    (9, True,  1, True,  "gray",    True,  True,  True,  ["stairs", "tiles", "area"],          1400, 10),
    (5, False, 0, False, "hatched", True,  True,  True,  ["color", "graphpaper"],              1100, 11),
    (9, True,  2, True,  "solid",   True,  True,  True,  ["axisgrid", "stairs", "title", "scalebar"], 900, 12),
]

# level -> degradation knobs (rot in degrees, CCW on screen; blur = gaussian sigma px; noise = sigma)
DEGRADE = {
    0: dict(jpeg=95),
    1: dict(jpeg=70, rot=1.0),
    2: dict(jpeg=66, blur=0.6),
    3: dict(jpeg=63, rot=-1.5, noise=4),
    4: dict(jpeg=60, tint=True, light=0.15),
    5: dict(jpeg=58, rot=2.0, blur=0.8, noise=5),
    6: dict(jpeg=55, tint=True, light=0.2, noise=6),
    7: dict(jpeg=52, rot=-2.5, blur=1.0, tint=True),
    8: dict(jpeg=50, rot=3.0, noise=8, light=0.25),
    9: dict(jpeg=48, blur=1.1, tint=True, light=0.3, specks=True),
    10: dict(jpeg=45, rot=-3.5, blur=1.2, noise=9, tint=True, fade=0.8),
    11: dict(jpeg=42, rot=4.0, noise=10, light=0.35, specks=True, tint=True),
    12: dict(jpeg=40, rot=-4.0, blur=1.4, noise=12, tint=True, light=0.4, fade=0.75, specks=True),
}


def config_for(idx):
    r = TABLE[idx]
    return dict(index=idx, seed=20261013 + 7919 * idx, n_rooms=r[0], l_shape=r[1], chamfers=r[2],
                corridor=r[3], wall_style=r[4], text=r[5], dims=r[6], furniture=r[7], extras=list(r[8]),
                long_px=r[9], degrade_level=r[10], degrade=dict(DEGRADE[r[10]]))


# ----------------------------------------------------------------------------------------------
# small geometry helpers (numpy 2-vectors, mm)
# ----------------------------------------------------------------------------------------------
def V(x, y=None):
    if y is None:
        return np.array(x, dtype=float)
    return np.array([x, y], dtype=float)


def unit(v):
    return v / np.linalg.norm(v)


def perp(u):  # +90 deg in the (x right, y down) frame
    return V(-u[1], u[0])


def snap(v, g=GRID):
    return float(round(v / g) * g)


def rect_poly(p0, p1, half_t, ext=0.0):
    p0, p1 = V(p0), V(p1)
    u = unit(p1 - p0)
    n = perp(u)
    a, b = p0 - u * ext, p1 + u * ext
    return Polygon([a + n * half_t, b + n * half_t, b - n * half_t, a - n * half_t])


def quad(a, b, n, d0, d1):
    return Polygon([a + n * d0, b + n * d0, b + n * d1, a + n * d1])


def line_intersect(p, d, q, e):
    M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    s = np.linalg.solve(M, q - p)[0]
    return p + s * d


def clean_ring(coords, tol=1e-6):
    pts = [V(c) for c in coords]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cr = (b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]
            if np.linalg.norm(b - a) < 1e-6 or abs(cr) < tol * max(1.0, np.linalg.norm(b - a) * np.linalg.norm(c - b)):
                pts.pop(i)
                changed = True
                break
    return pts


def largest(g):
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda p: p.area)
    if g.geom_type == "GeometryCollection":
        ps = [p for p in g.geoms if p.geom_type == "Polygon"]
        return max(ps, key=lambda p: p.area)
    return g


def poly_parts(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in g.geoms if isinstance(p, Polygon)]


def offset_vertices(verts, dist):
    """Mitred offset of a positively oriented polygon; dist>0 = outward."""
    n = len(verts)
    out = []
    for i in range(n):
        pp, p, pn = verts[i - 1], verts[i], verts[(i + 1) % n]
        d1, d2 = unit(p - pp), unit(pn - p)
        n1, n2 = V(d1[1], -d1[0]), V(d2[1], -d2[0])
        out.append(line_intersect(pp + n1 * dist, d1, p + n2 * dist, d2))
    return out


def merge_iv(ivs):
    out = []
    for lo, hi in sorted(ivs):
        if out and lo <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def subtract_iv(iv, cuts):
    res = [list(iv)]
    for clo, chi in cuts:
        new = []
        for lo, hi in res:
            if chi <= lo or clo >= hi:
                new.append([lo, hi])
                continue
            if clo > lo:
                new.append([lo, clo])
            if chi < hi:
                new.append([chi, hi])
        res = new
    return [r for r in res if r[1] - r[0] > 1]


def axis_of(a, b):
    if abs(a[1] - b[1]) < 1e-6:
        return ("h", round(float(a[1]), 3), float(min(a[0], b[0])), float(max(a[0], b[0])))
    if abs(a[0] - b[0]) < 1e-6:
        return ("v", round(float(a[0]), 3), float(min(a[1], b[1])), float(max(a[1], b[1])))
    return None


def axis_pts(axis, c, lo, hi):
    if axis == "h":
        return V(lo, c), V(hi, c)
    return V(c, lo), V(c, hi)


def ring_edges(poly):
    pts = list(poly.exterior.coords)[:-1]
    return [(V(pts[i]), V(pts[(i + 1) % len(pts)])) for i in range(len(pts))]


def arc_points(center, p_from, p_to, n=24):
    r = np.linalg.norm(p_from - center)
    a0 = math.atan2(*(p_from - center)[::-1])
    a1 = math.atan2(*(p_to - center)[::-1])
    d = (a1 - a0 + 3 * math.pi) % (2 * math.pi) - math.pi
    return [center + r * V(math.cos(a0 + d * k / n), math.sin(a0 + d * k / n)) for k in range(n + 1)]


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------
def bsp(rect, k, rng):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if k == 1:
        if min(w, h) < MIN_DIM or max(w, h) / min(w, h) > 3.0:
            return None
        return [rect]
    for _ in range(14):
        k1 = rng.randint(1, k - 1)
        vertical = (w >= h) if rng.random() < 0.85 else (w < h)
        L = w if vertical else h
        cut = snap(L * min(0.8, max(0.2, k1 / k + rng.uniform(-0.1, 0.1))))
        if cut < MIN_DIM or L - cut < MIN_DIM:
            continue
        if vertical:
            A, B = (x0, y0, x0 + cut, y1), (x0 + cut, y0, x1, y1)
        else:
            A, B = (x0, y0, x1, y0 + cut), (x0, y0 + cut, x1, y1)
        ra = bsp(A, k1, rng)
        if ra is None:
            continue
        rb = bsp(B, k - k1, rng)
        if rb is None:
            continue
        return ra + rb
    return None


def alloc(areas, k):
    tot = sum(areas)
    raw = [max(1, round(k * a / tot)) for a in areas]
    while sum(raw) > k:
        i = max(range(len(raw)), key=lambda j: raw[j])
        raw[i] -= 1
    while sum(raw) < k:
        i = max(range(len(raw)), key=lambda j: areas[j] / raw[j])
        raw[i] += 1
    return raw


def make_layout(rng, cfg):
    n = cfg["n_rooms"]
    has_corr = cfg["corridor"] and n >= 4
    for _ in range(400):
        per = rng.uniform(11.5e6, 17e6)
        area = n * per
        aspect = rng.uniform(1.2, 1.75)
        nwf, nhf = rng.uniform(0.3, 0.45), rng.uniform(0.3, 0.5)
        if cfg["l_shape"]:
            area /= (1 - nwf * nhf)
        W = snap(math.sqrt(area * aspect))
        H = snap(area / W)
        if cfg["l_shape"]:
            nw, nh = snap(W * nwf), snap(H * nhf)
            corner = rng.choice(["tl", "tr", "bl", "br"])
            zones = {"tr": [(0, 0, W - nw, H), (W - nw, nh, W, H)],
                     "tl": [(nw, 0, W, H), (0, nh, nw, H)],
                     "br": [(0, 0, W - nw, H), (W - nw, 0, W, H - nh)],
                     "bl": [(nw, 0, W, H), (0, 0, nw, H - nh)]}[corner]
        else:
            zones = [(0, 0, W, H)]
        k_rooms = n - (1 if has_corr else 0)
        areas = [(z[2] - z[0]) * (z[3] - z[1]) for z in zones]
        if k_rooms < len(zones):
            continue
        ks = alloc(areas, k_rooms)
        rects, corr = [], None
        ok = True
        for zi, (z, kz) in enumerate(zip(zones, ks)):
            subz = [(z, kz)]
            if has_corr and zi == max(range(len(zones)), key=lambda j: areas[j]):
                x0, y0, x1, y1 = z
                cw = snap(rng.uniform(1150, 1500))
                if kz < 2:
                    ok = False
                    break
                if x1 - x0 >= y1 - y0:
                    pos = snap(y0 + (y1 - y0 - cw) * rng.uniform(0.35, 0.65))
                    s1, corr, s2 = (x0, y0, x1, pos), (x0, pos, x1, pos + cw), (x0, pos + cw, x1, y1)
                else:
                    pos = snap(x0 + (x1 - x0 - cw) * rng.uniform(0.35, 0.65))
                    s1, corr, s2 = (x0, y0, pos, y1), (pos, y0, pos + cw, y1), (pos + cw, y0, x1, y1)
                a1 = (s1[2] - s1[0]) * (s1[3] - s1[1])
                a2 = (s2[2] - s2[0]) * (s2[3] - s2[1])
                k1, k2 = alloc([a1, a2], kz)
                subz = [(s1, k1), (s2, k2)]
            for sz, kk in subz:
                r = bsp(sz, kk, rng)
                if r is None:
                    ok = False
                    break
                rects += r
            if not ok:
                break
        if not ok:
            continue
        if corr is not None:
            rects.append(corr)
        if len(rects) != n:
            continue
        corr_i = len(rects) - 1 if corr is not None else None
        if rng.random() < 0.5:  # transpose for orientation variety
            rects = [(r[1], r[0], r[3], r[2]) for r in rects]
        return rects, corr_i
    return None, None


ROOM_COLORS = {"living": (200, 228, 250), "kitchen": (190, 240, 240), "bedroom": (235, 220, 200),
               "bath": (245, 235, 205), "corridor": (225, 225, 225), "storage": (215, 215, 230),
               "other": (210, 240, 215)}


def assign_types(rooms, corr_i, rng):
    order = sorted([r for r in rooms if r["i"] != corr_i], key=lambda r: -r["poly"].area)
    for r in rooms:
        r["type"] = None
    if corr_i is not None:
        rooms[corr_i]["type"] = "corridor"
        rooms[corr_i]["label"] = rng.choice(["HALL", "CORRIDOR", "ENTRY"])
    nr = len(order)
    order[0]["type"] = "living"
    order[-1]["type"] = "bath"
    if nr >= 4:
        rng.choice(order[1:-1])["type"] = "kitchen"
    rest = [r for r in order if r["type"] is None]
    if nr >= 6 and rest and rest[-1]["poly"].area < 7e6:
        rest[-1]["type"] = "bath"
        rest[-1]["label"] = "WC"
        rest = rest[:-1]
    if nr >= 7 and rest and rest[-1]["poly"].area < 6e6:
        rest[-1]["type"] = "storage"
        rest = rest[:-1]
    if nr >= 8 and len(rest) >= 3:
        rest[-1]["type"] = "other"
        rest[-1]["label"] = rng.choice(["STUDY", "OFFICE"])
        rest = rest[:-1]
    for r in rest:
        r["type"] = "bedroom"
    bed_word = rng.choice(["BEDROOM", "BED"])
    nb = sum(1 for r in rooms if r["type"] == "bedroom")
    k = 1
    for r in rooms:
        if "label" in r:
            continue
        t = r["type"]
        if t == "living":
            r["label"] = rng.choice(["LIVING", "LIVING ROOM", "LOUNGE"]) if nr >= 4 else "LIVING / KITCHEN"
        elif t == "kitchen":
            r["label"] = "KITCHEN"
        elif t == "bath":
            r["label"] = rng.choice(["BATH", "BATHROOM"])
        elif t == "storage":
            r["label"] = rng.choice(["STORAGE", "CLOSET"])
        elif t == "bedroom":
            r["label"] = f"{bed_word} {k}" if nb > 1 else bed_word
            k += 1


PRIV = {"corridor": 0, "living": 1, "kitchen": 2, "other": 3, "bedroom": 4, "bath": 5, "storage": 5}


# ----------------------------------------------------------------------------------------------
# plan building
# ----------------------------------------------------------------------------------------------
class Plan:
    pass


def build_plan(rng, cfg):
    rects, corr_i = make_layout(rng, cfg)
    if rects is None:
        return None
    P = Plan()
    P.cfg = cfg
    P.t_ext = snap(rng.uniform(250, 400), 10)
    P.t_int = snap(rng.uniform(100, 150), 10)
    P.m_end = P.t_ext / 2 + 120
    rooms = []
    for i, r in enumerate(rects):
        rooms.append(dict(i=i, id=f"r{i + 1}", rect=r, poly=box(*r)))
    assign_types(rooms, corr_i, rng)

    # --- diagonal (chamfered) exterior corners
    outline = orient(unary_union([r["poly"] for r in rooms]), 1.0)
    P.chamfered = []
    if cfg["chamfers"]:
        verts = clean_ring(outline.exterior.coords)
        cands = []
        for vi, v in enumerate(verts):
            pp, pn = verts[vi - 1], verts[(vi + 1) % len(verts)]
            cr = (v - pp)[0] * (pn - v)[1] - (v - pp)[1] * (pn - v)[0]
            if cr <= 0:
                continue
            for r in rooms:
                x0, y0, x1, y1 = r["rect"]
                if r["type"] in ("corridor",) or min(x1 - x0, y1 - y0) < 2800:
                    continue
                if any(np.allclose(v, c) for c in [V(x0, y0), V(x1, y0), V(x1, y1), V(x0, y1)]):
                    cands.append((vi, v, pp, pn, r))
        rng.shuffle(cands)
        used = set()
        for vi, v, pp, pn, r in cands:
            if len(P.chamfered) >= cfg["chamfers"]:
                break
            if r["i"] in used:
                continue
            x0, y0, x1, y1 = r["rect"]
            leg = snap(rng.uniform(900, min(2000, 0.42 * min(x1 - x0, y1 - y0))))
            tri = Polygon([v, v + leg * unit(pp - v), v + leg * unit(pn - v)])
            r["poly"] = r["poly"].difference(tri)
            used.add(r["i"])
            P.chamfered.append(r["id"])
        outline = orient(unary_union([r["poly"] for r in rooms]), 1.0)
    for r in rooms:
        r["poly"] = orient(Polygon(clean_ring(r["poly"].exterior.coords)), 1.0)
    overts = clean_ring(outline.exterior.coords)
    outline = orient(Polygon(overts), 1.0)
    overts = clean_ring(outline.exterior.coords)
    P.outline = outline
    P.outer = Polygon(offset_vertices(overts, P.t_ext / 2))

    # --- walls
    walls = []
    outv = offset_vertices(overts, P.t_ext / 2)
    inv = offset_vertices(overts, -P.t_ext / 2)
    out_iv = {}
    for i in range(len(overts)):
        a, b = overts[i], overts[(i + 1) % len(overts)]
        j = (i + 1) % len(overts)
        qp = Polygon([inv[i], inv[j], outv[j], outv[i]])
        walls.append(dict(id=f"w{len(walls) + 1}", p0=a, p1=b, t=P.t_ext, kind="exterior", gt_poly=qp, render=qp))
        s = axis_of(a, b)
        if s:
            out_iv.setdefault((s[0], s[1]), []).append((s[2], s[3]))
    int_iv = {}
    for r in rooms:
        for a, b in ring_edges(r["poly"]):
            s = axis_of(a, b)
            if not s:
                continue
            for piece in subtract_iv((s[2], s[3]), out_iv.get((s[0], s[1]), [])):
                int_iv.setdefault((s[0], s[1]), []).append(tuple(piece))
    for key in sorted(int_iv):
        for lo, hi in merge_iv(int_iv[key]):
            a, b = axis_pts(key[0], key[1], lo, hi)
            trims = []
            for p in (a, b):
                trims.append(P.t_ext / 2 if P.outline.exterior.distance(Point(p)) < 1 else P.t_int / 2)
            u = unit(b - a)
            gp = rect_poly(a + u * trims[0], b - u * trims[1], P.t_int / 2)
            walls.append(dict(id=f"w{len(walls) + 1}", p0=a, p1=b, t=P.t_int, kind="interior", gt_poly=gp,
                              render=rect_poly(a, b, P.t_int / 2, ext=P.t_int / 2), axis=key, iv=(lo, hi)))
    P.walls = walls
    P.wall_union = unary_union([w["render"] for w in walls])
    for r in rooms:
        r["interior"] = orient(largest(r["poly"].difference(P.wall_union)).simplify(0.5), 1.0)
    P.rooms = rooms
    P.room_by_id = {r["id"]: r for r in rooms}

    # --- slots (wall pieces where an opening may go)
    def side_rooms(p0, p1, t):
        u = unit(p1 - p0)
        n = perp(u)
        mid = (p0 + p1) / 2
        res = {}
        for sgn in (1, -1):
            q = Point(mid + sgn * n * (t / 2 + 40))
            res[sgn] = next((r["id"] for r in rooms if r["poly"].contains(q)), "outside")
        return res

    pair_slots = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            best = None
            for a1, b1 in ring_edges(rooms[i]["poly"]):
                s1 = axis_of(a1, b1)
                if not s1:
                    continue
                for a2, b2 in ring_edges(rooms[j]["poly"]):
                    s2 = axis_of(a2, b2)
                    if not s2 or s2[:2] != s1[:2]:
                        continue
                    lo, hi = max(s1[2], s2[2]), min(s1[3], s2[3])
                    if hi - lo > 1 and (best is None or hi - lo > best[3] - best[2]):
                        best = (s1[0], s1[1], lo, hi)
            if best:
                w = next(w for w in walls if w["kind"] == "interior" and w["axis"] == best[:2]
                         and w["iv"][0] <= best[2] + 1 and w["iv"][1] >= best[3] - 1)
                p0, p1 = axis_pts(*best)
                ms = [P.t_ext / 2 + 100 if P.outline.exterior.distance(Point(q)) < 1 else P.t_int / 2 + 100 for q in (p0, p1)]
                pair_slots.append(dict(i=i, j=j, p0=p0, p1=p1, wall=w, t=w["t"], side=side_rooms(p0, p1, w["t"]),
                                       m0=ms[0], m1=ms[1]))
    ext_slots = {r["id"]: [] for r in rooms}
    for r in rooms:
        for a, b in ring_edges(r["poly"]):
            s = axis_of(a, b)
            pieces = []
            if s:
                for lo, hi in out_iv.get((s[0], s[1]), []):
                    l2, h2 = max(lo, s[2]), min(hi, s[3])
                    if h2 - l2 > 1:
                        pieces.append(axis_pts(s[0], s[1], l2, h2))
            else:
                pieces.append((a, b))
            for p0, p1 in pieces:
                w = next(w for w in walls if w["kind"] == "exterior" and
                         _seg_dist(p0, w["p0"], w["p1"]) < 1 and _seg_dist(p1, w["p0"], w["p1"]) < 1)
                ms = [P.t_ext / 2 + 100 if any(np.allclose(q, v) for v in overts) else P.t_int / 2 + 100 for q in (p0, p1)]
                ext_slots[r["id"]].append(dict(p0=p0, p1=p1, wall=w, t=w["t"], side=side_rooms(p0, p1, w["t"]),
                                               diagonal=s is None, m0=ms[0], m1=ms[1]))
    P.pair_slots, P.ext_slots = pair_slots, ext_slots
    P.hard, P.soft, P.gaps, P.cuts, P.furn = [], [], [], [], []
    P.openings = []

    # --- interior doors: Kruskal over adjacency, accept an edge only if a door fits
    parent = list(range(len(rooms)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def weight(sl):
        ti, tj = rooms[sl["i"]]["type"], rooms[sl["j"]]["type"]
        w = rng.random()
        if "corridor" in (ti, tj):
            w -= 1.0
        if "living" in (ti, tj):
            w -= 0.5
        if ti in ("bath", "storage") and tj in ("bath", "storage"):
            w += 3
        if {"bath", "kitchen"} == {ti, tj}:
            w += 1
        return w

    edges = sorted(pair_slots, key=weight)
    leftovers = []
    for sl in edges:
        if find(sl["i"]) == find(sl["j"]):
            leftovers.append(sl)
            continue
        if place_edge_door(P, sl, rng):
            parent[find(sl["i"])] = find(sl["j"])
    if len({find(i) for i in range(len(rooms))}) > 1:
        return None
    for sl in leftovers:
        ti, tj = rooms[sl["i"]]["type"], rooms[sl["j"]]["type"]
        if ti in ("bath", "storage") or tj in ("bath", "storage"):
            continue
        if rng.random() < 0.2:
            place_edge_door(P, sl, rng)

    # --- entrance + optional patio door
    def ext_door(room_types, types, required):
        cands = [r for t in room_types for r in rooms if r["type"] == t]
        for r in cands:
            slots = [s for s in ext_slots[r["id"]] if not s["diagonal"]]
            rng.shuffle(slots)
            for s in slots:
                for typ in types:
                    w = door_width(typ, rng, entrance=True)
                    into = r["id"] if rng.random() < 0.65 else "outside"
                    op = place_opening(P, s, typ, w, rng, pref_into=into)
                    if op:
                        op["role"] = "entrance" if required else "patio"
                        return op
        return None

    ent_types = ["single", "double"] if rng.random() < 0.25 else ["single"]
    if not ext_door(["corridor", "living", "kitchen", "other", "bedroom"], ent_types, True):
        return None
    if cfg["n_rooms"] >= 5 and rng.random() < 0.6:
        ext_door(["living", "bedroom"], rng.choice([["sliding"], ["double"], ["single"]]), False)

    # --- windows
    for r in rooms:
        if r["type"] in ("corridor", "storage"):
            continue
        slots = sorted(ext_slots[r["id"]], key=lambda s: -np.linalg.norm(s["p1"] - s["p0"]))
        nwin = 0
        for s in slots:
            L = np.linalg.norm(s["p1"] - s["p0"])
            if L < 1500 or nwin >= 3:
                continue
            k = 2 if (L > 5000 and rng.random() < 0.6) else 1
            for _ in range(k):
                if r["type"] == "bath":
                    w = snap(rng.uniform(600, 900))
                else:
                    w = snap(rng.uniform(900, min(2400, 0.6 * L / k)))
                if place_opening(P, s, "window", w, rng):
                    nwin += 1
    # --- stairs
    P.stairs = []
    if "stairs" in cfg["extras"]:
        place_stairs(P, rng)
    return P


def _seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / np.dot(ab, ab))))
    return float(np.linalg.norm(p - (a + ab * t)))


def door_width(typ, rng, entrance=False, small=False):
    if typ == "single":
        return snap(rng.uniform(900, 1000) if entrance else (rng.uniform(700, 800) if small else rng.uniform(750, 950)))
    if typ == "double":
        return snap(rng.uniform(1300, 1600))
    if typ == "sliding":
        return snap(rng.uniform(1300, 1800))
    if typ == "pocket":
        return snap(rng.uniform(700, 900))
    if typ == "opening":
        return snap(rng.uniform(800, 1400))
    if typ == "window":
        return snap(rng.uniform(900, 1800))
    raise ValueError(typ)


def place_edge_door(P, sl, rng):
    ri, rj = P.rooms[sl["i"]], P.rooms[sl["j"]]
    ts = {ri["type"], rj["type"]}
    x = rng.random()
    if ts & {"bath", "storage"}:
        order = ["pocket", "single"] if x < 0.18 else (["sliding", "single"] if x < 0.25 else ["single"])
    elif ts == {"living", "kitchen"}:
        order = (["opening"] if x < 0.4 else ["double"] if x < 0.6 else ["sliding"] if x < 0.8 else []) + ["single"]
    elif "living" in ts and "corridor" in ts:
        order = (["double"] if x < 0.25 else ["opening"] if x < 0.45 else ["sliding"] if x < 0.55 else []) + ["single"]
    else:
        order = (["pocket"] if x < 0.1 else ["sliding"] if x < 0.18 else ["double"] if x < 0.24 else []) + ["single"]
    order.append("opening")
    # preferred swing room: the more private one, 75 %
    a, b = (ri, rj) if PRIV[ri["type"]] >= PRIV[rj["type"]] else (rj, ri)
    pref = a["id"] if rng.random() < 0.75 else b["id"]
    small = bool(ts & {"bath", "storage"})
    for typ in order:
        for shrink in (0, 1):
            w = door_width(typ, rng, small=small or shrink == 1)
            if shrink:
                w = {"single": 700.0, "opening": 700.0}.get(typ, w)
            op = place_opening(P, sl, typ, w, rng, pref_into=pref)
            if op:
                return op
    return None


def place_opening(P, sl, typ, w, rng, pref_into=None):
    p0, p1, t = sl["p0"], sl["p1"], sl["t"]
    L = float(np.linalg.norm(p1 - p0))
    u = unit(p1 - p0)
    n = perp(u)
    side = sl["side"]
    m0, m1 = sl["m0"], sl["m1"]
    if typ == "window":
        m0, m1 = m0 + 250, m1 + 250
    pockets = [None]
    if typ == "pocket":
        pockets = ["a", "b"]
        rng.shuffle(pockets)
    for pk in pockets:
        ea = w + 80 if pk == "a" else 0.0
        eb = w + 80 if pk == "b" else 0.0
        lo, hi = m0 + ea, L - m1 - w - eb
        if hi < lo:
            continue
        rnd = [rng.uniform(lo, hi) for _ in range(4)]
        ends = [lo, hi]
        rng.shuffle(ends)
        cands = (ends + [(lo + hi) / 2] + rnd) if rng.random() < 0.65 else (rnd[:2] + ends + [(lo + hi) / 2] + rnd[2:])
        for s in cands:
            s = round(s / 10) * 10
            s = min(max(s, lo), hi)
            a, b = p0 + u * s, p0 + u * (s + w)
            gap_chk = rect_poly(a, b, t / 2, ext=120)
            cut = rect_poly(a, b, t / 2 + 4)
            pocket_geom = None
            if pk == "a":
                pocket_geom = rect_poly(a - u * (w + 40), a, t / 6)
            elif pk == "b":
                pocket_geom = rect_poly(b, b + u * (w + 40), t / 6)
            chk = gap_chk if pocket_geom is None else unary_union([gap_chk, pocket_geom.buffer(60)])
            if any(chk.intersects(g) for g in P.gaps):
                continue
            if typ == "window":
                op = dict(category="window", type="unknown", a=a, b=b, t=t, u=u, n=n, w=w, leaves=[],
                          slide=None, connects=[x for x in (side[1], side[-1])], swing_into=None,
                          handedness=None, wall=sl["wall"])
                commit(P, op, gap_chk, [cut], [], [])
                return op
            if typ in ("single", "double"):
                sides = [1, -1]
                if pref_into is not None and side[-1] == pref_into:
                    sides = [-1, 1]
                hinges = ["a", "b"]
                rng.shuffle(hinges)
                opts = [(sd, h) for sd in sides for h in (hinges if typ == "single" else ["ab"])]
                for sd, h in opts:
                    into = side[sd]
                    fa, fb = a + sd * n * t / 2, b + sd * n * t / 2
                    if h == "a":
                        leaves = [(fa, fb, fa + sd * n * w)]
                    elif h == "b":
                        leaves = [(fb, fa, fb + sd * n * w)]
                    else:
                        mid = (fa + fb) / 2
                        leaves = [(fa, mid, fa + sd * n * w / 2), (fb, mid, fb + sd * n * w / 2)]
                    sweep = unary_union([Polygon([hg] + arc_points(hg, ct, ot)) for hg, ct, ot in leaves])
                    if into == "outside":
                        if sweep.buffer(-3).intersects(P.outer):
                            continue
                    elif not P.room_by_id[into]["interior"].buffer(3).contains(sweep):
                        continue
                    if any(sweep.buffer(-3).intersects(g) for g in P.hard + P.soft):
                        continue
                    clear = quad(a, b, -sd * n, t / 2, t / 2 + 600)
                    if any(clear.buffer(-3).intersects(g) for g in P.hard):
                        continue
                    other = side[-sd]
                    hand = None
                    if typ == "single":
                        f = sd * n
                        right = V(-f[1], f[0])
                        hand = "RH" if float(np.dot(leaves[0][0] - (a + b) / 2, right)) > 0 else "LH"
                    op = dict(category="door", type=typ, a=a, b=b, t=t, u=u, n=n, w=w,
                              leaves=[dict(hinge=hg, closed_tip=ct, open_tip=ot) for hg, ct, ot in leaves],
                              slide=None, connects=[other, into], swing_into=into, handedness=hand,
                              wall=sl["wall"], swing_sign=sd)
                    commit(P, op, gap_chk, [cut], [sweep.buffer(70)], [clear])
                    return op
                continue
            # sliding / pocket / plain opening: keep-clear zones on both sides
            clears = [quad(a, b, n, t / 2, t / 2 + 500), quad(a, b, -n, t / 2, t / 2 + 500)]
            if any(c.buffer(-3).intersects(g) for c in clears for g in P.hard):
                continue
            cat = "opening" if typ == "opening" else "door"
            op = dict(category=cat, type="none" if typ == "opening" else typ, a=a, b=b, t=t, u=u, n=n, w=w,
                      leaves=[], slide=None, connects=[side[1], side[-1]], swing_into=None, handedness=None,
                      wall=sl["wall"])
            cuts = [cut]
            if typ == "sliding":
                op["slide"] = dict(track_side=None, direction=None, panels=2, kind="bypass")
            if typ == "pocket":
                pdir = -u if pk == "a" else u
                op["slide"] = dict(track_side=pk, direction=pdir, panels=1, kind="pocket")
                op["pocket_geom"] = pocket_geom
                cuts.append(pocket_geom)
            commit(P, op, chk, cuts, [], clears)
            return op
    return None


def commit(P, op, gap_chk, cuts, hard, soft):
    P.gaps.append(gap_chk)
    P.cuts += cuts
    P.hard += hard
    P.soft += soft
    op["sweep"] = hard[0] if hard else None
    P.openings.append(op)


def place_stairs(P, rng):
    cands = [r for t in ("living", "other", "corridor", "bedroom", "kitchen") for r in P.rooms if r["type"] == t]
    for r in cands:
        nt = rng.randint(9, 13)
        tread = rng.choice([250.0, 260.0, 280.0])
        Ls, Ws = nt * tread, snap(rng.uniform(950, 1200))
        for _ in range(250):
            fr = wall_frame(r, Ls, Ws, rng)
            if fr is None:
                continue
            O, ex, ey = fr
            poly = Polygon([O, O + ex * Ls, O + ex * Ls + ey * Ws, O + ey * Ws])
            if not r["interior"].buffer(-5).contains(poly):
                continue
            if any(poly.intersects(g) for g in P.hard + P.soft + P.furn):
                continue
            up = 1 if rng.random() < 0.5 else -1
            P.stairs.append(dict(id=f"s{len(P.stairs) + 1}", O=O, ex=ex, ey=ey, L=Ls, W=Ws, nt=nt, up=up,
                                 poly=poly, room=r["id"]))
            P.hard.append(poly)
            return


def wall_frame(r, Wd, D, rng, against=True):
    """Random placement frame for an item W (along wall) x D (away from wall) inside room r."""
    x0, y0, x1, y1 = r["interior"].bounds
    side = rng.choice(["top", "bottom", "left", "right"])
    gap = 15.0
    if side in ("top", "bottom"):
        if x1 - x0 < Wd + 2 * gap:
            return None
        s = rng.uniform(x0 + gap, x1 - gap - Wd)
        if side == "top":
            return V(s, y0 + gap), V(1, 0), V(0, 1)
        return V(s + Wd, y1 - gap), V(-1, 0), V(0, -1)
    if y1 - y0 < Wd + 2 * gap:
        return None
    s = rng.uniform(y0 + gap, y1 - gap - Wd)
    if side == "left":
        return V(x0 + gap, s + Wd), V(0, -1), V(1, 0)
    return V(x1 - gap, s), V(0, 1), V(-1, 0)


# ----------------------------------------------------------------------------------------------
# furniture library (local frame: x along wall 0..W, y away from wall 0..D)
# ----------------------------------------------------------------------------------------------
def R(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def RR(x0, y0, x1, y1, r, k=5):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(k + 1):
            a = math.radians(a0 + 90 * i / k)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def ELL(cx, cy, rx, ry, k=28):
    return [(cx + rx * math.cos(2 * math.pi * i / k), cy + ry * math.sin(2 * math.pi * i / k)) for i in range(k)]


def item(kind, rng):
    if kind == "bed":
        dbl = rng.random() < 0.6
        bw, bd = (1600.0 if dbl else 900.0), 2050.0
        ns = 450.0 if rng.random() < 0.7 else 0.0
        x = ns + 40 if ns else 0.0
        W = bw + 2 * x
        P = [("poly", R(x, 0, x + bw, bd))]
        if dbl:
            P += [("poly", RR(x + 100, 80, x + bw / 2 - 50, 450, 60)), ("poly", RR(x + bw / 2 + 50, 80, x + bw - 100, 450, 60))]
        else:
            P += [("poly", RR(x + 120, 80, x + bw - 120, 450, 60))]
        P += [("line", (x, 650), (x + bw, 650)), ("line", (x + bw - 500, bd), (x + bw, bd - 500))]
        if ns:
            P += [("poly", R(0, 0, ns, 420)), ("circle", (ns / 2, 210), 110),
                  ("poly", R(W - ns, 0, W, 420)), ("circle", (W - ns / 2, 210), 110)]
        return W, bd, P
    if kind == "wardrobe":
        W, D = snap(rng.uniform(1000, 2000)), 600.0
        P = [("poly", R(0, 0, W, D)), ("line", (0, D / 2), (W, D / 2))]
        for k in range(1, int(W // 250)):
            P.append(("line", (k * 250, D / 2 - 120), (k * 250 + 60, D / 2 + 120)))
        return W, D, P
    if kind == "desk":
        W, D = 1200.0, 600.0
        return W, D + 450, [("poly", R(0, 0, W, D)), ("circle", (W / 2, D + 230), 220)]
    if kind == "sofa":
        W, D = snap(rng.uniform(1800, 2400)), 900.0
        P = [("poly", R(0, 0, W, D)), ("poly", R(0, 0, W, 220)), ("poly", R(0, 0, 220, D)), ("poly", R(W - 220, 0, W, D))]
        k = 3 if W > 2000 else 2
        for i in range(1, k):
            xx = 220 + (W - 440) * i / k
            P.append(("line", (xx, 220), (xx, D)))
        return W, D, P
    if kind == "armchair":
        W = D = 850.0
        return W, D, [("poly", R(0, 0, W, D)), ("poly", R(0, 0, W, 200)), ("poly", R(0, 0, 180, D)), ("poly", R(W - 180, 0, W, D))]
    if kind == "tv":
        W, D = snap(rng.uniform(1400, 2000)), 420.0
        return W, D, [("poly", R(0, 0, W, D)), ("poly", R(W * 0.2, 80, W * 0.8, 160))]
    if kind == "counter":
        W, D = snap(rng.uniform(1800, 3200)), 620.0
        P = [("poly", R(0, 0, W, D))]
        sx = W * rng.uniform(0.55, 0.7)
        P += [("poly", RR(sx, 90, sx + 650, 520, 60)), ("poly", RR(sx + 50, 130, sx + 600, 480, 50))]
        cx = 200
        for (dx, dy) in ((120, 170), (420, 170), (120, 450), (420, 450)):
            P.append(("circle", (cx + dx, dy), 95))
        P.append(("poly", R(cx, 30, cx + 560, 590)))
        return W, D, P
    if kind == "fridge":
        return 680.0, 680.0, [("poly", R(0, 0, 680, 680)), ("line", (0, 0), (680, 680)), ("line", (680, 0), (0, 680))]
    if kind == "dining":
        if rng.random() < 0.5:
            W, D = 1600.0, 1500.0
            P = [("poly", R(250, 300, 1350, 1200))]
            for x in (450, 1050):
                P += [("poly", R(x - 210, 0, x + 210, 280)), ("poly", R(x - 210, 1220, x + 210, 1500))]
            return W, D, P
        W = D = 1500.0
        P = [("circle", (750, 750), 480)]
        for a in (0, 90, 180, 270):
            c = (750 + 620 * math.cos(math.radians(a)), 750 + 620 * math.sin(math.radians(a)))
            P.append(("circle", c, 200))
        return W, D, P
    if kind == "coffee":
        if rng.random() < 0.5:
            return 1100.0, 600.0, [("poly", R(0, 0, 1100, 600))]
        return 800.0, 800.0, [("circle", (400, 400), 400)]
    if kind == "toilet":
        W, D = 400.0, 720.0
        return W, D, [("poly", R(0, 0, W, 190)), ("poly", ELL(W / 2, 190 + 260, 180, 260))]
    if kind == "basin":
        W, D = 560.0, 460.0
        return W, D, [("poly", R(0, 0, W, D)), ("poly", ELL(W / 2, D / 2 + 20, 200, 150)), ("circle", (W / 2, 70), 25)]
    if kind == "tub":
        W, D = 1700.0, 750.0
        return W, D, [("poly", R(0, 0, W, D)), ("poly", RR(80, 80, W - 80, D - 80, 200)), ("circle", (W - 250, D / 2), 35)]
    if kind == "shower":
        W = D = 900.0
        return W, D, [("poly", R(0, 0, W, D)), ("line", (0, 0), (W, D)), ("line", (W, 0), (0, D)), ("circle", (W / 2, D / 2), 45)]
    if kind == "shelves":
        W, D = snap(rng.uniform(900, 1600)), 400.0
        P = [("poly", R(0, 0, W, D))]
        for k in range(1, 4):
            P.append(("line", (0, D * k / 4), (W, D * k / 4)))
        return W, D, P
    raise ValueError(kind)


FURN_BY_TYPE = {
    "bedroom": [("bed", "wall"), ("wardrobe", "wall"), ("desk", "wall")],
    "living": [("sofa", "wall"), ("tv", "wall"), ("coffee", "free"), ("armchair", "wall"), ("dining", "free")],
    "kitchen": [("counter", "wall"), ("fridge", "wall"), ("dining", "free")],
    "bath": [("toilet", "wall"), ("basin", "wall"), ("tub", "wall"), ("shower", "wall")],
    "storage": [("shelves", "wall")],
    "other": [("desk", "wall"), ("shelves", "wall"), ("armchair", "wall")],
    "corridor": [],
}


def place_furniture(P, rng):
    items = []
    for r in P.rooms:
        lst = list(FURN_BY_TYPE.get(r["type"], []))
        if r["type"] == "bath" and r.get("label") == "WC":
            lst = [("toilet", "wall"), ("basin", "wall")]
        elif r["type"] == "bath":
            drop = "shower" if rng.random() < 0.6 else "tub"
            lst = [x for x in lst if x[0] != drop]
        if r["type"] == "living" and any(rr["type"] == "kitchen" for rr in P.rooms):
            lst = [x for x in lst if x[0] != "dining"] if rng.random() < 0.5 else lst
        if r["type"] == "living" and r.get("label") == "LIVING / KITCHEN":
            lst = [("counter", "wall"), ("sofa", "wall"), ("coffee", "free"), ("tv", "wall")]
        for kind, mode in lst:
            Wd, D, prims = item(kind, rng)
            for _ in range(50):
                if mode == "wall":
                    fr = wall_frame(r, Wd, D, rng)
                    if fr is None:
                        continue
                    O, ex, ey = fr
                else:
                    x0, y0, x1, y1 = r["interior"].bounds
                    k = rng.randrange(4)
                    ex = [V(1, 0), V(0, 1), V(-1, 0), V(0, -1)][k]
                    ey = perp(ex)
                    c = V(rng.uniform(x0, x1), rng.uniform(y0, y1))
                    O = c - ex * Wd / 2 - ey * D / 2
                foot = Polygon([O, O + ex * Wd, O + ex * Wd + ey * D, O + ey * D])
                if not r["interior"].buffer(-10).contains(foot):
                    continue
                if any(foot.intersects(g) for g in P.hard + P.soft + P.furn):
                    continue
                P.furn.append(foot.buffer(60))
                items.append(dict(kind=kind, O=O, ex=ex, ey=ey, prims=prims, room=r["id"]))
                break
    return items


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
SH = 4
F = 16


class Canvas:
    def __init__(self, P, cfg, rng):
        b = P.outer.bounds
        dims = cfg["dims"]
        mx = 1300.0 + (2100.0 if dims else 0.0) + (1100.0 if "axisgrid" in cfg["extras"] else 0.0)
        mb = mx + (1300.0 if ("title" in cfg["extras"] or "scalebar" in cfg["extras"]) else 0.0)
        self.ox, self.oy = b[0] - mx, b[1] - mx
        Wmm, Hmm = (b[2] - b[0]) + 2 * mx, (b[3] - b[1]) + mx + mb
        self.mmpp = round(max(Wmm, Hmm) / cfg["long_px"], 3)
        self.W = int(math.ceil(Wmm / self.mmpp))
        self.H = int(math.ceil(Hmm / self.mmpp))
        self.margin_mm = mx

    def X(self, p):
        return ((p[0] - self.ox) / self.mmpp, (p[1] - self.oy) / self.mmpp)

    def Xi(self, q):
        return V(q[0] * self.mmpp + self.ox, q[1] * self.mmpp + self.oy)


def ipt(q):
    return (int(round(q[0] * F)), int(round(q[1] * F)))


def arr(qs):
    return np.array([ipt(q) for q in qs], np.int32)


def draw_line(img, C, a, b, color, th):
    cv2.line(img, ipt(C.X(a)), ipt(C.X(b)), color, max(1, int(th)), cv2.LINE_AA, SH)


def draw_poly(img, C, pts, color, th, closed=True, fill=None):
    q = arr([C.X(p) for p in pts])
    if fill is not None:
        cv2.fillPoly(img, [q], fill, cv2.LINE_AA, SH)
    cv2.polylines(img, [q], closed, color, max(1, int(th)), cv2.LINE_AA, SH)


def draw_dashed(img, C, pts, color, th, dash_mm, gap_mm):
    seq = [V(p) for p in pts]
    acc, on = 0.0, True
    cur = [seq[0]]
    for a, b in zip(seq[:-1], seq[1:]):
        seg = np.linalg.norm(b - a)
        pos = 0.0
        while pos < seg:
            lim = dash_mm if on else gap_mm
            step = min(seg - pos, lim - acc)
            pos += step
            acc += step
            pt = a + (b - a) * (pos / seg)
            if on:
                cur.append(pt)
            if acc >= lim - 1e-9:
                if on and len(cur) > 1:
                    draw_poly(img, C, cur, color, th, closed=False)
                on = not on
                acc = 0.0
                cur = [pt]
    if on and len(cur) > 1:
        draw_poly(img, C, cur, color, th, closed=False)


def fill_geom(img, C, geom, color):
    rings = []
    for p in poly_parts(geom):
        rings.append(arr([C.X(c) for c in p.exterior.coords]))
        rings += [arr([C.X(c) for c in h.coords]) for h in p.interiors]
    if rings:
        cv2.fillPoly(img, rings, color, cv2.LINE_AA, SH)


def outline_geom(img, C, geom, color, th):
    for p in poly_parts(geom):
        cv2.polylines(img, [arr([C.X(c) for c in p.exterior.coords])], True, color, th, cv2.LINE_AA, SH)
        for h in p.interiors:
            cv2.polylines(img, [arr([C.X(c) for c in h.coords])], True, color, th, cv2.LINE_AA, SH)


def mask_geom(shape, C, geom):
    m = np.zeros(shape, np.uint8)
    rings = []
    for p in poly_parts(geom):
        rings.append(arr([C.X(c) for c in p.exterior.coords]))
        rings += [arr([C.X(c) for c in h.coords]) for h in p.interiors]
    if rings:
        cv2.fillPoly(m, rings, 255, cv2.LINE_AA, SH)
    return m


def blend(img, alpha_u8, color):
    a = (alpha_u8.astype(np.float32) / 255.0)[..., None]
    img[:] = (img.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


class TextStyle:
    def __init__(self, rng):
        self.font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                                cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN])

    def fit(self, text, h_px):
        (w1, h1), _ = cv2.getTextSize(text, self.font, 1.0, 1)
        sc = max(h_px, 8.0) / h1
        th = max(1, int(round(sc * (1.6 if self.font == cv2.FONT_HERSHEY_PLAIN else 1.1))))
        (w, h), base = cv2.getTextSize(text, self.font, sc, th)
        return sc, th, w, h, base


def put_text(img, ts, text, org, sc, th, color):
    cv2.putText(img, text, (int(round(org[0])), int(round(org[1]))), ts.font, sc, color, th, cv2.LINE_AA)


def put_text_rot90(img, ts, text, center, sc, th, color):
    (w, h), base = cv2.getTextSize(text, ts.font, sc, th)
    can = np.zeros((h + base + 6, w + 6), np.uint8)
    cv2.putText(can, text, (3, h + 3), ts.font, sc, 255, th, cv2.LINE_AA)
    rot = cv2.rotate(can, cv2.ROTATE_90_COUNTERCLOCKWISE)
    hh, ww = rot.shape
    x0, y0 = int(round(center[0] - ww / 2)), int(round(center[1] - hh / 2))
    sub = img[y0:y0 + hh, x0:x0 + ww]
    blend(sub, rot[:sub.shape[0], :sub.shape[1]], color)
    return [x0 + 3, y0 + 3, x0 + ww - 3, y0 + hh - 3]


def render(P, cfg, rng):
    C = Canvas(P, cfg, rng)
    mm = C.mmpp
    img = np.full((C.H, C.W, 3), 255, np.uint8)
    ex = cfg["extras"]
    ink = (20, 20, 20)
    lw_wall = max(1, int(round(max(18.0 / mm, 1.0))))
    lw_thin = max(1, int(round(10.0 / mm)))
    lw_furn = max(1, int(round(8.0 / mm)))
    furn_col = (90, 90, 90) if rng.random() < 0.6 else (40, 40, 40)
    labels = []
    extras_gt = {}

    # --- page background
    if "graphpaper" in ex:
        step = max(6, int(round(200.0 / mm)))
        for k in range(0, max(C.W, C.H), step):
            col = (215, 205, 190) if (k // step) % 5 == 0 else (238, 232, 222)
            cv2.line(img, (k, 0), (k, C.H), col, 1)
            cv2.line(img, (0, k), (C.W, k), col, 1)

    # --- room fills / tiles
    for r in P.rooms:
        if "color" in ex:
            fill_geom(img, C, r["interior"], ROOM_COLORS.get(r["type"], (230, 230, 230)))
        elif "graphpaper" in ex:
            fill_geom(img, C, r["interior"], (255, 255, 255))
        if "tiles" in ex and r["type"] in ("bath", "kitchen"):
            m = mask_geom(img.shape[:2], C, r["interior"])
            layer = np.zeros(img.shape[:2], np.uint8)
            step = 300.0 if r["type"] == "bath" else 400.0
            x0, y0, x1, y1 = r["interior"].bounds
            x = x0
            while x < x1:
                cv2.line(layer, ipt(C.X((x, y0))), ipt(C.X((x, y1))), 255, 1, cv2.LINE_AA, SH)
                x += step
            y = y0
            while y < y1:
                cv2.line(layer, ipt(C.X((x0, y))), ipt(C.X((x1, y))), 255, 1, cv2.LINE_AA, SH)
                y += step
            blend(img, (np.minimum(m, layer) * 0.55).astype(np.uint8), (120, 120, 120))

    # --- axis grid (structural grid, dash-dot, crossing the plan)
    if "axisgrid" in ex:
        b = P.outer.bounds
        xs = sorted({round(float(w["p0"][0]), 1) for w in P.walls if abs(w["p0"][0] - w["p1"][0]) < 1e-6})
        ys = sorted({round(float(w["p0"][1]), 1) for w in P.walls if abs(w["p0"][1] - w["p1"][1]) < 1e-6})
        xs = xs[::max(1, len(xs) // 5)]
        ys = ys[::max(1, len(ys) // 4)]
        ts = TextStyle(rng)
        top = b[1] - C.margin_mm + 600
        left = b[0] - C.margin_mm + 600
        for k, x in enumerate(xs):
            draw_dashed(img, C, [(x, top + 300), (x, b[3] + 600)], (150, 150, 150), lw_thin, 600, 150)
            cv2.circle(img, ipt(C.X((x, top))), int(300 / mm * F), (60, 60, 60), lw_thin, cv2.LINE_AA, SH)
            t = chr(65 + k)
            sc, th, w, h, base = ts.fit(t, 260 / mm)
            q = C.X((x, top))
            org = (q[0] - w / 2, q[1] + h / 2)
            put_text(img, ts, t, org, sc, th, (40, 40, 40))
            labels.append(dict(text=t, bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))
        for k, y in enumerate(ys):
            draw_dashed(img, C, [(left + 300, y), (b[2] + 600, y)], (150, 150, 150), lw_thin, 600, 150)
            cv2.circle(img, ipt(C.X((left, y))), int(300 / mm * F), (60, 60, 60), lw_thin, cv2.LINE_AA, SH)
            t = str(k + 1)
            sc, th, w, h, base = ts.fit(t, 260 / mm)
            q = C.X((left, y))
            org = (q[0] - w / 2, q[1] + h / 2)
            put_text(img, ts, t, org, sc, th, (40, 40, 40))
            labels.append(dict(text=t, bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))

    # --- walls
    cut_geom = unary_union(P.cuts)
    walls_g = P.wall_union.difference(cut_geom)
    style = cfg["wall_style"]
    if style == "solid":
        fill_geom(img, C, walls_g, (15, 15, 15))
    elif style == "gray":
        fill_geom(img, C, walls_g, (125, 125, 125))
        outline_geom(img, C, walls_g, ink, lw_thin)
    elif style == "hollow":
        fill_geom(img, C, walls_g, (255, 255, 255))
        outline_geom(img, C, walls_g, ink, lw_wall)
    elif style == "hatched":
        fill_geom(img, C, walls_g, (255, 255, 255))
        m = mask_geom(img.shape[:2], C, walls_g)
        layer = np.zeros(img.shape[:2], np.uint8)
        step = max(4, int(round(70.0 / mm)))
        for k in range(-C.H, C.W, step):
            cv2.line(layer, (k, C.H), (k + C.H, 0), 255, 1, cv2.LINE_AA)
        blend(img, np.minimum(m, layer), (30, 30, 30))
        outline_geom(img, C, walls_g, ink, lw_wall)

    # --- windows
    for op in P.openings:
        if op["category"] != "window":
            continue
        a, b, n, t = op["a"], op["b"], op["n"], op["t"]
        for d in (t / 2, -t / 2):
            draw_line(img, C, a + n * d, b + n * d, ink, lw_thin)
        draw_line(img, C, a, b, ink, lw_thin)
        draw_line(img, C, a + n * t / 2, a - n * t / 2, ink, lw_thin)
        draw_line(img, C, b + n * t / 2, b - n * t / 2, ink, lw_thin)

    # --- doors
    leaf_mode = rng.choice(["line", "rect"])
    arc_dashed = rng.random() < 0.25
    lw_leaf = max(1, int(round(max(30.0 / mm, 1.5))))
    for op in P.openings:
        if op["category"] != "door":
            continue
        a, b, n, u, t = op["a"], op["b"], op["n"], op["u"], op["t"]
        for lf in op["leaves"]:
            hg, ct, ot = lf["hinge"], lf["closed_tip"], lf["open_tip"]
            pts = arc_points(hg, ct, ot, 32)
            if arc_dashed:
                draw_dashed(img, C, pts, ink, lw_thin, 120, 80)
            else:
                draw_poly(img, C, pts, ink, lw_thin, closed=False)
            if leaf_mode == "line":
                draw_line(img, C, hg, ot, ink, lw_leaf)
            else:
                d = unit(ot - hg)
                q = perp(d) * 20.0
                side = 1 if np.dot(q, ct - hg) < 0 else -1  # leaf body on the side away from the opening
                draw_poly(img, C, [hg, ot, ot + side * 2 * q, hg + side * 2 * q], ink, lw_thin, fill=(255, 255, 255))
        if op["type"] == "sliding":
            mid = (a + b) / 2
            ov = 60.0
            pt = max(t / 5, 35.0)
            for s0, s1, off in ((a, mid + u * ov, t / 6), (mid - u * ov, b, -t / 6)):
                draw_poly(img, C, [s0 + n * (off - pt / 2), s1 + n * (off - pt / 2), s1 + n * (off + pt / 2), s0 + n * (off + pt / 2)],
                          ink, lw_thin, fill=(255, 255, 255))
        if op["type"] == "pocket":
            pdir = op["slide"]["direction"]
            pk = op["slide"]["track_side"]
            P_, F_ = (a, b) if pk == "a" else (b, a)
            pt = max(t / 5, 35.0)
            s0, s1 = P_ + pdir * 0.45 * op["w"], F_ + pdir * 0.45 * op["w"]
            draw_poly(img, C, [s0 + n * pt / 2, s1 + n * pt / 2, s1 - n * pt / 2, s0 - n * pt / 2], ink, lw_thin, fill=(170, 170, 170))

    # --- stairs
    for s in P.stairs:
        O, exv, eyv, L, Wd = s["O"], s["ex"], s["ey"], s["L"], s["W"]
        draw_poly(img, C, [O, O + exv * L, O + exv * L + eyv * Wd, O + eyv * Wd], ink, lw_thin, fill=(255, 255, 255))
        tr = L / s["nt"]
        for k in range(1, s["nt"]):
            draw_line(img, C, O + exv * tr * k, O + exv * tr * k + eyv * Wd, ink, lw_thin)
        c0, c1 = O + exv * tr * 0.5 + eyv * Wd / 2, O + exv * (L - tr * 0.5) + eyv * Wd / 2
        if s["up"] < 0:
            c0, c1 = c1, c0
        draw_line(img, C, c0, c1, ink, lw_thin)
        d = unit(c1 - c0)
        for sg in (1, -1):
            draw_line(img, C, c1, c1 - d * 250 + perp(d) * 120 * sg, ink, lw_thin)
        cv2.circle(img, ipt(C.X(c0)), max(2, int(40 / mm * F)), ink, -1, cv2.LINE_AA, SH)
        s["dir"] = d
        if cfg["text"]:
            ts = TextStyle(rng)
            sc, th, w, h, base = ts.fit("UP", 180 / mm)
            q = C.X(c0 + d * 260)
            org = (q[0] + 2, q[1] - 6)
            put_text(img, ts, "UP", org, sc, th, ink)
            labels.append(dict(text="UP", bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))

    # --- room labels (placed before furniture; furniture avoids them)
    if cfg["text"]:
        ts = TextStyle(rng)
        name_mm = rng.uniform(220, 320)
        for r in P.rooms:
            name = r["label"]
            lines = [(name, name_mm)]
            if "area" in ex:
                lines.append((f"{r['interior'].area / 1e6:.1f} m2", name_mm * 0.7))
            fits = [ts.fit(t, hmm / mm) for t, hmm in lines]
            wpx = max(f[2] for f in fits)
            hpx = sum(f[3] + f[4] for f in fits) + 4 * (len(fits) - 1)
            wmm, hmm_ = wpx * mm, hpx * mm
            c0 = V(r["interior"].representative_point().coords[0])
            ctr = V(r["interior"].centroid.coords[0])
            if r["interior"].contains(Point(ctr)):
                c0 = ctr
            placed = None
            for k in range(0, 400):
                ring = int(math.sqrt(k))
                ang = k * 2.39996
                c = c0 + 150.0 * ring * V(math.cos(ang), math.sin(ang))
                bx = box(c[0] - wmm / 2 - 60, c[1] - hmm_ / 2 - 60, c[0] + wmm / 2 + 60, c[1] + hmm_ / 2 + 60)
                if not r["interior"].buffer(-30).contains(bx):
                    continue
                if any(bx.intersects(g) for g in P.hard + P.soft + P.furn):
                    continue
                placed = (c, bx)
                break
            if placed is None:
                r["label_drawn"] = False
                continue
            r["label_drawn"] = True
            c, bx = placed
            P.furn.append(bx)
            q = C.X(c)
            y = q[1] - hpx / 2
            for (t, _), (sc, th, w, h, base) in zip(lines, fits):
                org = (q[0] - w / 2, y + h)
                put_text(img, ts, t, org, sc, th, ink)
                labels.append(dict(text=t, bbox=[org[0], org[1] - h, org[0] + w, org[1] + base],
                                   kind="room_name" if t == name else "other", room=r["id"]))
                y += h + base + 4

    # --- furniture
    if cfg["furniture"]:
        for it in place_furniture(P, rng):
            O, exv, eyv = it["O"], it["ex"], it["ey"]
            T = lambda p: O + exv * p[0] + eyv * p[1]
            for pr in it["prims"]:
                if pr[0] == "poly":
                    draw_poly(img, C, [T(p) for p in pr[1]], furn_col, lw_furn, fill=(255, 255, 255))
                elif pr[0] == "line":
                    draw_line(img, C, T(pr[1]), T(pr[2]), furn_col, lw_furn)
                elif pr[0] == "circle":
                    c = ipt(C.X(T(pr[1])))
                    rr = int(round(pr[2] / mm * F))
                    cv2.circle(img, c, rr, (255, 255, 255), -1, cv2.LINE_AA, SH)
                    cv2.circle(img, c, rr, furn_col, lw_furn, cv2.LINE_AA, SH)
        P.furniture_drawn = True

    # --- dimension chains
    if cfg["dims"]:
        ts = TextStyle(rng)
        sides = rng.choice([("top", "left"), ("bottom", "right"), ("top", "right"), ("bottom", "left")])
        ob = P.outline.bounds
        off1 = P.t_ext / 2 + 1400
        th_mm = rng.uniform(170, 230)
        dim_col = (50, 50, 50)
        for side in sides:
            horiz = side in ("top", "bottom")
            if horiz:
                edge = ob[1] if side == "top" else ob[3]
                pos = {ob[0], ob[2]}
                for w in P.walls:
                    if abs(w["p0"][0] - w["p1"][0]) < 1e-6:
                        ys = (w["p0"][1], w["p1"][1])
                        if (side == "top" and min(ys) <= edge + 1) or (side == "bottom" and max(ys) >= edge - 1):
                            pos.add(float(w["p0"][0]))
                sgn = -1 if side == "top" else 1
            else:
                edge = ob[0] if side == "left" else ob[2]
                pos = {ob[1], ob[3]}
                for w in P.walls:
                    if abs(w["p0"][1] - w["p1"][1]) < 1e-6:
                        xs = (w["p0"][0], w["p1"][0])
                        if (side == "left" and min(xs) <= edge + 1) or (side == "right" and max(xs) >= edge - 1):
                            pos.add(float(w["p0"][1]))
                sgn = -1 if side == "left" else 1
            pos = sorted(round(p, 1) for p in pos)
            chains = ((0, pos), (1, [pos[0], pos[-1]])) if len(pos) > 2 else ((1, pos),)
            for chain, plist in chains:
                d = edge + sgn * (off1 + 750 * chain)
                Pt = (lambda s_, d_=d: V(s_, d_)) if horiz else (lambda s_, d_=d: V(d_, s_))
                Nv = V(0, sgn) if horiz else V(sgn, 0)
                draw_line(img, C, Pt(plist[0] - 200), Pt(plist[-1] + 200), dim_col, lw_thin)
                for s_ in plist:
                    draw_line(img, C, Pt(s_) - Nv * 150, Pt(s_) + Nv * 350 * (0 if chain else 1) - Nv * 0, dim_col, lw_thin)
                    tick = (V(1, -1) if horiz else V(1, -1)) * 80
                    draw_line(img, C, Pt(s_) - tick, Pt(s_) + tick, dim_col, lw_thin + 1)
                for s0, s1 in zip(plist[:-1], plist[1:]):
                    val = f"{int(round(s1 - s0))}"
                    sc, th, w, h, base = ts.fit(val, th_mm / mm)
                    if max(w, h) * mm > (s1 - s0) - 150:
                        continue
                    mid = Pt((s0 + s1) / 2)
                    if horiz:
                        q = C.X(mid - V(0, 70))
                        org = (q[0] - w / 2, q[1])
                        put_text(img, ts, val, org, sc, th, dim_col)
                        bb = [org[0], org[1] - h, org[0] + w, org[1] + base]
                    else:
                        q = C.X(mid - V(70 + (h + base) * mm / 2, 0))
                        bb = put_text_rot90(img, ts, val, q, sc, th, dim_col)
                    labels.append(dict(text=val, bbox=bb, kind="dimension"))

    # --- title / scale bar / north arrow
    b = P.outer.bounds
    base_y = b[3] + (2100 + 1700 if cfg["dims"] else 1500)
    if "title" in ex:
        ts = TextStyle(rng)
        t = rng.choice(["FLOOR PLAN", "APARTMENT TYPE B", "LEVEL 1 PLAN", "UNIT 3A"])
        sc, th, w, h, base = ts.fit(t, 360 / mm)
        q = C.X(((b[0] + b[2]) / 2, base_y + 360))
        org = (q[0] - w / 2, q[1])
        put_text(img, ts, t, org, sc, th, ink)
        labels.append(dict(text=t, bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))
    if "scalebar" in ex:
        ts = TextStyle(rng)
        x0 = b[0] + 200
        y0 = base_y + 250
        for k in range(5):
            pts = [V(x0 + 1000 * k, y0), V(x0 + 1000 * (k + 1), y0), V(x0 + 1000 * (k + 1), y0 + 150), V(x0 + 1000 * k, y0 + 150)]
            draw_poly(img, C, pts, ink, lw_thin, fill=(20, 20, 20) if k % 2 == 0 else (255, 255, 255))
        for txt, xx in (("0", x0), ("5 m", x0 + 5000)):
            sc, th, w, h, base = ts.fit(txt, 180 / mm)
            q = C.X((xx, y0 - 60))
            org = (q[0] - w / 2, q[1])
            put_text(img, ts, txt, org, sc, th, ink)
            labels.append(dict(text=txt, bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))
        extras_gt["scale_bar"] = dict(p0=V(x0, y0 + 75), p1=V(x0 + 5000, y0 + 75), length_mm=5000)
    if "north" in ex:
        c = V(b[2] + C.margin_mm - 700, b[1] - C.margin_mm + 700)
        if not any(Point(c).buffer(450).intersects(g) for g in P.hard):
            cv2.circle(img, ipt(C.X(c)), int(400 / mm * F), ink, lw_thin, cv2.LINE_AA, SH)
            draw_poly(img, C, [c + V(0, -380), c + V(150, 250), c + V(0, 150), c + V(-150, 250)], ink, lw_thin, fill=(20, 20, 20))
            ts = TextStyle(rng)
            sc, th, w, h, base = ts.fit("N", 220 / mm)
            q = C.X(c + V(0, -470))
            org = (q[0] - w / 2, q[1])
            put_text(img, ts, "N", org, sc, th, ink)
            labels.append(dict(text="N", bbox=[org[0], org[1] - h, org[0] + w, org[1] + base], kind="other"))
    return C, img, labels, extras_gt


# ----------------------------------------------------------------------------------------------
# degradations
# ----------------------------------------------------------------------------------------------
def degrade(img, d, seed):
    nrng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    M = np.array([[1, 0, 0], [0, 1, 0]], float)
    paper = (255, 255, 255)
    if d.get("rot"):
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), d["rot"], 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=paper)
    f = img.astype(np.float32)
    if d.get("fade"):
        f = 255 - (255 - f) * d["fade"]
    if d.get("tint"):
        tint = np.array([nrng.uniform(0.80, 0.90), nrng.uniform(0.90, 0.96), nrng.uniform(0.97, 1.0)], np.float32)
        f = f * tint
    if d.get("light"):
        g = nrng.uniform(1 - d["light"], 1.0, size=(4, 4)).astype(np.float32)
        field = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        f = f * np.clip(field, 0.4, 1.05)[..., None]
    if d.get("blur"):
        f = cv2.GaussianBlur(f, (0, 0), d["blur"])
    if d.get("specks"):
        n = int(W * H / 40000)
        for _ in range(n):
            x, y = int(nrng.integers(0, W)), int(nrng.integers(0, H))
            r = int(nrng.integers(1, 3))
            v = float(nrng.uniform(60, 160))
            cv2.circle(f, (x, y), r, (v, v, v), -1)
    if d.get("noise"):
        f = f + nrng.normal(0, d["noise"], size=(H, W, 1)).astype(np.float32)
    img = np.clip(f, 0, 255).astype(np.uint8)
    return img, M


# ----------------------------------------------------------------------------------------------
# GT assembly
# ----------------------------------------------------------------------------------------------
def generate(idx):
    cfg = config_for(idx)
    rng = random.Random(cfg["seed"])
    P = None
    for _ in range(200):
        P = build_plan(rng, cfg)
        if P is not None:
            break
    if P is None:
        raise RuntimeError(f"layout failed for sample {idx}")
    C, img, labels, extras_gt = render(P, cfg, rng)
    img, M = degrade(img, cfg["degrade"], cfg["seed"])
    A, t0 = M[:, :2], M[:, 2]

    def T(p):
        q = A @ np.array(C.X(p)) + t0
        return [round(float(q[0]), 2), round(float(q[1]), 2)]

    def Tpx(q):
        q = A @ np.array(q, float) + t0
        return [round(float(q[0]), 2), round(float(q[1]), 2)]

    def Tdir(v):
        q = A @ np.array(v, float)
        q = q / np.linalg.norm(q)
        return [round(float(q[0]), 5), round(float(q[1]), 5)]

    def ring(poly):
        return [T(c) for c in list(orient(poly, 1.0).exterior.coords)[:-1]]

    mm = C.mmpp
    walls = [dict(id=w["id"], centerline=[T(w["p0"]), T(w["p1"])], thickness_px=round(w["t"] / mm, 3),
                  kind=w["kind"], polygon=ring(w["gt_poly"]), thickness_mm=w["t"]) for w in P.walls]
    openings = []
    nd = nw = no = 0
    for op in P.openings:
        if op["category"] == "window":
            nw += 1
            oid = f"win{nw}"
        elif op["category"] == "door":
            nd += 1
            oid = f"d{nd}"
        else:
            no += 1
            oid = f"o{no}"
        op["id"] = oid
        slide = None
        if op["slide"]:
            slide = dict(op["slide"])
            if slide.get("direction") is not None:
                slide["direction"] = Tdir(slide["direction"])
        o = dict(id=oid, category=op["category"], type=op["type"], wall_id=op["wall"]["id"],
                 jambs=[T(op["a"]), T(op["b"])], width_px=round(op["w"] / mm, 3), width_mm=op["w"],
                 leaves=[dict(hinge=T(lf["hinge"]), closed_tip=T(lf["closed_tip"]), open_tip=T(lf["open_tip"]),
                              max_angle_deg=90) for lf in op["leaves"]],
                 slide=slide, connects=op["connects"], swing_into=op["swing_into"], handedness=op["handedness"])
        if op.get("role"):
            o["role"] = op["role"]
        openings.append(o)
    rooms = [dict(id=r["id"], polygon=ring(r["interior"]), polygon_centerline=ring(r["poly"]), label=r["label"],
                  type=r["type"], label_source="gt", label_drawn=bool(r.get("label_drawn", False)),
                  area_m2=round(r["interior"].area / 1e6, 3)) for r in P.rooms]
    stairs = [dict(id=s["id"], polygon=ring(s["poly"]), direction=Tdir(s["dir"]), room=s["room"]) for s in P.stairs]
    labs = []
    Hh, Ww = img.shape[:2]
    for lb in labels:
        x0, y0, x1, y1 = lb["bbox"]
        cs = [Tpx(c) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        bb = [round(min(c[0] for c in cs), 1), round(min(c[1] for c in cs), 1),
              round(max(c[0] for c in cs), 1), round(max(c[1] for c in cs), 1)]
        # verifier fix: rotation can push a label partly off the page; clip the bbox to the image and
        # drop labels that are less than 75 % visible (their glyphs are cut and not legible)
        cb = [max(bb[0], 0.0), max(bb[1], 0.0), min(bb[2], float(Ww)), min(bb[3], float(Hh))]
        full = max(bb[2] - bb[0], 1e-6) * max(bb[3] - bb[1], 1e-6)
        vis = max(cb[2] - cb[0], 0.0) * max(cb[3] - cb[1], 0.0)
        if vis < 0.75 * full:
            continue
        bb = [round(v, 1) for v in cb]
        d = dict(text=lb["text"], bbox=bb, kind=lb["kind"])
        if "room" in lb:
            d["room"] = lb["room"]
        labs.append(d)
    name = f"{KEY}_{idx:02d}"
    H, W = img.shape[:2]
    gt = {
        "schema": "floorplan/v0",
        "image": {"file": f"{name}.jpg", "width": int(W), "height": int(H)},
        "scale": {"mm_per_px": mm, "source": "synthetic (exact)"},
        "walls": walls, "openings": openings, "rooms": rooms, "stairs": stairs, "labels": labs,
        "gt_fidelity": {"walls": "full", "door_swing": "full", "door_type": True, "windows": True, "rooms": True,
                        "room_types": True, "derived_from": f"synthetic (tools/synthetic_gen.py index {idx} seed {cfg['seed']})"},
        "conventions": {
            "jambs": "wall centerline points at the two ends of the opening (a=jambs[0], b=jambs[1])",
            "leaves": "hinge/closed_tip = jamb corners on the wall face of the swing side (where leaf+arc are drawn); open_tip = leaf tip at 90 deg",
            "handedness": "schema/US letters viewed from the side the door swings AWAY from: hinge on viewer's left=LH, right=RH (LHR/RHR never used); single leaves only",
            "connects": "swing doors: [non-swing-side room, swing_into]; others: [room on +normal side, room on -normal side]",
            "rooms": "polygon = clear wall-face polygon; polygon_centerline = wall-centerline polygon",
            "winding": "positive shoelace area in image coords (x right, y down): mathematically CCW in the image frame, appears clockwise on screen",
            "slide": "pocket: track_side = jamb whose side holds the wall pocket, direction = unit vector the panel moves to open; bypass sliding: track_side/direction null, panels=2",
            "labels": "bbox is axis-aligned in the final (possibly rotated) image",
        },
        "generator": {"tool": "tools/synthetic_gen.py", "index": idx, "seed": cfg["seed"],
                      "knobs": {k: cfg[k] for k in ("n_rooms", "l_shape", "chamfers", "corridor", "wall_style", "text",
                                                    "dims", "furniture", "extras", "long_px", "degrade_level", "degrade")},
                      "rotation_matrix": [[round(float(v), 8) for v in row] for row in M],
                      "wall_thickness_mm": {"exterior": P.t_ext, "interior": P.t_int},
                      "diagonal_walls_in_rooms": P.chamfered},
    }
    if "scale_bar" in extras_gt:
        sb = extras_gt["scale_bar"]
        gt["scale_bar"] = dict(p0=T(sb["p0"]), p1=T(sb["p1"]), length_mm=sb["length_mm"])
    return name, img, gt, cfg, P


def manifest_entry(name, gt, cfg, P):
    ops = gt["openings"]
    doors = [o for o in ops if o["category"] == "door"]
    d = cfg["degrade"]
    types = {}
    for o in ops:
        types[f"{o['category']}:{o['type']}"] = types.get(f"{o['category']}:{o['type']}", 0) + 1
    deg = ", ".join(f"{k}={v}" for k, v in d.items()) if cfg["degrade_level"] else "none (JPEG q95)"
    notes = (f"seed {cfg['seed']}; {cfg['n_rooms']} rooms; {'L-shape' if cfg['l_shape'] else 'rectangular'}; "
             f"{len(P.chamfered)} diagonal wall(s); corridor={cfg['corridor']}; walls={cfg['wall_style']}; "
             f"extras={cfg['extras']}; degradations: {deg}; openings by type: {types}; "
             f"plain openings (category=opening): {sum(1 for o in ops if o['category'] == 'opening')}. "
             "Handedness: schema/US letters viewed from the side the door swings away from (hinge left=LH, right=RH). "
             "Hinge/closed_tip on the wall face of the swing side; jambs on the wall centerline. "
             "Text glyphs: OpenCV 5.0 putText renders the FONT_HERSHEY_* ids with fonts embedded in the OpenCV "
             "binary (Rubik, SIL OFL 1.1; WenQuanYi Micro Hei also embedded); OFL places no restriction on "
             "images rendered with the font.")
    return dict(file=f"{name}.jpg", gt_file=f"{name}.gt.json", orig_gt_file=None,
                source_page="local generator: tools/synthetic_gen.py (this project; see testdata/synthetic/GENERATOR.md)",
                file_url=None, author="plan-test synthetic generator (own work)",
                license="Own synthetic data, unrestricted (no third-party content)", style="synthetic",
                has_furniture=bool(cfg["furniture"]), has_text=bool(gt["labels"]), has_color="color" in cfg["extras"],
                approx_px=f"{gt['image']['width']}x{gt['image']['height']}", gt_fidelity=gt["gt_fidelity"],
                n_walls=len(gt["walls"]), n_doors=len(doors), n_doors_with_swing=sum(1 for o in doors if o["leaves"]),
                n_windows=sum(1 for o in ops if o["category"] == "window"), n_rooms=len(gt["rooms"]), notes=notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    idxs = list(range(len(TABLE))) if args.all else (args.index or [])
    os.makedirs(args.out, exist_ok=True)
    entries = []
    for idx in idxs:
        name, img, gt, cfg, P = generate(idx)
        q = cfg["degrade"].get("jpeg", 95)
        cv2.imwrite(os.path.join(args.out, f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        with open(os.path.join(args.out, f"{name}.gt.json"), "w") as fh:
            json.dump(gt, fh, indent=1)
        entries.append(manifest_entry(name, gt, cfg, P))
        e = entries[-1]
        print(f"{name}: {e['approx_px']} mmpp={gt['scale']['mm_per_px']} walls={e['n_walls']} doors={e['n_doors']} "
              f"swing={e['n_doors_with_swing']} win={e['n_windows']} rooms={e['n_rooms']}", flush=True)
    if args.all:
        with open(os.path.join(args.out, "manifest.json"), "w") as fh:
            json.dump(entries, fh, indent=1)


if __name__ == "__main__":
    main()
