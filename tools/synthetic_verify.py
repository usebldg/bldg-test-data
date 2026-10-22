"""Independent pixel-level verifier for synthetic/ GT (adversarial check).

For every door it builds the 4 (single) / 2 (double) possible swing hypotheses from the jambs,
wall normal and wall thickness alone, scores how much ink lies on each hypothesis' arc and leaf
in the JPG, and checks the GT leaf matches the best-scoring hypothesis. It also checks
swing_into / connects against room polygons, handedness from geometry, slide fields, wall
faces / gaps against pixels and manifest counts.

Usage: python3 -I tools/synthetic_verify.py synthetic
"""
import json
import math
import os
import sys

import cv2
import numpy as np


def load_ink(path, gt):
    img = cv2.imread(path)
    g = img.min(axis=2).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    bg = cv2.dilate(g, np.ones((31, 31), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    dark = np.clip(bg - g, 0, 255)
    deg = gt["generator"]["knobs"].get("degrade_level", 0) or 0
    k = 5 if deg >= 5 or gt["image"]["width"] > 2500 else 3
    dark = cv2.dilate(dark, np.ones((k, k), np.uint8))
    return img, dark


def sample(dark, p):
    x, y = int(round(p[0])), int(round(p[1]))
    if 0 <= x < dark.shape[1] and 0 <= y < dark.shape[0]:
        return dark[y, x]
    return 0.0


def arc_pts(h, c, o, f0=0.12, f1=0.88, n=40):
    h, c, o = map(np.asarray, (h, c, o))
    r = np.linalg.norm(c - h)
    a0 = math.atan2(c[1] - h[1], c[0] - h[0])
    a1 = math.atan2(o[1] - h[1], o[0] - h[0])
    d = (a1 - a0 + 3 * math.pi) % (2 * math.pi) - math.pi
    return [h + r * np.array([math.cos(a0 + d * f), math.sin(a0 + d * f)]) for f in np.linspace(f0, f1, n)]


def seg_pts(a, b, f0=0.2, f1=0.9, n=30):
    a, b = np.asarray(a), np.asarray(b)
    return [a + (b - a) * f for f in np.linspace(f0, f1, n)]


def score(dark, pts, thr):
    v = np.array([sample(dark, p) for p in pts])
    return float((v > thr).mean())


def inside(poly, p):
    return cv2.pointPolygonTest(np.array(poly, np.float32), (float(p[0]), float(p[1])), False) >= 0


def room_at(gt, p):
    for r in gt["rooms"]:
        if inside(r["polygon"], p):
            return r["id"]
    for r in gt["rooms"]:
        if inside(r["polygon_centerline"], p):
            return r["id"] + "?"
    return "outside"


def hyps(op, t, kind):
    a, b = np.array(op["jambs"][0]), np.array(op["jambs"][1])
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    w = np.linalg.norm(b - a)
    out = {}
    for s in (1, -1):
        fa, fb = a + s * n * t / 2, b + s * n * t / 2
        if kind == "single":
            out[("a", s)] = [(fa, fb, fa + s * n * w)]
            out[("b", s)] = [(fb, fa, fb + s * n * w)]
        else:
            m = (fa + fb) / 2
            out[("ab", s)] = [(fa, m, fa + s * n * w / 2), (fb, m, fb + s * n * w / 2)]
    return out, n, u, w


def main(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    tot = dict(doors=0, swing_checked=0, swing_ok=0, issues=0)
    for e in man:
        gtp = os.path.join(d, e["gt_file"])
        gt = json.load(open(gtp))
        img, dark = load_ink(os.path.join(d, e["file"]), gt)
        H, W = img.shape[:2]
        issues = []
        if (W, H) != (gt["image"]["width"], gt["image"]["height"]):
            issues.append(f"image size {W}x{H} != GT")
        if e["approx_px"] != f"{W}x{H}":
            issues.append(f"approx_px {e['approx_px']} != {W}x{H}")
        walls = {w["id"]: w for w in gt["walls"]}
        thr = 45.0 if (gt["generator"]["knobs"].get("degrade_level") or 0) < 9 else 30.0
        margins = []
        for op in gt["openings"]:
            wl = walls[op["wall_id"]]
            t = wl["thickness_px"]
            a, b = np.array(op["jambs"][0]), np.array(op["jambs"][1])
            # jambs on wall centerline
            p0, p1 = np.array(wl["centerline"][0]), np.array(wl["centerline"][1])
            uu = (p1 - p0) / np.linalg.norm(p1 - p0)
            for j in (a, b):
                off = abs(uu[0] * (j - p0)[1] - uu[1] * (j - p0)[0])
                if off > 1.0:
                    issues.append(f"{op['id']} jamb off centerline by {off:.1f}px")
            if abs(np.linalg.norm(b - a) - op["width_px"]) > 1.0:
                issues.append(f"{op['id']} width_px mismatch")
            if op["category"] == "window" or op.get("role"):
                if wl["kind"] != "exterior":
                    issues.append(f"{op['id']} ({op['category']},{op.get('role')}) in non-exterior wall")
            if op["category"] != "door":
                continue
            tot["doors"] += 1
            typ = op["type"]
            if typ in ("single", "double"):
                H_, n, u, w = hyps(op, t, typ)
                # match GT to hypothesis
                key = None
                for k, lv in H_.items():
                    if len(lv) == len(op["leaves"]) and all(
                            max(np.linalg.norm(np.array(L[q]) - P) for q, P in zip(("hinge", "closed_tip", "open_tip"), hl)) < 1.5
                            for L, hl in zip(op["leaves"], lv)):
                        key = k
                if key is None:
                    issues.append(f"{op['id']} GT leaves match no geometric hypothesis")
                    continue
                sc = {}
                for k, lv in H_.items():
                    s_arc = np.mean([score(dark, arc_pts(*hl), thr) for hl in lv])
                    s_leaf = np.mean([score(dark, seg_pts(hl[0], hl[2]), thr) for hl in lv])
                    sc[k] = (s_arc + s_leaf) / 2
                best = max(sc, key=sc.get)
                second = max(v for k, v in sc.items() if k != key) if len(sc) > 1 else 0
                tot["swing_checked"] += 1
                margins.append(sc[key] - second)
                if best == key and sc[key] > 0.55:
                    tot["swing_ok"] += 1
                else:
                    issues.append(f"{op['id']} {typ} swing pixel check FAILED gt={key} {sc[key]:.2f} best={best} {sc[best]:.2f}")
                s = key[1]
                # swing_into
                lv = H_[key]
                hg, ct, ot = lv[0]
                pin = hg + 0.4 * (ct - hg) + 0.4 * (ot - hg)
                rin = room_at(gt, pin)
                if rin != op["swing_into"]:
                    issues.append(f"{op['id']} swing_into={op['swing_into']} but sweep point in {rin}")
                pout = (a + b) / 2 - s * n * (t / 2 + 0.4 * w)
                rout = room_at(gt, pout)
                if op["connects"] != [rout, op["swing_into"]]:
                    issues.append(f"{op['id']} connects={op['connects']} geometry says [{rout},{rin}]")
                if typ == "single":
                    f = s * n
                    left = np.array([f[1], -f[0]])
                    hand = "LH" if np.dot(hg - (a + b) / 2, left) > 0 else "RH"
                    if hand != op["handedness"]:
                        issues.append(f"{op['id']} handedness {op['handedness']} but geometry says {hand}")
                elif op["handedness"] is not None:
                    issues.append(f"{op['id']} double with handedness")
            else:
                if op["leaves"]:
                    issues.append(f"{op['id']} {typ} has leaves")
                sl = op.get("slide")
                if typ in ("sliding", "pocket") and not sl:
                    issues.append(f"{op['id']} {typ} missing slide")
                u = (b - a) / np.linalg.norm(b - a)
                n = np.array([-u[1], u[0]])
                # ink inside the gap (panels)
                band = [a + (b - a) * f + n * t * o for f in np.linspace(0.1, 0.9, 25) for o in (-0.25, 0, 0.25)]
                ink_in_gap = score(dark, band, thr)
                if ink_in_gap < 0.3:
                    issues.append(f"{op['id']} {typ} no panel ink in gap ({ink_in_gap:.2f})")
                if typ == "pocket":
                    tr = sl["track_side"]
                    J, O = (a, b) if tr == "a" else (b, a)
                    dv = np.array(sl["direction"])
                    if np.dot(dv, J - O) <= 0:
                        issues.append(f"{op['id']} pocket direction does not point to track side")
                    g = img.min(axis=2)
                    w = np.linalg.norm(b - a)
                    vin = np.median([g[int(round(q[1])), int(round(q[0]))] for q in seg_pts(J + dv * 0.1 * w, J + dv * 0.4 * w, 0, 1, 15)])
                    vout = np.median([g[int(round(q[1])), int(round(q[0]))] for q in seg_pts(O - dv * 0.1 * w, O - dv * 0.4 * w, 0, 1, 15)])
                    vgap = np.median([g[int(round(q[1])), int(round(q[0]))] for q in seg_pts(J - dv * 0.6 * w, O + dv * 0.1 * w, 0, 1, 15)])
                    op["_pk"] = (round(float(vin)), round(float(vout)), round(float(vgap)))
        # walls: face lines dark away from openings
        wall_bad = []
        ops_by_wall = {}
        for op in gt["openings"]:
            ops_by_wall.setdefault(op["wall_id"], []).append(op)
        for wl in gt["walls"]:
            p0, p1 = np.array(wl["centerline"]), None
            p0, p1 = np.array(wl["centerline"][0]), np.array(wl["centerline"][1])
            L = np.linalg.norm(p1 - p0)
            u = (p1 - p0) / L
            n = np.array([-u[1], u[0]])
            t = wl["thickness_px"]
            iv = []
            for op in ops_by_wall.get(wl["id"], []):
                sa = np.dot(np.array(op["jambs"][0]) - p0, u)
                sb = np.dot(np.array(op["jambs"][1]) - p0, u)
                iv.append((min(sa, sb), max(sa, sb), op))
            hits = tot_s = 0
            gap_hits = gap_n = 0
            for s in np.linspace(t, L - t, max(10, int(L / 4))):
                ins = [o for lo, hi, o in iv if lo - 3 <= s <= hi + 3]
                for side in (1, -1):
                    q = p0 + u * s + n * side * t * 0.47
                    v = sample(dark, q) > thr
                    if not ins:
                        tot_s += 1
                        hits += v
                    elif all(o["category"] in ("door", "opening") and o["type"] in ("single", "double", "none") for o in ins):
                        lo_hi = [(lo, hi) for lo, hi, o in iv if lo - 3 <= s <= hi + 3][0]
                        if lo_hi[0] + 0.15 * (lo_hi[1] - lo_hi[0]) < s < lo_hi[1] - 0.15 * (lo_hi[1] - lo_hi[0]):
                            gap_n += 1
                            gap_hits += v
            if tot_s and hits / tot_s < 0.8:
                wall_bad.append(f"{wl['id']} face ink {hits / tot_s:.2f}")
            if gap_n and gap_hits / gap_n > 0.5:
                wall_bad.append(f"{wl['id']} door gap inked {gap_hits / gap_n:.2f}")
        issues += wall_bad

        # windows: centre line inked, glass band light (solid/gray walls)
        style = gt["generator"]["knobs"]["wall_style"]
        for op in gt["openings"]:
            if op["category"] != "window":
                continue
            t = walls[op["wall_id"]]["thickness_px"]
            a, b = np.array(op["jambs"][0]), np.array(op["jambs"][1])
            u = (b - a) / np.linalg.norm(b - a)
            n = np.array([-u[1], u[0]])
            c = score(dark, seg_pts(a, b, 0.1, 0.9, 30), thr)
            gl = score(dark, [q + n * sg * t * 0.25 for q in seg_pts(a, b, 0.1, 0.9, 30) for sg in (1, -1)], thr)
            if c < 0.8 or (style in ("solid", "gray") and t > 8 and gl > 0.3):
                issues.append(f"{op['id']} window symbol not found (centre {c:.2f}, glass {gl:.2f})")
        # rooms: edges on wall faces, labels inside, area
        for r in gt["rooms"]:
            poly = np.array(r["polygon"])
            pts = []
            for i in range(len(poly)):
                p_, q_ = poly[i], poly[(i + 1) % len(poly)]
                pts += seg_pts(p_, q_, 0.05, 0.95, max(5, int(np.linalg.norm(q_ - p_) / 5)))
            f = score(dark, pts, thr)
            if f < 0.6:
                issues.append(f"{r['id']} polygon edges on ink only {f:.2f}")
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            if area <= 0:
                issues.append(f"{r['id']} winding negative")
            mm = gt["scale"]["mm_per_px"]
            if abs(area * mm * mm / 1e6 - r["area_m2"]) > 0.02 * r["area_m2"]:
                issues.append(f"{r['id']} area_m2 {r['area_m2']} vs polygon {area * mm * mm / 1e6:.2f}")
        rid = {r["id"]: r for r in gt["rooms"]}
        for lb in gt["labels"]:
            x0, y0, x1, y1 = lb["bbox"]
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                issues.append(f"label {lb['text']} bbox out of image")
            sub = dark[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
            if sub.size == 0 or (sub > thr).mean() < 0.05:
                issues.append(f"label {lb['text']} bbox has no ink")
            if lb.get("room"):
                if not inside(rid[lb["room"]]["polygon"], ((x0 + x1) / 2, (y0 + y1) / 2)):
                    issues.append(f"label {lb['text']} not inside room {lb['room']}")
                if lb["kind"] == "room_name" and lb["text"] != rid[lb["room"]]["label"]:
                    issues.append(f"label {lb['text']} != room label {rid[lb['room']]['label']}")
        for st in gt["stairs"]:
            c = np.mean(np.array(st["polygon"]), axis=0)
            if room_at(gt, c) != st.get("room"):
                issues.append(f"stair {st['id']} not in room {st.get('room')}")
        mm = gt["scale"]["mm_per_px"]
        for op in gt["openings"]:
            if abs(op["width_px"] * mm - op["width_mm"]) > 15:
                issues.append(f"{op['id']} width_mm inconsistent with scale")
        if "scale_bar" in gt:
            sb = gt["scale_bar"]
            if abs(math.dist(sb["p0"], sb["p1"]) * mm - sb["length_mm"]) > 30:
                issues.append("scale bar length inconsistent")
        # exterior walls: one side must be outside
        for wl in gt["walls"]:
            p0, p1 = np.array(wl["centerline"][0]), np.array(wl["centerline"][1])
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            n = np.array([-u[1], u[0]])
            m_ = (p0 + p1) / 2
            sides = [room_at(gt, m_ + n * sg * (wl["thickness_px"] / 2 + 6)) for sg in (1, -1)]
            ext = "outside" in sides
            if ext != (wl["kind"] == "exterior"):
                issues.append(f"{wl['id']} kind {wl['kind']} but sides {sides}")
        # manifest counts
        doors = [o for o in gt["openings"] if o["category"] == "door"]
        cnt = dict(n_walls=len(gt["walls"]), n_doors=len(doors), n_doors_with_swing=sum(1 for o in doors if o["leaves"]),
                   n_windows=sum(1 for o in gt["openings"] if o["category"] == "window"), n_rooms=len(gt["rooms"]))
        for k, v in cnt.items():
            if e[k] != v:
                issues.append(f"manifest {k}={e[k]} actual {v}")
        if e["gt_fidelity"] != gt["gt_fidelity"]:
            issues.append("manifest gt_fidelity differs")
        pk = [(o["id"], o["slide"]["track_side"], o["_pk"]) for o in gt["openings"] if "_pk" in o]
        tot["issues"] += len(issues)
        print(f"{e['file']}: doors={cnt['n_doors']} minmargin={min(margins) if margins else None:.2f} issues={len(issues)} pocket(in,out,gap)={pk}")
        for i in issues:
            print("   -", i)
    print(tot)


if __name__ == "__main__":
    main(sys.argv[1])
