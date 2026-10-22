"""Validate and visualise floorplan/v0 ground-truth JSON (the format: README.md, "Ground truth: floorplan/v0").

Usage:
  python3 -I tools/fpgt.py validate GT.json [GT.json ...]
  python3 -I tools/fpgt.py overlay IMAGE.jpg GT.json OUT.png
"""
import json
import math
import sys

CATEGORIES = {"door", "window", "opening"}
TYPES = {"single", "double", "sliding", "folding", "pocket", "revolving", "garage", "none", "unknown"}
FIDELITY = {"walls": {"full", "geometry_only", "none"}, "door_swing": {"full", "hinge_only", "none"}}


def _pt(p):
    return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)


def validate(gt):
    errs = []
    if gt.get("schema") != "floorplan/v0":
        errs.append("schema != floorplan/v0")
    img = gt.get("image", {})
    w, h = img.get("width"), img.get("height")
    if not (isinstance(w, int) and isinstance(h, int)):
        errs.append("image.width/height missing")
        w = h = None
    fid = gt.get("gt_fidelity")
    if not isinstance(fid, dict):
        errs.append("gt_fidelity missing")
    else:
        for k, allowed in FIDELITY.items():
            if fid.get(k) not in allowed:
                errs.append(f"gt_fidelity.{k} not in {sorted(allowed)}")

    def inside(p, where):
        if w is not None and not (-0.05 * w <= p[0] <= 1.05 * w and -0.05 * h <= p[1] <= 1.05 * h):
            errs.append(f"{where}: point {p} outside image")

    wall_ids = set()
    for i, wl in enumerate(gt.get("walls", [])):
        where = f"walls[{i}]"
        wall_ids.add(wl.get("id"))
        cl = wl.get("centerline")
        if not (isinstance(cl, list) and len(cl) == 2 and all(_pt(p) for p in cl)):
            errs.append(f"{where}: centerline must be 2 points")
            continue
        for p in cl:
            inside(p, where)
        if not isinstance(wl.get("thickness_px"), (int, float)):
            errs.append(f"{where}: thickness_px missing")

    room_ids = {r.get("id") for r in gt.get("rooms", [])}
    for i, r in enumerate(gt.get("rooms", [])):
        poly = r.get("polygon")
        if not (isinstance(poly, list) and len(poly) >= 3 and all(_pt(p) for p in poly)):
            errs.append(f"rooms[{i}]: polygon needs >=3 points")

    for i, o in enumerate(gt.get("openings", [])):
        where = f"openings[{i}]({o.get('id')})"
        if o.get("category") not in CATEGORIES:
            errs.append(f"{where}: bad category")
        if o.get("type") not in TYPES and o.get("category") == "door":
            errs.append(f"{where}: bad door type {o.get('type')}")
        jambs = o.get("jambs")
        if not (isinstance(jambs, list) and len(jambs) == 2 and all(_pt(p) for p in jambs)):
            errs.append(f"{where}: jambs must be 2 points")
            continue
        if o.get("wall_id") is not None and wall_ids and o["wall_id"] not in wall_ids:
            errs.append(f"{where}: wall_id {o['wall_id']} not found")
        leaves = o.get("leaves", [])
        if o.get("type") == "single" and len(leaves) != 1 and fid and fid.get("door_swing") == "full":
            errs.append(f"{where}: single door needs 1 leaf")
        if o.get("type") == "double" and len(leaves) != 2 and fid and fid.get("door_swing") == "full":
            errs.append(f"{where}: double door needs 2 leaves")
        for j, lf in enumerate(leaves):
            if not all(_pt(lf.get(k)) for k in ("hinge", "closed_tip", "open_tip")):
                errs.append(f"{where}.leaves[{j}]: hinge/closed_tip/open_tip required")
                continue
            r1 = math.dist(lf["hinge"], lf["closed_tip"])
            r2 = math.dist(lf["hinge"], lf["open_tip"])
            if r1 < 1 or abs(r1 - r2) > 0.25 * max(r1, r2):
                errs.append(f"{where}.leaves[{j}]: |hinge-closed| {r1:.1f} vs |hinge-open| {r2:.1f} mismatch")
        si = o.get("swing_into")
        if si not in (None, "outside") and room_ids and si not in room_ids:
            errs.append(f"{where}: swing_into {si} not a room id")
    return errs


def overlay(image_path, gt, out_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"cannot read {image_path}")
    base = (img * 0.45 + 140).astype(np.uint8)
    t = max(1, round(max(img.shape[:2]) / 800))
    ip = lambda p: (int(round(p[0])), int(round(p[1])))

    for r in gt.get("rooms", []):
        poly = np.array([ip(p) for p in r["polygon"]], np.int32)
        cv2.polylines(base, [poly], True, (200, 120, 0), t)
        c = poly.mean(axis=0).astype(int)
        cv2.putText(base, f"{r.get('id')}:{r.get('type', '')}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (200, 120, 0), t)
    for wl in gt.get("walls", []):
        a, b = wl["centerline"]
        cv2.line(base, ip(a), ip(b), (0, 0, 220), max(t, int(wl.get("thickness_px", 2) * 0.5)))
    for o in gt.get("openings", []):
        a, b = o["jambs"]
        color = (0, 170, 0) if o["category"] == "door" else (220, 0, 220)
        cv2.line(base, ip(a), ip(b), color, 2 * t)
        for lf in o.get("leaves", []):
            hge, ct, ot = lf["hinge"], lf["closed_tip"], lf["open_tip"]
            cv2.circle(base, ip(hge), 3 * t, (0, 0, 0), -1)
            cv2.line(base, ip(hge), ip(ot), (0, 120, 255), 2 * t)
            r = math.dist(hge, ct)
            a0 = math.degrees(math.atan2(ct[1] - hge[1], ct[0] - hge[0]))
            a1 = math.degrees(math.atan2(ot[1] - hge[1], ot[0] - hge[0]))
            d = (a1 - a0 + 540) % 360 - 180
            cv2.ellipse(base, ip(hge), (int(r), int(r)), 0, a0, a0 + d, (0, 120, 255), t)
        label = f"{o.get('id')}:{o.get('type')}" + (f">{o['swing_into']}" if o.get("swing_into") else "")
        cv2.putText(base, label, ip(a), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, color, t)
    cv2.imwrite(out_path, base)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "validate":
        bad = 0
        for path in sys.argv[2:]:
            errs = validate(json.load(open(path)))
            bad += bool(errs)
            print(f"{'FAIL' if errs else 'OK  '} {path}")
            for e in errs[:20]:
                print(f"     - {e}")
        sys.exit(1 if bad else 0)
    elif len(sys.argv) == 5 and sys.argv[1] == "overlay":
        overlay(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)
