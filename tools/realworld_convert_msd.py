"""Render a Swiss Dwellings (Archilyse, Zenodo 7788422, CC BY 4.0) floor to a plan image and floorplan/v0 GT.

Usage:
  python3 -I tools/realworld_convert_msd.py geometries.csv FLOOR_ID OUT_PREFIX [--long-side 2400]

Source semantics (geometries.csv, one row per entity, metres, y up):
  entity_type  area      -> rooms (entity_subtype = BEDROOM, KITCHEN, ...)
               separator -> WALL (walls), COLUMN, RAILING
               opening   -> DOOR, ENTRANCE_DOOR, WINDOW: rectangles lying inside the wall
               feature   -> KITCHEN, SINK, TOILET, BATHTUB, SHOWER, STAIRS, ELEVATOR, ...
  Door geometry is only the rectangle in the wall: there is NO hinge, swing side, leaf
  count or swing angle in the source. The rendered image therefore draws doors as a
  plain gap with a thin threshold outline (no swing arc), and gt door_swing = none.

Rendering: walls/columns filled dark grey, windows as a white rectangle with a centre line,
doors as white gaps with an outline, railings thin, features as thin outlines, room-type
text labels at room centroids. Scale is known exactly (scale.mm_per_px).
"""
import csv
import json
import math
import os
import re
import sys

import cv2
import numpy as np

ROOM_TYPE = {
    "BEDROOM": "bedroom", "BATHROOM": "bath", "SANITARY_ROOMS": "bath", "KITCHEN": "kitchen",
    "KITCHEN_DINING": "kitchen", "LIVING_ROOM": "living", "LIVING_DINING": "living", "DINING": "living",
    "CORRIDOR": "corridor", "CORRIDORS_AND_HALLS": "corridor", "LOBBY": "corridor", "FOYER": "corridor",
    "STOREROOM": "storage", "BASEMENT_COMPARTMENT": "storage", "BALCONY": "balcony", "LOGGIA": "balcony",
    "TERRACE": "outdoor", "GARDEN": "outdoor", "WINTERGARTEN": "balcony",
}


def parse_wkt(s):
    """Return list of polygons, each a list of rings (outer first) of (x, y)."""
    s = s.strip()
    out = []
    if s.startswith("POLYGON"):
        bodies = [s[s.index("(") :]]
    elif s.startswith("MULTIPOLYGON"):
        inner = s[s.index("(") + 1 : s.rindex(")")]
        bodies = re.findall(r"\(\([^()]*\)(?:\s*,\s*\([^()]*\))*\)", inner)
    else:
        return out
    for b in bodies:
        rings = []
        for ring in re.findall(r"\(([^()]*)\)", b):
            pts = [tuple(map(float, p.split()[:2])) for p in ring.split(",")]
            rings.append(pts)
        out.append(rings)
    return out


def ccw(poly):
    a = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    return poly if a < 0 else poly[::-1]


def r2(p):
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def load_floor(csv_path, floor_id):
    csv.field_size_limit(10 ** 9)
    rows, seen = [], set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if row["floor_id"] != floor_id:
                    continue
            except (KeyError, TypeError):
                continue
            key = (row["entity_type"], row["entity_subtype"], row["geometry"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def main(csv_path, floor_id, out_prefix, long_side):
    rows = load_floor(csv_path, floor_id)
    if not rows:
        raise SystemExit("floor not found")
    ents = []
    for row in rows:
        for poly in parse_wkt(row["geometry"]):
            ring = poly[0][:-1] if poly[0][0] == poly[0][-1] else poly[0]
            ents.append((row["entity_type"], row["entity_subtype"], ring, row))
    xs = [p[0] for e in ents for p in e[2]]
    ys = [p[1] for e in ents for p in e[2]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    margin_m = 1.0
    ppm = long_side / (max(maxx - minx, maxy - miny) + 2 * margin_m)  # pixels per metre
    W = int(math.ceil((maxx - minx + 2 * margin_m) * ppm))
    H = int(math.ceil((maxy - miny + 2 * margin_m) * ppm))
    T = lambda p: (float((p[0] - minx + margin_m) * ppm), float((maxy - p[1] + margin_m) * ppm))
    P = lambda ring: np.array([[round(x), round(y)] for x, y in map(T, ring)], np.int32)

    img = np.full((H, W, 3), 255, np.uint8)
    lw = max(1, round(ppm * 0.02))
    # rooms: very light fill so the plan reads like a CAD print; then features, walls, openings
    for et, st, ring, _ in ents:
        if et == "feature":
            cv2.polylines(img, [P(ring)], True, (110, 110, 110), lw, cv2.LINE_AA)
    for et, st, ring, _ in ents:
        if et == "separator" and st in ("WALL", "COLUMN"):
            cv2.fillPoly(img, [P(ring)], (60, 60, 60), cv2.LINE_AA)
        elif et == "separator" and st == "RAILING":
            cv2.polylines(img, [P(ring)], True, (60, 60, 60), lw, cv2.LINE_AA)
    for et, st, ring, _ in ents:
        if et == "opening":
            cv2.fillPoly(img, [P(ring)], (255, 255, 255), cv2.LINE_AA)
            cv2.polylines(img, [P(ring)], True, (60, 60, 60), lw, cv2.LINE_AA)
            if st == "WINDOW":
                (cx, cy), (w, h), ang = cv2.minAreaRect(P(ring).astype(np.float32))
                a = math.radians(ang)
                u = (math.cos(a), math.sin(a)) if w >= h else (-math.sin(a), math.cos(a))
                L = max(w, h) / 2
                cv2.line(img, (round(cx - u[0] * L), round(cy - u[1] * L)), (round(cx + u[0] * L), round(cy + u[1] * L)),
                         (60, 60, 60), lw, cv2.LINE_AA)

    walls, openings, rooms, stairs, labels = [], [], [], [], []
    room_polys = []
    for et, st, ring, row in ents:
        if et == "area":
            poly = [T(p) for p in ring]
            rid = f"r{len(rooms) + 1}"
            rooms.append({"id": rid, "polygon": [r2(p) for p in ccw(poly)], "label": st,
                          "type": ROOM_TYPE.get(st, "other"), "label_source": "gt", "source_subtype": st,
                          "unit_id": row["unit_id"], "area_id": row["area_id"]})
            room_polys.append((rid, np.array(poly, np.float32)))
    # room labels (text rendered in the image = labels GT)
    fs = ppm * 0.012
    for rm in rooms:
        if rm["source_subtype"] in ("SHAFT", "VOID", "OUTDOOR_VOID", "ELEVATOR", "LIGHTWELL"):
            continue
        poly = np.array(rm["polygon"], np.float32)
        M = cv2.moments(poly)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if cv2.pointPolygonTest(poly, (cx, cy), False) < 0:
            continue
        text = rm["source_subtype"].replace("_", " ")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw))
        org = (int(cx - tw / 2), int(cy + th / 2))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), max(1, lw), cv2.LINE_AA)
        labels.append({"text": text, "bbox": [org[0], org[1] - th, org[0] + tw, org[1]], "kind": "room_name"})

    def room_at(pt):
        for rid, poly in room_polys:
            if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                return rid
        return None

    for et, st, ring, row in ents:
        if et == "separator" and st == "WALL":
            poly = [T(p) for p in ring]
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.array(poly, np.float32))
            a = math.radians(ang)
            u = (math.cos(a), math.sin(a)) if w >= h else (-math.sin(a), math.cos(a))
            L = max(w, h) / 2
            walls.append({"id": f"w{len(walls) + 1}", "centerline": [r2((cx - u[0] * L, cy - u[1] * L)), r2((cx + u[0] * L, cy + u[1] * L))],
                          "thickness_px": round(float(min(w, h)), 2), "kind": "unknown",
                          "polygon": [r2(p) for p in ccw(poly)]})
        elif et == "feature" and st == "STAIRS":
            stairs.append({"id": f"s{len(stairs) + 1}", "polygon": [r2(p) for p in ccw([T(p) for p in ring])], "direction": None})

    def wall_for(c, u):
        best, bd = None, 1e18
        for wl in walls:
            a, b = np.array(wl["centerline"][0]), np.array(wl["centerline"][1])
            d = b - a
            Ld = np.linalg.norm(d)
            if Ld < 1 or abs(np.dot(d / Ld, u)) < 0.9:
                continue
            t = np.clip(np.dot(c - a, d) / Ld ** 2, 0, 1)
            dist = np.linalg.norm(a + t * d - c)
            if dist < bd:
                bd, best = dist, wl["id"]
        return best if bd < ppm * 0.5 else None

    nd = nw = 0
    for et, st, ring, row in ents:
        if et != "opening":
            continue
        poly = [T(p) for p in ring]
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.array(poly, np.float32))
        a = math.radians(ang)
        u = np.array((math.cos(a), math.sin(a)) if w >= h else (-math.sin(a), math.cos(a)))
        n = np.array([-u[1], u[0]])
        L, D = max(w, h), min(w, h)
        c = np.array([cx, cy])
        if L / max(D, 1e-6) < 1.3:
            # near-square opening (e.g. small window in a thick wall): the minAreaRect long axis
            # is arbitrary, so take the axis of the nearest wall instead (verifier fix)
            bw, bd = None, 1e18
            for wl in walls:
                wa, wb = np.array(wl["centerline"][0]), np.array(wl["centerline"][1])
                dd = wb - wa
                Ld = np.linalg.norm(dd)
                if Ld < 1:
                    continue
                t = np.clip(np.dot(c - wa, dd) / Ld ** 2, 0, 1)
                dist = np.linalg.norm(wa + t * dd - c)
                if dist < bd:
                    bd, bw = dist, dd / Ld
            if bw is not None and bd < ppm * 0.5:
                u = bw
                n = np.array([-u[1], u[0]])
                pu = [float(np.dot(np.array(q) - c, u)) for q in poly]
                pn = [float(np.dot(np.array(q) - c, n)) for q in poly]
                L, D = max(pu) - min(pu), max(pn) - min(pn)
                c = c + u * (max(pu) + min(pu)) / 2 + n * (max(pn) + min(pn)) / 2
        ja, jb = c - u * L / 2, c + u * L / 2
        sides = []
        for sg in (1, -1):
            rr = room_at(c + sg * n * (D / 2 + 0.3 * ppm))
            sides.append(rr if rr else "outside")
        base = {"wall_id": wall_for(c, u), "jambs": [r2(ja), r2(jb)], "width_px": round(float(L), 2),
                "depth_px": round(float(D), 2), "polygon": [r2(p) for p in ccw(poly)], "leaves": [],
                "source_subtype": st}
        if st == "WINDOW":
            nw += 1
            openings.append({"id": f"win{nw}", "category": "window", "type": "unknown", **base,
                             "connects": sides})
        else:
            nd += 1
            openings.append({"id": f"d{nd}", "category": "door", "type": "unknown", **base, "slide": None,
                             "connects": sides, "swing_into": None, "handedness": None,
                             "is_entrance": st == "ENTRANCE_DOOR", "swing_scorable": False})

    site = rows[0]["site_id"]
    gt = {
        "schema": "floorplan/v0",
        "image": {"file": os.path.basename(out_prefix) + ".jpg", "width": W, "height": H},
        "scale": {"mm_per_px": round(1000.0 / ppm, 4), "source": "gt (Swiss Dwellings metric coordinates)"},
        "walls": walls, "openings": openings, "rooms": rooms, "stairs": stairs, "labels": labels,
        "gt_fidelity": {"walls": "full", "door_swing": "none", "door_type": False, "windows": True,
                        "rooms": True, "room_types": True,
                        "derived_from": f"Swiss Dwellings v3.0.0 geometries.csv floor_id={floor_id} site_id={site}; "
                                        "image rendered from the GT geometry (doors drawn as outlined white rectangles in the wall, no swing arc: source has no swing)"},
        "source": {"dataset": "Swiss Dwellings v3.0.0", "floor_id": floor_id, "site_id": site,
                   "plan_id": rows[0]["plan_id"], "building_id": rows[0]["building_id"],
                   "unit_ids": sorted({r["unit_id"] for r in rows}), "px_per_m": round(ppm, 4),
                   "transform": f"x_px=(x_m-{minx:.4f}+{margin_m})*ppm, y_px=({maxy:.4f}-y_m+{margin_m})*ppm"},
    }
    cv2.imwrite(out_prefix + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(gt, open(out_prefix + ".gt.json", "w"), indent=1)
    # original GT rows for this floor
    with open(out_prefix + ".orig.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(out_prefix, W, H, "walls", len(walls), "doors", nd, "windows", nw, "rooms", len(rooms), "stairs", len(stairs))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    ls = int(sys.argv[sys.argv.index("--long-side") + 1]) if "--long-side" in sys.argv else 2400
    main(sys.argv[1], sys.argv[2], sys.argv[3], ls)
