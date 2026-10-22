"""Convert the chosen Swiss Dwellings floors with realworld_convert_msd.py and write their manifest entries.

    python -I tools/swiss_dwellings_batch.py GEOMETRIES_CSV                        # FLOORS -> swiss-dwellings/
    python -I tools/swiss_dwellings_batch.py GEOMETRIES_CSV 1627 9106 --out /tmp/x  # some floors elsewhere

GEOMETRIES_CSV is ``swiss-dwellings-v3.0.0/geometries.csv`` of https://zenodo.org/records/7788422 (Swiss Dwellings
v3.0.0, Archilyse AG, CC BY 4.0). Each floor is written as ``floor_<floor_id>`` (.jpg, .gt.json, .orig.csv) by
``tools/realworld_convert_msd.py``, the converter that made realworld_08 / 09 / 10, and gets an entry in
``<out>/manifest.json`` in the same format (added, or replaced when the file is already listed). The floors' rows
are first copied, in file order, to a temporary file, which the converter then reads instead of the 1 GB source
(once per floor otherwise); its output is the same.

Byte for byte reproducible with opencv-python-headless 4.10.0.84 and numpy 2.5.4 (Python 3.12): these give
realworld_08 / 09 / 10 back exactly. OpenCV 4.12 and 5.0 report ``minAreaRect`` angles differently, which turns
some centrelines and jambs end for end (and the order of their ``connects``) and moves a few pixels.

FLOORS, one floor per site, none from the sites of realworld_08 / 09 / 10:

- flats: floors with no commercial unit, whose walls are all four-point polygons, with no holes in any polygon, at
  most 34 m across and with at least four rooms, drawn at random (seed 20261022) within strata of unit count (1, 2,
  3, 4+) and size (< 14, < 24, <= 34 m), then looked at: floors with no or almost no windows, detached parts,
  undefined rooms or only one or two rooms were dropped, and a few more to keep the strata even;
- offices, shops, a practice, a workshop: floors with such rooms (OFFICE, OFFICE_SPACE, OPEN_PLAN_OFFICE,
  MEETING_ROOM, SALESROOM, MEDICAL_ROOM, FACTORY_ROOM, WORKSHOP, ...) and at most three non-rectangular walls and
  one small hole (a column or shaft inside a staircase or foyer; the converter keeps a polygon's outer ring only,
  so the manifest note names it), picked by hand from the 197 floors that have them. Two floors were left out
  because their staircase name, drawn at the ring's centroid, falls into the void in its hole (floor 49015, 49022).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realworld_convert_msd as msd  # noqa: E402

FLOORS = {
    "509": "offices and flats", "1627": "flats (2 units)", "1771": "flats (1 unit)", "1795": "flats (3 units)",
    "2304": "flats (2 units)", "3320": "flats (4 units)", "4889": "flats (2 units)", "5954": "flats (3 units)",
    "6002": "flats (4 units)", "6407": "flats (1 unit)", "6863": "flats (1 unit)", "7396": "offices and a flat",
    "8166": "flats (2 units)", "8432": "flats (3 units)", "9106": "flats (2 units)", "9776": "flats (1 unit)",
    "10355": "medical practice and flats", "11019": "flats (3 units)", "11977": "flats (4 units)",
    "12960": "flats (6 units)", "13773": "flats (4 units)", "13837": "flats (3 units)", "14814": "flats (6 units)",
    "15050": "flats (1 unit)", "15108": "flats (2 units)", "15408": "flats (5 units)", "24468": "flats (4 units)",
    "42450": "flats (1 unit)", "43081": "flats (3 units)", "47886": "shop", "48780": "offices and a workshop",
    "48978": "offices", "48983": "offices", "49063": "shop", "49870": "offices", "49889": "factory floor and offices",
    "49980": "flats (2 units)", "50411": "open-plan office and flats", "50496": "flats (2 units)",
    "51300": "offices and a shop",
}
RECORD = "https://zenodo.org/records/7788422 (doi:10.5281/zenodo.7788422)"
FILE_URL = ("https://zenodo.org/api/records/7788422/files/swiss-dwellings-v3.0.0.zip/content "
            "(member swiss-dwellings-v3.0.0/geometries.csv) :: floor_id {}")
AUTHOR = ("Archilyse AG: M. Standfest, M. Franzen, Y. Schroeder, L. Gonzalez Medina, Y. Villanueva Hernandez, "
          "J. H. Buck, Y.-L. Tan, M. Niedzwiecka, R. Colmegna - Swiss Dwellings v3.0.0")
LICENSE = ("CC BY 4.0 (Zenodo record metadata). Image rendered by tools/realworld_convert_msd.py from the CC BY 4.0 "
           "geometry (adaptation; attribute Archilyse / Swiss Dwellings)")
NOTE = ("Source has door/window rectangles only (no hinge/swing/leaf count): doors drawn as outlined white rectangles "
        "in the wall (no arc; windows look similar but carry a centre line), type unknown, leaves []. Walls, windows, "
        "rooms with types, stairs, scale (mm_per_px) are exact GT. Room-type text is rendered from GT and listed in "
        "labels.")
FURNITURE = {"KITCHEN", "BATHTUB", "SINK", "TOILET", "SHOWER", "BUILT_IN_FURNITURE", "WASHING_MACHINE"}


def copy_rows(src: Path, floor_ids: set[str], dst: Path) -> None:
    """The rows of the given floors, header and file order kept."""
    csv.field_size_limit(10 ** 9)
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        rd, wr = csv.reader(fi), csv.writer(fo)
        head = next(rd)
        wr.writerow(head)
        k = head.index("floor_id")
        wr.writerows(row for row in rd if row[k] in floor_ids)


def _area(ring) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))) / 2


def turn_deg(rows: list[dict]) -> int:
    """How far the building is turned from the axes, from its walls' directions (0 .. 45 degrees)."""
    vx = vy = 0.0
    for r in rows:
        if r["entity_subtype"] != "WALL":
            continue
        for rings in msd.parse_wkt(r["geometry"]):
            ring = rings[0]
            for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
                a, w = 4 * math.atan2(y1 - y0, x1 - x0), math.hypot(x1 - x0, y1 - y0)
                vx, vy = vx + w * math.cos(a), vy + w * math.sin(a)
    t = round(math.degrees(math.atan2(vy, vx)) / 4) % 90
    return min(t, 90 - t)


def manifest_entry(name: str, fid: str, gt: dict, rows: list[dict]) -> dict:
    turn = turn_deg(rows)
    turned = f"building turned ~{turn} deg in the source frame (kept)" if turn >= 3 else "axis-aligned"
    rooms = iter(gt["rooms"])
    holes = []
    for r in rows:                                     # area rows in order, as the converter numbers its rooms
        if r["entity_type"] != "area":
            continue
        for rings in msd.parse_wkt(r["geometry"]):
            rid = next(rooms)["id"]
            if len(rings) > 1:
                holes.append(f"{rid} ({r['entity_subtype']}, {sum(_area(h) for h in rings[1:]):.2f} m2)")
    holes_note = f" Room polygons without their holes: {', '.join(holes)}." if holes else ""
    doors = [o for o in gt["openings"] if o["category"] == "door"]
    return {
        "file": f"{name}.jpg", "gt_file": f"{name}.gt.json", "orig_gt_file": f"{name}.orig.csv",
        "source_page": RECORD, "file_url": FILE_URL.format(fid), "author": AUTHOR, "license": LICENSE,
        "style": "synthetic", "has_furniture": any(r["entity_subtype"] in FURNITURE for r in rows
                                                   if r["entity_type"] == "feature"),
        "has_text": bool(gt["labels"]), "has_color": False,
        "notes": f"Rendered from geometry; {FLOORS.get(fid, 'floor')}, {turned}. {NOTE}{holes_note}",
        "approx_px": [gt["image"]["width"], gt["image"]["height"]], "gt_fidelity": gt["gt_fidelity"],
        "n_walls": len(gt["walls"]), "n_doors": len(doors),
        "n_doors_with_swing": sum(1 for o in doors if o.get("leaves") and o.get("swing_scorable", True)),
        "n_windows": sum(o["category"] == "window" for o in gt["openings"]), "n_rooms": len(gt["rooms"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("csv", type=Path, help="swiss-dwellings-v3.0.0/geometries.csv")
    ap.add_argument("floor_ids", nargs="*", help="default: FLOORS")
    ap.add_argument("--out", type=Path, default=HERE.parent / "swiss-dwellings")
    a = ap.parse_args(argv)
    ids = sorted(a.floor_ids or FLOORS, key=int)
    a.out.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        part = Path(tmp) / "floors.csv"
        copy_rows(a.csv, set(ids), part)
        for fid in ids:
            name = f"floor_{fid}"
            msd.main(str(part), fid, str(a.out / name), 2400)
            gt = json.loads((a.out / f"{name}.gt.json").read_text())
            with open(a.out / f"{name}.orig.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            entries.append(manifest_entry(name, fid, gt, rows))
    path = a.out / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else []
    new = {e["file"] for e in entries}
    path.write_text(json.dumps([e for e in old if e["file"] not in new] + entries, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
