"""Merge the per-folder manifests into all_manifest.json and write splits.json.

Usage: python3 -I tools/build_manifest.py [ROOT]      (ROOT: this repository, the default)

all_manifest.json: every image sample of FOLDERS, in that order and in its folder manifest's order, with the paths
made relative to ROOT and the counts re-derived from its .gt.json with one definition for all sources:

  n_doors            openings with category == "door"
  n_plain_openings   openings with category == "opening" (doorless gap, type none)
  n_doors_with_swing doors with >= 1 leaf and swing_scorable != false
                     (i.e. hinge / closed_tip / open_tip usable for scoring)
  n_windows          openings with category == "window"

A folder manifest's own counts that differ are kept as source_manifest_counts.

splits.json: a fixed dev / test split. A sample whose name ends in an odd number is dev, an even one test
(synthetic_19 dev, floor_1588 test).

ifc/ holds IFC models, not images with ground truth: it has its own manifest.json and is not listed here.
"""
import collections
import json
import os
import sys

FOLDERS = ["swiss-dwellings", "wikimedia-commons", "synthetic"]

# licence class of each folder (Wikimedia Commons: per file) -> (short licence, redistribution)
LIC = {
    "swiss-dwellings": ("CC BY 4.0", "redistributable-with-attribution"),
    "commons-pd": ("Public domain", "redistributable"),
    "commons-cc0": ("CC0 1.0", "redistributable"),
    "commons-by-sa-4": ("CC BY-SA 4.0", "redistributable-share-alike"),
    "commons-by-sa-3": ("CC BY-SA 3.0", "redistributable-share-alike"),
    "synthetic": ("own synthetic, unrestricted", "redistributable"),
}

FIRST = ["file", "gt_file", "orig_gt_file", "source_page", "file_url", "author", "license", "style",
         "has_furniture", "has_text", "has_color", "approx_px", "gt_fidelity", "n_walls", "n_doors",
         "n_doors_with_swing", "n_windows", "n_rooms", "notes"]
SOURCE_COUNTS = ("n_walls", "n_doors", "n_doors_with_swing", "n_windows", "n_rooms")


def lic_key(folder, license_text):
    if folder != "wikimedia-commons":
        return folder
    t = license_text.lower()
    if "public domain" in t:
        return "commons-pd"
    if "cc0" in t:
        return "commons-cc0"
    if "by-sa 4.0" in t:
        return "commons-by-sa-4"
    return "commons-by-sa-3"


def gt_counts(g):
    ops = g.get("openings", [])
    doors = [o for o in ops if o["category"] == "door"]
    swing = [o for o in doors if o.get("leaves") and o.get("swing_scorable", True) is not False]
    return {
        "n_walls": len(g.get("walls", [])),
        "n_doors": len(doors),
        "n_plain_openings": sum(o["category"] == "opening" for o in ops),
        "n_doors_with_swing": len(swing),
        "n_leaves_scorable": sum(len(o["leaves"]) for o in swing),
        "n_doors_swing_into": sum(o.get("swing_into") is not None for o in swing),
        "n_doors_handedness": sum(o.get("handedness") not in (None, "") for o in swing),
        "n_windows": sum(o["category"] == "window" for o in ops),
        "n_rooms": len(g.get("rooms", [])),
        "n_rooms_typed": sum(r.get("type") not in (None, "unknown") for r in g.get("rooms", [])),
        "n_stairs": len(g.get("stairs", [])),
        "n_walls_with_kind": sum(wl.get("kind") in ("exterior", "interior") for wl in g.get("walls", [])),
        "door_types": dict(sorted(collections.Counter(o["type"] for o in doors).items())),
        "swing_door_types": dict(sorted(collections.Counter(o["type"] for o in swing).items())),
        "window_types": dict(sorted(collections.Counter(o["type"] for o in ops if o["category"] == "window").items())),
        "room_types": dict(sorted(collections.Counter(r.get("type") for r in g.get("rooms", [])).items())),
        "mm_per_px": (g.get("scale") or {}).get("mm_per_px"),
    }


def merged(root):
    out = []
    for folder in FOLDERS:
        for m in json.load(open(os.path.join(root, folder, "manifest.json"), encoding="utf-8")):
            c = gt_counts(json.load(open(os.path.join(root, folder, m["gt_file"]), encoding="utf-8")))
            short, redistribution = LIC[lic_key(folder, m["license"])]
            e = dict(m)
            e.update({k: c[k] for k in SOURCE_COUNTS})
            e["file"] = f"{folder}/{m['file']}"
            e["gt_file"] = f"{folder}/{m['gt_file']}"
            e["orig_gt_file"] = f"{folder}/{m['orig_gt_file']}" if m.get("orig_gt_file") else None
            e.update(source_dir=folder, dataset=folder, license_short=short, redistribution=redistribution, counts=c)
            if any(m[k] != c[k] for k in SOURCE_COUNTS):
                e["source_manifest_counts"] = {k: m[k] for k in SOURCE_COUNTS}
            out.append({**{k: e[k] for k in FIRST}, **{k: v for k, v in e.items() if k not in FIRST}})
    return out


def splits(manifest):
    keys = [e["file"].rsplit(".", 1)[0] for e in manifest]
    odd = lambda k: int(k.rsplit("_", 1)[1]) % 2 == 1          # noqa: E731
    return {"dev": [k for k in keys if odd(k)], "test": [k for k in keys if not odd(k)]}


def write(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = merged(root)
    write(os.path.join(root, "all_manifest.json"), m)
    s = splits(m)
    write(os.path.join(root, "splits.json"), s)
    print(f"{len(m)} samples -> all_manifest.json; splits.json: {len(s['dev'])} dev, {len(s['test'])} test")


if __name__ == "__main__":
    main()
