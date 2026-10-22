"""Fetch visual-only floor plan images from Wikimedia Commons with licence metadata.

Usage:
  python3 -I tools/realworld_fetch_commons.py RAW_DIR OUT_DIR KEY_START "File:A.jpg=style" ["File:B.png=style" ...]

For each title: queries the Commons API (imageinfo + extmetadata), downloads the original
file into RAW_DIR (untrusted data, never executed), converts it to JPG (long side clamped
to 800..4000 px) as OUT_DIR/realworld_NN.jpg, writes a visual-only floorplan/v0 GT
(no walls/openings/rooms; gt_fidelity all none/false) and a sidecar
OUT_DIR/realworld_NN.commons.json with the licence metadata used for the manifest.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

UA = {"User-Agent": "plan-test-research/0.1 (local floor plan test set; contact: none)"}
API = "https://commons.wikimedia.org/w/api.php?"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def main(raw_dir, out_dir, start, items):
    os.makedirs(raw_dir, exist_ok=True)
    k = start
    for item in items:
        title, style = item.rsplit("=", 1)
        q = API + urllib.parse.urlencode({"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                                          "iiprop": "url|size|mime|extmetadata|sha1"})
        page = next(iter(json.loads(get(q))["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        em = {k2: v.get("value") for k2, v in ii.get("extmetadata", {}).items()}
        raw_path = os.path.join(raw_dir, re.sub(r"[^A-Za-z0-9._-]+", "_", title[5:]))
        data = get(ii["url"])
        open(raw_path, "wb").write(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("SKIP undecodable", title)
            continue
        h, w = img.shape[:2]
        s = 1.0
        if max(h, w) > 4000:
            s = 4000 / max(h, w)
        elif max(h, w) < 800:
            print("SKIP too small", title, w, h)
            continue
        if s != 1.0:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        name = f"realworld_{k:02d}"
        cv2.imwrite(os.path.join(out_dir, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        gt = {"schema": "floorplan/v0",
              "image": {"file": name + ".jpg", "width": img.shape[1], "height": img.shape[0]},
              "scale": {"mm_per_px": None, "source": None},
              "walls": [], "openings": [], "rooms": [], "stairs": [], "labels": [],
              "gt_fidelity": {"walls": "none", "door_swing": "none", "door_type": False, "windows": False,
                              "rooms": False, "room_types": False,
                              "derived_from": "none (visual-only Wikimedia Commons image; no ground truth)"},
              "source": {"dataset": "Wikimedia Commons", "title": title, "scale_applied": s}}
        json.dump(gt, open(os.path.join(out_dir, name + ".gt.json"), "w"), indent=1)
        meta = {"title": title, "style": style, "source_page": ii["descriptionurl"], "file_url": ii["url"],
                "sha1": ii.get("sha1"), "orig_size": [w, h], "mime": ii.get("mime"),
                "license": em.get("LicenseShortName"), "license_url": em.get("LicenseUrl"),
                "usage_terms": em.get("UsageTerms"), "attribution_required": em.get("AttributionRequired"),
                "author": strip_html(em.get("Artist")), "credit": strip_html(em.get("Credit")),
                "description": strip_html(em.get("ImageDescription"))[:500], "date": strip_html(em.get("DateTimeOriginal"))}
        json.dump(meta, open(os.path.join(out_dir, name + ".commons.json"), "w"), indent=1)
        print(name, title, w, h, "->", img.shape[1], img.shape[0], meta["license"], "|", meta["author"][:80])
        k += 1


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])
