"""Per-door contact sheet for verifying door leaves (realworld verifier).

Usage: python3 -I tools/realworld_verify_doors.py IMAGE.jpg GT.json OUT.png [cell_px]

Each cell: left = raw crop, right = same crop with hinge (red dot), closed leaf
(blue), open leaf (green), arc (magenta), swing_into label. Doors without leaves
are shown with jambs only (yellow).
"""
import json
import math
import sys

import cv2
import numpy as np


def main():
    img = cv2.imread(sys.argv[1])
    gt = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    cell = int(sys.argv[4]) if len(sys.argv) > 4 else 260
    doors = [o for o in gt["openings"] if o["category"] == "door"]
    cells = []
    for d in doors:
        pts = [d["jambs"][0], d["jambs"][1]]
        for lf in d.get("leaves", []):
            pts += [lf["hinge"], lf["closed_tip"], lf["open_tip"]]
        pts = np.array(pts, float)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        s = max(x1 - x0, y1 - y0, 20)
        m = 0.6 * s
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = s / 2 + m
        X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
        X1, Y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        crop = img[Y0:Y1, X0:X1].copy()
        k = cell / max(crop.shape[:2])
        raw = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)))
        ov = raw.copy()

        def P(p):
            return (int(round((p[0] - X0) * k)), int(round((p[1] - Y0) * k)))

        if not d.get("leaves"):
            cv2.line(ov, P(d["jambs"][0]), P(d["jambs"][1]), (0, 200, 255), 3)
        for lf in d.get("leaves", []):
            h, c, o = (np.array(lf[q], float) for q in ("hinge", "closed_tip", "open_tip"))
            cv2.line(ov, P(h), P(c), (255, 0, 0), 2)
            cv2.line(ov, P(h), P(o), (0, 180, 0), 2)
            r = np.linalg.norm(c - h)
            a0 = math.atan2(c[1] - h[1], c[0] - h[0])
            a1 = math.atan2(o[1] - h[1], o[0] - h[0])
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            arc = [h + r * np.array([math.cos(a0 + da * t), math.sin(a0 + da * t)]) for t in np.linspace(0, 1, 20)]
            for i in range(len(arc) - 1):
                cv2.line(ov, P(arc[i]), P(arc[i + 1]), (255, 0, 255), 1)
            cv2.circle(ov, P(h), 5, (0, 0, 255), -1)
        canvas = np.full((cell + 22, 2 * cell + 6, 3), 255, np.uint8)
        canvas[:raw.shape[0], :raw.shape[1]] = raw
        canvas[:ov.shape[0], cell + 6:cell + 6 + ov.shape[1]] = ov
        lab = f"{d['id']} {d.get('type')} into={d.get('swing_into')} sc={d.get('swing_scorable')}"
        cv2.putText(canvas, lab, (3, cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cells.append(canvas)
    if not cells:
        print("no doors")
        return
    cols = 3
    rows = (len(cells) + cols - 1) // cols
    ch, cw = cells[0].shape[:2]
    sheet = np.full((rows * (ch + 8), cols * (cw + 8), 3), 200, np.uint8)
    for i, c in enumerate(cells):
        r, q = divmod(i, cols)
        sheet[r * (ch + 8):r * (ch + 8) + ch, q * (cw + 8):q * (cw + 8) + cw] = c
    cv2.imwrite(out, sheet)
    print(out, len(cells))


if __name__ == "__main__":
    main()
