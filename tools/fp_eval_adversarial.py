"""Adversarial / independent checks for tools/fp_eval.py (complements fp_eval_selftest.py).

  .venv-eval/bin/python -I tools/fp_eval_adversarial.py [--testdata DIR] [--quick]

What is different from the selftest:
  * tiny hand-made plans with hand-computed expected values (gates at their exact edges,
    hinge/swing signs, double-door leaf pairing, swing_into through RENAMED P room ids,
    exact wall IoU / centerline numbers, rooms with bow-ties / holes, unicode),
  * a perturbation oracle on every real door: each scorable door gets a random, *recorded*
    action (delete, clear leaves, wrong hinge side, wrong swing side, both, jamb reversal,
    leaf-order reversal, 1 px jitter, wrong swing_into) and the expected per-dataset counts are
    derived from the recorded actions, not from fp_eval's formulas,
  * geometric invariance: mirroring / rotating G and P together must not change any metric
    (catches sign errors in the y-down frame math),
  * an independent shapely oracle for wall IoU and centerline P/R,
  * garbage predictions, a corrupt PRED_DIR through the CLI, 10k-item predictions for runtime.
Exit code 0 when every check passes.
"""
import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import fp_eval as fe  # noqa: E402

TESTDATA = ROOT
if "--testdata" in sys.argv:
    TESTDATA = sys.argv[sys.argv.index("--testdata") + 1]
QUICK = "--quick" in sys.argv
MANIFEST = fe.load_manifest(TESTDATA)
IDS = sorted(MANIFEST)
GTS = {sid: json.load(open(os.path.join(TESTDATA, MANIFEST[sid]["gt_file"]))) for sid in IDS}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def ev(P, G):
    c, info = fe.evaluate_pair(P, G)
    m = fe.finalize(c)
    return {k: m[k]["value"] for k in m}, m, c, info


FULL = {"walls": "full", "door_swing": "full", "door_type": True, "windows": True, "rooms": True, "room_types": True}


# =========================================================================== 1. tiny single door
# 200x100 image, one horizontal wall y=50 (t=10 -> τ1=5, τ2=10), one 40 px door with jambs
# (50,50)-(90,50). Frame: m=(70,50), u=(1,0), n=(0,1) (points DOWN in image coords).
# G leaf: hinge (50,50) -> h=-1; open_tip (50,90) -> (0,40)·n = +40 -> s=+1. Door gate = max(10, 20) = 20.
def tiny_single():
    return {"schema": "floorplan/v0", "image": {"width": 200, "height": 100}, "gt_fidelity": dict(FULL),
            "walls": [{"id": "w1", "centerline": [[0, 50], [200, 50]], "thickness_px": 10, "kind": "exterior",
                       "polygon": [[0, 45], [200, 45], [200, 55], [0, 55]]}],
            "openings": [{"id": "d1", "category": "door", "type": "single", "wall_id": "w1",
                          "jambs": [[50, 50], [90, 50]], "width_px": 40,
                          "leaves": [{"hinge": [50, 50], "closed_tip": [90, 50], "open_tip": [50, 90]}],
                          "swing_into": "r2"},
                         {"id": "d2", "category": "door", "type": "single", "wall_id": "w1",
                          "jambs": [[150, 50], [180, 50]], "width_px": 30,
                          "leaves": [{"hinge": [180, 50], "closed_tip": [150, 50], "open_tip": [180, 20]}],
                          "swing_into": "outside"}],
            "rooms": [{"id": "r1", "polygon": [[0, 0], [200, 0], [200, 45], [0, 45]], "type": "kitchen"},
                      {"id": "r2", "polygon": [[0, 55], [200, 55], [200, 100], [0, 100]], "type": "bedroom"}]}


def pred_of(G):
    P = copy.deepcopy(G)
    P.pop("gt_fidelity", None)
    return P


def door(P, i):
    return P["openings"][i]


def test_tiny_single():
    G = tiny_single()
    v, m, c, info = ev(pred_of(G), G)
    check("tiny: τ from median thickness 10 -> τ1=5, τ2=10", info["tau"] == {"t": 10.0, "tau1": 5.0, "tau2": 10.0}, info["tau"])
    bad = {k: x for k, x in v.items() if x is not None and k not in fe.DIAGNOSTIC and x != 1.0}
    check("tiny: P=G perfect", not bad, bad)

    # (1) reversed jambs + hinge at the far jamb + swing up -> h and s both wrong
    P = pred_of(G)
    door(P, 0).update(jambs=[[90, 50], [50, 50]],
                      leaves=[{"hinge": [88, 52], "closed_tip": [52, 52], "open_tip": [88, 10]}])
    v, *_ = ev(P, G)
    check("tiny: hinge+swing wrong -> hinge 1/2, swing 1/2, dir 1/2",
          close(v["hinge_side_acc"], 0.5) and close(v["swing_side_acc"], 0.5) and close(v["door_dir_acc"], 0.5),
          (v["hinge_side_acc"], v["swing_side_acc"], v["door_dir_acc"]))
    # (2) reversed jambs only, leaf correct (P hinge on the centerline, G-like) -> all right
    P = pred_of(G)
    door(P, 0)["jambs"] = [[90, 50], [50, 50]]
    v, *_ = ev(P, G)
    check("tiny: P jamb order irrelevant", v["door_dir_acc"] == 1.0)
    # (3) hinge right, swing wrong; small open angle (10°) still gives the right swing side
    P = pred_of(G)
    door(P, 0)["leaves"] = [{"hinge": [52, 50], "closed_tip": [90, 50], "open_tip": [52, 15]}]
    door(P, 1)["leaves"] = [{"hinge": [180, 50], "closed_tip": [150, 50],
                             "open_tip": [180 - 30 * math.cos(math.radians(10)), 50 - 30 * math.sin(math.radians(10))]}]
    v, *_ = ev(P, G)
    check("tiny: swing wrong on d1, 10° leaf on d2 right -> hinge 1, swing 1/2",
          v["hinge_side_acc"] == 1.0 and close(v["swing_side_acc"], 0.5), (v["hinge_side_acc"], v["swing_side_acc"]))
    # (4) hinge exactly at the midpoint is not a side
    P = pred_of(G)
    door(P, 0)["leaves"] = [{"hinge": [70, 50], "closed_tip": [90, 50], "open_tip": [70, 90]}]
    v, *_ = ev(P, G)
    check("tiny: hinge at jamb midpoint counts as wrong hinge side", close(v["hinge_side_acc"], 0.5))
    # (5) matched P door with no leaves -> wrong for hinge/swing/dir, still in n
    P = pred_of(G)
    door(P, 0)["leaves"] = []
    v, m, *_ = ev(P, G)
    check("tiny: matched P door without leaves is wrong (n stays 2)",
          close(v["door_dir_acc"], 0.5) and m["door_dir_acc"]["n"] == 2 and close(v["door_dir_recall"], 0.5))

    # gates at their edges: centre distance <= 20 (= 0.5·w_G > τ2)
    for dx, exp in ((19.9, 1), (20.5, 0)):
        P = pred_of(G)
        door(P, 0)["jambs"] = [[50 + dx, 50], [90 + dx, 50]]
        door(P, 0)["leaves"] = []
        _, m, *_ = ev(P, G)
        check(f"tiny: centre offset {dx} px vs gate 20 -> tp(d1)={exp}", m["door_p"]["tp"] == 1 + exp, m["door_p"])
    for ang, exp in ((19.5, 1), (20.5, 0), (180 - 19.5, 1)):
        P = pred_of(G)
        r = math.radians(ang)
        hv = np.array([math.cos(r), math.sin(r)]) * 20
        door(P, 0)["jambs"] = [(np.array([70, 50]) - hv).tolist(), (np.array([70, 50]) + hv).tolist()]
        door(P, 0)["leaves"] = []
        _, m, *_ = ev(P, G)
        check(f"tiny: jamb angle {ang}° vs gate 20° (undirected) -> tp(d1)={exp}", m["door_p"]["tp"] == 1 + exp)
    for half, exp in ((39.4, 1), (40.6, 0), (10.4, 1), (9.6, 0)):
        P = pred_of(G)
        door(P, 0)["jambs"] = [[70 - half, 50], [70 + half, 50]]
        door(P, 0).pop("width_px")
        door(P, 0)["leaves"] = []
        _, m, *_ = ev(P, G)
        check(f"tiny: width {2 * half} vs G 40 (ratio gate 0.5..2) -> tp(d1)={exp}", m["door_p"]["tp"] == 1 + exp)

    # door type: P plain opening matched to a single door = type none (wrong); P unknown = wrong
    P = pred_of(G)
    door(P, 0).update(category="opening", type="none", leaves=[])
    door(P, 1)["type"] = "unknown"
    v, m, *_ = ev(P, G)
    check("tiny: P opening + P unknown -> door_type 0/2, detection still 2/2",
          v["door_type_acc"] == 0.0 and m["door_type_acc"]["n"] == 2 and v["door_f1"] == 1.0
          and m["door_type_acc"]["confusion"] == {"single": {"none": 1, "unknown": 1}}, m["door_type_acc"])

    # swing_into is judged through room MATCHING: P uses its own room ids
    P = pred_of(G)
    P["rooms"][0]["id"], P["rooms"][1]["id"] = "B", "A"      # P "A" == G r2
    door(P, 0)["swing_into"] = "A"
    v, *_ = ev(P, G)
    check("tiny: swing_into via renamed P room ids -> 1.0", v["swing_into_acc"] == 1.0)
    door(P, 0)["swing_into"] = "r2"                          # G's id, not a P room id
    v, *_ = ev(P, G)
    check("tiny: swing_into using G's id instead of the matched P id -> wrong", close(v["swing_into_acc"], 0.5))
    P2 = pred_of(G)
    P2["rooms"] = []
    door(P2, 1)["swing_into"] = "r1"
    v, m, *_ = ev(P2, G)
    check("tiny: no P rooms -> room swing_into wrong; 'outside' vs room wrong", v["swing_into_acc"] == 0.0 and m["swing_into_acc"]["n"] == 2)

    # door <-> window confusion
    P = pred_of(G)
    door(P, 0)["category"] = "window"
    v, m, *_ = ev(P, G)
    check("tiny: d1 predicted as window -> door_r 1/2, door_as_window 1/1, window metrics not scored (G has none)",
          close(v["door_r"], 0.5) and v["door_as_window"] == 1.0 and m["door_as_window"]["n"] == 1
          and v["window_f1"] is None, (v["door_r"], m["door_as_window"], v["window_f1"]))
    # duplicate prediction: one TP, one FP; direction counted once
    P = pred_of(G)
    P["openings"].append(copy.deepcopy(door(P, 0)))
    v, m, *_ = ev(P, G)
    check("tiny: duplicated door -> door_p 2/3, r 1, direction n 2",
          close(v["door_p"], 2 / 3) and v["door_r"] == 1.0 and m["door_dir_acc"]["n"] == 2)

    # walls, exact numbers (G polygon = 200x10 rectangle, P without polygon = flat-capped rectangle)
    P = pred_of(G)
    P["walls"] = [{"id": "p", "centerline": [[0, 50], [100, 50]], "thickness_px": 10, "kind": "exterior"}]
    v, *_ = ev(P, G)
    check("tiny: half-length wall -> IoU exactly 0.5", close(v["wall_iou"], 0.5, 1e-9), v["wall_iou"])
    check("tiny: half-length wall -> CL P=1, R@τ1=(100+5)/200", close(v["wall_cl_p@τ1"], 1.0) and close(v["wall_cl_r@τ1"], 0.525, 1e-9),
          (v["wall_cl_p@τ1"], v["wall_cl_r@τ1"]))
    for dy, iou, p1, p2 in ((3, 7 / 13, 1.0, 1.0), (6, 4 / 16, 0.0, 1.0), (10.5, 0.0, 0.0, 0.0)):
        P["walls"] = [{"id": "p", "centerline": [[0, 50 + dy], [200, 50 + dy]], "thickness_px": 10, "kind": "interior"}]
        v, *_ = ev(P, G)
        check(f"tiny: wall shifted {dy} px -> IoU {iou:.4f}, CL F1@τ1 {p1}, @τ2 {p2}",
              close(v["wall_iou"], iou, 1e-9) and close(v["wall_cl_f1@τ1"], p1) and close(v["wall_cl_f1@τ2"], p2),
              (v["wall_iou"], v["wall_cl_f1@τ1"], v["wall_cl_f1@τ2"]))
    P["walls"] = [{"id": "p", "centerline": [[0, 50.4], [200, 50.4]], "thickness_px": 10, "kind": "interior"}]
    v, *_ = ev(P, G)
    check("tiny: 0.4 px shift is visible in IoU (9.6/10.4: sub-pixel areas)", close(v["wall_iou"], 9.6 / 10.4, 1e-9), v["wall_iou"])
    check("tiny: interior vs exterior -> wall_kind_acc 0", v["wall_kind_acc"] == 0.0)
    P["walls"] = [{"id": "p", "centerline": [[0, 50], [200, 50]], "thickness_px": 10, "kind": "exterior",
                   "polygon": [[0, 50], [200, 50], [200, 50], [0, 50]]}]
    v, *_ = ev(P, G)
    check("tiny: zero-area P wall polygon has zero area", v["wall_iou"] == 0.0 and v["wall_cl_f1@τ1"] == 1.0, v["wall_iou"])
    # a 10 px wall split into 1000 pieces gives the same CL numbers
    P["walls"] = [{"id": f"p{i}", "centerline": [[i * 0.1, 53], [(i + 1) * 0.1, 53]], "thickness_px": 10, "kind": "exterior"}
                  for i in range(1000)]
    v1, *_ = ev(P, G)
    P["walls"] = [{"id": "p", "centerline": [[0, 53], [100, 53]], "thickness_px": 10, "kind": "exterior"}]
    v2, *_ = ev(P, G)
    check("tiny: 1000 sub-pixel pieces == one segment (CL P/R, IoU)",
          all(close(v1[k], v2[k], 2e-3) for k in ("wall_cl_p@τ1", "wall_cl_r@τ1", "wall_iou")),
          {k: (v1[k], v2[k]) for k in ("wall_cl_r@τ1", "wall_iou")})


# =========================================================================== 2. tiny double door
# jambs a=(100,100), b=(180,100); L1 hinge a (h=-1) and L2 hinge b (h=+1), both open downwards (s=+1).
def tiny_double():
    G = {"schema": "floorplan/v0", "image": {"width": 300, "height": 200}, "gt_fidelity": dict(FULL, rooms=False, room_types=False),
         "walls": [{"id": "w1", "centerline": [[0, 100], [300, 100]], "thickness_px": 10, "kind": "interior"}],
         "openings": [{"id": "d1", "category": "door", "type": "double", "jambs": [[100, 100], [180, 100]], "width_px": 80,
                       "leaves": [{"hinge": [100, 100], "closed_tip": [140, 100], "open_tip": [100, 140]},
                                  {"hinge": [180, 100], "closed_tip": [140, 100], "open_tip": [180, 140]}]}]}
    return G


def test_tiny_double():
    G = tiny_double()
    L1 = {"hinge": [100, 100], "closed_tip": [140, 100], "open_tip": [100, 140]}
    L2 = {"hinge": [180, 100], "closed_tip": [140, 100], "open_tip": [180, 140]}
    L2up = {"hinge": [180, 100], "closed_tip": [140, 100], "open_tip": [180, 60]}
    L1b = {"hinge": [104, 100], "closed_tip": [140, 100], "open_tip": [104, 140]}
    cases = [  # leaves, jambs reversed?, expected (swing, dir, leafcnt)
        ("P=G", [L1, L2], False, (1.0, 1.0, 1.0)),
        ("leaves + jambs reversed", [L2, L1], True, (1.0, 1.0, 1.0)),
        ("right leaf swings up", [L1, L2up], False, (0.5, 0.0, 1.0)),
        ("only left leaf", [L1], False, (0.5, 0.0, 0.0)),
        ("two leaves both hinged left", [L1, L1b], False, (0.5, 0.0, 0.0)),
        ("three leaves", [L1, L2, L1b], False, (1.0, 0.0, 0.0)),
        ("no leaves", [], False, (0.0, 0.0, 0.0)),
    ]
    for name, leaves, rev, (sw, dr, lc) in cases:
        P = pred_of(G)
        P["openings"][0]["leaves"] = copy.deepcopy(leaves)
        if rev:
            P["openings"][0]["jambs"] = P["openings"][0]["jambs"][::-1]
        v, m, *_ = ev(P, G)
        ok = close(v["swing_side_acc"], sw) and close(v["door_dir_acc"], dr) and close(v["double_leaf_count_acc"], lc)
        ok = ok and v["hinge_side_acc"] is None and m["swing_side_acc"]["n"] == 1
        check(f"double: {name} -> swing {sw}, dir {dr}, leafcount {lc}, hinge_side n/a", ok,
              (v["swing_side_acc"], v["door_dir_acc"], v["double_leaf_count_acc"], v["hinge_side_acc"]))
    # a G single matched by a P double: P's FIRST leaf is used
    G1 = tiny_double()
    G1["openings"][0].update(type="single", leaves=[L2])
    for first, exp in ((L2, 1.0), (L1, 0.0)):
        P = pred_of(G1)
        P["openings"][0].update(type="double", leaves=[first, L1 if first is L2 else L2])
        v, *_ = ev(P, G1)
        check(f"double: P double vs G single uses P's first leaf -> hinge {exp}", v["hinge_side_acc"] == exp)
    # G sliding (no leaves) with P leaves: not direction-scorable
    G2 = tiny_double()
    G2["openings"][0].update(type="sliding", leaves=[])
    P = pred_of(G)
    v, m, *_ = ev(P, G2)
    check("double: P leaves on a G sliding door -> no direction items, type wrong",
          m["door_dir_acc"]["n"] == 0 and m["door_dir_recall"]["n"] == 0 and v["door_type_acc"] == 0.0)
    # off-image hinge and swing_scorable=false are excluded
    G3 = tiny_double()
    G3["openings"][0]["leaves"][0]["hinge"] = [-3, 100]
    _, m, *_ = ev(pred_of(G3), G3)
    G4 = tiny_double()
    G4["openings"][0]["swing_scorable"] = False
    _, m4, *_ = ev(pred_of(G4), G4)
    check("double: off-image hinge / swing_scorable=false excluded", m["door_dir_recall"]["n"] == 0 and m4["door_dir_recall"]["n"] == 0)


# =========================================================================== 3. rooms
def test_rooms():
    G = tiny_single()
    P = pred_of(G)
    # bow-tie over r2 (y 55..100): crossing at (100, 77.5); two triangles of 0.5*45*100 = 2250 each
    P["rooms"] = [{"id": "x", "polygon": [[0, 55], [200, 100], [200, 55], [0, 100]], "type": "bedroom"}]
    v, m, *_ = ev(P, G)
    check("rooms: self-intersecting bow-tie is repaired, IoU 4500/9000 = 0.5 -> matched", close(v["room_sq"], 0.5, 1e-9) and m["room_f1@0.5"]["tp"] == 1,
          (v["room_sq"], m["room_f1@0.5"]))
    # keyhole polygon: r2 with a 20x20 hole reached through a zero-width slit
    kh = [[0, 55], [200, 55], [200, 100], [100, 100], [100, 90], [110, 90], [110, 70], [90, 70], [90, 90], [100, 90], [100, 100], [0, 100]]
    P["rooms"] = [{"id": "x", "polygon": kh, "type": "bedroom"}, copy.deepcopy(G["rooms"][0])]
    v, *_ = ev(P, G)
    check("rooms: keyhole (hole via slit) -> IoU (9000-400)/9000", close(v["room_sq"], (1 + 8600 / 9000) / 2, 1e-9), v["room_sq"])
    # clockwise winding, duplicated room, unicode labels, capitalised type
    P["rooms"] = [{"id": "ü", "polygon": G["rooms"][1]["polygon"][::-1], "type": "Bedroom", "label": "Schlafzimmer ☃ 卧室"},
                  {"id": "ü2", "polygon": [[0, 55], [200, 55], [200, 99], [0, 99]], "type": "bedroom", "label": "‮\u0000"},
                  {"id": "k", "polygon": G["rooms"][0]["polygon"], "type": "kitchen", "label": "Küche"}]
    v, m, *_ = ev(P, G)
    check("rooms: CW winding ok; duplicate = FP; 'Bedroom' (not enum) wrong; unicode fine",
          close(v["room_p@0.5"], 2 / 3) and v["room_r@0.5"] == 1.0 and close(v["room_type_acc"], 0.5), (v["room_p@0.5"], v["room_type_acc"]))
    # IoU just below / at the gate
    P["rooms"] = [{"id": "a", "polygon": [[0, 55], [200, 55], [200, 77.4], [0, 77.4]], "type": "x"},
                  {"id": "b", "polygon": [[0, 0], [200, 0], [200, 22.6], [0, 22.6]], "type": "x"}]
    v, m, *_ = ev(P, G)
    check("rooms: IoU 22.4/45 < 0.5 unmatched, 22.6/45 >= 0.5 matched", m["room_f1@0.5"]["tp"] == 1, m["room_f1@0.5"])


# =========================================================================== 4. garbage predictions
def test_garbage():
    G = tiny_single()
    nan, inf = float("nan"), float("inf")
    garbage = [
        None, [], "str", 3, {"walls": "x", "openings": {"a": 1}, "rooms": 5},
        {"walls": [None, 1, {"centerline": [[nan, 0], [1, 1]]}, {"centerline": [[0, 0]]}, {"centerline": [[0, 0], [1e308, -1e308]], "thickness_px": inf},
                   {"centerline": [[True, False], [1, 1]]}, {"centerline": [["1", "2"], [3, 4]], "thickness_px": "10"},
                   {"centerline": [[0, 50], [200, 50]], "thickness_px": -5, "polygon": [[nan, 1], [2, 3], [4, 5]]},
                   {"centerline": [[0, 50], [200, 50]], "thickness_px": 1e300, "polygon": [[0, 0], [1e300, 0], [1e300, 1e300]]}],
         "openings": [None, {"category": "door"}, {"category": "door", "jambs": [[50, 50], [50, 50]], "leaves": []},
                      {"category": "door", "jambs": [[50, 50], [90, 50]], "width_px": nan, "leaves": [None, {"hinge": [nan, 1]}, {"hinge": [50, 50]}, 7],
                       "type": ["single"], "swing_into": {"r": 1}},
                      {"category": "window", "jambs": [[inf, 0], [1, 1]]}, {"category": "DOOR", "jambs": [[50, 50], [90, 50]]}],
         "rooms": [None, {"polygon": [[0, 0], [1, 1]]}, {"polygon": [[0, 0], [1, 1], [2, 2]], "type": 5},
                   {"polygon": [[0, 0], [nan, 1], [2, 2]]}, {"polygon": [[0, 0], [10, 10], [10, 0], [0, 10], [5, 5]], "id": ["x"]}]},
    ]
    ok, detail = True, []
    for i, P in enumerate(garbage):
        try:
            st = {"has_pred": True, "pred_valid": True, "pred_errors": []}
            c, info = fe.score_sample("x", P, G, st)
            m = fe.finalize(c)
            json.dumps(fe._clean(m), allow_nan=False)
            if "eval_error" in st:
                detail.append((i, st["eval_error"]))
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail.append((i, repr(exc)))
    check("garbage: no crash, no NaN/inf in the report, no eval_error fallback needed", ok and not detail, detail)
    # the jambs-ok garbage door IS matched (1 of 2) and its broken leaves make direction wrong, not crash
    c, _ = fe.evaluate_pair(garbage[-1], G)
    m = fe.finalize(c)
    check("garbage: valid jambs with broken leaves -> matched, direction wrong", m["door_p"]["tp"] == 1 and m["door_dir_acc"]["value"] == 0.0,
          (m["door_p"], m["door_dir_acc"]))
    # an evaluator failure in one group only drops that group of the prediction
    orig = fe.eval_walls

    def broken(c, Gw, Pw, *a):
        if Pw:
            raise RuntimeError("boom")
        return orig(c, Gw, Pw, *a)
    fe.eval_walls = broken
    try:
        st = {"has_pred": True, "pred_valid": True, "pred_errors": []}
        c, _ = fe.score_sample("x", pred_of(G), G, st)
        m = fe.finalize(c)
    finally:
        fe.eval_walls = orig
    check("garbage: a crash in wall scoring keeps doors/rooms scored, walls scored as empty, flagged",
          m["door_f1"]["value"] == 1.0 and m["room_f1@0.5"]["value"] == 1.0 and m["wall_cl_r@τ1"]["value"] == 0.0
          and "walls dropped" in st.get("eval_error", "") and not st["pred_valid"], st.get("eval_error"))
    v = fe.validate_pred([], G) + fe.validate_pred({"image": {"width": 1, "height": 1}, "walls": [1]}, G)
    check("garbage: validate_pred never raises and flags problems", len(v) >= 2)


# =========================================================================== 5. real-data oracle
def reflect_across(p, origin, axis):
    """Reflect point p across the line through origin with unit direction ⟂ axis (i.e. flip the axis component)."""
    p, origin = np.asarray(p, float), np.asarray(origin, float)
    return (p - 2 * ((p - origin) @ axis) * axis).tolist()


def my_scorable(o, w, h):
    lv = o.get("leaves") or []
    return (o.get("category") == "door" and len(lv) > 0 and o.get("swing_scorable") is not False
            and all(0 <= l["hinge"][0] <= w and 0 <= l["hinge"][1] <= h for l in lv))


def perturb(sid, rng):
    """Random recorded perturbation of all doors (+ room ids). Returns (P, expected counts)."""
    G = GTS[sid]
    w, h = G["image"]["width"], G["image"]["height"]
    fid = G["gt_fidelity"]
    P = pred_of(G)
    rid = {r["id"]: f"P{k}_{r['id']}" for k, r in enumerate(P.get("rooms", []))}
    for r in P.get("rooms", []):
        r["id"] = rid[r["id"]]
    e = dict(n=0, hinge_n=0, hinge_ok=0, swing_n=0, swing_sum=0.0, dir_n=0, dir_ok=0, lc_n=0, lc_ok=0,
             si_n=0, si_ok=0, tp=0, n_pred=0, ng=0)
    keep = []
    for o in P["openings"]:
        if o["category"] == "window":
            keep.append(o)
            continue
        e["ng"] += 1
        go = copy.deepcopy(o)
        if o.get("swing_into") not in (None, "outside"):
            o["swing_into"] = rid.get(o["swing_into"], o["swing_into"])
        a, b = np.array(o["jambs"], float)
        L = np.hypot(*(b - a))
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        mid = (a + b) / 2
        sc = fid["door_swing"] == "full" and my_scorable(go, w, h)
        act = "keep"
        if sc:
            nl = len(o["leaves"])
            acts = ["keep", "delete", "clear", "hswap", "sflip", "both"] if nl == 1 else ["keep", "delete", "clear", "sflip1"]
            act = rng.choice(acts)
            e["n"] += 1
        if act == "delete":
            continue
        if act == "clear":
            o["leaves"] = []
        if act in ("hswap", "both"):   # mirror the leaf across the jamb bisector: hinge side flips, swing side kept
            for lf in o["leaves"]:
                for k in ("hinge", "closed_tip", "open_tip"):
                    lf[k] = reflect_across(lf[k], mid, u)
        if act in ("sflip", "both", "sflip1"):  # mirror open_tip across the line through the hinge along u
            lf = o["leaves"][rng.randrange(len(o["leaves"]))]
            lf["open_tip"] = reflect_across(lf["open_tip"], lf["hinge"], n)
        if rng.random() < 0.5:
            o["jambs"] = o["jambs"][::-1]
        if len(o.get("leaves", [])) > 1 and rng.random() < 0.5:
            o["leaves"] = o["leaves"][::-1]
        d = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])  # 1 px jitter of the whole door
        o["jambs"] = [(np.array(p) + d).tolist() for p in o["jambs"]]
        for lf in o.get("leaves", []):
            for k in ("hinge", "closed_tip", "open_tip"):
                lf[k] = (np.array(lf[k]) + d).tolist()
        si_wrong = False
        if sc and go.get("swing_into") is not None and rng.random() < 0.25:
            si_wrong = True
            o["swing_into"] = "outside" if go["swing_into"] != "outside" else next(iter(rid.values()), "nowhere")
        keep.append(o)
        e["tp"] += 1
        e["n_pred"] += 1
        if not sc:
            continue
        if len(go["leaves"]) == 1:
            e["hinge_n"] += 1
            e["swing_n"] += 1
            e["dir_n"] += 1
            e["hinge_ok"] += act in ("keep", "sflip")
            e["swing_sum"] += act in ("keep", "hswap")
            e["dir_ok"] += act == "keep"
        else:
            e["swing_n"] += 1
            e["dir_n"] += 1
            e["lc_n"] += 1
            e["swing_sum"] += {"keep": 1.0, "sflip1": 1 - 1 / len(go["leaves"]), "clear": 0.0}[act]
            e["lc_ok"] += act in ("keep", "sflip1")
            e["dir_ok"] += act == "keep"
        if go.get("swing_into") is not None:
            e["si_n"] += 1
            e["si_ok"] += not si_wrong
    P["openings"] = keep
    return P, e


def test_oracle():
    seeds = [1] if QUICK else [1, 2, 3]
    for seed in seeds:
        rng = random.Random(seed)
        per_ds, overall_c = {}, fe.new_counters()
        exp_ds = {}
        for sid in IDS:
            if not GTS[sid].get("openings"):
                continue
            ds = MANIFEST[sid]["dataset"]
            P, e = perturb(sid, rng)
            c, _ = fe.evaluate_pair(P, GTS[sid])
            fe.add_counters(per_ds.setdefault(ds, fe.new_counters()), c)
            fe.add_counters(overall_c, c)
            acc = exp_ds.setdefault(ds, {k: 0 for k in e})
            for k in e:
                acc[k] += e[k]
        tot = {k: sum(x[k] for x in exp_ds.values()) for k in next(iter(exp_ds.values()))}
        bad = []
        for ds, e in list(exp_ds.items()) + [("overall", tot)]:
            m = fe.finalize(per_ds[ds] if ds != "overall" else overall_c)
            want = {"door_r": (e["tp"] / e["ng"], e["ng"]), "door_p": (e["tp"] / e["n_pred"], e["n_pred"]),
                    "hinge_side_acc": (e["hinge_ok"] / e["hinge_n"] if e["hinge_n"] else None, e["hinge_n"]),
                    "swing_side_acc": (e["swing_sum"] / e["swing_n"] if e["swing_n"] else None, e["swing_n"]),
                    "door_dir_acc": (e["dir_ok"] / e["dir_n"] if e["dir_n"] else None, e["dir_n"]),
                    "door_dir_recall": (e["dir_ok"] / e["n"] if e["n"] else None, e["n"]),
                    "double_leaf_count_acc": (e["lc_ok"] / e["lc_n"] if e["lc_n"] else None, e["lc_n"]),
                    "swing_into_acc": (e["si_ok"] / e["si_n"] if e["si_n"] else None, e["si_n"])}
            for k, (val, n) in want.items():
                got = m[k]
                if got["n"] != n or not ((val is None and got["value"] is None) or close(got["value"], val, 1e-9)):
                    bad.append((ds, k, "want", round(val, 6) if val is not None else None, n, "got", got["value"], got["n"]))
        check(f"oracle seed {seed}: per-dataset + overall door/direction/swing_into counts match the recorded perturbations",
              not bad, bad[:6] or f"overall n_dir={tot['n']} dir_ok={tot['dir_ok']} swing_into n={tot['si_n']}")


# =========================================================================== 6. invariance
def transform_doc(doc, f, dims):
    d = copy.deepcopy(doc)
    d["image"]["width"], d["image"]["height"] = dims
    for wl in d.get("walls", []):
        wl["centerline"] = [f(p) for p in wl["centerline"]]
        if wl.get("polygon"):
            wl["polygon"] = [f(p) for p in wl["polygon"]]
    for o in d.get("openings", []):
        o["jambs"] = [f(p) for p in o["jambs"]]
        for key in ("leaves", "alt_leaves"):
            for lf in o.get(key) or []:
                for k in ("hinge", "closed_tip", "open_tip"):
                    lf[k] = f(lf[k])
    for r in d.get("rooms", []):
        r["polygon"] = [f(p) for p in r["polygon"]]
    return d


def test_invariance():
    rng = random.Random(7)
    sids = [s for s in IDS if GTS[s].get("walls") or GTS[s].get("openings")]
    if QUICK:
        sids = sids[::4]
    worst = 0.0
    where = None
    for sid in sids:
        G = GTS[sid]
        W, H = G["image"]["width"], G["image"]["height"]
        P, _ = perturb(sid, rng)
        for wl in P.get("walls", []):  # walls: random small shifts so wall metrics are not trivially 1
            dy = rng.uniform(-1, 1) * 8
            wl["centerline"] = [[x, y + dy] for x, y in wl["centerline"]]
            wl.pop("polygon", None)
        base = fe.finalize(fe.evaluate_pair(P, G)[0])
        for name, f, dims in (("mirror-x", lambda p: [W - p[0], p[1]], (W, H)),
                              ("mirror-y", lambda p: [p[0], H - p[1]], (W, H)),
                              ("rot90", lambda p: [H - p[1], p[0]], (H, W))):
            m = fe.finalize(fe.evaluate_pair(transform_doc(P, f, dims), transform_doc(G, f, dims))[0])
            for k in base:
                a, b = base[k]["value"], m[k]["value"]
                if (a is None) != (b is None) or base[k]["n"] != m[k]["n"]:
                    worst, where = 1.0, (sid, name, k, a, b)
                elif a is not None:
                    tol = 2e-3 if k.startswith("wall_cl") or k == "wall_kind_acc" else 1e-6
                    if abs(a - b) > tol and abs(a - b) > worst:
                        worst, where = abs(a - b), (sid, name, k, a, b)
    check("invariance: mirror-x / mirror-y / rot90 of G and P together leave every metric unchanged", worst == 0.0, where)


# =========================================================================== 7. wall oracle
def test_wall_oracle():
    rng = random.Random(11)
    picks = {}
    for sid in IDS:
        ds = MANIFEST[sid]["dataset"]
        if GTS[sid].get("walls") and ds not in picks:
            picks[ds] = sid
    worst = {}
    for ds, sid in picks.items():
        G = GTS[sid]
        W, H = G["image"]["width"], G["image"]["height"]
        tau = fe.tolerances(G)
        th = sorted(w["thickness_px"] for w in G["walls"])
        t_med = float(np.median(th))
        check(f"wall oracle {ds}: τ from the median GT thickness", close(tau["t"], t_med) and close(tau["tau1"], max(2, t_med / 2))
              and close(tau["tau2"], max(4, t_med)), (tau, t_med))
        P = pred_of(G)
        walls = []
        for wl in P["walls"]:
            if rng.random() < 0.15:
                continue
            a, b = np.array(wl["centerline"], float)
            L = np.hypot(*(b - a))
            if L < 1e-6:
                continue
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / L
            off = rng.choice([0, 0.3, 0.7, 1.5]) * t_med * rng.choice([-1, 1])
            walls.append({"id": wl["id"], "centerline": [(a + off * nrm).tolist(), (b + off * nrm).tolist()],
                          "thickness_px": wl["thickness_px"] * rng.uniform(0.7, 1.3), "kind": wl.get("kind", "unknown")})
        walls.append({"id": "far", "centerline": [[-5000, -5000], [-4000, 9000]], "thickness_px": 10})  # off-image
        P["walls"] = walls
        v, m, *_ = ev(P, G)
        # exact areas
        gpoly = shapely.union_all([fe._valid_polygon(np.array(w["polygon"], float)) for w in G["walls"]]).intersection(box(0, 0, W, H))
        ppoly = shapely.union_all([LineString(w["centerline"]).buffer(w["thickness_px"] / 2, cap_style="flat") for w in walls]).intersection(box(0, 0, W, H))
        iou = gpoly.intersection(ppoly).area / gpoly.union(ppoly).area
        # pixel-centre raster at 4x supersampling (independent of shapely areas)
        S = 4

        def rast(geom):
            mk = np.zeros((H * S, W * S), np.uint8)
            for g in getattr(geom, "geoms", [geom]):
                if g.geom_type != "Polygon" or g.is_empty:
                    continue
                ext = np.round((np.array(g.exterior.coords) * S - 0.5) * 16).astype(np.int32)
                tmp = np.zeros_like(mk)
                import cv2
                cv2.fillPoly(tmp, [ext], 1, shift=4)
                for hole in g.interiors:
                    cv2.fillPoly(tmp, [np.round((np.array(hole.coords) * S - 0.5) * 16).astype(np.int32)], 0, shift=4)
                mk |= tmp
            return mk
        if not QUICK:
            mg, mp = rast(gpoly), rast(ppoly)
            iou_r = np.count_nonzero(mg & mp) / np.count_nonzero(mg | mp)
        else:
            iou_r = iou
        # centerline P/R, continuous
        gl = shapely.union_all([LineString(w["centerline"]) for w in G["walls"]])
        plines = [LineString(w["centerline"]) for w in walls]
        res = {}
        for key, tv in (("τ1", tau["tau1"]), ("τ2", tau["tau2"])):
            gb = gl.buffer(tv, quad_segs=64)
            pb = shapely.union_all(plines).buffer(tv, quad_segs=64)
            p_in = sum(pl.intersection(gb).length for pl in plines) / sum(pl.length for pl in plines)
            r_in = sum(LineString(w["centerline"]).intersection(pb).length for w in G["walls"]) / sum(LineString(w["centerline"]).length for w in G["walls"])
            res[key] = (p_in, r_in)
        d = {"iou_exact": abs(v["wall_iou"] - iou), "iou_raster4x": abs(v["wall_iou"] - iou_r)}
        for key in ("τ1", "τ2"):
            d[f"p{key}"] = abs(v[f"wall_cl_p@{key}"] - res[key][0])
            d[f"r{key}"] = abs(v[f"wall_cl_r@{key}"] - res[key][1])
        worst[ds] = d
        ok = d["iou_exact"] < 1e-6 and d["iou_raster4x"] < 0.01 and all(d[k] < 0.005 for k in d if k[0] in "pr")
        check(f"wall oracle {ds} ({sid}): IoU = exact shapely area and ~ 4x pixel-centre raster; CL P/R = continuous shapely length",
              ok, {k: round(x, 5) for k, x in d.items()} | {"iou": round(v["wall_iou"], 4)})


# =========================================================================== 8. split / dataset attribution via CLI
def test_cli():
    with tempfile.TemporaryDirectory() as td:
        pd = os.path.join(td, "pred")
        os.makedirs(pd)
        fe.pred_from_gt(pd, "all", TESTDATA)
        # corrupt some files in different ways
        bad = {"synthetic_01": b"{not json", "synthetic_02": b"[1,2,3]", "realworld_09": "\"✓ unicode string\"".encode(),
               "synthetic_04": b"\xff\xfe\x00garbage", "realworld_11": b"{}"}
        for k, raw in bad.items():
            with open(os.path.join(pd, k + ".json"), "wb") as f:
                f.write(raw)
        os.remove(os.path.join(pd, "synthetic_07.json"))
        # nested-layout prediction overrides nothing (flat file wins) but must not crash
        for split in ("dev", "test"):
            out = os.path.join(td, f"r_{split}.json")
            p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "fp_eval.py"), "--pred-dir", pd, "--split", split,
                                "--out", out, "--testdata", TESTDATA], capture_output=True, text=True, timeout=900)
            if p.returncode != 0:
                check(f"cli {split}: exit 0", False, p.stderr[-400:])
                continue
            rep = json.load(open(out, encoding="utf-8"))
            ids = json.load(open(os.path.join(TESTDATA, "splits.json")))[split]
            per = {s["id"]: s for s in rep["per_sample"]}
            want_ds = {}
            for sid in ids:
                want_ds[MANIFEST[sid]["dataset"]] = want_ds.get(MANIFEST[sid]["dataset"], 0) + 1
            got_ds = {d: b["robustness"]["n_samples"] for d, b in rep["per_dataset"].items()}
            attr_ok = got_ds == want_ds and all(per[s]["dataset"] == MANIFEST[s]["dataset"] for s in ids) and set(per) == set(ids)
            check(f"cli {split}: per-dataset sample attribution matches manifest", attr_ok, (got_ds, want_ds))
            exp_bad = {s for s in ids if os.path.basename(s) in bad}
            exp_miss = {s for s in ids if os.path.basename(s) == "synthetic_07"}
            got_inv = {s for s, x in per.items() if x["has_pred"] and not x["pred_valid"]}
            got_miss = {s for s, x in per.items() if not x["has_pred"]}
            check(f"cli {split}: corrupt files flagged invalid, missing flagged missing, others valid",
                  got_inv == exp_bad and got_miss == exp_miss, (sorted(got_inv), sorted(got_miss)))
            # micro-average: overall counts are the sums of the per-sample counts
            ov = rep["overall"]["metrics"]
            tp = sum(x["metrics"]["door_f1"].get("tp", 0) for x in per.values())
            ng = sum(x["metrics"]["door_f1"]["n"] for x in per.values())
            npred = sum(x["metrics"]["door_f1"].get("n_pred", 0) for x in per.values())
            f1 = 2 * tp / (ng + npred) if ng + npred else None
            rd = sum(x["metrics"]["door_dir_recall"]["n"] for x in per.values())
            check(f"cli {split}: overall door F1 / dir-recall n are micro sums of per-sample counts",
                  close(ov["door_f1"]["value"], f1, 1e-6) and ov["door_dir_recall"]["n"] == rd, (ov["door_f1"], f1))
            # every sample whose prediction is fine must be perfect; corrupt ones must be 0 where scored
            imperfect = [s for s, x in per.items() if s not in exp_bad | exp_miss and any(
                m["value"] not in (None, 1.0) for k, m in x["metrics"].items() if k not in fe.DIAGNOSTIC)]
            check(f"cli {split}: untouched P=G samples still perfect", not imperfect, imperfect[:5])
            for sid in ids:
                tau = per[sid]["tau"]
                G = GTS[sid]
                th = [w["thickness_px"] for w in G.get("walls", [])]
                t = float(np.median(th)) if th else 0.006 * math.hypot(G["image"]["width"], G["image"]["height"])
                if not close(tau["t"], round(t, 3), 1e-3):
                    check(f"cli {split}: τ for {sid}", False, (tau, t))
                    break
            else:
                check(f"cli {split}: per-sample τ = median GT thickness (or 0.006·diag)", True)


def test_locale():
    g = os.path.join(TESTDATA, MANIFEST["synthetic/synthetic_01"]["gt_file"])
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, LC_ALL="en_US.ISO8859-1", LANG="en_US.ISO8859-1")
        p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "fp_eval.py"), "--pred", g, "--gt", g,
                            "--out", os.path.join(td, "r.json"), "--md", os.path.join(td, "r.md")],
                           capture_output=True, env=env, timeout=300)
        ok = p.returncode == 0 and "τ" in open(os.path.join(td, "r.json"), encoding="utf-8").read()
    check("cli: latin-1 locale does not crash on 'τ' (files are UTF-8)", ok, p.stderr[-300:])


# =========================================================================== 9. runtime
def test_runtime():
    G = GTS[max(GTS, key=lambda s: len(GTS[s].get("walls", [])))]           # the largest plan of the set
    W, H = G["image"]["width"], G["image"]["height"]
    rs = np.random.default_rng(3)
    P = {"schema": "floorplan/v0", "image": {"width": W, "height": H},
         "walls": [{"id": f"w{i}", "centerline": (c + rs.uniform(-60, 60, (2, 2))).tolist(), "thickness_px": 12, "kind": "interior"}
                   for i, c in enumerate(rs.uniform(0, W, (10000, 1, 2)))],
         "openings": [{"id": f"d{i}", "category": "door", "type": "single", "jambs": rs.uniform(0, W, (2, 2)).tolist(),
                       "leaves": [{"hinge": [1, 1], "closed_tip": [2, 2], "open_tip": [3, 1]}]} for i in range(10000)],
         "rooms": [{"id": f"r{i}", "polygon": (rs.uniform(0, W, 2) + rs.uniform(-80, 80, (5, 2))).tolist(), "type": "other"} for i in range(10000)]}
    t = time.time()
    c, _ = fe.evaluate_pair(P, G)
    dt = time.time() - t
    check("runtime: 10k walls + 10k doors + 10k rooms on a 2400 px sample < 30 s", dt < 30, f"{dt:.1f} s")
    P2 = {"walls": [{"id": f"w{i}", "centerline": [[-1e7, i], [1e7, i + 3]], "thickness_px": 1e6} for i in range(2000)]}
    t = time.time()
    v, *_ = ev(P2, G)
    dt = time.time() - t
    check("runtime: 2000 walls 2e7 px long with 1e6 px thickness < 20 s, P ~ 0", dt < 20 and v["wall_cl_p@τ2"] < 1e-3, f"{dt:.1f} s, p={v['wall_cl_p@τ2']}")


def main():
    t0 = time.time()
    for fn in (test_tiny_single, test_tiny_double, test_rooms, test_garbage, test_oracle, test_invariance,
               test_wall_oracle, test_cli, test_locale, test_runtime):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised", False, repr(exc))
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
