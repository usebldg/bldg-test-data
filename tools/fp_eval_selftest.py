"""Self-test for tools/fp_eval.py: P = G must be perfect, and controlled perturbations must move
exactly the metrics they should.

  .venv-eval/bin/python -I tools/fp_eval_selftest.py [--testdata DIR]

Exit code 0 when every check passes.
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import fp_eval as fe  # noqa: E402

TESTDATA = ROOT
if "--testdata" in sys.argv:
    TESTDATA = sys.argv[sys.argv.index("--testdata") + 1]

MANIFEST = fe.load_manifest(TESTDATA)
IDS = sorted(MANIFEST)
GTS = {sid: json.load(open(os.path.join(TESTDATA, MANIFEST[sid]["gt_file"]))) for sid in IDS}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def run(make_pred, ids=IDS):
    """make_pred(sid, gt_copy) -> prediction (dict or None). Returns (overall metrics, {sid: metrics})."""
    acc = fe.new_counters()
    per = {}
    for sid in ids:
        pred = make_pred(sid, copy.deepcopy(GTS[sid]))
        c, _ = fe.evaluate_pair(pred, GTS[sid])
        fe.add_counters(acc, c)
        per[sid] = fe.finalize(c)
    return fe.finalize(acc), per


def v(m, k):
    return m[k]["value"]


def identity(sid, p):
    p.pop("gt_fidelity", None)
    return p


# ---------------------------------------------------------------------- door selection helpers

def scorable(sid):
    """[(opening index in GT 'openings', n_leaves)] of direction-scorable G doors in this sample."""
    gt = GTS[sid]
    if gt["gt_fidelity"].get("door_swing") != "full":
        return []
    w, h = fe.image_size(gt)
    out = []
    for i, o in enumerate(gt["openings"]):
        if o.get("category") != "door":
            continue
        po = fe.parse_openings({"openings": [o]})["door"]
        if po and fe.scorable_direction(po[0], w, h):
            out.append((i, len(po[0]["leaves"])))
    return out


ALL_SCORABLE = [(sid, i, nl) for sid in IDS for i, nl in scorable(sid)]
SINGLES = [(sid, i) for sid, i, nl in ALL_SCORABLE if nl == 1]
PICK = SINGLES[::max(1, len(SINGLES) // 7)][:7]  # k = 7 single doors spread over the datasets
K = len(PICK)
PICK_BY = {}
for sid, i in PICK:
    PICK_BY.setdefault(sid, set()).add(i)


def mirror_leaf(lf):
    """Reflect open_tip across the wall line through the hinge (direction hinge -> closed_tip)."""
    hx, hy = lf["hinge"]
    cx, cy = lf["closed_tip"]
    ox, oy = lf["open_tip"]
    ux, uy = cx - hx, cy - hy
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    nx, ny = -uy, ux
    d = (ox - hx) * nx + (oy - hy) * ny
    lf["open_tip"] = [ox - 2 * d * nx, oy - 2 * d * ny]


def shift_doc(p, dx, dy):
    sh = lambda q: [q[0] + dx, q[1] + dy]
    for w in p.get("walls", []):
        w["centerline"] = [sh(q) for q in w["centerline"]]
        if w.get("polygon"):
            w["polygon"] = [sh(q) for q in w["polygon"]]
    for o in p.get("openings", []):
        o["jambs"] = [sh(q) for q in o["jambs"]]
        if o.get("polygon"):
            o["polygon"] = [sh(q) for q in o["polygon"]]
        for lf in o.get("leaves", []) + (o.get("alt_leaves") or []):
            for k in ("hinge", "closed_tip", "open_tip"):
                lf[k] = sh(lf[k])
    for r in p.get("rooms", []):
        r["polygon"] = [sh(q) for q in r["polygon"]]
    return p


def tval(sid):
    return fe.tolerances(GTS[sid])["t"]


def split_walls(p, which="longest"):
    walls = p.get("walls", [])
    if not walls:
        return p
    lens = [math.dist(*w["centerline"]) for w in walls]
    targets = [max(range(len(walls)), key=lens.__getitem__)] if which == "longest" else list(range(len(walls)))
    out = []
    for i, w in enumerate(walls):
        if i not in targets:
            out.append(w)
            continue
        (ax, ay), (bx, by) = w["centerline"]
        cuts = [0.0, 0.3, 0.55, 1.0]
        for j in range(3):
            q = copy.deepcopy(w)
            f0, f1 = cuts[j], cuts[j + 1]
            q["id"] = f"{w['id']}_{j}"
            q["centerline"] = [[ax + f0 * (bx - ax), ay + f0 * (by - ay)], [ax + f1 * (bx - ax), ay + f1 * (by - ay)]]
            q.pop("polygon", None)  # pieces fall back to buffered centerlines (only the centerline metric is checked)
            out.append(q)
    p["walls"] = out
    return p


# ------------------------------------------------------------------------------------------ tests

def test_identity():
    ov, per = run(identity)
    bad = []
    for scope, m in [("overall", ov)] + list(per.items()):
        for k, x in m.items():
            if x["n"] == 0 or x["value"] is None:
                continue
            want = 0.0 if k in fe.DIAGNOSTIC else 1.0
            if not close(x["value"], want):
                bad.append(f"{scope}:{k}={x['value']}")
    check("P=G: every applicable metric is perfect (1.0; diagnostics 0) per sample and overall", not bad, "; ".join(bad[:5]))
    need = ["wall_iou", "wall_cl_f1@τ1", "wall_cl_f1@τ2", "wall_kind_acc", "door_p", "door_r", "door_f1",
            "window_f1", "door_type_acc", "hinge_side_acc", "swing_side_acc", "door_dir_acc", "door_dir_recall",
            "double_leaf_count_acc", "swing_into_acc", "room_f1@0.5", "room_sq", "room_pq", "room_type_acc"]
    missing = [k for k in need if ov[k]["n"] == 0]
    check("P=G: all primary metrics are applicable somewhere (n > 0)", not missing, str(missing))
    n_doors = sum(o.get("category") == "door" for g in GTS.values() for o in g.get("openings", []))
    n_into = sum(o.get("category") == "door" and o.get("swing_into") is not None
                 for g in GTS.values() for o in g.get("openings", []))
    check(f"P=G: scorable doors exclude swing_scorable=false and off-image hinges ({len(ALL_SCORABLE)} of {n_doors})",
          ov["door_dir_recall"]["n"] == len(ALL_SCORABLE) < n_doors, f"n={ov['door_dir_recall']['n']}")
    check(f"P=G: swing_into n = {n_into} (G swing_into non-null)", ov["swing_into_acc"]["n"] == n_into,
          str(ov["swing_into_acc"]["n"]))
    return ov


def test_a(base):
    def make(sid, p):
        for i in PICK_BY.get(sid, ()):
            lf = p["openings"][i]["leaves"][0]
            lf["hinge"], lf["closed_tip"] = lf["closed_tip"], lf["hinge"]
        return identity(sid, p)
    ov, _ = run(make)
    n = ov["hinge_side_acc"]["n"]
    check(f"(a) swap hinge/closed_tip on k={K} doors: hinge_side_acc = 1 - k/n exactly",
          close(v(ov, "hinge_side_acc"), 1 - K / n), f"{v(ov, 'hinge_side_acc'):.6f} vs {1 - K / n:.6f}, n={n}")
    check("(a) swing_side_acc unchanged (1.0)", close(v(ov, "swing_side_acc"), 1.0), f"{v(ov, 'swing_side_acc')}")
    check("(a) door_dir_acc drops by k/n_dir", close(v(ov, "door_dir_acc"), 1 - K / ov["door_dir_acc"]["n"]))
    check("(a) detection unchanged", close(v(ov, "door_f1"), 1.0))


def test_b():
    def make(sid, p):
        for i in PICK_BY.get(sid, ()):
            for lf in p["openings"][i]["leaves"]:
                mirror_leaf(lf)
        return identity(sid, p)
    ov, _ = run(make)
    n = ov["swing_side_acc"]["n"]
    check(f"(b) mirror open_tip across the wall on k={K} doors: swing_side_acc = 1 - k/n",
          close(v(ov, "swing_side_acc"), 1 - K / n), f"{v(ov, 'swing_side_acc'):.6f}, n={n}")
    check("(b) hinge_side_acc unchanged (1.0)", close(v(ov, "hinge_side_acc"), 1.0))

    allsc = {}
    for sid, i, _ in ALL_SCORABLE:
        allsc.setdefault(sid, set()).add(i)

    def make_all(sid, p):
        for i in allsc.get(sid, ()):
            for lf in p["openings"][i]["leaves"]:
                mirror_leaf(lf)
        return identity(sid, p)
    ov2, _ = run(make_all)
    check("(b) mirror every scorable door: swing_side_acc = 0, door_dir_acc = 0, hinge 1.0, leaf count 1.0",
          close(v(ov2, "swing_side_acc"), 0.0) and close(v(ov2, "door_dir_acc"), 0.0)
          and close(v(ov2, "hinge_side_acc"), 1.0) and close(v(ov2, "double_leaf_count_acc"), 1.0),
          f"swing={v(ov2, 'swing_side_acc')}, dir={v(ov2, 'door_dir_acc')}")


def test_c():
    def make(sid, p):
        drop = PICK_BY.get(sid, set())
        p["openings"] = [o for i, o in enumerate(p["openings"]) if i not in drop]
        return identity(sid, p)
    ov, _ = run(make)
    nd, nr = ov["door_r"]["n"], ov["door_dir_recall"]["n"]
    check(f"(c) delete k={K} doors: door_r = 1 - k/n_doors", close(v(ov, "door_r"), 1 - K / nd), f"{v(ov, 'door_r'):.6f}")
    check("(c) door_dir_recall = 1 - k/n_scorable", close(v(ov, "door_dir_recall"), 1 - K / nr), f"{v(ov, 'door_dir_recall'):.6f}")
    check("(c) door_p, hinge_side_acc, swing_side_acc stay 1.0 (matched denominators)",
          all(close(v(ov, k), 1.0) for k in ("door_p", "hinge_side_acc", "swing_side_acc")))


def test_d():
    def mk(f):
        def make(sid, p):
            s = f * tval(sid) / math.sqrt(2)
            return identity(sid, shift_doc(p, s, s))
        return make
    ov04, per04 = run(mk(0.4))
    ov3, _ = run(mk(3.0))
    worst = min(m["wall_cl_f1@τ1"]["value"] for m in per04.values() if m["wall_cl_f1@τ1"]["value"] is not None)
    check("(d) shift 0.4·t: centerline F1@τ1 stays high (overall >= 0.99, worst sample >= 0.99)",
          v(ov04, "wall_cl_f1@τ1") >= 0.99 and worst >= 0.99, f"overall={v(ov04, 'wall_cl_f1@τ1'):.4f}, worst={worst:.4f}")
    check("(d) shift 3·t: centerline F1@τ1 drops (< 0.6) and F1@τ2 drops (< 0.7)",
          v(ov3, "wall_cl_f1@τ1") < 0.6 and v(ov3, "wall_cl_f1@τ2") < 0.7,
          f"F1@τ1={v(ov3, 'wall_cl_f1@τ1'):.4f}, F1@τ2={v(ov3, 'wall_cl_f1@τ2'):.4f}")
    check("(d) shift 3·t: wall IoU drops below the 0.4·t shift",
          v(ov3, "wall_iou") < v(ov04, "wall_iou") - 0.2, f"{v(ov04, 'wall_iou'):.3f} -> {v(ov3, 'wall_iou'):.3f}")


def test_e():
    def make(sid, p):
        for o in p["openings"]:
            if o["category"] in ("door", "opening"):
                o["category"], o["type"], o["leaves"] = "window", "unknown", []
        return identity(sid, p)
    ov, _ = run(make)
    nd = ov["door_r"]["n"]
    check("(e) relabel doors as windows: door_r = 0", close(v(ov, "door_r"), 0.0))
    check("(e) door_as_window confusion = 1.0 over all G doors",
          close(v(ov, "door_as_window"), 1.0) and ov["door_as_window"]["n"] == nd,
          f"{v(ov, 'door_as_window')} (n={ov['door_as_window']['n']}, G doors {nd})")
    nw = ov["window_r"]["n"]
    check("(e) window precision drops to n_w/(n_w+n_d), window recall stays 1.0",
          close(v(ov, "window_p"), nw / (nw + nd)) and close(v(ov, "window_r"), 1.0), f"{v(ov, 'window_p'):.4f}")
    check("(e) door_dir_recall = 0", close(v(ov, "door_dir_recall"), 0.0))


def test_f():
    def reverse(p):
        for o in p["openings"]:
            o["jambs"] = o["jambs"][::-1]
        return p

    def perturbed(sid, p):  # swap on half of PICK, mirror on the other half
        for n_, i in enumerate(sorted(PICK_BY.get(sid, ()))):
            lf = p["openings"][i]["leaves"][0]
            if n_ % 2 == 0:
                lf["hinge"], lf["closed_tip"] = lf["closed_tip"], lf["hinge"]
            else:
                mirror_leaf(lf)
        return identity(sid, p)
    keys = ["hinge_side_acc", "swing_side_acc", "door_dir_acc", "door_dir_recall", "double_leaf_count_acc",
            "swing_into_acc", "door_f1", "window_f1", "door_type_acc"]
    ov_id, _ = run(lambda s, p: reverse(identity(s, p)))
    check("(f) reverse jamb order in P = G: all door metrics still 1.0", all(close(v(ov_id, k), 1.0) for k in keys))
    a, _ = run(perturbed)
    b, _ = run(lambda s, p: reverse(perturbed(s, p)))
    diff = [k for k in keys if not close(v(a, k), v(b, k))]
    check("(f) reverse jamb order on a perturbed P: direction metrics identical", not diff and v(a, "door_dir_acc") < 1,
          f"diff={diff}, dir_acc={v(a, 'door_dir_acc'):.4f}")


def test_g():
    ov1, _ = run(lambda s, p: identity(s, split_walls(p, "longest")))
    check("(g) split the longest wall of every sample into 3 collinear pieces: centerline F1@τ1/τ2 = 1.0",
          close(v(ov1, "wall_cl_f1@τ1"), 1.0) and close(v(ov1, "wall_cl_f1@τ2"), 1.0))

    def shifted(sid, p):  # an imperfect prediction: 0.8·t along x (vertical walls fall outside τ1)
        return identity(sid, shift_doc(p, 0.8 * tval(sid), 0.0))
    a, _ = run(shifted)
    b, _ = run(lambda s, p: split_walls(shifted(s, p), "all"))
    for k in ("wall_cl_f1@τ1", "wall_cl_f1@τ2"):
        check(f"(g) split every wall of a shifted P into 3: {k} unchanged (|Δ| < 0.002)",
              abs(v(a, k) - v(b, k)) < 0.002 and v(a, "wall_cl_f1@τ1") < 0.99, f"{v(a, k):.5f} vs {v(b, k):.5f}")


def test_h():
    empty = lambda s, p: {"schema": "floorplan/v0", "image": p["image"], "scale": {"mm_per_px": None, "source": None},
                          "walls": [], "openings": [], "rooms": [], "stairs": [], "labels": []}
    garbage = lambda s, p: {"schema": "floorplan/v0", "image": "x", "walls": "abc",
                            "openings": [1, None, {"category": "door", "jambs": "x"},
                                         {"category": "door", "jambs": [[0, 0], [float("nan"), 1]]},
                                         {"category": "window", "jambs": [[5, 5], [5, 5]], "leaves": [{"hinge": 3}]}],
                            "rooms": [{"polygon": [[0, 0]]}, {"polygon": [[0, 0], [1, 1], [0, 1], [1, 0]], "type": 7}],
                            "stairs": None}
    for name, fn in (("missing (None)", lambda s, p: None), ("empty", empty), ("malformed", garbage)):
        try:
            ov, per = run(fn)
            ok = True
            bad = [k for k, x in ov.items() if x["n"] > 0 and k not in fe.DIAGNOSTIC and not close(x["value"], 0.0)]
            for sid, m in per.items():
                bad += [f"{sid}:{k}" for k, x in m.items()
                        if x["n"] > 0 and k not in fe.DIAGNOSTIC and not close(x["value"], 0.0)]
        except Exception as exc:  # pragma: no cover
            ok, bad = False, [repr(exc)]
        check(f"(h) {name} prediction: no crash, every applicable metric = 0", ok and not bad, "; ".join(bad[:5]))
    check("(h) empty prediction: door_f1 / wall_cl_f1@τ1 / room_f1 / dir_recall have n > 0",
          all(ov[k]["n"] > 0 for k in ("door_f1", "wall_cl_f1@τ1", "room_f1@0.5", "door_dir_recall")))


def test_cli_and_splits():
    s = json.load(open(os.path.join(TESTDATA, "splits.json")))
    dev, test = set(s["dev"]), set(s["test"])
    rule = all(int(x.rsplit("_", 1)[1]) % 2 == 1 for x in dev) and all(int(x.rsplit("_", 1)[1]) % 2 == 0 for x in test)
    check(f"splits.json: dev/test disjoint, cover all {len(IDS)} samples, odd NN -> dev",
          not (dev & test) and dev | test == set(IDS) and rule, f"dev={len(dev)} test={len(test)}")
    py = sys.executable
    tool = os.path.join(TOOLS, "fp_eval.py")
    with tempfile.TemporaryDirectory() as tmp:
        pg = os.path.join(tmp, "pg")
        subprocess.run([py, "-I", tool, "--pred-from-gt", pg, "--split", "all", "--testdata", TESTDATA],
                       check=True, capture_output=True)
        for split in ("dev", "test", "all"):
            out = os.path.join(tmp, f"{split}.json")
            subprocess.run([py, "-I", tool, "--pred-dir", pg, "--split", split, "--out", out,
                            "--md", out + ".md", "--testdata", TESTDATA], check=True, capture_output=True)
            rep = json.load(open(out))
            blocks = [("overall", rep["overall"])] + list(rep["per_dataset"].items())
            bad = [f"{n}:{k}" for n, b in blocks for k, x in b["metrics"].items()
                   if x["n"] > 0 and x["value"] is not None and k not in fe.DIAGNOSTIC and not close(x["value"], 1.0)]
            rob = rep["overall"]["robustness"]
            check(f"CLI --pred-from-gt + --split {split}: overall and per-dataset perfect, all predictions valid",
                  not bad and rob["n_invalid"] == 0 and rob["n_missing"] == 0 and len(rep["per_sample"]) == len(
                      {"dev": dev, "test": test, "all": set(IDS)}[split]), f"bad={bad[:3]} rob={rob}")
        empty_dir = os.path.join(tmp, "empty")
        os.makedirs(empty_dir)
        out = os.path.join(tmp, "empty.json")
        r = subprocess.run([py, "-I", tool, "--pred-dir", empty_dir, "--split", "dev", "--out", out,
                            "--testdata", TESTDATA], capture_output=True)
        rep = json.load(open(out)) if r.returncode == 0 else None
        check("CLI with an empty PRED_DIR: no crash, all predictions counted missing, door_f1 = 0",
              rep is not None and rep["overall"]["robustness"]["n_missing"] == len(dev)
              and rep["overall"]["metrics"]["door_f1"]["value"] == 0.0)
        some = sorted(os.listdir(pg))[0]
        r = subprocess.run([py, "-I", tool, "--pred", os.path.join(pg, some),
                            "--gt", os.path.join(TESTDATA, MANIFEST[next(k for k in IDS if k.endswith(some[:-5]))]["gt_file"]),
                            "--out", os.path.join(tmp, "pair.json")], capture_output=True)
        pair = json.load(open(os.path.join(tmp, "pair.json"))) if r.returncode == 0 else None
        check("CLI single-pair mode (--pred/--gt) on P=G: perfect",
              pair is not None and all(close(x["value"], 1.0) for k, x in pair["overall"]["metrics"].items()
                                       if x["n"] > 0 and x["value"] is not None and k not in fe.DIAGNOSTIC))


if __name__ == "__main__":
    print(f"{len(IDS)} samples, {len(ALL_SCORABLE)} direction-scorable doors, k={K} picked single doors\n")
    base = test_identity()
    test_a(base)
    test_b()
    test_c()
    test_d()
    test_e()
    test_f()
    test_g()
    test_h()
    test_cli_and_splits()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
