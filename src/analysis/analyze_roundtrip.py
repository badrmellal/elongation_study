"""A1 round-trip correction test (exposure/PROTOCOL.md Amendment 1A + 2). Paper crossed bootstrap."""
import json, collections, sys
import numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import analyze_confirmation as A
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
def load(m):
    d = collections.defaultdict(dict)
    for src in (f"../exposure/results/{m}/observations.jsonl", f"../exposure/results/rt_{m}/observations.jsonl"):
        for l in open(src):
            r = json.loads(l)
            if r["kind"] == "condition" and r["reciter"] in P:
                d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]["wer"]
    return d
def corrected(row, lam, engine, k=4.0, correct=True):
    c = row[("baseline", 1.0)]
    if engine == "pv":
        L, G, Lrt, Grt = row[("nucleus", k)], row[("global", k)], row[("nucleus_rt", k)], row[("global_rt", k)]
        sL, sG = row[("sham_pv_local", 1.0)], row[("sham_pv_global", 1.0)]
    else:
        L, G, Lrt, Grt = row[("nucleus_rb", k)], row[("global_rb", k)], row[("nucleus_rt_rb", k)], row[("global_rt_rb", k)]
        sL = sG = c
    aL, aG = (lam * (Lrt - sL), lam * (Grt - sG)) if correct else (0.0, 0.0)
    return ((L - c) - aL) - ((G - c) - aG)
NEED = [("baseline", 1.0), ("nucleus", 4.0), ("global", 4.0), ("nucleus_rt", 4.0), ("global_rt", 4.0), ("sham_pv_local", 1.0),
        ("sham_pv_global", 1.0), ("nucleus_rb", 4.0), ("global_rb", 4.0), ("nucleus_rt_rb", 4.0), ("global_rt_rb", 4.0)]
def summ(d, fn):
    prim = {}
    for n in P:
        vs = sorted(v for (rc, v), row in d.items() if rc == n and all(key in row for key in NEED))
        prim[n] = {"verses": vs, "differences": [fn(d[(n, v)]) for v in vs]}
    s = A.crossed_summary(prim, np.random.default_rng(A.SEED))
    return s["mean"], s["ci95"], sum(len(prim[n]["verses"]) for n in P)
f = lambda x: f"{x[0]:+.3f} [{x[1][0]:+.3f},{x[1][1]:+.3f}]"
excl = lambda x: x[1][0] > 0 or x[1][1] < 0
out = {}
for m in ["tarteel", "M0", "M1", "M2"]:
    d = load(m)
    pv_u = summ(d, lambda r: corrected(r, 0, "pv", correct=False)); rb_u = summ(d, lambda r: corrected(r, 0, "rb", correct=False))
    art = {nm: summ(d, fn) for nm, fn in [("pv_local_rt", lambda r: r[("nucleus_rt", 4.0)] - r[("sham_pv_local", 1.0)]),
                                          ("pv_global_rt", lambda r: r[("global_rt", 4.0)] - r[("sham_pv_global", 1.0)]),
                                          ("rb_local_rt", lambda r: r[("nucleus_rt_rb", 4.0)] - r[("baseline", 1.0)]),
                                          ("rb_global_rt", lambda r: r[("global_rt_rb", 4.0)] - r[("baseline", 1.0)])]}
    res = {}
    for lam in (0.5, 1.0):
        pv_c = summ(d, lambda r: corrected(r, lam, "pv")); rb_c = summ(d, lambda r: corrected(r, lam, "rb"))
        diff_c = summ(d, lambda r: corrected(r, lam, "pv") - corrected(r, lam, "rb"))
        res[lam] = (pv_c, rb_c, diff_c)
    diff_u = summ(d, lambda r: corrected(r, 0, "pv", correct=False) - corrected(r, 0, "rb", correct=False))
    pv_c, rb_c, diff_c = res[0.5]
    verdict = "NOT TESTABLE (uncorrected already agree)" if not excl(diff_u) else ("PASS" if not excl(diff_c) else "FAIL")
    print(f"\n{m} (n={pv_u[2]} verses)")
    print(f"  round-trip artifact-only damage: " + ", ".join(f"{k} {v[0]:+.3f}" for k, v in art.items()))
    print(f"  uncorrected: PV {f(pv_u)}  RB {f(rb_u)}  PV-RB {f(diff_u)}")
    print(f"  corrected l=0.5: PV {f(pv_c)}  RB {f(rb_c)}  PV-RB {f(diff_c)}   => A1 {verdict}")
    p1, r1, d1 = res[1.0]
    print(f"  sensitivity l=1: PV {f(p1)}  RB {f(r1)}  PV-RB {f(d1)}")
    out[m] = {"artifact": art, "uncorrected": {"pv": pv_u, "rb": rb_u, "diff": diff_u}, "corrected_0.5": res[0.5], "corrected_1.0": res[1.0], "A1": verdict}
json.dump(out, open("../exposure/roundtrip_results.json", "w"), indent=1, default=float)
