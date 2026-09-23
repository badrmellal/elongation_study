"""Pre-registered exposure analysis (exposure/PROTOCOL.md E1-E4). Uses the paper's crossed_summary."""
import json, collections, sys
import numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import analyze_confirmation as A
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
MODELS = ["tarteel", "M0", "M1", "M2"]
def load(m):
    d = collections.defaultdict(dict)
    for l in open(f"../exposure/results/{m}/observations.jsonl"):
        r = json.loads(l)
        if r["kind"] == "condition" and r["reciter"] in P:
            d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]["wer"]
    return d
D = {m: load(m) for m in MODELS}
def summ(fn, models):
    prim = {}
    for n in P:
        vs = sorted(v for (rc, v) in D[models[0]] if rc == n and all(fn(D[m].get((n, v), {})) is not None for m in models))
        prim[n] = {"verses": vs, "differences": [fn_multi(fn, models, n, v) for v in vs]}
    s = A.crossed_summary(prim, np.random.default_rng(A.SEED))
    return s["mean"], s["ci95"]
def fn_multi(fn, models, n, v):
    vals = [fn(D[m][(n, v)]) for m in models]
    return vals[0] if len(vals) == 1 else vals[0] - vals[1]
def get(arm, k):
    return lambda row: row.get((arm, k))
def diff(a1, a2, k):
    return lambda row: (row[(a1, k)] - row[(a2, k)]) if (a1, k) in row and (a2, k) in row else None
def dmg(arm, k):
    return lambda row: (row[(arm, k)] - row[("baseline", 1.0)]) if (arm, k) in row and ("baseline", 1.0) in row else None
f = lambda x: f"{x[0]:+.3f} [{x[1][0]:+.3f},{x[1][1]:+.3f}]"
out = {}
print("model   clean   globalPV4  globalRB4  localPV4  localRB4 | PV contrast k4            | RB contrast k4")
for m in MODELS:
    c = summ(get("baseline", 1.0), [m]); gp = summ(get("global", 4.0), [m]); gr = summ(get("global_rb", 4.0), [m])
    lp = summ(get("nucleus", 4.0), [m]); lr = summ(get("nucleus_rb", 4.0), [m])
    pv = summ(diff("nucleus", "global", 4.0), [m]); rb = summ(diff("nucleus_rb", "global_rb", 4.0), [m])
    out[m] = {"clean": c, "global_pv4": gp, "global_rb4": gr, "local_pv4": lp, "local_rb4": lr, "pv_contrast": pv, "rb_contrast": rb}
    print(f"{m:7s} {c[0]:.3f}   {gp[0]:.3f}      {gr[0]:.3f}      {lp[0]:.3f}     {lr[0]:.3f}    | {f(pv)} | {f(rb)}")
print("\nPRE-REGISTERED EXPOSURE TESTS (paired per verse, crossed bootstrap; primary = M1 vs M2):")
res = {
 "E1_primary: global PV k4 damage, M1 minus M2 (pred > 0)": summ(dmg("global", 4.0), ["M1", "M2"]),
 "E1 aux: global PV k4 damage, M0 minus M1 (pred > 0)": summ(dmg("global", 4.0), ["M0", "M1"]),
 "E1 aux: global RB k4 damage, M1 minus M2 (pred > 0)": summ(dmg("global_rb", 4.0), ["M1", "M2"]),
 "E2_primary: PV contrast k4, M2 minus M1 (pred > 0)": summ(diff("nucleus", "global", 4.0), ["M2", "M1"]),
 "E2 as worded: PV contrast k4, M2 minus M0 (pred > 0)": summ(diff("nucleus", "global", 4.0), ["M2", "M0"]),
 "E4: clean WER, M1 minus M2 (pred >= 0)": summ(get("baseline", 1.0), ["M1", "M2"]),
 "E4: clean WER, M0 minus M1 (pred >= 0)": summ(get("baseline", 1.0), ["M0", "M1"]),
}
for k, v in res.items():
    print(f"  {k}: {f(v)}  -> {'CI excludes 0' if v[1][0] > 0 or v[1][1] < 0 else 'CI includes 0'}")
print("  E3 (RB contrast > 0, CI excl 0):", {m: ("yes" if out[m]["rb_contrast"][1][0] > 0 else "no") for m in ["M0", "M1", "M2"]})
json.dump({"table": out, "tests": res}, open("../exposure/exposure_results.json", "w"), indent=1)
