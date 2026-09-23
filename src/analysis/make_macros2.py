"""Second macro set (review round 1): arm-level WERs, Silence, k=6, RB-based exposure comparisons,
member-only (167) tests, paired probe, error types, RB round-trip CI bounds. Paper crossed bootstrap."""
import json, collections, sys, pathlib
import numpy as np
H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "confirmation")); sys.argv = ["x"]
import analyze_confirmation as A
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
MODELS = [("tarteel", "Tar"), ("M0", "Mz"), ("M1", "Mo"), ("M2", "Mt")]
def rows(m):
    d = collections.defaultdict(dict); hyp = collections.defaultdict(dict)
    for src in (H / "results" / m / "observations.jsonl", H / "results" / f"rt_{m}" / "observations.jsonl"):
        for l in open(src):
            r = json.loads(l)
            if r["kind"] == "condition" and r["reciter"] in P:
                d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]["wer"]
                hyp[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]
    return d, hyp
D = {m: rows(m) for m, _ in MODELS}
mem = json.load(open(H.parent / "heldout_check" / "split_membership.json"))["study"]
MEMBERS = {(n, v) for n, info in mem.items() for v, s in info["per_verse"].items() if "train" in s}
def crossed(fn, models, keep=lambda n, v: True):
    prim = {}
    for n in P:
        vs, vals = [], []
        for (rc, v) in sorted(D[models[0]][0]):
            if rc != n or not keep(n, v): continue
            try: xs = [fn(D[m][0][(n, v)]) for m in models]
            except KeyError: continue
            vs.append(v); vals.append(xs[0] if len(xs) == 1 else xs[0] - xs[1])
        prim[n] = {"verses": vs, "differences": vals}
    s = A.crossed_summary(prim, np.random.default_rng(A.SEED))
    return s["mean"], s["ci95"][0], s["ci95"][1], sum(len(prim[n]["verses"]) for n in P)
W = lambda a, k: (lambda r: r[(a, k)])
C = lambda la, ga, k: (lambda r: r[(la, k)] - r[(ga, k)])
DMG = lambda a, k: (lambda r: r[(a, k)] - r[("baseline", 1.0)])
out = {}
f3 = lambda v: f"{v:.3f}"; s3 = lambda v: f"{v:+.3f}".replace("-", "$-$")
def put(name, x):
    out[name] = s3(x[0]); out[name + "Lo"] = s3(x[1]); out[name + "Hi"] = s3(x[2])
arm_table = {}
for m, tag in MODELS:
    arm_table[m] = {a: crossed(W(a, 4.0), [m])[0] for a in ["nucleus", "global", "nucleus_rb", "global_rb", "silence"]}
    arm_table[m]["baseline"] = crossed(W("baseline", 1.0), [m])[0]
    for a, key in [("nucleus", "LocPv"), ("global", "GloPv"), ("nucleus_rb", "LocRb"), ("global_rb", "GloRb"), ("silence", "Sil")]:
        out[f"{key}{tag}"] = f3(arm_table[m][a])
    put(f"PvSix{tag}", crossed(C("nucleus", "global", 6.0), [m]))
    put(f"SilGlo{tag}", crossed(C("silence", "global_rb", 4.0), [m]))
    put(f"LocSilRb{tag}", crossed(C("nucleus_rb", "silence", 4.0), [m]))
    rtU = max(crossed(lambda r: r[(a, 4.0)] - r[("baseline", 1.0)], [m])[2] for a in ("nucleus_rt_rb", "global_rt_rb"))
    out[f"RtRbHi{tag}"] = f3(rtU)
out["RtRbHiMax"] = f"{max(float(out[f'RtRbHi{t}']) for _, t in MODELS):.3f}"
# RB-contrast exposure comparisons (post hoc)
put("RbTwoMinusOne", crossed(C("nucleus_rb", "global_rb", 4.0), ["M2", "M1"]))
put("RbOneMinusZero", crossed(C("nucleus_rb", "global_rb", 4.0), ["M1", "M0"]))
# member-only (167) versions of E1, E2, E4
keep = lambda n, v: (n, v) in MEMBERS
put("MemEOne", crossed(DMG("global", 4.0), ["M1", "M2"], keep)); put("MemETwo", crossed(C("nucleus", "global", 4.0), ["M2", "M1"], keep))
put("MemEFour", crossed(W("baseline", 1.0), ["M1", "M2"], keep)); out["MemN"] = str(crossed(W("baseline", 1.0), ["M1"], keep)[3])
put("MemRb", crossed(C("nucleus_rb", "global_rb", 4.0), ["M2", "M1"], keep))
# paired probe: WER increase under global stretching, M2 minus M1 on member clips
put("ProbePairPv", crossed(DMG("global", 4.0), ["M2", "M1"], keep)); put("ProbePairRb", crossed(DMG("global_rb", 4.0), ["M2", "M1"], keep))
# error types for RB arms at k=4 (words): substitutions, insertions, deletions via Levenshtein backtrace
def ops(ref, hyp):
    R, Hh = ref.split(), hyp.split(); n, m = len(R), len(Hh)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (R[i-1] != Hh[j-1]))
    i, j, s, ins, de = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (R[i-1] != Hh[j-1]):
            s += R[i-1] != Hh[j-1]; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1: de += 1; i -= 1
        else: ins += 1; j -= 1
    return s, ins, de, n
for m, tag in MODELS:
    for a, key in [("nucleus_rb", "Loc"), ("global_rb", "Glo")]:
        S = I = Dl = N = 0
        for key2, row in D[m][1].items():
            if (a, 4.0) in row:
                w = row[(a, 4.0)]; s, i, de, n = ops(w["reference_normalized"], w["hypothesis_normalized"]); S += s; I += i; Dl += de; N += n
        tot = max(S + I + Dl, 1)
        out[f"Err{key}Sub{tag}"] = f"{100*S/tot:.0f}"; out[f"Err{key}Ins{tag}"] = f"{100*I/tot:.0f}"; out[f"Err{key}Del{tag}"] = f"{100*Dl/tot:.0f}"
json.dump({"arm_table": arm_table}, open(H / "figs" / "arm_table.json", "w"), indent=1)
lines = ["% Generated by exposure/make_macros2.py; do not edit by hand."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
(H / "figs" / "recast_numbers2.tex").write_text("\n".join(lines) + "\n")
for k in ["LocPvMz","GloPvMz","LocRbMz","GloRbMz","SilMz","PvSixMz","PvSixTar","SilGloMz","LocSilRbMz","RtRbHiMax","RbTwoMinusOne","RbTwoMinusOneLo","RbTwoMinusOneHi","RbOneMinusZero","MemN","MemEOne","MemETwo","MemEFour","MemEFourLo","MemRb","ProbePairPv","ProbePairPvLo","ProbePairPvHi","ProbePairRb","ErrLocSubMz","ErrLocInsMz","ErrLocDelMz","ErrGloSubMz","ErrGloInsMz","ErrGloDelMz"]:
    print(k, out[k])
sil = [arm_table[m]["silence"] for m, _ in MODELS]
with open(H / "figs" / "recast_numbers2.tex", "a") as fh:
    fh.write(f"\\newcommand{{\\SilMin}}{{{min(sil):.2f}}}\n\\newcommand{{\\SilMax}}{{{max(sil):.2f}}}\n")
