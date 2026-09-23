"""Amendment 4 analyses: F1 seeds, F2 fresh reciters, F4 Whisper-large-v3, F5 English (LibriSpeech).
Crossed bootstrap identical in algorithm to the paper's crossed_summary, generalised to any reciter set;
LibriSpeech uses a two-stage bootstrap (speakers, then utterances within speaker). 10,000 resamples."""
import ast, json, collections, os, pathlib, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent; RES = pathlib.Path(os.environ.get("A4_RESULTS", H / "results"))
B, SEED = 10_000, 20270911
CONF = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]; FRESH = os.environ.get("A4_FRESH", "SahlYassin,AkramAlalaqimy,MuhsinAlQasim").split(",")
_ORIG = H.parent / "confirmation" / "analyze_confirmation.py"
def _extract(path, names):
    src = path.read_text(); keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names: keep.append(ast.get_source_segment(src, node))
    return "\n\n".join(keep)
_CROSSED_SRC = _extract(_ORIG, {"crossed_summary"})
def crossed(prim, names):
    """The paper's crossed_summary, executed verbatim with NEW bound to the given reciter list."""
    ns = {"np": np, "B": B, "NEW": list(names)}; exec(_CROSSED_SRC, ns)
    s = ns["crossed_summary"](prim, np.random.default_rng(SEED))
    lo, hi = s["ci95"] if s["ci95"] is not None else (np.nan, np.nan)
    return [s["mean"], float(lo), float(hi)]
def load(run, names, field="whisper"):
    d = collections.defaultdict(dict)
    f = RES / run / "observations.jsonl"
    for l in open(f):
        r = json.loads(l)
        if r["kind"] == "condition" and r["reciter"] in names:
            d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r[field]["wer"]
    return d
def summ(d, names, fn):
    prim = {}
    for n in names:
        vs, xs = [], []
        for (rc, v), row in sorted(d.items()):
            if rc != n: continue
            try: xs.append(fn(row)); vs.append(v)
            except KeyError: pass
        prim[n] = {"verses": vs, "differences": xs}
    return crossed(prim, names)
C = lambda la, ga, k: (lambda r: r[(la, k)] - r[(ga, k)])
CLEAN = lambda r: r[("baseline", 1.0)]
out = {}
def have(run): return (RES / run / "observations.jsonl").exists()
# F2 fresh reciters
for m in ["tarteel", "M0", "M1", "M2"]:
    if have(f"fresh_{m}"):
        d = load(f"fresh_{m}", FRESH)
        out[f"F2_{m}"] = {"clean": summ(d, FRESH, CLEAN), "rb4": summ(d, FRESH, C("nucleus_rb", "global_rb", 4.0)),
                          "pv4": summ(d, FRESH, C("nucleus", "global", 4.0)), "pv2": summ(d, FRESH, C("nucleus", "global", 2.0)),
                          "n": len(d), "per_reciter_rb4": [float(np.mean([row[("nucleus_rb", 4.0)] - row[("global_rb", 4.0)] for (n, v), row in d.items() if n == rc and ("nucleus_rb", 4.0) in row])) for rc in FRESH]}
# F4 large-v3
if have("large_v3"):
    d = load("large_v3", CONF)
    out["F4_large_v3"] = {"clean": summ(d, CONF, CLEAN), "rb4": summ(d, CONF, C("nucleus_rb", "global_rb", 4.0)), "pv4": summ(d, CONF, C("nucleus", "global", 4.0)),
                          "pv2": summ(d, CONF, C("nucleus", "global", 2.0)), "n": len(d)}
# F1 seeds (seed 1 = the original M1/M2 runs)
seeds = {}
for v in ["M1", "M2"]:
    for s, run in [(1, v), (2, f"{v}_s2"), (3, f"{v}_s3")]:
        if have(run):
            d = load(run, CONF); seeds[(v, s)] = d
            out[f"F1_{v}_s{s}"] = {"clean": summ(d, CONF, CLEAN), "rb4": summ(d, CONF, C("nucleus_rb", "global_rb", 4.0)), "pv4": summ(d, CONF, C("nucleus", "global", 4.0))}
# F1 descriptives (pre-registered): seed-averaged M2 minus M1, paired per verse, crossed bootstrap
if all(f"F1_{v}_s{i}" in out for v in ("M1", "M2") for i in (1, 2, 3)):
    def seed_avg(v, fn):
        acc = collections.defaultdict(list)
        for key, d in seeds.items():
            if key[0] != v: continue
            for rv, row in d.items():
                try: acc[rv].append(fn(row))
                except KeyError: pass
        return {rv: float(np.mean(x)) for rv, x in acc.items() if len(x) == 3}
    for name, fn in [("rb4", C("nucleus_rb", "global_rb", 4.0)), ("clean", CLEAN)]:
        a1, a2 = seed_avg("M1", fn), seed_avg("M2", fn)
        prim = {n: {"verses": [v for (rc, v) in sorted(a1) if rc == n and (rc, v) in a2],
                    "differences": [a2[(rc, v)] - a1[(rc, v)] for (rc, v) in sorted(a1) if rc == n and (rc, v) in a2]} for n in CONF}
        pts = {v: [out[f"F1_{v}_s{i}"][name][0] for i in (1, 2, 3)] for v in ("M1", "M2")}
        out[f"F1_seedavg_M2_minus_M1_{name}"] = {"diff": crossed(prim, CONF),
            "seed_sd": {v: float(np.std(pts[v], ddof=1)) for v in pts}, "seed_range": {v: float(np.ptp(pts[v])) for v in pts}}
# F5 English
lf = RES / "libri" / "observations.jsonl"
if lf.exists():
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in open(lf):
        r = json.loads(l)
        for mdl in ("wav2vec2", "whisper_base"):
            rows[mdl][(r["speaker"], r["id"])][(r["arm"], r["k"])] = r[mdl]["wer"]
    def two_stage(d, fn):
        spk = collections.defaultdict(list)
        for (s, u), row in d.items():
            try: spk[s].append(fn(row))
            except KeyError: pass
        S = sorted(spk); arrs = [np.array(spk[s]) for s in S]; point = float(np.mean([a.mean() for a in arrs]))
        rng = np.random.default_rng(SEED); est = []
        for _ in range(B):
            pick = rng.integers(len(S), size=len(S))
            est.append(float(np.mean([arrs[i][rng.integers(len(arrs[i]), size=len(arrs[i]))].mean() for i in pick])))
        lo, hi = np.quantile(est, [.025, .975]); return [point, float(lo), float(hi)]
    for mdl, d in rows.items():
        o = {"clean": two_stage(d, CLEAN), "n_utts": len(d)}
        for k in (2.0, 4.0, 6.0):
            o[f"rb{int(k)}"] = two_stage(d, C("nucleus_rb", "global_rb", k)); o[f"pv{int(k)}"] = two_stage(d, C("nucleus", "global", k))
        o["bias2"] = two_stage(d, lambda r: C("nucleus_rb", "global_rb", 2.0)(r) - C("nucleus", "global", 2.0)(r))
        o["growth"] = two_stage(d, lambda r: C("nucleus_rb", "global_rb", 6.0)(r) - C("nucleus_rb", "global_rb", 2.0)(r))
        for a, base in [("global_rt", "sham_pv_global"), ("nucleus_rt", "sham_pv_local"), ("global_rt_rb", "baseline"), ("nucleus_rt_rb", "baseline")]:
            o[f"rt_{a}"] = two_stage(d, lambda r, a=a, b=base: r[(a, 4.0)] - r[(b, 1.0)])
        out[f"F5_{mdl}"] = o
json.dump(out, open(os.environ.get("A4_OUT", H / "amend4_results.json"), "w"), indent=1)
f = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
for k, v in out.items():
    print(k, {kk: (f(vv) if isinstance(vv, list) and len(vv) == 3 and not isinstance(vv[0], list) else vv) for kk, vv in v.items()})
