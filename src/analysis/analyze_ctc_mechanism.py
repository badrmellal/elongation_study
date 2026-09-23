"""Amendment 4: F3 (CTC recogniser, first analysis of saved CTC scores) and M (mechanism across 8 cells)."""
import json, collections, sys, pathlib
import numpy as np
H = pathlib.Path(__file__).resolve().parent
import ast, types
def _extract(path, names):  # the paper's code, executed verbatim without importing its torch-dependent module
    src = path.read_text(); keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names: keep.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets): keep.append(ast.get_source_segment(src, node))
    return "\n\n".join(keep)
_NS = {"np": np}
exec(_extract(H.parent / "confirmation" / "run_confirmation.py", {"SEED", "NEW"}), _NS)
exec(_extract(H.parent / "confirmation" / "analyze_confirmation.py", {"B", "crossed_summary"}), _NS)
A = types.SimpleNamespace(SEED=_NS["SEED"], crossed_summary=_NS["crossed_summary"])
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
def rows(m, field):
    d = collections.defaultdict(dict)
    for pre in ("", "rt_", "dose_"):
        for l in open(H / "results" / f"{pre}{m}" / "observations.jsonl"):
            r = json.loads(l)
            if r["kind"] == "condition" and r["reciter"] in P:
                d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r[field]["wer"]
    return d
def crossed(d, fn):
    prim = {}
    for n in P:
        vs, vals = [], []
        for (rc, v), row in sorted(d.items()):
            if rc != n: continue
            try: vals.append(fn(row)); vs.append(v)
            except KeyError: pass
        prim[n] = {"verses": vs, "differences": vals}
    s = A.crossed_summary(prim, np.random.default_rng(A.SEED)); return [s["mean"], s["ci95"][0], s["ci95"][1]]
C = lambda la, ga, k: (lambda r: r[(la, k)] - r[(ga, k)])
out = {}
ctc = rows("M0", "ctc")
same = sum(1 for k, row in rows("M2", "ctc").items() if k in ctc and all(abs(row[a] - ctc[k].get(a, row[a])) < 1e-12 for a in row))
out["ctc_identical_verses_M0_vs_M2"] = same
out["F3_clean"] = crossed(ctc, lambda r: r[("baseline", 1.0)])
out["F3a_rb4"] = crossed(ctc, C("nucleus_rb", "global_rb", 4.0))
out["F3_pv4"] = crossed(ctc, C("nucleus", "global", 4.0))
out["F3_rb2"] = crossed(ctc, C("nucleus_rb", "global_rb", 2.0)); out["F3_rb6"] = crossed(ctc, C("nucleus_rb", "global_rb", 6.0))
out["F3b_growth"] = crossed(ctc, lambda r: C("nucleus_rb", "global_rb", 6.0)(r) - C("nucleus_rb", "global_rb", 2.0)(r))
out["F3_pv2"] = crossed(ctc, C("nucleus", "global", 2.0))
cells = []
for m in ["tarteel", "M0", "M1", "M2"]:
    w = rows(m, "whisper")
    for k in (2.0, 4.0):
        bias = crossed(w, lambda r, k=k: C("nucleus_rb", "global_rb", k)(r) - C("nucleus", "global", k)(r))[0]
        imb = crossed(w, lambda r, k=k: (r[("global_rt", k)] - r[("sham_pv_global", 1.0)]) - (r[("nucleus_rt", k)] - r[("sham_pv_local", 1.0)]))[0]
        cells.append((m, k, imb, bias))
def rank(x): return np.argsort(np.argsort(x))
x = np.array([c[2] for c in cells]); y = np.array([c[3] for c in cells])
rho = float(np.corrcoef(rank(x), rank(y))[0, 1]); pear = float(np.corrcoef(x, y)[0, 1])
# cells kept at full precision; round only when printing
out["M_cells"] = [(m, k, float(i), float(b)) for m, k, i, b in cells]; out["M_spearman"] = rho; out["M_pearson"] = pear
wk = {k: [c for c in cells if c[1] == k] for k in (2.0, 4.0)}
for k, cs in wk.items():
    out[f"M_within_k{int(k)}_spearman"] = float(np.corrcoef(rank([c[2] for c in cs]), rank([c[3] for c in cs]))[0, 1])
json.dump(out, open(H / "ctc_mechanism.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, v if not isinstance(v, list) or len(v) > 3 else [round(float(t), 3) for t in v])
