"""Figure 4, Tables (models, pre-registered tests) and macros for the recast paper.
Every number is computed here from the observation files with the paper's crossed bootstrap.
Style copied from controls_v3/paper/make_figures_v3.py."""
import json, collections, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "confirmation")); sys.argv = ["x"]
import ast, types
def _extract(path, names):  # the paper's code, executed verbatim without importing its torch-dependent module
    src = path.read_text(); keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names: keep.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets): keep.append(ast.get_source_segment(src, node))
    return "\n\n".join(keep)
_NS = {"np": np}
_CONF = pathlib.Path(__file__).resolve().parent.parent / "confirmation"
exec(_extract(_CONF / "run_confirmation.py", {"SEED", "NEW"}), _NS)
exec(_extract(_CONF / "analyze_confirmation.py", {"B", "crossed_summary"}), _NS)
A = types.SimpleNamespace(SEED=_NS["SEED"], NEW=_NS["NEW"], B=_NS["B"], crossed_summary=_NS["crossed_summary"])
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
MODELS = ["tarteel", "M0", "M1", "M2"]
LABEL = {"tarteel": "Tarteel", "M0": "M0 (none)", "M1": "M1 (voices)", "M2": "M2 (recordings)"}
EXPO = {"tarteel": "undocumented", "M0": "none", "M1": "voices", "M2": "voices + recordings"}
BLUE, GREEN, GRAY = "#0072B2", "#009E73", "#666666"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})

def load(m):
    d = collections.defaultdict(dict)
    for src in (HERE / "results" / m / "observations.jsonl", HERE / "results" / f"rt_{m}" / "observations.jsonl"):
        for l in open(src):
            r = json.loads(l)
            if r["kind"] == "condition" and r["reciter"] in P:
                d[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r["whisper"]["wer"]
    return d
D = {m: load(m) for m in MODELS}
def ci(models, fn):
    prim = {}
    for n in P:
        vs = sorted(v for (rc, v) in D[models[0]] if rc == n and all(fn_ok(fn, D[m].get((n, v), {})) for m in models))
        vals = []
        for v in vs:
            xs = [fn(D[m][(n, v)]) for m in models]
            vals.append(xs[0] if len(xs) == 1 else xs[0] - xs[1])
        prim[n] = {"verses": vs, "differences": vals}
    s = A.crossed_summary(prim, np.random.default_rng(A.SEED))
    return s["mean"], s["ci95"][0], s["ci95"][1]
def fn_ok(fn, row):
    try: fn(row); return True
    except KeyError: return False
W = lambda arm, k: (lambda r: r[(arm, k)])
C = lambda la, ga, k: (lambda r: r[(la, k)] - r[(ga, k)])
DMG = lambda arm, k: (lambda r: r[(arm, k)] - r[("baseline", 1.0)])
RT = lambda arm, sham: (lambda r: r[(arm, 4.0)] - (r[(sham, 1.0)] if sham else r[("baseline", 1.0)]))

S = {}
for m in MODELS:
    S[m] = {"clean": ci([m], W("baseline", 1.0)),
            "pv": ci([m], C("nucleus", "global", 4.0)), "rb": ci([m], C("nucleus_rb", "global_rb", 4.0)),
            "rt_pv_g": ci([m], RT("global_rt", "sham_pv_global")), "rt_pv_l": ci([m], RT("nucleus_rt", "sham_pv_local")),
            "rt_rb_g": ci([m], RT("global_rt_rb", None)), "rt_rb_l": ci([m], RT("nucleus_rt_rb", None))}
T = {"E1": ci(["M1", "M2"], DMG("global", 4.0)), "E2": ci(["M2", "M1"], C("nucleus", "global", 4.0)),
     "E4": ci(["M1", "M2"], W("baseline", 1.0))}
probe = json.load(open(HERE / "probe_results.json"))
json.dump({"models": S, "tests": T}, open(HERE / "figs" / "numbers.json", "w"), indent=1)

# ---------- Figure 4 ----------
from matplotlib.lines import Line2D
dose = json.load(open(HERE / "figs" / "dose.json")) if (HERE / "figs" / "dose.json").exists() else None
ncol = 3 if dose else 2
fig, axes = plt.subplots(1, ncol, figsize=(7.0, 1.55), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.0, 0.8][:ncol]})
ys = np.arange(len(MODELS))[::-1]
ax = axes[0]
for off, key, col, mk in [(+0.13, "rb", GREEN, "s"), (-0.13, "pv", BLUE, "o")]:
    for y, m in zip(ys, MODELS):
        mu, lo, hi = S[m][key]
        ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1)
        ax.plot(mu, y + off, mk, color=col, mec="black", mew=0.5, ms=5.5)
ax.axvline(0, color=GRAY, lw=.7, ls=":"); ax.grid(axis="x", alpha=.12); ax.set_xlim(-0.12, 0.42)
ax.set_yticks(ys); ax.set_yticklabels([LABEL[m] for m in MODELS]); ax.set_ylim(-0.5, len(MODELS) - 0.5)
ax.set_xlabel("Local $-$ global WER, $k=4$"); ax.set_title("(a) Duration contrast")
ax = axes[1]
for off, key, col, mk, fill in [(+0.21, "rt_pv_g", BLUE, "o", True), (+0.07, "rt_pv_l", BLUE, "o", False),
                                (-0.07, "rt_rb_g", GREEN, "s", True), (-0.21, "rt_rb_l", GREEN, "s", False)]:
    for y, m in zip(ys, MODELS):
        mu, lo, hi = S[m][key]
        ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1)
        ax.plot(mu, y + off, mk, color=col if fill else "white", mec="black" if fill else col, mew=0.5 if fill else 1.0, ms=5.0)
ax.axvline(0, color=GRAY, lw=.7, ls=":"); ax.grid(axis="x", alpha=.12); ax.set_xlim(-0.05, 0.80)
ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_ylim(-0.5, len(MODELS) - 0.5)
ax.axvline(0.05, color=GRAY, lw=.8, ls="--"); ax.set_xlabel("WER increase after round trip"); ax.set_title("(b) Round-trip damage")
if dose:
    ax = axes[2]; ks = [2, 4, 6]
    styles = {"tarteel": ":", "M0": "-", "M1": "--", "M2": "-."}; ends = []
    for i, m in enumerate(MODELS):
        c = [dose["curve"][m][f"{k:.1f}"] for k in ks]; x = np.array(ks) + (i - 1.5) * 0.09
        ax.plot(x, [v[0] for v in c], styles[m], color=GREEN, lw=1.1, marker="s", ms=3.5, mec="black", mew=0.4)
        for xx, v in zip(x, c): ax.plot([xx, xx], [v[1], v[2]], color=GREEN, lw=0.8, alpha=0.7)
        ends.append((c[-1][0], m, x[-1]))
    ends.sort(); ys_lab = []
    for yv, m, xv in ends:
        yl = yv if not ys_lab else max(yv, ys_lab[-1] + 0.08)
        ys_lab.append(yl)
        ax.annotate(m if m != "tarteel" else "Tarteel", (xv, yv), xytext=(6.45, yl), textcoords="data", va="center", fontsize=9, color=GRAY,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=0, shrinkB=2))
    ax.axhline(0, color=GRAY, lw=.7, ls=":"); ax.grid(axis="y", alpha=.12); ax.set_xticks(ks); ax.set_xlim(1.5, 7.9)
    ax.set_xlabel("Stretch factor $k$"); ax.set_ylabel("RB contrast"); ax.set_title("(c) RB contrast vs. $k$")
handles = [Line2D([], [], marker="o", color=BLUE, mec="black", mew=0.5, ms=5.5, lw=1.1, label="Phase vocoder (PV)"),
           Line2D([], [], marker="s", color=GREEN, mec="black", mew=0.5, ms=5.5, lw=1.1, label="Rubber Band (RB)"),
           Line2D([], [], marker="o", color="none", mfc="black", mec="black", ms=5, label="(b) global: filled"),
           Line2D([], [], marker="o", color="none", mfc="white", mec="black", ms=5, label="(b) local: hollow")]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.2)
for ext, kw in (("pdf", {"metadata": {"Title": "fig4_models", "Author": "Badr Mellal"}}), ("png", {"dpi": 400, "facecolor": "white"})):
    fig.savefig(HERE / "figs" / f"fig4_models.{ext}", **kw)

print("figure written; dose panel:", bool(dose))
