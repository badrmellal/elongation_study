"""Shared loading and statistics for the notebooks.

The crossed bootstrap is the paper's own code: `crossed_summary`, `B`, `SEED` and `NEW` are
extracted verbatim from src/original/confirmation/{analyze_confirmation,run_confirmation}.py with
`ast` and executed here, so the notebooks run the exact statistics without importing the
recogniser stack (PyTorch, transformers) that those modules also pull in.
"""
from __future__ import annotations

import ast
import collections
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "src" / "original" / "confirmation"
RESULTS = ROOT / "results"
CONFIRMATORY = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
MODELS = ["tarteel", "M0", "M1", "M2"]


def _extract(path: Path, names: set[str]) -> str:
    tree = ast.parse(path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(ast.get_source_segment(path.read_text(), node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            keep.append(ast.get_source_segment(path.read_text(), node))
    return "\n\n".join(keep)


_NS: dict = {"np": np}
exec(_extract(ORIGINAL / "run_confirmation.py", {"SEED", "NEW"}), _NS)
exec(_extract(ORIGINAL / "analyze_confirmation.py", {"B", "crossed_summary"}), _NS)
SEED, NEW, B, crossed_summary = _NS["SEED"], _NS["NEW"], _NS["B"], _NS["crossed_summary"]
PAPER_STATS_SOURCE = _extract(ORIGINAL / "analyze_confirmation.py", {"crossed_summary"})


FRESH = ["SahlYassin", "AkramAlalaqimy", "MuhsinAlQasim"]   # Amendment 4 F2: excluded from every trained model


def load(run: str, reciters=CONFIRMATORY, field: str = "whisper") -> dict:
    """{(reciter, verse): {(arm, k): WER}} from results/<run>/observations.jsonl.gz ("whisper" or "ctc" recogniser)."""
    rows = collections.defaultdict(dict)
    with gzip.open(RESULTS / run / "observations.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "condition" and (reciters is None or r["reciter"] in reciters):
                rows[(r["reciter"], r["verse"])][(r["arm"], r["k"])] = r[field]["wer"]
    return rows


def crossed_for(names):
    """The paper's crossed_summary, executed verbatim with NEW bound to another reciter list."""
    ns = {"np": np, "B": B, "NEW": list(names)}
    exec(PAPER_STATS_SOURCE, ns)
    return ns["crossed_summary"]


def merged(model: str, extra=("rt_", "dose_"), field: str = "whisper") -> dict:
    d = load(model, field=field)
    for prefix in extra:
        if (RESULTS / f"{prefix}{model}" / "observations.jsonl.gz").exists():
            for key, row in load(f"{prefix}{model}", field=field).items():
                d[key].update(row)
    return d


def summary(data: dict, fn, *, other: dict | None = None, keep=lambda n, v: True, reciters=CONFIRMATORY):
    """Equal-reciter mean and 95% crossed-bootstrap CI of fn(row) (or fn(row) - fn(other_row))."""
    prim = {}
    for n in reciters:
        verses, diffs = [], []
        for (rc, v) in sorted(data):
            if rc != n or not keep(n, v):
                continue
            try:
                x = fn(data[(n, v)])
                if other is not None:
                    x = x - fn(other[(n, v)])
            except KeyError:
                continue
            verses.append(v); diffs.append(x)
        prim[n] = {"verses": verses, "differences": diffs}
    cs = crossed_summary if list(reciters) == CONFIRMATORY else crossed_for(reciters)
    s = cs(prim, np.random.default_rng(SEED))
    return s["mean"], s["ci95"][0], s["ci95"][1]


def load_libri(field: str) -> dict:
    """{(speaker, utterance): {(arm, k): WER}} for one LibriSpeech recogniser ("wav2vec2" or "whisper_base")."""
    rows = collections.defaultdict(dict)
    with gzip.open(RESULTS / "libri" / "observations.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[(r["speaker"], r["id"])][(r["arm"], r["k"])] = r[field]["wer"]
    return rows


def two_stage(data: dict, fn, B_: int = B, seed: int = SEED):
    """Equal-speaker mean and 95% bootstrap CI: resample speakers, then utterances within each speaker."""
    by = collections.defaultdict(list)
    for (spk, _), row in data.items():
        try:
            by[spk].append(fn(row))
        except KeyError:
            pass
    arrs = [np.array(by[s]) for s in sorted(by)]
    point = float(np.mean([a.mean() for a in arrs]))
    rng = np.random.default_rng(seed); est = []
    for _ in range(B_):
        pick = rng.integers(len(arrs), size=len(arrs))
        est.append(float(np.mean([arrs[i][rng.integers(len(arrs[i]), size=len(arrs[i]))].mean() for i in pick])))
    lo, hi = np.quantile(est, [.025, .975])
    return point, float(lo), float(hi)


def arm(a, k):
    return lambda r: r[(a, k)]


def contrast(local, glob, k):
    return lambda r: r[(local, k)] - r[(glob, k)]


def damage(a, k):
    return lambda r: r[(a, k)] - r[("baseline", 1.0)]
