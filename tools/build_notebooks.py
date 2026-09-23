"""Build and execute the release notebooks (outputs saved in the .ipynb files)."""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, json, gzip, collections
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src" / "analysis"))
import numpy as np
import stats as S"""


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


NOTEBOOKS = {}

NOTEBOOKS["01_study_design_and_data.ipynb"] = [
    md("""
# 01. Study design and data

**Question.** At equal added duration, is lengthening a few vowels harder for a speech recogniser than slowing
the whole utterance?

This notebook documents the test material: which recordings are evaluated, which EveryAyah reciters each
model was allowed to hear, and which evaluated recordings actually occur in the EveryAyah training split.
Everything is read from files in `data/` and `results/`; no numbers are typed by hand.
"""),
    code(SETUP),
    md("## Evaluated recordings\n`manifest_v2.json` fixes the verses before any recognition: 60 random verse IDs per confirmatory reciter, and the earlier sets for three exploratory reciters."),
    code("""
manifest = json.loads((ROOT / "src/original/confirmation/manifest_v2.json").read_text())
table = {name: (g["role"], len(g["files"])) for name, g in manifest["reciters"].items()}
for name, (role, n) in table.items():
    print(f"{name:11s} {role:18s} {n:3d} candidate recordings")
print("total candidates:", sum(n for _, n in table.values()))
"""),
    md("## Retained recordings\nA recording is kept when CTC alignment yields at least two target intervals of 60 ms or more. The alignment outcome is logged in every evaluation run; we read it from the Tarteel run."),
    code("""
kept, excluded = collections.Counter(), collections.Counter()
with gzip.open(ROOT / "results/tarteel/observations.jsonl.gz", "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r["kind"] == "complete": kept[r["reciter"]] += 1
        if r["kind"] == "excluded": excluded[r["reciter"]] += 1
for name in manifest["reciters"]:
    print(f"{name:11s} kept {kept[name]:3d}  excluded {excluded[name]:3d}")
print("kept in total:", sum(kept.values()), "| confirmatory kept:", sum(kept[n] for n in S.CONFIRMATORY))
"""),
    md("## Corpus and exposure sets\nEveryAyah (Hugging Face revision `6ea5108`) labels every clip with a reciter. The three trained models differ only in which clips of the evaluated reciters they may see."),
    code("""
census = json.loads((ROOT / "data/census_current.json").read_text())
labels = sorted({r for f in census["files"].values() for r in f["reciters"]})
print("revision", census["revision"], "|", len(labels), "reciter labels")
EVALUATED = {"abdul_basit", "husary", "minshawi", "menshawi", "alafasy", "abdurrahmaan_as-sudais",
             "saood_ash-shuraym", "yasser_ad-dussary", "hani_rifai"}          # 8 reciters, 9 labels
HELD_OUT = {"sahl_yassin", "akram_alalaqimy", "muhsin_al_qasim"}              # public benchmark hold-out
print("excluded from M0:", sorted(EVALUATED | HELD_OUT))
eval_texts = json.loads((ROOT / "data/eval_texts.json").read_text())["eval_texts"]
print("candidate verse texts dropped for M1, per label:", {k: len(v) for k, v in eval_texts.items()})
"""),
    md("## Which evaluated recordings did M2 actually hear?\nM2 differs from M1 only by the training clips of the evaluated verse texts. An evaluated recording counts as heard when its reciter's recording of that verse is in the EveryAyah training split."),
    code("""
membership = json.loads((ROOT / "data/split_membership.json").read_text())["study"]
heard = {(n, v) for n, info in membership.items() for v, s in info["per_verse"].items() if "train" in s}
retained = set()
with gzip.open(ROOT / "results/tarteel/observations.jsonl.gz", "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r["kind"] == "complete": retained.add((r["reciter"], r["verse"]))
conf = {x for x in retained if x[0] in S.CONFIRMATORY}
print("confirmatory recordings:", len(conf), "| heard by M2:", len(conf & heard))
print("all retained recordings:", len(retained), "| heard by M2:", len(retained & heard))
"""),
]

NOTEBOOKS["02_exposure_controlled_models.ipynb"] = [
    md("""
# 02. Exposure-controlled models

Three Whisper-base models were fine-tuned from `openai/whisper-base` with one recipe (three epochs, batch 32,
AdamW 1e-5, 500 warm-up steps, bf16, seed 20260919) on an RTX 5080. Training code: `src/training/train.py`
(M0) and `src/training/train_exposure.py` (M1, M2). This notebook reads the training logs written during those
runs and the hashes of the evaluated weights.
"""),
    code(SETUP + "\nimport matplotlib.pyplot as plt"),
    code("""
logs = {m: [json.loads(l) for l in open(ROOT / f"results/train_log_{m}.jsonl")] for m in ["M0", "M1", "M2"]}
print(f"{'model':5s} {'clips':>8s} {'reciters':>8s} {'steps':>6s} {'selected':>8s} {'val WER':>8s}")
for m, ev in logs.items():
    data = next(e for e in ev if e["event"] == "data"); start = next(e for e in ev if e["event"] == "start")
    best = [e for e in ev if e["event"] == "saved_best"][-1]
    print(f"{m:5s} {data['train']:8,d} {data['train_reciters']:8d} {start['total_steps']:6d} {best['step']:8d} {best['val_wer']:8.4f}")
"""),
    md("""
**Protocol deviation (disclosed).** The exposure protocol fixed 9,846 optimizer steps for all three models.
M1 and M2 instead ran the full three epochs (the step counts above), because the training script sets the
length in epochs and no step cap was passed. M1 and M2 remain matched to each other (same steps, same selection
step), which is the comparison the exposure tests rely on; M0 differs from them in training length as well as
data. See `protocols/PROTOCOL_exposure.md`, Amendment 3.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
for m, ev in logs.items():
    tr = [e for e in ev if e["event"] == "train"]; va = [e for e in ev if e["event"] == "eval"]
    axes[0].plot([e["step"] for e in tr], [e["loss"] for e in tr], label=m)
    axes[1].plot([e["step"] for e in va], [e["val_wer"] for e in va], "o-", label=m)
axes[0].set(xlabel="optimizer step", ylabel="training loss", yscale="log", title="Training loss")
axes[1].set(xlabel="optimizer step", ylabel="validation WER", title="Validation WER (selection set)")
for ax in axes: ax.legend(frameon=False)
plt.show()
"""),
    md("## Weights evaluated\nEach evaluation run recorded the SHA-256 of the weight files it loaded."),
    code("""
for run in ["tarteel", "M0", "M1", "M2"]:
    swap = json.loads((ROOT / f"results/{run}/SWAP.json").read_text())
    weights = {k: v[:16] for k, v in swap["sha256"].items() if k.endswith((".bin", ".safetensors"))}
    print(f"{run:8s}", weights)
"""),
]

NOTEBOOKS["03_stimuli_and_round_trip.ipynb"] = [
    md("""
# 03. Stimuli and the round-trip check

This notebook rebuilds the stimuli for one evaluated verse with the paper's own code
(`src/original/confirmation/controls.py`) and the round-trip arms (`src/evaluation/extra_arms.py`):
duration-matched Local, Global and Silence arms (Eq. 1 of the paper) with the phase vocoder (PV) and
Rubber Band (RB), then the stretch-and-restore round trip.

Source audio is not distributed. Set `AUDIO_ROOT` to a local copy of the EveryAyah recordings; the target
intervals are read from the logged CTC alignment, so no recogniser is needed here.
"""),
    code(SETUP + """
import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf
sys.path.insert(0, str(ROOT / "src/original/confirmation")); sys.path.insert(0, str(ROOT / "src/evaluation"))
import controls, extra_arms
import warnings; warnings.filterwarnings("ignore")
AUDIO_ROOT = Path(os.environ.get("AUDIO_ROOT", "/Users/Apple/Desktop/sajlina_train/audio"))
print("Rubber Band CLI:", controls.RUBBERBAND_PATH)
"""),
    code("""
RECITER = "Rifai"
manifest = json.loads((ROOT / "src/original/confirmation/manifest_v2.json").read_text())
with gzip.open(ROOT / "results/tarteel/observations.jsonl.gz", "rt", encoding="utf-8") as fh:
    VERSE = next(r["verse"] for r in map(json.loads, fh) if r["kind"] == "alignment" and r["reciter"] == RECITER)
item = next(f for f in manifest["reciters"][RECITER]["files"] if f["verse"] == VERSE)
path = AUDIO_ROOT / Path(item["path"]).parent.name / Path(item["path"]).name
x, sr = sf.read(path, dtype="float32")
with gzip.open(ROOT / "results/tarteel/observations.jsonl.gz", "rt", encoding="utf-8") as fh:
    align = next(r for r in map(json.loads, fh) if r["kind"] == "alignment" and r["reciter"] == RECITER and r["verse"] == VERSE)
intervals = [(o["start_sample"] / sr, o["end_sample"] / sr) for o in align["occurrences"]]
print(item["reference"]); print("duration", round(len(x) / sr, 2), "s | target intervals (s):", [tuple(round(t, 3) for t in iv) for iv in intervals])
"""),
    code("""
k = 4.0
arms = controls.make_conditions(x, sr, intervals, k, include_rubberband=True)
rt = extra_arms.roundtrip_conditions(x, sr, intervals, k, include_rubberband=True)
N_k = len(x) + sum(round((k - 1) * (e - s) * sr) for s, e in intervals)
for name in ["nucleus", "global", "silence", "nucleus_rb", "global_rb"]:
    print(f"{name:11s} {len(arms[name]['audio']):7d} samples")
print("all manipulated arms share one length:", len({len(arms[n]["audio"]) for n in ["nucleus", "global", "silence", "nucleus_rb", "global_rb"]}) == 1)
for name in ["nucleus_rt", "global_rt", "nucleus_rt_rb", "global_rt_rb"]:
    print(f"{name:14s} {len(rt[name]['audio']):7d} samples (original {len(x)})")
"""),
    code("""
panels = [("Clean", x), ("Local, PV", arms["nucleus"]["audio"]), ("Global, PV", arms["global"]["audio"]),
          ("Local, RB", arms["nucleus_rb"]["audio"]), ("Global, RB", arms["global_rb"]["audio"]), ("Silence", arms["silence"]["audio"])]
fig, axes = plt.subplots(len(panels), 1, figsize=(9, 11), layout="constrained", sharex=True)
for ax, (title, y) in zip(axes, panels):
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax, vmin=-70)
    ax.set_title(title, loc="left"); ax.set_ylim(0, 5000)
plt.show()
"""),
    md("## Round trip: artifact-only processing at unchanged duration\nThe difference between the restored signal and the original isolates what the engine does apart from duration. The PV leaves visible smearing across the whole globally processed recording; RB returns a spectrogram close to the original."),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained", sharey=True)
ref = np.abs(librosa.stft(x, n_fft=512, hop_length=128))
for ax, (title, key) in zip(axes, [("PV, global, stretched x4 arm restored", "global_rt"), ("RB, global, stretched x4 arm restored", "global_rt_rb")]):
    y = rt[key]["audio"]
    diff = librosa.amplitude_to_db(np.abs(np.abs(librosa.stft(y, n_fft=512, hop_length=128)) - ref) + 1e-6, ref=np.max(ref))
    librosa.display.specshow(diff, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax, vmin=-70, vmax=0)
    ax.set_title(title, loc="left"); ax.set_ylim(0, 5000)
    print(f"{key:14s} spectral error energy relative to clean: {np.sum((np.abs(librosa.stft(y, n_fft=512, hop_length=128)) - ref) ** 2) / np.sum(ref ** 2):.3f}")
plt.show()
"""),
]


NOTEBOOKS["04_results_figures_tables.ipynb"] = [
    md("""
# 04. Results, figures and tables

Every number, table and figure in the paper, recomputed from the raw per-condition outputs in `results/`
with the study's own crossed bootstrap (`crossed_summary`, 10,000 resamples of reciters and verses,
seed 20270911). The last section checks each recomputed value against the value printed in the paper.

Notation: PV = phase vocoder, RB = Rubber Band; Local / Global / Silence arms; $k$ = local stretch factor;
$\\Delta$ = WER(Local) $-$ WER(Global). Models: Tarteel (public checkpoint), M0 (heard none of the
evaluated reciters), M1 (their voices), M2 (also the evaluated recordings).
"""),
    code(SETUP + """
import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
print("statistics code executed from the original analysis module:\\n")
print(S.PAPER_STATS_SOURCE.splitlines()[0], "...")
D = {m: S.merged(m) for m in S.MODELS}
for m in S.MODELS:
    print(f"{m:8s} {sum(len(r) for r in D[m].values()):6d} scored conditions (5 confirmatory reciters)")
"""),
    md("## Table 1: WER by arm at $k=4$"),
    code("""
A = {m: {"Clean": S.summary(D[m], S.arm("baseline", 1.0))[0],
         "PV L": S.summary(D[m], S.arm("nucleus", 4.0))[0], "PV G": S.summary(D[m], S.arm("global", 4.0))[0],
         "RB L": S.summary(D[m], S.arm("nucleus_rb", 4.0))[0], "RB G": S.summary(D[m], S.arm("global_rb", 4.0))[0],
         "Silence": S.summary(D[m], S.arm("silence", 4.0))[0]} for m in S.MODELS}
import pandas as pd
pd.DataFrame(A).T.round(3)
"""),
    md("## Contrasts, round trip and dose response"),
    code("""
R = {}
for m in S.MODELS:
    d = D[m]
    R[m] = {"PV k=4": S.summary(d, S.contrast("nucleus", "global", 4.0)), "RB k=4": S.summary(d, S.contrast("nucleus_rb", "global_rb", 4.0)),
            "PV k=2": S.summary(d, S.contrast("nucleus", "global", 2.0)), "PV k=6": S.summary(d, S.contrast("nucleus", "global", 6.0)),
            "RB k=2": S.summary(d, S.contrast("nucleus_rb", "global_rb", 2.0)), "RB k=6": S.summary(d, S.contrast("nucleus_rb", "global_rb", 6.0)),
            "RT PV global": S.summary(d, lambda r: r[("global_rt", 4.0)] - r[("sham_pv_global", 1.0)]),
            "RT PV local": S.summary(d, lambda r: r[("nucleus_rt", 4.0)] - r[("sham_pv_local", 1.0)]),
            "RT RB global": S.summary(d, lambda r: r[("global_rt_rb", 4.0)] - r[("baseline", 1.0)]),
            "RT RB local": S.summary(d, lambda r: r[("nucleus_rt_rb", 4.0)] - r[("baseline", 1.0)])}
fmt = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
pd.DataFrame({m: {k: fmt(v) for k, v in R[m].items()} for m in S.MODELS})
"""),
    code("""
BLUE, GREEN, GRAY = "#0072B2", "#009E73", "#666666"
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
ys = np.arange(4)[::-1]
for off, key, col, mk in [(+0.13, "RB k=4", GREEN, "s"), (-0.13, "PV k=4", BLUE, "o")]:
    for y, m in zip(ys, S.MODELS):
        mu, lo, hi = R[m][key]; axes[0].plot([lo, hi], [y + off] * 2, color=col); axes[0].plot(mu, y + off, mk, color=col, mec="k", mew=.5)
axes[0].set_yticks(ys); axes[0].set_yticklabels(S.MODELS); axes[0].axvline(0, color=GRAY, ls=":"); axes[0].set_title("(a) Local - global WER, k=4")
for off, key, col, mk, fill in [(.21, "RT PV global", BLUE, "o", 1), (.07, "RT PV local", BLUE, "o", 0), (-.07, "RT RB global", GREEN, "s", 1), (-.21, "RT RB local", GREEN, "s", 0)]:
    for y, m in zip(ys, S.MODELS):
        mu, lo, hi = R[m][key]; axes[1].plot([lo, hi], [y + off] * 2, color=col); axes[1].plot(mu, y + off, mk, color=col if fill else "w", mec="k" if fill else col)
axes[1].set_yticks(ys); axes[1].set_yticklabels([]); axes[1].axvline(0, color=GRAY, ls=":"); axes[1].set_title("(b) Round-trip damage")
for m, ls in zip(S.MODELS, [":", "-", "--", "-."]):
    c = [R[m][f"RB k={k}"] for k in (2, 4, 6)]
    axes[2].errorbar([2, 4, 6], [v[0] for v in c], yerr=[[v[0] - v[1] for v in c], [v[2] - v[0] for v in c]], ls=ls, color=GREEN, marker="s", capsize=2, label=m)
axes[2].axhline(0, color=GRAY, ls=":"); axes[2].set_xticks([2, 4, 6]); axes[2].legend(frameon=False); axes[2].set_title("(c) RB dose response")
plt.show()
"""),
    md("## Pre-specified tests (paper Table 2)"),
    code("""
own = ["M0", "M1", "M2"]
dmg = lambda a, k: (lambda r: r[(a, k)] - r[("baseline", 1.0)])
T = {"E1 global PV damage, M1 - M2": S.summary(D["M1"], dmg("global", 4.0), other=D["M2"]),
     "E2 PV contrast, M2 - M1": S.summary(D["M2"], S.contrast("nucleus", "global", 4.0), other=D["M1"]),
     "E4 clean WER, M1 - M2": S.summary(D["M1"], S.arm("baseline", 1.0), other=D["M2"]),
     "RB contrast, M2 - M1 (post hoc)": S.summary(D["M2"], S.contrast("nucleus_rb", "global_rb", 4.0), other=D["M1"]),
     "D2 RB growth k=6 - k=2, M0": S.summary(D["M0"], lambda r: (r[("nucleus_rb", 6.0)] - r[("global_rb", 6.0)]) - (r[("nucleus_rb", 2.0)] - r[("global_rb", 2.0)])),
     "D2 RB growth k=6 - k=2, M1": S.summary(D["M1"], lambda r: (r[("nucleus_rb", 6.0)] - r[("global_rb", 6.0)]) - (r[("nucleus_rb", 2.0)] - r[("global_rb", 2.0)])),
     "D2 RB growth k=6 - k=2, M2": S.summary(D["M2"], lambda r: (r[("nucleus_rb", 6.0)] - r[("global_rb", 6.0)]) - (r[("nucleus_rb", 2.0)] - r[("global_rb", 2.0)]))}
for k, v in T.items(): print(f"{k:34s} {fmt(v)}")
print("E3 RB contrast > 0 in M0-M2:", all(R[m]["RB k=4"][1] > 0 for m in own))
print("D1 RB contrast at k=6 > 0 in M0-M2:", all(R[m]["RB k=6"][1] > 0 for m in own))
print("A1 round-trip correction: not testable (uncorrected PV and RB agree within their intervals)")
"""),
    md("### Membership probe (B1)\nScore = WER increase under global stretching. The same recordings are scored under M2 (heard) and M1 (not heard); recordings counted as heard are those whose reciter's recording of the verse is in the EveryAyah training split."),
    code("""
full = {m: S.load(m, reciters=None) for m in ["M1", "M2"]}
mem = json.loads((ROOT / "data/split_membership.json").read_text())["study"]
members = [(n, v) for n, info in mem.items() for v, s in info["per_verse"].items() if "train" in s]
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
rng = np.random.default_rng(20260921)
for arm_name, k in [("global", 4.0), ("global_rb", 4.0)]:
    keys = [key for key in members if all((arm_name, k) in full[m].get(key, {}) and ("baseline", 1.0) in full[m].get(key, {}) for m in ["M1", "M2"])]
    s = {m: np.array([full[m][key][(arm_name, k)] - full[m][key][("baseline", 1.0)] for key in keys]) for m in ["M1", "M2"]}
    a = auc(-s["M2"], -s["M1"]); boots = []
    for _ in range(2000):
        idx = rng.integers(len(keys), size=len(keys)); boots.append(auc(-s["M2"][idx], -s["M1"][idx]))
    print(f"{arm_name}@{k:g}: n = {len(keys)} recordings, AUC = {a:.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
"""),
    md("## Recogniser-independent round-trip check\nMedian relative spectral error energy of the stretch-and-restore arms against the clean recording (computed from audio with `src/analysis`; audio is not redistributed, so the saved summary is loaded)."),
    code("""
json.loads((ROOT / "results/spectral_rt.json").read_text())
"""),
    md("## Check against the paper\nThe paper's numbers are generated into LaTeX macros. Each recomputed value below is formatted the same way and compared with the macro printed in the paper."),
    code("""
import re
paper = {}
for f in sorted((ROOT / "results/paper_macros").glob("*.tex")):
    paper.update(dict(re.findall(r"\\\\newcommand\\{\\\\(\\w+)\\}\\{([^}]*)\\}", f.read_text())))
s3 = lambda v: f"{v:+.3f}".replace("-", "$-$"); f3 = lambda v: f"{v:.3f}"
tag = {"tarteel": "Tar", "M0": "Mz", "M1": "Mo", "M2": "Mt"}
expected = {}
for m, t in tag.items():
    expected[f"Clean{t}"] = f3(A[m]["Clean"])
    for key, nm in [("PV k=4", "Pv"), ("RB k=4", "Rb")]:
        expected[f"{nm}{t}"], expected[f"{nm}{t}Lo"], expected[f"{nm}{t}Hi"] = map(s3, R[m][key])
    expected[f"RtPvG{t}"], expected[f"RtPvL{t}"] = f3(R[m]["RT PV global"][0]), f3(R[m]["RT PV local"][0])
    expected[f"LocRb{t}"], expected[f"GloRb{t}"], expected[f"Sil{t}"] = f3(A[m]["RB L"]), f3(A[m]["RB G"]), f3(A[m]["Silence"])
    expected[f"PvTwo{t}"] = s3(R[m]["PV k=2"][0])
expected.update({"TestEOne": s3(T["E1 global PV damage, M1 - M2"][0]), "TestETwo": s3(T["E2 PV contrast, M2 - M1"][0]),
                 "TestEFour": s3(T["E4 clean WER, M1 - M2"][0]), "RbTwoMinusOne": s3(T["RB contrast, M2 - M1 (post hoc)"][0])})
ok = {k: paper.get(k) == v for k, v in expected.items()}
print(f"{sum(ok.values())} of {len(ok)} checked values match the paper exactly")
for k, v in ok.items():
    if not v: print("MISMATCH", k, "paper:", paper.get(k), "recomputed:", expected[k])
"""),
]


NOTEBOOKS["05_confirmatory_tests_new_data.ipynb"] = [
    md("""
# 05. Confirmatory tests on new data (Amendment 4)

The main study's RB prediction for M0-M2 was written after M0's result was known. Amendment 4 therefore fixed a
set of predictions **before any of their data existed**, on material and models outside the main study, and the
paper reports them whatever they show. Each prediction is that the RB contrast at $k=4$,
$\\Delta_{\\mathrm{RB}}(4) = \\mathrm{WER(Local)} - \\mathrm{WER(Global)}$ with Rubber Band, is positive
(95% interval above zero), unless stated otherwise.

| Test | New material | Prediction |
|---|---|---|
| F1 | M1 and M2 retrained with data-order seeds 2 and 3 | $\\Delta_{\\mathrm{RB}}(4)>0$ in all six models |
| F2 | three reciters no trained model heard | $\\Delta_{\\mathrm{RB}}(4)>0$ in M0, M1, M2 |
| F3 | the wav2vec2 CTC recogniser (also the aligner) | $\\Delta_{\\mathrm{RB}}(4)>0$; growth from $k=2$ to $k=6$ |
| F4 | Whisper-large-v3, not tuned on the Qur'an | gate: clean WER $\\le 0.50$; then $\\Delta_{\\mathrm{RB}}(4)>0$ |
| F5 | English read speech (LibriSpeech test-clean) | $\\Delta_{\\mathrm{RB}}(4)>0$; RB passes the round-trip check; RB minus PV contrast at $k=2$ positive (the protocol calls this the PV bias) |
| M | existing data | the RB minus PV contrast grows with the PV's artifact imbalance (descriptive) |
"""),
    code(SETUP + """
import hashlib, re, warnings; warnings.filterwarnings("ignore")
proto = (ROOT / "protocols/PROTOCOL_exposure.md").read_bytes()
print("PROTOCOL_exposure.md sha256:", hashlib.sha256(proto).hexdigest())
print("recorded before the runs:    ", (ROOT / "protocols/PROTOCOL_exposure.sha256").read_text().split()[0])
text = proto.decode()
print(text[text.index("## Amendment 4"):][:1500], "...")
fmt = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
verdict = lambda x: "SUPPORTED" if x[1] > 0 else "NOT SUPPORTED"
rb4 = S.contrast("nucleus_rb", "global_rb", 4.0)
"""),
    md("## F2. Reciters no trained model heard\nThe fresh manifest (`data/manifest_fresh.json`, built by `src/evaluation/make_fresh.py`) draws 60 verse IDs common to the three reciters with numpy seed 20260921. The paper's crossed bootstrap runs with its reciter list set to these three."),
    code("""
F2 = {}
for m in ["tarteel", "M0", "M1", "M2"]:
    d = S.load(f"fresh_{m}", reciters=S.FRESH)
    F2[m] = {"rb4": S.summary(d, rb4, reciters=S.FRESH), "clean": S.summary(d, S.arm("baseline", 1.0), reciters=S.FRESH)}
    per = [np.mean([row[("nucleus_rb", 4.0)] - row[("global_rb", 4.0)] for (n, v), row in d.items() if n == r]) for r in S.FRESH]
    tag = "(descriptive)" if m == "tarteel" else verdict(F2[m]["rb4"])
    print(f"{m:8s} n={len(d)}  RB contrast {fmt(F2[m]['rb4'])}  {tag:13s} per reciter", np.round(per, 3))
"""),
    md("## F1. Retraining seeds\nSame recipe, three epochs, same validation subset and selection rule; only the batch order changes. Seed 1 is the original M1/M2. Pre-registered descriptives: the seed spread, and the seed-averaged M2 minus M1 difference against it."),
    code("""
runs = {("M1", 1): "M1", ("M2", 1): "M2", ("M1", 2): "M1_s2", ("M2", 2): "M2_s2", ("M1", 3): "M1_s3", ("M2", 3): "M2_s3"}
F1 = {}
for key, run in runs.items():
    d = S.load(run); F1[key] = (d, S.summary(d, rb4), S.summary(d, S.arm("baseline", 1.0)))
    print(f"{key[0]} seed {key[1]}: RB contrast {fmt(F1[key][1])} {verdict(F1[key][1])}   clean WER {F1[key][2][0]:.3f}")
for v in ("M1", "M2"):
    pts = [F1[(v, s)][1][0] for s in (1, 2, 3)]
    print(f"{v}: seed sd {np.std(pts, ddof=1):.3f}, range {np.ptp(pts):.3f}")
def seed_avg(v, fn):
    acc = collections.defaultdict(list)
    for s in (1, 2, 3):
        for key, row in F1[(v, s)][0].items():
            try: acc[key].append(fn(row))
            except KeyError: pass
    return {k: [float(np.mean(x))] for k, x in acc.items() if len(x) == 3}
for name, fn in [("RB contrast", rb4), ("clean WER", S.arm("baseline", 1.0))]:
    a1, a2 = seed_avg("M1", fn), seed_avg("M2", fn)
    wrap = lambda d: {k: {("x", 0): val[0]} for k, val in d.items()}
    print(f"seed-averaged M2 - M1, {name}: {fmt(S.summary(wrap(a2), lambda r: r[('x', 0)], other=wrap(a1)))}")
"""),
    md("## F3. A CTC recogniser\nThe wav2vec2 CTC model also places the target intervals, so it is not independent of them (stated in the paper). Its scores are identical in every run because the audio is identical; they are read from M0's runs."),
    code("""
ctc = S.merged("M0", field="ctc"); ctc2 = S.merged("M2", field="ctc")
print("CTC rows identical between the M0 and M2 runs:", all(ctc[k] == ctc2[k] for k in ctc))
F3 = {"rb4": S.summary(ctc, rb4),
      "growth": S.summary(ctc, lambda r: (r[("nucleus_rb", 6.0)] - r[("global_rb", 6.0)]) - (r[("nucleus_rb", 2.0)] - r[("global_rb", 2.0)])),
      "pv2": S.summary(ctc, S.contrast("nucleus", "global", 2.0))}
print("F3a RB contrast k=4:", fmt(F3["rb4"]), verdict(F3["rb4"]))
print("F3b growth k=2 to 6:", fmt(F3["growth"]), verdict(F3["growth"]))
print("    PV contrast k=2: ", fmt(F3["pv2"]))
"""),
    md("## F4. Whisper-large-v3\nEvaluated from a real-file copy of the Hugging Face files with `config.json` dtype set to float32 (weights identical, upcast); see `results/large_v3/SWAP.json` and the README."),
    code("""
v3 = S.load("large_v3")
F4 = {"clean": S.summary(v3, S.arm("baseline", 1.0)), "rb4": S.summary(v3, rb4), "pv4": S.summary(v3, S.contrast("nucleus", "global", 4.0))}
print("gate, clean WER <= 0.50:", fmt(F4["clean"]), "PASSED" if F4["clean"][0] <= 0.5 else "FAILED")
print("F4 RB contrast k=4:     ", fmt(F4["rb4"]), verdict(F4["rb4"]))
print("   PV contrast k=4:     ", fmt(F4["pv4"]))
print("per reciter RB contrast:", {r: round(np.mean([row[('nucleus_rb', 4.0)] - row[('global_rb', 4.0)] for (n, v), row in v3.items() if n == r]), 3) for r in S.CONFIRMATORY})
"""),
    md("## F5. English read speech\n300 LibriSpeech test-clean utterances (15 from each of 20 speakers), targets = vowels with primary stress of at least 60 ms in forced alignments. Bootstrap: speakers, then utterances within speaker. Round-trip damage is relative to the clean recording for RB and to the unit-rate PV copy for the PV."),
    code("""
F5 = {}
for rec in ["wav2vec2", "whisper_base"]:
    d = S.load_libri(rec); o = {}
    o["clean"] = S.two_stage(d, S.arm("baseline", 1.0))
    for k in (2.0, 4.0, 6.0):
        o[f"rb{k:g}"] = S.two_stage(d, S.contrast("nucleus_rb", "global_rb", k)); o[f"pv{k:g}"] = S.two_stage(d, S.contrast("nucleus", "global", k))
    o["bias2"] = S.two_stage(d, lambda r: S.contrast("nucleus_rb", "global_rb", 2.0)(r) - S.contrast("nucleus", "global", 2.0)(r))
    for a, ref in [("global_rt", "sham_pv_global"), ("nucleus_rt", "sham_pv_local"), ("global_rt_rb", "baseline"), ("nucleus_rt_rb", "baseline")]:
        o[a] = S.two_stage(d, lambda r, a=a, ref=ref: r[(a, 4.0)] - r[(ref, 1.0)])
    F5[rec] = o
    print(f"== {rec}: clean WER {o['clean'][0]:.3f}")
    print("  F5a RB contrast k=4:", fmt(o["rb4"]), verdict(o["rb4"]), "| k=2", fmt(o["rb2"]), "| k=6", fmt(o["rb6"]))
    print("  F5b RB round trips (upper bound < 0.05):", fmt(o["global_rt_rb"]), fmt(o["nucleus_rt_rb"]),
          "PASSED" if max(o["global_rt_rb"][2], o["nucleus_rt_rb"][2]) < 0.05 else "FAILED")
    print("      PV round trips:", fmt(o["global_rt"]), fmt(o["nucleus_rt"]))
    print("  F5c RB minus PV contrast at k=2:", fmt(o["bias2"]), verdict(o["bias2"]), "| PV contrast k=2", fmt(o["pv2"]))
"""),
    md("## M. Engine difference and the PV's artifact imbalance\nRubber Band is not artifact-free, so the RB minus PV difference is not a direct estimate of PV bias. Across 8 cells (4 models $\\times$ $k\\in\\{2,4\\}$): the between-engine difference $\\Delta_{\\mathrm{RB}}-\\Delta_{\\mathrm{PV}}$ against the PV's artifact imbalance $A_{\\mathrm{PV}}(\\mathrm{Global})-A_{\\mathrm{PV}}(\\mathrm{Local})$."),
    code("""
cells = []
for m in S.MODELS:
    d = S.merged(m)
    for k in (2.0, 4.0):
        bias = S.summary(d, lambda r, k=k: S.contrast("nucleus_rb", "global_rb", k)(r) - S.contrast("nucleus", "global", k)(r))[0]
        imb = S.summary(d, lambda r, k=k: (r[("global_rt", k)] - r[("sham_pv_global", 1.0)]) - (r[("nucleus_rt", k)] - r[("sham_pv_local", 1.0)]))[0]
        cells.append((m, k, imb, bias))
rank = lambda x: np.argsort(np.argsort(x))
x, y = np.array([c[2] for c in cells]), np.array([c[3] for c in cells])
for c in cells: print(f"{c[0]:8s} k={c[1]:g}  imbalance {c[2]:.3f}  RB-PV difference {c[3]:+.3f}")
print("Spearman over 8 cells:", round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 2))
k4 = [c for c in cells if c[1] == 4.0]
print("within k=4:", round(float(np.corrcoef(rank([c[2] for c in k4]), rank([c[3] for c in k4]))[0, 1]), 2))
MECH = (float(np.corrcoef(rank(x), rank(y))[0, 1]), float(np.corrcoef(rank([c[2] for c in k4]), rank([c[3] for c in k4]))[0, 1]))
"""),
    md("## Check against the paper\nEvery Amendment 4 number printed in the paper comes from `results/paper_macros/recast_numbers5.tex`. Each is recomputed above and compared here."),
    code("""
paper = dict(re.findall(r"\\\\newcommand\\{\\\\(\\w+)\\}\\{([^}]*)\\}", (ROOT / "results/paper_macros/recast_numbers5.tex").read_text()))
s3, u3 = (lambda v: f"{v:+.3f}"), (lambda v: f"{v:.3f}")
exp = {}
for m, t in {"tarteel": "Tar", "M0": "Mz", "M1": "Mo", "M2": "Mt"}.items():
    exp[f"FrRb{t}"], exp[f"FrRb{t}Lo"], exp[f"FrRb{t}Hi"] = map(s3, F2[m]["rb4"])
own = [F1[(v, s)][1] for v in ("M1", "M2") for s in (1, 2, 3)]
exp["SeedRbMin"], exp["SeedRbMax"] = s3(min(o[0] for o in own)), s3(max(o[0] for o in own))
exp["CtcRb"], exp["CtcRbLo"], exp["CtcRbHi"] = map(s3, F3["rb4"]); exp["CtcGrowth"], exp["CtcGrowthLo"], exp["CtcGrowthHi"] = map(s3, F3["growth"])
exp["VthClean"] = u3(F4["clean"][0]); exp["VthRb"], exp["VthRbLo"], exp["VthRbHi"] = map(s3, F4["rb4"]); exp["VthPv"] = s3(F4["pv4"][0])
for rec, t in [("wav2vec2", "Wv"), ("whisper_base", "Wb")]:
    o = F5[rec]; exp[f"En{t}Clean"] = u3(o["clean"][0])
    exp[f"En{t}Rb"], exp[f"En{t}RbLo"], exp[f"En{t}RbHi"] = map(s3, o["rb4"]); exp[f"En{t}Bias"] = s3(o["bias2"][0])
    exp[f"En{t}RbTwo"], exp[f"En{t}RbSix"], exp[f"En{t}PvTwo"] = s3(o["rb2"][0]), s3(o["rb6"][0]), s3(o["pv2"][0])
    exp[f"En{t}RtPvG"] = u3(o["global_rt"][0])
exp["EnRtRbHiMax"] = u3(max(F5[r][a][2] for r in F5 for a in ("global_rt_rb", "nucleus_rt_rb")))
exp["MechRho"] = f"{MECH[0]:.2f}"
import itertools
split = [np.corrcoef(list(range(8)), list(p) + [4 + i for i in q])[0, 1] for p in itertools.permutations(range(4)) for q in itertools.permutations(range(4))]
exp["MechRhoSplit"] = f"{np.mean(split):.2f}"
for kk, nm in [(2.0, "Two"), (4.0, "Four")]:
    im = [c[2] for c in cells if c[1] == kk]; exp[f"MechImb{nm}Min"], exp[f"MechImb{nm}Max"] = f"{min(im):.2f}", f"{max(im):.2f}"
# post hoc checks printed in the paper: reciter-level t intervals and the boundary-blended local arm
from scipy import stats as st
def tint(x):
    x = np.array(x); se = x.std(ddof=1) / np.sqrt(len(x)); q = st.t.ppf(.975, len(x) - 1); return (x.mean(), x.mean() - q * se, x.mean() + q * se)
per = lambda d, names: [np.mean([row[("nucleus_rb", 4.0)] - row[("global_rb", 4.0)] for (n, v), row in d.items() if n == r]) for r in names]
exp["TintFrMt"], exp["TintFrMtLo"], exp["TintFrMtHi"] = map(s3, tint(per(S.load("fresh_M2", reciters=S.FRESH), S.FRESH)))
exp["TintVth"], exp["TintVthLo"], exp["TintVthHi"] = map(s3, tint(per(v3, S.CONFIRMATORY)))
blend = lambda d: np.mean([np.mean([row[("nucleus_edge", 4.0)] - row[("nucleus", 4.0)] for (n, v), row in d.items() if n == r]) for r in S.CONFIRMATORY])
exp["BlendMax"] = u3(max(abs(blend(S.load(m))) for m in S.MODELS))
print("post hoc t intervals: unseen M2", fmt(tint(per(S.load("fresh_M2", reciters=S.FRESH), S.FRESH))), "| large-v3", fmt(tint(per(v3, S.CONFIRMATORY))))
ok = {k: paper.get(k) == v for k, v in exp.items()}
print(f"{sum(ok.values())} of {len(ok)} checked values match the paper exactly")
for k, v in ok.items():
    if not v: print("MISMATCH", k, "paper:", paper.get(k), "recomputed:", exp[k])
"""),
]


def build(names):
    NB.mkdir(exist_ok=True)
    for name in names:
        nb = nbf.v4.new_notebook(); nb.cells = NOTEBOOKS[name]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, NB / name)
        print("executed", name)


if __name__ == "__main__":
    build(sys.argv[1:] or list(NOTEBOOKS))
