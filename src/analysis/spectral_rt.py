"""ASR-independent round-trip check: relative spectral error energy of stretch-and-restore arms
(k=4) against the clean recording, for PV and RB, over the 210 confirmatory recordings."""
import json, sys, warnings
import numpy as np, soundfile as sf, librosa
warnings.filterwarnings("ignore")
sys.path.insert(0, "confirmation"); sys.path.insert(0, "exposure/ws")
import extra_arms
P = ["Alafasy", "Sudais", "Shuraym", "Dussary", "Rifai"]
man = json.load(open("confirmation/manifest_v2.json"))
paths = {(n, it["verse"]): it["path"] for n, g in man["reciters"].items() for it in g["files"]}
align = {}
for l in open("exposure/results/tarteel/observations.jsonl"):
    r = json.loads(l)
    if r["kind"] == "alignment" and r["reciter"] in P:
        align[(r["reciter"], r["verse"])] = r["occurrences"]
def err(y, ref):
    Y = np.abs(librosa.stft(y, n_fft=512, hop_length=128)); R = np.abs(librosa.stft(ref, n_fft=512, hop_length=128))
    n = min(Y.shape[1], R.shape[1]); return float(np.sum((Y[:, :n] - R[:, :n]) ** 2) / np.sum(R[:, :n] ** 2))
out = {a: [] for a in ["global_rt", "nucleus_rt", "global_rt_rb", "nucleus_rt_rb"]}
for i, (key, occ) in enumerate(sorted(align.items())):
    x, sr = sf.read(paths[key], dtype="float32")
    iv = [(o["start_sample"] / sr, o["end_sample"] / sr) for o in occ]
    c = extra_arms.roundtrip_conditions(x, sr, iv, 4.0, include_rubberband=True)
    for a in out: out[a].append(err(c[a]["audio"], x))
res = {a: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "n": len(v)} for a, v in out.items()}
json.dump(res, open("exposure/spectral_rt.json", "w"), indent=1); print(json.dumps(res, indent=1))
