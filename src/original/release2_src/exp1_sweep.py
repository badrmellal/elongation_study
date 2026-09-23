"""
EXPERIMENT 1: the three matched arms, synthetic sweep.

Base audio is MURATTAL (short natural madd), so stretching moves us across the
natural mujawwad range rather than beyond anything ever recited.

PREDICTIONS, from the measured law A_in ~ L^beta with beta ~ 0.1-0.3:
  arm 1 nucleus : L grows, budget near-fixed -> enrichment falls as L^(beta-1)
  arm 2 global  : every segment scales together, relative structure preserved
                  -> enrichment FLAT in k
  arm 3 silence : voiced L unchanged -> enrichment FLAT in k

Falsification criterion: if arm 2 falls like arm 1, the effect is global tempo
(Katkov) and the dissociation fails. The prediction was fixed in advance and the
outcome is reported either way.

Madd positions in the manipulated audio are taken from the manipulation itself
(nuclei_out), NOT by re-aligning. Re-aligning would confound the measurement with
the CTC model's own behaviour on stretched audio.

Writes results/exp1_<tag>.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
import torch

from madd import forced_align, madd_intervals
from manipulations import arm_nucleus, arm_global, arm_silence
from config import IDX_PATH, AUDIO, W2V, WHI, results_path# noqa: E402

BASE_STYLE = "Abdul_Basit_Murattal_192kbps"
WFRAME = 0.02
KS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # overridable via --ks


def norm_ar(s):
    import re
    s = re.sub(r"[ً-ْٰٓـ]", "", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    return " ".join(s.split())


def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def attn_on(whp, whm, audio, sr, text, nuclei, decode=False):
    """Attention mass + enrichment on given intervals; optionally free-decode."""
    dur = len(audio) / sr
    if dur > 29:
        return None
    feats = whp(audio, sampling_rate=sr, return_tensors="pt").input_features
    lab = whp.tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        o = whm(input_features=feats, decoder_input_ids=lab, output_attentions=True)
    A = torch.stack([t[0].mean(0) for t in o.cross_attentions])
    n_valid = min(A.shape[-1], int(np.ceil(dur / WFRAME)))
    deep = list(range(A.shape[0]))[-2:]

    rec = []
    for (s, e) in nuclei:
        i0, i1 = int(s / WFRAME), min(int(e / WFRAME), n_valid)
        if i1 - i0 < 3:
            continue
        frac = (i1 - i0) / n_valid
        for li in deep:
            al = A[li, :, :n_valid].numpy()
            al = al / (al.sum(1, keepdims=True) + 1e-12)
            u = int(np.argmax(al[:, i0:i1].sum(1)))
            A_in = float(al[u, i0:i1].sum())
            rec.append({"layer": li, "L_ms": (e - s) * 1000, "frac": frac,
                        "A_in": A_in, "enrich": A_in / max(frac, 1e-9)})
    wer = None
    if decode:
        with torch.no_grad():
            g = whm.generate(feats, max_new_tokens=180)
        hyp = norm_ar(whp.tokenizer.decode(g[0], skip_special_tokens=True)).split()
        ref = norm_ar(text).split()
        wer = lev(ref, hyp) / max(len(ref), 1)
    return rec, wer


def main(n_verses=60, seed=0, decode=True, base_style=None):
    from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor,
                              WhisperForConditionalGeneration, WhisperProcessor)
    w2p = Wav2Vec2Processor.from_pretrained(W2V)
    w2m = Wav2Vec2ForCTC.from_pretrained(W2V).eval()
    whp = WhisperProcessor.from_pretrained(WHI)
    whm = WhisperForConditionalGeneration.from_pretrained(
        WHI, attn_implementation="eager").eval()
    blank = w2m.config.pad_token_id
    vocab = w2p.tokenizer.get_vocab()
    IDX = json.load(open(IDX_PATH))["verse_text"]

    keys = [k for k in IDX if k.split(":")[0] in
            ("1","36","55","78","112","67","93","94","103","108","110","87","88","91",
             "99","100","101","102","104","105","106","107","109","111","113","114","97","98")]
    # PRE-FILTER on base duration BEFORE sampling. Without this, slower reciters
    # (Husary median 11.9 s vs Abdul Basit 7.3 s) lose most of their sample to the
    # 8 s cap and end up underpowered relative to the others.
    ok = []
    for v in keys:
        ss, aa = v.split(":")
        f = f"{AUDIO}/{base_style or BASE_STYLE}/{int(ss):03d}{int(aa):03d}.wav"
        if not os.path.exists(f):
            continue
        try:
            i = sf.info(f)
            if i.frames / i.samplerate <= 8.0:
                ok.append(v)
        except Exception:
            pass
    n = min(n_verses, len(ok))
    keys = list(np.random.default_rng(seed).choice(ok, n, replace=False))

    out = []
    for v in keys:
        s, a = v.split(":")
        wav = f"{AUDIO}/{base_style or BASE_STYLE}/{int(s):03d}{int(a):03d}.wav"
        if not os.path.exists(wav):
            continue
        text = IDX[v]
        x, sr = sf.read(wav); x = np.asarray(x, np.float32)
        if len(x) / sr > 8:            # leave headroom for 6x nucleus stretch
            continue
        ids, ioc = [], []
        for ch in text:
            tk = "|" if ch == " " else ch
            if tk in vocab:
                ioc.append(len(ids)); ids.append(vocab[tk])
            else:
                ioc.append(-1)
        if not ids:
            continue
        iv = w2p(x, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            lg = w2m(iv).logits[0]
        logp = torch.log_softmax(lg, -1).numpy().astype(np.float64)
        if logp.shape[0] < len(ids):
            continue
        fs = (len(x) / sr) / logp.shape[0]
        _, f2t = forced_align(logp, ids, blank=blank)
        mm = madd_intervals(text, ids, ioc, f2t, fs, min_counts=2)
        nuclei = [(m["start"], m["end"]) for m in mm if m["dur_ms"] >= 60]
        if len(nuclei) < 2:
            continue

        for k in KS:
            for name, fn in (("nucleus", arm_nucleus), ("global", arm_global),
                             ("silence", arm_silence)):
                man = fn(x, sr, nuclei, k)
                r = attn_on(whp, whm, man.audio.astype(np.float32), sr, text,
                            man.nuclei_out, decode=decode)
                if r is None:
                    continue
                rec, wer = r
                for e in rec:
                    out.append({**e, "verse": v, "k": k, "arm": name,
                                "total_s": man.total_dur_s, "wer": wer})
    return out


def report(rows):
    print(f"\n{'='*76}\nEXPERIMENT 1  n_obs={len(rows)}  verses={len({r['verse'] for r in rows})}")
    print(f"{'='*76}")
    print(f"  {'arm':>8} {'k':>5} {'median L(ms)':>12} {'enrich':>8} {'A_in':>7} {'WER':>6} {'total_s':>8}")
    for arm in ("nucleus", "global", "silence"):
        for k in KS:
            R = [r for r in rows if r["arm"] == arm and r["k"] == k]
            if not R:
                continue
            w = [r["wer"] for r in R if r["wer"] is not None]
            print(f"  {arm:>8} {k:>5.1f} {np.median([r['L_ms'] for r in R]):>12.0f} "
                  f"{np.median([r['enrich'] for r in R]):>8.2f} {np.median([r['A_in'] for r in R]):>7.3f} "
                  f"{(np.median(w) if w else float('nan')):>6.2f} {np.median([r['total_s'] for r in R]):>8.2f}")
        print()
    print("  SLOPES  d log(enrich) / d log(k):")
    for arm in ("nucleus", "global", "silence"):
        R = [r for r in rows if r["arm"] == arm]
        if len(R) < 10:
            continue
        x = np.log([r["k"] for r in R]); y = np.log([r["enrich"] + 1e-9 for r in R])
        n = len(x); b, a = np.polyfit(x, y, 1); res = y - (a + b * x)
        se = np.sqrt((res @ res) / (n - 2) / ((x - x.mean()) @ (x - x.mean())))
        lo, hi = b - 1.96 * se, b + 1.96 * se
        v = "FALLS" if hi < -0.05 else ("flat" if lo > -0.05 else "?")
        print(f"    {arm:>8}: {b:+.3f}  [{lo:+.3f},{hi:+.3f}]   {v}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default=BASE_STYLE)
    ap.add_argument("--tag", default="AbdulBasit")
    ap.add_argument("--ks", default="")
    a = ap.parse_args()
    if a.ks:
        KS[:] = [float(z) for z in a.ks.split(",")]
    rows = main(base_style=a.style)
    for r in rows:
        r["base_style"] = a.style
    report(rows)
    json.dump(rows, open(results_path(f"exp1_{a.tag}.json"), "w"))
    print(f"n_obs={len(rows)} verses={len({r['verse'] for r in rows})} -> results/exp1_{a.tag}.json")
