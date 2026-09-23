"""Amendment 4 F5: LibriSpeech arms with the paper's stimulus code; recognisers wav2vec2-base-960h and whisper-base."""
import json, pathlib, re, sys, time, hashlib
import numpy as np, soundfile as sf, torch
EVAL = pathlib.Path(r"C:\phon\eval")
sys.path.insert(0, str(EVAL / "phonation_study" / "confirmation")); sys.path.insert(0, r"C:\phon\code")
import controls, extra_arms  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402
controls.RUBBERBAND_PATH = r"C:\phon\tools\rb\rubberband-4.0.0-gpl-executable-windows\rubberband.exe"
OUT = pathlib.Path(r"C:\phon\results\libri"); OUT.mkdir(parents=True, exist_ok=True)
man = json.load(open(r"C:\phon\libri\libri_manifest.json", encoding="utf-8"))
dev = "cuda"
w2p = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"); w2m = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().to(dev)
wp = WhisperProcessor.from_pretrained("openai/whisper-base"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").eval().to(dev)
def norm(s): return " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split())
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
@torch.inference_mode()
def rec(y):
    iv = w2p(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
    h1 = w2p.batch_decode(w2m(iv).logits.argmax(-1))[0]
    f = wp(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
    g = wm.generate(input_features=f, language="en", task="transcribe", max_new_tokens=180, num_beams=1, do_sample=False)
    h2 = wp.batch_decode(g, skip_special_tokens=True)[0]
    return {"wav2vec2": h1, "whisper_base": h2}
# resume: keep only rows of items recorded as finished, skip those items
DONE = OUT / "done_ids.txt"; done = set(DONE.read_text().split()) if DONE.exists() else set()
if (OUT / "observations.jsonl").exists():
    keep = [l for l in open(OUT / "observations.jsonl", encoding="utf-8") if l.endswith("\n") and json.loads(l)["id"] in done]
    open(OUT / "observations.jsonl", "w", encoding="utf-8").writelines(keep)
log = open(OUT / "observations.jsonl", "a", encoding="utf-8"); t0 = time.time(); n = 0
for idx, it in enumerate(man["items"], 1):
    if it["id"] in done: continue
    x, sr = sf.read(it["path"], dtype="float32"); iv = [tuple(v) for v in it["intervals"]]
    conds = {("baseline", 1.0): x}
    for arm, c in controls.sham_conditions(x, sr, iv).items(): conds[(arm, 1.0)] = c["audio"]
    for k in (2.0, 4.0, 6.0):
        for arm, c in controls.make_conditions(x, sr, iv, k, include_rubberband=True).items(): conds[(arm, k)] = c["audio"]
    for arm, c in extra_arms.roundtrip_conditions(x, sr, iv, 4.0, include_rubberband=True).items(): conds[(arm, 4.0)] = c["audio"]
    ref = norm(it["reference"]).split()
    for (arm, k), y in conds.items():
        hyp = rec(y)
        row = {"id": it["id"], "speaker": it["speaker"], "arm": arm, "k": k, "samples": len(y)}
        for mname, h in hyp.items():
            hn = norm(h).split(); row[mname] = {"hypothesis": h, "wer": lev(ref, hn) / len(ref)}
        log.write(json.dumps(row) + "\n"); n += 1
    log.flush(); open(DONE, "a").write(it["id"] + "\n")
    print(f"{idx}/{len(man['items'])} {it['id']} {n} conditions {time.time()-t0:.0f}s", flush=True)
print("LIBRI DONE", n, flush=True)
