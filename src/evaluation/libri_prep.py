"""Amendment 4 F5: select LibriSpeech test-clean utterances and stressed-vowel targets (MFA alignments).
20 speakers x 15 utterances (numpy seed 20260921); utterances <= 8 s with >= 2 primary-stress vowels >= 60 ms."""
import io, json, pathlib, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
SRC = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(r"C:\phon\libri"); (OUT / "wav").mkdir(parents=True, exist_ok=True)
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
t = pq.read_table(SRC).to_pylist()
elig = {}
for r in t:
    y_bytes = r["audio"]["bytes"]
    info = sf.info(io.BytesIO(y_bytes)); dur = info.frames / info.samplerate
    if dur > 8.0: continue
    tg = [(p["start"], p["end"]) for p in r["phonemes"] if p["phoneme"][:-1] in VOWELS and p["phoneme"].endswith("1") and p["end"] - p["start"] >= 0.06]
    if len(tg) >= 2:
        elig.setdefault(r["id"].split("-")[0], []).append((r, tg, dur))
rng = np.random.default_rng(20260921)
spk = sorted(s for s, v in elig.items() if len(v) >= 15)
chosen = [spk[i] for i in rng.permutation(len(spk))[:20]]
items = []
for s in sorted(chosen):
    pool = sorted(elig[s], key=lambda z: z[0]["id"])
    for j in rng.permutation(len(pool))[:15]:
        r, tg, dur = pool[j]
        y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True); y = y.mean(1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        p = OUT / "wav" / f"{r['id']}.wav"; sf.write(p, y, 16000, subtype="FLOAT")
        items.append({"id": r["id"], "speaker": s, "sex": r.get("sex"), "path": str(p), "reference": r["transcript"],
                      "intervals": [[a, b] for a, b in tg], "duration_s": len(y) / 16000})
json.dump({"seed": 20260921, "source": "gilkeyio/librispeech-alignments test_clean (MFA)", "speakers": sorted(chosen), "items": items},
          open(OUT / "libri_manifest.json", "w", encoding="utf-8"), indent=1)
print("eligible speakers", len(spk), "| chosen", len(chosen), "| utterances", len(items),
      "| targets/utt median", float(np.median([len(i["intervals"]) for i in items])))
