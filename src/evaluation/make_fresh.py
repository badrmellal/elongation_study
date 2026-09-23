"""Amendment 4 F2: fresh-reciter manifest. 60 verse IDs common to the three held-out reciters, drawn from the
paper's 214-verse candidate pool with numpy seed 20260921; audio decoded from EveryAyah (revision 6ea5108)."""
import io, json, hashlib, re, pathlib, datetime
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
EVAL = pathlib.Path(r"C:\phon\eval"); DATA = pathlib.Path(r"C:\phon\everyayah")
FRESH = {"sahl_yassin": ("SahlYassin", "Sahl_Yassin_EA"), "akram_alalaqimy": ("AkramAlalaqimy", "Akram_AlAlaqimy_EA"),
         "muhsin_al_qasim": ("MuhsinAlQasim", "Muhsin_AlQasim_EA")}
m2 = json.loads((EVAL / "phonation_study/confirmation/manifest_v2.json").read_text(encoding="utf-8"))
verses = json.loads((EVAL / "phonation_study/release 2/data/quran_verses.json").read_text(encoding="utf-8"))["verse_text"]
def norm_ar(s):
    s = re.sub(r"[ً-ْٰٓـ]", "", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    return " ".join(re.sub(r"[ۖۗۚۛۜ۞]", " ", s).split())
pool = m2["candidate_pool"]
bytext = {}
for v in pool: bytext.setdefault(norm_ar(verses[v]), []).append(v)
unique = {t: vs[0] for t, vs in bytext.items() if len(vs) == 1}
census = json.loads(pathlib.Path(r"C:\phon\census_current.json").read_text())
files = [k for k, f in census["files"].items() if set(f["reciters"]) & set(FRESH)]
found = {r: {} for r in FRESH}
for name in files:
    path = DATA / name
    if not path.exists(): continue
    for b in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["audio", "duration", "text", "reciter"]):
        d = b.to_pydict()
        for a, dur, text, rec in zip(d["audio"], d["duration"], d["text"], d["reciter"]):
            if rec in FRESH and dur <= 8.0:
                v = unique.get(norm_ar(text))
                if v and v not in found[rec]: found[rec][v] = a["bytes"]
common = [v for v in pool if all(v in found[r] for r in FRESH)]
order = [pool[i] for i in np.random.default_rng(20260921).permutation(len(pool))]
chosen = [v for v in order if v in set(common)][:60]
print("files read", len(files), "| per reciter found", {r: len(v) for r, v in found.items()}, "| common", len(common), "| chosen", len(chosen))
man = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seed": 20260921,
       "protocol_sha256": m2["protocol_sha256"], "index_sha256": m2["index_sha256"], "candidate_pool": pool,
       "confirmation_verses": chosen, "source": "tarteel-ai/everyayah revision 6ea5108 (train split)", "reciters": {}}
for rec, (name, folder) in FRESH.items():
    out_dir = EVAL / "sajlina_train" / "audio" / folder; out_dir.mkdir(parents=True, exist_ok=True); items = []
    for v in chosen:
        y, sr = sf.read(io.BytesIO(found[rec][v]), dtype="float32", always_2d=True); y = y.mean(1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        s, a = v.split(":"); fn = f"{int(s):03d}{int(a):03d}.wav"; p = out_dir / fn
        sf.write(p, y, 16000, subtype="FLOAT")
        items.append({"verse": v, "path": f"/Users/Apple/Desktop/sajlina_train/audio/{folder}/{fn}",
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "duration_s": len(y) / 16000, "reference": verses[v]})
    man["reciters"][name] = {"folder": folder, "role": "fresh_confirmatory", "files": items}
(EVAL / "phonation_study/confirmation/manifest_fresh.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("manifest written:", {n: len(g["files"]) for n, g in man["reciters"].items()})
