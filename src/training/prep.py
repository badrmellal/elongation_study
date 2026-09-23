"""Decode usable rows of each downloaded parquet file to 16 kHz int16 audio.
Per file writes PREP/<stem>.npy (concatenated int16) and PREP/<stem>.json (index).
Rows are filtered by the reciter COLUMN (not by file), so mixed files are safe."""
import io, json, sys, concurrent.futures as cf
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
from common import DATA, PREP, EXCLUDED, MIN_S, MAX_S, ROOT

def decode(b):
    try:
        y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        y = y.mean(1)
    except Exception:
        y, sr = librosa.load(io.BytesIO(b), sr=None, mono=True)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.clip(y * 32767, -32768, 32767).astype(np.int16)

def one(path):
    out = PREP / (path.parent.name + "__" + path.stem)
    if out.with_suffix(".json").exists():
        return path.name, "cached"
    pf = pq.ParquetFile(path)
    chunks, index, pos, kept, dropped = [], [], 0, 0, 0
    for batch in pf.iter_batches(batch_size=256, columns=["audio", "duration", "text", "reciter"]):
        d = batch.to_pydict()
        for a, dur, text, rec in zip(d["audio"], d["duration"], d["text"], d["reciter"]):
            if rec in EXCLUDED or not (MIN_S <= dur <= MAX_S):
                continue
            try:
                y = decode(a["bytes"])
            except Exception:
                dropped += 1; continue
            if not (MIN_S * 16000 <= len(y) <= MAX_S * 16000):
                dropped += 1; continue
            chunks.append(y); index.append([pos, len(y), text, rec]); pos += len(y); kept += 1
    np.save(out.with_suffix(".npy"), np.concatenate(chunks) if chunks else np.zeros(0, np.int16))
    json.dump({"source": path.name, "rows": index, "dropped_decode": dropped}, open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False)
    return path.name, f"kept {kept} dropped {dropped}"

if __name__ == "__main__":
    PREP.mkdir(parents=True, exist_ok=True)
    sizes = {k.split("/")[-1]: v["size"] for k, v in json.load(open(ROOT / "census_current.json"))["files"].items()}
    present = sorted(DATA.rglob("*.parquet"))
    files = [f for f in present if f.stat().st_size == sizes.get(f.name)]
    print(len(present), "parquet present,", len(files), "complete", flush=True)
    with cf.ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as ex:
        for name, msg in ex.map(one, files):
            print(name, msg, flush=True)
    print("PREP DONE", flush=True)
