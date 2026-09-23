"""Download the parquet files that contain usable (non-excluded) rows, pinned to the census revision.
Train files in descending usable-row order, plus the 3 validation files with most usable rows."""
import json, time, concurrent.futures as cf
from huggingface_hub import hf_hub_download
from common import ROOT, DATA

c = json.load(open(ROOT / "census_current.json"))
rev, files = c["revision"], c["files"]
train = sorted((k for k, v in files.items() if "train-" in k and v["usable_rows"] > 0), key=lambda k: -files[k]["usable_rows"])
val = sorted((k for k, v in files.items() if "validation-" in k and v["usable_rows"] > 0), key=lambda k: -files[k]["usable_rows"])[:3]
todo = val + train
print(f"revision {rev}: {len(todo)} files, {sum(files[k]['size'] for k in todo)/1e9:.1f} GB", flush=True)
t0, done_bytes = time.time(), 0

def get(name):
    for attempt in range(5):
        try:
            return name, hf_hub_download("tarteel-ai/everyayah", name, repo_type="dataset", revision=rev, local_dir=DATA)
        except Exception as e:  # network hiccup: retry
            print("retry", name, attempt, repr(e)[:200], flush=True); time.sleep(10 * (attempt + 1))
    raise RuntimeError(name)

with cf.ThreadPoolExecutor(6) as ex:
    for i, (name, path) in enumerate(ex.map(get, todo), 1):
        done_bytes += files[name]["size"]
        el = time.time() - t0
        print(f"{i}/{len(todo)} {name} {done_bytes/1e9:.1f} GB {done_bytes/1e6/el:.1f} MB/s", flush=True)
print("DOWNLOAD DONE", flush=True)
