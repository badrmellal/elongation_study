"""Census of the CURRENT tarteel-ai/everyayah revision: reciter counts per parquet file.
Reads metadata columns only (range requests). Writes C:/phon/census_current.json."""
import collections, concurrent.futures as cf, json
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from common import EXCLUDED, MIN_S, MAX_S, ROOT

info = HfApi().dataset_info("tarteel-ai/everyayah", files_metadata=True)
rev = info.sha
files = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".parquet")}
fs = HfFileSystem()

def one(name):
    with fs.open(f"datasets/tarteel-ai/everyayah@{rev}/{name}", "rb") as fh:
        t = pq.ParquetFile(fh).read(columns=["reciter", "duration"]).to_pydict()
    counts = collections.Counter(t["reciter"])
    usable = sum(1 for r, d in zip(t["reciter"], t["duration"]) if r not in EXCLUDED and MIN_S <= d <= MAX_S)
    return name, {"size": files[name], "rows": len(t["reciter"]), "reciters": dict(counts), "usable_rows": usable}

with cf.ThreadPoolExecutor(16) as ex:
    out = dict(ex.map(one, sorted(files)))
json.dump({"revision": rev, "last_modified": str(info.last_modified), "files": out}, open(ROOT / "census_current.json", "w"), indent=1)
labels = sorted({r for v in out.values() for r in v["reciters"]})
print("revision", rev, "modified", info.last_modified)
print("reciter labels", len(labels), "excluded present:", sorted(set(labels) & EXCLUDED), "unknown excluded names:", sorted(EXCLUDED - set(labels)))
for split in ("train", "validation", "test"):
    sel = {k: v for k, v in out.items() if f"/{split}-" in k or k.split("/")[-1].startswith(split)}
    use = {k: v for k, v in sel.items() if v["usable_rows"] > 0}
    print(split, "files", len(sel), "with usable rows", len(use), "usable rows", sum(v["usable_rows"] for v in use.values()),
          "GB to download", round(sum(v["size"] for v in use.values()) / 1e9, 1))
