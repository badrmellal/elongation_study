"""Relocate source paths without changing the frozen sample or audio hashes."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--audio-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
source = Path(__file__).parent / "confirmation" / "manifest_v2.json"
manifest = json.loads(source.read_text())
for group in manifest["reciters"].values():
    for item in group["files"]:
        chapter, verse = map(int, item["verse"].split(":"))
        path = a.audio_root.resolve() / group["folder"] / f"{chapter:03d}{verse:03d}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        item["path"] = str(path)
with a.output.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"Verified and relocated manifest: {a.output}")
