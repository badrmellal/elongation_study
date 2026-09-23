"""Manifest-first, resumable inference. No audio or historical results overwritten."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from inference import ROOT, Models, digest, score

HERE = Path(__file__).resolve().parent
AUDIO = Path(os.environ.get("QURAN_AUDIO", "/Users/Apple/Desktop/sajlina_train/audio"))
INDEX = ROOT / "release 2" / "data" / "quran_verses.json"
SEED = 20270911
CHAPTERS = {1,36,55,78,112,67,93,94,103,108,110,87,88,91,99,100,101,102,104,105,106,107,109,111,113,114,97,98}
NEW = {"Alafasy": "Alafasy_128kbps", "Sudais": "Abdurrahmaan_As-Sudais_192kbps",
       "Shuraym": "Saood_ash-Shuraym_128kbps", "Dussary": "Yasser_Ad-Dussary_128kbps",
       "Rifai": "Hani_Rifai_192kbps"}
OLD = {"AbdulBasit": "Abdul_Basit_Murattal_192kbps", "Husary": "Husary_64kbps",
       "Minshawy": "Minshawy_Murattal_128kbps"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wav_path(folder, verse):
    s, a = map(int, verse.split(":"))
    return AUDIO / folder / f"{s:03d}{a:03d}.wav"


def sorted_ids(values):
    return sorted(values, key=lambda key: tuple(map(int, key.split(":"))))


def build_manifest(output):
    if output.exists():
        raise FileExistsError(f"Refusing to replace manifest: {output}")
    texts = json.loads(INDEX.read_text())["verse_text"]
    common = []
    for verse in sorted_ids(texts):
        if int(verse.split(":")[0]) not in CHAPTERS:
            continue
        eligible = True
        for folder in NEW.values():
            path = wav_path(folder, verse)
            if not path.is_file():
                eligible = False; break
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"Unexpected audio format: {path}")
            if not 0 < info.duration <= 8:
                eligible = False; break
        if eligible:
            common.append(verse)
    chosen = np.random.default_rng(SEED).choice(common, 60, replace=False).tolist()
    manifest = {"created_utc": now(), "seed": SEED, "protocol_sha256": digest(HERE / "PROTOCOL.md"),
                "index_sha256": digest(INDEX), "candidate_pool": common,
                "confirmation_verses": chosen, "reciters": {}}
    for name, folder in {**OLD, **NEW}.items():
        if name in NEW:
            verses = chosen
        else:
            original = ROOT / "release 2" / "results" / f"exp1_{name}.json"
            rows = json.loads(original.read_text())
            verses = sorted_ids({r["verse"] for r in rows if r["k"] == 1 and r["arm"] == "nucleus"})
        files = []
        for verse in verses:
            path = wav_path(folder, verse)
            info = sf.info(path)
            files.append({"verse": verse, "path": str(path), "sha256": digest(path),
                          "duration_s": info.duration, "reference": texts[verse]})
        manifest["reciters"][name] = {"folder": folder,
            "role": "confirmation" if name in NEW else "exploratory_rerun", "files": files}
    save(output, manifest)
    print(json.dumps({"manifest": str(output), "common_pool": len(common),
                      "candidates": {k: len(v["files"]) for k, v in manifest["reciters"].items()}}, indent=2))


def record(handle, value):
    handle.write(json.dumps({"written_utc": now(), **value}, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def logged_transform(handle, base, k, operation, function):
    try:
        return function()
    except Exception as error:
        record(handle, {**base, "kind": "transform_failure", "k": k,
                        "operation": operation, "exception_type": type(error).__name__,
                        "exception": str(error)})
        raise


def source_hashes():
    paths = [HERE / "inference.py", HERE / "controls.py", Path(__file__),
             ROOT / "release 2" / "src" / "madd.py", ROOT / "release 2" / "src" / "exp1_sweep.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def reference_regions(arm, condition, sr):
    intervals = condition["intervals"]
    if arm == "silence":
        gaps = condition["metadata"]["interval_added_samples"]
        return [(start, end + gap / sr) for (start, end), gap in zip(intervals, gaps)]
    return intervals


def execute(args):
    from controls import make_conditions, sham_conditions
    manifest = json.loads(args.manifest.read_text())
    if digest(HERE / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after manifest: record a new manifest before inference")
    if digest(INDEX) != manifest["index_sha256"]:
        raise ValueError("Reference corpus changed")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "observations.jsonl"
    if log_path.exists() and log_path.stat().st_size and not (args.output / "metadata.json").is_file():
        raise ValueError("Existing observations lack original metadata; refusing to relabel provenance")
    if log_path.exists() and log_path.stat().st_size:
        with log_path.open("rb") as boundary:
            boundary.seek(-1, os.SEEK_END)
            if boundary.read(1) != b"\n":
                raise ValueError("Observation log lacks final newline; refusing unsafe append")
    done, terminal = set(), set()
    if log_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume for this existing experiment")
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "condition":
                done.add((row["reciter"], row["verse"], row["arm"], row["k"]))
            if row["kind"] in {"excluded", "complete"}:
                terminal.add((row["reciter"], row["verse"]))
    models = Models(args.device)
    meta = {"created_utc": now(), "manifest_sha256": digest(args.manifest),
            "source_sha256": source_hashes(), "models": models.metadata(), "platform": platform.platform()}
    meta_path = args.output / "metadata.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ["manifest_sha256", "source_sha256"]:
            if previous[key] != meta[key]:
                raise ValueError(f"Resume would mix different {key}")
        if previous["models"] != meta["models"]:
            raise ValueError("Resume would mix model settings or runtime")
    else:
        save(meta_path, meta)
    names = args.reciters.split(",") if args.reciters else list(manifest["reciters"])
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as handle:
        for name in names:
            group = manifest["reciters"][name]
            for ordinal, item in enumerate(group["files"], 1):
                verse, path, text = item["verse"], Path(item["path"]), item["reference"]
                if (name, verse) in terminal:
                    continue
                if digest(path) != item["sha256"]:
                    raise ValueError(f"Audio changed: {path}")
                x, sr = sf.read(path, dtype="float32")
                base = {"reciter": name, "verse": verse, "role": group["role"]}
                if sr != 16000 or x.ndim != 1 or not np.isfinite(x).all():
                    raise ValueError(f"Invalid waveform: {path}")
                alignment = models.align(x, text, sr)
                if "excluded" in alignment:
                    record(handle, {**base, "kind": "excluded", "reason": alignment["excluded"], "alignment": alignment})
                    print(name, ordinal, verse, "EXCLUDED", alignment["excluded"], flush=True)
                    continue
                occurrences = alignment["occurrences"]
                intervals = [(i["start_sample"] / sr, i["end_sample"] / sr) for i in occurrences]
                record(handle, {**base, "kind": "alignment", **alignment})
                factors = [(1.0, {"baseline": {"audio": x, "intervals": intervals,
                                              "backend": "none", "metadata": {}}})]
                factors.append((1.0, logged_transform(handle, base, 1.0, "PV local/global unity shams",
                                                       lambda: sham_conditions(x, sr, intervals))))
                for k in [2.0, 4.0, 6.0]:
                    # The exact expected target is independent of the ASR outcome.
                    target = len(x) + sum(round((k - 1) * (i["end_sample"] - i["start_sample"])) for i in occurrences)
                    if target / sr > 29:
                        record(handle, {**base, "kind": "factor_excluded", "k": k, "reason": "Matched duration >29 s"})
                        continue
                    conditions = logged_transform(handle, base, k,
                        "PV local/global/silence/edge" + (" and RubberBand local/global" if k == 4 else ""),
                        lambda: make_conditions(x, sr, intervals, k, include_rubberband=(k == 4)))
                    factors.append((k, conditions))
                for k, conditions in factors:
                    for arm, condition in conditions.items():
                        if (name, verse, arm, k) in done:
                            continue
                        y = condition["audio"]
                        tick = time.monotonic()
                        excluded_regions = reference_regions(arm, condition, sr)
                        result = models.whisper(y, text, occurrences, condition["intervals"], sr,
                                                reference_exclusions=excluded_regions)
                        ctc = alignment["ctc"] if arm == "baseline" else score(text, models.ctc_text(models.ctc_logits(y, sr)))
                        record(handle, {**base, "kind": "condition", "arm": arm, "k": k,
                            "backend": condition["backend"], "transform_metadata": condition["metadata"],
                            "audio_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                            "reference_exclusion_intervals": excluded_regions,
                            "samples": len(y), "whisper": result, "ctc": ctc,
                            "elapsed_s": time.monotonic() - tick})
                        done.add((name, verse, arm, k))
                record(handle, {**base, "kind": "complete"})
                terminal.add((name, verse))
                print(f"{name} {ordinal}/{len(group['files'])} {verse}: complete; {len(done)} conditions; elapsed {time.monotonic()-started:.0f}s", flush=True)
    save(args.output / "completion.json", {"finished_utc": now(), "reciters": names,
         "terminal_verses": len(terminal), "conditions": len(done), "manifest_sha256": digest(args.manifest)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["manifest", "run"])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest_v2.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--reciters")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build_manifest(args.manifest) if args.mode == "manifest" else execute(args)


if __name__ == "__main__":
    main()
