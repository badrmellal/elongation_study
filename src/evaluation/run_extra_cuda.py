"""Round-trip artifact arms on SERVER-AI (CUDA), schema-compatible with the paper's runner.
Arms: nucleus_rt / global_rt (PV) at k=2 and 4, plus nucleus_rt_rb / global_rt_rb at k=4."""
import argparse, hashlib, json, pathlib, subprocess, sys, time
EVAL = pathlib.Path(r"C:\phon\eval")
sys.path.insert(0, str(EVAL / "phonation_study" / "confirmation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, soundfile as sf  # noqa: E402
import inference, controls, run_confirmation, extra_arms  # noqa: E402
from inference import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reciters")
a = ap.parse_args()
out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
real = EVAL / "modelcache"; fake = out / "hfcache"
for link, target in {fake / "models--tarteel-ai--whisper-base-ar-quran" / "snapshots" / inference.WHISPER_REV: pathlib.Path(a.ckpt).resolve(),
                     fake / "models--rabah2026--wav2vec2-large-xlsr-53-arabic-quran-v2" / "snapshots" / inference.CTC_REV:
                     real / "models--rabah2026--wav2vec2-large-xlsr-53-arabic-quran-v2" / "snapshots" / inference.CTC_REV}.items():
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
inference.CACHE = fake
controls.RUBBERBAND_PATH = r"C:\phon\tools\rb\rubberband-4.0.0-gpl-executable-windows\rubberband.exe"

manifest = json.loads((EVAL / "phonation_study" / "confirmation" / "manifest_v2.json").read_text())
models = inference.Models("cuda")
ck = pathlib.Path(a.ckpt).resolve()
(out / "SWAP.json").write_text(json.dumps({"device": "cuda", "whisper_weights_dir": str(ck), "arms": "roundtrip",
    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ck.iterdir()) if p.is_file()}}, indent=1))
log = (out / "observations.jsonl").open("a", encoding="utf-8")
names = a.reciters.split(",") if a.reciters else list(manifest["reciters"])
t0 = time.monotonic(); n = 0
for name in names:
    group = manifest["reciters"][name]
    for ordinal, item in enumerate(group["files"], 1):
        verse, text = item["verse"], item["reference"]
        path = pathlib.Path(str(item["path"]).replace("/Users/Apple/Desktop", str(EVAL)))
        x, sr = sf.read(path, dtype="float32")
        alignment = models.align(x, text, sr)
        if "excluded" in alignment:
            continue
        occurrences = alignment["occurrences"]
        intervals = [(i["start_sample"] / sr, i["end_sample"] / sr) for i in occurrences]
        for k in (2.0, 4.0):
            added = sum(round((k - 1) * (i["end_sample"] - i["start_sample"])) for i in occurrences)
            if (len(x) + added) / sr > 29:
                continue
            conditions = extra_arms.roundtrip_conditions(x, sr, intervals, k, include_rubberband=(k == 4.0))
            for arm, condition in conditions.items():
                y = condition["audio"]
                result = models.whisper(y, text, occurrences, condition["intervals"], sr,
                                        reference_exclusions=condition["intervals"])
                ctc = score(text, models.ctc_text(models.ctc_logits(y, sr)))
                log.write(json.dumps({"reciter": name, "verse": verse, "role": group["role"], "kind": "condition",
                    "arm": arm, "k": k, "backend": condition["backend"],
                    "audio_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "samples": len(y),
                    "whisper": result, "ctc": ctc}, ensure_ascii=False) + "\n")
                log.flush(); n += 1
        print(f"{name} {ordinal}/{len(group['files'])} {verse}: {n} conditions, {time.monotonic()-t0:.0f}s", flush=True)
log.close()
(out / "completion.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "reciters": names, "conditions": n}, indent=1))
print("EXTRA DONE", n, flush=True)
