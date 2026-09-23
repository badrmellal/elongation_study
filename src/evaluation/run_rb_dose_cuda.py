"""Rubber Band dose-response (exposure/PROTOCOL.md Amendment 3): RB Local/Global and RB round trips
at k=2 and k=6, schema-compatible with the paper's runner. Paper code imported unchanged."""
import argparse, hashlib, json, pathlib, subprocess, sys, time
EVAL = pathlib.Path(r"C:\phon\eval")
sys.path.insert(0, str(EVAL / "phonation_study" / "confirmation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import soundfile as sf  # noqa: E402
import inference, controls, extra_arms  # noqa: E402
from controls import make_conditions  # noqa: E402
from inference import score  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
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
log = (out / "observations.jsonl").open("a", encoding="utf-8"); t0 = time.monotonic(); n = 0
for name, group in manifest["reciters"].items():
    for ordinal, item in enumerate(group["files"], 1):
        verse, text = item["verse"], item["reference"]
        x, sr = sf.read(pathlib.Path(str(item["path"]).replace("/Users/Apple/Desktop", str(EVAL))), dtype="float32")
        alignment = models.align(x, text, sr)
        if "excluded" in alignment:
            continue
        occ = alignment["occurrences"]; intervals = [(i["start_sample"] / sr, i["end_sample"] / sr) for i in occ]
        for k in (2.0, 6.0):
            if (len(x) + sum(round((k - 1) * (i["end_sample"] - i["start_sample"])) for i in occ)) / sr > 29:
                continue
            c = make_conditions(x, sr, intervals, k, include_rubberband=True)
            rt = extra_arms.roundtrip_conditions(x, sr, intervals, k, include_rubberband=True)
            arms = {"nucleus_rb": c["nucleus_rb"], "global_rb": c["global_rb"], "nucleus_rt_rb": rt["nucleus_rt_rb"], "global_rt_rb": rt["global_rt_rb"]}
            for arm, cond in arms.items():
                y = cond["audio"]
                excl = cond["intervals"]
                res = models.whisper(y, text, occ, cond["intervals"], sr, reference_exclusions=excl)
                log.write(json.dumps({"reciter": name, "verse": verse, "role": group["role"], "kind": "condition", "arm": arm, "k": k,
                    "backend": cond["backend"], "audio_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "samples": len(y),
                    "whisper": res, "ctc": score(text, models.ctc_text(models.ctc_logits(y, sr)))}, ensure_ascii=False) + "\n")
                log.flush(); n += 1
        print(f"{name} {ordinal}: {n} conditions {time.monotonic()-t0:.0f}s", flush=True)
(out / "completion.json").write_text(json.dumps({"conditions": n}, indent=1)); print("RB DOSE DONE", n, flush=True)
