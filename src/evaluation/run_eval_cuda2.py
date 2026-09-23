"""(v2: --manifest) Paper's confirmation pipeline on SERVER-AI (CUDA). The paper's code is imported unchanged; the
only substitutions are: device=cuda, audio path prefix (Mac -> C:/phon/eval), model cache root,
the Whisper weights directory (--ckpt), and the Rubber Band 4.0.0 Windows CLI path."""
import argparse, hashlib, json, pathlib, subprocess, sys
EVAL = pathlib.Path(r"C:\phon\eval")
sys.path.insert(0, str(EVAL / "phonation_study" / "confirmation"))
import inference, controls, run_confirmation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="Whisper weights dir")
ap.add_argument("--out", required=True)
ap.add_argument("--reciters")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--manifest", default="manifest_v2.json", help="file name in confirmation/")
a = ap.parse_args()
out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
real = EVAL / "modelcache"
fake = out / "hfcache"
links = {fake / "models--tarteel-ai--whisper-base-ar-quran" / "snapshots" / inference.WHISPER_REV: pathlib.Path(a.ckpt).resolve(),
         fake / "models--rabah2026--wav2vec2-large-xlsr-53-arabic-quran-v2" / "snapshots" / inference.CTC_REV:
         real / "models--rabah2026--wav2vec2-large-xlsr-53-arabic-quran-v2" / "snapshots" / inference.CTC_REV}
for link, target in links.items():
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
inference.CACHE = fake
controls.RUBBERBAND_PATH = r"C:\phon\tools\rb\rubberband-4.0.0-gpl-executable-windows\rubberband.exe"
_Path = pathlib.Path
run_confirmation.Path = lambda p: _Path(str(p).replace("/Users/Apple/Desktop", str(EVAL)))
ck = pathlib.Path(a.ckpt).resolve()
(out / "SWAP.json").write_text(json.dumps({"device": "cuda", "whisper_weights_dir": str(ck),
    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ck.iterdir()) if p.is_file()},
    "note": "metadata whisper_revision is a path label only; weights are from whisper_weights_dir"}, indent=1))
args = argparse.Namespace(mode="run", manifest=EVAL / "phonation_study" / "confirmation" / a.manifest,
                          output=out, device="cuda", reciters=a.reciters, resume=a.resume)
run_confirmation.execute(args)
