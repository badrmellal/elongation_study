"""Shared constants for the unseen-voice retrain (runs on SERVER-AI)."""
import re
from pathlib import Path

ROOT = Path("C:/phon")
DATA = ROOT / "everyayah"          # downloaded parquet shards (pinned revision)
PREP = ROOT / "prep"               # decoded 16 kHz int16 audio + index
PROC = ROOT / "tarteel_processor"  # the paper's WhisperProcessor files (no weights)
RUNS = ROOT / "runs"

# Voices that must never be seen in training: the paper's 8 reciters (both
# EveryAyah spellings of Minshawi) and the 3 Quran-Lab "held-out" reciters.
EXCLUDED = {
    "abdul_basit", "husary", "minshawi", "menshawi", "alafasy",
    "abdurrahmaan_as-sudais", "saood_ash-shuraym", "yasser_ad-dussary", "hani_rifai",
    "sahl_yassin", "akram_alalaqimy", "muhsin_al_qasim",
}
MIN_S, MAX_S = 0.5, 29.0  # paper runner excludes > 29 s from Whisper

def norm_ar(s: str) -> str:
    """Verbatim copy of release 2/src/exp1_sweep.py norm_ar (paper scorer)."""
    s = re.sub(r"[ً-ْٰٓـ]", "", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    return " ".join(s.split())

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
