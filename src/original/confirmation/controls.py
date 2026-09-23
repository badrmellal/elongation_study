"""Matched duration and audio-artifact controls; no model or inference code.

Inputs are finite, nonempty, real mono audio in [-1, 1], a positive integer
sample rate, and intervals in seconds. Audio is converted once to float32;
unity controls preserve those bits (including signed zero). Expansion k must
be finite and >= 1. Invalid interval entries are discarded, finite bounds are
clipped to the recording, floored to samples, sorted, and merged when the gap
is at most one sample. A one-ULP upward adjustment before flooring makes
integer sample -> seconds -> sample conversion stable. Reversed/empty bounds
are discarded. Added sample counts use round(), with ties to even.

Every result contains ``audio``, ``intervals`` (seconds on the output timeline),
``backend``, and ``metadata``. Authoritative integer bounds are in metadata's
``input_intervals_samples`` and ``output_intervals_samples``. Metadata's
``source_interval_indices`` maps each canonical interval to the zero-based
input entries that formed it. Already merged sample-grid inputs retain their
interval count and order across every arm and factor. All arms add
exactly sum(round((k - 1) * (end - start))) over the canonical input bounds.
Silence intervals exclude the inserted zeros; global bounds use the matched
whole-recording ratio, rounded once per endpoint.

Finite synthesized peaks outside [-1, 1] are preserved without clipping or
gain normalization. ``absmax`` and ``samples_exceeding_one`` are diagnostics
on the returned audio. ``length_corrections`` records each processed cut
(one for global), including raw/target lengths, signed correction (positive
means zero-padding; negative means tail trimming), raw peak diagnostics, and synthesis.
Empty/nonfinite backend output or discrepancies above two samples raise
RuntimeError; no resampling or substantial padding is used.

PV uses a 2048-sample FFT and hop min(512, input_cut_length), also used by the
shams. Reducing the hop only for cuts shorter than 512 samples guarantees at
least two centered STFT frames; librosa 1.0's interpolator produces NaNs with
one frame. Each processing record reports the actual FFT and hop settings.

``nucleus_edge`` is a boundary sensitivity control, NOT proof of artifact
removal. It blends matching original edges into the same PV-stretched cuts,
using raised-cosine weights over min(sr // 100, original_length // 4) samples
per edge. This tests sensitivity to boundary treatment, not interior artifacts.

Rubber Band is optional and pinned to installed CLI version 4.0.0. Help and
version are checked before selecting offline R3/fine settings, consistently
used for local and global processing. Each subprocess uses an isolated temp
directory and FLOAT WAV input/output; no package installation or network I/O.
"""

from functools import lru_cache
import math
from pathlib import Path
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


__all__ = ["make_conditions", "sham_conditions"]
RUBBERBAND_PATH = "/opt/homebrew/bin/rubberband"
RUBBERBAND_VERSION = "4.0.0"
_N_FFT = 2048
_HOP_LENGTH = 512
_PV_BACKEND = "librosa.effects.time_stretch"
_SHAM_BACKEND = "librosa.stft -> phase_vocoder(rate=1) -> istft"


def _prepare(x, sr, intervals):
    if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sr must be a positive integer")
    raw = np.asarray(x)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "fiu":
        raise ValueError("x must be nonempty real mono audio")
    if not np.isfinite(raw).all() or np.any(np.abs(raw.astype(np.float64)) > 1):
        raise ValueError("x must be finite and within [-1, 1]")
    audio = np.asarray(raw, dtype=np.float32)
    sr = int(sr)
    duration = len(audio) / sr
    bounds = []
    for source_index, entry in enumerate(intervals):
        try:
            start, end = entry
            start, end = float(start), float(end)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        i0 = min(len(audio), math.floor(np.nextafter(max(0.0, min(duration, start)) * sr, np.inf)))
        i1 = min(len(audio), math.floor(np.nextafter(max(0.0, min(duration, end)) * sr, np.inf)))
        if i0 < i1:
            bounds.append((i0, i1, [source_index]))
    merged = []
    for i0, i1, sources in sorted(bounds):
        if merged and i0 <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], i1), sorted(merged[-1][2] + sources))
        else:
            merged.append((i0, i1, sources))
    return audio, sr, [(i0, i1) for i0, i1, _ in merged], [sources for _, _, sources in merged]


def _finish(raw, target, input_length, synthesized):
    raw = np.asarray(raw)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise RuntimeError("backend returned empty, non-mono, non-real, or nonfinite audio")
    correction = target - len(raw)
    if abs(correction) > 2:
        raise RuntimeError(
            f"backend length discrepancy exceeds 2 samples: target={target}, actual={len(raw)}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        audio = raw.astype(np.float32)
    if not np.isfinite(audio).all():
        raise RuntimeError("backend output cannot be represented as finite float32")
    if correction > 0:
        audio = np.pad(audio, (0, correction))
    elif correction < 0:
        audio = audio[:target].copy()
    return audio, {
        "input_samples": input_length,
        "target_samples": target,
        "raw_output_samples": len(raw),
        "correction_samples": correction,
        "raw_absmax": float(np.max(np.abs(raw))),
        "raw_samples_exceeding_one": int(np.count_nonzero(np.abs(raw) > 1)),
        "synthesized": synthesized,
    }


def _pv(segment, target):
    synthesized = target != len(segment)
    hop = min(_HOP_LENGTH, len(segment))
    raw = librosa.effects.time_stretch(
        segment, rate=len(segment) / target, n_fft=_N_FFT, hop_length=hop
    ) if synthesized else segment.copy()
    audio, record = _finish(raw, target, len(segment), synthesized)
    return audio, {**record, "n_fft": _N_FFT, "hop_length": hop}


def _sham(segment):
    # Deliberately call every synthesis stage even at unity. Never use _pv here.
    hop = min(_HOP_LENGTH, len(segment))
    spectrum = librosa.stft(segment, n_fft=_N_FFT, hop_length=hop)
    spectrum = librosa.phase_vocoder(spectrum, rate=1.0, hop_length=hop, n_fft=_N_FFT)
    raw = librosa.istft(spectrum, n_fft=_N_FFT, hop_length=hop,
                        length=len(segment), dtype=np.float32)
    audio, record = _finish(raw, len(segment), len(segment), True)
    return audio, {**record, "n_fft": _N_FFT, "hop_length": hop}


def _join(x, bounds, segments, gaps=None):
    pieces, output_bounds = [], []
    cursor, offset = 0, 0
    for index, ((i0, i1), segment) in enumerate(zip(bounds, segments)):
        pieces.extend((x[cursor:i0], segment))
        start = i0 + offset
        output_bounds.append((start, start + len(segment)))
        gap = 0 if gaps is None else gaps[index]
        if gap:
            pieces.append(np.zeros(gap, dtype=np.float32))
        offset += len(segment) - (i1 - i0) + gap
        cursor = i1
    pieces.append(x[cursor:])
    return np.concatenate(pieces).astype(np.float32, copy=False), output_bounds


def _blend_edges(original, stretched, sr):
    result = stretched.copy()
    count = min(sr // 100, len(original) // 4)
    if count:
        original_weight = (1 + np.cos(np.linspace(0, np.pi, count))) / 2
        result[:count] = original[:count] * original_weight + result[:count] * (1 - original_weight)
        result[-count:] = original[-count:] * original_weight[::-1] + result[-count:] * (1 - original_weight[::-1])
    return result, count


def _result(audio, sr, k, bounds, output_bounds, additions, backend, records, **extra):
    metadata = {
        "sr": sr,
        "k": k,
        "input_samples": len(audio) - sum(additions),
        "output_samples": len(audio),
        "added_samples": sum(additions),
        "interval_added_samples": list(additions),
        "input_intervals_samples": list(bounds),
        "output_intervals_samples": list(output_bounds),
        "length_corrections": [dict(record) for record in records],
        "synthesis_calls": sum(record["synthesized"] for record in records),
        "absmax": float(np.max(np.abs(audio))),
        "samples_exceeding_one": int(np.count_nonzero(np.abs(audio) > 1)),
        **extra,
    }
    return {"audio": audio, "intervals": [(i0 / sr, i1 / sr) for i0, i1 in output_bounds],
            "backend": backend, "metadata": metadata}


@lru_cache(maxsize=1)
def _rubberband_info():
    help_result = subprocess.run([RUBBERBAND_PATH, "--help"], capture_output=True, text=True, timeout=15)
    full_help = subprocess.run([RUBBERBAND_PATH, "--full-help"], capture_output=True, text=True, timeout=15)
    version = subprocess.run([RUBBERBAND_PATH, "--version"], capture_output=True, text=True, timeout=15)
    if help_result.returncode not in (0, 2) or full_help.returncode not in (0, 2) or version.returncode:
        raise RuntimeError("Rubber Band help/version query failed")
    version_text = (version.stdout + version.stderr).strip()
    if version_text != RUBBERBAND_VERSION:
        raise RuntimeError(f"Rubber Band version must be {RUBBERBAND_VERSION}; found {version_text!r}")
    help_text = help_result.stdout + help_result.stderr + full_help.stdout + full_help.stderr
    flags = ["--fine", "--quiet", "--ignore-clipping"]
    if any(flag not in help_text for flag in [*flags, "--time"]):
        raise RuntimeError("Rubber Band CLI lacks required offline R3 options")
    return {"path": RUBBERBAND_PATH, "version": version_text, "engine": "R3",
            "mode": "offline", "flags": flags, "wav_subtype": "FLOAT"}


def _rb(segment, sr, target, info):
    if target == len(segment):
        return _finish(segment.copy(), target, len(segment), False)
    with tempfile.TemporaryDirectory(prefix="phonation-controls-rb-") as directory:
        source, dest = Path(directory) / "input.wav", Path(directory) / "output.wav"
        sf.write(source, segment, sr, format="WAV", subtype="FLOAT")
        command = [info["path"], *info["flags"], "--time", format(target / len(segment), ".17g"),
                   str(source), str(dest)]
        completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise RuntimeError(f"Rubber Band failed ({completed.returncode}): {completed.stderr.strip()}")
        output_info = sf.info(dest)
        if output_info.samplerate != sr or output_info.channels != 1 or output_info.subtype != "FLOAT":
            raise RuntimeError("Rubber Band output must be mono FLOAT WAV at the input sample rate")
        raw, _ = sf.read(dest, dtype="float32")
    return _finish(raw, target, len(segment), True)


def make_conditions(x: np.ndarray, sr: int, intervals: list[tuple[float, float]],
                    k: float, include_rubberband: bool = False) -> dict[str, dict]:
    """Return nucleus, global, silence, nucleus_edge; optionally nucleus_rb/global_rb.

    See module documentation for input validation, sample rounding, metadata,
    peak diagnostics, and correction policy. Rubber Band failures propagate so
    the caller can gate that backend using pre-run tests; there is no fallback.
    """
    x, sr, bounds, sources = _prepare(x, sr, intervals)
    k = float(k)
    if not math.isfinite(k) or k < 1:
        raise ValueError("k must be finite and >= 1")
    additions = [round((k - 1) * (i1 - i0)) for i0, i1 in bounds]
    target = len(x) + sum(additions)
    output_global = [(round(i0 * target / len(x)), round(i1 * target / len(x))) for i0, i1 in bounds]
    originals = [x[i0:i1] for i0, i1 in bounds]
    stretched, records = [], []
    for original, added in zip(originals, additions):
        audio, record = _pv(original, len(original) + added)
        stretched.append(audio)
        records.append(record)
    nucleus, output_local = _join(x, bounds, stretched)
    global_audio, global_record = _pv(x, target)
    silence, output_silence = _join(x, bounds, originals, additions)
    edged, edge_counts = [], []
    for original, audio in zip(originals, stretched):
        # Preserve unity bits even where floating-point blending could differ.
        blended, count = (audio.copy(), 0) if k == 1 else _blend_edges(original, audio, sr)
        edged.append(blended)
        edge_counts.append(count)
    edge_audio, _ = _join(x, bounds, edged)
    canonical_metadata = {"source_interval_indices": sources,
                          "interval_quantization": "floor with one-ULP grid stabilization; merge gap <= 1 sample"}
    pv_metadata = {**canonical_metadata, "librosa_version": librosa.__version__,
                   "n_fft": _N_FFT, "hop_length_max": _HOP_LENGTH}
    conditions = {
        "nucleus": _result(nucleus, sr, k, bounds, output_local, additions, _PV_BACKEND, records, **pv_metadata),
        "global": _result(global_audio, sr, k, bounds, output_global, additions, _PV_BACKEND, [global_record], **pv_metadata),
        "silence": _result(silence, sr, k, bounds, output_silence, additions, "silence-insertion", [], **canonical_metadata),
        "nucleus_edge": _result(edge_audio, sr, k, bounds, output_local, additions, _PV_BACKEND, records,
                                edge_samples=edge_counts, control="boundary sensitivity",
                                interpretation="Boundary sensitivity control; not proof of artifact removal.", **pv_metadata),
    }
    if include_rubberband:
        info = _rubberband_info()
        rb_segments, rb_records = [], []
        for original, added in zip(originals, additions):
            audio, record = _rb(original, sr, len(original) + added, info)
            rb_segments.append(audio)
            rb_records.append(record)
        local_rb, _ = _join(x, bounds, rb_segments)
        global_rb, rb_global_record = _rb(x, sr, target, info)
        for name, audio, output_bounds, backend_records in (
            ("nucleus_rb", local_rb, output_local, rb_records),
            ("global_rb", global_rb, output_global, [rb_global_record]),
        ):
            conditions[name] = _result(audio, sr, k, bounds, output_bounds, additions, "rubberband",
                                       backend_records, rubberband={**info, "flags": list(info["flags"])}, **canonical_metadata)
    return conditions


def sham_conditions(x: np.ndarray, sr: int,
                    intervals: list[tuple[float, float]]) -> dict[str, dict]:
    """Force unity STFT -> phase_vocoder -> ISTFT locally and globally.

    Returns sham_pv_local and sham_pv_global at exactly the input length. Local
    synthesis uses the same canonical cuts as nucleus; all outside samples are
    preserved. With no valid cuts, local synthesis_calls is honestly zero;
    global still synthesizes the whole recording. Numerical near-identity is
    possible and is not evidence that synthesis was skipped.
    """
    x, sr, bounds, sources = _prepare(x, sr, intervals)
    segments, records = [], []
    for i0, i1 in bounds:
        audio, record = _sham(x[i0:i1])
        segments.append(audio)
        records.append(record)
    local, _ = _join(x, bounds, segments)
    global_audio, global_record = _sham(x)
    return {
        name: _result(audio, sr, 1.0, bounds, bounds, [0] * len(bounds), _SHAM_BACKEND, backend_records,
                      librosa_version=librosa.__version__, n_fft=_N_FFT, hop_length_max=_HOP_LENGTH,
                      forced_round_trip=True, source_interval_indices=sources,
                      interval_quantization="floor with one-ULP grid stabilization; merge gap <= 1 sample")
        for name, audio, backend_records in (
            ("sham_pv_local", local, records), ("sham_pv_global", global_audio, [global_record])
        )
    }
