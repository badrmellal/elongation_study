"""Round-trip artifact arms (exposure/PROTOCOL.md Amendment 1A).

Stretch by k, then stretch back to the ORIGINAL duration with the same engine.
Duration is unchanged, artifact exposure is roughly doubled, so WER minus clean WER
measures artifact-only damage. Engine calls reuse the paper's controls module
unchanged (_pv, _rb, _prepare, _join, _result), so settings match the main arms.
"""
from __future__ import annotations
import controls
from controls import _finish, _join, _prepare, _pv, _rb, _result, _rubberband_info  # noqa: F401


def _roundtrip_pv(segment, stretched_len):
    out, rec_up = _pv(segment, stretched_len)
    back, rec_down = _pv(out, len(segment))
    return back, {"up": rec_up, "down": rec_down}


def _roundtrip_rb(segment, sr, stretched_len, info):
    out, rec_up = _rb(segment, sr, stretched_len, info)
    back, rec_down = _rb(out, sr, len(segment), info)
    return back, {"up": rec_up, "down": rec_down}


def roundtrip_conditions(x, sr, intervals, k, include_rubberband=False):
    """nucleus_rt / global_rt (PV) and optionally nucleus_rt_rb / global_rt_rb."""
    x, sr, bounds, sources = _prepare(x, sr, intervals)
    k = float(k)
    additions = [round((k - 1) * (i1 - i0)) for i0, i1 in bounds]
    target = len(x) + sum(additions)
    originals = [x[i0:i1] for i0, i1 in bounds]
    zeros = [0] * len(bounds)
    meta = {"source_interval_indices": sources, "roundtrip": True,
            "interpretation": "Artifact-only control: same duration as baseline, ~2x engine passes.",
            "librosa_version": controls.librosa.__version__}

    local, recs = [], []
    for original, added in zip(originals, additions):
        audio, rec = _roundtrip_pv(original, len(original) + added)
        local.append(audio); recs.append(rec["up"])
    nucleus_rt, _ = _join(x, bounds, local)
    global_rt, grec = _roundtrip_pv(x, target)
    out = {
        "nucleus_rt": _result(nucleus_rt, sr, k, bounds, bounds, zeros, controls._PV_BACKEND, recs, **meta),
        "global_rt": _result(global_rt, sr, k, bounds, bounds, zeros, controls._PV_BACKEND, [grec["up"]], **meta),
    }
    if include_rubberband:
        info = _rubberband_info()
        local_rb, recs_rb = [], []
        for original, added in zip(originals, additions):
            audio, rec = _roundtrip_rb(original, sr, len(original) + added, info)
            local_rb.append(audio); recs_rb.append(rec["up"])
        nucleus_rt_rb, _ = _join(x, bounds, local_rb)
        global_rt_rb, grec_rb = _roundtrip_rb(x, sr, target, info)
        rb_meta = {**meta, "rubberband": info}
        out["nucleus_rt_rb"] = _result(nucleus_rt_rb, sr, k, bounds, bounds, zeros, "rubberband", recs_rb, **rb_meta)
        out["global_rt_rb"] = _result(global_rt_rb, sr, k, bounds, bounds, zeros, "rubberband", [grec_rb["up"]], **rb_meta)
    return out
