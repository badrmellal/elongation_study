"""Validate complete coverage, then compute prespecified verse-paired summaries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from inference import digest, score
from run_confirmation import HERE, NEW, OLD, SEED, save

B = 10_000
ARMS = {1.0: {"baseline", "sham_pv_local", "sham_pv_global"},
        2.0: {"nucleus", "global", "silence", "nucleus_edge"},
        4.0: {"nucleus", "global", "silence", "nucleus_edge", "nucleus_rb", "global_rb"},
        6.0: {"nucleus", "global", "silence", "nucleus_edge"}}


def paired(values, rng):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Paired estimates contain non-finite values")
    if not len(values):
        return {"n": 0}
    estimates = values[rng.integers(len(values), size=(B, len(values)))].mean(1)
    signs = rng.choice([-1, 1], size=(B, len(values)))
    permutations = (signs * values).mean(1)
    p = (1 + int((np.abs(permutations) >= abs(values.mean()) - 1e-12).sum())) / (B + 1)
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(), "p_two_sided": p}


def holm(cells):
    if any("p_two_sided" not in c for c in cells):
        raise ValueError("Prespecified test family has a non-estimable member")
    ordered = sorted(cells, key=lambda c: c["p_two_sided"])
    maximum = 0
    for i, cell in enumerate(ordered):
        maximum = min(1, max(maximum, (len(ordered) - i) * cell["p_two_sided"]))
        cell["p_holm"] = maximum


def load_and_validate(folder, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["manifest_sha256"] != digest(manifest_path):
        raise AssertionError("Result/manifest hash mismatch")
    if manifest["protocol_sha256"] != digest(HERE / "PROTOCOL.md"):
        raise AssertionError("Protocol/manifest hash mismatch")
    if manifest["seed"] != SEED or not set(NEW) <= manifest["reciters"].keys():
        raise AssertionError("Confirmation sampling or five-reciter family changed")
    chosen = np.random.default_rng(SEED).choice(manifest["candidate_pool"], 60, replace=False).tolist()
    if chosen != manifest["confirmation_verses"]:
        raise AssertionError("Confirmation sample differs from the seeded draw")
    references, source_samples = {}, {}
    for name, group in manifest["reciters"].items():
        verses = [item["verse"] for item in group["files"]]
        if len(set(verses)) != len(verses):
            raise AssertionError(f"Duplicate manifest candidates: {name}")
        if name in NEW and (verses != chosen or group["role"] != "confirmation"):
            raise AssertionError(f"Different confirmation candidates/role: {name}")
        for item in group["files"]:
            if not np.isfinite(item["duration_s"]) or not 0 < item["duration_s"] <= 8:
                raise AssertionError(f"Ineligible source duration: {name}, {item['verse']}")
            key = name, item["verse"]
            references[key] = item["reference"]
            source_samples[key] = round(item["duration_s"] * 16000)
    expected = {(r, f["verse"]) for r, g in manifest["reciters"].items() for f in g["files"]}
    observations, states, dropped_factors, exclusions = {}, {}, set(), []
    alignments = {}
    with (folder / "observations.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row["reciter"], row["verse"]
            if key not in expected:
                raise AssertionError(f"Unexpected candidate {key}")
            kind = row["kind"]
            if kind == "condition":
                cid = (*key, row["arm"], row["k"])
                if cid in observations:
                    raise AssertionError(f"Duplicate condition {cid}")
                for model in ["whisper", "ctc"]:
                    s = row[model]
                    recomputed = score(s["reference"], s["hypothesis"])
                    if s["reference"] != references[key]:
                        raise AssertionError(f"Reference differs from manifest: {cid}")
                    if (not np.isfinite(s["wer"]) or s["errors"] != recomputed["errors"]
                            or abs(s["wer"] - recomputed["wer"]) > 1e-12):
                        raise AssertionError(f"WER does not match retained transcript: {cid}")
                    if "<|" in s["hypothesis"]:
                        raise AssertionError(f"Control token leaked: {cid}")
                observations[cid] = row
            elif kind in {"excluded", "complete"}:
                if key in states:
                    raise AssertionError(f"Duplicate terminal state: {key}")
                states[key] = kind
                if kind == "excluded": exclusions.append(row)
            elif kind == "factor_excluded":
                if row["k"] not in {2., 4., 6.}:
                    raise AssertionError(f"Invalid excluded factor: {key}, {row['k']}")
                dropped_factors.add((*key, row["k"]))
            elif kind == "alignment":
                if key in alignments and alignments[key]["occurrences"] != row["occurrences"]:
                    raise AssertionError(f"Resumed alignment differs: {key}")
                alignments[key] = row
    missing = expected - states.keys()
    if missing:
        raise AssertionError(f"Incomplete experiment: {len(missing)} candidates, e.g. {sorted(missing)[:3]}")
    reasons = {"Fewer than two merged CTC intervals >=60 ms", "Incomplete CTC alignment",
               "CTC target longer than feasible frame path"}
    if any(row["reason"] not in reasons for row in exclusions):
        raise AssertionError("Unprespecified candidate exclusion")
    for key, state in states.items():
        present = {(cid[2], cid[3]) for cid in observations if cid[:2] == key}
        required = {(a, k) for k, arms in ARMS.items() for a in arms if (*key, k) not in dropped_factors}
        if state == "excluded": required = set()
        if present != required:
            raise AssertionError(f"Coverage mismatch {key}: missing={required-present}, extra={present-required}")
        for k in [1., 2., 4., 6.]:
            lengths = {observations[(*key, a, k)]["samples"] for a in ARMS[k] if (*key, a, k) in observations}
            if len(lengths) > 1:
                raise AssertionError(f"Arms are not sample matched: {key}, {k}")
        if state == "excluded":
            if key in alignments or any(cid[:2] == key for cid in dropped_factors):
                raise AssertionError(f"Excluded candidate has an included alignment/factor: {key}")
            continue
        if key not in alignments:
            raise AssertionError(f"Complete candidate lacks alignment: {key}")
        occurrences = alignments[key]["occurrences"]
        if len(occurrences) < 2:
            raise AssertionError(f"Fewer than two aligned intervals: {key}")
        previous_end = -2
        for item in occurrences:
            lo, hi = item["start_sample"], item["end_sample"]
            if not (0 <= lo < hi <= source_samples[key]) or hi - lo < 960 or lo <= previous_end + 1:
                raise AssertionError(f"Invalid aligned interval: {key}, {item}")
            previous_end = hi
        for k, arms in ARMS.items():
            target = source_samples[key] + sum(round((k - 1) * (i["end_sample"] - i["start_sample"])) for i in occurrences)
            if (*key, k) in dropped_factors:
                if k == 1 or target <= 29 * 16000:
                    raise AssertionError(f"Unjustified factor exclusion: {key}, {k}")
                continue
            for arm in arms:
                row = observations[(*key, arm, k)]
                if row["samples"] != target or target > 29 * 16000:
                    raise AssertionError(f"Incorrect matched target length: {key}, {arm}, {k}")
                if abs(row["whisper"]["duration_s"] - target / 16000) > 1e-12:
                    raise AssertionError(f"Duration/sample mismatch: {key}, {arm}, {k}")
                measures = row["whisper"]["attention"]
                ids = [a["interval_id"] for a in measures]
                if len(ids) != len(set(ids)) or set(ids) != set(range(len(occurrences))):
                    raise AssertionError(f"Missing/duplicate/unknown attention interval: {key}, {arm}, {k}")
                for a in measures:
                    if "excluded" in a:
                        continue
                    if a["query_rows"] != occurrences[a["interval_id"]]["query_rows"]:
                        raise AssertionError(f"Attention query changed across conditions: {key}")
                    if "log_density_ratio" in a and (a["reference_frames"] < 10 or not np.isfinite(a["log_density_ratio"])):
                        raise AssertionError(f"Invalid relative attention support/value: {key}")
    return manifest, observations, states, exclusions, dropped_factors


def compare(observations, reciter, model, arm_a, arm_b, k_a, k_b, rng):
    verses = sorted({key[1] for key in observations if key[0] == reciter})
    diffs, used = [], []
    for verse in verses:
        a, b = (reciter, verse, arm_a, k_a), (reciter, verse, arm_b, k_b)
        if a in observations and b in observations:
            diffs.append(observations[a][model]["wer"] - observations[b][model]["wer"])
            used.append(verse)
    return {**paired(diffs, rng), "verses": used, "differences": diffs}


def attention_change(observations, reciter, arm, k, rng):
    changes = []
    used = []
    paired_intervals, excluded_intervals = 0, 0
    reasons = Counter()
    for key, row in observations.items():
        if key[0] != reciter or key[2:] != (arm, k): continue
        baseline = observations[(reciter, key[1], "baseline", 1.)]
        orig = {r["interval_id"]: r for r in baseline["whisper"]["attention"]}
        deltas = []
        for r in row["whisper"]["attention"]:
            b = orig.get(r["interval_id"], {})
            if "log_density_ratio" in r and "log_density_ratio" in b:
                deltas.append(r["log_density_ratio"] - b["log_density_ratio"])
                paired_intervals += 1
            else:
                excluded_intervals += 1
                for label, value in [("baseline", b), ("condition", r)]:
                    if "log_density_ratio" not in value:
                        reasons[f"{label}: {value.get('excluded', 'Fewer than ten reference frames')}"] += 1
        if deltas:
            changes.append(float(np.mean(deltas))); used.append(key[1])
    return {**paired(changes, rng), "verses": used, "differences": changes,
            "paired_intervals": paired_intervals, "excluded_intervals": excluded_intervals,
            "exclusion_reasons": dict(reasons)}


def crossed_summary(primary, rng):
    """Equal-reciter mean; resample reciter and common verse IDs independently."""
    names = list(NEW)
    verses = sorted({v for name in names for v in primary[name]["verses"]})
    matrix = np.full((len(names), len(verses)), np.nan)
    for i, name in enumerate(names):
        for verse, diff in zip(primary[name]["verses"], primary[name]["differences"]):
            matrix[i, verses.index(verse)] = diff
    reciter_means = np.nanmean(matrix, axis=1)
    if not np.isfinite(reciter_means).all():
        raise ValueError("Equal-reciter estimate requires eligible pairs for all five reciters")
    estimates = []
    for _ in range(B):
        ri = rng.integers(len(names), size=len(names))
        vi = rng.integers(len(verses), size=len(verses))
        sample = matrix[ri][:, vi]
        if np.isfinite(sample).any(axis=1).all():
            estimates.append(float(np.nanmean(sample, axis=1).mean()))
    # Do not silently condition on a data-dependent subset of bootstrap draws.
    # If any resampled row is empty, retain the point estimate and withhold CI.
    ci = np.quantile(estimates, [.025, .975]).tolist() if len(estimates) == B else None
    return {"mean": float(reciter_means.mean()), "ci95": ci,
            "bootstrap_valid": len(estimates), "reciters": names,
            "bootstrap_requested": B, "bootstrap_empty_row_draws": B - len(estimates),
            "ci_status": "available" if ci is not None else "withheld: empty resampled reciter rows",
            "leave_one_reciter_out": {n: float(np.delete(reciter_means, i).mean()) for i, n in enumerate(names)},
            "scope": "Five observed confirmation reciters; not a universal population claim"}


def analyze(folder, manifest_path, output, lexical_scoring=False):
    manifest, obs, states, exclusions, dropped = load_and_validate(folder, manifest_path)
    changed = set()
    if lexical_scoring:
        # Validate the immutable historical scores first, then amend scoring only.
        for cid, row in obs.items():
            for model in ("whisper", "ctc"):
                cell = row[model]
                corrected = score(cell["reference"].replace("\u06de", ""),
                                  cell["hypothesis"].replace("\u06de", ""))
                if corrected["wer"] != cell["wer"]:
                    changed.add(cid)
                for field in ("reference_normalized", "hypothesis_normalized",
                              "errors", "reference_words", "wer"):
                    cell[field] = corrected[field]
    missing_primary = [name for name in NEW if not any(key[0] == name and key[2:] == ("nucleus", 4.) for key in obs)]
    if missing_primary:
        raise ValueError(f"Primary comparison is not estimable for prespecified reciters: {missing_primary}")
    rng = np.random.default_rng(SEED)
    result = {"protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": digest(manifest_path),
              "observations_sha256": digest(folder / "observations.jsonl"),
              "analysis_source_sha256": digest(Path(__file__)), "bootstrap_replicates": B,
              "seed": SEED, "coverage": {}, "models": {}, "attention": {}, "quality": {}}
    for name in manifest["reciters"]:
        result["coverage"][name] = {"role": manifest["reciters"][name]["role"],
            "candidates": len(manifest["reciters"][name]["files"]),
            "included": sum(r == name and state == "complete" for (r, v), state in states.items()),
            "excluded": dict(Counter(r["reason"] for r in exclusions if r["reciter"] == name)),
            "factor_exclusions": {str(k): sum(r == name and factor == k for r, v, factor in dropped) for k in [2.,4.,6.]}}
    for model in ["whisper", "ctc"]:
        model_summary = {}
        for name in manifest["reciters"]:
            contrasts = {str(k): compare(obs, name, model, "nucleus", "global", k, k, rng) for k in [2.,4.,6.]}
            means = {}
            for k, arms in ARMS.items():
                for arm in sorted(arms):
                    values = [row[model]["wer"] for key, row in obs.items() if key[0] == name and key[2:] == (arm, k)]
                    if values: means[f"{arm}@{k:g}"] = {"n": len(values), "mean": float(np.mean(values))}
            model_summary[name] = {"local_global": contrasts, "means": means,
                "edge_global_k4": compare(obs, name, model, "nucleus_edge", "global", 4., 4., rng),
                "rubberband_local_global_k4": compare(obs, name, model, "nucleus_rb", "global_rb", 4., 4., rng),
                "sham_local_baseline": compare(obs, name, model, "sham_pv_local", "baseline", 1., 1., rng),
                "sham_global_baseline": compare(obs, name, model, "sham_pv_global", "baseline", 1., 1., rng)}
        if model == "whisper":
            holm([model_summary[n]["local_global"]["4.0"] for n in NEW])
            result["primary_equal_reciter_mean"] = crossed_summary({n: model_summary[n]["local_global"]["4.0"] for n in NEW}, rng)
        result["models"][model] = model_summary
    for name in manifest["reciters"]:
        result["attention"][name] = {f"{arm}@{k:g}": attention_change(obs, name, arm, k, rng)
            for k in [2.,4.,6.] for arm in ["nucleus", "global", "silence", "nucleus_edge"]}
    result["quality"] = {"conditions": len(obs), "token_cap_hits": sum(r["whisper"]["token_cap_hit"] for r in obs.values()),
        "mass_floor_events": sum(a.get("mass_floor_count", 0) for r in obs.values() for a in r["whisper"]["attention"]),
        "min_target_or_reference_mass": min((a["min_target_or_reference_mass"] for r in obs.values()
            for a in r["whisper"]["attention"] if "min_target_or_reference_mass" in a), default=None),
        "sample_match_verified": True, "transcript_WER_recomputed": True,
        "excluded_candidates": len(exclusions), "terminal_candidates": len(states)}
    if lexical_scoring:
        result["scoring_amendment"] = {
            "date": "2026-09-13", "timing": "post-hoc, after inference",
            "rule": "Remove nonlexical U+06DE from both reference and hypothesis before WER",
            "changed_conditions": len(changed),
            "affected_recordings": sorted({f"{c[0]}:{c[1]}" for c in changed}),
            "historical_log_modified": False, "inference_rerun": False}
    save(output, result)
    print(json.dumps({"coverage": result["coverage"], "primary_equal_reciter_mean": result["primary_equal_reciter_mean"],
                      "primary": {n: result["models"]["whisper"][n]["local_global"]["4.0"] for n in NEW}, "quality": result["quality"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--manifest", type=Path, default=HERE / "manifest_v2.json")
    p.add_argument("--output", type=Path, default=HERE / "analysis.json")
    p.add_argument("--lexical-scoring", action="store_true",
                   help="Apply documented post-hoc U+06DE scoring correction")
    a = p.parse_args()
    analyze(a.results, a.manifest, a.output, a.lexical_scoring)
