"""Offline, exploratory S/D/I audit of retained transcripts; no model inference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from inference import digest, norm_ar, score
from run_confirmation import HERE, save


def edit_counts(reference: str, hypothesis: str) -> dict:
    """One optimal alignment, preferring match/substitution, deletion, insertion.

    Different minimum-distance paths may split S/D/I differently. Total edit
    distance is unique; this deterministic decomposition is descriptive only.
    """
    ref, hyp = norm_ar(reference).split(), norm_ar(hypothesis).split()
    if not ref:
        raise ValueError("Empty normalized reference")
    cost = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): cost[i][0] = i
    for j in range(len(hyp) + 1): cost[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0,
              "correct": 0, "reference_words": len(ref)}
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            counts["correct" if ref[i - 1] == hyp[j - 1] else "substitutions"] += 1
            i -= 1; j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            counts["deletions"] += 1; i -= 1
        else:
            counts["insertions"] += 1; j -= 1
    counts["errors"] = sum(counts[k] for k in ["substitutions", "deletions", "insertions"])
    if counts["errors"] != score(reference, hypothesis)["errors"]:
        raise AssertionError("Edit decomposition disagrees with inference scorer")
    counts["wer"] = counts["errors"] / counts["reference_words"]
    return counts


def audit(results: Path, output: Path):
    # Require validated full coverage, not an opportunistic subset of a live run.
    from analyze_confirmation import load_and_validate
    _, observations, _, _, _ = load_and_validate(results, HERE / "manifest_v2.json")
    groups = defaultdict(list)
    for (reciter, verse, arm, k), row in observations.items():
        for model in ["whisper", "ctc"]:
            counts = edit_counts(row[model]["reference"], row[model]["hypothesis"])
            groups[(reciter, model, arm, k)].append(counts)
    summaries = []
    for (reciter, model, arm, k), values in sorted(groups.items()):
        total_words = sum(v["reference_words"] for v in values)
        result = {"reciter": reciter, "model": model, "arm": arm, "k": k, "n": len(values),
                  "reference_words": total_words}
        for field in ["substitutions", "deletions", "insertions", "errors"]:
            result[field] = sum(v[field] for v in values)
            result[f"macro_{field}_rate"] = sum(v[field] / v["reference_words"] for v in values) / len(values)
        result["micro_wer"] = result["errors"] / total_words
        result["macro_wer"] = result["macro_errors_rate"]
        result["perfect_transcripts"] = sum(v["errors"] == 0 for v in values)
        summaries.append(result)
    save(output, {"status": "Exploratory offline decomposition, not a new confirmatory endpoint",
                  "tie_break": "diagonal (match/substitution), deletion, insertion",
                  "observations_sha256": digest(results / "observations.jsonl"),
                  "groups": summaries})
    print(f"Audited {len(observations)} conditions, {len(summaries)} reciter/model/arm groups")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--output", type=Path, default=HERE / "error_audit.json")
    args = p.parse_args()
    audit(args.results, args.output)
