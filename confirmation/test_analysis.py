"""CPU-only, synthetic acceptance tests for the frozen confirmation analyzer.

"""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np


HERE = Path(__file__).resolve().parent


def definitions(path, names, namespace):
    """Compile selected, inspected pure definitions, never module side effects."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    assert names <= namespace.keys(), names - namespace.keys()


def load_analyzer():
    inference = ModuleType("inference")
    inference.__dict__.update(hashlib=hashlib, Path=Path)
    definitions(HERE.parent / "release 2/src/exp1_sweep.py", {"norm_ar", "lev"}, inference.__dict__)
    definitions(HERE / "inference.py", {"digest", "score"}, inference.__dict__)
    runner = ModuleType("run_confirmation")
    runner.__dict__.update(Path=Path, __file__=str(HERE / "run_confirmation.py"))
    definitions(HERE / "run_confirmation.py", {"HERE", "NEW", "OLD", "SEED"}, runner.__dict__)
    runner.save = Mock(side_effect=AssertionError("Disk output is forbidden in this audit"))
    spec = importlib.util.spec_from_file_location("confirmation_analysis_under_test", HERE / "analyze_confirmation.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"inference": inference, "run_confirmation": runner}):
        spec.loader.exec_module(module)
    return module


analysis = load_analyzer()


class MemoryPath:
    """The Path subset needed by the analyzer and its real digest function."""

    def __init__(self, files, name):
        self.files, self.name = files, name

    def __truediv__(self, child):
        return MemoryPath(self.files, f"{self.name}/{child}")

    def read_text(self):
        return self.files[self.name]

    def open(self, mode="r"):
        if mode == "r":
            return io.StringIO(self.read_text())
        if mode == "rb":
            return io.BytesIO(self.read_text().encode())
        raise AssertionError("Only in-memory reads are allowed")


def attention(interval_id, value=0.0, query=None):
    return {"interval_id": interval_id, "log_density_ratio": value,
            "per_head_log_density_ratio": [value],
            "query_rows": [interval_id + 1] if query is None else query,
            "reference_frames": 30, "mass_floor_count": 0,
            "min_target_or_reference_mass": 0.01}


class SyntheticExperiment:
    """Five reciters, the seeded 60 candidates, two eligible verses per reciter."""

    def __init__(self):
        self.names = list(analysis.NEW)
        pool = [f"{chapter}:{verse}" for chapter, count in [(36, 83), (55, 78), (67, 30), (78, 23)]
                for verse in range(1, count + 1)]
        chosen = np.random.default_rng(analysis.SEED).choice(pool, 60, replace=False).tolist()
        self.verse = chosen[0]
        self.protocol = "Synthetic protocol fixture; no real experiment records."
        self.manifest = {"seed": analysis.SEED, "protocol_sha256": hashlib.sha256(self.protocol.encode()).hexdigest(),
                         "candidate_pool": pool, "confirmation_verses": chosen, "reciters": {}}
        self.rows = []
        self.reference = "alpha beta"
        for name, folder in analysis.NEW.items():
            self.manifest["reciters"][name] = {"role": "confirmation", "folder": folder, "files": [
                {"verse": verse, "duration_s": 1.0, "reference": self.reference,
                 "path": f"/synthetic-audio/{name}/{verse}.wav", "sha256": "0" * 64}
                for verse in chosen]}
            for verse in chosen:
                base = {"reciter": name, "verse": verse, "role": "confirmation"}
                if verse not in chosen[:2]:
                    self.rows.append({**base, "kind": "excluded",
                                      "reason": "Fewer than two merged CTC intervals >=60 ms",
                                      "alignment": {"detected": [], "excluded": "Fewer than two merged CTC intervals >=60 ms"}})
                    continue
                occurrences = [{"start_sample": 1600, "end_sample": 3200, "query_rows": [1]},
                               {"start_sample": 8000, "end_sample": 9600, "query_rows": [2]}]
                self.rows.append({**base, "kind": "alignment", "occurrences": occurrences})
                for k, arms in analysis.ARMS.items():
                    for arm in sorted(arms):
                        samples = int(16000 + (k - 1) * 3200)
                        scored = analysis.score(self.reference, self.reference)
                        self.rows.append({**base, "kind": "condition", "arm": arm, "k": k,
                                          "samples": samples, "ctc": deepcopy(scored),
                                          "whisper": {**scored, "duration_s": samples / 16000,
                                                      "token_cap_hit": False,
                                                      "attention": [attention(0), attention(1)]}})
                self.rows.append({**base, "kind": "complete"})

    def row(self, kind="condition", arm="nucleus", k=4.0):
        return next(row for row in self.rows if row["reciter"] == self.names[0]
                    and row["verse"] == self.verse and row["kind"] == kind
                    and (kind != "condition" or (row["arm"], row["k"]) == (arm, k)))

    def drop_factor(self, k):
        self.rows = [row for row in self.rows if not (
            row["reciter"] == self.names[0] and row["verse"] == self.verse
            and row["kind"] == "condition" and row["k"] == k)]
        self.rows.append({"kind": "factor_excluded", "reciter": self.names[0],
                          "verse": self.verse, "k": k, "reason": "Matched duration >29 s"})

    def paths(self):
        files = {"audit/PROTOCOL.md": self.protocol, "audit/manifest.json": json.dumps(self.manifest),
                 "audit/results/observations.jsonl": "\n".join(json.dumps(row) for row in self.rows) + "\n"}
        root = MemoryPath(files, "audit")
        files["audit/results/metadata.json"] = json.dumps({"manifest_sha256": analysis.digest(root / "manifest.json")})
        return root, root / "results", root / "manifest.json"

    def validate(self):
        root, folder, manifest = self.paths()
        with patch.object(analysis, "HERE", root):
            return analysis.load_and_validate(folder, manifest)

    def summarize(self):
        root, folder, manifest = self.paths()
        # Match save()'s finite-JSON requirement without creating any file.
        with patch.object(analysis, "HERE", root), patch.object(analysis, "B", 16), \
             patch.object(analysis, "save", side_effect=lambda path, data: json.dumps(data, allow_nan=False)) as saved, \
             redirect_stdout(io.StringIO()):
            analysis.analyze(folder, manifest, root / "analysis.json")
        saved.assert_called_once()
        return saved.call_args.args[1]


class StatisticalTests(unittest.TestCase):
    def test_frozen_seed_and_draw_count(self):
        self.assertEqual(analysis.SEED, 20270911)
        self.assertEqual(analysis.B, 10000)

    def test_paired_bootstrap_and_two_sided_sign_flips_match_oracle(self):
        values = np.array([1.0, 2.0, 4.0])
        indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2],
                            [0, 0, 1], [0, 1, 1], [1, 2, 2], [0, 2, 2]])
        signs = np.array(list(product([-1, 1], repeat=3)))
        rng = Mock()
        rng.integers.return_value = indices
        rng.choice.return_value = signs
        with patch.object(analysis, "B", 8):
            result = analysis.paired(values, rng)
        boot = [sum(values[j] for j in draw) / 3 for draw in indices]
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 7 / 3)
        np.testing.assert_allclose(result["ci95"], np.percentile(boot, [2.5, 97.5]))
        # Only the all-positive/all-negative patterns tie the observed |mean|.
        self.assertEqual(result["p_two_sided"], (1 + 2) / (8 + 1))
        rng.integers.assert_called_once_with(3, size=(8, 3))
        rng.choice.assert_called_once_with([-1, 1], size=(8, 3))

    def test_mixed_sign_differences_use_absolute_mean_not_mean_absolute(self):
        rng = Mock()
        rng.integers.return_value = np.tile(np.arange(3), (8, 1))
        rng.choice.return_value = np.array(list(product([-1, 1], repeat=3)))
        with patch.object(analysis, "B", 8):
            result = analysis.paired([1, -2, 4], rng)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["p_two_sided"], 7 / 9)  # Six of eight patterns meet |mean| >= 1.

    def test_zero_singleton_and_empty_pairs(self):
        with patch.object(analysis, "B", 16):
            for values, ci in [([0, 0, 0], [0, 0]), ([-2], [-2, -2])]:
                result = analysis.paired(values, np.random.default_rng(4))
                self.assertEqual(result["ci95"], ci)
                self.assertEqual(result["p_two_sided"], 1)
            self.assertEqual(analysis.paired([], np.random.default_rng(4)), {"n": 0})

    def test_nonfinite_paired_value_is_rejected(self):
        with patch.object(analysis, "B", 16), self.assertRaises((AssertionError, ValueError)):
            analysis.paired([0.1, float("nan")], np.random.default_rng(4))

    def test_holm_exact_five_family_monotonic_ties_and_cap(self):
        cells = [{"p_two_sided": p} for p in [0.04, 0.001, 0.02, 0.011, 0.04]]
        analysis.holm(cells)
        np.testing.assert_allclose([cell["p_holm"] for cell in cells], [0.08, 0.005, 0.06, 0.044, 0.08])
        cells = [{"p_two_sided": p} for p in [0, 0.4, 0.5, 0.6, 1]]
        analysis.holm(cells)
        self.assertEqual([cell["p_holm"] for cell in cells], [0, 1, 1, 1, 1])

    def test_compare_uses_same_reciter_and_verse_intersection(self):
        obs = {(r, v, arm, 4.0): {"whisper": {"wer": value}} for r, v, arm, value in [
            ("r", "v2", "global", 0.3), ("r", "v1", "nucleus", 0.8),
            ("r", "v1", "global", 0.2), ("r", "v2", "nucleus", 0.1),
            ("r", "unpaired", "nucleus", 100), ("other", "v1", "global", 100)]}
        with patch.object(analysis, "B", 16):
            result = analysis.compare(obs, "r", "whisper", "nucleus", "global", 4.0, 4.0, np.random.default_rng(4))
        self.assertEqual(result["verses"], ["v1", "v2"])
        np.testing.assert_allclose(result["differences"], [0.6, -0.2])
        self.assertAlmostEqual(result["mean"], 0.2)

    def test_complete_crossed_bootstrap_uses_shared_columns_and_equal_reciters(self):
        names = list(analysis.NEW)
        matrix = np.arange(15, dtype=float).reshape(5, 3)
        primary = {name: {"verses": ["v0", "v1", "v2"], "differences": matrix[i].tolist()}
                   for i, name in enumerate(names)}
        draws = [(np.array(ri), np.array(vi)) for ri, vi in [
            ([0, 1, 2, 3, 4], [0, 1, 2]), ([4, 4, 4, 4, 4], [2, 2, 2]),
            ([1, 1, 1, 1, 1], [0, 0, 0]), ([0, 2, 4, 4, 2], [0, 2, 2])]]
        rng = Mock()
        rng.integers.side_effect = [indices for draw in draws for indices in draw]
        with patch.object(analysis, "B", len(draws)):
            result = analysis.crossed_summary(primary, rng)
        expected = [sum(sum(matrix[r, v] for v in vi) / len(vi) for r in ri) / len(ri)
                    for ri, vi in draws]
        self.assertEqual(result["mean"], 7.0)
        self.assertEqual(result["bootstrap_valid"], 4)
        np.testing.assert_allclose(result["ci95"], np.percentile(expected, [2.5, 97.5]))
        for i, name in enumerate(names):
            self.assertEqual(result["leave_one_reciter_out"][name], np.delete(matrix.mean(1), i).mean())

    def test_crossed_point_estimate_keeps_equal_reciter_weights_with_missing_cells(self):
        names = list(analysis.NEW)
        primary = {name: {"verses": ["a"], "differences": [1.0]} for name in names}
        primary[names[0]] = {"verses": ["a", "b"], "differences": [0.0, 0.0]}
        with patch.object(analysis, "B", 16):
            result = analysis.crossed_summary(primary, np.random.default_rng(4))
        self.assertAlmostEqual(result["mean"], 0.8)  # Pooled-cell mean would be 4/6.

    def test_crossed_ci_does_not_silently_drop_empty_row_draws(self):
        names = list(analysis.NEW)
        primary = {name: {"verses": ["a", "b"], "differences": [0.0, 0.0]} for name in names}
        primary[names[-1]] = {"verses": ["a"], "differences": [1.0]}
        rng = Mock()
        rng.integers.side_effect = [np.arange(5), np.array([1, 1]), np.arange(5), np.array([0, 0])]
        with patch.object(analysis, "B", 2):
            result = analysis.crossed_summary(primary, rng)
        self.assertEqual(result["bootstrap_valid"], 1)
        self.assertEqual(result["bootstrap_empty_row_draws"], 1)
        self.assertIsNone(result["ci95"], "A conditional bootstrap CI must be withheld, not relabeled as unconditional")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.experiment = SyntheticExperiment()

    def test_valid_complete_synthetic_experiment(self):
        _, obs, states, exclusions, dropped = self.experiment.validate()
        self.assertEqual(len(obs), 170)
        self.assertEqual(len(states), 300)
        self.assertEqual(len(exclusions), 290)
        self.assertEqual(dropped, set())

    def test_existing_hash_coverage_duplicate_and_sample_guards(self):
        for defect, message in [("hash", "hash mismatch"), ("missing", "Coverage mismatch"),
                                ("duplicate", "Duplicate condition"), ("length", "sample matched"),
                                ("terminal", "Incomplete experiment"), ("wer", "WER does not match")]:
            with self.subTest(defect=defect):
                fixture = SyntheticExperiment()
                row = fixture.row()
                if defect == "missing": fixture.rows.remove(row)
                if defect == "duplicate": fixture.rows.append(deepcopy(row))
                if defect == "length": row["samples"] += 1
                if defect == "terminal": fixture.rows.remove(fixture.row("complete"))
                if defect == "wer": row["whisper"]["wer"] = 0.5
                root, folder, manifest = fixture.paths()
                if defect == "hash": root.files["audit/results/metadata.json"] = '{"manifest_sha256": "wrong"}'
                with patch.object(analysis, "HERE", root), self.assertRaisesRegex(AssertionError, message):
                    analysis.load_and_validate(folder, manifest)

    def test_legitimate_long_factor_exclusion_removes_all_arms(self):
        fixture = self.experiment
        fixture.manifest["reciters"][fixture.names[0]]["files"][0]["duration_s"] = 8.0
        occurrences = fixture.row("alignment")["occurrences"]
        occurrences[0].update(start_sample=1600, end_sample=40000)
        occurrences[1].update(start_sample=60000, end_sample=112000)
        for row in fixture.rows:
            if (row["reciter"], row["verse"], row["kind"]) == (fixture.names[0], fixture.verse, "condition"):
                row["samples"] = int(128000 + (row["k"] - 1) * 90400)
                row["whisper"]["duration_s"] = row["samples"] / 16000
        fixture.drop_factor(6.0)  # Expected 36.25 s; k=4 remains eligible at 24.95 s.
        _, obs, _, _, dropped = fixture.validate()
        self.assertEqual(dropped, {(fixture.names[0], fixture.verse, 6.0)})
        self.assertFalse(any(key[:2] == (fixture.names[0], fixture.verse) and key[3] == 6.0 for key in obs))

    def test_complete_candidate_requires_alignment(self):
        self.experiment.rows.remove(self.experiment.row("alignment"))
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.validate()

    def test_alignment_requires_two_intervals_at_least_60ms(self):
        for defect in ["one_interval", "short_interval"]:
            with self.subTest(defect=defect):
                fixture = SyntheticExperiment()
                occurrences = fixture.row("alignment")["occurrences"]
                if defect == "one_interval": occurrences.pop()
                else: occurrences[0]["end_sample"] = occurrences[0]["start_sample"] + 959
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_manifest_sampling_duration_and_five_reciter_family_are_validated(self):
        for defect in ["duplicate", "duration", "missing_reciter", "different_candidates", "seed", "protocol_hash"]:
            with self.subTest(defect=defect):
                fixture = SyntheticExperiment()
                files = fixture.manifest["reciters"][fixture.names[0]]["files"]
                if defect == "duplicate": files.append(deepcopy(files[0]))
                if defect == "duration": files[0]["duration_s"] = 8.01
                if defect == "missing_reciter":
                    del fixture.manifest["reciters"][fixture.names[-1]]
                    fixture.rows = [r for r in fixture.rows if r["reciter"] != fixture.names[-1]]
                if defect == "different_candidates":
                    removed = files.pop()["verse"]
                    fixture.rows = [r for r in fixture.rows if (r["reciter"], r["verse"]) != (fixture.names[0], removed)]
                if defect == "seed": fixture.manifest["seed"] += 1
                if defect == "protocol_hash": fixture.manifest["protocol_sha256"] = "wrong"
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_factor_exclusion_requires_over_29_seconds_and_valid_factor(self):
        for k in [1.0, 4.0]:
            with self.subTest(k=k):
                fixture = SyntheticExperiment()
                fixture.drop_factor(k)  # Source 1 s; k=4 target 1.6 s, so neither exclusion is legal.
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_candidate_exclusion_reason_must_be_prespecified(self):
        next(row for row in self.experiment.rows if row["kind"] == "excluded")["reason"] = "High WER"
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.validate()

    def test_matched_lengths_must_equal_expected_target_and_not_exceed_29_seconds(self):
        for samples in [25601, 29 * 16000 + 1]:
            with self.subTest(samples=samples):
                fixture = SyntheticExperiment()
                for arm in analysis.ARMS[4.0]:
                    row = fixture.row(arm=arm)
                    row["samples"] = samples
                    row["whisper"]["duration_s"] = samples / 16000
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_unity_sham_must_match_baseline_length(self):
        self.experiment.row(arm="sham_pv_local", k=1.0)["samples"] += 1
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.validate()

    def test_scoring_reference_must_match_manifest(self):
        row = self.experiment.row()
        row["whisper"].update(analysis.score("different reference", "different reference"))
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.validate()

    def test_nan_wer_must_not_pass_transcript_recomputation(self):
        self.experiment.row()["whisper"]["wer"] = float("nan")
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.validate()

    def test_analysis_corrects_exactly_five_whisper_k4_tests(self):
        result = self.experiment.summarize()
        adjusted = [(model, name, k) for model, reciters in result["models"].items()
                    for name, record in reciters.items() for k, cell in record["local_global"].items()
                    if "p_holm" in cell]
        self.assertEqual(set(adjusted), {("whisper", name, "4.0") for name in self.experiment.names})
        self.assertEqual(result["seed"], 20270911)

    def test_empty_confirmation_reciter_has_explicit_preflight_error(self):
        name = self.experiment.names[0]
        self.experiment.rows = [r for r in self.experiment.rows if r["reciter"] != name]
        for item in self.experiment.manifest["reciters"][name]["files"]:
            self.experiment.rows.append({"kind": "excluded", "reciter": name, "verse": item["verse"],
                                         "reason": "Fewer than two merged CTC intervals >=60 ms",
                                         "alignment": {"detected": [], "excluded": "Fewer than two merged CTC intervals >=60 ms"}})
        with self.assertRaises((AssertionError, ValueError)):
            self.experiment.summarize()

    def test_quality_reports_minimum_mass(self):
        self.experiment.row()["whisper"]["attention"][0].update(
            mass_floor_count=1, min_target_or_reference_mass=1e-14)
        quality = self.experiment.summarize()["quality"]
        self.assertEqual(quality["mass_floor_events"], 1)
        self.assertEqual(quality.get("min_target_or_reference_mass"), 1e-14)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.obs = {
            ("r", "v1", "baseline", 1.0): {"whisper": {"attention": [attention(0, 10), attention(1, -10)]}},
            ("r", "v1", "nucleus", 4.0): {"whisper": {"attention": [attention(1, -8), attention(0, 14)]}},
            ("r", "v2", "baseline", 1.0): {"whisper": {"attention": [attention(0, 100), {"interval_id": 1, "excluded": "Fewer than three attention frames"}]}},
            ("r", "v2", "nucleus", 4.0): {"whisper": {"attention": [attention(0, 99), attention(1, 999)]}},
        }

    def summarize(self):
        with patch.object(analysis, "B", 16):
            return analysis.attention_change(self.obs, "r", "nucleus", 4.0, np.random.default_rng(4))

    def test_same_occurrence_baseline_then_equal_verse_aggregation(self):
        result = self.summarize()
        self.assertEqual(result["verses"], ["v1", "v2"])
        self.assertEqual(result["differences"], [3.0, -1.0])
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["n"], 2)  # Three paired intervals are not three replicates.

    def test_duplicate_occurrence_ids_are_rejected(self):
        for arm, k in [("baseline", 1.0), ("nucleus", 4.0)]:
            with self.subTest(arm=arm):
                fixture = SyntheticExperiment()
                values = fixture.row(arm=arm, k=k)["whisper"]["attention"]
                values.append(attention(0, 1000))
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_query_rows_and_reference_support_are_validated(self):
        for defect in ["query", "reference_frames", "unknown_interval"]:
            with self.subTest(defect=defect):
                fixture = SyntheticExperiment()
                value = fixture.row()["whisper"]["attention"][0]
                if defect == "query": value["query_rows"] = [99]
                if defect == "reference_frames": value["reference_frames"] = 9
                if defect == "unknown_interval": value["interval_id"] = 99
                with self.assertRaises((AssertionError, ValueError)):
                    fixture.validate()

    def test_attention_reports_pair_and_exclusion_counts(self):
        result = self.summarize()
        self.assertEqual(result.get("paired_intervals"), 3)
        self.assertEqual(result.get("excluded_intervals"), 1)


if __name__ == "__main__":
    unittest.main()
