"""Synthetic, inference-free checks. Run with python -m unittest confirmation.test_controls -v."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from confirmation import controls


def voiced(length=16003, sr=16000):
    t = np.arange(length, dtype=np.float64) / sr
    phase = 2 * np.pi * (173 * t + 13 * t ** 2)
    envelope = 0.7 + 0.2 * np.sin(2 * np.pi * 2.7 * t)
    return (envelope * (0.35 * np.sin(phase + 0.3) + 0.12 * np.sin(2 * phase + 0.1))).astype(np.float32)


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.x = voiced()
        self.bounds = [(1601, 4404), (6407, 9810), (11013, 14018)]
        self.intervals = [(start / self.sr, end / self.sr) for start, end in self.bounds]

    def assert_valid(self, result, length):
        self.assertEqual(set(result), {"audio", "intervals", "backend", "metadata"})
        audio, metadata = result["audio"], result["metadata"]
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (length,))
        self.assertGreater(audio.size, 0)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(float(np.max(np.abs(audio))), 0.01)
        self.assertEqual(metadata["absmax"], float(np.max(np.abs(audio))))
        self.assertEqual(metadata["samples_exceeding_one"], int(np.count_nonzero(np.abs(audio) > 1)))
        self.assertEqual(metadata["input_samples"], len(self.x))
        self.assertEqual(metadata["output_samples"], length)
        previous = 0
        for (i0, i1), seconds in zip(metadata["output_intervals_samples"], result["intervals"]):
            self.assertIs(type(i0), int)
            self.assertIs(type(i1), int)
            self.assertGreaterEqual(i0, previous)
            self.assertLess(i0, i1)
            self.assertLessEqual(i1, length)
            self.assertEqual(seconds, (i0 / self.sr, i1 / self.sr))
            previous = i1
        self.assertEqual(len(result["intervals"]), len(metadata["output_intervals_samples"]))
        self.assertIsInstance(result["backend"], str)
        json.dumps(metadata, allow_nan=False)

    def assert_outside_unchanged(self, result, gaps=None):
        audio = result["audio"]
        input_cursor = output_cursor = 0
        for index, ((i0, i1), (o0, o1)) in enumerate(zip(
            result["metadata"]["input_intervals_samples"], result["metadata"]["output_intervals_samples"]
        )):
            self.assertEqual(self.x[input_cursor:i0].tobytes(), audio[output_cursor:o0].tobytes())
            input_cursor, output_cursor = i1, o1 + (0 if gaps is None else gaps[index])
        self.assertEqual(self.x[input_cursor:].tobytes(), audio[output_cursor:].tobytes())

    def test_canonicalize_merge_clip_and_discard_invalid(self):
        sr = 1000
        x = voiced(1000, sr)
        intervals = [(.4, .5), (.0996, .2), (.2, .3), (.15, .25), (-.02, .05),
                     (.8, 2), (.5, .5001), (.7, .6), (0, 0), (float("nan"), .6),
                     (.6, float("inf")), (float("-inf"), .2), (-2, -1), (2, 3),
                     None, (1,), (1, 2, 3), ("bad", .5), (.301, .35)]
        output = controls.make_conditions(x, sr, intervals, 1)
        for result in output.values():
            self.assertEqual(result["metadata"]["input_intervals_samples"],
                             [(0, 50), (99, 350), (400, 500), (800, 1000)])
            self.assertEqual(result["metadata"]["source_interval_indices"], [[4], [1, 2, 3, 18], [0], [5]])
            self.assertEqual(result["audio"].tobytes(), x.tobytes())

    def test_floor_endpoints_before_merging(self):
        result = controls.make_conditions(voiced(128, 128), 128,
                                          [(10.5 / 128, 20.5 / 128), (20.49 / 128, 30.5 / 128)], 1)["nucleus"]
        self.assertEqual(result["metadata"]["input_intervals_samples"], [(10, 30)])

    def test_exact_matched_lengths_awkward_boundaries_and_fractional_k(self):
        # Subsample offsets floor to the independently specified odd lengths.
        intervals = [((i0 + .24) / self.sr, (i1 + .76) / self.sr) for i0, i1 in self.bounds]
        original = self.x.tobytes()
        for k in (1, 2, 4, 6, 1.5):
            with self.subTest(k=k):
                additions = [round((k - 1) * (i1 - i0)) for i0, i1 in self.bounds]
                target = len(self.x) + sum(additions)
                results = controls.make_conditions(self.x, self.sr, intervals, k)
                self.assertEqual(set(results), {"nucleus", "global", "silence", "nucleus_edge"})
                for name, result in results.items():
                    self.assert_valid(result, target)
                    self.assertEqual(result["metadata"]["input_intervals_samples"], self.bounds)
                    self.assertEqual(result["metadata"]["interval_added_samples"], additions)
                    self.assertEqual(result["metadata"]["added_samples"], sum(additions))
                    self.assertTrue(all(record["correction_samples"] == 0 for record in result["metadata"]["length_corrections"]))
                    if name in ("nucleus", "nucleus_edge"):
                        self.assert_outside_unchanged(result)
                offset = 0
                for (i0, i1), added, local, silence in zip(
                    self.bounds, additions, results["nucleus"]["metadata"]["output_intervals_samples"],
                    results["silence"]["metadata"]["output_intervals_samples"]
                ):
                    self.assertEqual(local, (i0 + offset, i1 + offset + added))
                    self.assertEqual(silence, (i0 + offset, i1 + offset))
                    np.testing.assert_array_equal(results["silence"]["audio"][silence[0]:silence[1]], self.x[i0:i1])
                    np.testing.assert_array_equal(results["silence"]["audio"][silence[1]:silence[1] + added], np.zeros(added))
                    offset += added
                self.assert_outside_unchanged(results["silence"], additions)
                self.assertEqual(results["global"]["metadata"]["output_intervals_samples"],
                                 [(round(i0 * target / len(self.x)), round(i1 * target / len(self.x))) for i0, i1 in self.bounds])
        self.assertEqual(self.x.tobytes(), original)

    def test_premerged_occurrence_count_and_sample_grid_are_stable(self):
        # This quotient multiplies back to one ULP below its integer sample.
        samples = np.arange(1, len(self.x))
        below_grid = samples[(samples / self.sr) * self.sr < samples]
        self.assertGreater(len(below_grid), 0)
        boundary = int(below_grid[len(below_grid) // 2])
        bounds = [(boundary - 2001, boundary), (boundary + 2, boundary + 2103)]
        intervals = [(i0 / self.sr, i1 / self.sr) for i0, i1 in bounds]
        for k in (1, 2, 4, 6):
            results = controls.make_conditions(self.x, self.sr, intervals, k)
            for result in results.values():
                self.assertEqual(result["metadata"]["input_intervals_samples"], bounds)
                self.assertEqual(result["metadata"]["source_interval_indices"], [[0], [1]])
                self.assertEqual(len(result["intervals"]), len(bounds))
        for result in controls.sham_conditions(self.x, self.sr, intervals).values():
            self.assertEqual(result["metadata"]["input_intervals_samples"], bounds)
            self.assertEqual(len(result["intervals"]), len(bounds))

    def test_unity_bit_identity_and_no_synthesis(self):
        self.x[17] = -0.0
        with patch.object(controls.librosa.effects, "time_stretch", side_effect=AssertionError("unity synthesized")):
            for result in controls.make_conditions(self.x, self.sr, self.intervals, 1).values():
                self.assertEqual(result["audio"].tobytes(), self.x.tobytes())
                self.assertEqual(result["metadata"]["synthesis_calls"], 0)
                self.assertFalse(np.shares_memory(result["audio"], self.x))

    def test_no_valid_intervals_identity_even_at_k6(self):
        for result in controls.make_conditions(self.x, self.sr, [(1, -1)], 6).values():
            self.assertEqual(result["audio"].tobytes(), self.x.tobytes())
            self.assertEqual(result["intervals"], [])
            self.assertEqual(result["metadata"]["added_samples"], 0)

    def test_edge_endpoints_weights_interior_and_original_quarter_limit(self):
        for bounds in (self.bounds, [(100, 504)]):
            intervals = [(i0 / self.sr, i1 / self.sr) for i0, i1 in bounds]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results = controls.make_conditions(self.x, self.sr, intervals, 4)
            nucleus, edge = results["nucleus"]["audio"], results["nucleus_edge"]["audio"]
            for (i0, i1), (o0, o1), count in zip(bounds, results["nucleus"]["metadata"]["output_intervals_samples"],
                                                results["nucleus_edge"]["metadata"]["edge_samples"]):
                self.assertEqual(count, min(self.sr // 100, (i1 - i0) // 4))
                self.assertEqual(edge[o0], self.x[i0])
                self.assertEqual(edge[o1 - 1], self.x[i1 - 1])
                self.assertEqual(edge[o0 + count - 1], nucleus[o0 + count - 1])
                self.assertEqual(edge[o1 - count], nucleus[o1 - count])
                weights = (1 + np.cos(np.linspace(0, np.pi, count))) / 2
                np.testing.assert_array_equal(edge[o0:o0 + count],
                                              (self.x[i0:i0 + count] * weights + nucleus[o0:o0 + count] * (1 - weights)).astype(np.float32))
                np.testing.assert_array_equal(edge[o1 - count:o1],
                                              (self.x[i1 - count:i1] * weights[::-1] + nucleus[o1 - count:o1] * (1 - weights[::-1])).astype(np.float32))
                np.testing.assert_array_equal(edge[o0 + count:o1 - count], nucleus[o0 + count:o1 - count])
            self.assert_outside_unchanged(results["nucleus_edge"])
            self.assertIn("not proof of artifact removal", results["nucleus_edge"]["metadata"]["interpretation"])

    def test_short_cuts_have_finite_audio_and_edge_counts_zero_or_one(self):
        for length in (1, 3, 4, 7):
            with self.subTest(length=length), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results = controls.make_conditions(self.x, self.sr, [(100 / self.sr, (100 + length) / self.sr)], 4)
                for result in results.values():
                    self.assert_valid(result, len(self.x) + 3 * length)
                edge = results["nucleus_edge"]
                self.assertEqual(edge["metadata"]["edge_samples"], [length // 4])
                self.assertEqual(results["nucleus"]["metadata"]["length_corrections"][0]["hop_length"], length)
                if length >= 4:
                    i0, i1 = edge["metadata"]["output_intervals_samples"][0]
                    self.assertEqual(edge["audio"][i0], self.x[100])
                    self.assertEqual(edge["audio"][i1 - 1], self.x[99 + length])
                for result in controls.sham_conditions(self.x, self.sr, [(100 / self.sr, (100 + length) / self.sr)]).values():
                    self.assert_valid(result, len(self.x))

    def test_forced_sham_calls_real_stft_phase_vocoder_and_istft(self):
        with patch.object(controls.librosa.effects, "time_stretch", side_effect=AssertionError("sham used time_stretch")), \
             patch.object(controls.librosa, "stft", wraps=controls.librosa.stft) as stft, \
             patch.object(controls.librosa, "phase_vocoder", wraps=controls.librosa.phase_vocoder) as pv, \
             patch.object(controls.librosa, "istft", wraps=controls.librosa.istft) as istft:
            results = controls.sham_conditions(self.x, self.sr, self.intervals)
        self.assertEqual(set(results), {"sham_pv_local", "sham_pv_global"})
        for stage in (stft, pv, istft):
            self.assertEqual(stage.call_count, len(self.bounds) + 1)
        for call in pv.call_args_list:
            self.assertEqual(call.kwargs["rate"], 1.0)
        for call, (i0, i1) in zip(stft.call_args_list, self.bounds):
            np.testing.assert_array_equal(call.args[0], self.x[i0:i1])
        np.testing.assert_array_equal(stft.call_args_list[-1].args[0], self.x)
        for result in results.values():
            self.assert_valid(result, len(self.x))
            self.assertEqual(result["metadata"]["output_intervals_samples"], self.bounds)
            self.assertTrue(result["metadata"]["forced_round_trip"])
            self.assertGreater(result["metadata"]["synthesis_calls"], 0)
        self.assert_outside_unchanged(results["sham_pv_local"])

    def test_sham_propagates_synthesis_output_not_original_copy(self):
        real_istft = controls.librosa.istft
        def attenuated_synthesis(*args, **kwargs):
            return real_istft(*args, **kwargs) * np.float32(.5)
        with patch.object(controls.librosa, "istft", side_effect=attenuated_synthesis):
            results = controls.sham_conditions(self.x, self.sr, self.intervals)
        np.testing.assert_allclose(results["sham_pv_global"]["audio"], self.x * .5, atol=2e-5, rtol=1e-4)
        for i0, i1 in self.bounds:
            np.testing.assert_allclose(results["sham_pv_local"]["audio"][i0:i1], self.x[i0:i1] * .5, atol=2e-5, rtol=1e-4)
        self.assert_outside_unchanged(results["sham_pv_local"])

    def test_empty_local_sham_still_runs_global_synthesis(self):
        with patch.object(controls.librosa, "istft", wraps=controls.librosa.istft) as istft:
            results = controls.sham_conditions(self.x, self.sr, [])
        self.assertEqual(istft.call_count, 1)
        self.assertEqual(results["sham_pv_local"]["metadata"]["synthesis_calls"], 0)
        self.assertEqual(results["sham_pv_local"]["audio"].tobytes(), self.x.tobytes())

    def test_invalid_audio_rate_and_factor(self):
        for x in (np.array([]), np.zeros((2, 4)), np.array([np.nan]), np.array([np.inf]),
                  np.array([1.1]), np.array([1j]), np.array(["x"])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                controls.make_conditions(x, self.sr, [], 1)
            with self.assertRaises(ValueError):
                controls.sham_conditions(x, self.sr, [])
        for sr in (0, -1, 16000.5, True):
            with self.subTest(sr=sr), self.assertRaises(ValueError):
                controls.make_conditions(self.x, sr, [], 1)
        for k in (0, .5, -1, float("nan"), float("inf")):
            with self.subTest(k=k), self.assertRaises(ValueError):
                controls.make_conditions(self.x, self.sr, [], k)

    def test_backend_small_corrections_reported_large_errors_rejected(self):
        intervals = [(0, len(self.x) / self.sr)]
        target = len(self.x) * 2
        for difference in (-2, -1, 0, 1, 2):
            with self.subTest(difference=difference), patch.object(
                controls.librosa.effects, "time_stretch", return_value=np.full(target + difference, .2, dtype=np.float32)
            ):
                results = controls.make_conditions(self.x, self.sr, intervals, 2)
                for name in ("nucleus", "global", "nucleus_edge"):
                    self.assertEqual(len(results[name]["audio"]), target)
                    self.assertEqual(results[name]["metadata"]["length_corrections"][0]["correction_samples"], -difference)
        for raw in (np.ones(target + 3), np.ones(target - 3), np.array([]), np.array([np.nan])):
            with patch.object(controls.librosa.effects, "time_stretch", return_value=raw), self.assertRaises(RuntimeError):
                controls.make_conditions(self.x, self.sr, intervals, 2)

    def test_overshoot_preserved_without_clipping_or_normalization(self):
        def overshoot(segment, *, rate, **kwargs):
            return np.full(round(len(segment) / rate), 1.2, dtype=np.float32)
        with patch.object(controls.librosa.effects, "time_stretch", side_effect=overshoot):
            results = controls.make_conditions(self.x, self.sr, self.intervals, 2)
        for name in ("nucleus", "nucleus_edge", "global"):
            self.assertGreater(results[name]["metadata"]["samples_exceeding_one"], 0)
            self.assertEqual(results[name]["metadata"]["absmax"], float(np.float32(1.2)))
        np.testing.assert_array_equal(results["global"]["audio"], np.full(len(results["global"]["audio"]), 1.2, dtype=np.float32))
        for i0, i1 in results["nucleus"]["metadata"]["output_intervals_samples"]:
            np.testing.assert_array_equal(results["nucleus"]["audio"][i0:i1], np.full(i1 - i0, 1.2, dtype=np.float32))
        self.assert_outside_unchanged(results["nucleus"])
        self.assert_outside_unchanged(results["nucleus_edge"])

    @unittest.skipUnless(os.access(controls.RUBBERBAND_PATH, os.X_OK), "Rubber Band CLI unavailable")
    def test_actual_rubberband_smoke_versions_float_io_unity_and_concurrent_temps(self):
        for k in (1, 2, 4, 6):
            with self.subTest(k=k):
                results = controls.make_conditions(self.x, self.sr, self.intervals, k, include_rubberband=True)
                target = len(self.x) + sum(round((k - 1) * (i1 - i0)) for i0, i1 in self.bounds)
                self.assertEqual(len(results), 6)
                for name, result in results.items():
                    self.assert_valid(result, target)
                    if k == 1:
                        self.assertEqual(result["audio"].tobytes(), self.x.tobytes())
                    if name.endswith("_rb"):
                        self.assertEqual(result["metadata"]["rubberband"]["version"], "4.0.0")
                        self.assertEqual(result["metadata"]["rubberband"]["wav_subtype"], "FLOAT")
                        self.assertEqual(result["metadata"]["rubberband"]["mode"], "offline")
                        self.assertTrue(all(abs(record["correction_samples"]) <= 2 for record in result["metadata"]["length_corrections"]))
                self.assertEqual(results["nucleus_rb"]["metadata"]["rubberband"], results["global_rb"]["metadata"]["rubberband"])
                self.assert_outside_unchanged(results["nucleus_rb"])
        info = controls._rubberband_info()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(controls._rb, self.x, self.sr, len(self.x) * 4, info) for _ in range(2)]
            outputs = [job.result()[0] for job in jobs]
        np.testing.assert_array_equal(*outputs)

    def test_rubberband_requires_queried_fixed_version_and_supported_flags(self):
        def run(command, **kwargs):
            output = "4.0.0\n" if command[-1] == "--version" else "--fine --quiet --ignore-clipping --time"
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        with patch.object(controls.subprocess, "run", side_effect=run) as process:
            info = controls._rubberband_info.__wrapped__()
        self.assertEqual([call.args[0][-1] for call in process.call_args_list], ["--help", "--full-help", "--version"])
        self.assertEqual(info["flags"], ["--fine", "--quiet", "--ignore-clipping"])
        with patch.object(controls.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="9.9.9", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "version must be"):
                controls._rubberband_info.__wrapped__()

    def test_rubberband_float_files_length_guard_reporting_and_temp_cleanup(self):
        info = {"path": controls.RUBBERBAND_PATH, "flags": ["--fine", "--quiet", "--ignore-clipping"]}
        target = len(self.x) * 4
        for difference in (-3, -2, 0, 2, 3):
            directories = []
            def fake_cli(command, **kwargs):
                source, dest = Path(command[-2]), Path(command[-1])
                directories.append(source.parent)
                self.assertEqual(controls.sf.info(source).subtype, "FLOAT")
                self.assertEqual(kwargs["cwd"], str(source.parent))
                controls.sf.write(dest, voiced(target + difference), self.sr, subtype="FLOAT")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            with self.subTest(difference=difference), patch.object(controls.subprocess, "run", side_effect=fake_cli):
                if abs(difference) > 2:
                    with self.assertRaisesRegex(RuntimeError, "exceeds 2 samples"):
                        controls._rb(self.x, self.sr, target, info)
                else:
                    audio, record = controls._rb(self.x, self.sr, target, info)
                    self.assertEqual(len(audio), target)
                    self.assertEqual(record["correction_samples"], -difference)
            self.assertTrue(directories)
            self.assertTrue(all(not directory.exists() for directory in directories))


if __name__ == "__main__":
    unittest.main()
