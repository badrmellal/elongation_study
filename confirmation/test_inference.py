import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import CACHE, WHISPER_REV, merge_occurrences, query_rows, score, word_span
from transformers import WhisperProcessor


class InferenceTests(unittest.TestCase):
    def test_control_tokens_are_not_recognition_errors(self):
        from inference import decode_transcript
        tokenizer = WhisperProcessor.from_pretrained(
            CACHE / "models--tarteel-ai--whisper-base-ar-quran" / "snapshots" / WHISPER_REV,
            local_files_only=True).tokenizer
        text = "أَلَا يَعْلَمُ مَنْ خَلَقَ وَهُوَ اللَّطِيفُ الْخَبِيرُ"
        ids = [50272, 50359, 50363] + tokenizer(text, add_special_tokens=False).input_ids
        self.assertEqual(score(text, decode_transcript(tokenizer, ids))["wer"], 0)

    def test_wer_counts(self):
        self.assertEqual(score("one two three", "one four")["wer"], 2 / 3)
        self.assertEqual(score("one", "one two three")["wer"], 2)
        self.assertEqual(score("قُلْ هُوَ", "قل هو")["errors"], 0)
        with self.assertRaises(ValueError):
            score("", "x")

    def test_query_shift_and_special_exclusion(self):
        offsets = [(0, 0), (0, 0), (0, 2), (2, 3), (4, 6), (0, 0)]
        self.assertEqual(query_rows(offsets, (0, 3)), [1, 2])
        self.assertEqual(query_rows(offsets, (4, 6)), [3])

    def test_word_spans(self):
        self.assertEqual(word_span("قل هو الله", 8), (6, 10))
        with self.assertRaises(ValueError):
            word_span("قل هو", 2)

    def test_merge_preserves_character_sources(self):
        merged = merge_occurrences([{"start": 0.1, "end": 0.2, "char_idx": 2},
                                    {"start": 0.19, "end": 0.3, "char_idx": 3}], 1000)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["char_indices"], [2, 3])
        self.assertEqual(merged[0]["end_sample"], 300)


if __name__ == "__main__":
    unittest.main()
