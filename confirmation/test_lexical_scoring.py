"""Regression checks for the nonlexical rub-el-hizb scoring amendment."""
import unittest
from inference import score


class LexicalScoringTest(unittest.TestCase):
    def test_decorative_symbol_is_not_a_word(self):
        reference = "۞ افلا يعلم اذا بعثر ما في القبور"
        hypothesis = "افلا يعلم اذا بعثر ما في القبور"
        self.assertEqual(score(reference, hypothesis)["wer"], 0.125)
        corrected = score(reference.replace("\u06de", ""), hypothesis.replace("\u06de", ""))
        self.assertEqual(corrected["reference_words"], 7)
        self.assertEqual(corrected["wer"], 0)

    def test_lexical_errors_still_count(self):
        self.assertEqual(score("افلا يعلم", "افلا")["wer"], 0.5)


if __name__ == "__main__":
    unittest.main()
