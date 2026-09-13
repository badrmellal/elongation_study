import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_audit import edit_counts
from inference import score


class ErrorAuditTests(unittest.TestCase):
    def test_perfect_and_empty(self):
        self.assertEqual(edit_counts("قُلْ هُوَ", "قل هو")["errors"], 0)
        out = edit_counts("one two", "")
        self.assertEqual(out["deletions"], 2)
        self.assertEqual(out["wer"], 1)

    def test_substitution_deletion_insertion(self):
        self.assertEqual(edit_counts("one two", "one three")["substitutions"], 1)
        self.assertEqual(edit_counts("one two three", "one three")["deletions"], 1)
        self.assertEqual(edit_counts("one two", "one extra two")["insertions"], 1)
        self.assertEqual(edit_counts("one", "one two three")["wer"], 2)

    def test_all_short_sequences_match_distance(self):
        seqs = [list(s) for n in range(4) for s in itertools.product(["a", "b"], repeat=n)]
        for ref in seqs[1:]:
            for hyp in seqs:
                a, b = " ".join(ref), " ".join(hyp)
                out = edit_counts(a, b)
                self.assertEqual(out["errors"], score(a, b)["errors"])
                self.assertEqual(out["correct"] + out["substitutions"] + out["deletions"], len(ref))


if __name__ == "__main__":
    unittest.main()
