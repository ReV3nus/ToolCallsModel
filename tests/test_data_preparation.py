"""Checks for held-out protection and source-preserving full-data preparation."""
import sys
from pathlib import Path
import unittest
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_full_data import allocate, group_id, source_line
from score import same_value, strict_json


def record(source_id, query):
    return {"id": source_id, "query": query, "tools": '[{"name":"f","parameters":{}}]',
            "answers": '[{"name":"f","arguments":{}}]'}


class FullDataTests(unittest.TestCase):
    def test_new_related_record_follows_held_out_anchor(self):
        first = record(1, "Fetch the product details for item 123 in the catalog.")
        related = record(2, "Fetch the product details for item 456 in the catalog.")
        other = record(3, "Return a random word.")
        parts, _, _ = allocate([first, related, other], {1: "test"}, 5300)
        self.assertEqual(group_id(first), group_id(related))
        self.assertEqual({r["id"] for r in parts["test"]}, {1, 2})
        self.assertEqual(sum(map(len, parts.values())), 3)

    def test_conflicting_group_anchors_abort(self):
        first = record(1, "Fetch the product details for item 123 in the catalog.")
        related = record(2, "Fetch the product details for item 456 in the catalog.")
        with self.assertRaises(ValueError):
            allocate([first, related], {1: "train", 2: "test"}, 5300)

    def test_unknown_anchor_and_duplicate_id_abort(self):
        first = record(1, "Return a random word.")
        for rows, anchors in (([first], {99: "dev"}), ([first, first], {})):
            with self.assertRaises(ValueError):
                allocate(rows, anchors, 5300)

    def test_original_numeric_literals_survive_preparation(self):
        original = record(1, "Return the supplied number.")
        original["answers"] = '[{"name":"f","arguments":{"x":1.00000000000000001,"n":9007199254740993}}]'
        prepared = source_line(original)
        self.assertIn("1.00000000000000001", prepared)
        self.assertTrue(same_value(strict_json(prepared)["answers"], strict_json(original["answers"])))

    def test_cli_refuses_to_overwrite_the_source_as_documentation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "source.json"
            source.write_text("[]", encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_full_data.py"
            result = subprocess.run([sys.executable, str(script), "--input", str(source), "--m1-archive", str(folder / "m1.tar.gz"),
                                     "--output", str(folder / "output"), "--documentation", str(source)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not overwrite source inputs", result.stderr)
            self.assertEqual(source.read_text(), "[]")


if __name__ == "__main__":
    unittest.main()
