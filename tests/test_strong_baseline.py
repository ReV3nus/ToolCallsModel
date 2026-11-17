"""Tests for target masking, loss padding, exact values, and inference isolation."""
import copy
from decimal import Decimal
import importlib.util
import unittest

from score import strict_json, same_value
from strong_baseline import json_text, messages, training_example, ResponseCollator


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, value, **kwargs):
        self.seen = value
        return [10, 11, 12]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


class StrongDataTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": 1, "query": "Convert 1.", "tools": [{"name": "convert", "parameters": {"x": {"type": "int"}}}],
                    "answers": [{"name": "convert", "arguments": {"x": 1}}]}

    def test_reference_or_metadata_changes_cannot_change_prompt(self):
        changed = copy.deepcopy(self.row)
        changed.update(id=55, answers=[{"name": "wrong", "arguments": {}}], reference_call_count=8, metadata={"label": "secret"})
        self.assertEqual(messages(self.row), messages(changed))
        payload = strict_json(messages(self.row)[1]["content"])
        self.assertEqual(set(payload), {"query", "tools"})

    def test_only_complete_response_and_eos_receive_loss(self):
        tokenizer = FakeTokenizer()
        feature = training_example(tokenizer, self.row, 1000)
        self.assertEqual(feature["labels"][:3], [-100]*3)
        self.assertEqual(feature["input_ids"][3:], feature["labels"][3:])
        self.assertEqual(feature["labels"][-1], tokenizer.eos_token_id)
        decoded = ''.join(chr(x) for x in feature["labels"][3:-1])
        self.assertTrue(same_value(strict_json(decoded), self.row["answers"]))

    def test_long_example_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            training_example(FakeTokenizer(), self.row, 4)

    def test_decimal_types_and_repeated_calls_survive_formatting(self):
        self.row["answers"] = [{"name": "convert", "arguments": {"x": Decimal("1.00000000000000001")}}]*2
        feature = training_example(FakeTokenizer(), self.row, 1000)
        decoded = ''.join(chr(x) for x in feature["labels"][3:-1])
        self.assertTrue(same_value(strict_json(decoded), self.row["answers"]))
        self.assertIn("1.00000000000000001", decoded)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training environment not installed")
    def test_padding_tokens_are_masked_out_of_loss(self):
        rows = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {"input_ids": [3], "attention_mask": [1], "labels": [3]}]
        batch = ResponseCollator(0)(rows)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])


if __name__ == "__main__":
    unittest.main()
