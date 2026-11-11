"""Metric regression tests with deliberately constructed examples, not model runs."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import score


def gold(arguments=None):
    return {1: {"id": 1, "answers": [{"name": "convert", "arguments": arguments if arguments is not None else {"x": 1}}]}}


def prediction(value):
    return {1: {"id": 1, "prediction": value}}


def call(arguments=None, name="convert"):
    return [{"name": name, "arguments": arguments if arguments is not None else {"x": 1}}]


class ScoreTests(unittest.TestCase):
    def test_structural_match_ignores_key_order_and_whitespace(self):
        text = ' [ { "arguments": {"b": [2, 3], "a": 1}, "name": "convert"} ] '
        result, _ = score.evaluate(gold({"a": 1, "b": [2, 3]}), prediction(text))
        self.assertEqual(result["metrics"]["complete_call_accuracy"], 1)

    def test_numeric_equality_is_exact_without_bool_or_string_coercion(self):
        for raw_value, expected in (("1.0", 1), ('"1"', 0), ("true", 0), ("1.00000000000000001", 0)):
            with self.subTest(value=raw_value):
                result, _ = score.evaluate(gold(), prediction('[{"name":"convert","arguments":{"x":' + raw_value + '}}]'))
                self.assertEqual(result["metrics"]["complete_call_accuracy"], expected)

    def test_nested_booleans_do_not_equal_integers(self):
        result, _ = score.evaluate(gold({"x": [{"a": True}]}), prediction(call({"x": [{"a": 1}]})))
        self.assertEqual(result["counts"]["wrong_arguments"], 1)

    def test_arrays_and_strings_preserve_order_case_and_whitespace(self):
        for expected, actual in (([1, 2], [2, 1]), ("US", "us"), ("US", "US "), (None, "null")):
            with self.subTest(expected=expected):
                result, _ = score.evaluate(gold({"x": expected}), prediction(call({"x": actual})))
                self.assertEqual(result["counts"]["wrong_arguments"], 1)

    def test_wrong_arguments_detail(self):
        result, details = score.evaluate(gold({"a": 1, "b": 2}), prediction(call({"a": 3, "c": 4})))
        self.assertEqual(result["metrics"]["tool_selection_accuracy"], 1)
        self.assertEqual(result["metrics"]["complete_call_accuracy"], 0)
        self.assertEqual(details[0]["missing_arguments"], ["b"])
        self.assertEqual(details[0]["extra_arguments"], ["c"])
        self.assertEqual(details[0]["incorrect_values"], ["a"])

    def test_empty_arguments_can_be_correct(self):
        result, _ = score.evaluate(gold({}), prediction(call({})))
        self.assertEqual(result["metrics"]["complete_call_accuracy"], 1)

    def test_wrong_tool_is_valid_format_and_conditional_rate_is_undefined(self):
        result, _ = score.evaluate(gold(), prediction(call(name="other")))
        self.assertEqual(result["metrics"]["format_valid_rate"], 1)
        self.assertEqual(result["counts"]["wrong_tool"], 1)
        self.assertIsNone(result["metrics"]["argument_accuracy_given_correct_tool"])

    def test_missing_rows_stay_in_denominator_and_order_does_not_matter(self):
        references = {1: gold()[1], 2: {"id": 2, "answers": call()}}
        result, details = score.evaluate(references, {2: {"id": 2, "prediction": call()}})
        self.assertEqual(result["metrics"]["complete_call_accuracy"], .5)
        self.assertEqual(result["counts"]["missing_prediction"], 1)
        self.assertEqual([d["id"] for d in details], [1, 2])

    def test_malformed_outputs_are_failures_without_repair(self):
        cases = [None, {}, [{"name": "convert", "arguments": "{}"}],
                 [{"name": "convert", "arguments": {}, "extra": 1}],
                 [{"name": " ", "arguments": {}}],
                 '```json\n[{"name":"convert","arguments":{"x":1}}]\n```',
                 'Here is the answer: [{"name":"convert","arguments":{"x":1}}]',
                 '[{"name":"convert","arguments":{"x":1,"x":2}}]',
                 '[{"name":"convert","arguments":{"x":NaN}}]']
        for value in cases:
            with self.subTest(value=value):
                result, _ = score.evaluate(gold(), prediction(value))
                self.assertEqual(result["counts"]["invalid_format"], 1)

    def test_call_reordering_matches_multiset_but_not_ordered_score(self):
        calls = call({"x": 1}) + call({"x": 2}, name="other")
        references = {1: {"id": 1, "answers": calls}}
        result, details = score.evaluate(references, prediction(list(reversed(calls))))
        self.assertEqual(result["metrics"]["complete_request_accuracy"], 1)
        self.assertEqual(result["metrics"]["ordered_exact_match_accuracy"], 0)
        self.assertEqual(details[0]["matched_calls"], 2)

    def test_repeated_calls_are_not_collapsed_to_a_set(self):
        references = {1: {"id": 1, "answers": call() * 2}}
        result, details = score.evaluate(references, prediction(call()))
        self.assertEqual(result["counts"]["wrong_call_count"], 1)
        self.assertEqual(result["metrics"]["call_precision"], 1)
        self.assertEqual(result["metrics"]["call_recall"], .5)
        self.assertEqual(details[0]["false_negative_calls"], 1)

    def test_argument_binding_for_repeated_tool_instances(self):
        references = {1: {"id": 1, "answers": call({"x": 1}) + call({"x": 2})}}
        result, _ = score.evaluate(references, prediction(call({"x": 1}) * 2))
        self.assertEqual(result["metrics"]["tool_selection_accuracy"], 1)
        self.assertEqual(result["metrics"]["call_precision"], .5)
        self.assertEqual(result["metrics"]["call_recall"], .5)
        self.assertEqual(result["counts"]["wrong_arguments"], 1)

    def test_extra_call_counts_as_false_positive(self):
        result, details = score.evaluate(gold(), prediction(call() + call(name="extra")))
        self.assertEqual(result["metrics"]["call_precision"], .5)
        self.assertEqual(result["metrics"]["call_recall"], 1)
        self.assertEqual(result["metrics"]["complete_request_accuracy"], 0)
        self.assertEqual(details[0]["false_positive_calls"], 1)

    def test_empty_prediction_array_is_valid_but_misses_all_calls(self):
        result, _ = score.evaluate(gold(), prediction([]))
        self.assertEqual(result["metrics"]["format_valid_rate"], 1)
        self.assertEqual(result["counts"]["wrong_call_count"], 1)
        self.assertEqual(result["metrics"]["call_f1"], 0)

    def test_invalid_array_gets_no_partial_credit_and_one_placeholder(self):
        invalid = call() + [{"name": "other", "arguments": "bad"}]
        result, details = score.evaluate(gold(), prediction(invalid))
        self.assertEqual(details[0]["matched_calls"], 0)
        self.assertEqual(result["totals"]["prediction_items"], 1)
        self.assertEqual(result["totals"]["invalid_prediction_placeholders"], 1)

    def test_micro_recall_includes_missing_multi_call_examples(self):
        references = {1: {"id": 1, "answers": call() * 2}, 2: {"id": 2, "answers": call()}}
        result, _ = score.evaluate(references, prediction(call() * 2))
        self.assertAlmostEqual(result["metrics"]["call_recall"], 2/3)
        self.assertEqual(result["metrics"]["complete_request_accuracy"], .5)

    def test_legacy_single_mode_rejects_multiple_and_empty_calls(self):
        for value in (call() * 2, []):
            result, _ = score.evaluate(gold(), prediction(value), mode="single")
            self.assertEqual(result["counts"]["invalid_format"], 1)
        with self.assertRaises(score.ScoringError):
            score.evaluate({1: {"id": 1, "answers": call() * 2}}, {}, mode="single")

    def test_category_and_quality_breakdowns(self):
        references = gold()
        references[1]["tools"] = [{"name": "convert"}, {"name": "other"}]
        metadata = {1: {"id": 1, "quality_flags": ["example_flag"]}, 999: {"id": 999, "quality_flags": []}}
        result, _ = score.evaluate(references, prediction(call()), metadata=metadata)
        self.assertEqual(result["breakdowns"]["category"]["multiple"]["total_examples"], 1)
        self.assertEqual(result["breakdowns"]["quality"]["flagged"]["metrics"]["complete_request_accuracy"], 1)
        for bad_meta in ({}, {1: {"quality_flags": "bad"}}):
            with self.assertRaises(score.ScoringError):
                score.evaluate(references, {}, metadata=bad_meta)

    def test_missing_prediction_field_is_an_invalid_output(self):
        result, _ = score.evaluate(gold(), {1: {"id": 1, "raw_output": "ignored"}})
        self.assertEqual(result["invalid_format_reasons"], {"missing_prediction_field": 1})

    def test_jsonl_integrity_errors_abort(self):
        for text in ('{"id":1}\n{"id":1}', '{"id":"1"}', '{"id":true}', '{bad}',
                     '{"id":1,"id":2}', '{"id":1,"prediction":Infinity}'):
            with self.subTest(text=text), self.assertRaises(score.ScoringError):
                score.read_rows(text.encode(), "predictions")

    def test_bad_references_and_unknown_ids_abort(self):
        for references, predictions in (({}, {}), ({1: {"id": 1, "answers": []}}, {}),
                                         (gold(), {2: {"id": 2, "prediction": call()}})):
            with self.subTest(gold=references), self.assertRaises(score.ScoringError):
                score.evaluate(references, predictions)

    def test_exact_large_integer(self):
        result, _ = score.evaluate(gold({"x": 9007199254740993}), prediction('[{"name":"convert","arguments":{"x":9007199254740992.0}}]'))
        self.assertEqual(result["counts"]["wrong_arguments"], 1)

    def test_cli_reads_archive_and_writes_consistent_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = path / "dev.jsonl"
            source.write_text(json.dumps(gold()[1]) + "\n", encoding="utf-8")
            with tarfile.open(path / "data.tar.gz", "w:gz") as archive:
                archive.add(source, arcname="data/dev.jsonl")
            predictions = path / "predictions.jsonl"
            predictions.write_text(json.dumps(prediction(call())[1]) + "\n", encoding="utf-8")
            args = [sys.executable, str(Path(score.__file__)), "--gold", str(path / "data.tar.gz"), "--split", "dev", "--predictions", str(predictions),
                    "--output", str(path / "summary.json"), "--details", str(path / "details.jsonl")]
            completed = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["metrics"]["complete_call_accuracy"], 1)
            self.assertEqual(summary, json.loads((path / "summary.json").read_text()))
            self.assertEqual(json.loads((path / "details.jsonl").read_text())["status"], "correct")
            # Refuse input overwrite and an implicit archive split.
            for extra in (["--output", str(predictions)], []):
                bad_args = args[:args.index("--output")] + extra
                if not extra:
                    del bad_args[bad_args.index("--split"):bad_args.index("--split") + 2]
                failed = subprocess.run(bad_args, capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
