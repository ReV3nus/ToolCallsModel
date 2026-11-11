"""Score xLAM call arrays by source ID. Python standard library only.

Predictions: JSONL rows {"id": <integer>, "prediction": <call array or JSON text>}.
Gold: processed xLAM JSONL, or an M1 tar.gz plus --split. See scoring.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, DecimalException
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from task_types import describe

VERSION = "2.0"


class ScoringError(ValueError):
    """The evaluation files cannot be aligned or their references are invalid."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def strict_json(text):
    # Decimal preserves exact JSON decimal values; bool must remain distinct.
    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_float=Decimal,
                          parse_constant=reject_constant)
    except DecimalException as exc:
        raise ValueError("numeric literal exceeds the decimal parser's range") from exc


def call_array(value, mode="multiset"):
    """Validate structure only; a well-formed but wrong tool is still valid JSON."""
    if isinstance(value, str):
        try:
            value = strict_json(value)
        except (ValueError, RecursionError):
            return None, "invalid_json"
    if not isinstance(value, list):
        return None, "expected_call_array"
    if mode == "single" and len(value) != 1:
        return None, "expected_exactly_one_call"
    for call in value:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            return None, "expected_name_and_arguments_only"
        if not isinstance(call["name"], str) or not call["name"].strip():
            return None, "invalid_tool_name"
        if not isinstance(call["arguments"], dict):
            return None, "arguments_not_object"
    return value, None


def frozen(value):
    """Hashable JSON value with exact numeric equality and tagged booleans."""
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, Decimal, float)):
        number = Decimal(str(value)) if isinstance(value, float) else value
        if isinstance(number, Decimal) and not number.is_finite():
            raise ValueError("non-finite numeric value")
        return ("number", number)
    if isinstance(value, dict):
        return ("object", tuple(sorted((k, frozen(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ("array", tuple(map(frozen, value)))
    if value is None:
        return ("null",)
    if isinstance(value, str):
        return ("string", value)
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def same_value(left, right):
    """JSON structural equality with exact numbers and no type coercion."""
    return frozen(left) == frozen(right)


def read_payload(path, split=None):
    if split is not None:
        try:
            with tarfile.open(path, "r:gz") as archive:
                member = archive.getmember(f"data/{split}.jsonl")
                if not member.isfile():
                    raise ScoringError("gold archive member must be a regular file")
                return archive.extractfile(member).read()
        except (tarfile.TarError, KeyError) as exc:
            raise ScoringError(f"cannot read data/{split}.jsonl from {path}: {exc}") from exc
    if path.name.endswith((".tar.gz", ".tgz")):
        raise ScoringError("--split is required when --gold is a tar archive")
    return path.read_bytes()


def read_rows(payload, label):
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ScoringError(f"{label}: expected UTF-8 JSONL") from exc
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = strict_json(line)
        except (ValueError, RecursionError) as exc:
            raise ScoringError(f"{label}, line {line_number}: malformed JSONL: {exc}") from exc
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ScoringError(f"{label}, line {line_number}: id must be a JSON integer")
        source_id = row["id"]
        if source_id in rows:
            raise ScoringError(f"{label}: duplicate id {source_id}")
        rows[source_id] = row
    return rows


def validate_gold(rows, mode="multiset"):
    if not rows:
        raise ScoringError("gold contains no examples")
    calls = {}
    for source_id, row in rows.items():
        call, error = call_array(row.get("answers"), mode)
        if error:
            raise ScoringError(f"gold id {source_id}: {error}")
        if not call:
            raise ScoringError(f"gold id {source_id}: empty reference list is outside this dataset's task")
        calls[source_id] = call
    return calls


def aggregate(details):
    n = len(details)
    counts = Counter(d["status"] for d in details)
    complete = sum(d["complete_call_correct"] for d in details)
    tools = sum(d["tool_correct"] for d in details)
    reference = sum(d["reference_call_count"] for d in details)
    predicted = sum(d["prediction_items"] for d in details)
    matched = sum(d["matched_calls"] for d in details)
    named = sum(d["matched_tool_names"] for d in details)
    precision, recall = matched / predicted if predicted else 0, matched / reference
    name_p, name_r = named / predicted if predicted else 0, named / reference
    return {
        "total_examples": n,
        "provided_predictions": n - counts["missing_prediction"],
        "counts": {key: counts[key] for key in ("correct", "wrong_call_count", "wrong_tool", "wrong_arguments", "invalid_format", "missing_prediction")},
        "metrics": {
            "complete_request_accuracy": complete / n,
            "complete_call_accuracy": complete / n,
            "ordered_exact_match_accuracy": sum(d["ordered_exact_match"] for d in details) / n,
            "call_count_accuracy": sum(d["call_count_correct"] for d in details) / n,
            "tool_selection_accuracy": tools / n,
            "format_valid_rate": sum(d["format_valid"] for d in details) / n,
            "argument_accuracy_given_correct_tool": complete / tools if tools else None,
            "call_precision": precision, "call_recall": recall,
            "call_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "tool_name_precision": name_p, "tool_name_recall": name_r,
            "tool_name_f1": 2 * name_p * name_r / (name_p + name_r) if name_p + name_r else 0,
        },
        "totals": {"reference_calls": reference, "prediction_items": predicted, "matched_calls": matched,
                   "matched_tool_names": named, "invalid_prediction_placeholders": counts["invalid_format"]},
        "denominators": {"request_metrics": n, "argument_accuracy_given_correct_tool": tools,
                         "micro_precision": predicted, "micro_recall": reference},
    }


def evaluate(gold_rows, prediction_rows, mode="multiset", metadata=None):
    if mode not in ("multiset", "single"):
        raise ScoringError("mode must be multiset or single")
    gold_calls = validate_gold(gold_rows, mode)
    unknown = sorted(prediction_rows.keys() - gold_rows.keys())
    if unknown:
        raise ScoringError(f"predictions contain ids outside gold: {unknown[:10]}")
    if metadata is not None and gold_rows.keys() - metadata.keys():
        raise ScoringError("metadata must cover every gold ID")
    details, format_errors = [], Counter()
    for source_id, gold in gold_calls.items():
        gold_names = Counter(c["name"] for c in gold)
        gold_inventory = Counter(frozen(c) for c in gold)
        detail = {"id": source_id, "gold_tools": [c["name"] for c in gold], "predicted_tools": [],
                  "reference_call_count": len(gold), "predicted_call_count": None,
                  "prediction_items": 0, "matched_calls": 0, "matched_tool_names": 0,
                  "status": "missing_prediction", "format_valid": False,
                  "tool_correct": False, "arguments_correct": None,
                  "complete_call_correct": False, "ordered_exact_match": False, "call_count_correct": False,
                  "task": describe(gold_rows[source_id], gold)}
        if metadata is not None:
            flags = metadata[source_id].get("quality_flags")
            if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
                raise ScoringError("metadata quality_flags must be an array of strings")
            detail["task"]["quality"] = "flagged" if flags else "unflagged"
        if source_id in prediction_rows:
            row = prediction_rows[source_id]
            if "prediction" in row:
                call, error = call_array(row["prediction"], mode)
            else:
                call, error = None, "missing_prediction_field"
            if error:
                detail.update(status="invalid_format", format_error=error, prediction_items=1)
                format_errors[error] += 1
            else:
                inventory = Counter(frozen(c) for c in call)
                names = Counter(c["name"] for c in call)
                match = inventory == gold_inventory
                same_tools = names == gold_names
                count_correct = len(call) == len(gold)
                detail.update(format_valid=True, predicted_tools=[c["name"] for c in call],
                              predicted_call_count=len(call), prediction_items=len(call),
                              matched_calls=sum((inventory & gold_inventory).values()),
                              matched_tool_names=sum((names & gold_names).values()),
                              complete_call_correct=match, ordered_exact_match=same_value(call, gold),
                              call_count_correct=count_correct, tool_correct=same_tools,
                              arguments_correct=match if same_tools else None,
                              status="correct" if match else "wrong_call_count" if not count_correct else "wrong_tool" if not same_tools else "wrong_arguments")
                if len(call) == len(gold) == 1 and same_tools and not match:
                    expected, predicted = gold[0]["arguments"], call[0]["arguments"]
                    detail.update(missing_arguments=sorted(expected.keys() - predicted.keys()),
                                  extra_arguments=sorted(predicted.keys() - expected.keys()),
                                  incorrect_values=sorted(k for k in expected.keys() & predicted.keys() if not same_value(expected[k], predicted[k])))
        detail["false_positive_calls"] = detail["prediction_items"] - detail["matched_calls"]
        detail["false_negative_calls"] = len(gold) - detail["matched_calls"]
        details.append(detail)
    breakdowns = {}
    for axis in ("category", "call_pattern", "candidate_scope", "selection_scope", "call_count_bucket", "quality"):
        groups = {}
        for detail in details:
            if axis in detail["task"]:
                groups.setdefault(detail["task"][axis], []).append(detail)
        if groups:
            breakdowns[axis] = {name: aggregate(rows) for name, rows in sorted(groups.items())}
    summary = {"scorer_version": VERSION, "matching_mode": mode, **aggregate(details),
               "invalid_format_reasons": dict(sorted(format_errors.items())), "breakdowns": breakdowns}
    return summary, details


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), help="Read data/<split>.jsonl inside --gold tar.gz")
    parser.add_argument("--mode", choices=("multiset", "single"), default="multiset")
    parser.add_argument("--metadata", type=Path, help="Offline sidecar for flagged/unflagged breakdowns")
    parser.add_argument("--output", type=Path, help="Also save the JSON summary")
    parser.add_argument("--details", type=Path, help="Save one diagnostic row per gold ID as JSONL")
    args = parser.parse_args(argv)
    try:
        inputs = {args.gold.resolve(), args.predictions.resolve()}
        if args.metadata:
            inputs.add(args.metadata.resolve())
        outputs = [p.resolve() for p in (args.output, args.details) if p is not None]
        if inputs.intersection(outputs) or len(outputs) != len(set(outputs)):
            raise ScoringError("output paths must be distinct and must not overwrite inputs")
        gold_bytes = read_payload(args.gold, args.split)
        prediction_bytes = args.predictions.read_bytes()
        metadata_bytes = args.metadata.read_bytes() if args.metadata else None
        metadata = read_rows(metadata_bytes, "metadata") if metadata_bytes is not None else None
        summary, details = evaluate(read_rows(gold_bytes, "gold"), read_rows(prediction_bytes, "predictions"), args.mode, metadata)
        summary["inputs"] = {"gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
                             "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(), "split": args.split}
        if metadata_bytes is not None:
            summary["inputs"]["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            write_text(args.output, rendered)
        if args.details:
            write_text(args.details, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in details))
        sys.stdout.write(rendered)
        return 0
    except (OSError, ValueError, RecursionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
