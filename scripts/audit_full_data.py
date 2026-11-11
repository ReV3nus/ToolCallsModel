"""Profile the complete xLAM release without filtering or rewriting its records.

Flags identify structural inconsistencies for review, not proven semantic errors.
This script produces a profile, not train/dev/test data or model results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from data_utils import canonical, explicit_type_matches, function_definition, normalized_query, parameter_definitions, query_template, summarize


def parse(value):
    return json.loads(value) if isinstance(value, str) else value


def audit(records):
    candidates, calls, kinds, flags = Counter(), Counter(), Counter(), Counter()
    lengths = {"query_characters": [], "input_characters": [], "target_characters": []}
    schemas, names, fingerprints, query_groups = set(), set(), Counter(), Counter()
    flagged = []
    ids = set()
    for record in records:
        source_id = record["id"]
        if type(source_id) is not int or source_id in ids:
            raise ValueError(f"Invalid or repeated source ID: {source_id!r}")
        ids.add(source_id)
        query, tools, answers = record["query"], parse(record["tools"]), parse(record["answers"])
        if not isinstance(query, str) or not isinstance(tools, list) or not isinstance(answers, list):
            raise ValueError(f"Source ID {source_id}: unrecognized top-level structure; no records were filtered")
        issues = set()
        definitions = [function_definition(t) for t in tools]
        by_name = defaultdict(list)
        for original, definition in zip(tools, definitions):
            by_name[definition["name"]].append(definition)
            schemas.add(canonical(original))
            names.add(definition["name"])
        if any(len(group) > 1 for group in by_name.values()):
            issues.add("repeated_candidate_name")
            if any(len({canonical(t) for t in group}) > 1 for group in by_name.values()):
                issues.add("same_candidate_name_with_different_definitions")
        if not query.strip():
            issues.add("empty_request")
        if not tools:
            issues.add("empty_candidate_list")
        if not answers:
            issues.add("empty_reference_list")
        for call in answers:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                issues.add("malformed_reference_call")
                continue
            matching = by_name.get(call["name"], [])
            if not matching:
                issues.add("reference_tool_absent")
                continue
            parameter_sets = [parameter_definitions(t) for t in matching]
            if not all(p is not None for p in parameter_sets):
                issues.add("unrecognized_parameter_schema")
            known = [p for p in parameter_sets if p is not None]
            compatible = [p for p in known if all(k in p for k in call["arguments"])]
            if known and not compatible:
                issues.add("reference_argument_absent_from_matching_schemas")
            if compatible and not any(all(not isinstance(p[k], dict) or explicit_type_matches(v, p[k].get("type"))
                                          for k, v in call["arguments"].items()) for p in compatible):
                issues.add("recognized_primitive_type_conflict")
        if issues:
            flagged.append({"id": source_id, "flags": sorted(issues)})
            flags.update(issues)
        candidates[len(tools)] += 1
        calls[len(answers)] += 1
        answer_names = {a.get("name") for a in answers if isinstance(a, dict)}
        kind = "single_call" if len(answers) == 1 else "multiple_calls_same_named_tool" if len(answer_names) == 1 else "multiple_calls_distinct_named_tools"
        kinds[kind] += 1
        lengths["query_characters"].append(len(query))
        lengths["input_characters"].append(len(canonical({"query": query, "tools": tools})))
        lengths["target_characters"].append(len(canonical(answers)))
        fingerprint = hashlib.sha256(canonical({"query": normalized_query(query), "tools": sorted(tools, key=canonical), "answers": answers}).encode("utf-8")).hexdigest()
        fingerprints[fingerprint] += 1
        query_groups[query_template(query)] += 1
    return {"records": len(records), "records_removed": 0, "records_rewritten": 0,
            "candidate_count_histogram": dict(sorted(candidates.items())),
            "reference_call_count_histogram": dict(sorted(calls.items())), "request_categories_by_tool_name": dict(kinds),
            "length_unit": "Unicode characters, not tokenizer tokens", "lengths": {k: summarize(v) for k, v in lengths.items()},
            "unique_candidate_names": len(names), "unique_candidate_schemas": len(schemas),
            "duplicate_complete_task_extra_records": sum(n - 1 for n in fingerprints.values()),
            "recognized_query_template_groups": len(query_groups), "max_query_template_group_size": max(query_groups.values()),
            "flag_counts": dict(sorted(flags.items())), "flagged_record_count": len(flagged), "flagged_records": flagged}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    records = json.loads(source_bytes.decode("utf-8-sig"))
    if isinstance(records, dict):
        records = records["data"]
    result = audit(records)
    result["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "full_data_profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": result["records"], "flagged": result["flagged_record_count"], "removed": 0, "rewritten": 0}))


if __name__ == "__main__":
    main()
