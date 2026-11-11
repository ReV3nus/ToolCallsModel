"""Descriptive task labels for offline analysis, never model input features."""
import json


def describe(record, calls=None):
    if calls is None:
        calls = record["answers"]
        if isinstance(calls, str):
            calls = json.loads(calls)
    names = {c["name"] for c in calls}
    result = {
        "reference_call_count": len(calls),
        "call_pattern": "single_call" if len(calls) == 1 else "repeated_tool" if len(names) == 1 else "distinct_tools",
        "call_count_bucket": str(len(calls)) if len(calls) <= 3 else "4+",
    }
    if "tools" in record:
        tools = record["tools"]
        if isinstance(tools, str):
            tools = json.loads(tools)
        candidates = {t.get("function", t)["name"] for t in tools}
        single_candidate = len(tools) == 1
        result.update(
            candidate_count=len(tools),
            category=("simple" if single_candidate else "multiple") if len(calls) == 1 else ("parallel" if single_candidate else "parallel_multiple"),
            candidate_scope="one" if single_candidate else "multiple",
            selection_scope="has_unselected_candidate" if candidates - names else "all_candidates_referenced",
        )
    return result
