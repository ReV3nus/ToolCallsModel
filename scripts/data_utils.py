"""Shared normalization, schema inspection, and descriptive statistics."""


import json
import math
import re
import statistics
import unicodedata


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_query(query):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query).casefold()).strip()


def query_template(query):
    """Group recognizable quote/number substitutions, not all paraphrases."""
    value = normalized_query(query)
    value = re.sub(r'"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)', " <quoted> ", value)
    value = re.sub(r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?!\w)", "<number>", value)
    value = re.sub(r"\s+", " ", value).strip()
    # Very short generic templates are too aggressive; normalized duplicates
    # still share their group through normalized_query().
    visible_words = re.findall(r"[a-z]+", re.sub(r"<[^>]+>", "", value))
    return value if len(visible_words) >= 5 else normalized_query(query)


def function_definition(tool):
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def parameter_definitions(tool):
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return None
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        return parameters["properties"]
    return parameters


def explicit_type_matches(value, description):
    """Return False only for an understood, unambiguous primitive mismatch."""
    if not isinstance(description, str):
        return True
    name = description.strip().casefold()
    name = re.sub(r",\s*optional\s*$", "", name)
    # Unknown unions, item schemas and unusual API annotations are not guessed.
    checks = {
        "str": lambda x: isinstance(x, str),
        "string": lambda x: isinstance(x, str),
        "int": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "float": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "bool": lambda x: isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "list": lambda x: isinstance(x, list),
        "array": lambda x: isinstance(x, list),
        "dict": lambda x: isinstance(x, dict),
        "object": lambda x: isinstance(x, dict),
    }
    return checks[name](value) if name in checks else True


def summarize(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    result = {"n": len(values), "min": values[0], "mean": round(statistics.mean(values), 4), "max": values[-1]}
    for name, p in (("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99)):
        position = (len(values) - 1) * p
        left, right = math.floor(position), math.ceil(position)
        result[name] = round(values[left] + (values[right] - values[left]) * (position - left), 4)
    return result
