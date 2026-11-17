"""Simple baseline: lexical tool matching plus rule-based argument extraction.

For each request the baseline
  1. splits the request into clauses (one clause ~ one intended call),
  2. scores every candidate tool against each clause by IDF-weighted word
     overlap with the tool's name, description and parameter names,
  3. keeps the best tool for each clause if it clears a score threshold,
  4. fills arguments with fixed rules: values that can be read off the request
     text (quoted strings, numbers, lists, URLs, e-mails, IPs, dates,
     capitalized names) are assigned to parameters by type and by nearby
     parameter-name cues, and
  5. "fans out" one call per value when a single-parameter tool is given
     several values (e.g. "search for 'a' and 'b'" -> two calls).

No neural model is trained. The only thing learned from training data is the
inverse document frequency (IDF) of words over the training tool inventory,
which down-weights generic words such as "fetch" or "retrieve". Thresholds
and on/off switches are chosen on the development split (--tune). The model
never reads `answers` or the metadata sidecar of the split it predicts.

Usage (from the repository root):
  python simple-baseline.py --train data/xlam_full_v1/train.jsonl \
      --input data/xlam_full_v1/dev.jsonl --output runs/simple/dev.predictions.jsonl
  python simple-baseline.py --train data/xlam_full_v1/train.jsonl \
      --input data/xlam_full_v1/dev.jsonl --tune
Python standard library only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import re

# Settings selected on the development split with --tune (see simple-baseline.md).
DEFAULTS = {"min_score": 0.0, "segment": True, "fanout": True, "fill_optional": True}

# --------------------------------------------------------------------------
# 1. Tokenization
# --------------------------------------------------------------------------

STOPWORDS = set("""a an the of for to in on at by with from and or as is are was were be been
it its this that these those i me my we our you your he she they them their what which who whom
how when where why can could would should will shall may might do does did please also some any
all each about into over than then there here if not no yes so such via using use given provide
get find fetch retrieve show give tell list information info details data me us want need like
let s t""".split())


def stem(word):
    """Very light suffix stripping so 'prices' matches 'price', 'searching' matches 'search'."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text):
    """Lower-cased, stemmed content words; splits snake_case and camelCase."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def tool_tokens(tool):
    """Bag of words describing a tool: name, description and parameter names."""
    params = tool.get("parameters") or {}
    text = " ".join([tool["name"], tool.get("description", ""), *params.keys()])
    return set(tokens(text))


# --------------------------------------------------------------------------
# 2. Tool scoring
# --------------------------------------------------------------------------


class IDF:
    """Inverse document frequency over the distinct tools seen in training."""

    def __init__(self, train_rows):
        seen, df = set(), Counter()
        for row in train_rows:
            for tool in row["tools"]:
                key = json.dumps(tool, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    df.update(tool_tokens(tool))
        self.n, self.df = len(seen), df

    def __call__(self, word):
        return math.log((self.n + 1) / (self.df[word] + 1)) + 1.0


def score_tool(query_words, tool, idf):
    """Sum of IDF weights of distinct request words that also describe the tool."""
    return sum(idf(w) for w in set(query_words) & tool_tokens(tool))


# --------------------------------------------------------------------------
# 3. Clause splitting
# --------------------------------------------------------------------------

VERBS = ("get|find|fetch|retrieve|check|calculate|compute|search|look|list|show|give|provide|"
         "determine|convert|generate|create|make|tell|what|how|is|are|can|could|obtain|"
         "identify|predict|estimate|sort|count|merge|reverse|validate|verify|download|send|"
         "translate|analyze|recommend|fetch|query|lookup|return|perform|display|i")
CLAUSE_BREAK = re.compile(
    r"(?<=[.?!;])\s+(?=[A-Z])"                         # sentence boundary
    r"|,?\s+\b(?:and\s+)?(?:also|additionally|then|plus|as well as)\b[,:]?\s+"
    r"|,?\s+\band\s+(?=(?:" + VERBS + r")\b)",            # ", and check ...", " and what ..."
    re.IGNORECASE,
)


def clauses(query):
    parts = [p.strip(" ,.") for p in CLAUSE_BREAK.split(query)]
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# 4. Value extraction from the request text
# --------------------------------------------------------------------------

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
PATTERNS = [  # (kind, regex); earlier kinds win when spans overlap
    ("list", r"\[[^\[\]]*\]"),
    ("quoted", r"\"([^\"]+)\"|(?<![\w])'([^']+?)'(?![\w])"),
    ("url", r"https?://[^\s,'\"]+[^\s,'\".?!)]"),
    ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+\w"),
    ("ip", r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    ("date", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("numseq", NUMBER + r"(?:\s*,\s*(?:and\s+)?" + NUMBER + r")+(?:,?\s+and\s+" + NUMBER + r")?"),
    ("number", r"(?<![\w.])" + NUMBER + r"(?![\w])"),
    ("capword", r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*"),
]
CAP_STOP = {"I", "What", "Could", "Can", "Please", "Also", "And", "Find", "Get", "Fetch", "Retrieve",
            "Check", "Calculate", "Show", "List", "Is", "Are", "How", "The", "ID", "API"}


def parse_number(text):
    text = text.replace(",", "")
    value = float(text)
    return int(value) if re.fullmatch(r"[-+]?\d+", text) else value


def extract_values(text):
    """Return candidate argument values as dicts with kind, value, raw text and span."""
    found, taken = [], [False] * len(text)
    # Bracketed text inside quotes (e.g. "'{[()]}'") is a string, not a list.
    quote_spans = [m.span() for m in re.finditer(PATTERNS[1][1], text)]
    for kind, pattern in PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.span()
            if kind == "list" and any(a < start < b for a, b in quote_spans):
                continue
            # Overlap rule: a number sequence may reuse single numbers, but other
            # spans already claimed by a higher-priority kind are skipped.
            if any(taken[start:end]) and kind != "number":
                continue
            raw = next((g for g in m.groups() if g is not None), None) if m.groups() else m.group(0)
            raw = raw if raw is not None else m.group(0)
            try:
                if kind == "list":
                    value = ast.literal_eval(raw)
                    if not isinstance(value, list):
                        continue
                elif kind == "numseq":
                    value = [parse_number(n) for n in re.findall(NUMBER, raw.replace(", ", " , "))]
                elif kind == "number":
                    if any(taken[start:end]):
                        continue
                    value = parse_number(raw)
                elif kind == "capword":
                    if raw in CAP_STOP:
                        continue
                    value = raw
                else:
                    value = raw
            except (ValueError, SyntaxError):
                continue
            found.append({"kind": kind, "value": value, "raw": raw, "start": start, "end": end})
            if kind != "numseq":  # numbers inside a sequence stay available individually
                for i in range(start, end):
                    taken[i] = True
    return sorted(found, key=lambda v: v["start"])


# --------------------------------------------------------------------------
# 5. Argument filling
# --------------------------------------------------------------------------


def param_kind(spec):
    t = str(spec.get("type", "")).lower()
    if "list" in t or "array" in t or t.startswith("tuple"):
        return "list"
    if t.startswith("bool"):
        return "bool"
    if t.startswith("int"):
        return "int"
    if t.startswith("float") or t.startswith("number"):
        return "float"
    if t.startswith("str"):
        return "str"
    return "other"


def is_optional(spec):
    return "optional" in str(spec.get("type", "")).lower()


# Preference order of value kinds for each parameter kind.
COMPATIBLE = {
    "int": ["number"],
    "float": ["number"],
    "list": ["list", "numseq"],
    "str": ["quoted", "url", "email", "ip", "date", "capword", "number"],
    "other": ["quoted", "number", "list"],
    "bool": [],
}


def convert(value, kind):
    """Cast an extracted value to the parameter's type."""
    if kind == "int" and isinstance(value["value"], float):
        return value["value"] if not value["value"].is_integer() else int(value["value"])
    if kind == "str" and value["kind"] == "number":
        return value["raw"]  # keep the request's spelling, e.g. "10"
    return value["value"]


def cue_match(param, value, text):
    """True if a word from the parameter name appears just before the value."""
    before = tokens(text[max(0, value["start"] - 40): value["start"]])[-4:]
    name_words = set(tokens(param)) - {"is"}
    return bool(name_words & set(before))


def fill(tool, values, text, fill_optional):
    """Assign extracted values to a tool's parameters; returns (arguments, unused values)."""
    params = tool.get("parameters") or {}
    kinds = {p: param_kind(s) for p, s in params.items()}
    args, used = {}, set()

    # Pass 1: parameter-name cues ("radius 5", "locale 'es_ES'"), any parameter.
    for p in params:
        if not fill_optional and is_optional(params[p]):
            continue
        for i, v in enumerate(values):
            if i not in used and v["kind"] in COMPATIBLE[kinds[p]] and cue_match(p, v, text):
                args[p] = convert(v, kinds[p])
                used.add(i)
                break

    # Pass 2: required parameters, in schema order, take the next compatible value
    # in preferred-kind order (e.g. a quoted string before a capitalized word).
    for p in params:
        if p in args or is_optional(params[p]):
            continue
        for kind in COMPATIBLE[kinds[p]]:
            i = next((i for i, v in enumerate(values) if i not in used and v["kind"] == kind), None)
            if i is not None:
                args[p] = convert(values[i], kinds[p])
                used.add(i)
                break
    return args, [v for i, v in enumerate(values) if i not in used]


def calls_for(tool, text, fanout, fill_optional):
    """Calls for one tool on one clause, fanning out over repeated values if enabled."""
    values = extract_values(text)
    args, rest = fill(tool, values, text, fill_optional)
    calls = [{"name": tool["name"], "arguments": args}]
    params = tool.get("parameters") or {}
    required = [p for p in params if not is_optional(params[p])]
    if fanout and len(required) == 1 and required[0] in args:
        p = required[0]
        kind = param_kind(params[p])
        extra = [v for v in rest if v["kind"] in COMPATIBLE[kind] and v["kind"] != "capword"]
        for v in extra:  # one more call per leftover value of the same kind
            calls.append({"name": tool["name"], "arguments": {**args, p: convert(v, kind)}})
    return calls


# --------------------------------------------------------------------------
# 6. Prediction
# --------------------------------------------------------------------------


def predict(row, idf, min_score, segment, fanout, fill_optional):
    tools = row["tools"]
    parts = clauses(row["query"]) if segment else [row["query"]]
    calls = []
    for part in parts:
        words = tokens(part)
        scored = [(score_tool(words, t, idf), i) for i, t in enumerate(tools)]
        best, index = max(scored)
        if best < min_score and calls:
            continue  # weak clause: probably a continuation of the previous one
        new = calls_for(tools[index], part, fanout, fill_optional)
        if calls and not new[0]["arguments"] and tools[index].get("parameters"):
            continue  # clause gave no values for a tool that takes some: not a new call
        calls.extend(new)
    if not calls:  # always produce at least one call
        best = max(range(len(tools)), key=lambda i: score_tool(tokens(row["query"]), tools[i], idf))
        calls = calls_for(tools[best], row["query"], fanout, fill_optional)
    return calls


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(rows, idf, settings):
    return [{"id": r["id"], "prediction": predict(r, idf, **settings)} for r in rows]


def tune(rows, idf):
    """Grid search on the development split using the project's scorer."""
    from score import evaluate, read_rows  # local import: same rules as score.py

    gold = read_rows("\n".join(json.dumps(r) for r in rows).encode(), "gold")
    grid = {"min_score": [0.0, 1.0, 2.0, 4.0, 6.0], "segment": [False, True],
            "fanout": [False, True], "fill_optional": [False, True]}
    results = []
    for combo in itertools.product(*grid.values()):
        settings = dict(zip(grid, combo))
        preds = run(rows, idf, settings)
        pred_rows = read_rows("\n".join(json.dumps(p) for p in preds).encode(), "predictions")
        m = evaluate(gold, pred_rows)[0]["metrics"]
        results.append((m["complete_request_accuracy"], m["tool_selection_accuracy"], settings))
        print(f"{json.dumps(settings)}  complete={m['complete_request_accuracy']:.4f}  tools={m['tool_selection_accuracy']:.4f}")
    best = max(results, key=lambda r: (r[0], r[1]))
    print("best:", json.dumps(best[2]), f"complete={best[0]:.4f} tools={best[1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="training JSONL (IDF statistics only)")
    parser.add_argument("--input", type=Path, required=True, help="JSONL to predict (answers are ignored)")
    parser.add_argument("--output", type=Path, help="prediction JSONL for score.py")
    parser.add_argument("--tune", action="store_true", help="grid-search settings on --input (dev only)")
    parser.add_argument("--min-score", type=float, default=DEFAULTS["min_score"])
    parser.add_argument("--no-segment", action="store_true")
    parser.add_argument("--no-fanout", action="store_true")
    parser.add_argument("--no-optional", action="store_true")
    args = parser.parse_args()

    idf = IDF(read_jsonl(args.train))
    # Drop gold answers before prediction so they can never leak into the rules.
    rows = read_jsonl(args.input)
    if args.tune:
        return tune(rows, idf)
    inputs = [{"id": r["id"], "query": r["query"], "tools": r["tools"]} for r in rows]
    settings = {"min_score": args.min_score, "segment": not args.no_segment,
                "fanout": not args.no_fanout, "fill_optional": not args.no_optional}
    preds = run(inputs, idf, settings)
    if args.output is None:
        parser.error("--output is required unless --tune is given")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for p in preds:
            stream.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
