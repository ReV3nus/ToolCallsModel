# Scoring tool-call arrays

`score.py` version 2 evaluates the project's single-turn xLAM task. Its primary metric is **complete-request accuracy**: the prediction must contain every reference call, with the correct tool names, arguments, and multiplicities. Order between calls is ignored. API execution and dependency correctness are outside this metric.

Structural evaluation is motivated by [BFCL (Patil et al., 2025)](https://proceedings.mlr.press/v267/patil25a.html). This implementation uses the project's [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) references and explicit JSON rules; it is not BFCL's official accepted-answer AST checker. The default `--mode multiset` handles single and multiple calls. `--mode single` retains the earlier one-call restriction for M1 comparisons.

## Inputs

Gold is UTF-8 JSONL with integer `id` and a nonempty `answers` array. Dataset layout and preparation commands are in [docs/data.md](docs/data.md). Every prediction is a JSONL row with the same ID:

```json
{"id": 5, "prediction": [{"name": "product_id", "arguments": {"is_id": 456789, "locale": "es_ES"}}]}
```

For generative models, `prediction` may be the unedited generated JSON text instead of a parsed array. This is the preferred way to retain malformed generations for evaluation. Outer metadata is ignored; `prediction` is authoritative. IDs determine alignment independently of row order.

A failed generation can be saved as an empty string. Missing rows remain in the gold denominator. Do not remove failed examples from the references. Model inputs must exclude `answers` and the offline metadata sidecar's reference-derived fields.

## Matching rules

A valid prediction is an array of call objects, each with exactly `name` and `arguments`. Names must be nonblank strings and arguments must be objects. An empty prediction array is valid format but misses every reference call. Gold arrays cannot be empty in this dataset. In legacy single mode, both gold and predictions must contain exactly one call.

| Content | Rule |
|---|---|
| Order of calls | Ignored for the primary metric; also report ordered exact match |
| Repeated calls | Preserve multiplicity; one call cannot match two references |
| Object keys and JSON whitespace | Compare after parsing; key order is irrelevant |
| Tool/argument names and strings | Exact, case-sensitive; string whitespace matters |
| Missing or extra arguments | Incorrect |
| Argument arrays | Preserve order and length, including nested arrays |
| Numbers | Exact numeric equality, so `1` matches `1.0`; no tolerance or rounding |
| Numeric strings and booleans | `"1"` and `true` do not match `1` |
| Null | Matches only null, not an absent key or `"null"` |
| Defaults and alternative valid tools | Never inserted, removed, or inferred |
| Prose, code fences, Python syntax | Invalid; no repair or JSON-substring extraction |
| Duplicate keys, NaN, Infinity | Rejected |

Decimal literals are parsed exactly. A malformed call makes the entire predicted array invalid; correctly shaped pieces do not receive partial credit. A well-formed call to an unknown tool has valid format but incorrect content. Matching references does not validate source schemas or prove that an API could execute successfully.

## Metrics

For request \(i\), let \(G_i\) be the multiset of reference calls and \(P_i\) the multiset of predicted calls. Each call contains its tool name and complete argument object. Define \(C_i=1\) only when the output is valid and \(P_i=G_i\). The main measure is

\[
\mathrm{CompleteRequestAccuracy}=\frac{1}{N}\sum_{i=1}^{N}C_i.
\]

`complete_call_accuracy` is retained as an alias for compatibility. Ordered exact-match accuracy additionally requires the source call order. This diagnostic does not establish that source order is necessary or that execution dependencies were satisfied.

For partial diagnostics, match exact calls one-to-one within each request, preserving duplicates:

\[
TP=\sum_i |P_i\cap G_i|,\quad P=\frac{TP}{\sum_i|P_i|},\quad R=\frac{TP}{\sum_i|G_i|},\quad F_1=\frac{2PR}{P+R}.
\]

These are micro-averaged `call_precision`, `call_recall`, and `call_f1`. Missing predictions contribute no predicted items and miss all reference calls. **An invalid response contributes one unmatched placeholder item**, no matches, and misses all reference calls. This explicit convention prevents malformed responses from disappearing from the precision denominator. A valid empty array contributes zero predicted items. Precision or F1 with a zero denominator is reported as zero.

Further diagnostics:

- `tool_selection_accuracy`: fraction of requests with exactly the right tool-name multiset, including repeated names, regardless of argument values.
- `tool_name_precision`, `tool_name_recall`, `tool_name_f1`: the same micro matching using names only. These measure final predicted calls, not the filter's separate Recall@k.
- `argument_accuracy_given_correct_tool`: complete-request matches divided by requests with an exact tool-name multiset. Returns `null` if that denominator is zero. For multi-call tasks, the legacy field name refers to the whole tool multiset.
- `call_count_accuracy`: fraction with valid output and exactly the reference number of calls.
- `format_valid_rate`: fraction with a valid array structure.

Request rates include every gold ID, including missing and invalid responses. All rates range from 0 to 1. Totals and denominators are saved explicitly.

The mutually exclusive status counts use this priority: missing prediction, invalid format, wrong call count, wrong tool names, wrong arguments, correct. Partial matching still records matched, extra, and missed calls. Single-call argument failures also list missing/extra keys and keys with incorrect values.

## Breakdown reports

The summary groups results by the four task categories, single/repeated-tool/distinct-tool call patterns, candidate scope, presence of unused candidates, and reference call-count bucket. These labels come from gold data solely for offline evaluation.

With `--metadata data/xlam_full_v1/metadata.jsonl`, it also reports flagged and unflagged records. The sidecar must cover every gold ID; additional IDs from the other partitions are allowed. Prediction IDs outside the selected gold partition are rejected. Each breakdown axis partitions the evaluated requests; different axes overlap and should not be added together.

## Commands

Only Python's standard library is required. Run the checked-in example:

```powershell
python score.py --gold examples/scoring/gold.jsonl --predictions examples/scoring/predictions.jsonl --output runs/scoring-demo/metrics.json --details runs/scoring-demo/details.jsonl
```

The gold file contains unchanged M1 training records 30011, 25351, 52957, and 4163. The predictions are deliberately constructed: one correct, one missing an argument, one wrong tool, and one omitted row. Their order is reversed to check ID alignment. **They are scorer examples, not baseline outputs.** Selected output fields are:

```json
{
  "total_examples": 4,
  "provided_predictions": 3,
  "metrics": {
    "complete_request_accuracy": 0.25,
    "tool_selection_accuracy": 0.5,
    "format_valid_rate": 0.75,
    "argument_accuracy_given_correct_tool": 0.5
  }
}
```

After generating a baseline's complete development predictions:

```powershell
python score.py --gold data/xlam_full_v1/dev.jsonl --predictions runs/simple/dev.predictions.jsonl --metadata data/xlam_full_v1/metadata.jsonl --output runs/simple/dev.metrics.json --details runs/simple/dev.details.jsonl
```

For a historical M1 comparison:

```powershell
python score.py --gold ../ms1/data.tar.gz --split dev --mode single --predictions runs/m1/dev.predictions.jsonl
```

Replace dev inputs with test inputs only after settings are fixed. Do not tune using test diagnostics. Summary JSON is printed and optionally saved by `--output`; `--details` writes a diagnostic row for every gold ID. Summary hashes identify the gold payload, predictions, and optional metadata.

Duplicate IDs, unknown prediction IDs, malformed outer JSONL, invalid references, and noninteger IDs abort evaluation with exit code 2. Malformed model text inside a valid `prediction` string is a per-request failure. A present row without `prediction` is also a failure. Output files cannot overwrite inputs, metadata, or one another.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover call order, duplicate multiplicities, argument binding, exact numeric types, invalid outputs, missing records, partial-credit denominators, category/quality groups, legacy mode, file alignment, and CLI/archive behavior. Integration checks on all 6,000 development records confirm perfect reference-as-prediction matching and zero recall for missing outputs. These are scorer sanity checks, not experimental model results.
