# Data

We use the complete [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) release from [APIGen](https://arxiv.org/abs/2406.18518v1), licensed CC BY 4.0. All 60,000 synthetic requests are retained, with their original candidate definitions, order, numeric values, and full call arrays. Compound requests remain intact.

Active version: `xlam_full_v1`. Source and generated-file hashes are recorded in [the split summary](../analysis/full_split_summary.json).

## Splits

| Split | Records | Flagged records |
|---|---:|---:|
| train | 48,000 | 1,103 |
| dev | 6,000 | 125 |
| test | 6,000 | 143 |

Seed 5300; 53,872 request-template groups are assigned toward 80/10/10. All 19,795 M1 IDs keep their original split, along with other members of their recognized groups. Remaining groups are shuffled and assigned to the largest target deficit.

Verification checks all source content and IDs, metadata consistency, and zero cross-split normalized-request/template overlap. Tools may recur across splits; this tests new requests rather than enforcing unseen tools. The grouping rule cannot identify every paraphrase.

## Format and storage

`data/xlam_full_v1/{train,dev,test}.jsonl` uses UTF-8 JSON Lines. Each row contains `id`, `query`, the complete `tools` array, and the complete `answers` array. Calls contain `name` and `arguments`. Original encoded arrays are embedded directly to preserve numeric literals.

The separate `metadata.jsonl` holds split/group IDs, reference-derived task labels, and quality flags. It is for offline analysis and must not enter model inputs. `manifest.json` records provenance and file hashes. Data files are ignored by Git; the original download and M1 submission remain unchanged.

## Corpus profile

The source has 3,605 tool names and 4,211 distinct full schemas. Name equality alone does not establish schema equality.

| Calls per request | Records |
|---|---:|
| 1 | 28,461 |
| 2 | 25,422 |
| 3 | 4,697 |
| 4 or more | 1,420 |

| Candidate tools | Records |
|---|---:|
| 1 | 15,410 |
| 2 | 12,726 |
| 3 | 12,655 |
| 4 | 13,078 |
| 5 | 2,427 |
| 6 | 1,464 |
| 7 | 1,108 |
| 8 | 1,132 |

| Length in characters | Median | P95 | Maximum |
|---|---:|---:|---:|
| query_characters | 100 | 220 | 1330 |
| input_characters | 1086 | 2706 | 9514 |
| target_characters | 117 | 236 | 4421 |

Lengths use compact JSON before prompts and chat templates. Measure actual tokens with the selected model before choosing context limits; long call lists remain in the data.

## Quality checks

The audit flags 1,371 records for recognized type conflicts or repeated candidate names. A flag may identify a schema issue rather than a wrong answer. All flagged records remain; report flagged/unflagged diagnostics and version any later corrections.

Structural checks do not detect every underspecified request or equivalent use of optional defaults. Manual development-set examples are discussed in the [strong-baseline analysis](strong-baseline.md).

The conservative task fingerprint finds 380 copies beyond the first occurrence. They remain in the data but share their request group. The largest recognized group contains 47 records. Exact distributions and per-record flags are in [full_data_profile.json](../analysis/full_data_profile.json).

Evaluation reports single/multiple candidates and calls, repeated/distinct tools, and unused candidates separately. See [scoring.md](../scoring.md) for definitions and [strong-baseline.md](strong-baseline.md) for model results. The earlier 98.1% lexical-selection diagnostic applies only to the historical M1 subset.

## Reproduction

Run from the repository root. Create the files once:

```powershell
python scripts/prepare_full_data.py --input ../xlam_function_calling_60k.json --m1-archive ../ms1/data.tar.gz --output data/xlam_full_v1 --summary analysis/full_split_summary.json --documentation docs/data.md
```

For existing files, verify without rewriting:

```powershell
python scripts/prepare_full_data.py --input ../xlam_function_calling_60k.json --m1-archive ../ms1/data.tar.gz --output data/xlam_full_v1 --verify-only
```

To refresh the detailed source audit:

```powershell
python scripts/audit_full_data.py --input ../xlam_function_calling_60k.json --output analysis
```
