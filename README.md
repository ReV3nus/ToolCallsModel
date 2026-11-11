# From User Requests to Reliable Tool Calls

We study how a single-turn request and candidate tools map to correct function calls, and whether filtering or similar-tool training improves reliability.

## Documentation

- [Research design](docs/design.md): four papers, two baselines, and two extensions.
- [Data](docs/data.md): source, splits, corpus profile, quality checks, and reproduction.
- [Scoring](scoring.md): metrics, matching rules, prediction format, and usage.

Documentation and code comments use English. Source data stays unchanged.

## Status

The complete xLAM release is prepared: **48,000 train / 6,000 dev / 6,000 test**. Preparation and multi-call scoring are implemented and verified. Original M1 assignments are preserved.

Remaining MS2 work: implement both baselines, select a model and compute environment, measure tokenizer lengths, run experiments, and write baseline instructions and the 1-2 page report. No model results are available yet. MS2 is due **November 18, 2026**; see the [course requirements](https://www.cis.upenn.edu/~myatskar/teaching/cis5300_fa26//term-project.html#milestone-2).

## Run

Current code uses only Python's standard library. Prepare the data using [these instructions](docs/data.md), then run from the repository root:

```powershell
python scripts/prepare_full_data.py --input ../xlam_function_calling_60k.json --m1-archive ../ms1/data.tar.gz --output data/xlam_full_v1 --verify-only
python score.py --gold examples/scoring/gold.jsonl --predictions examples/scoring/predictions.jsonl
python -m unittest discover -s tests -v
```

The scoring example deliberately returns 0.25; it is a demonstration, not a model score.

## Layout

| Location | Contents |
|---|---|
| `docs/` | Design and consolidated data documentation |
| `score.py`, `task_types.py` | Scoring and offline task classification |
| `scripts/` | Preparation, auditing, and shared data utilities |
| `data/xlam_full_v1/` | Local splits, metadata, and manifest; ignored by Git |
| `analysis/` | Detailed JSON statistics and split provenance |
| `examples/scoring/`, `tests/` | Scoring demonstration and regression checks |

The raw download and submitted M1 materials remain in the parent project directory. Gold-derived metadata is for offline analysis only. Future baseline instructions will use the course's required filenames; they have not been created yet.
