# From User Requests to Reliable Tool Calls

We study how a single-turn request and candidate tools map to correct function calls, and whether filtering or similar-tool training improves reliability.

## Documentation

- [Research design](docs/design.md): four papers, two baselines, and two extensions.
- [Data](docs/data.md): source, splits, corpus profile, quality checks, and reproduction.
- [Scoring](scoring.md): metrics, matching rules, prediction format, and usage.
- [Strong baseline](docs/strong-baseline.md): environment, LoRA training, prediction, and reproduction.

Documentation and code comments use English. Source data stays unchanged.

## Status

The complete xLAM release is prepared: **48,000 train / 6,000 dev / 6,000 test**. Preparation and multi-call scoring are implemented and verified. Original M1 assignments are preserved.

The Qwen2.5-1.5B LoRA baseline completed one epoch on all 48,000 training examples. The selected adapter is saved under `runs/strong/full/adapter`. Full evaluation is complete: **81.63% dev / 81.77% test complete-request accuracy**, with 6,000 predictions per split. [Verified results](analysis/strong_results.json) include category breakdowns, data hashes, and model provenance; the [guide](docs/strong-baseline.md) explains the errors and evaluation limits.

The lexical simple baseline is implemented in [simple-baseline.py](simple-baseline.py). Its standalone guide and full-split evaluation results still need to be added to the repository.

Remaining MS2 work: add the simple-baseline guide and verified scores, compare the two systems, and write the 1-2 page report. MS2 is due **November 18, 2026**; see the [course requirements](https://www.cis.upenn.edu/~myatskar/teaching/cis5300_fa26//term-project.html#milestone-2).

## Run

Data preparation and scoring use Python's standard library. Strong-baseline dependencies and commands are in its [guide](docs/strong-baseline.md). Prepare the data using [these instructions](docs/data.md), then run from the repository root:

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
| `simple-baseline.py` | Lexical tool matching and rule-based argument extraction |
| `strong_baseline.py`, `configs/` | SFT, generation, and fixed strong-baseline settings |
| `scripts/` | Preparation, auditing, and shared data utilities |
| `data/xlam_full_v1/` | Local splits, metadata, and manifest; ignored by Git |
| `analysis/` | Data/token statistics, split provenance, and verified baseline results |
| `examples/scoring/`, `tests/` | Scoring demonstration and regression checks |

The raw download and submitted M1 materials remain in the parent project directory. Gold-derived metadata is for offline analysis only. Keep training environments, weights, and run artifacts outside Git.
