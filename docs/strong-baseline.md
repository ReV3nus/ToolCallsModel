# Strong baseline: Qwen2.5 with LoRA SFT

The strong baseline adapts [APIGen's supervised tool-calling approach](https://arxiv.org/abs/2406.18518v1) to a generic Qwen instruction model and our grouped xLAM splits. The task is to generate the complete reference call array from the request and all supplied tool definitions. Baseline training uses the original candidates; filtering and similar-tool augmentation belong to later extensions.

## Model and training settings

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Training data | All 48,000 records in `xlam_full_v1/train.jsonl` |
| Epochs | 1 |
| LoRA | Rank 8, alpha 16, dropout 0.05; attention and MLP projections |
| Trainable parameters | 9,232,384, approximately 0.59% of the adapted model |
| Learning rate | 0.0002, 3% warmup, weight decay 0.01 |
| Batch | 1 example per GPU step, accumulated over 16 steps |
| Precision | BF16 on supported GPUs; otherwise FP16 |
| Context limit | 4,096 tokens; examples exceeding it cause an error |
| Checkpoint selection | Lowest assistant-token loss on the first 128 fixed dev records, checked every 250 optimizer steps |
| Seed | 5300 |

Settings are in [configs/strong-baseline.json](../configs/strong-baseline.json). The implementation uses Transformers, PEFT, gradient checkpointing, and PyTorch SDPA. It requires one CUDA GPU. The setup has been exercised on an RTX 4080 Laptop GPU with 12GB VRAM.

## Inputs, targets, and length checks

The prompt consists of a fixed system instruction and a user message containing only `query` and the complete `tools` array. IDs, answers, reference call counts, and offline metadata are excluded. Tool schemas are serialized as text without converting them into a different schema standard.

The target is the original JSON call array followed by EOS. Only target tokens receive loss; prompt and padding labels are `-100`. Source decimal values are preserved. Training and prediction share the same chat-template prompt. Tests cover answer/metadata isolation, padding masks, complete targets, and truncation rejection.

The [actual tokenizer profile](../analysis/strong_token_profile.json) covers train and dev:

| Split | Mean total tokens | P95 | Maximum | Over 4,096 |
|---|---:|---:|---:|---:|
| Train | 389.86 | 741 | 2,272 | 0 |
| Dev | 389.12 | 737 | 2,272 | 0 |

The longest training target is 1,771 tokens including EOS. Generation uses a fixed maximum of 2,048 new tokens; no reference-specific output budget is used. The test targets were not used to choose these limits.

## Environment

Create an isolated environment from the repository root. The following Windows commands match the tested CUDA 12.8 setup:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements-training.txt
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"
```

For Linux or Colab, use its Python executable and a compatible CUDA-enabled PyTorch build, then install the same project dependencies. The base model is public and is downloaded at the pinned revision into `models/hf` when needed. This workstation also has a local mirror in `models/qwen2.5-1.5b-instruct`; it is used only when its `.model_revision` marker matches the configuration. Model files are ignored by Git.

Each training run saves its configuration, data hashes, dependency versions, complete package list, model revision, progress logs, and selected checkpoint. These files belong under `runs/`, which is ignored by Git.

## Commands

Profile the training template before changing model or context settings:

```powershell
.venv/Scripts/python strong_baseline.py profile --output analysis/strong_token_profile.json
```

A short end-to-end check:

```powershell
.venv/Scripts/python strong_baseline.py train --output runs/strong/smoke --train-limit 32 --eval-limit 8 --max-steps 2
.venv/Scripts/python strong_baseline.py predict --adapter runs/strong/smoke/adapter --split dev --limit 16 --output runs/strong/smoke/dev.predictions.jsonl
```

The recorded check trained on 32 examples, saved/reloaded the adapter, and generated/scored 16 dev requests. It verifies the pipeline, not final model quality. Its short training run peaked at about 4.9GiB of allocated GPU memory; full-data peaks can differ.

Formal training and evaluation:

```powershell
.venv/Scripts/python strong_baseline.py train --output runs/strong/full
.venv/Scripts/python strong_baseline.py predict --adapter runs/strong/full/adapter --split dev --output runs/strong/full/dev.predictions.jsonl
.venv/Scripts/python strong_baseline.py predict --adapter runs/strong/full/adapter --split test --output runs/strong/full/test.predictions.jsonl
```

Use a fresh training directory. To resume a run, append `--resume` to restore its latest checkpoint, or supply `--resume path/to/checkpoint`. Training data, limits, and configuration must match the recorded run. To resume prediction, repeat its command with `--resume`; completed IDs are checked and skipped. Both commands flush progress so interrupted work is recoverable.

## Outputs and evaluation

Training produces an `adapter/` directory with LoRA weights and tokenizer, resumable `checkpoint-*` directories, `run.json`, `progress.jsonl`, and `train_metrics.json`. The saved adapter is taken from the checkpoint selected by the declared development-loss rule.

Generation is greedy, with fixed settings and all original candidate schemas. Input-length sorting reduces padding; IDs retain alignment with the fixed split. Raw generated text is saved without JSON repair. Each prediction includes input/output token counts and an EOS/length finish reason.

For each prediction file the command writes:

- `*.metrics.json`: existing scorer results, category/quality breakdowns, and explicit full-split or limited-sample scope.
- `*.details.jsonl`: per-request error diagnostics.
- `*.run.json`: configuration, adapter/data hashes, progress sessions, and generation timing.

Limited diagnostic runs also save their exact gold subset. Full runs evaluate all 6,000 requests per split; missing and invalid outputs stay in the denominator. The same [scorer](../scoring.md) can be run independently:

```powershell
.venv/Scripts/python score.py --gold data/xlam_full_v1/dev.jsonl --predictions runs/strong/full/dev.predictions.jsonl --metadata data/xlam_full_v1/metadata.jsonl
```

## Results and error analysis

Formal training completed one epoch on all 48,000 examples in 14,832 seconds (about 4 hours 7 minutes). The selected checkpoint is step 3,000, with development-subset loss 0.02644885 and peak allocated training memory 6.98GiB. The final adapter weights match that checkpoint.

Both full splits have 6,000 unique predictions and no missing IDs. Development generation was paused after 4,720 requests and resumed with the same settings. Active generation time was 7,390 seconds for dev and 5,057 seconds for test, excluding model loading and the overnight pause. Scores were recomputed from saved raw predictions and matched the recorded metrics and per-request diagnostics. Source-file and prediction hashes were also checked. The [verified results snapshot](../analysis/strong_results.json) preserves these results and provenance.

| Metric (%) | Dev | Test |
|---|---:|---:|
| Complete-request accuracy | 81.63 | 81.77 |
| Complete-call micro F1 | 86.24 | 86.97 |
| Request-level tool selection | 98.35 | 98.55 |
| Call-count accuracy | 98.82 | 98.92 |
| Valid JSON call-array rate | 99.83 | 99.82 |
| Argument accuracy given correct tools | 83.00 | 82.97 |

The main metric requires the entire call multiset and all argument values to match the reference. There were 4,898 correct dev requests and 4,906 correct test requests. Argument mismatches account for 1,003 of 1,102 dev failures and 1,007 of 1,094 test failures. The other test failures were 54 call-count mismatches, 22 tool mismatches, and 11 invalid JSON outputs; these are mutually exclusive scorer categories. Three dev outputs reached the 2,048-token limit, while no test output did. All raw outputs were scored.

| Candidate tools / required calls | Dev accuracy (%) | Test accuracy (%) | Test requests |
|---|---:|---:|---:|
| One tool, one call | 83.48 | 80.95 | 761 |
| Multiple tools, one call | 86.91 | 85.14 | 2,039 |
| One tool, multiple calls | 81.87 | 85.25 | 732 |
| Multiple tools, multiple calls | 76.46 | 78.20 | 2,468 |

Tool selection is already strong on the original candidates. Matched distractor experiments are still needed to isolate the effect of tool similarity. More calls also create more opportunities for an argument mismatch: test accuracy is 84.00% for one call and 68.15% for four or more calls.

Manual inspection of development errors shows why exact-reference mismatches need interpretation:

- **ID 36873:** the model omitted the requested playlist ID (`is_id`) and kept only the cursor. This misses information explicitly provided by the user.
- **ID 24531:** the model added `lang: "en"`, which the reference omits. The supplied schema defines `en` as the default, so an exact-match failure can still follow the documented language setting.
- **ID 26649:** the request asks for cosine similarity but supplies no vector values, and the tool definition supplies none either. The reference and prediction use different vectors; the reference values cannot be recovered from the supplied input.

These examples illustrate limits of strict reference matching and of the synthetic data; they do not measure how common those issues are. Flagged/unflagged test accuracy is 33.57% / 82.94%, and structural flags do not identify every semantic problem. The reported primary scores use the unchanged data and scorer. This run adapts APIGen's SFT approach to a smaller model with LoRA and our grouped xLAM splits; its results are not directly comparable to APIGen's published BFCL scores.
