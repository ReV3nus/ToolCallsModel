# Research design

## Question and scope

Can a small instruction model produce all required tool calls and arguments when its candidate set contains similar distractors? We will measure whether inference-time filtering and similar-tool training improve reliability and input cost.

The task uses a single request and supplied tool definitions. The output is the complete call array, including repeated calls to the same tool. We retain the full xLAM release. This is static call generation; execution feedback and multi-turn agent policies are outside the experiment.

## Role of the four papers

| Paper | What we use | Limit on interpretation |
|---|---|---|
| [APIGen](https://arxiv.org/abs/2406.18518v1) | Existing verified request-call pairs and supervised fine-tuning | We adapt the training approach; we do not reproduce the generation pipeline or its historical BFCL score. |
| [BFCL](https://proceedings.mlr.press/v267/patil25a.html) | Structural evaluation and distinct calling categories | Our JSON reference matcher differs from the official accepted-answer AST checker. |
| [Robustness](https://aclanthology.org/2025.trustnlp-main.20/) | Hold requests fixed while adding related candidates | Matched random distractors help separate candidate-count effects from similarity. |
| [ToolRet](https://aclanthology.org/2025.findings-acl.1258/) | Candidate filtering, retrieval-miss analysis, and negative selection | Small-set filtering adapts its ideas; it does not reproduce large-corpus retrieval. |

## Two baselines

**Simple:** lexical tool scoring with fixed rules for call count and argument extraction. It provides a cheap reference without training a neural model from scratch. Rules and thresholds are selected on development data.

**Strong:** Qwen2.5-1.5B-Instruct with response-only LoRA SFT on the original training pairs and all supplied candidates. [Training settings and commands](strong-baseline.md) define the reproducible run. Neither baseline may read reference calls, reference call count, or offline metadata at inference. Avoid an initialization already fine-tuned on the complete xLAM release when treating our xLAM holdout as independent.

## Two extensions

**1. Inference-time filtering.** Rank candidates using a fixed lexical ranker or pretrained embeddings and pass the selected tools to the unchanged SFT generator. No new retriever training is required initially. Report per-required-tool recall, all-required-tools recall, complete-request accuracy, and actual input tokens. Include ranking overhead in any latency measurement. Losing one required tool remains an end-to-end failure.

**2. Similar-tool training.** Add similar but incorrect candidates to training examples, keeping requests and reference arrays unchanged, then fine-tune a separate model from the same initialization. Draw additions from the training tool inventory. Check for duplicate tools and valid alternatives: similarity alone does not establish a negative label.

Compare similar additions with random additions at the same candidate count and on the same base requests. This is an extension control, not a third main baseline. Keep training settings comparable and record token-budget differences. Test the extensions separately before combining them. Their benefits are hypotheses, not established results.

## Evaluation

The primary metric is exact agreement of the complete call multiset, preserving duplicate counts but ignoring order between calls. Ordered exact match is a separate diagnostic. Argument arrays remain ordered. [Scoring rules and commands](../scoring.md) define partial call/name precision and recall, count errors, and invalid outputs.

Report overall scores and breakdowns by task category, repeated versus distinct tools, candidate competition, input length, and quality flags. Compare original inputs with random and similar distractors at matched counts. Keep candidate ordering paired across comparisons or use the same fixed permutations. Do not tune from test errors.

## Data and next steps

[Data documentation](data.md) contains the fixed splits, profiles, grouping rules, and known issues. Original requests and schemas remain intact; derived stress tests inherit the parent's partition. Tools may recur across splits. Unseen-tool evaluation and external BFCL scoring are optional later checks.

The historical 98.1% lexical-selection diagnostic applies only to the old M1 single-call development subset. It is not a complete-call result on the full release. Strong-baseline training and full dev/test evaluation are complete; [results and error analysis](strong-baseline.md#results-and-error-analysis) describe the current starting point. The simple-baseline implementation has been integrated. Remaining MS2 work is to record its full-split scores and usage, compare both systems, and write the report.
