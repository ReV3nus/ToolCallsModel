"""Response-only LoRA SFT and generation for the fixed full xLAM splits.

Run profile, train, or predict; all settings and real outputs are saved separately
from the source data. Only query/tools enter inference prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from score import strict_json

SYSTEM = ('Generate all function calls needed to satisfy the request using only the supplied tools. '
          'Return only a JSON array. Each element must contain exactly "name" and "arguments". '
          'Include repeated calls when required. Preserve argument value types. Do not add prose or code fences.')
PROMPT_VERSION = "xlam-json-v1"


def json_text(value):
    """Serialize JSON without rounding source decimal literals through floats."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite decimal")
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + json_text(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(map(json_text, value)) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not config.get("model_revision"):
        raise ValueError("An immutable model revision is required")
    return config


def records(path, limit=None):
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            yield strict_json(line)


def messages(row):
    # Whitelist fields: answers, call counts, IDs, and metadata are excluded.
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json_text({"query": row["query"], "tools": row["tools"]})}]


def prompt_ids(tokenizer, row):
    return tokenizer.apply_chat_template(messages(row), tokenize=True, add_generation_prompt=True)


def training_example(tokenizer, row, max_length):
    prefix = prompt_ids(tokenizer, row)
    target = tokenizer(json_text(row["answers"]), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prefix) + len(target) > max_length:
        raise ValueError(f"ID {row['id']}: {len(prefix)+len(target)} tokens exceed {max_length}; no truncation performed")
    return {"input_ids": prefix + target, "attention_mask": [1] * (len(prefix) + len(target)),
            "labels": [-100] * len(prefix) + target}


def get_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source(config), revision=config["model_revision"], cache_dir=config["cache_dir"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_source(config):
    location = config.get("model_local_dir")
    if location and Path(location, ".model_revision").exists():
        if Path(location, ".model_revision").read_text(encoding="utf-8").strip() != config["model_revision"]:
            raise ValueError("Local model revision differs from configured revision")
        return location
    return config["model_id"]


def percentile_summary(values):
    values = sorted(values)
    return {"n": len(values), "min": values[0], "mean": round(sum(values)/len(values), 2),
            "median": values[(len(values)-1)//2], "p95": values[round((len(values)-1)*.95)],
            "p99": values[round((len(values)-1)*.99)], "max": values[-1]}


def profile(config, output):
    tokenizer = get_tokenizer(config)
    result = {"model": config["model_id"], "revision": config["model_revision"], "prompt_version": PROMPT_VERSION,
              "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "splits": {}}
    for split in ("train", "dev"):
        path = Path(config["data_dir"]) / f"{split}.jsonl"
        inputs, targets = [], []
        for row in records(path):
            inputs.append(len(prompt_ids(tokenizer, row)))
            targets.append(len(tokenizer(json_text(row["answers"]), add_special_tokens=False)["input_ids"]) + 1)
        totals = [a+b for a, b in zip(inputs, targets)]
        result["splits"][split] = {"input_tokens": percentile_summary(inputs), "target_tokens": percentile_summary(targets),
                                    "total_tokens": percentile_summary(totals), "source_sha256": sha(path),
                                    "over_configured_context": sum(n > config["max_seq_length"] for n in totals),
                                    "targets_over_generation_budget": sum(n > config["max_new_tokens"] for n in targets)}
        print(json.dumps({"profiled": split, **result["splits"][split]}, ensure_ascii=False), flush=True)
    write_json(output, result)


class TokenDataset:
    def __init__(self, path, tokenizer, max_length, limit=None):
        self.features = [training_example(tokenizer, row, max_length) for row in records(path, limit)]
        if not self.features:
            raise ValueError("Empty training/evaluation dataset")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class ResponseCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        length = max(len(f["input_ids"]) for f in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            extra = length - len(feature["input_ids"])
            result["input_ids"].append(feature["input_ids"] + [self.pad_id] * extra)
            result["attention_mask"].append(feature["attention_mask"] + [0] * extra)
            result["labels"].append(feature["labels"] + [-100] * extra)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def provenance(config):
    import torch
    return {"config": config, "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM,
            "source_hashes": {s: sha(Path(config["data_dir"]) / f"{s}.jsonl") for s in ("train", "dev")},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "started_utc": datetime.now(timezone.utc).isoformat()}


def train(config, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training configuration")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    tokenizer = get_tokenizer(config)
    train_set = TokenDataset(Path(config["data_dir"])/"train.jsonl", tokenizer, config["max_seq_length"], args.train_limit)
    eval_limit = args.eval_limit or config["eval_examples"]
    eval_set = TokenDataset(Path(config["data_dir"])/"dev.jsonl", tokenizer, config["max_seq_length"], eval_limit)
    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(model_source(config), revision=config["model_revision"], cache_dir=config["cache_dir"],
                                                torch_dtype=torch.bfloat16 if bf16 else torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                            bias="none", task_type="CAUSAL_LM", revision=config["model_revision"]))
    model.peft_config["default"].base_model_name_or_path = config["model_id"]
    model.print_trainable_parameters()
    info = provenance(config)
    info.update(train_examples=len(train_set), evaluation_examples=len(eval_set),
                supervised_tokens=sum(sum(t != -100 for t in f["labels"]) for f in train_set.features),
                total_training_tokens=sum(len(f["input_ids"]) for f in train_set.features), max_steps=args.max_steps)
    if args.resume:
        old = json.loads((output/"run.json").read_text(encoding="utf-8"))
        for key in ("config", "prompt_version", "source_hashes", "train_examples", "evaluation_examples", "max_steps"):
            if old[key] != info[key]:
                raise ValueError(f"Resume configuration changed: {key}")
    else:
        write_json(output/"run.json", info)
        (output/"environment.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+"\n", encoding="utf-8")
    class LogCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            entry = {"step": state.global_step, **(logs or {}), "gpu_allocated_gib": round(torch.cuda.memory_allocated()/2**30, 3),
                     "gpu_peak_gib": round(torch.cuda.max_memory_allocated()/2**30, 3), "time_utc": datetime.now(timezone.utc).isoformat()}
            with (output/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    intervals = min(config["save_steps"], args.max_steps) if args.max_steps > 0 else config["save_steps"]
    training_args = TrainingArguments(output_dir=str(output), num_train_epochs=config["epochs"], max_steps=args.max_steps,
        learning_rate=config["learning_rate"], per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"], warmup_ratio=config["warmup_ratio"], weight_decay=config["weight_decay"],
        bf16=bf16, fp16=not bf16, tf32=True, optim="adamw_torch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, eval_strategy="steps", eval_steps=intervals, save_steps=intervals,
        save_total_limit=2, logging_steps=min(config["logging_steps"], 5) if args.max_steps > 0 else config["logging_steps"],
        prediction_loss_only=True, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", disable_tqdm=True, dataloader_num_workers=0, seed=config["seed"], data_seed=config["seed"],
        remove_unused_columns=False, group_by_length=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=eval_set,
                      data_collator=ResponseCollator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[LogCallback()])
    torch.cuda.reset_peak_memory_stats()
    resume_path = args.resume
    if resume_path == "latest":
        from transformers.trainer_utils import get_last_checkpoint
        resume_path = get_last_checkpoint(str(output))
        if resume_path is None:
            raise ValueError("No checkpoint available to resume")
    result = trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(output/"adapter"))
    tokenizer.save_pretrained(output/"adapter")
    trainer.save_state()
    write_json(output/"train_metrics.json", {**result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
                                            "best_eval_loss": trainer.state.best_metric, "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30})
    print(json.dumps({"saved_adapter": str(output/"adapter"), **result.metrics}), flush=True)


def predict(config, args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, set_seed
    from score import evaluate, read_rows
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    set_seed(config["seed"])
    source = Path(config["data_dir"])/f"{args.split}.jsonl"
    selected = list(records(source, args.limit))
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ValueError("Prediction output exists; choose a new path or --resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".run.json")
    settings = {"config": config, "prompt_version": PROMPT_VERSION, "source_sha256": sha(source),
                "adapter_config_sha256": sha(Path(args.adapter)/"adapter_config.json"),
                "adapter_weights_sha256": sha(Path(args.adapter)/"adapter_model.safetensors"),
                "split": args.split, "limit": args.limit, "batch_size": args.batch_size or config["generation_batch_size"],
                "selected_ids": [r["id"] for r in selected]}
    if args.resume:
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("Inference settings changed; use a new output")
    else:
        previous = {"settings": settings, "sessions": []}
        write_json(sidecar, previous)
    completed = read_rows(output.read_bytes(), "existing predictions") if output.exists() else {}
    if completed.keys() - set(settings["selected_ids"]):
        raise ValueError("Existing output has IDs outside selected records")
    tokenizer = get_tokenizer(config)
    tokenizer.padding_side = "left"
    pending = [(r["id"], prompt_ids(tokenizer, r)) for r in selected if r["id"] not in completed]
    pending.sort(key=lambda pair: len(pair[1]))
    model = AutoModelForCausalLM.from_pretrained(model_source(config), revision=config["model_revision"], cache_dir=config["cache_dir"],
                                                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()
    model.config.use_cache = True
    context = min(tokenizer.model_max_length, model.config.max_position_embeddings)
    if pending and max(len(ids) for _, ids in pending) + config["max_new_tokens"] > context:
        raise ValueError("Generation would exceed model context; no truncation performed")
    stop_ids = model.generation_config.eos_token_id
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids)
    stop_ids = list(dict.fromkeys([tokenizer.eos_token_id, *stop_ids]))
    previous["stop_token_ids"] = stop_ids
    start, generated = time.monotonic(), 0
    batch_size = args.batch_size or config["generation_batch_size"]
    session = {"examples": 0, "seconds": 0, "generated_tokens": 0, "complete": False}
    previous["sessions"].append(session)
    write_json(sidecar, previous)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index in range(0, len(pending), batch_size):
            batch = pending[index:index+batch_size]
            encoded = tokenizer.pad({"input_ids": [ids for _, ids in batch], "attention_mask": [[1]*len(ids) for _, ids in batch]}, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                tokens = model.generate(**encoded, do_sample=False, max_new_tokens=config["max_new_tokens"], pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=stop_ids)
            for (source_id, ids), sequence in zip(batch, tokens[:, encoded["input_ids"].shape[1]:].tolist()):
                eos = next((j for j, token in enumerate(sequence) if token in stop_ids), None)
                content = sequence[:eos] if eos is not None else sequence
                text = tokenizer.decode(content, skip_special_tokens=False)
                item = {"id": source_id, "prediction": text, "input_tokens": len(ids), "generated_tokens": len(content)+(eos is not None),
                        "finish_reason": "eos" if eos is not None else "length"}
                generated += item["generated_tokens"]
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            session.update(examples=min(index+batch_size,len(pending)), seconds=time.monotonic()-start, generated_tokens=generated)
            write_json(sidecar, previous)
            if index % (batch_size*10) == 0:
                print(json.dumps({"predicted": min(index+batch_size,len(pending)), "remaining": len(pending)-min(index+batch_size,len(pending)),
                                  "seconds": round(time.monotonic()-start, 1)}), flush=True)
    session.update(complete=True, seconds=time.monotonic()-start)
    previous["prediction_sha256"] = sha(output)
    write_json(sidecar, previous)
    gold = {r["id"]: r for r in selected}
    metadata = read_rows((Path(config["data_dir"])/"metadata.jsonl").read_bytes(), "metadata")
    scored, details = evaluate(gold, read_rows(output.read_bytes(), "predictions"), metadata=metadata)
    scored["evaluation_scope"] = {"split": args.split, "examples": len(selected), "full_split": args.limit is None}
    scored["inputs"] = {"gold_source_sha256": settings["source_sha256"], "predictions_sha256": sha(output),
                         "model_revision": config["model_revision"], "adapter_weights_sha256": settings["adapter_weights_sha256"]}
    write_json(output.with_suffix(".metrics.json"), scored)
    output.with_suffix(".details.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False)+"\n" for d in details), encoding="utf-8")
    if args.limit is not None:
        output.with_suffix(".gold.jsonl").write_text("".join(json_text(r)+"\n" for r in selected), encoding="utf-8")
    print(json.dumps({"metrics": scored["metrics"], "scope": scored["evaluation_scope"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "train", "predict"))
    parser.add_argument("--config", type=Path, default=Path("configs/strong-baseline.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume latest/path checkpoint, or resume prediction output")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if any(v is not None and v <= 0 for v in (args.train_limit, args.eval_limit, args.limit, args.batch_size)):
        parser.error("Example limits and batch size must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_config(args.config)
    if args.command == "profile":
        profile(config, args.output)
    elif args.command == "train":
        train(config, args)
    else:
        if not args.adapter:
            parser.error("--adapter is required for prediction")
        predict(config, args)


if __name__ == "__main__":
    main()
