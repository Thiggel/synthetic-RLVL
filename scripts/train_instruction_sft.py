#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_sft import make_sft_data_collator


def _first_user_assistant(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    user_text: str | None = None
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user" and user_text is None:
            user_text = content
        elif role == "assistant" and user_text is not None:
            return user_text, content
    return None


def _row_to_prompt_target(
    row: dict[str, Any],
    *,
    wrap_question_tags: bool,
    wrap_answer_tags: bool,
) -> dict[str, str] | None:
    if isinstance(row.get("messages"), list):
        pair = _first_user_assistant(row["messages"])
        if pair is None:
            return None
        prompt_text, answer_text = pair
    elif "instruction" in row and "output" in row:
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        answer_text = str(row.get("output", "")).strip()
        prompt_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
    else:
        return None

    if not prompt_text.strip() or not answer_text.strip():
        return None

    prompt = prompt_text.strip()
    if wrap_question_tags:
        prompt = f"<question>\n{prompt}\n</question>\n"
    if wrap_answer_tags:
        target = f"<answer>\n{answer_text.strip()}\n</answer>"
    else:
        target = answer_text.strip()
    return {"prompt": prompt, "target": target}


def _format_dataset(
    raw: Dataset,
    *,
    limit: int,
    seed: int,
    format_mode: str,
    wrap_answer_tags: bool,
    single_turn_only: bool = False,
) -> Dataset:
    if limit > 0 and len(raw) > limit:
        raw = raw.shuffle(seed=seed)
    rows: list[dict[str, str]] = []
    for row in raw:
        if single_turn_only:
            messages = row.get("messages")
            if not (
                isinstance(messages, list)
                and len(messages) == 2
                and str(messages[0].get("role", "")).lower() == "user"
                and str(messages[1].get("role", "")).lower() == "assistant"
            ):
                continue
        formatted = _row_to_prompt_target(
            dict(row),
            wrap_question_tags=format_mode == "tagged",
            wrap_answer_tags=wrap_answer_tags and format_mode == "tagged",
        )
        if formatted is not None:
            rows.append(formatted)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("No usable instruction rows after formatting.")
    return Dataset.from_list(rows)


def _tokenize_dataset(ds: Dataset, tokenizer, *, max_length: int, format_mode: str) -> Dataset:
    def tokenize_row(row: dict[str, str]) -> dict[str, list[int]]:
        if format_mode == "chat":
            user = {"role": "user", "content": row["prompt"]}
            assistant = {"role": "assistant", "content": row["target"]}
            prompt_ids = tokenizer.apply_chat_template(
                [user], tokenize=True, add_generation_prompt=True
            )
            input_ids = tokenizer.apply_chat_template(
                [user, assistant], tokenize=True, add_generation_prompt=False
            )
            if input_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("Chat-template assistant response does not extend the user prompt.")
        else:
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[min(len(prompt_ids), len(input_ids)) :]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized = ds.map(tokenize_row, remove_columns=ds.column_names)
    tokenized = tokenized.filter(lambda row: any(label != -100 for label in row["labels"]))
    if len(tokenized) == 0:
        raise ValueError("No instruction rows retain assistant target tokens after truncation.")
    return tokenized


def _load_instruction_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.prepared_dataset:
        prepared = load_from_disk(args.prepared_dataset)
        if not isinstance(prepared, DatasetDict) or not {"train", "eval"}.issubset(prepared):
            raise ValueError("Prepared dataset must be a DatasetDict with train and eval splits.")
        return prepared["train"], prepared["eval"]
    load_kwargs = {"revision": args.dataset_revision} if args.dataset_revision else {}
    if args.train_split == args.eval_split:
        raw = load_dataset(args.dataset, split=args.train_split, **load_kwargs).shuffle(seed=int(args.seed))
        requested = int(args.train_samples) + int(args.eval_samples)
        if len(raw) < requested:
            raise ValueError(f"Dataset has {len(raw)} rows, fewer than requested {requested}.")
        # Use a generous disjoint candidate pool because optional row filters
        # can reject examples before the requested count is reached.
        candidate_count = min(len(raw), max(requested * 3, requested))
        raw = raw.select(range(candidate_count))
        split_at = min(len(raw), max(int(args.train_samples) * 2, int(args.train_samples)))
        raw_train = raw.select(range(split_at))
        raw_eval = raw.select(range(split_at, len(raw)))
    else:
        raw_train = load_dataset(args.dataset, split=args.train_split, **load_kwargs)
        raw_eval = load_dataset(args.dataset, split=args.eval_split, **load_kwargs)
    train = _format_dataset(
        raw_train,
        limit=int(args.train_samples),
        seed=int(args.seed),
        format_mode=str(args.format_mode),
        wrap_answer_tags=bool(args.wrap_answer_tags),
        single_turn_only=bool(args.single_turn_only),
    )
    eval_limit = int(args.eval_samples)
    if len(raw_eval) > eval_limit:
        raw_eval = raw_eval.shuffle(seed=int(args.seed) + 1).select(range(eval_limit))
    eval_ds = _format_dataset(
        raw_eval,
        limit=0,
        seed=int(args.seed) + 1,
        format_mode=str(args.format_mode),
        wrap_answer_tags=bool(args.wrap_answer_tags),
        single_turn_only=bool(args.single_turn_only),
    )
    return train, eval_ds


def _resolve_resume_checkpoint(output_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    if value == "auto":
        if not output_dir.is_dir():
            return None
        candidates = sorted(
            (
                path
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
            ),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
            reverse=True,
        )
        for checkpoint in candidates:
            # A failed FSDP save can leave a newer directory that looks like a
            # checkpoint to get_last_checkpoint but cannot restore optimizer,
            # scheduler, or RNG state. Resume only a fully committed Trainer
            # state so recovery does not silently replay or corrupt training.
            if (
                (checkpoint / "trainer_state.json").is_file()
                and (checkpoint / "scheduler.pt").is_file()
                and any(checkpoint.glob("optimizer.*"))
            ):
                return str(checkpoint)
        return None
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise ValueError(f"Instruction SFT resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA instruction SFT control for OLMo-style causal LMs.")
    parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--prepared-dataset", default=None)
    parser.add_argument("--prepared-dataset-output", default=None)
    parser.add_argument("--train-split", default="train_sft")
    parser.add_argument("--eval-split", default="test_sft")
    parser.add_argument("--run-name", default="sft_instruction_ultrachat200k_olmo3_7b_10k_seed3407")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=12)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--format-mode", choices=["chat", "tagged"], default="tagged")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    parser.add_argument("--no-wrap-answer-tags", dest="wrap_answer_tags", action="store_false")
    parser.set_defaults(wrap_answer_tags=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--single-turn-only", action="store_true")
    parser.add_argument("--full-parameter", action="store_true")
    parser.add_argument("--fsdp", default=None)
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default=None)
    parser.add_argument("--ddp-timeout", type=int, default=1800)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint path, or 'auto' to use the latest checkpoint in output-dir.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--report-to", nargs="*", default=["wandb"])
    parser.add_argument("--dry-run", action="store_true", help="Only load and tokenize data; do not load/train model.")
    parser.add_argument("--audit-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    random.seed(int(args.seed))

    output_dir = Path(args.output_dir or Path(os.environ.get("WORK", os.environ["HOME"])) / "synthetic-RLVL" / "runs" / args.run_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw, eval_raw = _load_instruction_splits(args)
    if args.prepared_dataset_output:
        if args.prepared_dataset:
            raise ValueError("Cannot rewrite an already prepared dataset.")
        prepared_path = Path(args.prepared_dataset_output)
        if prepared_path.exists():
            raise FileExistsError(f"Prepared dataset output already exists: {prepared_path}")
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": train_raw, "eval": eval_raw}).save_to_disk(str(prepared_path))
    if args.format_mode == "chat" and not tokenizer.chat_template:
        raise ValueError(f"Tokenizer for {args.model} does not define a chat template.")
    train_ds = _tokenize_dataset(
        train_raw,
        tokenizer,
        max_length=int(args.max_length),
        format_mode=str(args.format_mode),
    )
    eval_ds = _tokenize_dataset(
        eval_raw,
        tokenizer,
        max_length=int(args.max_length),
        format_mode=str(args.format_mode),
    )

    if args.dry_run:
        lengths = [len(row["input_ids"]) for row in train_ds.select(range(min(128, len(train_ds))))]
        first = train_ds[0]
        supervised_ids = [
            token_id
            for token_id, label in zip(first["input_ids"], first["labels"], strict=True)
            if label != -100
        ]
        def fingerprint(dataset: Dataset) -> str:
            digest = hashlib.sha256()
            for row in dataset:
                digest.update(str(row["prompt"]).encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(row["target"]).encode("utf-8"))
                digest.update(b"\n")
            return digest.hexdigest()

        audit = {
                "format_mode": args.format_mode,
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
                "seed": int(args.seed),
                "formatted_train_rows": len(train_raw),
                "retained_train_rows": len(train_ds),
                "formatted_eval_rows": len(eval_raw),
                "retained_eval_rows": len(eval_ds),
                "train_fingerprint": fingerprint(train_raw),
                "eval_fingerprint": fingerprint(eval_raw),
                "min_len": min(lengths),
                "mean_len": sum(lengths) / len(lengths),
                "max_len": max(lengths),
                "first_prompt": train_raw[0]["prompt"][:500],
                "first_target": train_raw[0]["target"][:500],
                "first_rendered": tokenizer.decode(first["input_ids"][:1000]),
                "first_supervised": tokenizer.decode(supervised_ids[:500]),
            }
        if audit["train_fingerprint"] == audit["eval_fingerprint"]:
            raise ValueError("Train and evaluation fingerprints unexpectedly match.")
        if args.audit_output:
            audit_path = Path(args.audit_output)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return

    model_kwargs = {"torch_dtype": torch.bfloat16 if bool(args.bf16) else torch.float16}
    if not bool(args.full_parameter):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if bool(args.gradient_checkpointing):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if not bool(args.full_parameter):
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            target_modules=list(args.lora_target_modules),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    report_to = [str(x).lower() for x in args.report_to]
    if "wandb" in report_to and not bool(args.full_parameter):
        group = os.environ.get("WANDB_RUN_GROUP") or os.environ.get("WANDB_GROUP")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=args.run_name,
            group=group,
            config=vars(args),
        )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=str(args.run_name),
        per_device_train_batch_size=int(args.per_device_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        max_steps=int(args.max_steps),
        num_train_epochs=float(args.num_train_epochs),
        warmup_steps=int(args.warmup_steps),
        warmup_ratio=float(args.warmup_ratio),
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bool(args.bf16),
        fp16=not bool(args.bf16),
        report_to=list(args.report_to),
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False} if bool(args.gradient_checkpointing) else None,
        fsdp=args.fsdp,
        ddp_timeout=int(args.ddp_timeout),
        fsdp_config=(
            {
                "transformer_layer_cls_to_wrap": str(args.fsdp_transformer_layer_cls_to_wrap),
                "use_orig_params": True,
                "limit_all_gathers": True,
                "sync_module_states": True,
            }
            if args.fsdp
            else None
        ),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=make_sft_data_collator(tokenizer),
    )
    resume_checkpoint = _resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if resume_checkpoint is not None:
        print(f"Resuming instruction SFT from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    tmp_dirs = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if int(args.save_total_limit) <= 0:
        for path in tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
