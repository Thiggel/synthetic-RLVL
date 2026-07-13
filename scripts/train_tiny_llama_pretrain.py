#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.datasets import MaterializedSyntheticDataset
from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


SIZE_CONFIGS = {
    "50m": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4),
    "100m": dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4),
    "200m": dict(hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small random-init Llama-style causal LMs on synthetic traces.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--size", choices=sorted(SIZE_CONFIGS), required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--dataset-id", default="flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--train-subset", default="train_fixedtarget_up_to_10_50k")
    parser.add_argument("--eval-subset", default="val_step_10_1k")
    parser.add_argument("--template", choices=[item.value for item in TemplateName], default="logic")
    parser.add_argument("--train-max-step", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument(
        "--require-unique-train-examples",
        action="store_true",
        help="Fail if the optimizer-step budget would require reusing a training row.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint path or 'auto' for the newest checkpoint in output-dir.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "synthetic-rlvl"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    return parser.parse_args()


def _task_cfg(template: str, train_max_step: int, seed: int) -> TaskConfig:
    return TaskConfig(
        template=TemplateName(template),
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        train_steps=StepRange(1, int(train_max_step)),
        val_steps=StepRange(1, int(train_max_step)),
        seed=int(seed),
    )


def _load_texts(args: argparse.Namespace, tokenizer) -> tuple[Dataset, Dataset]:
    mat = MaterializedSyntheticDataset()
    cfg = _task_cfg(args.template, args.train_max_step, args.seed)

    def rows_to_dataset(rows: list[dict]) -> Dataset:
        items = []
        for row in rows:
            sample = task_sample_from_materialized_row(row, cfg=cfg)
            text = sample.prompt + sample.target + (tokenizer.eos_token or "")
            items.append({"text": text, "depth": int(row.get("depth", sample.depth))})
        ds = Dataset.from_list(items)

        def tokenize(batch: dict) -> dict:
            encoded = tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=int(args.max_length),
            )
            encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
            return encoded

        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    train_rows = mat.load_rows(
        subset=args.train_subset,
        dataset_id=args.dataset_id,
        local_root=args.local_root,
        split="train",
        limit=int(args.train_samples),
    )
    eval_rows = mat.load_rows(
        subset=args.eval_subset,
        dataset_id=args.dataset_id,
        local_root=args.local_root,
        split="train",
        limit=int(args.eval_samples),
    )
    return rows_to_dataset(train_rows), rows_to_dataset(eval_rows)


def _collator(tokenizer):
    pad_id = int(tokenizer.pad_token_id)

    def collate(features: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            cur = list(item["input_ids"])
            pad = max_len - len(cur)
            input_ids.append(cur + [pad_id] * pad)
            attention_mask.append([1] * len(cur) + [0] * pad)
            labels.append(cur + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _resolve_resume_checkpoint(output_dir: Path, value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "false", "0"}:
        return None
    if str(value).strip().lower() != "auto":
        return str(value)
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / "trainer_state.json").is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints, key=lambda item: item[0])[1]) if checkpoints else None


def _required_train_examples(
    *,
    max_steps: int,
    per_device_batch_size: int,
    grad_accum: int,
    world_size: int = 1,
) -> int:
    return (
        int(max_steps)
        * int(per_device_batch_size)
        * int(grad_accum)
        * int(world_size)
    )


def _validate_unique_train_budget(
    *,
    dataset_rows: int,
    max_steps: int,
    per_device_batch_size: int,
    grad_accum: int,
    world_size: int = 1,
) -> int:
    required = _required_train_examples(
        max_steps=max_steps,
        per_device_batch_size=per_device_batch_size,
        grad_accum=grad_accum,
        world_size=world_size,
    )
    if required > int(dataset_rows):
        raise ValueError(
            "Unique-example training requires at least "
            f"{required} rows, but the loaded dataset has {dataset_rows}."
        )
    return required


def main() -> None:
    args = _parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir or Path(os.environ.get("WORK", ".")) / "synthetic-RLVL" / "runs" / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg_kwargs = dict(SIZE_CONFIGS[args.size])
    model_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max(
            8192,
            int(args.max_length),
            int(args.max_position_embeddings or 0),
        ),
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **cfg_kwargs,
    )
    model = LlamaForCausalLM(model_cfg)
    if bool(args.bf16):
        model = model.to(dtype=torch.bfloat16)
    train_ds, eval_ds = _load_texts(args, tokenizer)
    required_train_examples = _required_train_examples(
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    if args.require_unique_train_examples:
        _validate_unique_train_budget(
            dataset_rows=len(train_ds),
            max_steps=args.max_steps,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            config=vars(args)
            | {
                "model_config": model_cfg.to_dict(),
                "required_train_examples": required_train_examples,
            },
        )

    arg_kwargs = dict(
        output_dir=str(output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=int(args.per_device_batch_size),
        per_device_eval_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        max_steps=int(args.max_steps),
        warmup_steps=int(args.warmup_steps),
        logging_steps=int(args.logging_steps),
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bool(args.bf16),
        fp16=not bool(args.bf16),
        report_to=["wandb"] if args.wandb_project else [],
        remove_unused_columns=False,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        arg_kwargs["evaluation_strategy"] = "steps"
    else:
        arg_kwargs["eval_strategy"] = "steps"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**arg_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collator(tokenizer),
    )
    trainer.train(
        resume_from_checkpoint=_resolve_resume_checkpoint(
            output_dir,
            args.resume_from_checkpoint,
        )
    )
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
