from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .task import TaskBuilder, task_sample_from_materialized_row
from .types import TaskConfig


@dataclass(frozen=True)
class SFTDatasetBundle:
    train: Dataset
    eval: Dataset


def build_sft_dataset(
    builder: TaskBuilder,
    *,
    train_samples: int,
    eval_samples: int,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> SFTDatasetBundle:
    train_rows = [s.__dict__ for s in builder.build_samples(train_samples, train=True)]
    eval_rows = [s.__dict__ for s in builder.build_samples(eval_samples, train=False)]
    return _tokenize_sft_rows(train_rows, eval_rows, tokenizer=tokenizer, max_length=max_length)


def build_sft_dataset_from_materialized_rows(
    *,
    train_rows: list[dict],
    eval_rows: list[dict],
    task_cfg: TaskConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> SFTDatasetBundle:
    train_task_rows = [task_sample_from_materialized_row(r, cfg=task_cfg).__dict__ for r in train_rows]
    eval_task_rows = [task_sample_from_materialized_row(r, cfg=task_cfg).__dict__ for r in eval_rows]
    return _tokenize_sft_rows(train_task_rows, eval_task_rows, tokenizer=tokenizer, max_length=max_length)


def _tokenize_sft_rows(
    train_rows: list[dict],
    eval_rows: list[dict],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> SFTDatasetBundle:

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    def tokenize_row(row: dict) -> dict:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[min(len(prompt_ids), len(input_ids)) :]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    cols_to_remove = train_ds.column_names
    train_tok = train_ds.map(tokenize_row, remove_columns=cols_to_remove)
    eval_tok = eval_ds.map(tokenize_row, remove_columns=cols_to_remove)
    return SFTDatasetBundle(train=train_tok, eval=eval_tok)
