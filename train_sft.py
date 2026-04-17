from __future__ import annotations

import os
import inspect

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from synthrlvl.config import eval_loop_config_from_sft, task_config_from_cfg
from synthrlvl.datasets import MaterializedSyntheticDataset
from synthrlvl.sft_data import build_sft_dataset_from_materialized_rows
from synthrlvl.types import TaskConfig
from synthrlvl.eval_loop import UnifiedEvaluator


class SFTTrainerWithGeneration(Trainer):
    def __init__(self, *args, tokenizer, task_cfg: TaskConfig, eval_cfg, log_generations_cfg, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self._tokenizer = tokenizer
        self._task_cfg = task_cfg
        self._eval_cfg = eval_cfg
        self._log_generations_cfg = log_generations_cfg
        self._evaluator = UnifiedEvaluator()

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        gen_metrics, samples = self._evaluator.evaluate_model(
            self.model,
            self._tokenizer,
            task_cfg=self._task_cfg,
            eval_cfg=self._eval_cfg,
            device=self.model.device,
            collect_samples=int(self._log_generations_cfg.num_samples),
        )
        self.log(gen_metrics)
        metrics.update(gen_metrics)
        if wandb.run is not None and samples:
            table = wandb.Table(columns=["source", "step", "prompt", "generation", "gold_answer", "format_ok", "correct", "valid"])
            for row in samples:
                table.add_data(
                    row["source"],
                    row["step"],
                    str(row["prompt"])[: int(self._log_generations_cfg.max_chars)],
                    str(row["generation"])[: int(self._log_generations_cfg.max_chars)],
                    row["gold_answer"],
                    row["format_ok"],
                    row["correct"],
                    row["valid"],
                )
            wandb.log({"val/generations": table}, step=int(self.state.global_step))
        return metrics


def make_sft_data_collator(tokenizer):
    pad_id = int(tokenizer.pad_token_id)

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


@hydra.main(config_path="conf", config_name="sft", version_base=None)
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    mat_ds = MaterializedSyntheticDataset()

    task_cfg = task_config_from_cfg(cfg)
    eval_cfg = eval_loop_config_from_sft(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16 if cfg.model.bf16 else torch.float16,
        device_map="auto",
    )

    if bool(cfg.model.lora.enabled):
        lora_cfg = LoraConfig(
            r=int(cfg.model.lora.r),
            lora_alpha=int(cfg.model.lora.alpha),
            target_modules=list(cfg.model.lora.target_modules),
            lora_dropout=float(cfg.model.lora.dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    if str(cfg.data.source) != "materialized":
        raise ValueError("SFT now expects materialized data (`data.source=materialized`).")

    train_subset = str(cfg.data.materialized.train_subset)
    if train_subset == "auto":
        train_subset = mat_ds.train_subset_for_max_step(int(cfg.task.train_max_step))
    eval_subset = str(cfg.data.materialized.eval_subset)
    if eval_subset == "auto":
        eval_subset = mat_ds.val_subset_name(int(cfg.task.val_max_step))
    train_rows = mat_ds.load_rows(
        subset=train_subset,
        dataset_id=cfg.data.materialized.dataset_id,
        local_root=cfg.data.materialized.local_root,
        split="train",
        limit=int(cfg.data.train_samples),
    )
    eval_rows = mat_ds.load_rows(
        subset=eval_subset,
        dataset_id=cfg.data.materialized.dataset_id,
        local_root=cfg.data.materialized.local_root,
        split="train",
        limit=int(cfg.data.eval_samples),
    )
    datasets = build_sft_dataset_from_materialized_rows(
        train_rows=train_rows,
        eval_rows=eval_rows,
        task_cfg=task_cfg,
        tokenizer=tokenizer,
        max_length=int(cfg.data.max_length),
    )

    report_to = [str(x).lower() for x in list(cfg.logging.report_to)]
    wandb_enabled = "wandb" in report_to
    if wandb_enabled:
        group = os.environ.get("WANDB_RUN_GROUP") or os.environ.get("WANDB_GROUP")
        os.environ.setdefault("WANDB_NAME", str(cfg.run_name))
        if group:
            os.environ["WANDB_RUN_GROUP"] = str(group)
            os.environ["WANDB_GROUP"] = str(group)
        if wandb.run is None:
            init_kwargs = {
                "name": str(cfg.run_name),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if os.environ.get("WANDB_PROJECT"):
                init_kwargs["project"] = os.environ["WANDB_PROJECT"]
            if os.environ.get("WANDB_ENTITY"):
                init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
            if group:
                init_kwargs["group"] = str(group)
            wandb.init(**init_kwargs)
        elif group and getattr(wandb.run, "group", None) != str(group):
            wandb.run.group = str(group)
            wandb.run.update()

    arg_kwargs = dict(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        per_device_train_batch_size=int(cfg.train.per_device_batch_size),
        per_device_eval_batch_size=int(cfg.train.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        num_train_epochs=float(cfg.train.num_epochs),
        learning_rate=float(cfg.train.lr),
        max_steps=int(cfg.train.max_steps),
        warmup_steps=int(cfg.train.warmup_steps),
        logging_steps=int(cfg.train.logging_steps),
        eval_steps=int(cfg.train.eval_steps),
        save_steps=int(cfg.train.save_steps),
        save_total_limit=int(cfg.train.save_total_limit),
        bf16=bool(cfg.model.bf16),
        fp16=not bool(cfg.model.bf16),
        report_to=list(cfg.logging.report_to),
        remove_unused_columns=False,
        label_names=["labels"],
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        arg_kwargs["evaluation_strategy"] = "steps"
    else:
        arg_kwargs["eval_strategy"] = "steps"
    args = TrainingArguments(**arg_kwargs)

    trainer = SFTTrainerWithGeneration(
        model=model,
        args=args,
        train_dataset=datasets.train,
        eval_dataset=datasets.eval,
        tokenizer=tokenizer,
        task_cfg=task_cfg,
        eval_cfg=eval_cfg,
        log_generations_cfg=cfg.log_generations,
        data_collator=make_sft_data_collator(tokenizer),
    )

    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "final"))


if __name__ == "__main__":
    main()
