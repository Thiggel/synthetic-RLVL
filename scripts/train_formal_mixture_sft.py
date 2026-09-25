#!/usr/bin/env python
"""Full-parameter SFT on a Dolci + rlvlgen formal-CoT mixture (transformers 5.x).

A sibling of train_instruction_sft.py for the Qwen3.5 mixture sweep; that script
is unchanged. Differences:
  * input is a prepared DatasetDict from scripts/data/build_formal_mixture_sft.py
    (train/eval rows with prompt, target, source, family, mask_tool_results);
  * rendering and label masking go through scripts/formal_chat_format.py, the
    same module the vLLM evaluator uses: the model's own chat template, loss on
    assistant tokens only, and for rows with mask_tool_results the tool-result
    contents plus the closing "</result>\\n" are masked (the opening "<result>"
    stays in the loss);
  * transformers 5 API (processing_class=, dtype=), no hydra/synthrlvl imports.
The optimisation recipe matches the Dolci post-SFT runs: seed 3407, lr 5e-6,
linear decay with 3% warmup, global batch 128, one epoch.

--dry-run tokenizes, writes a token/mask audit (--audit-output) and exits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formal_chat_format import encode_example, template_matches  # noqa: E402


class DivergenceGuard(TrainerCallback):
    """Raise on a non-finite or exploding loss (copied from train_instruction_sft.py)."""

    def __init__(self, max_loss: float = 50.0):
        self.max_loss = float(max_loss)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        grad_norm = logs.get("grad_norm")
        bad = not math.isfinite(loss) or loss > self.max_loss
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            bad = True
        if bad:
            raise RuntimeError(f"training diverged at global step {state.global_step}: loss={loss} grad_norm={grad_norm}")


def make_collator(pad_id: int):
    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = n - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [pad_id] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
    return collate


def tokenize_split(ds, tokenizer, *, max_length: int, max_truncated_frac: float | None, name: str, num_proc: int):
    def row_fn(row):
        enc = encode_example(tokenizer, row["prompt"], row["target"], bool(row["mask_tool_results"]))
        ids, labels = enc["input_ids"], enc["labels"]
        truncated = len(ids) > max_length
        ids, labels = ids[:max_length], labels[:max_length]
        return {"input_ids": ids, "labels": labels, "_truncated": truncated,
                "_n_loss": sum(l != -100 for l in labels),
                "_n_masked_tool": sum(1 for i, l in enumerate(labels) if l == -100 and i >= enc["n_prompt"])}

    keep = [c for c in ("source", "family") if c in ds.column_names]
    out = ds.map(row_fn, remove_columns=[c for c in ds.column_names if c not in keep], num_proc=num_proc)
    n_trunc = sum(out["_truncated"])
    frac = n_trunc / max(len(out), 1)
    print(f"[truncation] {name}: {n_trunc}/{len(out)} = {frac:.4%} exceeded max_length={max_length}", flush=True)
    # Per-source counts. A truncated synthetic proof loses its conclusion and
    # answer, so any rlvlgen truncation fails closed regardless of the budget.
    if "source" in out.column_names:
        by_src: dict[str, list[int]] = {}
        for s, t in zip(out["source"], out["_truncated"]):
            by_src.setdefault(s, [0, 0])
            by_src[s][0] += int(t)
            by_src[s][1] += 1
        print(f"[truncation] {name} by source: " + ", ".join(f"{s} {a}/{b}" for s, (a, b) in sorted(by_src.items())), flush=True)
        if max_truncated_frac is not None and by_src.get("rlvlgen", [0, 0])[0] > 0:
            raise ValueError(f"Refusing to train: {by_src['rlvlgen'][0]} rlvlgen rows exceed max_length={max_length}")
    if max_truncated_frac is not None and frac > max_truncated_frac:
        raise ValueError(f"Refusing to train: {name} truncation {frac:.4%} > {max_truncated_frac:.4%}")
    out = out.filter(lambda r: r["_n_loss"] > 0)
    return out


def audit(train_ds, tokenizer, raw_train) -> dict:
    lengths = [len(x) for x in train_ds["input_ids"]]
    n_loss, n_tool = train_ds["_n_loss"], train_ds["_n_masked_tool"]
    srcs = train_ds["source"] if "source" in train_ds.column_names else ["?"] * len(train_ds)
    stats: dict = {}
    for src in sorted(set(srcs)):
        idx = [i for i, s in enumerate(srcs) if s == src]
        lens = [lengths[i] for i in idx]
        stats[src] = {"rows": len(idx), "tokens": sum(lens), "mean_len": sum(lens) / len(lens), "max_len": max(lens),
                      "loss_tokens": sum(n_loss[i] for i in idx), "masked_tool_tokens": sum(n_tool[i] for i in idx)}
    example = None
    for i, fam in enumerate(train_ds["family"] if "family" in train_ds.column_names else []):
        if fam == "tools":
            row = train_ds[i]
            spans, cur, buf = [], None, []
            for t, l in zip(row["input_ids"], row["labels"]):
                m = l == -100
                if m != cur and buf:
                    spans.append({"loss": not cur, "text": tokenizer.decode(buf)})
                    buf = []
                cur = m
                buf.append(t)
            spans.append({"loss": not cur, "text": tokenizer.decode(buf)})
            example = spans
            break
    return {"per_source": stats, "tools_example_spans": example}


def resolve_resume(output_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    if value != "auto":
        return value
    cands = sorted((p for p in output_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                   key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for c in cands:
        if (c / "trainer_state.json").is_file() and (c / "scheduler.pt").is_file() and any(c.glob("optimizer*")):
            return str(c)
    return None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prepared-dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-truncated-frac", type=float, default=0.005)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--eval-rows", type=int, default=None, help="cap the eval split")
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--fsdp", default=None, help='e.g. "full_shard auto_wrap"; omit for DDP')
    ap.add_argument("--fsdp-layer-cls", default="Qwen3_5DecoderLayer")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--resume-from-checkpoint", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit-output", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise ValueError(f"{args.model} has no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_from_disk(args.prepared_dataset)
    if not isinstance(data, DatasetDict) or not {"train", "eval"} <= set(data):
        raise ValueError("prepared dataset must be a DatasetDict with train and eval")
    probe = data["train"].select(range(min(500, len(data["train"]))))
    n_bad = sum(not template_matches(tokenizer, p, t) for p, t in zip(probe["prompt"], probe["target"]))
    print(f"[template] {n_bad}/{len(probe)} probe rows differ from the model chat template", flush=True)
    if n_bad > 0.01 * len(probe):
        raise ValueError("the model chat template is not ChatML-compatible; formal_chat_format needs updating")
    eval_raw = data["eval"]
    if args.eval_rows is not None:
        eval_raw = eval_raw.select(range(min(args.eval_rows, len(eval_raw))))
    train_ds = tokenize_split(data["train"], tokenizer, max_length=args.max_length,
                              max_truncated_frac=args.max_truncated_frac, name="train", num_proc=args.num_proc)
    # The Dolci eval split only feeds the eval-loss monitor, so its truncation
    # is reported, not gated (4/200 = 2% of the smoke eval rows exceed 4096).
    eval_ds = tokenize_split(eval_raw, tokenizer, max_length=args.max_length,
                             max_truncated_frac=None, name="eval", num_proc=args.num_proc)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        rep = audit(train_ds, tokenizer, data["train"])
        rep.update({"model": args.model, "prepared_dataset": args.prepared_dataset,
                    "train_rows": len(train_ds), "eval_rows": len(eval_ds)})
        print(json.dumps(rep["per_source"], indent=2), flush=True)
        if args.audit_output:
            Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.audit_output).write_text(json.dumps(rep, indent=2) + "\n")
    if args.dry_run:
        return
    drop = [c for c in train_ds.column_names if c not in ("input_ids", "labels")]
    train_ds, eval_ds = train_ds.remove_columns(drop), eval_ds.remove_columns(drop)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_ratio,  # transformers v5: a float in [0,1) is a ratio of total steps
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        fsdp=args.fsdp or "",
        fsdp_config={"transformer_layer_cls_to_wrap": [args.fsdp_layer_cls], "use_orig_params": True,
                     "limit_all_gathers": True, "sync_module_states": True} if args.fsdp else None,
        ddp_timeout=1800,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=make_collator(tokenizer.pad_token_id),
        callbacks=[DivergenceGuard()],
    )
    resume = resolve_resume(output_dir, args.resume_from_checkpoint)
    if resume:
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    final = output_dir / "final"
    if args.fsdp:
        # Under FSDP (9B) the post-train trainer.evaluate()/save_model() hung in
        # smoke 6595: rank 1 exited cleanly while rank 0 spun without I/O for
        # >15 min. The end-of-training checkpoint that Trainer writes with
        # save_strategy="steps" already holds the full consolidated
        # model.safetensors (verified: all 427 text-model tensors, fp32 because
        # FSDP keeps fp32 master weights), so promote it instead.
        trainer.accelerator.wait_for_everyone()
        metrics = next((dict(h) for h in reversed(trainer.state.log_history) if "eval_loss" in h), {})
        metrics["note"] = "last in-training eval; no post-train evaluate under FSDP"
        if trainer.is_world_process_zero():
            ckpt = output_dir / f"checkpoint-{trainer.state.global_step}"
            if not (ckpt / "model.safetensors").exists() and not (ckpt / "model.safetensors.index.json").exists():
                raise RuntimeError(f"no consolidated model in {ckpt}")
            final.mkdir(parents=True, exist_ok=True)
            for f in ckpt.iterdir():
                if f.name in {"config.json", "generation_config.json", "chat_template.jinja"} or (
                    f.name.startswith("model") and f.name.endswith((".safetensors", ".json"))
                ):
                    shutil.move(str(f), str(final / f.name))
    else:
        metrics = trainer.evaluate()
        trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final))
        (final / "final_eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        for p in output_dir.glob("checkpoint-*"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
