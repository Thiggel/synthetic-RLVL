#!/usr/bin/env python
"""Stage-2 GRPO on the Olmo 3 RL prompts (allenai/Dolci-Instruct-RL) with checker rewards.

docs/research_plan.md, Stage 2. One run = one arm:
  --arm correct          G0 (--no-tag, the X=0 policy) / G1 (tagged, the best-X policy)
  --arm correct_x_valid  G2     --arm gvc  G3     --arm valid  G4     --arm lines  G5 (dense line credit)   --arm lines_fmt  G5c (+ format gate)
  --arm frac   the user's "%grammatical lines + %valid lines + correct"
  --arm frac_hard  G6: frac with the line credit gated on premises (numeric check; exact faithfulness on
                   --benches gen items), format and no circular given; G6g adds gen to the benches
  --arm cvf  G7: correct x valid x faithful premises, all-or-nothing (no partial credit to hack)
  --arm cvf_fmt  cvf x format_ok (G10 drifted out of the Answer: format under cvf, then looped; L1_cvf from step 51)
  --no-mask-truncated --overlong-penalty 0.5  G14: keep completions cut at the length limit in the loss and
                   subtract 0.5 from their reward (TRL masks them by default; most are proof-line loops)
  --no-proof-penalty 0.5  G16: also subtract 0.5 from answers without a <proof> block (G14 fled into informal prose,
                   which scored 0 like a failed proof but could not be truncated)
  --prompt-filter  G11: train only on prompts whose sampled reward rate (scripts/rl_prompt_filter.py) is in
                   (lo, hi), i.e. prompts whose rollout groups are likely to have non-zero advantage
Rewards: scripts/formal_rewards.py (the other components are logged with weight 0).

Prompts: the checkable pool of scripts/build_rl_gate_set.py (math, DAPO, persona
word problems, yes/no) minus the gate set's held-out rows. Each prompt is rendered
as a raw string with the SFT chat rendering (formal_chat_format.render_prompt of
"<formal>\\n{prompt}", or of the plain prompt with --no-tag), so GRPO sees exactly
the token boundaries of SFT and of the tagged eval; TRL does not re-template it.

Environment: /vol/tmp2/laitenbf/.venv_rlvl_grpo (TRL 1.14, vLLM 0.30 colocate,
transformers 5.17). Launch with accelerate (scripts/slurm/jobs/grpo_formal.slurm; on alex
scripts/slurm/jobs/alex_grpo_formal.slurm, which sets RLVL_DATA_ROOT and --stop-at before the walltime).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_rl_gate_set import OUT as GATE_DIR, classify  # noqa: E402
from formal_chat_format import render_prompt  # noqa: E402
from formal_rewards import reward_funcs  # noqa: E402

DATA_ROOT = Path(os.environ.get("RLVL_DATA_ROOT", "/vol/tmp2/laitenbf/rlvl_data"))
GEN_POOL = DATA_ROOT / "datasets/formal_mixture_20260925/pool/train.jsonl"


def build_dataset(tok, tag: bool, n: int | None, seed: int, benches: list[str], max_per_bench: int | None = None,
                  keep_ids: set | None = None, exclude_ids: set | None = None):
    from datasets import Dataset, load_dataset
    rows = []
    if "gen" in benches:  # generator items carry their gold sentences: frac_hard then checks exact faithfulness
        import re
        for i, ln in enumerate(open(GEN_POOL)):
            r = json.loads(ln)
            a = r["answer"]
            t = "yesno" if a in ("yes", "no") else "number" if re.fullmatch(r"-?\d+(/\d+)?", a) else None
            if t is None or r.get("tools"):  # no tool loop in GRPO
                continue
            rows.append({"prompt": render_prompt(tok, f"<formal>\n{r['prompt']}" if tag else r["prompt"]),
                         "raw_prompt": r["prompt"], "id": f"gen/{r['family']}/{i}", "bench": "gen", "gold": a,
                         "answer_type": t, "system_answerable": True, "sentences_json": json.dumps(r["sentences"])})
    held = set(json.loads((GATE_DIR / "heldout_rows.json").read_text()))
    ds = load_dataset("allenai/Dolci-Instruct-RL", split="train")
    for i, r in enumerate(ds):
        p = r["prompt"]
        if i in held or not p.startswith("user: ") or "\nassistant:" in p:
            continue
        c = classify(r)
        if c is None or c[0] not in benches:
            continue
        b, t, g, a = c
        q = p.removeprefix("user: ").strip()
        rows.append({"prompt": render_prompt(tok, f"<formal>\n{q}" if tag else q), "raw_prompt": q,
                     "id": f"{b}/{i}", "bench": b, "gold": g, "answer_type": t, "system_answerable": a,
                     "sentences_json": ""})
    if "gsm8k_train" in benches:  # more word problems for EI/GRPO (the Dolci wordprob pool is 1.4k); the
        import re  # gsm8k *test* split is a benchmark (build_formal_bench_tagged.py), never use it here
        for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
            g = r["answer"].split("####")[-1].strip().replace(",", "")
            if not re.fullmatch(r"-?\d+", g):
                continue
            q = r["question"].strip()
            rows.append({"prompt": render_prompt(tok, f"<formal>\n{q}" if tag else q), "raw_prompt": q,
                         "id": f"gsm8k_train/{i}", "bench": "gsm8k_train", "gold": g, "answer_type": "number",
                         "system_answerable": True, "sentences_json": ""})
    if exclude_ids:  # --exclude-ids: near-duplicates of gate items (analysis/gate_contamination.py)
        rows = [r for r in rows if r["id"] not in exclude_ids]
    if keep_ids is not None:  # --prompt-filter: only prompts whose sampled reward rate is in (lo, hi)
        rows = [r for r in rows if r["id"] in keep_ids]
    random.Random(seed).shuffle(rows)
    if max_per_bench:  # dolci_math is 96% of the pool and mostly outside the system; balance it
        seen, kept = {}, []
        for r in rows:
            seen[r["bench"]] = seen.get(r["bench"], 0) + 1
            if seen[r["bench"]] <= max_per_bench:
                kept.append(r)
        rows = kept
    if n:
        rows = rows[:n]
    return Dataset.from_list(rows)


def _ckpts(out_dir: str) -> list[Path]:
    return sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def _resumable(c: Path) -> bool:
    return (c / "trainer_state.json").is_file() and (any(c.glob("global_step*")) or (c / "optimizer.pt").is_file())


def _prune_optimizer_state(out_dir: str, keep_every: int = 0, keep_full: int = 2) -> None:
    """Long runs (--resumable) save often for resume granularity. The newest keep_full checkpoints stay complete
    (the newest may be half-written when the walltime kills a job during a save). Older ones lose the optimizer /
    DeepSpeed state (~6x the weights); their weights are kept only at multiples of keep_every (gate evals)."""
    import shutil
    for c in _ckpts(out_dir)[:-keep_full]:
        if keep_every and int(c.name.split("-")[1]) % keep_every:
            shutil.rmtree(c, ignore_errors=True)
            continue
        for f in list(c.glob("global_step*")) + [c / "optimizer.pt", c / "scheduler.pt"]:
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            elif f.exists():
                f.unlink()


def _callback_base():
    from transformers import TrainerCallback
    return TrainerCallback


class PruneOptimizerState(_callback_base()):
    def __init__(self, keep_every: int):
        self.keep_every = keep_every

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            _prune_optimizer_state(args.output_dir, self.keep_every)


class StopAtDeadline(_callback_base()):
    """Checkpoint and stop once the wall clock passes `stop_at` (unix time), e.g. before a Slurm walltime.

    All ranks take the decision together (all-reduce), so they save the same step."""

    def __init__(self, stop_at: float):
        self.stop_at = stop_at

    def on_step_end(self, args, state, control, **kwargs):
        import time

        import torch
        import torch.distributed as dist
        stop = time.time() > self.stop_at
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([float(stop)], device="cuda" if torch.cuda.is_available() else "cpu")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
        if stop:
            if state.is_world_process_zero:
                print(f"stop-at deadline reached at step {state.global_step}: saving and stopping", flush=True)
            control.should_save = True
            control.should_training_stop = True
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", choices=["correct", "correct_x_valid", "gvc", "valid", "lines", "lines_fmt", "lines_raw", "frac", "frac_hard", "cvf", "cvf_fmt"], required=True)
    ap.add_argument("--no-tag", action="store_true", help="prompt without <formal> (G0, G1-NL)")
    ap.add_argument("--benches", default="dolci_math,dolci_dapo,dolci_wordprob,dolci_yesno",
                    help="also: gen = formal_mixture generator pool train (yes/no + numeric, no tools)")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--max-per-bench", type=int, default=None,
                    help="cap prompts per subset (pool: math 43.5k, wordprob 1.2k, yesno 0.5k)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--per-device-batch", type=int, default=8, help="completions per device per micro-step")
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-mask-truncated", action="store_true",
                    help="keep completions cut at --max-completion-length in the loss (TRL masks them by default), so "
                         "repetition loops that run into the length limit get their reward-0 negative advantage")
    ap.add_argument("--overlong-penalty", type=float, default=0.0,
                    help="subtract this from the reward of completions that reach --max-completion-length "
                         "(only has a gradient with --no-mask-truncated)")
    ap.add_argument("--no-proof-penalty", type=float, default=0.0,
                    help="subtract this from the reward of completions without a <proof> block (G16)")
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--vllm-max-model-len", type=int, default=None,
                    help="cap vLLM max_model_len (default: the model config's 262144, whose KV cache needs ~3 GiB)")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--prompt-filter", default=None,
                    help="scripts/rl_prompt_filter.py output: keep prompts with --filter-lo < rate < --filter-hi")
    ap.add_argument("--filter-lo", type=float, default=0.0)
    ap.add_argument("--filter-hi", type=float, default=1.0)
    ap.add_argument("--resumable", action="store_true",
                    help="long runs across walltime limits: save optimizer state, resume from the latest checkpoint "
                         "in --out-dir, and keep optimizer state only in the newest checkpoints")
    ap.add_argument("--keep-every", type=int, default=0,
                    help="with --resumable: keep weights of older checkpoints only at multiples of this step")
    ap.add_argument("--exclude-ids", default=None,
                    help="contamination.json (scripts/analysis/gate_contamination.py): drop its pool_exclude_ids, "
                         "training prompts that near-duplicate a gate item")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="unix time: save a checkpoint and stop after the first step past it (no final/); "
                         "with --resumable the next job continues from there")
    args = ap.parse_args()
    if args.overlong_penalty and not args.no_mask_truncated:
        ap.error("--overlong-penalty needs --no-mask-truncated (masked completions get no gradient)")

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    keep = None
    if args.prompt_filter:
        rates = json.loads(Path(args.prompt_filter).read_text())["rates"]
        keep = {i for i, v in rates.items() if args.filter_lo < v["rate"] < args.filter_hi}
    exclude = set(json.loads(Path(args.exclude_ids).read_text())["pool_exclude_ids"]) if args.exclude_ids else None
    train = build_dataset(tok, not args.no_tag, args.n_prompts, args.seed, args.benches.split(","),
                          args.max_per_bench, keep, exclude)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    completions_per_step = args.prompts_per_step * args.num_generations
    grad_accum = max(1, completions_per_step // (args.per_device_batch * world))
    funcs, weights = reward_funcs(args.arm, args.overlong_penalty, args.max_completion_length, args.no_proof_penalty)
    cfg = GRPOConfig(
        output_dir=args.out_dir, learning_rate=args.lr, beta=args.beta, max_steps=args.max_steps,
        num_generations=args.num_generations, per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum, max_completion_length=args.max_completion_length,
        temperature=args.temperature, mask_truncated_completions=not args.no_mask_truncated, reward_weights=weights,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_enable_sleep_mode=True, bf16=True, gradient_checkpointing=True, logging_steps=1,
        save_steps=args.save_steps, save_only_model=not args.resumable, seed=args.seed, report_to=args.report_to,
        log_completions=True, num_completions_to_print=2, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
    )
    import collections
    print(json.dumps({"benches": collections.Counter(train["bench"])}), flush=True)
    print(json.dumps({"arm": args.arm, "tag": not args.no_tag, "n_prompts": len(train), "grad_accum": grad_accum,
                      "world": world, "rewards": [f.__name__ for f in funcs]}), flush=True)
    callbacks = [PruneOptimizerState(args.keep_every)] if args.resumable else []
    if args.stop_at:
        callbacks.append(StopAtDeadline(args.stop_at))
    resume = [c for c in _ckpts(args.out_dir) if _resumable(c)] if args.resumable else []
    if args.resumable and not resume:  # recovery: no optimizer state left, resume from weights + trainer state
        resume = [c for c in _ckpts(args.out_dir)
                  if (c / "trainer_state.json").is_file() and (c / "model.safetensors").is_file()]
    if resume:
        print(f"resuming from {resume[-1]}", flush=True)
    # Load the policy weights from the checkpoint itself: Trainer's resume loads model.safetensors with a raw
    # load_state_dict, and save_pretrained writes Qwen3.5 keys as `model.language_model.*` while the model expects
    # `model.*`, so every key "misses" and the policy silently restarts from args.model (bug found 2026-10-03).
    trainer = GRPOTrainer(model=str(resume[-1]) if resume else args.model, reward_funcs=funcs, args=cfg,
                          train_dataset=train, processing_class=tok, callbacks=callbacks or None)
    trainer.train(resume_from_checkpoint=str(resume[-1]) if resume else None)
    if args.stop_at and trainer.state.global_step < args.max_steps:  # stopped at --stop-at: next job resumes
        print(f"stopped at step {trainer.state.global_step} of {args.max_steps}; not saving final", flush=True)
    else:
        trainer.save_model(str(Path(args.out_dir) / "final"))
    if trainer.accelerator.is_main_process:
        json.dump(trainer.state.log_history, open(Path(args.out_dir) / "log_history.json", "w"))


if __name__ == "__main__":
    main()
