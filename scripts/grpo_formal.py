#!/usr/bin/env python
"""Stage-2 GRPO on the Olmo 3 RL prompts (allenai/Dolci-Instruct-RL) with checker rewards.

docs/research_plan.md, Stage 2. One run = one arm:
  --arm correct          G0 (--no-tag, the X=0 policy) / G1 (tagged, the best-X policy)
  --arm correct_x_valid  G2     --arm gvc  G3     --arm valid  G4
Rewards: scripts/formal_rewards.py (the other components are logged with weight 0).

Prompts: the checkable pool of scripts/build_rl_gate_set.py (math, DAPO, persona
word problems, yes/no) minus the gate set's held-out rows. Each prompt is rendered
as a raw string with the SFT chat rendering (formal_chat_format.render_prompt of
"<formal>\\n{prompt}", or of the plain prompt with --no-tag), so GRPO sees exactly
the token boundaries of SFT and of the tagged eval; TRL does not re-template it.

Environment: /vol/tmp2/laitenbf/.venv_rlvl_grpo (TRL 1.14, vLLM 0.30 colocate,
transformers 5.17). Launch with accelerate (scripts/slurm/jobs/grpo_formal.slurm).
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


def build_dataset(tok, tag: bool, n: int | None, seed: int, benches: list[str]):
    from datasets import Dataset, load_dataset
    held = set(json.loads((GATE_DIR / "heldout_rows.json").read_text()))
    ds = load_dataset("allenai/Dolci-Instruct-RL", split="train")
    rows = []
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
                     "id": f"{b}/{i}", "bench": b, "gold": g, "answer_type": t, "system_answerable": a})
    random.Random(seed).shuffle(rows)
    if n:
        rows = rows[:n]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", choices=["correct", "correct_x_valid", "gvc", "valid"], required=True)
    ap.add_argument("--no-tag", action="store_true", help="prompt without <formal> (G0, G1-NL)")
    ap.add_argument("--benches", default="dolci_math,dolci_dapo,dolci_wordprob,dolci_yesno")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--per-device-batch", type=int, default=8, help="completions per device per micro-step")
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    train = build_dataset(tok, not args.no_tag, args.n_prompts, args.seed, args.benches.split(","))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    completions_per_step = args.prompts_per_step * args.num_generations
    grad_accum = max(1, completions_per_step // (args.per_device_batch * world))
    funcs, weights = reward_funcs(args.arm)
    cfg = GRPOConfig(
        output_dir=args.out_dir, learning_rate=args.lr, beta=args.beta, max_steps=args.max_steps,
        num_generations=args.num_generations, per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum, max_completion_length=args.max_completion_length,
        temperature=args.temperature, mask_truncated_completions=True, reward_weights=weights,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_enable_sleep_mode=True, bf16=True, gradient_checkpointing=True, logging_steps=1,
        save_steps=args.save_steps, save_only_model=True, seed=args.seed, report_to=args.report_to,
        log_completions=True, num_completions_to_print=2, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
    )
    print(json.dumps({"arm": args.arm, "tag": not args.no_tag, "n_prompts": len(train), "grad_accum": grad_accum,
                      "world": world, "rewards": [f.__name__ for f in funcs]}), flush=True)
    trainer = GRPOTrainer(model=args.model, reward_funcs=funcs, args=cfg, train_dataset=train,
                          processing_class=tok)
    trainer.train()
    trainer.save_model(str(Path(args.out_dir) / "final"))
    json.dump(trainer.state.log_history, open(Path(args.out_dir) / "log_history.json", "w"))


if __name__ == "__main__":
    main()
