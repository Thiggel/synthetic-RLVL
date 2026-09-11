#!/usr/bin/env python3
"""GRPO with a program checker as the reward, LoRA, one GPU.

Three reward arms, selected with --reward:

  correct   1 if the <answer> matches gold, else 0. The standard baseline;
            needs labels.
  valid     1 if the written derivation is accepted by the forward-chaining
            checker (every proof line follows from the item's own premises and
            the conclusion is the last line), else 0. USES NO LABELS.
  both      1 only if the answer is right AND the derivation checks out.

The prompts are deep BranchProof items in the midtraining document format, the
only format in which these models write derivations. Completions are capped at
--max-completion-length and generation stops at </answer>.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lm_eval_tasks", "synthrlvl_ood"))
from check_native_derivations import check  # noqa: E402
import utils  # noqa: E402

ANS = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.DOTALL)


def answer_of(text):
    m = ANS.search(text)
    return utils.normalize_answer(m.group(1).strip().split("\n")[0]) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/vol/tmp2/laitenbf/rlvl_data/datasets/deep_branchproof_20260911")
    ap.add_argument("--depths", nargs="+", type=int, default=[30, 35])
    ap.add_argument("--reward", choices=["correct", "valid", "both"], required=True)
    ap.add_argument("--formal", action="store_true", help="completions are formal notation")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-prompt-length", type=int, default=6000)
    ap.add_argument("--max-completion-length", type=int, default=9000)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--eval-holdout", type=int, default=64)
    a = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = []
    for d in a.depths:
        for line in open(f"{a.data}/branchproof_nl_d{d}.jsonl"):
            r = json.loads(line)
            rows.append(dict(prompt=utils.doc_to_text_deduction_bp_native(r), doc=json.dumps(r)))
    rng = __import__("random").Random(a.seed)
    rng.shuffle(rows)
    hold, train = rows[: a.eval_holdout], rows[a.eval_holdout:]
    ds = Dataset.from_list(train)
    os.makedirs(a.output_dir, exist_ok=True)
    json.dump(hold, open(os.path.join(a.output_dir, "holdout.json"), "w"))

    stats = dict(n=0, correct=0.0, valid=0.0, both=0.0, has_proof=0.0)

    def reward_fn(completions, doc, **kwargs):
        out = []
        for text, dj in zip(completions, doc):
            item = json.loads(dj)
            res = check(item, text + "</answer>", a.formal)
            valid = float(res["all_valid"] > 0 and res["conclusion_matches_last"] > 0)
            corr = float(answer_of(text) == utils.normalize_answer(item["answer"]))
            stats["n"] += 1
            stats["correct"] += corr
            stats["valid"] += valid
            stats["both"] += corr * valid
            stats["has_proof"] += res["has_proof"]
            out.append({"correct": corr, "valid": valid, "both": corr * valid}[a.reward])
        if stats["n"] and stats["n"] % (a.num_generations * 16) < a.num_generations:
            n = stats["n"]
            print(f"[rollouts {n}] correct {stats['correct']/n:.3f} valid {stats['valid']/n:.3f} "
                  f"both {stats['both']/n:.3f} has_proof {stats['has_proof']/n:.3f}", flush=True)
        return out

    cfg = GRPOConfig(
        output_dir=a.output_dir,
        learning_rate=a.lr,
        per_device_train_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations,
        max_prompt_length=a.max_prompt_length,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        beta=a.beta,
        max_steps=a.max_steps,
        logging_steps=1,
        save_steps=max(50, a.max_steps // 3),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        report_to="none",
        seed=a.seed,
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"],
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(os.path.join(a.output_dir, "final"))
    n = max(1, stats["n"])
    json.dump({k: (v / n if k != "n" else v) for k, v in stats.items()},
              open(os.path.join(a.output_dir, "rollout_stats.json"), "w"), indent=2)
    print("done", a.reward, stats["n"], "rollouts")


if __name__ == "__main__":
    main()
