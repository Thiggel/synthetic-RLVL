#!/usr/bin/env python
"""Per-prompt GRPO reward rate on the Stage-2 training pool, for a prompt-filtered GRPO run (2026-09-30).

Why: at 8 rollouts, 86% of G8's groups had zero reward std (no gradient), and on the Dolci gate the cvf
signal rate (mixed@8) is ~5% even after GRPO (reports/2026-09-28_stage1_interim.md, pass@k section).
Most rollouts go to prompts the policy never solves (dolci_math, DAPO) or always solves. This samples
--n completions per training prompt at the GRPO temperature, scores them with the GRPO reward
(scripts/formal_rewards.py), and writes the rate per prompt id. grpo_formal.py --prompt-filter then keeps
the prompts with lo < rate < hi.

Prompts are exactly grpo_formal.build_dataset (same rendering, gate held-out rows excluded).
Output: JSON {"model", "arm", "n", "temperature", "rates": {id: {"bench", "rate", "k"}}}, and
<out>.passing.jsonl: every completion with reward > 0 (id, bench, raw_prompt, completion), the harvest for
self-distillation (expert iteration) on real prompts.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from formal_rewards import make_reward  # noqa: E402
from grpo_formal import build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="cvf")
    ap.add_argument("--benches", default="gen,dolci_math,dolci_wordprob,dolci_yesno")
    ap.add_argument("--max-per-bench", type=int, default=3000)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--exclude-ids", default=None,
                    help="contamination.json (scripts/analysis/gate_contamination.py): drop its pool_exclude_ids")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    exclude = set(json.loads(Path(args.exclude_ids).read_text())["pool_exclude_ids"]) if args.exclude_ids else None
    ds = build_dataset(tok, True, None, args.seed, args.benches.split(","), args.max_per_bench, exclude_ids=exclude)
    print(json.dumps({"benches": collections.Counter(ds["bench"]), "n": args.n}), flush=True)
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_completion_length,
                        seed=args.seed)
    t0 = time.time()
    outs = llm.generate(ds["prompt"], sp)
    print(f"generated in {time.time() - t0:.0f}s", flush=True)
    reward = make_reward(args.arm)
    cols = ["id", "gold", "answer_type", "system_answerable", "raw_prompt", "sentences_json"]
    rates = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    harvest = open(str(args.out) + ".passing.jsonl", "w")
    for i, o in enumerate(outs):
        # truncated completions are masked in GRPO (mask_truncated_completions); count them as 0 here
        comps = [c.text if c.finish_reason != "length" else "" for c in o.outputs]
        kw = {c: [ds[i][c]] * len(comps) for c in cols}
        r = reward([ds[i]["prompt"]] * len(comps), comps, **kw)
        rates[ds[i]["id"]] = {"bench": ds[i]["bench"], "rate": sum(r) / len(r), "k": sum(x > 0 for x in r)}
        for c, x in zip(comps, r):
            if x > 0:
                harvest.write(json.dumps({"id": ds[i]["id"], "bench": ds[i]["bench"], "raw_prompt": ds[i]["raw_prompt"],
                                          "gold": ds[i]["gold"], "reward": x, "completion": c}) + "\n")
    harvest.close()
    summ = collections.defaultdict(collections.Counter)
    for v in rates.values():
        summ[v["bench"]]["all"] += 1
        summ[v["bench"]]["mixed" if 0 < v["k"] < args.n else "zero" if v["k"] == 0 else "full"] += 1
    print(json.dumps(summ), flush=True)
    args.out.write_text(json.dumps({"model": args.model, "arm": args.arm, "n": args.n, "exclude_ids": args.exclude_ids,
                                    "temperature": args.temperature, "summary": summ, "rates": rates}) + "\n")


if __name__ == "__main__":
    main()
