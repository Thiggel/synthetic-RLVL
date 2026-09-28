#!/usr/bin/env python
"""Stage-2 gate, sampling version: does a GRPO group see any reward signal?

The greedy gate (eval_formal_bench_vllm on the rl_gate set) scores one output per
prompt; GRPO learns only from groups whose rewards differ. For every gate item this
samples K completions per temperature from the policy (prompt rendered exactly as
in scripts/grpo_formal.py), scores each with scripts/formal_rewards.components and
reports per bench and temperature:
  mean_<c>       mean component over all samples
  pass_<c>       fraction of items with >= 1 sample where c = 1   (pass@K)
  signal_<arm>   fraction of items whose K rewards under that arm are not all equal
                 (a group with nonzero advantage; GRPO's effective batch fraction)
Output: <out-dir>/probe.json and samples.jsonl (the valid samples are the seed
material for a rejection-sampling bootstrap).

--source train samples the Stage-2 RL training prompts instead (grpo_formal.build_dataset,
gate rows excluded, --per-bench prompts per bench): the expert-iteration collection step
(scripts/data/build_ei_mixture.py turns the in-system samples into SFT rows).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from formal_chat_format import render_prompt  # noqa: E402
from formal_rewards import components  # noqa: E402

ARMS = {"correct": lambda c: c["correct"], "correct_x_valid": lambda c: c["correct"] * c["valid"],
        "gvc": lambda c: (c["grammatical"] + c["valid"] + c["correct"]) / 3, "valid": lambda c: c["valid"]}
COMP = ["has_proof", "grammatical", "valid", "correct", "in_system"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-jsonl", default="/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl")
    ap.add_argument("--benches", default="dolci_math,dolci_dapo,dolci_wordprob,dolci_yesno")
    ap.add_argument("--per-bench", type=int, default=100)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperatures", default="0.6,1.0")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-tag", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--source", choices=["gate", "train"], default="gate")
    ap.add_argument("--seed", type=int, default=0, help="prompt shuffle seed for --source train")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    keep = set(args.benches.split(","))
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.source == "train":
        from grpo_formal import build_dataset
        pool = [{**r, "prompt": r["raw_prompt"]} for r in build_dataset(tok, True, None, args.seed, sorted(keep))]
    else:
        pool = [json.loads(line) for line in open(args.test_jsonl)]
    recs, seen = [], collections.Counter()
    for r in pool:
        if r["bench"] in keep and seen[r["bench"]] < args.per_bench:
            seen[r["bench"]] += 1
            recs.append(r)
    prompts = [render_prompt(tok, r["prompt"] if args.no_tag else f"<formal>\n{r['prompt']}") for r in recs]
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result, fh = {}, open(out / "samples.jsonl", "w")
    for t in map(float, args.temperatures.split(",")):
        sp = SamplingParams(n=args.k, temperature=t, max_tokens=args.max_new_tokens, seed=0)
        gens = llm.generate(prompts, sp)
        per = collections.defaultdict(list)
        for r, g in zip(recs, gens):
            cs = [components(r, o.text) for o in g.outputs]
            per[r["bench"]].append(cs)
            per["all"].append(cs)
            for o, c in zip(g.outputs, cs):
                if c["valid"]:
                    fh.write(json.dumps({"id": r["id"], "bench": r["bench"], "prompt": r["prompt"], "gold": r["gold"],
                                         "temperature": t, "text": o.text, **c}) + "\n")
        res = {}
        for b, groups in per.items():
            d = {"n_items": len(groups)}
            for c in COMP:
                d[f"mean_{c}"] = sum(x[c] for cs in groups for x in cs) / sum(len(cs) for cs in groups)
                d[f"pass_{c}"] = sum(any(x[c] for x in cs) for cs in groups) / len(groups)
            for a, f in ARMS.items():
                d[f"signal_{a}"] = sum(len({f(x) for x in cs}) > 1 for cs in groups) / len(groups)
            res[b] = d
        result[str(t)] = res
        print(json.dumps({"T": t, "all": res["all"]}), flush=True)
    json.dump(result, open(out / "probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
