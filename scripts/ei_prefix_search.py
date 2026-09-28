#!/usr/bin/env python
"""Expert-iteration collection with checker-guided prefix resampling (line-level search).

Stage-2 gate remedy, round 1 finding: whole-proof rejection sampling on the RL prompts
(rl_signal_probe.py --source train) yields ~0.2% in-system samples, far too few to
train on. Most failures are one bad line inside an otherwise checked proof (a `subst`
with one citation, `calc 40 + 25`, prose after `;`). rlvl.check reports the character
offset of the first error, and every line before it has passed the checker. So:

  round 0   K samples per prompt from the policy (as rl_signal_probe --source train)
  round r   for each unsolved prompt, take the B deepest distinct verified prefixes
            ("<proof>\\n" + the lines before the first failing line, >= 1 step) and
            sample K continuations from prompt + prefix
  keep      completions (prefix + continuation) that are in_system

Every kept proof is still a sample of the policy, conditioned on its own checked
prefix; build_ei_mixture.py turns them into SFT rows. This is also the smallest
version of the Stage-3 idea (search over proof lines with the checker as a
step-level verifier), and stats.json records the yield per round.

Output: <out-dir>/samples.jsonl (the valid completions, the rl_signal_probe schema plus
"round" and "prefix_lines"), stats.json.
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

OPEN = "<proof>\n"


def verified_prefix(prompt: str, text: str):
    """(prefix of text up to the first failing proof line, number of checked step lines) or None."""
    import rlvl
    s = text.find(OPEN)
    if s < 0:
        return None
    e = text.find("</proof>", s)
    body = text[s + len(OPEN): e if e >= 0 else len(text)]
    try:
        res = rlvl.check(prompt, body, strict=True)
    except Exception:
        return None
    bad = []
    err = res.get("first_error") or res.get("fatal")
    if isinstance(err, dict) and isinstance(err.get("pos"), int):
        bad.append(err["pos"])
    lines = res.get("lines") or []
    bad += [ln["start"] for ln in lines if not ln.get("ok")]
    if not bad:
        return None
    cut = min(bad)
    cut = body.rfind("\n", 0, cut) + 1  # start of the line holding the first error
    n_ok = sum(1 for ln in lines if ln.get("ok") and ln.get("kind") == "step" and ln["end"] <= cut)
    if n_ok < 1:
        return None
    return text[: s + len(OPEN) + cut], n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benches", default="dolci_math,dolci_wordprob,dolci_yesno")
    ap.add_argument("--per-bench", type=int, default=1500)
    ap.add_argument("--k0", type=int, default=8, help="round-0 samples per prompt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4, help="prefixes resampled per unsolved prompt and round")
    ap.add_argument("--k", type=int, default=4, help="continuations per prefix")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1, help="prompt shuffle seed (round 1 of EI used 0)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from grpo_formal import build_dataset

    keep = sorted(set(args.benches.split(",")))
    tok = AutoTokenizer.from_pretrained(args.model)
    pool = [{**r, "prompt": r["raw_prompt"]} for r in build_dataset(tok, True, None, args.seed, keep)]
    recs, seen = [], collections.Counter()
    for r in pool:
        if seen[r["bench"]] < args.per_bench:
            seen[r["bench"]] += 1
            recs.append(r)
    heads = [render_prompt(tok, f"<formal>\n{r['prompt']}") for r in recs]
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "samples.jsonl", "w")
    solved = collections.defaultdict(int)             # item index -> in-system completions
    frontier = collections.defaultdict(dict)          # item index -> {prefix: n_ok}
    tried = collections.defaultdict(set)
    stats = {"n_items": len(recs), "per_bench": dict(seen), "rounds": []}

    for rnd in range(args.rounds + 1):
        jobs = []  # (item, prefix)
        if rnd == 0:
            jobs = [(i, "") for i in range(len(recs))]
            k = args.k0
        else:
            k = args.k
            for i in range(len(recs)):
                if solved[i]:
                    continue
                cands = sorted(((n, p) for p, n in frontier[i].items() if p not in tried[i]), key=lambda x: -x[0])
                for n, p in cands[: args.beam]:
                    tried[i].add(p)
                    jobs.append((i, p))
        if not jobs:
            break
        sp = SamplingParams(n=k, temperature=args.temperature, max_tokens=args.max_new_tokens, seed=rnd)
        gens = llm.generate([heads[i] + p for i, p in jobs], sp)
        new_solved, n_valid, n_samples = collections.Counter(), 0, 0
        for (i, p), g in zip(jobs, gens):
            r = recs[i]
            for o in g.outputs:
                text = p + o.text
                c = components(r, text)
                n_samples += 1
                if c["valid"]:
                    n_valid += 1
                    fh.write(json.dumps({"id": r["id"], "bench": r["bench"], "prompt": r["prompt"], "gold": r["gold"],
                                         "temperature": args.temperature, "round": rnd,
                                         "prefix_lines": p.count("\n") - 1 if p else 0, "text": text, **c}) + "\n")
                if c["in_system"]:
                    if not solved[i]:
                        new_solved[r["bench"]] += 1
                    solved[i] += 1
                elif not solved[i] and c["has_proof"]:
                    vp = verified_prefix(r["prompt"], text)
                    if vp and len(vp[0]) > len(p):  # only strictly deeper prefixes
                        frontier[i][vp[0]] = vp[1]
        fh.flush()
        tot = collections.Counter(recs[i]["bench"] for i in range(len(recs)) if solved[i])
        row = {"round": rnd, "n_jobs": len(jobs), "n_samples": n_samples, "n_valid": n_valid,
               "new_solved": dict(new_solved), "solved": dict(tot),
               "solved_frac": {b: tot[b] / seen[b] for b in seen},
               "n_in_system": sum(solved.values()),
               "items_with_frontier": sum(1 for i in frontier if frontier[i] and not solved[i])}
        stats["rounds"].append(row)
        print(json.dumps(row), flush=True)
        json.dump(stats, open(out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
