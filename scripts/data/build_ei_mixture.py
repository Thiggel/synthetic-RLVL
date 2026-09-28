#!/usr/bin/env python
"""Expert-iteration SFT mixture: the policy's own in-system proofs on RL prompts + replay.

Stage-2 gate remedy (docs/research_plan.md, change log 2026-09-28). The formal-mixture
policies almost never write a checkable proof for the Olmo 3 RL prompts (greedy
in_system ~0.3%), so validity rewards have no signal. This builds a short SFT set
from the samples of scripts/rl_signal_probe.py --source train that are in_system
(valid, grounded, the proof's ans agrees with the Answer: line and equals the
reference), i.e. rejection sampling / STaR on the RL prompt distribution.

Rows (the schema of build_formal_mixture_sft.py):
  prompt  "<formal>\\n{RL prompt}"      target  "<proof>...</proof>\\nAnswer: x"
  source  "ei"                          family  "ei_<bench>"
At most --max-per-prompt distinct proofs per prompt (shortest first). Replay:
--replay rows of the base mixture's train split (Dolci and generator rows in their
original ratio) so the continued SFT keeps both skills. eval = the base mixture's eval.

  .venv_rlvl_vllm/bin/python scripts/data/build_ei_mixture.py --samples A/samples.jsonl [B ...] \\
      --out /vol/tmp2/laitenbf/rlvl_data/datasets/formal_ei_20260928/<name>
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

BASE = Path("/vol/tmp2/laitenbf/rlvl_data/datasets/formal_mixture_20260925/mixtures/dolci_rlvlgen_p50")
ANSWER_LINE = re.compile(r"(?m)^Answer:[ \t]*.*$")


def trim(text: str) -> str | None:
    """Proof plus the first Answer: line after it; None if either is missing."""
    s, e = text.find("<proof>\n"), text.find("</proof>")
    if s < 0 or e < 0:
        return None
    m = ANSWER_LINE.search(text, e)
    return None if m is None else text[s:e + len("</proof>")] + "\n" + m.group(0).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--max-per-prompt", type=int, default=2)
    ap.add_argument("--replay", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

    by_prompt = collections.defaultdict(dict)
    meta = {}
    for p in args.samples:
        for line in open(p):
            r = json.loads(line)
            if not r.get("in_system"):
                continue
            t = trim(r["text"])
            if t:
                by_prompt[r["id"]][t] = True
                meta[r["id"]] = (r["prompt"], r["bench"])
    rows = []
    for pid, targets in sorted(by_prompt.items()):
        prompt, bench = meta[pid]
        for t in sorted(targets, key=len)[: args.max_per_prompt]:
            rows.append({"prompt": f"<formal>\n{prompt}", "target": t, "source": "ei", "family": f"ei_{bench}",
                         "example_id": pid, "mask_tool_results": False})
    base = load_from_disk(str(args.base))
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(base["train"])), min(args.replay, len(base["train"])))
    ei = Dataset.from_list(rows).cast(base["train"].features)
    train = concatenate_datasets([ei, base["train"].select(idx)]).shuffle(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": train, "eval": base["eval"]}).save_to_disk(str(args.out))
    manifest = {"n_ei": len(rows), "n_prompts": len(by_prompt), "n_replay": len(idx), "base": str(args.base),
                "samples": args.samples, "max_per_prompt": args.max_per_prompt, "seed": args.seed,
                "ei_benches": dict(collections.Counter(r["family"] for r in rows))}
    (args.out / "mixture_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
