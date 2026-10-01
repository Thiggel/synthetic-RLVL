#!/usr/bin/env python3
"""Near-duplicates between the Dolci gate (950 held-out prompts) and the training-prompt pool (2026-10-01).

The gate holds out Dolci rows by index (build_rl_gate_set.py), but Dolci-Instruct-RL repeats problems under
other row indices (reworded, re-spaced, with an image link, ...), and GSM8K train shares problems with Dolci.
Pool = every Dolci train row that classify() accepts (all gate benches, held-out rows excluded) + GSM8K train,
i.e. everything grpo_formal.build_dataset and the EI harvest can draw from.
Match on word 12-grams of the normalised text (lowercase alphanumerics), ignoring template 12-grams (in more
than --template-df prompts: instruction boilerplate). coverage(gate item, pool row) = share of the gate item's
non-template 12-grams that occur in the pool row; an item is contaminated if some pool row covers >= --threshold.
Writes <gate dir>/contamination.json: per-item best coverage, clean_ids (the gate minus contaminated items) and
pool_exclude_ids (pool rows covering a gate item >= threshold; grpo_formal.py --exclude-ids drops them), plus
analysis/gate_contamination.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from build_rl_gate_set import OUT as GATE_DIR, classify  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def shingles(s: str, n: int = 12) -> set[str]:
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--template-df", type=int, default=50)
    args = ap.parse_args()
    from datasets import load_dataset

    gate = [json.loads(l) for l in open(GATE_DIR / "test.jsonl")]
    held = set(json.loads((GATE_DIR / "heldout_rows.json").read_text()))
    pool: list[tuple[str, set[str]]] = []
    for i, r in enumerate(load_dataset("allenai/Dolci-Instruct-RL", split="train")):
        p = r["prompt"]
        if i in held or not p.startswith("user: ") or "\nassistant:" in p:
            continue
        c = classify(r)
        if c is not None:
            pool.append((f"{c[0]}/{i}", shingles(p.removeprefix("user: "))))
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        pool.append((f"gsm8k_train/{i}", shingles(r["question"])))
    gsh = {g["id"]: shingles(g["prompt"]) for g in gate}
    df = collections.Counter(x for _, s in pool for x in s)
    df.update(x for s in gsh.values() for x in s)
    gsh = {k: {x for x in s if df[x] <= args.template_df} for k, s in gsh.items()}
    index: dict[str, set[str]] = collections.defaultdict(set)
    for k, s in gsh.items():
        for x in s:
            index[x].add(k)
    best: dict[str, float] = {k: 0.0 for k in gsh}
    best_row: dict[str, str] = {}
    exclude = set()
    for pid, s in pool:
        hits = collections.Counter(k for x in s if x in index for k in index[x])
        for k, n in hits.items():
            cov = n / len(gsh[k])
            if cov > best[k]:
                best[k], best_row[k] = cov, pid
            if cov >= args.threshold:
                exclude.add(pid)
    dirty = {k for k, v in best.items() if v >= args.threshold}
    out = {"threshold": args.threshold, "template_df": args.template_df, "n_gate": len(gate), "n_pool": len(pool),
           "best_coverage": {k: round(v, 4) for k, v in best.items()}, "best_row": best_row,
           "clean_ids": sorted(set(gsh) - dirty), "pool_exclude_ids": sorted(exclude)}
    (GATE_DIR / "contamination.json").write_text(json.dumps(out, indent=0) + "\n")

    benches = sorted({g["bench"] for g in gate})
    n = collections.Counter(g["bench"] for g in gate)
    lines = ["# Gate contamination: near-duplicates of gate prompts in the training-prompt pool", "",
             f"Pool: {len(pool)} prompts (Dolci train rows of all gate benches, held-out rows excluded, + GSM8K "
             f"train). 12-gram coverage, template 12-grams (df > {args.template_df}) ignored "
             "(scripts/analysis/gate_contamination.py).", "",
             "| gate bench | items | best coverage >= .3 | >= .6 | >= .9 | clean (< "
             f"{args.threshold}) |", "|---|---:|---:|---:|---:|---:|"]
    for b in benches + ["all"]:
        ks = [g["id"] for g in gate if b == "all" or g["bench"] == b]
        c = [sum(best[k] >= t for k in ks) for t in (.3, .6, .9)]
        lines.append(f"| {b} | {len(ks)} | {c[0]} | {c[1]} | {c[2]} | {sum(best[k] < args.threshold for k in ks)} |")
    src = collections.Counter(best_row[k].split("/")[0] for k in dirty)
    lines += ["", f"Contaminated items by source of the closest pool row: {dict(src)}. Pool rows to exclude from "
              f"training prompts: {len(exclude)} "
              f"({dict(collections.Counter(x.split('/')[0] for x in exclude))})."]
    (REPO / "analysis/gate_contamination.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), dict(n))


if __name__ == "__main__":
    main()
