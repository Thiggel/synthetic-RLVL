#!/usr/bin/env python3
"""Deeper BranchProof items (d30-d45) for reinforcement-learning headroom.

The graded set stops at depth 25, where every derivation-trained model is at
or near 1.000, so it cannot serve as an RL training distribution. This builds
the same generator's items at greater depth, with a fresh seed so nothing
collides with training or with the graded evaluation set.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="+", type=int, default=[30, 35, 40, 45])
    ap.add_argument("--per-depth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", default="/vol/tmp2/laitenbf/rlvl_data/datasets/deep_branchproof_20260911")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, ".")
    from synthetic_dataset import DatasetConfig, LogicDatasetGenerator
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for depth in a.depths:
        gen = LogicDatasetGenerator(DatasetConfig(
            depth=depth, distractor_ratio=0.5, difficulty="hard_fsa_schema",
            branching_factor=4, shortcut_rate=0.0, shortcut_kind="schema",
            require_unique_solution=True, seed=a.seed))
        rows = []
        for i in range(a.per_depth):
            ex = gen.generate(i)
            lines = []
            for l in ex.premises_nl:
                s = str(l).strip()
                if ". " in s and s.split(". ", 1)[0].isdigit():
                    s = s.split(". ", 1)[1]
                lines.append(s)
            rows.append(dict(context="\n".join(lines), question=str(ex.question_nl).strip(),
                             answer=str(ex.answer).strip(), depth=depth,
                             source_id=f"bp_deep_seed{a.seed}_d{depth}_i{i}"))
        with open(out / f"branchproof_nl_d{depth}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest[depth] = dict(n=len(rows), mean_premise_lines=sum(len(r["context"].splitlines()) for r in rows) / len(rows))
        print(f"depth {depth}: {len(rows)} items, {manifest[depth]['mean_premise_lines']:.0f} premise lines")
    (out / "manifest.json").write_text(json.dumps(dict(seed=a.seed, depths=manifest), indent=2))


if __name__ == "__main__":
    main()
