#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic_dataset import DatasetConfig, LogicDatasetGenerator


def evaluate_split(*, depths: list[int], n_per_depth: int, shortcut_rate: float, seed: int, branching_factor: int) -> dict:
    rows = []
    failures = []
    for depth in depths:
        gen = LogicDatasetGenerator(
            DatasetConfig(
                depth=depth,
                difficulty="hard_fsa_schema",
                branching_factor=branching_factor,
                shortcut_rate=shortcut_rate,
                seed=seed,
            )
        )
        for idx in range(n_per_depth):
            try:
                ex = gen.generate(idx)
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append({"depth": depth, "idx": idx, "error": repr(exc)})
                continue
            meta = ex.metadata
            cands = list(meta.get("candidate_answers", []))
            rows.append(
                {
                    "depth": depth,
                    "answer": ex.answer,
                    "candidate_answers": cands,
                    "gold_candidate_position": int(meta.get("gold_candidate_position", -1)),
                    "schema_prediction_correct": bool(meta.get("schema_prediction_correct", False)),
                    "shortcut_enabled": bool(meta.get("shortcut_enabled", False)),
                    "first_candidate_correct": bool(cands and cands[0] == ex.answer),
                    "last_candidate_correct": bool(cands and cands[-1] == ex.answer),
                    "alphabetic_first_correct": bool(cands and sorted(cands)[0] == ex.answer),
                    "alphabetic_last_correct": bool(cands and sorted(cands)[-1] == ex.answer),
                }
            )
    total = max(1, len(rows))
    pos = Counter(r["gold_candidate_position"] for r in rows)
    metrics = {
        "num_rows": len(rows),
        "num_failures": len(failures),
        "failure_examples": failures[:10],
        "shortcut_enabled_rate": mean([float(r["shortcut_enabled"]) for r in rows]) if rows else 0.0,
        "schema_predictor_accuracy": mean([float(r["schema_prediction_correct"]) for r in rows]) if rows else 0.0,
        "first_candidate_accuracy": mean([float(r["first_candidate_correct"]) for r in rows]) if rows else 0.0,
        "last_candidate_accuracy": mean([float(r["last_candidate_correct"]) for r in rows]) if rows else 0.0,
        "alphabetic_first_accuracy": mean([float(r["alphabetic_first_correct"]) for r in rows]) if rows else 0.0,
        "alphabetic_last_accuracy": mean([float(r["alphabetic_last_correct"]) for r in rows]) if rows else 0.0,
        "gold_candidate_position_counts": dict(sorted(pos.items())),
        "gold_candidate_position_rates": {str(k): v / total for k, v in sorted(pos.items())},
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shortcut-rate", type=float, default=0.8)
    parser.add_argument("--eval-shortcut-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--n-per-depth", type=int, default=200)
    parser.add_argument("--train-depth-max", type=int, default=15)
    parser.add_argument("--eval-depth-max", type=int, default=20)
    parser.add_argument("--output", default="tmp/hard_fsa_schema_probe.json")
    args = parser.parse_args()

    train = evaluate_split(
        depths=list(range(1, args.train_depth_max + 1)),
        n_per_depth=args.n_per_depth,
        shortcut_rate=args.train_shortcut_rate,
        seed=args.seed,
        branching_factor=args.branching_factor,
    )
    eval_ = evaluate_split(
        depths=list(range(1, args.eval_depth_max + 1)),
        n_per_depth=args.n_per_depth,
        shortcut_rate=args.eval_shortcut_rate,
        seed=args.seed + 1_000_000,
        branching_factor=args.branching_factor,
    )
    chance = 1.0 / args.branching_factor
    tolerance = max(0.05, 3.0 / max(1, args.n_per_depth * args.eval_depth_max) ** 0.5)
    report = {
        "chance": chance,
        "train_shortcut_rate": args.train_shortcut_rate,
        "eval_shortcut_rate": args.eval_shortcut_rate,
        "train": train,
        "eval": eval_,
        "accepted": (
            train["num_failures"] == 0
            and eval_["num_failures"] == 0
            and abs(train["shortcut_enabled_rate"] - args.train_shortcut_rate) <= max(0.05, tolerance)
            and abs(train["schema_predictor_accuracy"] - args.train_shortcut_rate) <= max(0.05, tolerance)
            and eval_["schema_predictor_accuracy"] <= 0.05
            and abs(eval_["first_candidate_accuracy"] - chance) <= tolerance
            and abs(eval_["last_candidate_accuracy"] - chance) <= tolerance
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
