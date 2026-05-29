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


def evaluate_split(
    *,
    depths: list[int],
    n_per_depth: int,
    shortcut_rate: float,
    shortcut_kind: str,
    seed: int,
    branching_factor: int,
) -> dict:
    rows = []
    failures = []
    for depth in depths:
        gen = LogicDatasetGenerator(
            DatasetConfig(
                depth=depth,
                difficulty="hard_fsa_schema",
                branching_factor=branching_factor,
                shortcut_rate=shortcut_rate,
                shortcut_kind=shortcut_kind,
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
            branch_orders = list(meta.get("branch_orders", []))
            path_markers = list(meta.get("path_markers", []))
            first_gold_branch = bool(branch_orders and all(str(order[0]).startswith("branch0:") for order in branch_orders if order))
            initial_marker_correct = bool(path_markers and str(path_markers[0]) == "north")
            rows.append(
                {
                    "depth": depth,
                    "answer": ex.answer,
                    "candidate_answers": cands,
                    "gold_candidate_position": int(meta.get("gold_candidate_position", -1)),
                    "schema_prediction_correct": bool(meta.get("schema_prediction_correct", False)),
                    "shortcut_enabled": bool(meta.get("shortcut_enabled", False)),
                    "first_gold_branch": first_gold_branch,
                    "initial_marker_correct": initial_marker_correct,
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
        "position_predictor_accuracy": mean([float(r["first_gold_branch"]) for r in rows]) if rows else 0.0,
        "initial_marker_predictor_accuracy": mean([float(r["initial_marker_correct"]) for r in rows]) if rows else 0.0,
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
    parser.add_argument("--shortcut-kind", choices=["schema", "position", "initial_marker"], default="schema")
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
        shortcut_kind=args.shortcut_kind,
        seed=args.seed,
        branching_factor=args.branching_factor,
    )
    eval_ = evaluate_split(
        depths=list(range(1, args.eval_depth_max + 1)),
        n_per_depth=args.n_per_depth,
        shortcut_rate=args.eval_shortcut_rate,
        shortcut_kind=args.shortcut_kind,
        seed=args.seed + 1_000_000,
        branching_factor=args.branching_factor,
    )
    chance = 1.0 / args.branching_factor
    tolerance = max(0.05, 3.0 / max(1, args.n_per_depth * args.eval_depth_max) ** 0.5)
    train_shortcut_expected = args.train_shortcut_rate
    eval_shortcut_expected = 0.0
    predictor_key = "schema_predictor_accuracy"
    train_predictor_expected = args.train_shortcut_rate
    eval_predictor_expected = 0.0
    if args.shortcut_kind in {"position", "initial_marker"}:
        predictor_key = "position_predictor_accuracy" if args.shortcut_kind == "position" else "initial_marker_predictor_accuracy"
        if args.shortcut_kind == "position":
            train_predictor_expected = args.train_shortcut_rate
            eval_predictor_expected = 0.0
        else:
            train_predictor_expected = args.train_shortcut_rate + (1.0 - args.train_shortcut_rate) * chance
            eval_predictor_expected = chance
    report = {
        "chance": chance,
        "shortcut_kind": args.shortcut_kind,
        "train_shortcut_rate": args.train_shortcut_rate,
        "eval_shortcut_rate": args.eval_shortcut_rate,
        "train": train,
        "eval": eval_,
        "accepted": (
            train["num_failures"] == 0
            and eval_["num_failures"] == 0
            and abs(train["shortcut_enabled_rate"] - train_shortcut_expected) <= max(0.05, tolerance)
            and abs(eval_["shortcut_enabled_rate"] - eval_shortcut_expected) <= max(0.05, tolerance)
            and abs(train[predictor_key] - train_predictor_expected) <= max(0.05, tolerance)
            and (
                eval_[predictor_key] <= max(0.05, tolerance)
                if args.shortcut_kind == "position"
                else abs(eval_[predictor_key] - eval_predictor_expected) <= max(0.05, tolerance)
            )
            and (
                True
                if args.shortcut_kind != "schema"
                else abs(eval_["first_candidate_accuracy"] - chance) <= tolerance
                and abs(eval_["last_candidate_accuracy"] - chance) <= tolerance
            )
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
