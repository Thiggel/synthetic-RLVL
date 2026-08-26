#!/usr/bin/env python
"""Build the graded-depth natural-language deduction eval (2026-08-26).

Tier A: ProofWriter OWA meta-test questions bucketed by QDep in {0,1,2,3,5}
        (raw test splits were deliberately reserved for eval by
        scripts/data/real_logic_corpus/build_real_logic_corpus.py).
Tier B: freshly generated deep BranchProof (hard_fsa_schema, unique answer)
        prose QA at depths {5,10,15,20,25}, rendered with the nl_exact
        premise text only (no proof, no training envelope).

Writes one jsonl per depth bucket with fields
  {context, question, answer, depth}
and prints per-depth counts plus prompt token-length stats (p50/p99/max)
under the Qwen2.5 tokenizer, using the exact doc_to_text functions from
lm_eval_tasks/synthrlvl_ood/utils.py, to guard against silent truncation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PW_DEPTH_DIRS = (0, 1, 2, 3, 5)
PW_KEEP_QDEPS = (0, 1, 2, 3, 5)
BP_DEPTHS = (5, 10, 15, 20, 25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pw-raw-root",
        default="/home/atuin/c107fa/c107fa12/synthetic-RLVL/data/raw/proofwriter/proofwriter-dataset-V2020.12.3",
    )
    p.add_argument(
        "--out-root",
        default="/home/vault/c107fa/c107fa12/synthetic-RLVL/datasets/graded_deduction_eval_20260826",
    )
    p.add_argument(
        "--tokenizer",
        default="/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_control_seed3407/final",
    )
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--pw-cap", type=int, default=500)
    p.add_argument("--bp-per-depth", type=int, default=200)
    return p.parse_args()


def _load_ood_utils():
    path = REPO_ROOT / "lm_eval_tasks" / "synthrlvl_ood" / "utils.py"
    spec = importlib.util.spec_from_file_location("synthrlvl_ood_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_proofwriter(raw_root: Path, seed: int, cap: int) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {d: [] for d in PW_KEEP_QDEPS}
    for ddir in PW_DEPTH_DIRS:
        path = raw_root / "OWA" / f"depth-{ddir}" / "meta-test.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                theory = str(rec.get("theory", "")).strip()
                if not theory:
                    continue
                for qkey, q in sorted(rec.get("questions", {}).items()):
                    qdep = q.get("QDep")
                    if qdep is None or int(qdep) not in buckets:
                        continue
                    answer = q.get("answer")
                    if isinstance(answer, bool):
                        gold = "true" if answer else "false"
                    else:
                        gold = str(answer).strip().lower()
                    if gold not in ("true", "false", "unknown"):
                        continue
                    buckets[int(qdep)].append(
                        {
                            "context": theory,
                            "question": str(q.get("question", "")).strip(),
                            "answer": gold,
                            "depth": int(qdep),
                            "source_id": f"{rec.get('id')}::{qkey}",
                            "source_dir": f"depth-{ddir}",
                        }
                    )
    out: dict[int, list[dict]] = {}
    for depth, rows in buckets.items():
        rng = random.Random(seed + depth)
        rng.shuffle(rows)
        out[depth] = rows[:cap]
    return out


def build_branchproof(seed: int, per_depth: int) -> dict[int, list[dict]]:
    from synthetic_dataset import DatasetConfig, LogicDatasetGenerator

    def unnumber(lines: list[str]) -> list[str]:
        out = []
        for line in lines:
            stripped = str(line).strip()
            if ". " in stripped and stripped.split(". ", 1)[0].isdigit():
                stripped = stripped.split(". ", 1)[1]
            out.append(stripped)
        return out

    result: dict[int, list[dict]] = {}
    for depth in BP_DEPTHS:
        # Mirror branchproof_unique_v2_20260710 generation settings
        # (manifest: hard_fsa_schema, distractor 0.5, branching 4,
        # shortcut_rate 0.0, require_unique_solution) with a NEW seed so
        # nothing collides with any training-era dataset (those used 3407).
        gen = LogicDatasetGenerator(
            DatasetConfig(
                depth=depth,
                distractor_ratio=0.5,
                difficulty="hard_fsa_schema",
                branching_factor=4,
                shortcut_rate=0.0,
                shortcut_kind="schema",
                require_unique_solution=True,
                seed=seed,
            )
        )
        rows = []
        for index in range(per_depth):
            ex = gen.generate(index)
            context = "\n".join(unnumber(ex.premises_nl))
            rows.append(
                {
                    "context": context,
                    "question": str(ex.question_nl).strip(),
                    "answer": str(ex.answer).strip(),
                    "depth": int(depth),
                    "source_id": f"bp_nl_exact_seed{seed}_d{depth}_i{index}",
                }
            )
        result[depth] = rows
    return result


def token_stats(prompts: list[str], tokenizer) -> dict[str, int]:
    lens = sorted(len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts)
    if not lens:
        return {"n": 0, "p50": 0, "p99": 0, "max": 0}
    def pct(q: float) -> int:
        return lens[min(len(lens) - 1, int(round(q * (len(lens) - 1))))]
    return {"n": len(lens), "p50": pct(0.50), "p99": pct(0.99), "max": lens[-1]}


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    utils = _load_ood_utils()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    manifest: dict = {
        "seed": args.seed,
        "pw_raw_root": str(args.pw_raw_root),
        "tokenizer": str(args.tokenizer),
        "pw_cap": args.pw_cap,
        "bp_per_depth": args.bp_per_depth,
        "bp_generator": {
            "difficulty": "hard_fsa_schema",
            "distractor_ratio": 0.5,
            "branching_factor": 4,
            "shortcut_rate": 0.0,
            "require_unique_solution": True,
            "seed": args.seed,
        },
        "files": {},
    }

    def write_bucket(name: str, rows: list[dict], doc_to_text) -> None:
        path = out_root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats = token_stats([doc_to_text(r) for r in rows], tokenizer)
        manifest["files"][name] = {"path": str(path), **stats}
        print(f"{name}: n={stats['n']} prompt_tokens p50={stats['p50']} p99={stats['p99']} max={stats['max']}")

    pw = build_proofwriter(Path(args.pw_raw_root), args.seed, args.pw_cap)
    for depth in PW_KEEP_QDEPS:
        write_bucket(f"proofwriter_owa_d{depth}", pw[depth], utils.doc_to_text_deduction_pw)

    bp = build_branchproof(args.seed, args.bp_per_depth)
    for depth in BP_DEPTHS:
        write_bucket(f"branchproof_nl_d{depth}", bp[depth], utils.doc_to_text_deduction_bp)

    max_prompt = max(v["max"] for v in manifest["files"].values())
    budget_note = f"max prompt {max_prompt} + max_gen_toks 64 vs 8192 context"
    manifest["max_prompt_tokens"] = max_prompt
    manifest["budget_note"] = budget_note
    print(budget_note)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
