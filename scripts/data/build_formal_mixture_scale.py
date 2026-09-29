#!/usr/bin/env python3
"""Build a SCALED p50 Dolci + rlvlgen mixture: the p50 recipe at SCALE x the rows (2026-09-29).

Question (user, 2026-09-29): is the 100k-row, one-epoch SFT too short? The mixture sweep says the
2B in-domain valid rate is still rising ~+0.2 per doubling of synthetic rows at p50 (5k .12, 10k .25,
25k .47, 50k .675), and the model cites lib lemmas that do not exist. This mixture keeps everything
of dolci_rlvlgen_p50 except the size:
  synthetic  rlvlgen train items 0 .. N-1 (the first 50k are the p50 pool, byte-identical), minus any
             prompt of pool/test.jsonl (the held-out in-domain eval)
  Dolci      the prepared 100k subset (the rows p50 used), then further single-turn rows of the SAME
             seed-3407 shuffle of Dolci-Instruct-SFT-No-Tools, taken from index 306144 on: past the
             candidate window from which the 100k train rows [0, 200k) and the eval rows [200k, 306144)
             were drawn, minus rows whose prompt text repeats an eval or earlier train prompt
  eval       the unchanged Dolci-only 2048-row eval split
Writes <out-root>/pool/train_<N>.jsonl and <out-root>/mixtures/dolci_rlvlgen_p50_x<SCALE>.
Run with .venv_rlvl_grpo; HF offline, Dolci from the local cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from build_formal_mixture_sft import load_synth, sha256  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data/datasets")
BASE = DATA / "formal_mixture_20260925"
DOLCI = DATA / "dolci_no_tools_single_turn_100k_seed3407_20260803"
DOLCI_HF = "allenai/Dolci-Instruct-SFT-No-Tools"
DOLCI_REV = "9156a5a542b2503100d4f1fabbf50be5eb25d977"
CANDIDATE_END = 306144  # train_instruction_sft.py: min(len, 3 * (100000 + 2048)) rows of the seed-3407 shuffle
RLVLGEN = "/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python"


def gen_pool(out: Path, n: int) -> Path:
    """rlvlgen train items 0..n-1 without the test prompts; the p50 pool must be its prefix."""
    path = out / "pool" / f"train_{n}.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": RLVLGEN}
        subprocess.run([sys.executable, "-m", "rlvlgen.dataset", "--split", "train", "--n", str(n),
                        "--out", str(path) + ".tmp", "--exclude", str(BASE / "pool/test.jsonl")], env=env, check=True)
        os.rename(str(path) + ".tmp", path)
    base = open(BASE / "pool/train.jsonl").read().splitlines()
    with open(path) as f:
        head = [next(f).rstrip("\n") for _ in range(len(base))]
    if head != base:
        raise SystemExit(f"{path}: the first {len(base)} rows differ from the p50 pool")
    return path


def sft_formatter():
    """train_instruction_sft._row_to_prompt_target without importing the training stack (wandb, hydra)."""
    import ast
    from typing import Any
    src = (HERE.parent / "train_instruction_sft.py").read_text()
    keep = [n for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in ("_first_user_assistant", "_row_to_prompt_target")]
    ns = {"Any": Any}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "train_instruction_sft.py", "exec"), ns)
    return ns["_row_to_prompt_target"]


def dolci_rows(n: int) -> list[dict]:
    from datasets import load_dataset, load_from_disk
    _row_to_prompt_target = sft_formatter()
    prep = load_from_disk(str(DOLCI))
    rows = [{"prompt": p, "target": t, "source": "dolci", "family": "", "example_id": f"dolci-train-{i}",
             "mask_tool_results": False} for i, (p, t) in enumerate(zip(prep["train"]["prompt"], prep["train"]["target"]))]
    if n > len(rows):
        # Dolci repeats prompts across rows: drop new rows whose prompt is an eval or earlier train prompt
        taken = set(prep["eval"]["prompt"]) | {r["prompt"] for r in rows}
        raw = load_dataset(DOLCI_HF, split="train", revision=DOLCI_REV).shuffle(seed=3407)
        for j in range(CANDIDATE_END, len(raw)):
            m = raw[j].get("messages")
            if not (isinstance(m, list) and len(m) == 2 and str(m[0].get("role", "")).lower() == "user"
                    and str(m[1].get("role", "")).lower() == "assistant"):
                continue
            f = _row_to_prompt_target(dict(raw[j]), wrap_question_tags=False, wrap_answer_tags=False)
            if f is not None and f["prompt"] not in taken:
                taken.add(f["prompt"])
                rows.append({**f, "source": "dolci", "family": "", "example_id": f"dolci-ext-{j}",
                             "mask_tool_results": False})
                if len(rows) >= n:
                    break
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} Dolci rows < {n}")
    return rows[:n], prep["eval"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=DATA / "formal_mixture_scale_20260929")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    from datasets import Dataset, DatasetDict
    half = 50000 * args.scale
    out_dir = args.out_root / "mixtures" / f"dolci_rlvlgen_p50_x{args.scale}"
    if out_dir.exists():
        print(f"exists: {out_dir}")
        return
    # the generator drops test prompts, so draw a few percent more items than needed
    pool = gen_pool(args.out_root, int(half * 1.03) + 1000)
    synth = load_synth(pool, half)
    dolci, ev = dolci_rows(half)
    rows = dolci + synth
    random.Random(args.seed).shuffle(rows)
    eval_rows = [{"prompt": p, "target": t, "source": "dolci", "family": "", "example_id": f"dolci-eval-{i}",
                  "mask_tool_results": False} for i, (p, t) in enumerate(zip(ev["prompt"], ev["target"]))]
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    DatasetDict({"train": Dataset.from_list(rows), "eval": Dataset.from_list(eval_rows)}).save_to_disk(str(tmp))
    fam: dict[str, int] = {}
    for r in synth:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    meta = {"percent_synthetic": 50, "scale": args.scale, "total": len(rows), "n_dolci": len(dolci),
            "n_dolci_ext": sum(r["example_id"].startswith("dolci-ext") for r in dolci), "n_synthetic": len(synth),
            "synthetic_families": dict(sorted(fam.items())), "n_eval": len(eval_rows), "seed": args.seed,
            "pool": {"train": str(pool), "train_sha256": sha256(pool), "test_excluded": str(BASE / "pool/test.jsonl")},
            "design": f"p50 at {args.scale}x rows; synthetic rows 0..{half - 1} (p50 pool = prefix); Dolci 100k subset "
                      f"+ rows >= {CANDIDATE_END} of the same seed-3407 shuffle; eval Dolci-only (unchanged)"}
    (tmp / "mixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    tmp.rename(out_dir)
    print(json.dumps({k: meta[k] for k in ("total", "n_dolci", "n_dolci_ext", "n_synthetic")}))


if __name__ == "__main__":
    main()
