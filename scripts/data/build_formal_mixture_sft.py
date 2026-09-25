#!/usr/bin/env python3
"""Build the Dolci + rlvlgen formal-CoT SFT mixture sweep.

Design, the same as build_reasoning_mixture_sft.py: REPLACEMENT, not addition.
Each mixture has exactly --total train rows,
    p{X} = Dolci train[: total - k]  +  synthetic pool[: k],   k = round(X/100 * total)
shuffled with random.Random(--seed). The Dolci rows are the first rows of the
prepared 100k subset (dolci_no_tools_single_turn_100k_seed3407), so the kept
Dolci rows are nested across X. The synthetic rows are also nested: every
mixture takes the first k records of one fixed pool. The eval split is the
Dolci-only 2048-row eval split in every condition.

The synthetic data is not stored anywhere; it is regenerated here through the
rlvlgen CLI, which is deterministic (every example is seeded from sha256 of
split and index) and strict-checks every record before writing:
    python -m rlvlgen.dataset --split train --n POOL --out pool/train.jsonl
    python -m rlvlgen.dataset --split test  --n TEST --out pool/test.jsonl --exclude pool/train.jsonl
The sha256 of both files goes into every manifest. If the pool files exist
already they are reused, and a changed hash against pool/pool_manifest.json fails closed.

Synthetic rows carry prompt = "<formal>\\n{prompt}", target = the rendered
"<proof>\\n...</proof>\\nAnswer: ..." (the rlvlgen `messages`), and
mask_tool_results = True, which tells train_formal_mixture_sft.py to take the
tool-result contents out of the loss.
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

from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_FRACS = list(range(0, 51, 5))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cli(python: str, pythonpath: str, argv: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [python, "-m", "rlvlgen.dataset", *argv]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def ensure_pool(args) -> dict:
    pool = args.out_root / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    train, test = pool / "train.jsonl", pool / "test.jsonl"
    if not train.is_file():
        run_cli(args.python, args.rlvlgen_path, ["--split", "train", "--n", str(args.pool_size), "--out", str(train)])
    if not test.is_file():
        run_cli(args.python, args.rlvlgen_path,
                ["--split", "test", "--n", str(args.test_size), "--out", str(test), "--exclude", str(train)])
    meta = {
        "train": str(train), "train_sha256": sha256(train),
        "test": str(test), "test_sha256": sha256(test),
        "pool_size": args.pool_size, "test_size": args.test_size,
        "rlvlgen_path": args.rlvlgen_path,
    }
    mpath = pool / "pool_manifest.json"
    if mpath.is_file():
        old = json.loads(mpath.read_text())
        for key in ("train_sha256", "test_sha256"):
            if old.get(key) != meta[key]:
                raise SystemExit(f"{key} changed since {mpath} was written; delete the pool to regenerate")
    else:
        mpath.write_text(json.dumps(meta, indent=2) + "\n")
    n_train = sum(1 for _ in open(train))
    n_test = sum(1 for _ in open(test))
    if n_train < args.pool_size or n_test < args.test_size:
        raise SystemExit(f"pool too small: train {n_train}/{args.pool_size}, test {n_test}/{args.test_size}")
    return meta


def load_synth(path: Path, k: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if len(rows) >= k:
                break
            rec = json.loads(line)
            user, asst = rec["messages"]
            assert user["role"] == "user" and asst["role"] == "assistant"
            rows.append({"prompt": user["content"], "target": asst["content"], "source": "rlvlgen",
                         "family": rec["family"], "example_id": rec["id"], "mask_tool_results": True})
    if len(rows) < k:
        raise SystemExit(f"synthetic pool has {len(rows)} < {k} rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dolci", type=Path, required=True, help="prepared Dolci DatasetDict (train/eval, prompt/target)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fracs", default=",".join(map(str, DEFAULT_FRACS)), help="percent synthetic, comma list")
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--eval-rows", type=int, default=None, help="cap the Dolci eval split (smoke tests)")
    ap.add_argument("--pool-size", type=int, default=50000)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--name-prefix", default="dolci_rlvlgen")
    ap.add_argument("--python", default=sys.executable, help="interpreter that runs the rlvlgen CLI")
    ap.add_argument("--rlvlgen-path",
                    default="/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python",
                    help="PYTHONPATH for the rlvlgen CLI (generator + rlvl checker binding)")
    args = ap.parse_args()

    pool_meta = ensure_pool(args)
    dolci = load_from_disk(str(args.dolci))
    if len(dolci["train"]) < args.total:
        raise SystemExit(f"dolci train has {len(dolci['train'])} < total {args.total}")
    eval_ds = dolci["eval"]
    if args.eval_rows is not None:
        eval_ds = eval_ds.select(range(min(args.eval_rows, len(eval_ds))))
    eval_rows = [{"prompt": p, "target": t, "source": "dolci", "family": "", "example_id": f"dolci-eval-{i}",
                  "mask_tool_results": False} for i, (p, t) in enumerate(zip(eval_ds["prompt"], eval_ds["target"]))]

    fracs = [int(x) for x in args.fracs.split(",") if x.strip()]
    for pct in fracs:
        k = int(round(pct / 100 * args.total))
        if k > args.pool_size:
            raise SystemExit(f"p{pct} needs {k} synthetic rows > pool {args.pool_size}")
        n_dolci = args.total - k
        out_dir = args.out_root / "mixtures" / f"{args.name_prefix}_p{pct:02d}"
        tmp = out_dir.with_name(out_dir.name + ".tmp")
        if out_dir.exists():
            print(f"exists, skipping: {out_dir}")
            continue
        kept = dolci["train"].select(range(n_dolci))
        rows = [{"prompt": p, "target": t, "source": "dolci", "family": "", "example_id": f"dolci-train-{i}",
                 "mask_tool_results": False} for i, (p, t) in enumerate(zip(kept["prompt"], kept["target"]))]
        synth = load_synth(Path(pool_meta["train"]), k)
        rows += synth
        random.Random(args.seed).shuffle(rows)
        shutil.rmtree(tmp, ignore_errors=True)
        DatasetDict({"train": Dataset.from_list(rows), "eval": Dataset.from_list(eval_rows)}).save_to_disk(str(tmp))
        fam: dict[str, int] = {}
        for r in synth:
            fam[r["family"]] = fam.get(r["family"], 0) + 1
        meta = {
            "percent_synthetic": pct, "total": args.total, "n_dolci": n_dolci, "n_synthetic": k,
            "synthetic_families": dict(sorted(fam.items())), "n_eval": len(eval_rows), "seed": args.seed,
            "dolci_source": str(args.dolci), "pool": pool_meta,
            "design": "replacement (constant example count); Dolci train[:total-k] + pool[:k]; eval Dolci-only",
        }
        (tmp / "mixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        tmp.rename(out_dir)
        print(json.dumps({k2: meta[k2] for k2 in ("percent_synthetic", "n_dolci", "n_synthetic")}), flush=True)


if __name__ == "__main__":
    main()
