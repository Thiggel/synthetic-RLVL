#!/usr/bin/env python3
"""Continued-SFT mixtures from the 2B p50 model: does the lib lemma catalog fix hallucinated cites? (2026-09-29)

Question (user, 2026-09-29): does the model know which lemmas exist in the persistent knowledge base?
It does not: the p50 SFT pool cites 56/136 lib lemmas and >90% of the lib cites on the Dolci gate name
lemmas that do not exist (analysis/lib_coverage.json). RLVL-next/gen/rlvlgen/families/lemmas.py (the
opt-in `lemmas` family) states and uses every lemma. Two matched arms, both continued from the p50
final with the same Dolci rows and the same size, differing only in 5k rows:
  lc  5k `lemmas` items + 5k fresh generator items (default families) + 10k Dolci
  ct  10k fresh generator items                                        + 10k Dolci   (control: just more SFT)
fresh generator items = train rows 50000.. of the x3 pool (the p50 model saw rows 0..49999)
Dolci = rows dolci-ext-* of dolci_rlvlgen_p50_x3 (never seen by the p50 model); eval = the p50 eval split.
Writes <out-root>/{lc,ct} (DatasetDict + mixture_manifest.json). Run with .venv_rlvl_grpo.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_formal_mixture_sft import load_synth, sha256  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data/datasets")
BASE = DATA / "formal_mixture_20260925"
SCALE = DATA / "formal_mixture_scale_20260929"
RLVLGEN = "/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-lemmas", type=int, default=5000)
    ap.add_argument("--n-gen", type=int, default=10000)
    ap.add_argument("--n-dolci", type=int, default=10000)
    ap.add_argument("--out-root", type=Path, default=DATA / "formal_lemma_continue_20260929")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk

    lem_path = args.out_root / "pool" / f"lemmas_{args.n_lemmas}.jsonl"
    if not lem_path.is_file():
        lem_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "rlvlgen.dataset", "--split", "train", "--n", str(int(args.n_lemmas * 1.02) + 50),
                        "--families", "lemmas", "--out", str(lem_path) + ".tmp", "--exclude", str(BASE / "pool/test.jsonl")],
                       env={**os.environ, "PYTHONPATH": RLVLGEN}, check=True)
        os.rename(str(lem_path) + ".tmp", lem_path)
    lem = load_synth(lem_path, args.n_lemmas)
    pool = SCALE / "pool/train_155500.jsonl"
    gen = load_synth(pool, 50000 + args.n_gen)[50000:]
    x3 = load_from_disk(str(SCALE / "mixtures/dolci_rlvlgen_p50_x3"))
    dolci = [r for r in x3["train"] if r["example_id"].startswith("dolci-ext-")]
    dolci.sort(key=lambda r: int(r["example_id"].rsplit("-", 1)[1]))
    dolci = dolci[:args.n_dolci]
    ev = load_from_disk(str(BASE / "mixtures/dolci_rlvlgen_p50"))["eval"]
    arms = {"lc": lem + gen[:args.n_gen - args.n_lemmas] + dolci, "ct": gen + dolci}
    for arm, rows in arms.items():
        out = args.out_root / arm
        if out.exists():
            print(f"exists: {out}")
            continue
        random.Random(args.seed).shuffle(rows)
        tmp = out.with_name(out.name + ".tmp")
        DatasetDict({"train": Dataset.from_list(rows), "eval": ev}).save_to_disk(str(tmp))
        fam: dict[str, int] = {}
        for r in rows:
            fam[r["family"] or "dolci"] = fam.get(r["family"] or "dolci", 0) + 1
        meta = {"arm": arm, "total": len(rows), "families": dict(sorted(fam.items())), "seed": args.seed,
                "init": "qwen35_2b_dolci_rlvlgen_p50_lr5em6_seed3407/final",
                "pools": {"lemmas": str(lem_path), "lemmas_sha256": sha256(lem_path), "gen": f"{pool} rows 50000..",
                          "dolci": "dolci_rlvlgen_p50_x3 dolci-ext-* rows (lowest indices)"},
                "design": __doc__.split("\n\n")[1]}
        (tmp / "mixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        tmp.rename(out)
        print(arm, json.dumps(meta["families"]))


if __name__ == "__main__":
    main()
