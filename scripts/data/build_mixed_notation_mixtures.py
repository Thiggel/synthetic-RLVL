#!/usr/bin/env python3
"""Build mixtures that carry both notations of the derivation data.

The headline sweep trains each notation on its own. This builder splits a
given share evenly between formal and English derivations, so a model sees
both substrates of the same procedure. Every mixture keeps the replacement
design, 100k examples in total with the derivations displacing instruction
examples. The two halves are drawn from disjoint proofs, so no proof appears
twice in different notation.

Each share is built twice. The plain variant leaves the derivation prompts as
they are, so the model has to pick a notation on its own. The conditioned
variant prefixes every derivation prompt with the same instruction the
mode-conditioned study used, naming the notation the answer must take, so the
model knows before reasoning begins which substrate to think in.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from scripts.data.build_mode_conditioned_mixture import ENGLISH_PREFIX, FORMAL_PREFIX
from scripts.data.build_reasoning_mixture_sft import BANDS, render_traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolci", type=Path, required=True)
    ap.add_argument("--bp-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--band", type=int, default=25, choices=sorted(BANDS))
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    dolci = load_from_disk(str(args.dolci))
    if len(dolci["train"]) < args.total:
        raise SystemExit(f"dolci train has {len(dolci['train'])} < total {args.total}")

    for frac in args.fracs:
        k = int(round(frac * args.total))
        half = k // 2
        n_dolci = args.total - 2 * half
        kept = dolci["train"].select(range(n_dolci))
        # one draw of 2*half proofs, split in two, so the halves never overlap
        formal = render_traces(args.bp_root, args.band, "logic", 2 * half, args.seed)
        english = render_traces(args.bp_root, args.band, "nl_exact", 2 * half, args.seed)
        formal, english = formal[:half], english[half:]
        base_rows = [{"prompt": p, "target": t} for p, t in zip(kept["prompt"], kept["target"])]
        tag = "p%02d" % round(frac * 100)
        for variant in ("plain", "cond"):
            fp = FORMAL_PREFIX if variant == "cond" else ""
            ep = ENGLISH_PREFIX if variant == "cond" else ""
            rows = list(base_rows)
            rows += [{"prompt": fp + r["prompt"], "target": r["target"]} for r in formal]
            rows += [{"prompt": ep + r["prompt"], "target": r["target"]} for r in english]
            random.Random(args.seed).shuffle(rows)
            out_dir = args.out_root / f"mixed_{variant}_{tag}"
            DatasetDict({"train": Dataset.from_list(rows), "eval": dolci["eval"]}).save_to_disk(str(out_dir))
            meta = {
                "condition": f"mixed_{variant}_band{args.band}",
                "variant": variant,
                "band": args.band,
                "frac": frac,
                "total": args.total,
                "n_dolci": n_dolci,
                "n_formal": half,
                "n_english": half,
                "seed": args.seed,
                "dolci_source": str(args.dolci),
                "bp_source": str(args.bp_root / BANDS[args.band]),
                "design": "replacement (constant example count); halves from disjoint proofs; eval split is Dolci-only",
            }
            (out_dir / "mixture_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(json.dumps(meta))


if __name__ == "__main__":
    main()
