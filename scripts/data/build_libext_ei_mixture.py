#!/usr/bin/env python3
"""Continued-SFT arms from the L1 base: self-distilled real-prompt proofs (EI) x new lemma families (2026-09-30).

Approved plan (user, 2026-09-30: "self-distillation, lemmas everywhere and dolci"). The Dolci gate fails on
transfer, not on in-domain skill: 70% of G8's gate samples are parse errors (notation and prose outside the
language, malformed `; rule`, multi-word names), and >60% of lib cites name lemmas that did not exist
(reports/2026-09-30_libext_and_gate_failures.md). Two remedies, crossed in a 2x2 of matched arms, all
continued from qwen35_2b_p50_cont_lc_fp32m (the L1 base) with the same Dolci rows and the same size
(14k ours + 6k Dolci = 70/30):
  c   14k fresh generator items (default families)                        + 6k Dolci   (control: more SFT)
  l   7k new-family items (lib extension) + 7k fresh generator             + 6k Dolci
  e   EI rows + fresh generator to 14k                                     + 6k Dolci
  le  EI rows + 7k new-family items + fresh generator to 14k               + 6k Dolci
EI rows = the policy's own reward-passing proofs (G8 final, cvf reward, 16 samples at T=1, frozen checker;
scripts/rl_prompt_filter.py) on real number-answer prompts only (Dolci wordprob/math train pool, GSM8K train),
at most --max-per-prompt distinct proofs per prompt (shortest first), repeated --ei-repeat times, both lowered
until the EI rows fit next to the 7k new-family rows (ei_rows). Dropped:
proofs with a `given` quoting the question (a `?` or how many/what/find/... in the quote: the solution
stated as a fact), and all Dolci yesno proofs: 110/111 answer yes with free-form givens the checker cannot
tie to their quote (`~crow(dog1, dog2) ; given "outside with luggage..."`), i.e. unfaithful by construction.
new-family items = formal_libext_20260930/pool/train_math_20000.jsonl (families mathlemmas, numth, geom,
rates, counting; new lemma library). Fresh generator items = train rows 60000.. of the x3 pool (the base
saw rows 0..54999). Dolci = dolci-ext-* rows 10000..15999 of dolci_rlvlgen_p50_x3 (the base saw 0..9999).
eval = the p50 eval split. Evaluate under the NEW checker (the new families cite new lemmas).
EI round 2 (2026-10-01): arm e2 = e with the harvest from a stronger, RL'd teacher (HARVEST2: G12 checkpoint-100,
i.e. G10@500 + 100 cvf_fmt steps; same pool sizes, n=16, T=1, cvf reward, frozen checker, gate near-duplicates
excluded). The 2x2 found EI drives gate validity (§8 of the report); e2 vs e asks whether a better teacher
gives a better student.
Ablation e2s (2026-10-02): e2 had a better teacher AND 52% more EI rows (6,776 from 2,512 prompts vs 4,457 from
1,778). e2s = the e2 harvest cut to e's size: per bench a random subset of as many prompts as e kept, <= 4 proofs
each, then random non-first proofs dropped down to e's row count per bench. e2s - e = teacher quality at fixed quantity.

EI round 3 (2026-10-03): arm e3 = e2 with the harvest from G16 checkpoint-250 (GRPO cvf_fmt + overlong + no-proof
penalty from e2 final; better than G15@250 on the clean gate: valid .229 vs .210, v*c .102 vs .091). Since the teacher
is a new-library model, the harvest is checked with the NEW checker snapshot (checker_snapshot_libext_20261001, a
superset of the old one); same recipe and EI row cap otherwise, so e3 - e2 = the next teacher-quality step.
Writes <out-root>/<arm> (DatasetDict + mixture_manifest.json); existing arms are skipped. Run with .venv_rlvl_grpo.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_ei_mixture import trim  # noqa: E402
from build_formal_mixture_sft import load_synth, sha256  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
DS = DATA / "datasets"
BASE = DS / "formal_mixture_20260925"
SCALE = DS / "formal_mixture_scale_20260929"
LIBEXT = DS / "formal_libext_20260930/pool/train_math_20000.jsonl"
HARVEST = [DATA / "rl_filter_20260930/G8final_cvf_n16.json.passing.jsonl",
           DATA / "rl_filter_20260930/G8final_cvf_n16_gsm8k.json.passing.jsonl"]
HARVEST2 = [DATA / "rl_filter_20261001/G12c100_cvf_n16_dolci.json.passing.jsonl",
            DATA / "rl_filter_20261001/G12c100_cvf_n16_gsm8k.json.passing.jsonl"]
HARVEST3 = [DATA / "rl_filter_20261003/G16c250_cvf_n16_dolci.json.passing.jsonl",
            DATA / "rl_filter_20261003/G16c250_cvf_n16_gsm8k.json.passing.jsonl"]
EI_SRC = {"e": HARVEST, "le": HARVEST, "e2": HARVEST2, "e2s": HARVEST2, "e3": HARVEST3}
MATCH = {"e2s": "e"}  # arm -> arm whose EI size (prompts per bench, rows) it copies
REAL = ("dolci_wordprob", "dolci_math", "gsm8k_train")
INIT = "formal_mixture_sft_20260925/qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407/final"
GIVEN = re.compile(r'(?m)^\d+ .*? ; given "(.*)"\s*$')
ASKS = re.compile(r"\?|\b(how (many|much|long|far|old|often)|what|which|find|calculate|compute|determine)\b", re.I)


def ei_rows(paths: list[Path], max_per_prompt: int, repeat: int, cap: int, seed: int,
            match: dict | None = None) -> tuple[list[dict], dict]:
    """Passing proofs on real prompts, grouped by prompt (shortest first). Of the per-prompt caps <= max_per_prompt
    and repeats <= repeat, take the largest cap, then the largest repeat, whose rows fit in `cap` (distinct proofs
    before upweighting); if one proof per prompt is still too many, one proof each for a random subset of prompts.
    match = another arm's ei_stats: keep that many prompts per bench and cut to that many rows (see e2s)."""
    by_id: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["bench"] not in REAL:
                continue
            st = stats[r["bench"]]
            st["passing"] += 1
            t = trim(r["completion"])
            if t is None:
                st["no_trim"] += 1
                continue
            if any(ASKS.search(q) for q in GIVEN.findall(t)):
                st["question_premise"] += 1
                continue
            by_id[r["id"]].setdefault(t, r)
    groups = [[{"prompt": f"<formal>\n{proofs[t]['raw_prompt']}", "target": t, "source": "ei",
                "family": f"ei_{proofs[t]['bench']}", "example_id": f"ei/{pid}", "mask_tool_results": False}
               for t in sorted(proofs, key=lambda x: (len(x), x))] for pid, proofs in sorted(by_id.items())]
    rng = random.Random(seed)
    if match:
        by_bench = collections.defaultdict(list)
        for g in groups:
            by_bench[g[0]["family"][3:]].append(g)
        groups = sorted((g for b, gs in by_bench.items() for g in rng.sample(gs, min(len(gs), match.get(b, {}).get("prompts", 0)))),
                        key=lambda g: g[0]["example_id"])
        repeat = 1
    fit = next(((k, r) for k in range(max_per_prompt, 0, -1) for r in range(repeat, 0, -1)
                if r * sum(min(k, len(g)) for g in groups) <= cap), None)
    if fit is None:
        groups, fit = random.Random(seed).sample(groups, cap), (1, 1)
    k, r = fit
    rows = [x for g in groups for x in g[:k]]
    if match:  # per bench, drop random non-first proofs down to the matched arm's row count
        drop = set()
        for bench in REAL:
            extra = [(i, j) for i, g in enumerate(groups) if g[0]["family"][3:] == bench for j in range(1, min(k, len(g)))]
            n = sum(min(k, len(g)) for g in groups if g[0]["family"][3:] == bench) - match.get(bench, {}).get("kept", 0)
            drop |= set(rng.sample(extra, max(0, min(n, len(extra)))))
        rows = [x for i, g in enumerate(groups) for j, x in enumerate(g[:k]) if (i, j) not in drop]
    for x in rows:
        stats[x["family"][3:]]["kept"] += 1
    for g in groups:
        stats[g[0]["family"][3:]]["prompts"] += 1
    out = {b: dict(c) for b, c in stats.items()}
    out["selection"] = {"per_prompt": k, "repeat": r, "prompts": len(groups), "rows": len(rows) * r, "cap": cap}
    return rows * r, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="c,l,e,le")
    ap.add_argument("--n-ours", type=int, default=14000)
    ap.add_argument("--n-lib", type=int, default=7000)
    ap.add_argument("--n-dolci", type=int, default=6000)
    ap.add_argument("--max-per-prompt", type=int, default=4)
    ap.add_argument("--ei-repeat", type=int, default=2)
    ap.add_argument("--out-root", type=Path, default=DS / "formal_libext_ei_20260930")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk

    arms = args.arms.split(",")
    ei: dict[str, list[dict]] = {}
    ei_stats: dict[str, dict] = {}
    for arm in arms:
        if arm in EI_SRC and not (args.out_root / arm).exists():
            src = EI_SRC[arm]
            missing = [str(p) for p in src if not p.is_file()]
            if missing:
                raise SystemExit(f"missing harvest: {missing}")
            match = None
            if arm in MATCH:
                match = json.loads((args.out_root / MATCH[arm] / "mixture_manifest.json").read_text())["ei_stats"]
            ei[arm], ei_stats[arm] = ei_rows(src, args.max_per_prompt, args.ei_repeat, args.n_ours - args.n_lib,
                                             args.seed, match)
            print("EI", arm, json.dumps(ei_stats[arm]), flush=True)
    lib = load_synth(LIBEXT, args.n_lib)
    pool = SCALE / "pool/train_155500.jsonl"
    gen = load_synth(pool, 60000 + args.n_ours)[60000:]
    x3 = load_from_disk(str(SCALE / "mixtures/dolci_rlvlgen_p50_x3"))
    dolci = [r for r in x3["train"] if r["example_id"].startswith("dolci-ext-")]
    dolci.sort(key=lambda r: int(r["example_id"].rsplit("-", 1)[1]))
    dolci = dolci[10000:10000 + args.n_dolci]
    assert len(dolci) == args.n_dolci
    ev = load_from_disk(str(BASE / "mixtures/dolci_rlvlgen_p50"))["eval"]
    n, k = args.n_ours, args.n_lib
    build = {"c": lambda: gen[:n],
             "l": lambda: lib + gen[:n - k],
             "e": lambda: ei["e"] + gen[:n - len(ei["e"])],
             "le": lambda: ei["le"] + lib + gen[:n - k - len(ei["le"])],
             "e2": lambda: ei["e2"] + gen[:n - len(ei["e2"])],
             "e2s": lambda: ei["e2s"] + gen[:n - len(ei["e2s"])],
             "e3": lambda: ei["e3"] + gen[:n - len(ei["e3"])]}
    for arm in arms:
        out = args.out_root / arm
        if out.exists():
            print(f"exists: {out}")
            continue
        rows = build[arm]() + dolci
        assert len(rows) == n + args.n_dolci, (arm, len(rows))
        random.Random(args.seed).shuffle(rows)
        tmp = out.with_name(out.name + ".tmp")
        DatasetDict({"train": Dataset.from_list(rows), "eval": ev}).save_to_disk(str(tmp))
        fam: dict[str, int] = {}
        for r in rows:
            fam[r["family"] or "dolci"] = fam.get(r["family"] or "dolci", 0) + 1
        meta = {"arm": arm, "total": len(rows), "families": dict(sorted(fam.items())), "seed": args.seed,
                "init": INIT, "args": {k_: str(v) for k_, v in vars(args).items()},
                "pools": {"libext": str(LIBEXT), "libext_sha256": sha256(LIBEXT), "gen": f"{pool} rows 60000..",
                          "dolci": "dolci_rlvlgen_p50_x3 dolci-ext-* rows 10000..",
                          "ei": [str(p) for p in EI_SRC[arm]] if arm in EI_SRC else None},
                "ei_stats": ei_stats.get(arm),
                "design": __doc__.split("\n\n")[1]}
        (tmp / "mixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        tmp.rename(out)
        print(arm, json.dumps(meta["families"]), flush=True)


if __name__ == "__main__":
    main()
