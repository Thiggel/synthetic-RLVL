#!/usr/bin/env python
"""Condensed formal surface rendering of BranchProof latent proofs (prototype).

A RENDERER change only: the latent proof (premises_fol / proof_fol /
question / answer) is unchanged and stays verifiable by LogicEngine via
``validate_logic_example`` after parsing the condensed surface back.

Differences vs the standard SFT-style ``logic`` rendering
(task_sample_from_materialized_row, template=logic; document = prompt+target):

- fully formal document: the numbered NL theory prompt is NOT restated; the
  formal premises appear exactly once (the standard document states every rule
  twice: once as NL in <question>, once as FOL in <premises>).
- boilerplate dropped: no trivial constants glossary ("c0 = c0"), no
  <conclusion> block (restates the last proof line).
- predicate glossary condensed to one "A=lime;B=maple;..." line (needed to
  ground the NL answer token).
- ASCII operator spacing removed: "J(c0) & I(c0) -> C(c1)" -> "J(c0)&I(c0)->C(c1)",
  " ; " justification separator -> ";".
- atom/predicate names are already minimal (A..Z, c0..cN) and are kept.

Round-trip check: parse the condensed document back into premise/proof lines,
re-attach the original numbering, require exact equality with the source row's
``premises_fol``/``proof_fol``, and require ``validate_logic_example`` to pass
on the reconstructed example.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRED_RE = re.compile(r"^(\w+?)\(?x\)?: x is (\w+)$")


def _body(line: str) -> str:
    stripped = line.strip()
    if ". " in stripped:
        return stripped.split(". ", 1)[1].strip()
    return stripped


def _condense_formula(body: str) -> str:
    # The justification (after " ; ") may itself contain "->" (e.g. "->E"),
    # so only the formula part gets operator-spacing removal.
    formula, sep, justification = body.partition(" ; ")
    formula = formula.replace(" & ", "&").replace(" -> ", "->")
    return formula + (";" + justification if sep else "")


def _expand_formula(body: str) -> str:
    formula, sep, justification = body.partition(";")
    formula = formula.replace("&", " & ").replace("->", " -> ")
    return formula + (" ; " + justification if sep else "")


def condensed_defs(predicates: list[str]) -> str:
    pairs = []
    for line in predicates:
        m = PRED_RE.match(_body(line))
        if not m:
            raise ValueError(f"unparseable predicate glossary line: {line!r}")
        pairs.append(f"{m.group(1)}={m.group(2)}")
    return ";".join(pairs)


def render_condensed(row: dict) -> str:
    premises = "\n".join(_condense_formula(_body(l)) for l in row["premises_fol"])
    proof = "\n".join(_condense_formula(_body(l)) for l in row["proof_fol"])
    return (
        "<cformal>\n"
        "<defs>\n" + condensed_defs(list(row["predicates"])) + "\n</defs>\n"
        "<premises>\n" + premises + "\n</premises>\n"
        "<q>\n" + str(row["question_fol"]) + "\n</q>\n"
        "<proof>\n" + proof + "\n</proof>\n"
        "</cformal>\n"
        "<answer>\n" + str(row["answer"]) + "\n</answer>"
    )


def _block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, re.S)
    if not m:
        raise ValueError(f"missing <{tag}> block")
    return m.group(1)


def parse_condensed(text: str) -> dict:
    defs = _block(text, "defs")
    predicates = []
    for i, pair in enumerate(defs.split(";")):
        name, _, word = pair.partition("=")
        if not name or not word:
            raise ValueError(f"bad defs pair: {pair!r}")
        predicates.append(f"{i + 1}. {name}x: x is {word}")
    premises = [_expand_formula(l) for l in _block(text, "premises").splitlines()]
    proof = [_expand_formula(l) for l in _block(text, "proof").splitlines()]
    return {
        "predicates": predicates,
        "premises_bodies": premises,
        "proof_bodies": proof,
        "question": _block(text, "q"),
        "answer": _block(text, "answer"),
    }


def roundtrip_validate(row: dict, strict_engine: bool = True) -> None:
    text = render_condensed(row)
    parsed = parse_condensed(text)
    src_premises = [_body(l) for l in row["premises_fol"]]
    src_proof = [_body(l) for l in row["proof_fol"]]
    if parsed["premises_bodies"] != src_premises:
        raise AssertionError("premises round-trip mismatch")
    if parsed["proof_bodies"] != src_proof:
        raise AssertionError("proof round-trip mismatch")
    src_pairs = [PRED_RE.match(_body(l)).groups() for l in row["predicates"]]
    parsed_pairs = [tuple(p.split(". ", 1)[1].split("x: x is ")) for p in parsed["predicates"]]
    if parsed_pairs != src_pairs:
        raise AssertionError("predicate glossary round-trip mismatch")
    if parsed["question"] != str(row["question_fol"]):
        raise AssertionError("question round-trip mismatch")
    if parsed["answer"] != str(row["answer"]):
        raise AssertionError("answer round-trip mismatch")
    if strict_engine:
        from synthetic_dataset import LogicExample
        from synthrlvl.datasets import validate_logic_example

        # Reconstruct the example from the PARSED surface (renumbered exactly
        # like the source row) and require engine validity.
        n_prem = len(row["premises_fol"])
        first_proof_no = int(str(row["proof_fol"][0]).split(".", 1)[0])
        example = LogicExample(
            constants=list(row["constants"]),
            predicates=list(row["predicates"]),
            premises_fol=[f"{i + 1}. {b}" for i, b in enumerate(parsed["premises_bodies"])],
            premises_nl=list(row["premises_nl"]),
            proof_fol=[f"{first_proof_no + i}. {b}" for i, b in enumerate(parsed["proof_bodies"])],
            proof_nl=list(row["proof_nl"]),
            question_fol=parsed["question"],
            question_nl=str(row["question_nl"]),
            answer=parsed["answer"],
            metadata=dict(row.get("metadata", {})),
        )
        # BranchProof-unique-v2 gold traces are citation-free
        # (metadata["citation_free_gold"]); the cited-strict mode rejects them
        # even in raw form, so validate through the citation-free engine path.
        result = validate_logic_example(example, citation_free=True)
        if not result.ok:
            raise AssertionError(f"LogicEngine validation failed: {result.error} {list(result.line_errors)[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sample", type=int, default=400, help="rows for the token-length measurement")
    ap.add_argument("--validate", type=int, default=100, help="rows for strict engine round-trip validation")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--band", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from synthrlvl.task import task_sample_from_materialized_row
    from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    table = pq.read_table(args.parquet)
    cols = table.column_names
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(table.num_rows, size=min(args.sample, table.num_rows), replace=False).tolist())
    cfg = TaskConfig(
        template=TemplateName("logic"),
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(1, args.band),
        val_steps=StepRange(1, args.band),
        seed=args.seed,
    )

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    std_lens, cond_lens, depths = [], [], []
    validated = 0
    for k, i in enumerate(idx):
        row = {c: table.column(c)[i].as_py() for c in cols}
        sample = task_sample_from_materialized_row(row, cfg=cfg)
        std_lens.append(n_tokens(sample.prompt + sample.target))
        cond_text = render_condensed(row)
        cond_lens.append(n_tokens(cond_text))
        depths.append(int(row["depth"]))
        if validated < args.validate:
            roundtrip_validate(row)
            validated += 1

    def stats(lens: list[int]) -> dict:
        arr = np.asarray(lens)
        return {
            "n": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "frac_gt_4096": float((arr > 4096).mean()),
            "frac_gt_8192": float((arr > 8192).mean()),
        }

    std, cond = stats(std_lens), stats(cond_lens)
    fit_limit = int(4096 * 0.9)
    payload = {
        "parquet": args.parquet,
        "tokenizer": args.tokenizer,
        "band": args.band,
        "sample": len(idx),
        "roundtrip_validated": validated,
        "standard_logic_doc": std,
        "condensed_logic_doc": cond,
        "reduction_ratio_mean": cond["mean"] / std["mean"],
        "reduction_ratio_p50": cond["p50"] / std["p50"],
        "arm5_gate": {
            "criterion": f"condensed p99 <= {fit_limit} (4096 with >=10% margin)",
            "condensed_p99": cond["p99"],
            "go": bool(cond["p99"] <= fit_limit),
        },
        "per_depth_condensed_max": {
            str(d): int(max(l for l, dd in zip(cond_lens, depths) if dd == d))
            for d in sorted(set(depths))
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
