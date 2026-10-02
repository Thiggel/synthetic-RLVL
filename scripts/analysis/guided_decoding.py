#!/usr/bin/env python3
"""Checker-guided DFS decoding (scripts/guided_decode.py) vs unguided greedy decoding on the Dolci gate (2026-10-02).

Same model (e2 SFT), same 950 gate items. Every proof the guided decoder returns passes rlvl strict and the Stage-2
hardening (formal_rewards.components: grounded, ans agrees, >= 1 checked derived ancestor, not circular, premise
numbers stated in their quotes); the question is how often it finds one, at what cost, and whether the answers are
right. Unguided generations are rescored with the same components, so `valid_s2` / `cvf` compare like with like.

Decoding variants:
  unguided           rl_gate_dolci (greedy, one pass)
  guided(+repair)    guided_gate_hard_{repair,norepair}: the returned text (an unfinished prefix when nothing found)
  guided + fallback  the guided proof when found, else the unguided greedy completion
  guided ∩ agree     found and its answer equals the unguided greedy answer (a self-consistency filter on found proofs)

Also: per-bench rates, the clean subset (analysis/gate_contamination.md), search cost (sampled tokens per item) and
the share of found proofs whose answer rests on a "relation premise" (a `given` without numbers, e.g.
`chips_total = grid_size ; given "5x5 grid"`), the remaining unverifiable step.
Writes analysis/guided_decoding.{md,json} and reports/figures/guided_decoding.{png,pdf}.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
TEST = DATA / "datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
CONTAM = DATA / "datasets/rl_gate_dolci_instruct_20260928/contamination.json"
MODELS = {"e2 SFT": DATA / "formal_mixture_sft_20260925/qwen35_2b_lc_libext_e2_lr5em6_seed3407/final",
          "G14@750 (RL)": DATA / "grpo_formal_20260928/2b_le_G14_cvffmt_overlong/checkpoint-750"}
GUIDED = {"guided+repair": "guided_gate_hard_repair", "guided": "guided_gate_hard_norepair"}
OUT = REPO / "analysis/guided_decoding"
OUT_FIG = REPO / "reports/figures/guided_decoding"
_GIVEN = re.compile(r"^\s*\d+(?:\.\d+)*\s+(.*?)\s*;\s*given\b")


def load(p: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(p))}


def relation_premise(text: str) -> bool:
    """A `given` line whose formula states no number: a modeling step the checker cannot verify."""
    return any((m := _GIVEN.match(ln)) and not re.search(r"\d", m.group(1)) for ln in text.split("\n"))


def rates(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in ("found", "valid", "valid_s2", "correct", "cvf"):
        out[k] = sum(float(r.get(k, 0)) for r in rows) / n
    out["tokens"] = sum(r["gen_tokens"] for r in rows) / n
    return out


def main():
    recs = load(TEST)
    clean = set(json.load(open(CONTAM))["clean_ids"])
    res = {}
    for mname, mdir in MODELS.items():
        ung = load(mdir / "rl_gate_dolci/generations.jsonl")
        for r in ung.values():
            c = components(recs[r["id"]], r["generation"])
            r.update(valid_s2=c["valid"] * c["prem_ok"], cvf=c["correct"] * c["valid"] * c["prem_ok"], found=0.0)
        variants = {"unguided": ung}
        for gname, sub in GUIDED.items():
            f = mdir / sub / "generations.jsonl"
            if not f.exists():
                continue
            g = load(f)
            variants[gname] = g
            fb, agree = {}, {}
            for i, r in g.items():
                u = ung[i]
                fb[i] = r if r["found"] else {**u, "gen_tokens": r["gen_tokens"] + u["gen_tokens"]}
                same = r["found"] and u.get("pred_answer") is not None and str(r.get("pred_answer")) == str(u["pred_answer"])
                agree[i] = {**r, **({} if same else {"found": 0.0, "valid": 0.0, "valid_s2": 0.0, "cvf": 0.0})}
            variants[gname + " + fallback"] = fb
            variants[gname + " ∩ agree"] = agree
        out = {}
        benches = sorted({r["bench"] for r in recs.values()})
        for v, rows in variants.items():
            ids = [i for i in recs if i in rows]
            out[v] = {"all": rates([rows[i] for i in ids]), "clean": rates([rows[i] for i in ids if i in clean]),
                      **{b: rates([rows[i] for i in ids if recs[i]["bench"] == b]) for b in benches}}
            if v in GUIDED and v in variants:
                found = [rows[i] for i in ids if rows[i]["found"]]
                out[v]["found_rows"] = {
                    "n": len(found),
                    "correct": sum(r["correct"] for r in found) / max(1, len(found)),
                    "relation_premise": sum(relation_premise(r["generation"]) for r in found) / max(1, len(found)),
                    "uses_know": sum(bool(r.get("uses_know")) for r in found) / max(1, len(found)),
                    "unguided_correct_same_items": sum(ung[r["id"]]["correct"] for r in found) / max(1, len(found)),
                    "status": dict(collections.Counter(rows[i]["finish_reason"] for i in ids)),
                    "proof_tokens": sum(r["proof_tokens"] for r in found) / max(1, len(found)),
                    "gen_tokens": sum(r["gen_tokens"] for r in found) / max(1, len(found)),
                }
        res[mname] = out
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")

    lines = ["# Checker-guided decoding vs unguided greedy on the Dolci gate", "",
             "valid = rlvl strict; valid_s2 = Stage-2 valid with stated premise numbers (the cvf reward's validity); "
             "cvf = valid_s2 · correct; tokens = sampled tokens per item (guided: every candidate line).", ""]
    for mname, out in res.items():
        lines += [f"## {mname}", "", "| decoding | subset | n | found | valid | valid_s2 | correct | cvf | tokens |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for v, d in out.items():
            for s in ("all", "clean"):
                x = d[s]
                lines.append(f"| {v} | {s} | {x['n']} | {x['found']:.3f} | {x['valid']:.3f} | {x['valid_s2']:.3f} | "
                             f"{x['correct']:.3f} | {x['cvf']:.3f} | {x['tokens']:.0f} |")
        lines += ["", "Per bench (all items): valid_s2 / cvf / correct", "",
                  "| decoding | " + " | ".join(b for b in out["unguided"] if b.startswith("dolci")) + " |",
                  "|---|" + "---|" * sum(b.startswith("dolci") for b in out["unguided"])]
        for v, d in out.items():
            lines.append(f"| {v} | " + " | ".join(f"{d[b]['valid_s2']:.3f} / {d[b]['cvf']:.3f} / {d[b]['correct']:.3f}"
                                                for b in d if b.startswith("dolci")) + " |")
        for v in GUIDED:
            if v in out:
                fr = out[v]["found_rows"]
                lines += ["", f"**{v}, found proofs** (n={fr['n']}): correct {fr['correct']:.3f} (unguided greedy on the "
                          f"same items {fr['unguided_correct_same_items']:.3f}); answer rests on >= 1 relation premise "
                          f"(`given` without numbers) {fr['relation_premise']:.3f}; uses `know` {fr['uses_know']:.3f}; "
                          f"proof tokens {fr['proof_tokens']:.0f}, sampled tokens {fr['gen_tokens']:.0f}; "
                          f"status {fr['status']}"]
        lines.append("")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    benches = ["all", "clean"] + [b for b in res["e2 SFT"]["unguided"] if b.startswith("dolci")]
    fig, axes = plt.subplots(len(res), 3, figsize=(16, 4.2 * len(res)), squeeze=False)
    for row, (mname, out) in zip(axes, res.items()):
        vs = [v for v in out if not v.endswith("agree")]
        w = 0.8 / len(vs)
        for ax, m in zip(row, ("valid_s2", "cvf", "correct")):
            for j, v in enumerate(vs):
                ax.bar([i + j * w for i in range(len(benches))], [out[v][b][m] for b in benches], w, label=v)
            ax.set_xticks([i + 0.4 - w / 2 for i in range(len(benches))])
            ax.set_xticklabels([b.replace("dolci_", "") for b in benches], rotation=30)
            ax.set_title(f"{mname}: {m}")
        row[0].legend(fontsize=7)
    fig.suptitle("Dolci gate: unguided greedy vs checker-guided DFS decoding")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG.with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
