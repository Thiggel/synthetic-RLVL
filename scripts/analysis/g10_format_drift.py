#!/usr/bin/env python3
"""How G10 (G8 final + 600 cvf GRPO steps) drifted out of the answer format (2026-10-01).

cvf = correct x valid x faithful premises scores the proof and its `ans` line; it ignores what follows </proof>.
Per training step (256 rollouts, <run>/completions/completions_*.parquet), the tail after the last </proof> is
classified, first match wins:
  `<formal>` loop       the completion contains a second `<formal>` (re-opens the prompt, repeats the proof)
  Answer: x             format_ok: one </proof>, then exactly `Answer: <answer>` (the SFT format)
  Answer: x ; n         format_ok as well, but the line copies the proof's `ans x ; n` (step reference)
  Answer x ; n          no colon: format_ok 0, the eval answer extractors return nothing
  other                 no </proof>, several </proof>, prose after it, ...
Also per step: cvf, format_ok, mean length, and cvf on format_ok = 0 rollouts (reward the format gate would remove).
Gate evals of the G10 checkpoints (greedy, 950 Dolci gate items; generator test, 2000 items; frozen checker) are
added where present. L1_cvf (since step 51 with the format-gated reward cvf_fmt) is drawn for comparison.
Writes analysis/g10_format_drift.json and reports/figures/g10_format_drift.{png,pdf}.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
G10, G8, L1 = RUNS / "2b_lcfp32m_G10_cvf_cont600", RUNS / "2b_lcfp32m_G8_cvf", RUNS / "L1_cvf"
REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "analysis/g10_format_drift.json"
OUT_FIG = REPO / "reports/figures/g10_format_drift"
TAILS = ["Answer: x", "Answer: x ; n", "Answer x ; n", "<formal> loop", "other"]
COLORS = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:gray"]
BIN = 10


def tail(c: str) -> str:
    if c.count("<formal>") >= 1:
        return TAILS[3]
    parts = c.split("</proof>")
    if len(parts) != 2:
        return TAILS[4]
    t = parts[1].strip()
    if re.fullmatch(r"Answer:[^\n;]*", t):
        return TAILS[0]
    if re.fullmatch(r"Answer:[^\n]*;[^\n]*", t):
        return TAILS[1]
    if re.fullmatch(r"Answer [^\n]*", t):
        return TAILS[2]
    return TAILS[4]


def per_step(run: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(run / "completions/completions_*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["step", "completion", "cvf", "format_ok"]) for f in fs],
                  ignore_index=True)
    d["tail"] = d.completion.map(tail)
    d["chars"] = d.completion.str.len()
    d["b"] = d.step // BIN * BIN
    g = d.groupby("b")
    out = pd.DataFrame({"cvf": g.cvf.mean(), "format_ok": g.format_ok.mean(), "chars": g.chars.mean(),
                        "cvf_unformatted": g.apply(lambda x: ((x.cvf > 0) & (x.format_ok == 0)).mean())})
    for t in TAILS:
        out[t] = g["tail"].apply(lambda x, t=t: (x == t).mean())
    return out[g.size() >= g.size().max() / 2]


def gate(ckpt: Path) -> dict | None:
    f = ckpt / "rl_gate_dolci/summary.json"
    if not f.is_file():
        return None
    o = json.loads(f.read_text())["overall"]["all"]
    r = {k: o[k] for k in ("valid", "valid_correct", "correct", "ans_correct", "has_proof")}
    gens = [json.loads(x) for x in open(ckpt / "rl_gate_dolci/generations.jsonl")]
    r["formal_loop"] = sum(g.get("generation", g.get("completion", "")).count("<formal>") >= 1 for g in gens) / len(gens)
    fe = ckpt / "formal_eval/summary.json"
    if fe.is_file():
        e = json.loads(fe.read_text())["overall"]
        r.update({f"gen_{k}": e[k] for k in ("valid", "faithful", "answer_acc", "mean_gen_tokens")})
    return r


def main() -> None:
    g10 = per_step(G10)
    l1 = per_step(L1) if L1.is_dir() else None
    gates = {0: gate(G8 / "final")}
    for s in (100, 200, 300, 400, 500):
        gates[s] = gate(G10 / f"checkpoint-{s}")
    gates[600] = gate(G10 / "final")
    gates = {s: r for s, r in gates.items() if r is not None}
    res = {"g10_per_step": g10.reset_index().to_dict(orient="records"), "g10_gates": gates,
           "l1_cvf_per_step": None if l1 is None else l1.reset_index().to_dict(orient="records")}
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    ax = axes[0]
    x = g10.index + BIN / 2
    ax.stackplot(x, *[g10[t] for t in TAILS], labels=TAILS, colors=COLORS, alpha=.8)
    ax.set_ylim(0, 1)
    ax.set_title("G10 rollouts: what follows </proof>")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    ax.plot(x, g10.cvf, color="tab:green", label="cvf (the training reward)")
    ax.plot(x, g10.format_ok, color="tab:blue", label="format_ok")
    ax.plot(x, g10.cvf_unformatted, color="tab:red", label="cvf > 0 but format_ok = 0")
    if l1 is not None:
        ax.plot(l1.index + BIN / 2, l1.format_ok, "--", color="tab:blue", alpha=.6, label="L1_cvf format_ok (cvf_fmt)")
    ax.set_ylim(-.02, 1.02)
    ax.set_title("G10 rollouts: reward vs format")
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(x, g10.chars, color="k", alpha=.35, lw=1)
    ax2.set_ylabel("mean chars (grey)")
    ax = axes[2]
    s = sorted(gates)
    for k, c, lab in (("valid", "tab:red", "valid"), ("valid_correct", "tab:purple", "valid·correct"),
                      ("correct", "tab:green", "correct (Answer line)"), ("ans_correct", "tab:olive", "proof ans correct"),
                      ("formal_loop", "k", "<formal> loop")):
        ax.plot(s, [gates[i][k] for i in s], marker="o", color=c, label=lab)
    ax.set_title("Dolci gate (950, greedy): G8 final = step 0")
    ax.legend(fontsize=7)
    ax = axes[3]
    s = [i for i in sorted(gates) if "gen_valid" in gates[i]]
    for k, c in (("gen_valid", "tab:red"), ("gen_faithful", "tab:blue"), ("gen_answer_acc", "tab:green")):
        ax.plot(s, [gates[i][k] for i in s], marker="o", color=c, label=k[4:])
    ax.set_title("generator test (2000, greedy)")
    ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("G10 GRPO step")
        ax.grid(alpha=.3)
    fig.suptitle("G10: cvf rewards the proof, not the answer line; the tail drifts (Answer: x -> Answer x ; n) and "
                 "then loops (<formal> + prompt copy)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=130)
    print(g10.round(3).iloc[::5].to_string())
    print(json.dumps(gates, indent=0))


if __name__ == "__main__":
    main()
