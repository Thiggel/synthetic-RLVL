#!/usr/bin/env python3
"""Repetition-loop collapse in GRPO (2026-10-01).

grpo_formal.py ran every arm with TRL's mask_truncated_completions=True: a completion that hits
--max-completion-length (2048 tokens) is dropped from the loss. Most such completions are proof-line repetition
loops, so a loop never receives the negative advantage its reward (0) would give it, while the shorter siblings in
its group are still pushed up or down. Symptoms over training: clipped ratio up, entropy down, more
zero-variance groups. G14 (--no-mask-truncated --overlong-penalty 0.5) is the fix arm against G13.
Trainer metrics per step: the newest checkpoint's trainer_state.json plus the `{'loss': ...}` lines of the job
logs (the step is recovered from the logged epoch). Rollouts (completions/*.parquet, 256 per step): share of
completions in which one non-empty line, with digit runs replaced by #, occurs >= LOOP_REPEATS times (a loop), and
share longer than 4000 characters, per BIN-step bucket.
Writes analysis/grpo_loop_collapse.{md,json} and reports/figures/grpo_loop_collapse.{png,pdf}.
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
OUT = REPO / "analysis/grpo_loop_collapse"
OUT_FIG = REPO / "reports/figures/grpo_loop_collapse"
# name: (run dir, job-log prefix, reward column of the rollout parquet)
ARMS = {"L1 correct (old lib)": ("L1_correct", "grpo_L1_correct", "correct"),
        "L1 cvf (old lib)": ("L1_cvf", "grpo_L1_cvf", "cvf"),
        "G12 cvf_fmt (G10@500, old lib)": ("2b_lcfp32m_G12_cvffmt_from_G10_500", "grpo_G12", "cvf"),
        "G13 cvf_fmt (le SFT, new lib)": ("2b_le_G13_cvffmt", "grpo_G13", "cvf"),
        "G14 = G13 + overlong penalty, unmasked": ("2b_le_G14_cvffmt_overlong", "grpo_G14", "cvf"),
        "G15 = G14 recipe from e2 SFT": ("2b_e2_G15_cvffmt_overlong", "grpo_G15", "cvf")}
COLORS = ["tab:green", "tab:red", "tab:purple", "tab:blue", "tab:orange", "tab:brown"]
TRAINER = ("completions/clipped_ratio", "entropy", "frac_reward_zero_std", "completions/mean_length", "reward")
BIN = 25
LOOP_REPEATS = 8
SMOOTH = 10


def trainer_log(run: str, prefix: str) -> pd.DataFrame | None:
    rows: dict[int, dict] = {}
    ck = sorted((RUNS / run).glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    per_epoch = None
    if ck:
        h = json.loads(ck[-1].read_text())["log_history"]
        rows = {x["step"]: x for x in h if "loss" in x}
        per_epoch = next((x["step"] / x["epoch"] for x in reversed(h) if x.get("epoch")), None)
    for f in sorted(glob.glob(str(LOGS / f"{prefix}_*.out")), key=lambda p: int(Path(p).stem.split("_")[-1])):
        for line in open(f, errors="replace"):
            if not line.startswith("{'loss'") or per_epoch is None:
                continue
            try:
                d = {k: float(v) for k, v in ast.literal_eval(line.strip()).items()}
            except (ValueError, SyntaxError):
                continue
            rows[round(d["epoch"] * per_epoch)] = d  # later jobs (resumed chains) overwrite
    if not rows:
        return None
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def is_loop(c: str) -> bool:
    # loops renumber their lines and step references (`89 ... ; subst 89 87`, `94 ... ; subst 94 92`)
    n = collections.Counter(re.sub(r"\d+", "#", x.strip()) for x in c.splitlines() if x.strip())
    return bool(n) and n.most_common(1)[0][1] >= LOOP_REPEATS


def rollouts(run: str, col: str) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(RUNS / run / "completions/completions_*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["step", "completion", col]) for f in fs], ignore_index=True)
    d["b"] = d.step // BIN * BIN
    d["loop"] = d.completion.map(is_loop)
    d["long"] = d.completion.str.len() > 4000
    g = d.groupby("b")
    r = pd.DataFrame({"loop": g.loop.mean(), "long": g.long.mean(), "reward": g[col].mean(), "n": g.size()})
    return r[r.n >= r.n.max() / 2]


def main() -> None:
    res = {}
    for arm, (run, prefix, col) in ARMS.items():
        t, r = trainer_log(run, prefix), rollouts(run, col)
        if t is None and r is None:
            continue
        res[arm] = {"trainer": None if t is None else
                    {k: {int(s): float(v) for s, v in t[k].dropna().items()} for k in TRAINER if k in t},
                    "rollouts": None if r is None else
                    {int(b): {k: float(v) for k, v in x.items()} for b, x in r.iterrows()}}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=0) + "\n")

    lines = ["# GRPO repetition-loop collapse", "",
             f"Trainer metrics averaged over the first and the last {BIN} logged steps; loop = a non-empty line repeated "
             f">= {LOOP_REPEATS} times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).", "",
             "| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, x in res.items():
        tr, ro = x["trainer"], x["rollouts"]
        if not tr:
            continue
        st = sorted(tr["entropy"])
        first, last = st[:BIN], st[-BIN:]

        def fl(k: str) -> str:
            v = tr.get(k, {})
            a = sum(v[s] for s in first if s in v) / max(1, sum(s in v for s in first))
            b = sum(v[s] for s in last if s in v) / max(1, sum(s in v for s in last))
            return f"{a:.3g} → {b:.3g}"
        lo = f"{ro[min(ro)]['loop']:.3f} → {ro[max(ro)]['loop']:.3f}" if ro else ""
        lines.append(f"| {arm} | {st[0]}–{st[-1]} | {fl(TRAINER[0])} | {fl(TRAINER[1])} | {fl(TRAINER[2])} | "
                     f"{fl(TRAINER[3])} | {lo} | {fl(TRAINER[4])} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    titles = {TRAINER[0]: "truncated at 2048 tokens (clipped ratio)", TRAINER[1]: "policy entropy",
              TRAINER[2]: "groups with zero reward variance", TRAINER[3]: "mean completion length (tokens)"}
    for ax, k in zip(axes.flat, TRAINER[:4]):
        for (arm, x), c in zip(res.items(), COLORS):
            if x["trainer"] and k in x["trainer"]:
                s = pd.Series(x["trainer"][k]).sort_index()
                ax.plot(s.index, s.rolling(SMOOTH, min_periods=1).mean(), color=c, label=arm)
        ax.set_title(f"{titles[k]}, {SMOOTH}-step mean")
    ax = axes[1, 1]
    for (arm, x), c in zip(res.items(), COLORS):
        if x["rollouts"]:
            s = sorted(x["rollouts"])
            ax.plot([b + BIN / 2 for b in s], [x["rollouts"][b]["loop"] for b in s], "o-", ms=3, color=c, label=arm)
    ax.set_title(f"training rollouts with a line repeated >= {LOOP_REPEATS}x (digits masked)")
    ax = axes[1, 2]
    for (arm, x), c in zip(res.items(), COLORS):
        if x["rollouts"]:
            s = sorted(x["rollouts"])
            ax.plot([b + BIN / 2 for b in s], [x["rollouts"][b]["reward"] for b in s], "o-", ms=3, color=c,
                    label=f"{arm}: {ARMS[arm][2]}")
    ax.set_title("training-rollout reward component (own prompt pools)")
    for ax in axes.flat:
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("GRPO with completions cut at 2048 tokens masked from the loss (TRL default): truncation, entropy, loops")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=130)


if __name__ == "__main__":
    main()
