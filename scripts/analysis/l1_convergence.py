#!/usr/bin/env python
"""L1 convergence study (user question, 2026-09-30): GRPO with the reward correct x valid x faithful premises
(cvf) vs correctness only, both from the same SFT policy (p50 + lemma catalog, fp32 master, tagged), up to
5000 steps. When does each arm converge in correctness and in validity, and at what value?
The cvf arm switched to cvf_fmt = cvf x format_ok at step 51 (chain 9647, resumed from checkpoint-50):
under plain cvf, G10 learned to drop or garble the `Answer:` line, which can only cost reward
(reports/2026-09-28_stage2_gate.md, 2026-10-01). Steps 1-50 got identical rewards under both.

Training curves: per-step reward components on the training prompts (T=1.0, 32 prompts x 8 rollouts),
split by domain (gen = generator items, dolci = dolci_math/wordprob/yesno; formal_rewards.BENCH_GROUPS).
They come from <run>/log_history.json once a run finished, else from the newest checkpoint's
trainer_state.json (its log_history survives the pruning), so they lag the run by up to 50 steps.
Convergence, per arm, domain and metric:
  - fit: y(t) = a - (a - y0) exp(-t / tau) on the 25-step means (grid search over tau, least squares for a
    and y0). Reports the asymptote a and t95 = 3 tau, the step where 95% of the change is done.
    If t95 is past the last logged step, the fit extrapolates and the curve has not converged yet.
  - empirical: plateau = mean of the last 250 steps; t_plateau = the first step after which the 100-step
    moving average stays within 0.02 of it. Only reported once the run has >= 1000 steps.
Held-out: greedy evals of the kept checkpoints (every 250 steps; scripts/submit_l1_gates.sh): the Dolci gate
(950 items; the 700 dolci_math/wordprob/yesno items match the training benches, dapo/knowledge are
out-of-domain) and the in-domain generator test (the 1837 tool-free items of 2000). Both are rescored here
with formal_rewards.components, i.e. with the training rewards. Step 0 is the SFT policy.

Run with .venv_rlvl_grpo and the frozen checker the runs were trained with:
  S=/vol/tmp2/laitenbf/rlvl_data/checker_snapshot_pre_libext_20260930 PYTHONPATH=$S/gen:$S/rlvl_python
Writes analysis/l1_convergence.json and reports/figures/l1_convergence_{train,heldout}.{png,pdf}.
Held-out rescores are cached in <ckpt>/l1_rescore.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
RUNS = DATA / "grpo_formal_20260928"
BASE = DATA / "formal_mixture_sft_20260925/qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407"
GATE_TEST = DATA / "datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
GEN_TEST = DATA / "datasets/formal_mixture_20260925/pool/test.jsonl"
ARMS = {"correct only": "L1_correct", "cvf (x format_ok from step 51)": "L1_cvf"}
COLORS = {"correct only": "C0", "cvf (x format_ok from step 51)": "C3"}
METRICS = ["correct", "valid", "cvf"]
DOMAINS = {"gen": "generator prompts", "dolci": "Dolci prompts (math, wordprob, yesno)"}
TRAIN_BENCHES = {"dolci_math", "dolci_wordprob", "dolci_yesno"}
BIN, MA, TOL, TAIL = 25, 100, 0.02, 250
OUT_JSON = REPO / "analysis/l1_convergence.json"
OUT_FIG = REPO / "reports/figures/l1_convergence"


def history(run: str) -> list[dict]:
    d = RUNS / run
    if (d / "log_history.json").is_file():
        h = json.loads((d / "log_history.json").read_text())
    else:
        states = sorted(d.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        if not states:
            return []
        h = json.loads(states[-1].read_text())["log_history"]
    return [e for e in h if "step" in e and "rewards/correct/mean" in e]


def series(h: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(e["step"], e[key]) for e in h if e.get(key) is not None and not math.isnan(e[key])]
    if not pts:
        return np.array([]), np.array([])
    s, y = map(np.array, zip(*pts))
    return s.astype(float), y.astype(float)


def binned(s: np.ndarray, y: np.ndarray, w: int = BIN) -> tuple[np.ndarray, np.ndarray]:
    b = ((s - 1) // w).astype(int)
    ks = np.unique(b)
    return np.array([s[b == k].mean() for k in ks]), np.array([y[b == k].mean() for k in ks])


def fit_exp(t: np.ndarray, y: np.ndarray) -> dict | None:
    """Saturating exponential y = a - (a - y0) exp(-t/tau); linear in (a, y0) for fixed tau. The asymptote of a
    rate is kept in [0, 1] (a collapse to 0 would otherwise extrapolate to a negative rate)."""
    if len(t) < 6:
        return None
    best = None
    for tau in np.geomspace(5, 50_000, 400):
        e = np.exp(-t / tau)
        X = np.stack([1 - e, e], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if not 0 <= coef[0] <= 1:
            a = min(max(coef[0], 0.0), 1.0)
            coef = np.array([a, float(e @ (y - a * (1 - e)) / max(float(e @ e), 1e-12))])
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, *coef)
    sse, tau, a, y0 = best
    r2 = 1 - sse / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    return {"a": float(a), "y0": float(y0), "tau": float(tau), "t95": float(3 * tau), "r2": float(r2),
            "converged": bool(3 * tau <= t[-1])}


def plateau(s: np.ndarray, y: np.ndarray) -> dict | None:
    if len(s) == 0 or s[-1] < 1000:
        return None
    level = float(y[s > s[-1] - TAIL].mean())
    ma = np.convolve(y, np.ones(MA) / MA, mode="valid")
    ma_s = s[MA - 1:]
    off = np.nonzero(np.abs(ma - level) > TOL)[0]
    t = float(ma_s[0]) if len(off) == 0 else float(ma_s[off[-1] + 1]) if off[-1] + 1 < len(ma_s) else None
    return {"level": level, "t_plateau": t}


def rescored(ckpt: Path, gate: Path, formal: Path, tests: tuple[dict, dict]) -> dict | None:
    """Mean training-reward components of a checkpoint's greedy held-out generations (cached)."""
    cache = ckpt / "l1_rescore.json"
    gf, ff = gate / "generations.jsonl", formal / "generations.jsonl"
    gate_test, gen_test = tests
    if not (gf.is_file() and (gf.parent / "summary.json").is_file()
            and ff.is_file() and (ff.parent / "summary.json").is_file()):
        return None
    if cache.is_file() and cache.stat().st_mtime > max(gf.stat().st_mtime, ff.stat().st_mtime):
        return json.loads(cache.read_text())
    out = {}
    sums: dict[str, dict] = {}
    for ln in open(gf):
        g = json.loads(ln)
        c = components(gate_test[g["id"]], g["generation"])
        grp = "gate_train_benches" if g["bench"] in TRAIN_BENCHES else "gate_ood"
        for k in (grp, "gate_all"):
            acc = sums.setdefault(k, {m: 0.0 for m in METRICS + ["n"]})
            _add(acc, c)
    for ln in open(ff):
        g = json.loads(ln)
        rec = gen_test.get(g["id"])
        if rec is None:  # tool items and non-number/yesno answers: not in the GRPO pool
            continue
        _add(sums.setdefault("gen_test", {m: 0.0 for m in METRICS + ["n"]}), components(rec, g["generation"]))
    for k, acc in sums.items():
        out[k] = {m: acc[m] / acc["n"] for m in METRICS} | {"n": int(acc["n"])}
    summ = json.loads((ff.parent / "summary.json").read_text())["overall"]
    out["gen_test_eval"] = {k: summ[k] for k in ("valid", "answer_acc", "faithful", "grammatical", "n")}
    cache.write_text(json.dumps(out, indent=1))
    return out


def _add(acc: dict, c: dict) -> None:
    acc["correct"] += c["correct"]
    acc["valid"] += c["valid"]
    acc["cvf"] += c["correct"] * c["valid"] * c["prem_ok"]
    acc["n"] += 1


def load_tests() -> tuple[dict, dict]:
    gate = {json.loads(ln)["id"]: json.loads(ln) for ln in open(GATE_TEST)}
    gen = {}
    for ln in open(GEN_TEST):
        r = json.loads(ln)
        a = r["answer"]
        t = "yesno" if a in ("yes", "no") else "number" if re.fullmatch(r"-?\d+(/\d+)?", a) else None
        if t is None or r.get("tools"):
            continue
        gen[r["id"]] = {"id": r["id"], "prompt": r["prompt"], "gold": a, "answer_type": t,
                        "system_answerable": True, "sentences_json": json.dumps(r["sentences"])}
    return gate, gen


def main() -> None:
    tests = load_tests()
    res: dict = {"train": {}, "heldout": {}}
    hist = {arm: history(run) for arm, run in ARMS.items()}

    for arm, h in hist.items():
        r = res["train"][arm] = {"steps": int(h[-1]["step"]) if h else 0}
        for dom in DOMAINS:
            for m in METRICS:
                s, y = series(h, f"rewards/{m}_{dom}/mean")
                if len(s):
                    bt, by = binned(s, y)
                    r[f"{m}_{dom}"] = {"fit": fit_exp(bt, by), "plateau": plateau(s, y),
                                       "last100": float(y[s > s[-1] - 100].mean()),
                                       "first25": float(y[s <= 25].mean())}

    # the SFT policy: gate under final/, in-domain eval on the run dir (eval_formal_vllm picks final/)
    base = rescored(BASE / "final", BASE / "final/rl_gate_dolci", BASE / "formal_eval", tests)
    for arm, run in ARMS.items():
        pts = [(0, base)] if base else []
        d = RUNS / run
        for c in sorted(list(d.glob("checkpoint-*")) + [d / "final"], key=_step_of):
            if c.is_dir():
                v = rescored(c, c / "rl_gate_dolci", c / "formal_eval", tests)
                if v:
                    pts.append((_step_of(c), v))
        res["heldout"][arm] = pts
    OUT_JSON.write_text(json.dumps(res, indent=1))
    plot_train(hist, res)
    plot_heldout(res)
    report(res)


def _step_of(c: Path) -> int:
    if c.name == "final":
        st = c / "trainer_state.json"
        return int(json.loads(st.read_text())["global_step"]) if st.is_file() else 10**9
    return int(c.name.split("-")[1])


def plot_train(hist: dict, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for i, dom in enumerate(DOMAINS):
        for j, m in enumerate(METRICS):
            ax = axes[i, j]
            for arm, h in hist.items():
                s, y = series(h, f"rewards/{m}_{dom}/mean")
                if not len(s):
                    continue
                col = COLORS[arm]
                bt, by = binned(s, y)
                ax.plot(bt, by, color=col, alpha=.35, lw=1)
                if len(y) >= MA:
                    ma = np.convolve(y, np.ones(MA) / MA, mode="valid")
                    ax.plot(s[MA - 1:], ma, color=col, lw=2, label=f"{arm}: {MA}-step mean")
                f = res["train"][arm].get(f"{m}_{dom}", {}).get("fit")
                if f:
                    tt = np.linspace(0, max(s[-1], min(f["t95"], 3 * s[-1])), 300)
                    ax.plot(tt, f["a"] - (f["a"] - f["y0"]) * np.exp(-tt / f["tau"]), "--", color=col, lw=1,
                            label=f"fit: a={f['a']:.3f}, t95={f['t95']:.0f}" + ("" if f["converged"] else " (extrap.)"))
                    ax.axhline(f["a"], color=col, ls=":", lw=.8)
                    if f["converged"]:
                        ax.axvline(f["t95"], color=col, ls=":", lw=.8)
            ax.set_title(f"{m} — {DOMAINS[dom]}", fontsize=10)
            ax.set_xlabel("GRPO step")
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
    diag = [("completions/mean_length", "mean completion length (tokens)"),
            ("frac_reward_zero_std", "fraction of groups with zero reward std"),
            ("rewards/has_proof/mean", "has_proof (all prompts)")]
    for ax, (key, title) in zip(axes[2], diag):
        for arm, h in hist.items():
            s, y = series(h, key)
            if len(s):
                bt, by = binned(s, y)
                ax.plot(bt, by, color=COLORS[arm], label=arm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("L1: GRPO reward cvf vs correct only (2B, p50 + lemma catalog SFT, T=1.0, 32x8 rollouts/step)\n"
                 f"training-prompt rewards, {BIN}-step means (faint), {MA}-step moving average; dashed: saturating "
                 "exponential fit, dotted: its asymptote and t95", fontsize=11)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}_train.{ext}", dpi=130)
    plt.close(fig)


def plot_heldout(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("gen_test", "in-domain generator test (1837 items)"),
              ("gate_train_benches", "Dolci gate, training benches (700 items)"),
              ("gate_ood", "Dolci gate, dapo + knowledge (250 items, OOD)")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, (g, gname) in enumerate(groups):
        for j, m in enumerate(METRICS):
            ax = axes[i, j]
            for arm, pts in res["heldout"].items():
                xs = [s for s, v in pts if g in v]
                ys = [v[g][m] for s, v in pts if g in v]
                if xs:
                    ax.plot(xs, ys, "o-", color=COLORS[arm], label=arm, ms=4)
            ax.set_title(f"{m} — {gname}", fontsize=10)
            ax.set_xlabel("GRPO step (0 = SFT policy)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
    fig.suptitle("L1 held-out (greedy), rescored with the training rewards (frozen pre-libext checker)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}_heldout.{ext}", dpi=130)
    plt.close(fig)


def report(res: dict) -> None:
    print("| arm | metric | steps | first 25 | last 100 | fit asymptote | t95 | converged | plateau | t_plateau |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm, r in res["train"].items():
        for dom in DOMAINS:
            for m in METRICS:
                v = r.get(f"{m}_{dom}")
                if not v:
                    continue
                f, p = v["fit"] or {}, v["plateau"] or {}
                print(f"| {arm} | {m}_{dom} | {r['steps']} | {v['first25']:.3f} | {v['last100']:.3f} | "
                      f"{f.get('a', float('nan')):.3f} | {f.get('t95', float('nan')):.0f} | {f.get('converged', '')} | "
                      f"{p.get('level', float('nan')):.3f} | {p.get('t_plateau', '')} |")
    for arm, pts in res["heldout"].items():
        for s, v in pts:
            print(arm, s, {g: {m: round(x[m], 3) for m in METRICS} for g, x in v.items() if g != "gen_test_eval"})


if __name__ == "__main__":
    main()
