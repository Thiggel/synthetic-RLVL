#!/usr/bin/env python
"""Stage-2 GRPO pilot curves: per-step reward components of each arm (all logged, whatever the primary reward).

Reads <run>/log_history.json when the run finished, else the trainer's printed log dicts in the job log.
Writes reports/figures/stage2_grpo_curves.png and analysis/stage2_grpo_curves.json (last-20-step means).
"""
import ast
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
# label -> (run dir, job log glob)
ARMS = {
    "G1 correct (p50)": ("2b_p50_G1_correct_bal", "grpo_2b_G1b_correct_6960.out"),
    "G5 lines (p50)": ("2b_p50_G5_lines_bal", "grpo_2b_G5G3_6964_lines.log"),
    "G3 gvc (p50)": ("2b_p50_G3_gvc_bal", "grpo_2b_G3c_gvc_6965.out"),
    "G0 correct (p0, no tag)": ("2b_p0_G0_correct_bal", "grpo_2b_G0b_correct_6962.out"),
}
COMP = ["correct", "grammatical", "valid", "in_system", "lines", "n_ok"]


def history(run, log):
    h = RUNS / run / "log_history.json"
    if h.exists():
        return [d for d in json.loads(h.read_text()) if "rewards/correct/mean" in d]
    p = REPO / "logs" / log
    if not p.exists():
        return []
    out = []
    for m in re.finditer(r"\{'loss'.*?\}(?=\n|$)", p.read_text(errors="ignore"), re.M):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        out.append({k: float(v) for k, v in d.items() if isinstance(v, (int, float, str)) and _num(v)})
    return out


def _num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def smooth(y, w=5):
    return [sum(y[max(0, i - w + 1): i + 1]) / len(y[max(0, i - w + 1): i + 1]) for i in range(len(y))]


def main():
    data = {k: history(*v) for k, v in ARMS.items()}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    keys = [(c, f"rewards/{c}/mean") for c in COMP] + [("completion length", "completions/mean_length"),
                                                     ("reward std (primary)", "reward_std")]
    summary = {}
    for ax, (name, key) in zip(axes.flat, keys):
        for arm, hs in data.items():
            ys = [d[key] for d in hs if key in d]
            if ys:
                ax.plot(range(1, len(ys) + 1), smooth(ys), label=f"{arm} ({len(ys)})")
        ax.set_title(name); ax.set_xlabel("GRPO step"); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=7)
    for arm, hs in data.items():
        if hs:
            tail = hs[-20:]
            summary[arm] = {"steps": len(hs), **{c: sum(d.get(k, 0) for d in tail) / len(tail) for c, k in keys}}
    fig.suptitle("Stage-2 GRPO pilots, 2B, balanced pool (5-step moving average; all components logged)")
    fig.tight_layout()
    fig.savefig(REPO / "reports/figures/stage2_grpo_curves.png", dpi=120)
    (REPO / "analysis/stage2_grpo_curves.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
