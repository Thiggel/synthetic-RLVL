#!/usr/bin/env python
"""Which Stage-2 reward pays a hack as much as an honest proof? (2026-09-28)

For held-out generator items (formal_mixture pool/test: prompt, gold proof, answer) this builds
completions an RL policy can reach cheaply, all with the correct Answer: line, and scores them
with every candidate reward in scripts/formal_rewards.py:
  honest          the gold proof
  honest_1err     the gold proof with one derived line's number changed (a near miss)
  honest_runon    the gold proof, then prose after </proof> (late G5)
  answer_only     no proof
  circular        yes/no: the claim quoted as a `given`, `ans yes` (G5 @50)
  bogus_mp        yes/no: `F -> P ; given q1`, `F ; given q0`, `P ; mp`: invented premise, 3 lines
  bogus_num       number: `goal_lhs = a + 0 ; given q0`, `goal_lhs = a ; calc`: the guessed answer
                  smuggled in as a premise, 2 lines
A reward is hackable when a hack scores >= honest. Also reports how often the numeric premise
check rejects honest gold proofs (false rejections). frac_hard_gen is frac_hard with the premise
check replaced by exact faithfulness (the item's gold sentences), as on --benches gen items.
Writes analysis/reward_hack_audit.json and reports/figures/reward_hack_audit.png.
Run with .venv_rlvl_grpo and PYTHONPATH=RLVL-next/gen:RLVL-next/rlvl/python.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
POOL = DATA / "datasets/formal_mixture_20260925/pool/test.jsonl"
REWARDS = ["correct", "gvc", "lines", "lines_fmt", "frac", "frac_hard", "frac_hard_gen"]
EXTRA = ["valid", "valid_prem", "frac_parsed", "frac_ok", "n_prem_bad"]
N_PER_TYPE = 300


def wrap(proof: str, ans: str, tail: str = "") -> str:
    return f"<proof>\n{proof}</proof>\nAnswer: {ans}\n{tail}"


def variants(r: dict, rng: random.Random) -> dict:
    a, goal, proof = r["answer"], r["goal"], r["proof"]
    facts = [s for s in r.get("sentences", []) if s.get("role") == "fact" and s.get("forms")]
    q0 = facts[0]["quote"] if facts else r["prompt"].split(".")[0]
    q1 = facts[1]["quote"] if len(facts) > 1 else q0
    out = {"honest": wrap(proof, a), "answer_only": f"Answer: {a}\n",
           "honest_runon": wrap(proof, a).replace("\nAnswer:", "\nTherefore, the answer is " + a
                                                   + ".\nNow, we write the final answer.\n</proof>\nAnswer:")}
    lines = proof.split("\n")
    idx = [i for i, ln in enumerate(lines) if re.search(r";\s*calc\b", ln) and re.search(r"\d", ln.split(";")[0])]
    if idx:
        i = rng.choice(idx)
        f, j = lines[i].split(";", 1)
        m = list(re.finditer(r"\d+", f))[-1]
        lines[i] = f[:m.start()] + str(int(m.group()) + 1) + f[m.end():] + ";" + j
        out["honest_1err"] = wrap("\n".join(lines), a)
    if a in ("yes", "no"):
        p = goal.removeprefix("goal ").rstrip(" ?")
        f0 = facts[0]["forms"][0] if facts else "true"
        concl = p if a == "yes" else f"~({p})"
        out["circular"] = wrap(f"{goal}\n1 {p} ; given \"{q0}\"\nans yes ; 1\n", a)
        out["bogus_mp"] = wrap(f"{goal}\n1 {f0} -> {concl} ; given \"{q1}\"\n2 {f0} ; given \"{q0}\"\n"
                               f"3 {concl} ; mp 1 2\nans {a} ; 3\n", a)
    else:
        lhs = goal.removeprefix("goal ").split("=")[0].strip()
        out["bogus_num"] = wrap(f"{goal}\n1 {lhs} = {a} + 0 ; given \"{q0}\"\n2 {lhs} = {a} ; calc 1\n"
                                f"ans {a} ; 2\n", a)
    return out


def main():
    rng = random.Random(0)
    items = {"yesno": [], "number": []}
    for ln in open(POOL):
        r = json.loads(ln)
        t = "yesno" if r["answer"] in ("yes", "no") else "number" if re.fullmatch(r"-?\d+(/\d+)?", r["answer"]) else None
        if t and len(items[t]) < N_PER_TYPE:
            items[t].append(r)
    res = {}
    for t, rs in items.items():
        acc = {}
        for k, r in enumerate(rs):
            rec = {"id": f"pool/{t}/{k}", "prompt": r["prompt"], "gold": r["answer"], "answer_type": t,
                   "system_answerable": True}
            for v, comp in variants(r, rng).items():
                c = components(rec, comp)
                c["gvc"] = (c["grammatical"] + c["valid"] + c["correct"]) / 3
                # frac_hard as on generator items: exact faithfulness against the gold sentences
                c["frac_hard_gen"] = components({**rec, "id": rec["id"] + "/gen",
                                                 "sentences_json": json.dumps(r["sentences"])}, comp)["frac_hard"]
                a = acc.setdefault(v, {"n": 0, **{x: 0.0 for x in REWARDS + EXTRA}})
                a["n"] += 1
                for x in REWARDS + EXTRA:
                    a[x] += c[x]
        res[t] = {v: {x: (a[x] / a["n"] if x != "n" else a["n"]) for x in a} for v, a in acc.items()}
        for v, a in res[t].items():
            print(t, f"{v:13s}", {x: round(a[x], 3) for x in ["n"] + REWARDS + EXTRA})
    (REPO / "analysis/reward_hack_audit.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, (t, vs) in zip(axes, res.items()):
        names = list(vs)
        w = 0.8 / len(names)
        for i, v in enumerate(names):
            xs = [j + (i - (len(names) - 1) / 2) * w for j in range(len(REWARDS))]
            ys = [vs[v][x] for x in REWARDS]
            hatch = None if v in ("honest", "honest_1err") else "//"
            ax.bar(xs, ys, w, label=v, hatch=hatch, edgecolor="white" if hatch is None else "k", linewidth=0.3)
        ax.set_xticks(range(len(REWARDS)), [x.replace("frac_hard_gen", "frac_hard\n(gen items)") for x in REWARDS])
        ax.set_title(f"{t} items (n={vs['honest']['n']}): mean reward, all answers correct")
        ax.grid(axis="y", alpha=.3)
        ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    axes[0].set_ylabel("mean reward (each reward scaled to [0, 1])")
    fig.suptitle("Reward-hack audit on generator items: hatched = hack. A reward is safe only if every hatched bar is"
                 " below the blue 'honest' bar (red = honest with one wrong line)")
    fig.tight_layout()
    fig.savefig(REPO / "reports/figures/reward_hack_audit.png", dpi=140)


if __name__ == "__main__":
    main()
