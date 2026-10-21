"""W4.1 prep (RLVL-next PLAN): AZ eval-only baseline of tl_mid4 (fresh value head) on the first 200 rows of the v1 and r3
TL evals, guided (full) and free decoding, greedy (sims 0) vs MCTS (sims 64). Search backs up the verifier reward (gold
answer) at terminal nodes, so sims 64 is oracle-assisted search, not a deployable accuracy; sims 0 is what AZ should raise."""
import json
import pathlib
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

A = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals/az0/mid4")
D = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/datasets")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_az0.md")
SETS = {"v1": D / "tl_mid_20261009/tl_eval.jsonl", "r3": D / "tl_eval_r3_20261010/tl_eval.jsonl"}
CELLS = [(s, g) for s in SETS for g in ("full", "free")]


def boot(a, b, n=2000, seed=0):
    """Paired bootstrap 95% CI of mean(a) - mean(b)."""
    rng, d = random.Random(seed), [x - y for x, y in zip(a, b)]
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


fam = {s: {r["example_id"]: r["family"] for r in map(json.loads, open(p))} for s, p in SETS.items()}
log = {c: json.loads(open(A / f"{c[0]}_{c[1]}" / "log.jsonl").readline())["eval"] for c in CELLS}
rows = {(c, k): {r["example_id"]: r for r in map(json.loads, open(A / f"{c[0]}_{c[1]}" / f"eval_it0_{k}.jsonl"))}
        for c in CELLS for k in ("sims0", "sims64")}

lines = ["# W4.1 prep: AZ eval-only baseline of tl_mid4 (fresh value head, n = 200 per cell, reward mul)", "",
         "| set | decoding | sims | r (valid ∧ correct) | valid | correct | dead end | tokens | decisions | s / problem | sims64 − sims0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for c in CELLS:
    ids = sorted(rows[(c, "sims0")])
    d = boot([rows[(c, "sims64")][i]["r"] for i in ids], [rows[(c, "sims0")][i]["r"] for i in ids])
    for k in ("sims0", "sims64"):
        v = log[c][k]
        dd = "{:+.3f} [{:+.3f}, {:+.3f}]".format(*d) if k == "sims64" else "–"
        lines.append(f"| {c[0]} | {'guided' if c[1] == 'full' else 'free'} | {k[4:]} | {v['r']:.3f} | {v['valid']:.3f} | "
                     f"{v['correct']:.3f} | {v['dead_end']:.3f} | {v['tokens']:.0f} | {v['decisions']:.0f} | "
                     f"{v['seconds'] / v['n']:.1f} | {dd} |")
lines += ["", "## Greedy (sims 0) failures and what search does with them", "",
          "| set | decoding | family | n | sims0 r | sims64 r | sims0 failures: valid wrong / invalid right / invalid wrong / dead end |",
          "|---|---|---|---|---|---|---|"]
for c in CELLS:
    r0, r64 = rows[(c, "sims0")], rows[(c, "sims64")]
    for f in sorted({fam[c[0]][i] for i in r0}):
        ids = [i for i in r0 if fam[c[0]][i] == f]
        bad = [r0[i] for i in ids if r0[i]["r"] < 1]
        if not bad:
            continue
        k = (sum(b["valid"] and not b["correct"] for b in bad), sum(not b["valid"] and b["correct"] for b in bad),
             sum(not b["valid"] and not b["correct"] for b in bad), sum(b["end"] == "dead_end" for b in bad))
        lines.append(f"| {c[0]} | {'guided' if c[1] == 'full' else 'free'} | {f} | {len(ids)} | "
                     f"{sum(r0[i]['r'] for i in ids) / len(ids):.3f} | {sum(r64[i]['r'] for i in ids) / len(ids):.3f} | "
                     f"{' / '.join(map(str, k))} |")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

lab = [f"{s}\n{'guided' if g == 'full' else 'free'}" for s, g in CELLS]
x = range(len(CELLS))
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ax = axs[0]
for j, (k, col) in enumerate((("sims0", "#8da0cb"), ("sims64", "#fc8d62"))):
    ys = [log[c][k]["r"] for c in CELLS]
    ax.bar([i + (j - 0.5) * 0.38 for i in x], ys, 0.38, color=col, edgecolor="k", lw=0.4,
           label="greedy (sims 0)" if k == "sims0" else "MCTS 64 sims (verifier at terminals)")
    for i, y in zip(x, ys):
        ax.text(i + (j - 0.5) * 0.38, y + 0.003, f"{y:.3f}", ha="center", fontsize=7)
ax.set_xticks(list(x), lab)
ax.set_ylim(0.85, 1.03)
ax.set_ylabel("r = valid ∧ correct")
ax.set_title("mid4, fresh value head, n = 200 per cell")
ax.legend(fontsize=8, loc="lower left")
ax = axs[1]
for j, (k, col) in enumerate((("sims0", "#8da0cb"), ("sims64", "#fc8d62"))):
    ys = [log[c][k]["seconds"] / log[c][k]["n"] for c in CELLS]
    ax.bar([i + (j - 0.5) * 0.38 for i in x], ys, 0.38, color=col, edgecolor="k", lw=0.4)
    for i, y in zip(x, ys):
        ax.text(i + (j - 0.5) * 0.38, y * 1.08, f"{y:.1f}", ha="center", fontsize=7)
ax.set_yscale("log")
ax.set_xticks(list(x), lab)
ax.set_ylabel("wall seconds / problem (1 L40, 64 parallel)")
ax.set_title("cost (free: every token is a decision)")
fig.suptitle("W4.1 prep: guided greedy is saturated; free greedy leaves 2.5–8.5 points that search recovers")
fig.tight_layout()
fig.savefig(FIG / "tl_az0.png", dpi=150)
