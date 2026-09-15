#!/usr/bin/env python3
"""Program-generated corpora for four further chain-propagation tasks.

The deduction corpus improves the benchmarks whose answer follows by carrying a
value along a dependency chain, and leaves everything else where it was. These
four generators keep that shape and vary what the value is, so that the
hypothesis can be tested where it predicts a gain today.

  trace     the value is a program variable and the step rule is assignment,
            arithmetic and a branch, which is the shape a coding benchmark has
  reach     the value is membership in a reachable set, and half the questions
            have no path, which is the case a forward chain cannot certify
  interval  the value is a position in a temporal order built by transitivity
  units     the value is a quantity with a unit, carried through conversions

Every item is solved by the generator itself, so the target is correct by
construction and no teacher model is involved. Each item is written in a formal
rendering and in a controlled English rendering of the same latent solution, as
in the deduction corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

VARS = list("abcdefghij")
NODES = [f"n{i}" for i in range(12)]
EVENTS = [f"e{i}" for i in range(10)]
UNITS = [("metre", "centimetre", 100), ("kilogram", "gram", 1000),
         ("hour", "minute", 60), ("litre", "millilitre", 1000)]


# --------------------------------------------------------------------------
# trace: the value is a program variable
# --------------------------------------------------------------------------
def gen_trace(rng: random.Random, steps: int) -> dict:
    names = rng.sample(VARS, k=min(4, len(VARS)))
    state: dict[str, int] = {}
    lines_formal, lines_nl, program = [], [], []
    target = names[0]
    state[target] = rng.randint(1, 9)
    program.append(f"{target} = {state[target]}")
    lines_formal.append(f"s1: {target} = {state[target]} [assign]")
    lines_nl.append(f"{target} starts at {state[target]}.")
    step = 2
    for _ in range(steps):
        kind = rng.choice(["arith", "arith", "copy", "branch"])
        if kind == "arith":
            op, k = rng.choice(["+", "-", "*"]), rng.randint(2, 9)
            before = state[target]
            state[target] = {"+": before + k, "-": before - k, "*": before * k}[op]
            program.append(f"{target} = {target} {op} {k}")
            lines_formal.append(f"s{step}: {target} = {before} {op} {k} = {state[target]} [arith]")
            lines_nl.append(f"Then {target} becomes {before} {op} {k}, which is {state[target]}.")
        elif kind == "copy":
            other = rng.choice([n for n in names if n != target])
            state[other] = state[target]
            program.append(f"{other} = {target}")
            lines_formal.append(f"s{step}: {other} = {target} = {state[other]} [assign]")
            lines_nl.append(f"Then {other} takes the value of {target}, which is {state[other]}.")
        else:
            threshold = rng.randint(1, 20)
            taken = state[target] > threshold
            k = rng.randint(1, 9)
            program.append(f"if {target} > {threshold}: {target} = {target} + {k}")
            lines_formal.append(
                f"s{step}: {target} > {threshold} is {'true' if taken else 'false'} [test]")
            lines_nl.append(
                f"Since {target} is {state[target]}, the test {target} > {threshold} is "
                f"{'true' if taken else 'false'}.")
            step += 1
            if taken:
                before = state[target]
                state[target] = before + k
                lines_formal.append(f"s{step}: {target} = {before} + {k} = {state[target]} [arith]")
                lines_nl.append(f"So {target} becomes {state[target]}.")
            else:
                lines_formal.append(f"s{step}: {target} = {state[target]} [unchanged]")
                lines_nl.append(f"So {target} is unchanged at {state[target]}.")
        step += 1
    body = "\n".join(program)
    prompt = (f"Trace this program and report the final value of {target}.\n\n"
              f"<program>\n{body}\n</program>\n\nWhat is the final value of {target}?")
    answer = str(state[target])
    formal = ("<trace>\n" + "\n".join(lines_formal) + f"\nconcl: {target} = {answer}\n</trace>\n"
              f"<answer> {answer} </answer>")
    nl = ("<think>\n" + "\n".join(lines_nl) + f"\nTherefore {target} is {answer}.\n</think>\n"
          f"<answer> {answer} </answer>")
    return dict(kind="trace", prompt=prompt, formal=formal, nl=nl, answer=answer)


# --------------------------------------------------------------------------
# reach: the value is membership in a reachable set, half the answers negative
# --------------------------------------------------------------------------
def gen_reach(rng: random.Random, n_edges: int, positive: bool) -> dict:
    nodes = NODES[:]
    rng.shuffle(nodes)
    half = len(nodes) // 2
    left, right = nodes[:half], nodes[half:]
    edges: list[tuple[str, str]] = []
    # two components, so a negative question has a certificate the model can write
    for group in (left, right):
        for i in range(len(group) - 1):
            edges.append((group[i], group[i + 1]))
        for _ in range(max(0, n_edges // 2 - (len(group) - 1))):
            a, b = rng.sample(group, 2)
            if (a, b) not in edges:
                edges.append((a, b))
    rng.shuffle(edges)
    succ: dict[str, list[str]] = {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)

    def closure(start: str) -> list[str]:
        seen, stack = [], [start]
        while stack:
            x = stack.pop()
            for y in succ.get(x, []):
                if y not in seen:
                    seen.append(y)
                    stack.append(y)
        return seen

    src = left[0]
    reachable = closure(src)
    if positive and reachable:
        dst = rng.choice(reachable)
    else:
        options = [n for n in right if n not in reachable]
        if not options:
            return gen_reach(rng, n_edges, positive=True)
        dst = rng.choice(options)
    answer = "yes" if dst in reachable else "no"
    edge_text = "\n".join(f"{a} -> {b}" for a, b in edges)
    prompt = (f"<edges>\n{edge_text}\n</edges>\n\n"
              f"Is there a path from {src} to {dst}? Answer yes or no.")
    if answer == "yes":
        # breadth-first, so a cycle cannot strand the walk part way
        prev: dict[str, str] = {}
        queue, seen = [src], {src}
        while queue:
            x = queue.pop(0)
            if x == dst:
                break
            for y in succ.get(x, []):
                if y not in seen:
                    seen.add(y)
                    prev[y] = x
                    queue.append(y)
        path, cur = [dst], dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        steps = [f"s{i + 1}: {path[i]} -> {path[i + 1]} [edge]" for i in range(len(path) - 1)]
        formal = ("<trace>\n" + "\n".join(steps) + f"\nconcl: {src} reaches {dst}\n</trace>\n"
                  "<answer> yes </answer>")
        nl = ("<think>\nFollow the edges from " + src + ": " + " then ".join(path[1:])
              + f".\nSo there is a path from {src} to {dst}.\n</think>\n<answer> yes </answer>")
    else:
        listed = ", ".join([src] + reachable)
        formal = (f"<trace>\ns1: closure({src}) = {{{listed}}} [closure]\n"
                  f"s2: {dst} not in closure({src}) [member]\n"
                  f"concl: {src} does not reach {dst}\n</trace>\n<answer> no </answer>")
        nl = (f"<think>\nStarting from {src} the nodes that can be reached are {listed}.\n"
              f"{dst} is not among them, so no path exists.\n</think>\n<answer> no </answer>")
    return dict(kind="reach", prompt=prompt, formal=formal, nl=nl, answer=answer)


# --------------------------------------------------------------------------
# interval: the value is a position in a temporal order
# --------------------------------------------------------------------------
def gen_interval(rng: random.Random, n_events: int, positive: bool) -> dict:
    events = EVENTS[:n_events]
    order = events[:]
    rng.shuffle(order)
    facts = [f"{order[i]} ends before {order[i + 1]} begins" for i in range(len(order) - 1)]
    extra = []
    for _ in range(rng.randint(1, 3)):
        i, j = sorted(rng.sample(range(len(order)), 2))
        if j - i > 1:
            extra.append(f"{order[i]} ends before {order[j]} begins")
    shown = facts + extra
    rng.shuffle(shown)
    i, j = sorted(rng.sample(range(len(order)), 2))
    a, b = (order[i], order[j]) if positive else (order[j], order[i])
    answer = "yes" if positive else "no"
    prompt = ("<facts>\n" + "\n".join(shown) + "</facts>\n\n"
              + f"Does {a} end before {b} begins? Answer yes or no.")
    chain = order[i:j + 1]
    steps = [f"s{k + 1}: {chain[k]} < {chain[k + 1]} [before]" for k in range(len(chain) - 1)]
    if positive:
        formal = ("<trace>\n" + "\n".join(steps) + f"\nconcl: {a} < {b}\n</trace>\n"
                  "<answer> yes </answer>")
        nl = ("<think>\n" + " ".join(f"{chain[k]} ends before {chain[k+1]} begins."
                                     for k in range(len(chain) - 1))
              + f"\nSo {a} ends before {b} begins.\n</think>\n<answer> yes </answer>")
    else:
        formal = ("<trace>\n" + "\n".join(steps) + f"\ns{len(steps)+1}: {b} < {a} [chain]\n"
                  f"concl: not ({a} < {b})\n</trace>\n<answer> no </answer>")
        nl = ("<think>\n" + " ".join(f"{chain[k]} ends before {chain[k+1]} begins."
                                     for k in range(len(chain) - 1))
              + f"\nThat puts {b} before {a}, so {a} does not end before {b} begins.\n"
              "</think>\n<answer> no </answer>")
    return dict(kind="interval", prompt=prompt, formal=formal, nl=nl, answer=answer)


# --------------------------------------------------------------------------
# units: the value is a quantity carried through conversions
# --------------------------------------------------------------------------
def gen_units(rng: random.Random, steps: int) -> dict:
    """Scale a quantity, then convert it once at the end.

    The conversion has to come last, because the prompt lists only the scaling
    operations and an interleaved conversion would leave the answer
    underdetermined. Divisions are drawn so that they divide exactly, so the
    order of the scalings cannot matter either.
    """
    big, small, factor = rng.choice(UNITS)
    first = value = rng.randint(2, 20)
    ops: list[str] = []
    lines_f, lines_n = [], []
    step = 1
    for _ in range(steps):
        if rng.random() < 0.5:
            k = rng.randint(2, 6)
            before, value = value, value * k
            ops.append(f"times by {k}")
            lines_f.append(f"s{step}: {before} {big} * {k} = {value} {big} [scale]")
            lines_n.append(f"{before} {big} times {k} is {value} {big}.")
        else:
            divisors = [d for d in (2, 3, 4, 5, 6) if value % d == 0]
            if not divisors:
                continue
            k = rng.choice(divisors)
            before, value = value, value // k
            ops.append(f"divided by {k}")
            lines_f.append(f"s{step}: {before} {big} / {k} = {value} {big} [scale]")
            lines_n.append(f"{before} {big} divided by {k} is {value} {big}.")
        step += 1
    converted = value * factor
    lines_f.append(f"s{step}: {value} {big} = {converted} {small} [convert]")
    lines_n.append(f"{value} {big} is {converted} {small}.")
    prompt = (f"Start with {first} {big}, then apply each operation in order.\n"
              f"<operations>\n" + "\n".join(ops) + "\n</operations>\n\n"
              f"What is the result, expressed in {small}?")
    answer = f"{converted} {small}"
    formal = ("<trace>\n" + "\n".join(lines_f) + f"\nconcl: {answer}\n</trace>\n"
              f"<answer> {answer} </answer>")
    nl = ("<think>\n" + " ".join(lines_n) + f"\nSo the result is {answer}.\n</think>\n"
          f"<answer> {answer} </answer>")
    return dict(kind="units", prompt=prompt, formal=formal, nl=nl, answer=answer)


GENERATORS = {
    "trace": lambda rng: gen_trace(rng, rng.randint(4, 12)),
    "reach": lambda rng: gen_reach(rng, rng.randint(8, 14), rng.random() < 0.5),
    "interval": lambda rng: gen_interval(rng, rng.randint(5, 9), rng.random() < 0.5),
    "units": lambda rng: gen_units(rng, rng.randint(3, 7)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--kinds", nargs="+", default=sorted(GENERATORS))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as handle:
        for i in range(args.n):
            kind = args.kinds[i % len(args.kinds)]
            row = GENERATORS[kind](rng)
            counts[kind] = counts.get(kind, 0) + 1
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(dict(out=str(args.out), n=args.n, counts=counts, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
