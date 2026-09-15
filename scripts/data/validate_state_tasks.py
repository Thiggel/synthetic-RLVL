#!/usr/bin/env python3
"""Re-derive the answer of every generated item from its prompt alone.

The generator writes the trace and the answer from the same internal state, so
agreement between them proves nothing. This script parses the problem out of
the prompt and solves it independently, which is what makes "correct by
construction" a checked claim instead of an assumption.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def solve_trace(prompt: str) -> str:
    body = re.search(r"<program>\n(.*?)\n</program>", prompt, re.S).group(1)
    state: dict[str, int] = {}
    for line in body.splitlines():
        line = line.strip()
        m = re.fullmatch(r"if (\w+) > (-?\d+): (\w+) = (\w+) ([-+*]) (-?\d+)", line)
        if m:
            var, thr, dst, src, op, k = m.groups()
            if state[var] > int(thr):
                state[dst] = {"+": state[src] + int(k), "-": state[src] - int(k),
                              "*": state[src] * int(k)}[op]
            continue
        m = re.fullmatch(r"(\w+) = (-?\d+)", line)
        if m:
            state[m.group(1)] = int(m.group(2))
            continue
        m = re.fullmatch(r"(\w+) = (\w+)", line)
        if m:
            state[m.group(1)] = state[m.group(2)]
            continue
        m = re.fullmatch(r"(\w+) = (\w+) ([-+*]) (-?\d+)", line)
        if m:
            dst, src, op, k = m.groups()
            state[dst] = {"+": state[src] + int(k), "-": state[src] - int(k),
                          "*": state[src] * int(k)}[op]
            continue
        raise ValueError(f"unparsed program line: {line}")
    target = re.search(r"final value of (\w+)\?", prompt).group(1)
    return str(state[target])


def solve_reach(prompt: str) -> str:
    edges = re.search(r"<edges>\n(.*?)\n</edges>", prompt, re.S).group(1)
    succ: dict[str, list[str]] = {}
    for line in edges.splitlines():
        a, b = line.split(" -> ")
        succ.setdefault(a.strip(), []).append(b.strip())
    src, dst = re.search(r"path from (\w+) to (\w+)\?", prompt).groups()
    seen, stack = set(), [src]
    while stack:
        x = stack.pop()
        for y in succ.get(x, []):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return "yes" if dst in seen else "no"


def solve_interval(prompt: str) -> str:
    facts = re.search(r"<facts>\n(.*?)</facts>", prompt, re.S).group(1)
    before: dict[str, set] = {}
    for line in facts.strip().splitlines():
        m = re.fullmatch(r"(\w+) ends before (\w+) begins", line.strip())
        a, b = m.groups()
        before.setdefault(a, set()).add(b)
    changed = True
    while changed:
        changed = False
        for a in list(before):
            for b in list(before[a]):
                for c in before.get(b, set()):
                    if c not in before[a]:
                        before[a].add(c)
                        changed = True
    a, b = re.search(r"Does (\w+) end before (\w+) begins\?", prompt).groups()
    return "yes" if b in before.get(a, set()) else "no"


def solve_units(prompt: str) -> str:
    start = re.search(r"Start with (\d+) (\w+)", prompt)
    value, unit = int(start.group(1)), start.group(2)
    target = re.search(r"expressed in (\w+)\?", prompt).group(1)
    pairs = {"metre": ("centimetre", 100), "kilogram": ("gram", 1000),
             "hour": ("minute", 60), "litre": ("millilitre", 1000)}
    ops = re.search(r"<operations>\n(.*?)\n</operations>", prompt, re.S).group(1)
    for line in ops.strip().splitlines():
        m = re.fullmatch(r"(times|divided) by (\d+)", line.strip())
        op, k = m.group(1), int(m.group(2))
        value = value * k if op == "times" else value // k
    if unit != target:
        small, factor = pairs[unit]
        if small != target:
            raise ValueError(f"unexpected target unit {target}")
        value *= factor
    return f"{value} {target}"


SOLVERS = {"trace": solve_trace, "reach": solve_reach,
           "interval": solve_interval, "units": solve_units}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ok = bad = 0
    failures: list[dict] = []
    for i, line in enumerate(args.corpus.open(encoding="utf-8")):
        if args.limit and i >= args.limit:
            break
        row = json.loads(line)
        try:
            got = SOLVERS[row["kind"]](row["prompt"])
        except Exception as exc:  # a parse failure is a generator bug too
            got = f"error: {exc}"
        if got == row["answer"]:
            ok += 1
        else:
            bad += 1
            if len(failures) < 5:
                failures.append(dict(kind=row["kind"], expected=row["answer"], got=got))
    print(json.dumps(dict(checked=ok + bad, agree=ok, disagree=bad, examples=failures), indent=2))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
