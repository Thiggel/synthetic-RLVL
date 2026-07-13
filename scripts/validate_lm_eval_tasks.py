#!/usr/bin/env python
"""Validate that lm-eval tasks resolve and their datasets construct."""

from __future__ import annotations

import argparse
import gc

from lm_eval.tasks import TaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Comma- or space-separated task/group names.")
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Additional lm-eval task directory. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [task for task in args.tasks.replace(",", " ").split() if task]
    if not tasks:
        raise SystemExit("No lm-eval tasks were provided.")

    manager = TaskManager(include_path=args.include_path or None)
    for task in tasks:
        if not manager.match_tasks([task]):
            raise SystemExit(f"lm-eval task/group is unavailable: {task}")
        loaded = manager.load_task_or_group(task)
        print(f"Validated lm-eval task/group {task}: {len(loaded)} expanded task(s)", flush=True)
        del loaded
        gc.collect()


if __name__ == "__main__":
    main()
