#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "lm_eval_tasks" / "synthrlvl_ood"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from utils import extract_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print raw and extracted answers from synthrlvl OOD lm-eval samples.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    seen = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            files = sorted(path.rglob("samples_*.jsonl"))
        else:
            files = [path]
        for file in files:
            print(f"### {file}")
            strict_qa = "longbench" in file.name
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    resps = row.get("resps") or []
                    raw = ""
                    if resps and isinstance(resps[0], list) and resps[0]:
                        raw = str(resps[0][0])
                    elif resps:
                        raw = str(resps[0])
                    print(json.dumps({
                        "task": row.get("task_name") or row.get("task"),
                        "doc_id": row.get("doc_id"),
                        "target": row.get("target"),
                        "filtered_resps": row.get("filtered_resps"),
                        "extracted": extract_answer(raw, allow_raw_fallback=not strict_qa),
                        "raw_prefix": raw[:500],
                    }, ensure_ascii=False))
                    seen += 1
                    if seen >= int(args.limit):
                        return


if __name__ == "__main__":
    main()
