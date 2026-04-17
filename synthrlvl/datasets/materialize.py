from __future__ import annotations

import argparse

from synthetic_dataset import MaterializedDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally push materialized synthetic logic subsets.")
    parser.add_argument("--output-root", required=True, help="Root directory for subset parquet outputs")
    parser.add_argument("--train-up-to-5-rows", type=int, default=1_000_000)
    parser.add_argument("--train-up-to-10-rows", type=int, default=1_000_000)
    parser.add_argument("--val-rows-per-step", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--distractor-ratio", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = MaterializedDatasetBuilder()
    builder.build(
        output_root=args.output_root,
        train_up_to_5_rows=args.train_up_to_5_rows,
        train_up_to_10_rows=args.train_up_to_10_rows,
        val_rows_per_step=args.val_rows_per_step,
        seed=args.seed,
        distractor_ratio=args.distractor_ratio,
        chunk_size=args.chunk_size,
    )

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when --push-to-hub is set")
        builder.push_to_hub(
            output_root=args.output_root,
            repo_id=args.hf_repo_id,
            private=bool(args.hf_private),
        )


if __name__ == "__main__":
    main()
