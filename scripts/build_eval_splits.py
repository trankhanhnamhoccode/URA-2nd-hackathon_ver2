from __future__ import annotations

import argparse
from pathlib import Path

from ura_ocr.io.csv_io import read_csv_keep_empty
from ura_ocr.eval.splits import (
    make_product_holdout_split,
    make_random_split,
    save_splits,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to train_labels.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/eval_splits",
        help="Output directory for split CSVs.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)

    print(f"[INFO] Loading labels: {labels_path}")
    df = read_csv_keep_empty(labels_path)

    if "image_id" not in df.columns:
        raise ValueError("labels CSV must contain image_id column.")

    if "product_name" not in df.columns:
        raise ValueError("labels CSV must contain product_name column.")

    print(f"[INFO] Rows: {len(df)}")

    random_splits = make_random_split(
        df,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_splits(
        random_splits,
        out_dir=out_dir,
        prefix="random",
    )

    holdout_splits = make_product_holdout_split(
        df,
        product_holdout_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_splits(
        holdout_splits,
        out_dir=out_dir,
        prefix="product_holdout",
    )

    print(f"[INFO] Saved splits to: {out_dir}")
    print(f"[INFO] random_train rows: {len(random_splits['train'])}")
    print(f"[INFO] random_val rows: {len(random_splits['val'])}")
    print(f"[INFO] product_holdout_train rows: {len(holdout_splits['train'])}")
    print(f"[INFO] product_holdout_val rows: {len(holdout_splits['val'])}")
    print(f"[INFO] holdout products: {len(holdout_splits['holdout_products'])}")


if __name__ == "__main__":
    main()