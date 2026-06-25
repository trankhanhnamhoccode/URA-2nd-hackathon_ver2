from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ura_ocr.utils.text_norm import norm_key


def add_norm_product_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "product_name" not in out.columns:
        out["product_name"] = ""

    out["norm_product_name"] = out["product_name"].map(norm_key)
    out["is_product_blank"] = out["norm_product_name"].eq("")

    return out


def make_random_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Random row-level split.
    This is useful but optimistic because same product can appear in both train and val.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_size = int(round(len(shuffled) * val_ratio))
    val_df = shuffled.iloc[:val_size].copy()
    train_df = shuffled.iloc[val_size:].copy()

    return {
        "train": train_df,
        "val": val_df,
    }


def make_product_holdout_split(
    df: pd.DataFrame,
    product_holdout_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Product-level holdout split.

    Nonblank product names in val are held out by normalized product_name.
    This simulates phase 2/open-set behavior better than random split.

    Blank product rows are split randomly and added to both train/val according to ratio.
    """
    if not 0 < product_holdout_ratio < 1:
        raise ValueError("product_holdout_ratio must be between 0 and 1.")

    work = add_norm_product_column(df)

    blank_df = work[work["is_product_blank"]].copy()
    nonblank_df = work[~work["is_product_blank"]].copy()

    unique_products = (
        nonblank_df["norm_product_name"]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )

    holdout_count = max(1, int(round(len(unique_products) * product_holdout_ratio)))
    holdout_products = set(unique_products[:holdout_count])

    val_nonblank = nonblank_df[
        nonblank_df["norm_product_name"].isin(holdout_products)
    ].copy()

    train_nonblank = nonblank_df[
        ~nonblank_df["norm_product_name"].isin(holdout_products)
    ].copy()

    blank_split = make_random_split(
        blank_df,
        val_ratio=product_holdout_ratio,
        seed=seed,
    )

    train_df = pd.concat(
        [train_nonblank, blank_split["train"]],
        ignore_index=True,
    )

    val_df = pd.concat(
        [val_nonblank, blank_split["val"]],
        ignore_index=True,
    )

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "holdout_products": pd.DataFrame(
            {
                "norm_product_name": sorted(holdout_products),
            }
        ),
    }


def save_splits(
    splits: Dict[str, pd.DataFrame],
    out_dir: str | Path,
    prefix: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        path = out_dir / f"{prefix}_{name}.csv"
        split_df.to_csv(path, index=False)