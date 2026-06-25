import pandas as pd

from ura_ocr.eval.splits import (
    make_product_holdout_split,
    make_random_split,
)


def test_make_random_split_row_count():
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(10)],
            "product_name": ["A"] * 10,
        }
    )

    splits = make_random_split(df, val_ratio=0.2, seed=42)

    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 2


def test_make_product_holdout_split_no_product_overlap():
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(10)],
            "product_name": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "",
                "",
            ],
        }
    )

    splits = make_product_holdout_split(
        df,
        product_holdout_ratio=0.25,
        seed=42,
    )

    train_products = set(
        splits["train"]
        .query("product_name != ''")["norm_product_name"]
        .unique()
    )

    val_products = set(
        splits["val"]
        .query("product_name != ''")["norm_product_name"]
        .unique()
    )

    assert train_products.isdisjoint(val_products)
    assert len(splits["holdout_products"]) >= 1