from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_csv_keep_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def require_columns(df: pd.DataFrame, columns: Iterable[str], file_name: str = "dataframe") -> None:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")


def reorder_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    require_columns(df, columns)
    return df[columns].copy()