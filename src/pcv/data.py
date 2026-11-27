"""The local NSE bhavcopy lake (PCV_LAKE_DIR), read with a date filter so nothing after LAST_DAY
is ever loaded: the reference model keeps the last 365 days before its latest data as a sealed holdout."""

import os
from pathlib import Path

import pandas as pd

LAST_DAY = "2025-11-26"  # the reference model's holdout starts the day after
PICK_END = "2022-12-31"  # every rule is chosen on data up to here; 2023 on is checked once
START = "2013-01-01"


def lake(lake_dir: str | Path | None = None) -> Path:
    d = lake_dir or os.environ.get("PCV_LAKE_DIR")
    if not d:
        raise FileNotFoundError("set PCV_LAKE_DIR to a bhavcopy lake folder with panel.parquet")
    return Path(d)


def read_panel(columns: list[str], lake_dir=None, end: str = LAST_DAY) -> pd.DataFrame:
    """Rows of panel.parquet up to `end` (never past LAST_DAY)."""
    end = min(pd.Timestamp(end), pd.Timestamp(LAST_DAY))
    return pd.read_parquet(lake(lake_dir) / "panel.parquet", columns=columns,
                           filters=[("date", "<=", end)])


def calendar(lake_dir=None, start: str = START, end: str = LAST_DAY) -> pd.DatetimeIndex:
    """The lake's trading days from `start` to `end`."""
    d = pd.DatetimeIndex(read_panel(["date"], lake_dir, end)["date"].unique()).sort_values()
    return d[d >= pd.Timestamp(start)]
