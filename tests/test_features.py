import numpy as np
import pandas as pd
import pytest

from pcv.data import LAST_DAY
from pcv.features import FEAT_A, build


@pytest.fixture
def lake(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.bdate_range(end="2025-12-10", periods=300)  # runs past LAST_DAY on purpose
    rows = []
    for s in "ABCD":
        r = rng.normal(0, 0.01, len(days))
        c = 100 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.002, len(days)))
        rows.append(
            pd.DataFrame(
                {
                    "date": days,
                    "symbol": s,
                    "ret": np.r_[0, np.expm1(np.diff(np.log(c)))],
                    "adj_open": o,
                    "adj_close": c,
                    "turnover": rng.uniform(1e7, 1e8, len(days)),
                    "in_uni": True,
                }
            )
        )
    pd.concat(rows).to_parquet(tmp_path / "panel.parquet")
    pd.DataFrame(
        {"date": days, "nifty": 1e4 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))}
    ).to_parquet(tmp_path / "market.parquet")
    return tmp_path


def test_no_row_after_last_day_and_columns(lake):
    f = build(lake)
    assert f["date"].max() <= pd.Timestamp(LAST_DAY)
    assert set(FEAT_A) | {"xs_5", "y_5", "fwd_5"} <= set(f.columns)


def test_labels_and_momentum(lake):
    raw = pd.read_parquet(lake / "panel.parquet").query("symbol == 'A'").reset_index(drop=True)
    raw = raw[raw["date"] <= LAST_DAY].reset_index(drop=True)
    f = build(lake, rank=False).query("symbol == 'A'").reset_index(drop=True)
    i = 260
    assert f.loc[i, "fwd_5"] == pytest.approx(
        raw.loc[i + 5, "adj_close"] / raw.loc[i + 1, "adj_open"] - 1
    )
    lr = np.log1p(raw["ret"])
    assert f.loc[i, "mom_1m"] == pytest.approx(lr[i - 20 : i + 1].sum())
    assert np.isnan(f["fwd_5"].iloc[-1])  # window runs past the data


def test_features_ranked_to_half_range(lake):
    f = build(lake)
    v = f[FEAT_A].to_numpy()
    assert np.nanmin(v) >= -0.5 and np.nanmax(v) <= 0.5
    assert f.groupby("date")["xs_5"].mean().abs().max() < 1e-12
