"""The reference model's slow stock features (FEAT_A) and labels, rebuilt from the lake with the same
formulas (the original implementation), never reading past LAST_DAY."""

import numpy as np
import pandas as pd

from .data import LAST_DAY, lake

FEAT_A = ["mom_12_1", "mom_6m", "mom_1m", "dist_ma200", "dist_ma50", "vol_20", "vol_252",
          "idio_vol_60", "beta_60", "turnover_rank"]
HORIZONS = (1, 5, 21, 63)


def _roll(g, col, w, fn, minp=None):
    return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=minp or w), fn)())


def build(lake_dir=None, rank: bool = True) -> pd.DataFrame:
    """Universe rows with FEAT_A (ranked per date to [-0.5, 0.5] when `rank`), fwd_h (next open
    to close h rows later), xs_h (minus the day's mean) and y_h (its daily rank, the target)."""
    d = lake(lake_dir)
    end = pd.Timestamp(LAST_DAY)
    p = pd.read_parquet(d / "panel.parquet", filters=[("date", "<=", end)],
                        columns=["date", "symbol", "ret", "adj_open", "adj_close", "turnover",
                                 "in_uni"])
    mkt = pd.read_parquet(d / "market.parquet", filters=[("date", "<=", end)],
                          columns=["date", "nifty"]).set_index("date").sort_index()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["r"] = np.log1p(p["ret"].astype("float64"))
    g = p.groupby("symbol", sort=False)
    p["cum"] = g["r"].cumsum()
    g = p.groupby("symbol", sort=False)

    def sh(col, k):
        return g[col].shift(k)

    p["mom_12_1"] = sh("cum", 21) - sh("cum", 252)
    p["mom_6m"] = p["cum"] - sh("cum", 126)
    p["mom_1m"] = p["cum"] - sh("cum", 21)
    p["vol_20"] = _roll(g, "r", 20, "std")
    p["vol_252"] = _roll(g, "r", 252, "std", 200)
    p["dist_ma50"] = p["adj_close"] / _roll(g, "adj_close", 50, "mean") - 1
    p["dist_ma200"] = p["adj_close"] / _roll(g, "adj_close", 200, "mean") - 1
    p["m"] = p["date"].map(np.log(mkt["nifty"]).diff()).astype("float64")
    p["rm"], p["m2"] = p["r"] * p["m"], p["m"] ** 2
    g = p.groupby("symbol", sort=False)
    e_r, e_m = _roll(g, "r", 60, "mean"), _roll(g, "m", 60, "mean")
    var_m = _roll(g, "m2", 60, "mean") - e_m**2
    p["beta_60"] = (_roll(g, "rm", 60, "mean") - e_r * e_m) / var_m
    p["idio_vol_60"] = np.sqrt((_roll(g, "r", 60, "var") - p["beta_60"] ** 2 * var_m).clip(lower=0))
    p["log_to"] = np.log(p["turnover"].clip(lower=1))
    g = p.groupby("symbol", sort=False)
    p["turnover_rank"] = _roll(g, "log_to", 63, "mean", 40)
    for h in HORIZONS:
        p[f"fwd_{h}"] = g["adj_close"].shift(-h) / g["adj_open"].shift(-1) - 1

    u = p[p["in_uni"]].copy()
    if rank:
        for c in FEAT_A:
            u[c] = u[c].replace([np.inf, -np.inf], np.nan)
            lo = u.groupby("date")[c].transform(lambda s: s.quantile(0.01))
            hi = u.groupby("date")[c].transform(lambda s: s.quantile(0.99))
            u[c] = u[c].clip(lo, hi)
            u[c] = (u.groupby("date")[c].rank(pct=True) - 0.5).astype("float32")
    for h in HORIZONS:
        u[f"xs_{h}"] = u[f"fwd_{h}"] - u.groupby("date")[f"fwd_{h}"].transform("mean")
        u[f"y_{h}"] = (u.groupby("date")[f"xs_{h}"].rank(pct=True) - 0.5).astype("float32")
    cols = ["date", "symbol"] + FEAT_A + [f"{k}_{h}" for k in ("fwd", "xs", "y") for h in HORIZONS]
    return u[cols].sort_values(["date", "symbol"]).reset_index(drop=True)
