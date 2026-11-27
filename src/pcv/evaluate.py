"""Scoring: per-day rank IC of predictions, out-of-fold for each cross-validation method."""

import numpy as np
import pandas as pd


def ic_series(pred, y, day) -> pd.Series:
    """Rank correlation between predictions and labels across stocks, one value per day."""
    df = pd.DataFrame({"p": pred, "y": y, "d": day}).dropna()
    r = df.groupby("d")[["p", "y"]].rank()
    r["d"] = df["d"]
    return r.groupby("d").apply(lambda g: g["p"].corr(g["y"]), include_groups=False)


def daily_ic(pred, y, day) -> float:
    """Mean over days of the rank correlation between predictions and labels across stocks."""
    return float(ic_series(pred, y, day).mean())


def cv_daily(make_model, p, splitter, from_day: int | None = None) -> pd.Series:
    """Daily IC of out-of-fold predictions (splitters that test each row at most once)."""
    joined = np.full(len(p.y), np.nan)
    for tr, te in splitter.split(p.X, p.y, groups=p.day):
        joined[te] = make_model().fit(p.X[tr], p.y[tr]).predict(p.X[te])
    keep = np.ones(len(p.y), bool) if from_day is None else p.day >= from_day
    return ic_series(joined[keep], p.y[keep], p.day[keep])


class ShuffledKFold:
    """Ordinary k-fold on shuffled rows: what a default scikit-learn KFold(shuffle=True) does."""

    def __init__(self, n_splits: int = 5, seed: int = 0):
        self.n_splits, self.seed = n_splits, seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        n = len(groups)
        folds = np.array_split(np.random.default_rng(self.seed).permutation(n), self.n_splits)
        for k, te in enumerate(folds):
            yield np.concatenate([f for j, f in enumerate(folds) if j != k]), np.sort(te)


def cv_score(make_model, p, splitter, from_day: int | None = None) -> float:
    """IC of out-of-fold predictions (only rows from `from_day` on, if given). For combinatorial
    purged CV, the mean IC over its paths."""
    keep = np.ones(len(p.y), bool) if from_day is None else p.day >= from_day
    preds = []
    for tr, te in splitter.split(p.X, p.y, groups=p.day):
        m = make_model().fit(p.X[tr], p.y[tr])
        out = np.full(len(p.y), np.nan)
        out[te] = m.predict(p.X[te])
        preds.append(out)
    if not hasattr(splitter, "paths"):
        joined = np.full(len(p.y), np.nan)
        for out in preds:  # each row is tested at most once (walk-forward: some never)
            joined = np.where(np.isnan(joined), out, joined)
        return daily_ic(joined[keep], p.y[keep], p.day[keep])
    return float(np.mean(_path_ics(preds, p, splitter, keep)))


def _path_ics(preds, p, splitter, keep) -> list[float]:
    blocks = np.array_split(np.unique(p.day), splitter.n_groups)
    group = np.searchsorted([b[0] for b in blocks], p.day, side="right") - 1
    out = []
    for path in splitter.paths():
        joined = np.choose(group, [preds[s] for s in path])
        out.append(daily_ic(joined[keep], p.y[keep], p.day[keep]))
    return out


def cpcv_paths(make_model, p, splitter, from_day: int | None = None) -> list[float]:
    """IC of each combinatorial purged CV backtest path."""
    preds = []
    for tr, te in splitter.split(p.X, p.y, groups=p.day):
        out = np.full(len(p.y), np.nan)
        out[te] = make_model().fit(p.X[tr], p.y[tr]).predict(p.X[te])
        preds.append(out)
    keep = np.ones(len(p.y), bool) if from_day is None else p.day >= from_day
    return _path_ics(preds, p, splitter, keep)


def true_score(make_model, p, fresh) -> float:
    """Train on the whole panel, score on a fresh panel from the same process."""
    return daily_ic(make_model().fit(p.X, p.y).predict(fresh.X), fresh.y, fresh.day)


def expected_max(n: float) -> float:
    """Expected best of n independent standard normals (Bailey and Lopez de Prado), as in
    backtest-engine's forecast."""
    from math import e
    from statistics import NormalDist

    if n <= 1:
        return 0.0
    z, g = NormalDist().inv_cdf, 0.5772156649
    return (1 - g) * z(1 - 1 / n) + g * z(1 - 1 / (n * e))
