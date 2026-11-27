"""Scoring: per-day rank IC of predictions, out-of-fold for each cross-validation method."""

import numpy as np
import pandas as pd


def daily_ic(pred, y, day) -> float:
    """Mean over days of the rank correlation between predictions and labels across stocks."""
    df = pd.DataFrame({"p": pred, "y": y, "d": day}).dropna()
    r = df.groupby("d")[["p", "y"]].rank()
    r["d"] = df["d"]
    ic = r.groupby("d").apply(lambda g: g["p"].corr(g["y"]), include_groups=False)
    return float(ic.mean())


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
    blocks = np.array_split(np.unique(p.day), splitter.n_groups)
    group = np.searchsorted([b[0] for b in blocks], p.day, side="right") - 1
    ics = []
    for path in splitter.paths():
        joined = np.choose(group, [preds[s] for s in path])
        ics.append(daily_ic(joined[keep], p.y[keep], p.day[keep]))
    return float(np.mean(ics))


def true_score(make_model, p, fresh) -> float:
    """Train on the whole panel, score on a fresh panel from the same process."""
    return daily_ic(make_model().fit(p.X, p.y).predict(fresh.X), fresh.y, fresh.day)
