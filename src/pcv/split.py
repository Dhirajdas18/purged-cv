"""Cross-validation splits for rows whose labels look ahead.

Each row is one (day, stock) pair. `day` is its trading-day number on the calendar (0, 1, 2, ...).
Its label covers days day+1 .. day+h: bought at the next open, sold at the close h days later.
Splitters follow scikit-learn: `split(X, y=None, groups=day)` yields (train, test) row indices.
"""

import numpy as np


def label_end(day: np.ndarray, h: int) -> np.ndarray:
    """Last day a row's label depends on."""
    return np.asarray(day) + h


def _days(groups) -> np.ndarray:
    if groups is None:
        raise ValueError("pass each row's trading-day number as groups=")
    return np.asarray(groups)


def _blocks(day: np.ndarray, n: int) -> list[np.ndarray]:
    """n contiguous blocks of the distinct day numbers."""
    return np.array_split(np.unique(day), n)


def _train_mask(day: np.ndarray, test_days: np.ndarray, h: int, embargo: int) -> np.ndarray:
    """Rows outside the test block whose label window [d+1, d+h] misses the test labels
    [a+1, b+h], and which are not in the embargo b+h <= d < b+h+embargo."""
    a, b = test_days[0], test_days[-1]
    keep = ~np.isin(day, test_days)
    keep &= ~((day >= a + 1 - h) & (day <= b + h - 1))
    keep &= ~((day >= b + h) & (day < b + h + embargo))
    return keep


class PurgedKFold:
    """K contiguous folds by day; training rows whose labels overlap the test labels are
    dropped (purge), and so are the `embargo` days after the test labels end."""

    def __init__(self, n_splits: int = 5, h: int = 0, embargo: int = 0):
        self.n_splits, self.h, self.embargo = n_splits, h, embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        day = _days(groups)
        for block in _blocks(day, self.n_splits):
            test = np.isin(day, block)
            yield (np.flatnonzero(_train_mask(day, block, self.h, self.embargo)),
                   np.flatnonzero(test))


class WalkForward:
    """Train only on the past. The day is cut into `n_splits` blocks; each block after the first is
    tested on everything before it, ending h + embargo days before the test starts, so the last
    training label ends before the first test label begins. The reference rule (train on dates
    before day[first test - (h + 5)]) is embargo = 6."""

    def __init__(self, n_splits: int = 5, h: int = 0, embargo: int = 0):
        self.n_splits, self.h, self.embargo = n_splits, h, embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits - 1

    def split(self, X=None, y=None, groups=None):
        day = _days(groups)
        for block in _blocks(day, self.n_splits)[1:]:
            train = day <= block[0] - self.h - self.embargo
            yield np.flatnonzero(train), np.flatnonzero(np.isin(day, block))


class CombinatorialPurged:
    """Combinatorial purged CV (Lopez de Prado): the days are cut into `n_groups` blocks and every
    choice of `k_test` blocks is a test set, purged and embargoed around each block. `paths()` joins
    the test pieces into full backtest paths."""

    def __init__(self, n_groups: int = 6, k_test: int = 2, h: int = 0, embargo: int = 0):
        from itertools import combinations

        self.n_groups, self.k_test, self.h, self.embargo = n_groups, k_test, h, embargo
        self.test_groups = list(combinations(range(n_groups), k_test))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.test_groups)

    def split(self, X=None, y=None, groups=None):
        day = _days(groups)
        blocks = _blocks(day, self.n_groups)
        for tg in self.test_groups:
            train = np.ones(len(day), bool)
            for g in tg:
                train &= _train_mask(day, blocks[g], self.h, self.embargo)
            test = np.isin(day, np.concatenate([blocks[g] for g in tg]))
            yield np.flatnonzero(train), np.flatnonzero(test)

    def paths(self) -> list[list[int]]:
        """Each path lists, for every group, the split whose test set supplies that group."""
        per_group = [[s for s, tg in enumerate(self.test_groups) if g in tg]
                     for g in range(self.n_groups)]
        return [[per_group[g][j] for g in range(self.n_groups)]
                for j in range(len(per_group[0]))]


class BlockWalkForward:
    """Walk-forward on fixed calendar blocks (e.g. every 6 months, as the reference model
    retrains): block i runs from starts[i] to the day before starts[i+1] (the last to the end);
    training ends h + embargo days before the block starts."""

    def __init__(self, starts: list[int], h: int = 0, embargo: int = 0):
        self.starts, self.h, self.embargo = list(starts), h, embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.starts)

    def split(self, X=None, y=None, groups=None):
        day = _days(groups)
        ends = self.starts[1:] + [day.max() + 1]
        for a, b in zip(self.starts, ends, strict=True):
            test = (day >= a) & (day < b)
            yield np.flatnonzero(day <= a - self.h - self.embargo), np.flatnonzero(test)
