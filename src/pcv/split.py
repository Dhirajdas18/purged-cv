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
