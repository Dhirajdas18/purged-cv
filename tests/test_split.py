import numpy as np
import pytest

from pcv.split import PurgedKFold, label_end


def days(n, stocks=1):
    return np.repeat(np.arange(n), stocks)


def train_days(day, split):
    return sorted(set(day[split[0]].tolist()))


def test_label_end():
    assert label_end(np.array([0, 3]), 5).tolist() == [5, 8]


def test_h0_no_embargo_is_plain_blocks():
    d = days(10, 2)
    sp = list(PurgedKFold(5).split(None, groups=d))
    assert len(sp) == 5
    for k, (tr, te) in enumerate(sp):
        assert sorted(set(d[te])) == [2 * k, 2 * k + 1] and len(te) == 4 and len(tr) == 16


def test_purge_drops_rows_whose_label_reaches_test():
    d = days(20)
    sp = list(PurgedKFold(4, h=3).split(None, groups=d))
    assert sorted(set(d[sp[1][1]])) == [5, 6, 7, 8, 9]
    assert train_days(d, sp[1]) == [0, 1, 2] + list(range(12, 20))


def test_embargo_drops_days_after_test():
    d = days(20)
    sp = list(PurgedKFold(4, h=3, embargo=2).split(None, groups=d))
    assert train_days(d, sp[1]) == [0, 1, 2] + list(range(14, 20))


def test_indices_follow_caller_order():
    d = days(20, 3)
    perm = np.random.default_rng(0).permutation(len(d))
    a = list(PurgedKFold(4, h=3).split(None, groups=d))
    b = list(PurgedKFold(4, h=3).split(None, groups=d[perm]))
    for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
        assert sorted(d[tr1]) == sorted(d[perm][tr2]) and sorted(d[te1]) == sorted(d[perm][te2])


def test_edges_clip():
    d = days(20)
    for tr, te in PurgedKFold(4, h=6, embargo=3).split(None, groups=d):
        assert tr.min(initial=0) >= 0 and te.max() < len(d)
        assert not set(tr) & set(te)


def test_day_gaps_split_by_day_number():
    d = np.array([0, 1, 2, 10, 11, 12])  # a long holiday between 2 and 10
    sp = list(PurgedKFold(2).split(None, groups=d))
    assert sorted(d[sp[0][1]]) == [0, 1, 2] and sorted(d[sp[1][1]]) == [10, 11, 12]


def test_needs_groups():
    with pytest.raises(ValueError):
        list(PurgedKFold(2).split(np.zeros(4)))
