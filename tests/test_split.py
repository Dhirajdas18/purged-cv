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


def test_walkforward_train_ends_purge_plus_embargo_before_test():
    from pcv.split import WalkForward

    d = days(100)
    sp = list(WalkForward(5, h=5).split(None, groups=d))
    assert len(sp) == 4  # the first block only trains
    first = min(d[sp[0][1]])
    assert max(d[sp[0][0]]) == first - 5  # its label ends on the day before the test labels start
    sp6 = list(WalkForward(5, h=5, embargo=6).split(None, groups=d))
    assert max(d[sp6[0][0]]) == first - 11  # the reference model's cut: train dates < day[first - (h + 5)]
    assert all(max(d[tr]) < min(d[te]) for tr, te in sp6)


def test_cpcv_counts_and_paths():
    from pcv.split import CombinatorialPurged

    cv = CombinatorialPurged(6, 2, h=2)
    d = days(60)
    sp = list(cv.split(None, groups=d))
    assert len(sp) == cv.get_n_splits() == 15
    paths = cv.paths()
    assert len(paths) == 5
    for p in paths:  # one split per group, and that split tests the group
        assert len(p) == 6
        assert all(g in cv.test_groups[s] for g, s in enumerate(p))
    used = [(g, s) for p in paths for g, s in enumerate(p)]
    assert len(used) == len(set(used)) == 15 * 2


def test_cpcv_k1_equals_purged_kfold():
    from pcv.split import CombinatorialPurged

    d = days(40, 2)
    a = list(CombinatorialPurged(4, 1, h=3, embargo=1).split(None, groups=d))
    b = list(PurgedKFold(4, h=3, embargo=1).split(None, groups=d))
    for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
        assert tr1.tolist() == tr2.tolist() and te1.tolist() == te2.tolist()


def test_block_walkforward_uses_given_starts():
    from pcv.split import BlockWalkForward

    d = days(100)
    sp = list(BlockWalkForward([40, 70], h=5, embargo=6).split(None, groups=d))
    assert [sorted(set(d[te]))[0] for _, te in sp] == [40, 70]
    assert sorted(set(d[sp[0][1]]))[-1] == 69 and sorted(set(d[sp[1][1]]))[-1] == 99
    assert max(d[sp[0][0]]) == 40 - 5 - 6
