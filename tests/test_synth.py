import numpy as np

from pcv.evaluate import ShuffledKFold, cv_score, daily_ic
from pcv.split import PurgedKFold, WalkForward
from pcv.synth import panel


def test_panel_shapes_and_demeaned_labels():
    p = panel(20, 300, h=5, edge=0.0, seed=1)
    assert p.X.shape == (len(p.y), 5) and len(p.day) == len(p.stock) == len(p.y)
    assert p.day.max() <= 300 - 5 - 1  # rows without a full label window are dropped
    means = np.bincount(p.day, p.y) / np.bincount(p.day)
    assert np.abs(means[np.bincount(p.day) > 0]).max() < 1e-12


def test_edge_shows_as_ic():
    p = panel(200, 400, h=1, edge=0.5, seed=2)
    assert abs(daily_ic(p.X[:, 0], p.y, p.day) - 0.5) < 0.05
    assert abs(daily_ic(p.X[:, 1], p.y, p.day)) < 0.03


def test_perfect_prediction_ic_one():
    p = panel(10, 100, h=5, edge=0.0, seed=3)
    assert daily_ic(p.y, p.y, p.day) > 0.999


def test_shuffled_kfold_covers_each_row_once():
    sp = ShuffledKFold(5, seed=0).split(None, groups=np.arange(97))
    tests = np.concatenate([te for _, te in sp])
    assert sorted(tests.tolist()) == list(range(97))


def test_cv_score_runs_for_kfold_and_walkforward():
    from sklearn.linear_model import Ridge

    p = panel(20, 300, h=5, edge=0.2, seed=4)
    for sp in (PurgedKFold(5, h=5), WalkForward(5, h=5)):
        s = cv_score(lambda: Ridge(alpha=1.0), p, sp)
        assert 0.05 < s < 0.6


def test_cv_score_from_day_scores_only_later_rows():
    from sklearn.linear_model import Ridge

    p = panel(20, 300, h=5, edge=0.2, seed=4)
    a = cv_score(lambda: Ridge(alpha=1.0), p, PurgedKFold(5, h=5))
    b = cv_score(lambda: Ridge(alpha=1.0), p, PurgedKFold(5, h=5), from_day=150)
    assert a != b


def test_cv_daily_series_mean_equals_score():
    from sklearn.linear_model import Ridge

    from pcv.evaluate import cv_daily

    p = panel(20, 300, h=5, edge=0.2, seed=4)
    mk = lambda: Ridge(alpha=1.0)  # noqa: E731
    s = cv_daily(mk, p, PurgedKFold(5, h=5))
    assert abs(s.mean() - cv_score(mk, p, PurgedKFold(5, h=5))) < 1e-12


def test_cpcv_paths_mean_equals_cv_score():
    from sklearn.linear_model import Ridge

    from pcv.evaluate import cpcv_paths
    from pcv.split import CombinatorialPurged

    p = panel(20, 300, h=5, edge=0.2, seed=4)
    mk = lambda: Ridge(alpha=1.0)  # noqa: E731
    cv = CombinatorialPurged(6, 2, h=5)
    paths = cpcv_paths(mk, p, cv)
    assert len(paths) == 5
    assert abs(np.mean(paths) - cv_score(mk, p, cv)) < 1e-12


def test_expected_max_matches_simulation():
    from pcv.evaluate import expected_max

    assert expected_max(1) == 0
    sim = np.random.default_rng(0).standard_normal((20000, 12)).max(axis=1).mean()
    assert abs(expected_max(12) - sim) / sim < 0.03
