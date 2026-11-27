import numpy as np
import pandas as pd


def test_rows_up_to_a_cut_keep_only_labels_that_end_by_it(monkeypatch):
    import pcv.reports as r

    cal = pd.bdate_range("2022-12-01", periods=40)
    f = pd.DataFrame({"date": np.repeat(cal, 2), "symbol": ["A", "B"] * 40})
    for c in r.TREND + r.RISK:
        f[c] = 0.0
    f["y_5"] = 0.1
    v = {"h": 5, "features": r.TREND, "model": "ridge", "param": 1.0}
    p = r._variant_panel(f, v, cal, hi="2022-12-30")
    last = int(np.searchsorted(cal, np.datetime64("2022-12-30"), side="right")) - 1
    assert p.day.max() == last - 5  # its label ends on the last day up to the cut
