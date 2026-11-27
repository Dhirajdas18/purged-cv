import pandas as pd

from pcv.data import LAST_DAY, calendar


def test_calendar_never_passes_last_day(tmp_path):
    days = pd.bdate_range("2025-11-20", "2025-12-05")
    pd.DataFrame({"date": days, "symbol": "A"}).to_parquet(tmp_path / "panel.parquet")
    cal = calendar(tmp_path, start="2025-01-01", end="2026-01-01")
    assert cal.max() <= pd.Timestamp(LAST_DAY) and len(cal) == 5
