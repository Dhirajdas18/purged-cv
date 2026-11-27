import json
from pathlib import Path

from pcv.site import build_html


def test_page_builds_from_committed_data():
    d = json.loads(Path("docs/data/splits.json").read_text())
    d["leak"] = json.loads(Path("docs/data/leak.json").read_text())
    d["feat"] = json.loads(Path("docs/data/features.json").read_text())
    d["check"] = json.loads(Path("docs/data/check.json").read_text())
    html = build_html(d)
    assert "$check" not in html and len(d["check"]["real"]) == 48
    assert "$feat" not in html and len(d["feat"]) == 8
    assert "$leak" not in html and len({r["method"] for r in d["leak"]}) == 6
    assert "Split lab" in html and "$data" not in html
    assert set(d["layouts"]["21/5"]) == {"plain", "purged", "walk-forward", "combinatorial"}
    for segs in d["layouts"]["63/0"]["purged"]:  # every split covers every day once
        assert segs[0][0] == 0 and segs[-1][1] == d["n"] - 1
        assert all(a[1] + 1 == b[0] for a, b in zip(segs, segs[1:], strict=False))
