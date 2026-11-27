"""Build site/index.html from the committed derived data in docs/data."""

import json
from pathlib import Path

from pcv.site import build_html


def main() -> None:
    d = json.loads(Path("docs/data/splits.json").read_text())
    d["leak"] = json.loads(Path("docs/data/leak.json").read_text())
    d["feat"] = json.loads(Path("docs/data/features.json").read_text())
    d["check"] = json.loads(Path("docs/data/check.json").read_text())
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(d), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
