"""The split lab: one static page carrying the committed split layouts (docs/data)."""

import datetime
import json
import re
from pathlib import Path
from string import Template

from pcv import plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 8h3l1-4 2 8 2-6 1 2h5%22 fill=%22none%22"
    " stroke=%22%232a78d6%22 stroke-width=%221.8%22 stroke-linejoin=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/purged-cv"


def lab_section(d: dict) -> str:
    lab = LAB.replace("$leak", json.dumps(d.get("leak", []), separators=(",", ":")))
    lab = lab.replace("$feat", json.dumps(d.get("feat", []), separators=(",", ":")))
    chk = d.get("check", {"rows": [], "real": {}, "best": 0})
    lab = lab.replace("$check", json.dumps({k: chk[k] for k in ("rows", "real", "best")},
                                           separators=(",", ":")))
    lab = lab.replace("$sizing", json.dumps(d.get("sizing", []), separators=(",", ":")))
    lab = lab.replace("$paths", json.dumps({"picks": d.get("paths", {}).get("picks", [])},
                                           separators=(",", ":")))
    rest = {k: v for k, v in d.items() if k not in ("leak", "feat", "check", "sizing", "paths")}
    return lab.replace("$data", json.dumps(rest, separators=(",", ":")))


def method_section(_: dict) -> str:
    return """<ul class="plain">
<li>Each row is one stock on one trading day. Its label is the return from the next day’s open to
the close h trading days later, so it depends on days day+1 to day+h.</li>
<li>Purged k-fold: the days are cut into blocks; one block is the test set. A training day is
dropped when its label window overlaps the test days’ label windows. An embargo drops a few more
days after the test labels end, for features that move slowly.</li>
<li>Walk-forward trains only on the past and stops h + embargo days before the test. The reference
model’s rule is the same with a 6-day embargo.</li>
<li>Combinatorial purged CV tests every choice of 2 of 6 blocks, purged around each, and joins the
pieces into 5 complete backtest paths.</li>
</ul>"""


SECTIONS = [
    ("Split lab", "Pick a method, a label length and an embargo, and see which trading days each "
     "split trains on. Research only, not investment advice.", lab_section),
    ("Method and caveats", "How the splits are made.", method_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(d: dict) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{make(d)}\n'
                     "</section>")
    dk, s = plots.DARK, plots.SERIES
    return PAGE.substitute(
        updated=datetime.date.today().strftime("%d %b %Y").lstrip("0"),
        first=d["first"][:7], last=d["last"][:7], icon=ICON, font=FONT, repo=REPO,
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=s[0], accent2=s[1], green=s[2], violet=s[6], muted=plots.MUTED,
        d_surface=dk[plots.SURFACE], d_text=dk[plots.TEXT], d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID], d_accent=dk[s[0]], d_accent2=dk[s[1]], d_green=dk[s[2]],
        d_violet=dk[s[6]], d_muted="#6b6a64",
        nav="\n".join(nav), sections="\n".join(parts),
    )
