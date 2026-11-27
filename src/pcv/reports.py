"""`pcv report <topic>`: each topic's numbers and charts. Every README figure is printed here."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcv import plots

DATA = Path("docs/data")
HORIZONS = [1, 5, 21, 63]  # the reference model's label horizons, trading days
STATES = ["unused", "train", "purged", "embargo", "test"]


def day_states(n: int, split, purge_only_train: np.ndarray, walk: bool) -> np.ndarray:
    """Per day: 0 unused (future, walk-forward), 1 train, 2 purged, 3 embargo, 4 test."""
    train, test = split
    s = np.full(n, 2)
    s[purge_only_train] = 3  # kept by the purge, dropped by the embargo
    s[train] = 1
    s[test] = 4
    if walk:
        s[np.arange(n) > test.max()] = 0
    return s


def segments(s: np.ndarray) -> list[list[int]]:
    """Runs of one state as [first day, last day, state]."""
    cut = np.flatnonzero(np.diff(s)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut - 1, len(s) - 1]
    return [[int(a), int(b), int(s[a])] for a, b in zip(starts, ends, strict=True)]


def layouts(n: int, h: int, embargo: int) -> dict:
    from pcv.split import CombinatorialPurged, PurgedKFold, WalkForward

    day = np.arange(n)
    out = {}
    for name, make, walk in [
            ("plain", lambda e: PurgedKFold(5, 0, 0), False),
            ("purged", lambda e: PurgedKFold(5, h, e), False),
            ("walk-forward", lambda e: WalkForward(5, h, e), True),
            ("combinatorial", lambda e: CombinatorialPurged(6, 2, h, e), False)]:
        full = list(make(embargo).split(None, groups=day))
        bare = list(make(0).split(None, groups=day))
        out[name] = [segments(day_states(n, sp, b[0], walk))
                     for sp, b in zip(full, bare, strict=True)]
    return out


def report_splits(out_dir: Path) -> dict:
    """How the splitters cut the lake's 2013-2022 calendar, and how much training data the purge
    and embargo cost at each of the reference model's label horizons."""
    from pcv.data import PICK_END, START, calendar
    from pcv.split import PurgedKFold, WalkForward

    cal = calendar(end=PICK_END)
    n = len(cal)
    day = np.arange(n)
    res = {"calendar": f"{cal[0]:%d %b %Y} - {cal[-1]:%d %b %Y}, {n} trading days (rule-picking "
                       f"data; start {START})"}
    lost = {}
    for h in HORIZONS:
        base = [len(tr) for tr, _ in PurgedKFold(5).split(None, groups=day)]
        p = [len(tr) for tr, _ in PurgedKFold(5, h).split(None, groups=day)]
        pe = [len(tr) for tr, _ in PurgedKFold(5, h, 5).split(None, groups=day)]
        w0 = [len(tr) for tr, _ in WalkForward(5, h).split(None, groups=day)]
        w6 = [len(tr) for tr, _ in WalkForward(5, h, 6).split(None, groups=day)]
        lost[h] = (1 - sum(p) / sum(base), 1 - sum(pe) / sum(base))
        res[f"h = {h} days, 5 folds"] = (
            f"purge drops {lost[h][0]:.1%} of the training days plain k-fold uses (rows whose "
            f"labels overlap the test labels), purge + 5-day embargo {lost[h][1]:.1%}; "
            f"walk-forward trains "
            f"on {np.mean(w0):.0f} days with an exact purge and {np.mean(w6):.0f} with the reference model's "
            f"h + 5 cut")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    colors = {1: plots.SERIES[0], 2: plots.SERIES[1], 3: plots.SERIES[3], 4: plots.TEXT}
    lay = layouts(n, 63, 5)["purged"]
    for k, segs in enumerate(lay):
        for a, b, st in segs:
            if st in colors:
                ax.barh(k, b - a + 1, left=a, color=colors[st], height=0.62)
    years = [i for i in range(1, n) if cal[i].year != cal[i - 1].year][::2]
    ax.set_xticks(years, [f"{cal[i]:%Y}" for i in years])
    ax.set_yticks(range(len(lay)), [f"fold {k + 1}" for k in range(len(lay))])
    ax.invert_yaxis()
    ax.set_xlim(0, n)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=colors[st], label=lab) for st, lab in
                       [(1, "train"), (2, "purged"), (3, "embargo (5 days)"), (4, "test")]],
              frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Purged 5-fold, 63-day labels, 5-day embargo", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    x = np.arange(len(HORIZONS))
    ax.bar(x - 0.2, [lost[h][0] * 100 for h in HORIZONS], 0.38, color=plots.SERIES[1],
           label="purge")
    ax.bar(x + 0.2, [lost[h][1] * 100 for h in HORIZONS], 0.38, color=plots.SERIES[3],
           label="purge + 5-day embargo")
    ax.set_xticks(x, [f"{h}-day" for h in HORIZONS])
    ax.set_ylabel("training days dropped (%)")
    ax.set_title("Training days lost, by label length", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    plots.save(fig, out_dir, "splits")

    DATA.mkdir(parents=True, exist_ok=True)
    lab = {f"{h}/{e}": layouts(n, h, e) for h in HORIZONS for e in (0, 5)}
    ticks = [[i, f"{cal[i]:%Y}"] for i in years]
    (DATA / "splits.json").write_text(json.dumps({"n": n, "ticks": ticks, "layouts": lab,
                                                  "first": f"{cal[0]:%Y-%m-%d}",
                                                  "last": f"{cal[-1]:%Y-%m-%d}"},
                                                 separators=(",", ":")))
    return res


METHODS = ["shuffled k-fold", "k-fold, no purge", "purged", "purged + embargo", "walk-forward",
           "combinatorial purged"]
REPS = 10


def _splitters(h: int):
    from pcv.evaluate import ShuffledKFold
    from pcv.split import CombinatorialPurged, PurgedKFold, WalkForward

    cvs = [ShuffledKFold(5), PurgedKFold(5), PurgedKFold(5, h), PurgedKFold(5, h, h),
           WalkForward(5, h), CombinatorialPurged(6, 2, h)]
    return dict(zip(METHODS, cvs, strict=True))


def _models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    return {"ridge": lambda: Ridge(alpha=1.0),
            "boosted trees": lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0)}


def leak_chart(rows: list[dict], out_dir: Path) -> None:
    """CV score minus true score by method, 21-day labels, no true edge."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (ax, mname) in enumerate(zip(axes, ("ridge", "boosted trees"), strict=True)):
        sel = [r for r in rows if r["h"] == 21 and r["edge"] == 0.0 and r["case"] == "base"
               and r["model"] == mname]
        y = np.arange(len(sel))
        ax.barh(y, [r["gap"] for r in sel], color=[plots.MUTED if r["method"] in
                ("shuffled k-fold", "k-fold, no purge") else plots.SERIES[0] for r in sel],
                xerr=[2 * r["gap_se"] for r in sel], ecolor=plots.TEXT_2, height=0.6)
        ax.axvline(0, color=plots.TEXT, lw=1)
        ax.set_yticks(y, [r["method"] for r in sel] if i == 0 else [])
        ax.invert_yaxis()
        ax.set_title(f"{mname}: CV score minus true score", loc="left")
        ax.set_xlabel("IC points (21-day labels, no true edge)")
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.08, left=0.17)
    plots.save(fig, out_dir, "leak")


def report_leak(out_dir: Path) -> dict:
    """Each CV method's score against the true score on fresh panels from the same process:
    synthetic panels, 50 stocks x 1,000 days, slow features, known edge."""
    from pcv.evaluate import cv_score, true_score
    from pcv.synth import panel

    cases = [(h, e, "base") for h in (5, 21) for e in (0.0, 0.03)]
    cases += [(21, 0.03, "edge flips halfway"), (21, 0.0, "returns with momentum")]
    rows, res = [], {}
    for h, edge, kind in cases:
        kw = {"flip": kind == "edge flips halfway",
              "ret_ar": 0.05 if kind == "returns with momentum" else 0.0}
        for mname, mk in _models().items():
            cv = {m: [] for m in METHODS}
            truth = []
            for rep in range(REPS):
                p = panel(50, 1000, h, edge, seed=rep, **kw)
                fresh = panel(50, 1000, h, edge, seed=1000 + rep, **kw)
                truth.append(true_score(mk, p, fresh))
                for m, sp in _splitters(h).items():
                    cv[m].append(cv_score(mk, p, sp))
            t = np.array(truth)
            for m in METHODS:
                gap = np.array(cv[m]) - t
                rows.append({"h": h, "edge": edge, "case": kind, "model": mname, "method": m,
                             "cv": float(np.mean(cv[m])), "true": float(t.mean()),
                             "gap": float(gap.mean()),
                             "gap_se": float(gap.std(ddof=1) / np.sqrt(REPS))})
            res[f"{kind}, h {h}, edge {edge}, {mname}"] = "; ".join(
                f"{r['method']} {r['cv']:+.3f} (gap {r['gap']:+.3f} ± {r['gap_se']:.3f})"
                for r in rows[-len(METHODS):]) + f"; true {t.mean():+.3f}"

    leak_chart(rows, out_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "leak.json").write_text(json.dumps(rows))
    return res


REPORTS = {"splits": report_splits, "leak": report_leak}
