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
            f"on {np.mean(w0):.0f} days with an exact purge and {np.mean(w6):.0f} with the "
            f"reference model's "
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


# the reference model's LightGBM settings, seed added for repeatability
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  max_bin=63, verbose=-1, random_state=0)
CHECK_END = "2025-08-01"  # rows whose 63-day labels are complete inside the data


def lake_models():
    """The reference model's two models: Ridge on gaps filled with 0, LightGBM on gaps as is."""
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    return {"ridge": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                                           Ridge(alpha=1.0)),
            "lightgbm": lambda: LGBMRegressor(**LGB_PARAMS)}


def lake_table(force: bool = False):
    """The rebuilt feature table, cached in out/ (never committed)."""
    import pandas as pd

    from pcv.features import build

    path = Path("out/features.parquet")
    if force or not path.exists():
        path.parent.mkdir(exist_ok=True)
        build().to_parquet(path)
    return pd.read_parquet(path)


def lake_panel(f, h: int, end: str):
    """Rows whose labels end by `end`, as a Panel (y = daily rank of the excess return, the
    training target; IC is scored against it, the same ranks as xs_h)."""
    import pandas as pd

    from pcv.data import calendar
    from pcv.features import FEAT_A
    from pcv.synth import Panel

    cal = calendar()
    last = int(np.searchsorted(cal, np.datetime64(pd.Timestamp(end)), side="right")) - 1
    f = f[(f["date"] <= cal[last - h]) & f[f"y_{h}"].notna()]  # labels end by `end`
    return Panel(f[FEAT_A].to_numpy(np.float32), f[f"y_{h}"].to_numpy(float),
                 np.searchsorted(cal, f["date"].to_numpy()), np.zeros(len(f))), cal


def features_chart(rows: list[dict], out_dir: Path) -> None:
    """Baseline IC by horizon: purged 5-fold bars, walk-forward dots."""
    from pcv.features import HORIZONS

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(HORIZONS))
    for k, (mname, col) in enumerate([("ridge", plots.SERIES[0]), ("lightgbm", plots.SERIES[2])]):
        sel = [r for r in rows if r["model"] == mname]
        ax.bar(x + (k - 0.5) * 0.38, [r["purged"] for r in sel], 0.36, color=col,
               label=f"{mname}, purged 5-fold")
        ax.scatter(x + (k - 0.5) * 0.38, [r["walk"] for r in sel], color=plots.TEXT, zorder=3,
                   s=24, label="walk-forward" if k == 0 else None)
    ax.axhline(0, color=plots.TEXT, lw=1)
    ax.set_xticks(x, [f"{h}-day labels" for h in HORIZONS])
    ax.set_ylabel("mean daily rank IC")
    ax.set_title("Reference model features on NSE stocks, scored 2016-2022", loc="left")
    ax.set_ylim(0, max(max(r["purged"], r["walk"]) for r in rows) * 1.35)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    plots.save(fig, out_dir, "features")


def report_features(out_dir: Path) -> dict:
    """The reference model's slow features and labels rebuilt from the lake and matched against
    the original implementation's features.parquet; then a baseline for Ridge and LightGBM on
    data up to Dec 2022."""
    import os

    import pandas as pd

    from pcv.data import PICK_END
    from pcv.evaluate import cv_score
    from pcv.features import FEAT_A, HORIZONS
    from pcv.split import BlockWalkForward, PurgedKFold

    f = lake_table(force=True)
    res = {"table": f"{len(f):,} universe rows, {f['date'].min():%d %b %Y} - "
                    f"{f['date'].max():%d %b %Y}, {f['symbol'].nunique()} stocks"}
    cols = FEAT_A + [f"xs_{h}" for h in HORIZONS] + [f"y_{h}" for h in HORIZONS]
    q = pd.read_parquet(os.environ["PCV_REF_FEATURES"], columns=["date", "symbol"] + cols,
                        filters=[("date", "<=", pd.Timestamp(CHECK_END))])
    m = f[f["date"] <= CHECK_END].merge(q, on=["date", "symbol"], suffixes=("", "_q"))
    worst = max(float((m[c].astype(float) - m[c + "_q"].astype(float)).abs().max())
                for c in cols)
    nan_diff = max(float((m[c].isna() != m[c + "_q"].isna()).mean()) for c in cols)
    res["port check against the original features.parquet"] = (
        f"{len(m):,} shared rows of {int((f['date'] <= CHECK_END).sum()):,} (to {CHECK_END}), "
        f"{len(cols)} columns: largest gap {worst:.1e}, missing-value mismatch {nan_diff:.2%}")

    rows = []
    for h in HORIZONS:
        p, cal = lake_panel(f, h, PICK_END)
        starts = [int(np.searchsorted(cal, pd.Timestamp(d)))
                  for d in pd.date_range("2016-01-01", PICK_END, freq="6MS")]
        for mname, mk in lake_models().items():
            pk = cv_score(mk, p, PurgedKFold(5, h), from_day=starts[0])  # same days as walk-forward
            wf = cv_score(mk, p, BlockWalkForward(starts, h, 6))
            rows.append({"h": h, "model": mname, "purged": pk, "walk": wf})
            res[f"h = {h}, {mname}"] = (f"IC on Jan 2016 - Dec 2022: purged 5-fold {pk:+.4f}; "
                                        f"reference walk-forward (6-month blocks, embargo 6) "
                                        f"{wf:+.4f}")

    features_chart(rows, out_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "features.json").write_text(json.dumps(rows))
    return res


TREND = ["mom_12_1", "mom_6m", "mom_1m", "dist_ma200", "dist_ma50"]
RISK = ["vol_20", "vol_252", "idio_vol_60", "beta_60", "turnover_rank"]
# real 2023 - Nov 2025 scores, computed once; a fixed path, whatever folder the report runs from
CHECK = Path(__file__).resolve().parents[2] / "runs" / "check.json"


def variants() -> list[dict]:
    """The 48 pre-registered model variants (plan-topic4)."""
    from pcv.features import FEAT_A, HORIZONS

    out = []
    for h in HORIZONS:
        for fs, cols in (("all", FEAT_A), ("trend", TREND), ("risk", RISK)):
            for model, prm in (("ridge", 1.0), ("ridge", 100.0), ("lightgbm", 7), ("lightgbm", 31)):
                name = f"h{h} {fs} {model}-{prm:g}"
                out.append({"name": name, "h": h, "features": cols, "model": model, "param": prm})
    return out


def make_variant(v: dict):
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    if v["model"] == "ridge":
        return lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                                     Ridge(alpha=v["param"]))
    return lambda: LGBMRegressor(**(LGB_PARAMS | {"num_leaves": int(v["param"])}))


def _variant_panel(f, v, cal, lo=None, hi=None):
    import pandas as pd

    from pcv.synth import Panel

    m = f[f"y_{v['h']}"].notna()
    if lo is not None:
        m &= f["date"] >= pd.Timestamp(lo)
    if hi is not None:  # only labels that end by the cut: no return after it reaches training
        last = cal[int(np.searchsorted(cal, np.datetime64(pd.Timestamp(hi)), side="right")) - 1]
        end = cal[max(int(np.searchsorted(cal, np.datetime64(last))) - v["h"], 0)]
        m &= f["date"] <= end
    g = f[m]
    return Panel(g[v["features"]].to_numpy(np.float32), g[f"y_{v['h']}"].to_numpy(float),
                 np.searchsorted(cal, g["date"].to_numpy()), np.zeros(len(g)))


def real_scores(f, cal) -> dict:
    """Each variant trained on every row whose label ends by the last 2022 day, scored on
    2023-01-01 .. 2025-11-26. Run once; the result is saved and read back afterwards."""
    from pcv.data import PICK_END
    from pcv.evaluate import daily_ic

    if CHECK.exists():
        return json.loads(CHECK.read_text())["real"]
    last = int(np.searchsorted(cal, np.datetime64(PICK_END), side="right")) - 1
    real = {}
    for v in variants():
        tr = _variant_panel(f, v, cal, hi=PICK_END)
        keep = tr.day <= last - v["h"]
        te = _variant_panel(f, v, cal, lo="2023-01-01")
        m = make_variant(v)().fit(tr.X[keep], tr.y[keep])
        real[v["name"]] = daily_ic(m.predict(te.X), te.y, te.day)
    CHECK.parent.mkdir(exist_ok=True)
    CHECK.write_text(json.dumps({"window": "2023-01-01 to 2025-11-26, trained on labels ending "
                                           "by 2022-12-30", "real": real}, indent=1) + "\n")
    return real


def report_check(out_dir: Path) -> dict:
    """Which cross-validation score predicts the real later score, over 48 model variants
    (pre-registered in plan-topic4): CV on data to Dec 2022, real = Jan 2023 - Nov 2025."""
    import pandas as pd

    from pcv.data import PICK_END, calendar
    from pcv.evaluate import ShuffledKFold, cv_score
    from pcv.split import BlockWalkForward, CombinatorialPurged, PurgedKFold

    f = lake_table()
    cal = calendar()
    starts = [int(np.searchsorted(cal, pd.Timestamp(d)))
              for d in pd.date_range("2016-01-01", PICK_END, freq="6MS")]
    cache = Path("out/check_cv.json")
    cv = json.loads(cache.read_text()) if cache.exists() else {}
    for v in variants():
        if v["name"] in cv:
            continue
        p = _variant_panel(f, v, cal, hi=PICK_END)
        h, mk = v["h"], make_variant(v)
        sps = dict(zip(METHODS, [ShuffledKFold(5), PurgedKFold(5), PurgedKFold(5, h),
                                 PurgedKFold(5, h, h), BlockWalkForward(starts, h, 6),
                                 CombinatorialPurged(6, 2, h)], strict=True))
        cv[v["name"]] = {m: cv_score(mk, p, sp, from_day=starts[0]) for m, sp in sps.items()}
        cache.write_text(json.dumps(cv))
    real = real_scores(f, cal)

    names = [v["name"] for v in variants()]
    r = np.array([real[n] for n in names])
    hz = np.array([v["h"] for v in variants()])
    res, rows = {"variants": f"{len(names)}; real mean IC {r.mean():+.4f} (range {r.min():+.4f} "
                             f"to {r.max():+.4f})"}, []
    base = None
    for m in METHODS:
        c = np.array([cv[n][m] for n in names])
        gap = c - r
        rho = pd.Series(c).rank().corr(pd.Series(r).rank())
        pick = []
        for h in sorted(set(hz)):
            idx = np.flatnonzero(hz == h)
            pick.append(r[idx[np.argmax(c[idx])]])
        row = {"method": m, "gap": float(gap.mean()), "abs_gap": float(np.abs(gap).mean()),
               "rho": float(rho), "picked_real": float(np.mean(pick)), "cv_mean": float(c.mean())}
        if m == "purged":
            base = np.abs(gap)
        rows.append((row, np.abs(gap)))
    best_possible = np.mean([r[hz == h].max() for h in sorted(set(hz))])
    for row, ag in rows:
        d = ag - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
        row["abs_gap_vs_purged_t"] = float(t)
        res[row["method"]] = (
            f"mean CV {row['cv_mean']:+.4f}, gap {row['gap']:+.4f}, "
            f"mean |gap| {row['abs_gap']:.4f} "
            f"(vs purged: t {t:+.2f}), rank correlation with real {row['rho']:.2f}, real IC of its "
            f"top pick per horizon {row['picked_real']:+.4f} (best possible {best_possible:+.4f})")
    rows = [row for row, _ in rows]
    check_chart(rows, {n: cv[n] for n in names}, real, out_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "check.json").write_text(json.dumps({"rows": rows, "cv": cv, "real": real,
                                                 "best": best_possible}))
    return res


def check_chart(rows, cv, real, out_dir: Path) -> None:
    """Left: CV vs real for purged 5-fold and shuffled; right: mean |gap| by method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = list(cv)
    rr = [real[n] for n in names]
    shown = (("walk-forward", plots.SERIES[2]), ("purged", plots.SERIES[0]))
    for m, col in shown:  # shuffled k-fold sits far right (up to 0.31); its miss is on the right
        ax.scatter([cv[n][m] for n in names], rr, s=22, color=col, label=m, alpha=0.85)
    vals = rr + [cv[n][m] for n in names for m, _ in shown]
    lo, hi = min(vals) - 0.005, max(vals) + 0.005
    ax.plot([lo, hi], [lo, hi], color=plots.TEXT_2, lw=1, ls="--")
    ax.set_xlabel("cross-validation IC, 2016-2022")
    ax.set_ylabel("real IC, Jan 2023 - Nov 2025")
    ax.set_title("48 models: CV score vs real later score", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    y = np.arange(len(rows))
    grey = ("shuffled k-fold", "k-fold, no purge")
    ax.barh(y, [r_["abs_gap"] for r_ in rows], height=0.6,
            color=[plots.MUTED if r_["method"] in grey else plots.SERIES[0] for r_ in rows])
    ax.set_yticks(y, [r_["method"] for r_ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("mean |CV - real| (IC points)")
    ax.set_title("Average miss by method", loc="left")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.45)
    plots.save(fig, out_dir, "check")


def _grid(h: int):
    """(kind, label, splitter args) per plan-topic5, duplicates dropped."""
    kf = sorted(set([(0, 0), (-(-h // 2), 0), (h, 0), (h, 6), (h, h), (h, 2 * h)]))
    # gap g: training ends g trading days before the test (1 = the day before)
    wf = sorted(set([1, h, h + 6, 2 * h, 3 * h]))  # smallest first: the rule takes the first pass
    return kf, wf


def report_sizing(out_dir: Path) -> dict:
    """How big the purge, embargo and walk-forward gap need to be, by label length, on data to
    Dec 2022 only (rule fixed in plan-topic5)."""
    import pandas as pd

    from pcv.data import PICK_END, calendar
    from pcv.evaluate import cv_daily
    from pcv.features import FEAT_A, HORIZONS
    from pcv.split import BlockWalkForward, PurgedKFold

    f = lake_table()
    cal = calendar()
    starts = [int(np.searchsorted(cal, pd.Timestamp(d)))
              for d in pd.date_range("2016-01-01", PICK_END, freq="6MS")]
    cache = Path("out/sizing.json")
    done = json.loads(cache.read_text()) if cache.exists() else {}
    rows, res, need_wf = [], {}, {}
    for h in HORIZONS:
        kf, wf = _grid(h)
        for mname in ("ridge", "lightgbm"):
            prm = 1.0 if mname == "ridge" else 31
            v = {"h": h, "features": FEAT_A, "model": mname, "param": prm}
            p = _variant_panel(f, v, cal, hi=PICK_END)
            mk = make_variant(v)
            series = {}
            for kind, settings in (("k-fold", kf), ("walk-forward", wf)):
                for st in settings:
                    key = f"{h}|{mname}|{kind}|{st}"
                    if key not in done:
                        sp = (PurgedKFold(5, st[0], st[1]) if kind == "k-fold"
                              else BlockWalkForward(starts, 0, st))
                        done[key] = cv_daily(mk, p, sp, from_day=starts[0]).to_dict()
                        cache.write_text(json.dumps(done))
                    series[(kind, st)] = pd.Series(done[key]).rename(int)
            for kind, settings in (("k-fold", kf), ("walk-forward", wf)):
                ref = series[(kind, (h, 2 * h) if kind == "k-fold" else 3 * h)]  # plan-topic5
                needed = None
                for st in settings:
                    d = (series[(kind, st)] - ref).dropna()
                    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d) / h) if d.std() > 0 else 0.0
                    rows.append({"h": h, "model": mname, "kind": kind,
                                 "setting": list(st) if kind == "k-fold" else st,
                                 "ic": float(series[(kind, st)].mean()), "diff": float(d.mean()),
                                 "t": float(t)})
                    if needed is None and t < 2:
                        needed = st
                lab = (f"purge {needed[0]}, embargo {needed[1]}" if kind == "k-fold"
                       else f"gap {needed} days")
                if kind == "walk-forward":
                    need_wf[(h, mname)] = needed
                parts = "; ".join(
                    (f"({r['setting'][0]},{r['setting'][1]}) " if kind == "k-fold"
                     else f"gap {r['setting']} ") + f"{r['ic']:+.4f} (t {r['t']:+.1f})"
                    for r in rows if r["h"] == h and r["model"] == mname and r["kind"] == kind)
                res[f"h = {h}, {mname}, {kind}"] = f"{parts}; needed: {lab}"
    for h in HORIZONS:
        for mname in ("ridge", "lightgbm"):
            w = {r["setting"]: r["ic"] for r in rows
                 if r["h"] == h and r["model"] == mname and r["kind"] == "walk-forward"}
            res[f"walk-forward, h = {h}, {mname}: gap 1 minus gap h"] = (
                f"{w[1] - w[h]:+.4f}; gap h to the widest {min(w[g] for g in w if g >= h):+.4f} .. "
                f"{max(w[g] for g in w if g >= h):+.4f}")
    qok = all(need_wf[(h, m)] <= h + 6 for h, m in need_wf)  # plan: at or above the needed gap
    res["reference gap (h + 6 days before the test block)"] = (
        "at or above the needed walk-forward gap for every label length and model" if qok
        else "below the needed walk-forward gap somewhere: see the lines above")
    sizing_chart(rows, out_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "sizing.json").write_text(json.dumps(rows))
    return res


def sizing_chart(rows: list[dict], out_dir: Path) -> None:
    """Walk-forward IC against the gap, one panel per label length, ridge and LightGBM."""
    from pcv.features import HORIZONS

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, h in zip(axes, HORIZONS, strict=True):
        for mname, col in (("ridge", plots.SERIES[0]), ("lightgbm", plots.SERIES[2])):
            sel = [r for r in rows if r["h"] == h and r["model"] == mname
                   and r["kind"] == "walk-forward"]
            ax.plot([r["setting"] for r in sel], [r["ic"] for r in sel], "o-", color=col, lw=2,
                    ms=5, label=mname)
        ax.axvline(h + 6, color=plots.TEXT_2, lw=1, ls="--")
        ax.set_title(f"{h}-day labels", loc="left")
        ax.set_xlabel("gap before test (trading days)")
    axes[0].set_ylabel("walk-forward IC, 2016-2022")
    axes[0].legend(frameon=False)
    fig.text(0.99, 0.01, "dashed: reference gap (h + 6)", ha="right", color=plots.TEXT_2,
             fontsize=9)
    fig.subplots_adjust(wspace=0.12, bottom=0.18)
    plots.save(fig, out_dir, "sizing")


def report_paths(out_dir: Path) -> dict:
    """Combinatorial purged CV as a spread: 7 paths per model (8 groups, 2 tested), on data to
    Dec 2022; checked against topic 4's real 2023 - Nov 2025 scores (measures in plan-topic6)."""
    from math import sqrt

    import pandas as pd

    from pcv.data import PICK_END, calendar
    from pcv.evaluate import cpcv_paths
    from pcv.split import CombinatorialPurged

    f = lake_table()
    cal = calendar()
    first = int(np.searchsorted(cal, pd.Timestamp("2016-01-01")))
    cache = Path("out/paths.json")
    paths = json.loads(cache.read_text()) if cache.exists() else {}
    for v in variants():
        if v["name"] not in paths:
            p = _variant_panel(f, v, cal, hi=PICK_END)
            cv = CombinatorialPurged(8, 2, v["h"], v["h"])
            paths[v["name"]] = cpcv_paths(make_variant(v), p, cv, from_day=first)
            cache.write_text(json.dumps(paths))
    chk = json.loads(CHECK.read_text())["real"]
    cvd = json.loads((DATA / "check.json").read_text())["cv"]
    names = [v["name"] for v in variants()]
    P = np.array([paths[n] for n in names])
    real = np.array([chk[n] for n in names])
    mean, sd = P.mean(1), P.std(1, ddof=1)
    in_range = (real >= P.min(1)) & (real <= P.max(1))
    in_2sd = np.abs(real - mean) <= 2 * sd
    rho = pd.Series(sd).rank().corr(pd.Series(np.abs(real - mean)).rank())
    hzs = np.array([v["h"] for v in variants()])
    within = {int(h): float(pd.Series(sd[hzs == h]).rank().corr(
        pd.Series(np.abs(real - mean)[hzs == h]).rank())) for h in sorted(set(hzs))}
    res = {"paths": f"7 per model, 48 models; path sd median {np.median(sd):.4f} (range "
                    f"{sd.min():.4f} to {sd.max():.4f}); mean real minus path mean "
                    f"{(real - mean).mean():+.4f}",
           "real score inside the model's path range": f"{in_range.sum()} of 48",
           "real score inside path mean ± 2 sd": f"{in_2sd.sum()} of 48",
           "rank correlation of path sd with |real - path mean|": (
               f"{rho:.2f} over all 48; within one label length (12 models each): "
               + ", ".join(f"h {h} {c:+.2f}" for h, c in within.items())
               + "; median path sd by label length: "
               + ", ".join(f"h {h} {np.median(sd[hzs == h]):.4f}" for h in within))}

    from pcv.evaluate import expected_max

    picks = []
    hz = np.array([v["h"] for v in variants()])
    for h in sorted(set(hz)):
        idx = np.flatnonzero(hz == h)
        k = idx[np.argmax(mean[idx])]
        c = np.corrcoef(P[idx])
        r_ = float(c[~np.eye(len(idx), dtype=bool)].mean())
        cut = sqrt(max(1 - r_, 0)) * sd[k] / sqrt(P.shape[1]) * expected_max(len(idx))
        wf = cvd[names[k]]["walk-forward"]
        picks.append({"h": int(h), "pick": names[k], "naive": float(mean[k]),
                      "forecast": float(mean[k] - cut), "haircut": float(cut),
                      "walk": float(wf), "real": float(real[k]), "corr": r_})
        res[f"h = {h} pick {names[k]}"] = (
            f"path mean {mean[k]:+.4f}, haircut {cut:.4f} (variant path correlation {r_:.2f}) -> "
            f"forecast {mean[k] - cut:+.4f}; walk-forward {wf:+.4f}; real {real[k]:+.4f}")
    err = {k: float(np.mean([abs(p_[k] - p_["real"]) for p_ in picks]))
           for k in ("naive", "forecast", "walk")}
    res["mean |forecast - real| over the 4 picks"] = (
        f"naive pick {err['naive']:.4f}, after haircut {err['forecast']:.4f}, walk-forward "
        f"{err['walk']:.4f}")
    paths_chart(names, P, real, out_dir)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "paths.json").write_text(json.dumps({"names": names, "paths": P.tolist(),
                                                 "real": real.tolist(), "picks": picks}))
    return res


def paths_chart(names, P, real, out_dir: Path) -> None:
    """Each model's 7 path scores (range bar) against its real later score (dot)."""
    order = np.argsort(P.mean(1))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(names))
    ax.vlines(x, P[order].min(1), P[order].max(1), color=plots.SERIES[0], lw=3, alpha=0.5,
              label="7 path scores, 2016-2022 (range)")
    ax.scatter(x, P[order].mean(1), s=14, color=plots.SERIES[0], label="path mean")
    ax.scatter(x, real[order], s=22, color=plots.SERIES[1], zorder=3,
               label="real score, Jan 2023 - Nov 2025")
    ax.axhline(0, color=plots.TEXT, lw=1)
    ax.set_xticks([])
    ax.set_xlabel("48 models, sorted by path mean")
    ax.set_ylabel("mean daily rank IC")
    ax.set_title("Combinatorial purged CV: the spread of 7 paths against the real later score",
                 loc="left")
    ax.legend(frameon=False, loc="upper left")
    plots.save(fig, out_dir, "paths")


REPORTS = {"splits": report_splits, "leak": report_leak, "features": report_features,
           "check": report_check, "sizing": report_sizing, "paths": report_paths}
