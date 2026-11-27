# Purged Cross-Validation

[![CI](https://github.com/Dhirajdas18/purged-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/purged-cv/actions/workflows/ci.yml)
[![Dashboard](https://github.com/Dhirajdas18/purged-cv/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/purged-cv/)

Cross-validation for stock-return models whose labels look days ahead, where ordinary k-fold trains on
days that overlap the test period. Splitters that purge those days, the leak measured on data with a known
answer, and the purge a quant-style model on NSE stocks needs (local bhavcopy archive, 2013 – Nov 2025;
only derived results are committed).

**Live split lab: [dhirajdas18.github.io/purged-cv](https://dhirajdas18.github.io/purged-cv/)**, pick a
method, a label length and an embargo, and see which trading days each split trains on.

## Features

- [x] Splitters: purged k-fold, embargo, walk-forward with a purge, combinatorial purged CV with backtest paths
- [ ] The leak on synthetic stock panels with a known true edge, including harder cases
- [ ] A quant-style model on NSE stocks: features and labels rebuilt and checked against the original
- [ ] Which cross-validation score predicts the score a model really gets later
- [ ] How big the purge and embargo need to be, by label length
- [ ] Combinatorial purged CV: a spread of out-of-sample results instead of one number
- [ ] One check on Jan 2023 – Nov 2025, after every rule is fixed on earlier data

## Findings

### 1. Purging costs at most 5% of the training days, even with 63-day labels

<a href="docs/img/splits.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/splits.png"><img src="docs/img/splits.gif" alt="Purged five-fold layout on the 2013-2022 calendar with 63-day labels, and the share of training days dropped by label length" width="720"></picture></a>

Each row is one stock on one trading day; its label is the return from the next day's open to the close
h trading days later, as in the reference model (h = 1, 5, 21, 63). Purged k-fold cuts the days into blocks; a
training day is dropped when its label window overlaps the test days' label windows, and an embargo drops
a few more days after them. Measured on the lake's trading calendar, Jan 2013 – Dec 2022 (2,468 days, the
data every rule here is chosen on), with 5 folds.

- **With 1-day labels nothing needs purging.** A 1-day label ends before the next test label starts, so
  purging drops 0.0% of the training days; a 5-day embargo drops 0.2%.
- **The cost grows with the label length, and stays small.** 5-day labels: 0.3% purged (0.5% with the
  embargo); 21-day: 1.6% (1.8%); 63-day: 5.0% (5.2%).
- **Walk-forward loses a little more to a wider gap.** Training only on the past, an exact purge leaves
  1,231 days on average for 5-day labels; the reference model's cut (train before day first test − (h + 5)) leaves
  1,225, the same as a purge plus a 6-day embargo.
- **Cheap is not the same as needed.** Whether those few days change the score is the next finding.

## Quick start

```bash
uv sync
PCV_LAKE_DIR=/path/to/lake uv run pcv report splits   # topics: splits
uv run python scripts/build_site.py                    # site/index.html from docs/data
uv run pytest
```

The findings need a local NSE bhavcopy archive with `panel.parquet` (`PCV_LAKE_DIR`); it is not committed,
and nothing after 26 Nov 2025 is read from it. Tests run on hand-made cases.

## Design

- `src/pcv/split.py`: `PurgedKFold`, `WalkForward`, `CombinatorialPurged` (with `paths()`), scikit-learn
  style: `split(X, y, groups=day)` with each row's trading-day number.
- `src/pcv/data.py`: the lake reader, filtered at read time to 26 Nov 2025; the trading calendar.
- `src/pcv/reports.py`, `plots.py`, `site.py`: report numbers and charts, chart style, the lab page.
- Conventions: a row's label covers days day+1 to day+h; rules are chosen on data up to Dec 2022 and
  checked once on Jan 2023 – Nov 2025; the reference model's last 365 days (from 27 Nov 2025) are never read.
- Data: NSE daily bhavcopy archive, local only.
- Charts: every chart is a PNG plus an animated GIF made by `pcv report`.

Research only, not investment advice. MIT licence.
