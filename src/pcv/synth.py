"""Synthetic stock panels with a known true edge, to check cross-validation against the truth."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Panel:
    X: np.ndarray  # rows x features
    y: np.ndarray  # label: return over day+1 .. day+h, minus that day's mean across stocks
    day: np.ndarray  # trading-day number of each row
    stock: np.ndarray


def panel(n_stocks: int, n_days: int, h: int, edge: float, phi: float = 0.98, seed: int = 0,
          k: int = 5, flip: bool = False, ret_ar: float = 0.0) -> Panel:
    """k slow features (AR(1), persistence `phi`, unit variance); feature 1 predicts the next day's
    return with daily rank-correlation about `edge` (sign flips halfway if `flip`); the rest are
    noise. Fat-tailed shocks (t, 4 df). `ret_ar` adds day-to-day momentum in returns."""
    rng = np.random.default_rng(seed)
    f = np.empty((n_days, n_stocks, k))
    f[0] = rng.standard_normal((n_stocks, k))
    e = rng.standard_normal((n_days, n_stocks, k)) * np.sqrt(1 - phi**2)
    for t in range(1, n_days):
        f[t] = phi * f[t - 1] + e[t]
    z = (f[:, :, 0] - f[:, :, 0].mean(1, keepdims=True)) / f[:, :, 0].std(1, keepdims=True)
    shock = rng.standard_t(4, (n_days, n_stocks)) / np.sqrt(2)
    sign = np.where(np.arange(n_days) >= n_days // 2, -1.0, 1.0) if flip else np.ones(n_days)
    r = np.zeros((n_days, n_stocks))
    for t in range(1, n_days):  # r[t]: the return on day t, driven by features known at t-1
        r[t] = sign[t - 1] * edge * z[t - 1] + np.sqrt(1 - edge**2) * shock[t] + ret_ar * r[t - 1]
    c = np.cumsum(r, axis=0)
    d = np.arange(n_days - h)  # rows with a full label window
    y = c[d + h] - c[d]
    y -= y.mean(1, keepdims=True)
    day = np.repeat(d, n_stocks)
    return Panel(f[d].reshape(-1, k), y.reshape(-1), day, np.tile(np.arange(n_stocks), len(d)))
