"""MTM-daily marking with full-coverage enforcement.

Bug classes addressed:
  #1 exit-day lumping   — whole-trade P&L booked on exit day inflates Sharpe 2-2.4x
                          for multi-day holds. Fix: one mark per held day.
  #4 missing-leg deferral — a leg with no quote on a stress day silently lumps its
                          move onto the next quoted day (tail-day timing inverts).
                          Fix: re-mark quote-less days (carry + intrinsic shift,
                          floored at intrinsic) and ASSERT one row per held day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mtm_daily_marks(day_value: dict[int, float], entry_day: int, exit_day: int,
                    entry_value: float, sign: int, trading_days: np.ndarray,
                    intrinsic: dict[int, float] | None = None) -> dict[int, float]:
    """Daily P&L increments for one leg over EVERY trading day in [entry_day, exit_day].

    day_value    {day -> mark} on days the leg has a quote (must include exit_day)
    entry_value  value basis at entry (increment 0 at entry before frictions)
    sign         +1 long, -1 short
    intrinsic    optional {day -> intrinsic}; quote-less days are re-marked as
                 max(intrinsic, prev + d(intrinsic)) — model-free, auditable.
    Increments telescope: their sum equals sign*(final - entry_value) exactly,
    so interior re-marks can never change the per-trade total, only its timing.
    """
    lo = int(np.searchsorted(trading_days, entry_day))
    hi = int(np.searchsorted(trading_days, exit_day))
    if not (lo < len(trading_days) and trading_days[lo] == entry_day
            and hi < len(trading_days) and trading_days[hi] == exit_day):
        raise AssertionError("entry/exit day not on the trading-day grid")
    days = trading_days[lo:hi + 1]
    vals = np.empty(len(days))
    prev = entry_value
    for i, d in enumerate(days):
        v = day_value.get(int(d))
        if v is None:
            if intrinsic is None:
                raise AssertionError(
                    f"day {d} has no quote and no intrinsic re-mark supplied "
                    f"(bug class #4: missing-leg deferral)")
            iv = intrinsic.get(int(d), 0.0)
            iv_prev = intrinsic.get(int(days[i - 1]), 0.0) if i else 0.0
            v = max(iv, prev + (iv - iv_prev))
        vals[i] = v
        prev = v
    gains = sign * (vals - entry_value)
    out = {int(days[0]): float(gains[0])}
    for i in range(1, len(days)):
        out[int(days[i])] = float(gains[i] - gains[i - 1])
    return out


def assert_full_coverage(marks: pd.DataFrame, trades: pd.DataFrame,
                         trading_days: np.ndarray) -> None:
    """One mark row per (trade, day) for every trading day in [entry_day, exit_day].

    marks:  columns [trade_id, day, pnl]
    trades: columns [trade_id, entry_day, exit_day]
    Raises AssertionError naming the offending trades (bug classes #1 and #4).
    """
    got = marks.groupby("trade_id").agg(n=("day", "size"), nu=("day", "nunique"),
                                        lo=("day", "min"), hi=("day", "max"))
    t = trades.set_index("trade_id")
    exp = (np.searchsorted(trading_days, t.exit_day.values, side="right")
           - np.searchsorted(trading_days, t.entry_day.values, side="left"))
    got = got.reindex(t.index)
    bad = ((got.n.values != exp) | (got.nu.values != exp)
           | (got.lo.values != t.entry_day.values)
           | (got.hi.values != t.exit_day.values))
    if bad.any():
        raise AssertionError(
            f"{int(bad.sum())} trades lack one-mark-per-held-day coverage, "
            f"e.g. {t.index[bad][:5].tolist()} (bug classes #1/#4)")
