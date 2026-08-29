"""Structural invariants beyond marking and lines.

Bug classes addressed:
  #2  trajectory truncation — exits located by array index read stale values when a
      leg drops out of the panel mid-hold (a "12+ Sharpe" was 100% this artifact).
  #3  boundary drops — period-split tapes silently drop cross-boundary trades
      (non-randomly: the long, escalating holds).
  #8  retry-until-win / dropped candidates — trade count must tie the entry log.
  #12 attribution failure — dollars, timing and coverage all correct, yet the P&L
      comes from an exposure the strategy was not testing. Detectors: per-leg
      attribution + side-randomized negative control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assert_calendar_exit(entry_day: int, hold_days: int,
                         leg_days: np.ndarray, trading_days: np.ndarray) -> int:
    """Return the exit DAY located on the calendar, never by trajectory index.

    leg_days: trading days on which this leg actually has data. If the leg
    disappears before the target day, the exit is its LAST REAL day (force-close),
    never index entry+hold into a truncated array (bug class #2).
    """
    tgt_i = int(np.searchsorted(trading_days, entry_day)) + hold_days
    target = int(trading_days[min(tgt_i, len(trading_days) - 1)])
    avail = leg_days[(leg_days >= entry_day) & (leg_days <= target)]
    if len(avail) == 0:
        raise AssertionError("leg has no data on/after entry (bad join)")
    return int(avail[-1])


def assert_trade_count_ties(n_tape: int, n_entry_log: int) -> None:
    """Every candidate that entered must appear in the tape — force-close, never
    drop (bug classes #3 and #8)."""
    if n_tape != n_entry_log:
        raise AssertionError(
            f"tape has {n_tape} trades but the entry log has {n_entry_log}: "
            f"{n_entry_log - n_tape} candidates were dropped (bug classes #3/#8)")


def attribution_decompose(marks: pd.DataFrame, leg_col: str = "leg") -> pd.Series:
    """Total P&L by leg. If the leg the THESIS lives in contributes a minority of
    P&L (or a loss), the tape is measuring something else (bug class #12)."""
    return marks.groupby(leg_col).pnl.sum().sort_values()


def negative_control_sides(run_tape, sides: np.ndarray, n_draws: int = 3,
                           seed: int = 0) -> dict:
    """Re-run a tape with randomized / constant sides, identical entries.

    run_tape(sides_array) -> float total P&L. Returns the signal total against
    random and always-short/always-long controls. If a constant side beats the
    signal, the signal is not the source of the P&L (bug class #12).
    """
    rng = np.random.default_rng(seed)
    out = {"signal": run_tape(sides)}
    out["random"] = float(np.mean(
        [run_tape(rng.choice([-1, 1], size=len(sides))) for _ in range(n_draws)]))
    out["always_short"] = run_tape(np.full(len(sides), -1))
    out["always_long"] = run_tape(np.full(len(sides), 1))
    best_const = max(out["always_short"], out["always_long"])
    out["signal_beats_constant"] = out["signal"] > best_const
    return out


def best_day_check(daily: pd.Series, spot_returns: pd.Series,
                   k: int = 5) -> pd.DataFrame:
    """The k best P&L days next to the underlying's move that day.

    A short-vol book whose BEST days are the biggest crash days is broken by
    inspection (the missing-leg deferral was caught exactly this way)."""
    top = daily.nlargest(k)
    return pd.DataFrame({"pnl": top, "spot_ret": spot_returns.reindex(top.index)})
