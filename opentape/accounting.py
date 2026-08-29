"""Three accounting lines, full-calendar annualization, sign check.

Bug classes addressed:
  #5 mid-fill fantasy   — a single mid-based line let modeled fills reach live deployment.
                          Fix: Line 3 (cross) is the only go/no-go number.
  #6 active-day annualization — Sharpe on active days only inflates up to 8x.
                          Fix: pad the daily series with 0 on inactive days.
  #7 sign bug           — net > gross means the spread was credited, not paid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252.0)


def three_lines(entry_mid, exit_mid, entry_bid, entry_ask, exit_bid, exit_ask,
                side: int, friction_aware: float = 0.5):
    """Per-trade P&L on the three lines. side=+1 buys at entry, sells at exit.

    Line 1 (mid): both fills at mid — the signal ceiling.
    Line 2 (friction-aware): pay `friction_aware` of the half-spread each way.
    Line 3 (cross): buy@ask / sell@bid, entry AND exit — the go/no-go floor.
    """
    he, hx = (entry_ask - entry_bid) / 2, (exit_ask - exit_bid) / 2
    l1 = side * (exit_mid - entry_mid)
    l2 = l1 - friction_aware * (he + hx)
    l3 = l1 - (he + hx)
    return l1, l2, l3


def full_calendar_sharpe(daily: pd.Series, calendar: pd.Index) -> float:
    """Annualized Sharpe on the FULL calendar: inactive days count as 0.

    Passing only active days is bug class #6 (inflates up to 8x on sparse books).
    """
    d = daily.reindex(calendar, fill_value=0.0)
    sd = d.std(ddof=1)
    return float(d.mean() / sd * ANN) if sd > 0 else float("nan")


def sign_check(line1_total: float, line3_total: float) -> None:
    """Line 3 must never exceed Line 1: the spread is always a cost."""
    if line3_total > line1_total + 1e-9:
        raise AssertionError(
            f"net {line3_total:.2f} > gross {line1_total:.2f}: spread credited "
            f"instead of paid (bug class #7)")
