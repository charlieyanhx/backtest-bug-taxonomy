"""Synthetic reproductions for bug classes 5, 9, 10, 11 — completing the twelve.

All data synthetic. Each test: reproduce the bug, then show the invariant catches/fixes it.
"""
import numpy as np
import pandas as pd
import pytest

from opentape import three_lines, full_calendar_sharpe

RNG = np.random.default_rng(7)


class TestBug5MidFillFantasy:
    """A strategy whose edge is smaller than the spread: profitable at mid,
    losing at cross. One accounting line hides this; three lines expose it."""

    def test_three_lines_expose_sub_spread_edge(self):
        n = 2000
        edge = 0.05                       # true mid-to-mid edge per trade
        half = 0.12                       # half-spread (> edge: untradeable)
        l1s, l3s = [], []
        for _ in range(n):
            em = 10.0
            xm = em + edge + RNG.normal(0, 0.5)
            l1, l2, l3 = three_lines(em, xm, em - half, em + half,
                                     xm - half, xm + half, side=1)
            l1s.append(l1); l3s.append(l3)
        assert np.mean(l1s) > 0                        # the mid line says "edge"
        assert np.mean(l3s) < 0                        # the cross line says "no"
        # the L1-L3 gap is exactly the round-trip spread — the friction sensitivity
        assert abs((np.mean(l1s) - np.mean(l3s)) - 2 * half) < 0.02


class TestBug9WrongInstrumentLookup:
    """Forward P&L looked up by delta-bucket instead of exact contract: as spot
    moves, the bucket re-maps to a different strike and the 'position' mutates."""

    def _panel(self):
        # two strikes; spot moves so the "0.30-delta bucket" flips from K=95 to K=105
        return pd.DataFrame({
            "day":    [0, 0, 1, 1],
            "strike": [95.0, 105.0, 95.0, 105.0],
            "delta":  [0.30, 0.15, 0.45, 0.30],     # spot fell: deltas rose
            "mid":    [2.00, 0.80, 3.50, 1.60],
        })

    def test_bucket_join_mutates_the_position(self):
        p = self._panel()
        # bucket lookup: "the ~0.30-delta contract" each day
        day0 = p[(p.day == 0)].iloc[(p[p.day == 0].delta - 0.30).abs().argmin()]
        day1 = p[(p.day == 1)].iloc[(p[p.day == 1].delta - 0.30).abs().argmin()]
        bucket_pnl = day1.mid - day0.mid              # 1.60 - 2.00 = -0.40
        # exact-identity lookup: the SAME strike both days
        exact = p[p.strike == day0.strike].sort_values("day")
        exact_pnl = exact.mid.iloc[1] - exact.mid.iloc[0]   # 3.50 - 2.00 = +1.50
        assert day0.strike != day1.strike             # the bucket silently switched
        assert bucket_pnl < 0 < exact_pnl             # opposite conclusions
        # invariant: joins must be on exact identity — enforced by construction
        merged = exact.drop_duplicates("strike")
        assert len(merged) == 1


class TestBug10SameSnapshotExecution:
    """Filling at the quote that generated the signal embeds the signal in the
    fill. With a mean-reverting mid, 'buy the dip at the dip quote' earns the
    dip back by construction; next-quote fills remove the artifact."""

    def test_same_snap_fill_manufactures_edge(self):
        n = 20000
        noise = RNG.normal(0, 1.0, n)                 # i.i.d. quote noise
        mid = 100 + noise                             # mean-reverting around 100
        signal = noise < -1.0                         # "dip" detected AT the quote
        idx = np.where(signal[:-2])[0]
        same_snap = (mid[idx + 1] - mid[idx]).mean()      # fill at the signal quote
        next_quote = (mid[idx + 2] - mid[idx + 1]).mean() # fill at the NEXT quote
        assert same_snap > 0.5                        # huge fake edge (buys the noise)
        assert abs(next_quote) < 0.1                  # gone at the honest fill


class TestBug11CalendarIndexedDifferencing:
    """diff() on calendar dates: weekend gaps make 'daily' changes 3-day changes,
    and a 48-hour hold silently becomes a 2-trading-day hold or spans a weekend."""

    def test_calendar_diff_mixes_horizons(self):
        dates = pd.bdate_range("2026-01-05", periods=10)      # business days
        cal = pd.date_range(dates[0], dates[-1])              # calendar days
        px = pd.Series(np.arange(len(dates), dtype=float), index=dates)
        on_cal = px.reindex(cal)                              # NaN weekends
        d_cal = on_cal.diff()
        d_trad = px.diff()
        # calendar diff: Mondays produce NaN (prev day missing) — dropped silently
        mondays = [d for d in dates if d.weekday() == 0][1:]
        assert d_cal.loc[mondays].isna().all()
        # trading-grid diff: every day after the first has a valid 1-step change
        assert d_trad.iloc[1:].notna().all()
        # a "48-hour" hold entered Friday exits Sunday on the calendar grid (no data),
        # but exits Tuesday on the trading grid — a different trade entirely
        fri = dates[4]
        cal_exit = fri + pd.Timedelta(hours=48)
        assert cal_exit not in px.index               # the calendar exit doesn't exist
        trad_exit = dates[dates.get_loc(fri) + 2]
        assert trad_exit.weekday() == 1               # Tuesday
