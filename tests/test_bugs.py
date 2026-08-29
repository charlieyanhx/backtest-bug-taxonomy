"""Synthetic reproductions of the bug classes. Each test: (a) reproduce the bug and
show the inflation/misattribution, (b) show the invariant catches or fixes it.
All data synthetic — no market data anywhere.
"""
import numpy as np
import pandas as pd
import pytest

from opentape import (
    mtm_daily_marks, assert_full_coverage, full_calendar_sharpe, sign_check,
    assert_calendar_exit, assert_trade_count_ties, attribution_decompose,
    negative_control_sides,
)

DAYS = np.arange(0, 252)          # one synthetic year of trading days
CAL = pd.Index(DAYS)
RNG = np.random.default_rng(42)


def _multi_day_trades(n=120, hold=10):
    """Synthetic short-vol-like trades sharing a COMMON daily factor.

    The inflation mechanism is not iid noise (under iid, lumping does not inflate):
    it is the common shock across concurrent positions. MTM-daily books the crash
    day as one large loss across every open trade; exit-day marking disperses that
    same shock across many later exit dates, smoothing the series (the program
    record: worst day = 16-21x worst trade).
    """
    factor = RNG.normal(0.15, 1.0, len(DAYS))
    factor[RNG.choice(len(DAYS), 5, replace=False)] -= 12.0   # five crash days
    trades = []
    for i in range(n):
        e = int(RNG.integers(0, len(DAYS) - hold - 1))
        idio = RNG.normal(0.4, 0.8, hold + 1)
        daily = factor[e:e + hold + 1] + idio
        trades.append((i, DAYS[e], DAYS[e + hold], daily))
    return trades


class TestBug1ExitDayLumping:
    def test_exit_day_marking_inflates_sharpe(self):
        trades = _multi_day_trades()
        lump = pd.Series(0.0, index=CAL)
        mtm = pd.Series(0.0, index=CAL)
        for _, e, x, daily in trades:
            lump[x] += daily.sum()                       # the bug
            for d, v in zip(range(e, x + 1), daily):     # the fix
                mtm[d] += v
        s_lump = full_calendar_sharpe(lump, CAL)
        s_mtm = full_calendar_sharpe(mtm, CAL)
        assert abs(lump.sum() - mtm.sum()) < 1e-9        # dollars identical...
        assert s_lump > 1.5 * s_mtm                      # ...Sharpe inflated >1.5x

    def test_coverage_assertion_catches_it(self):
        marks = pd.DataFrame({"trade_id": [0], "day": [10], "pnl": [50.0]})
        trades = pd.DataFrame({"trade_id": [0], "entry_day": [0], "exit_day": [10]})
        with pytest.raises(AssertionError, match="coverage"):
            assert_full_coverage(marks, trades, DAYS)


class TestBug2TrajectoryTruncation:
    def test_index_exit_reads_stale_day(self):
        # leg quoted daily, then drops out of the panel after day 6
        leg_days = np.array([0, 1, 2, 3, 4, 5, 6])
        exit_day = assert_calendar_exit(entry_day=0, hold_days=10,
                                        leg_days=leg_days, trading_days=DAYS)
        assert exit_day == 6                              # force-close at last real day
        # the buggy pattern: index entry+hold into the truncated trajectory
        buggy = leg_days[min(0 + 10, len(leg_days) - 1)]
        assert buggy == 6  # same day here, but the VALUE read is day-6's, stale
        # while a naive iloc on a padded/stale array would read day 10's row —
        # the invariant is that the exit is located on the CALENDAR, asserted above.


class TestBug3And8DroppedTrades:
    def test_trade_count_must_tie(self):
        with pytest.raises(AssertionError, match="dropped"):
            assert_trade_count_ties(n_tape=97, n_entry_log=128)


class TestBug4MissingLegDeferral:
    def test_quoteless_crash_day_shifts_timing_not_total(self):
        # short put, quote missing on the crash day (day 2)
        tdays = np.arange(0, 4)
        quotes = {0: 5.0, 1: 5.5, 3: 20.0}               # day 2 missing
        intrinsic = {0: 0.0, 1: 0.0, 2: 12.0, 3: 14.0}   # crash lands on day 2
        inc = mtm_daily_marks(quotes, 0, 3, 5.0, sign=-1,
                              trading_days=tdays, intrinsic=intrinsic)
        assert abs(sum(inc.values()) - (-15.0)) < 1e-9   # total unchanged
        assert inc[2] < -10                              # loss ON the crash day
        assert inc[3] > -3                               # not lumped onto day 3
        with pytest.raises(AssertionError, match="missing-leg"):
            mtm_daily_marks(quotes, 0, 3, 5.0, -1, tdays, intrinsic=None)


class TestBug6ActiveDayAnnualization:
    def test_sparse_book_inflates_without_padding(self):
        daily = pd.Series(RNG.normal(5, 10, 25), index=CAL[:25])  # 25 active days
        s_active = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
        s_full = full_calendar_sharpe(daily, CAL)
        assert s_active > 2.0 * s_full


class TestBug7SignBug:
    def test_net_above_gross_raises(self):
        with pytest.raises(AssertionError, match="spread credited"):
            sign_check(line1_total=100.0, line3_total=140.0)


class TestBug12AttributionFailure:
    """The failure that survives every other check: dollars, timing, coverage all
    right — and the P&L is a premium/beta harvest, not the signal."""

    def _tape(self, sides, prem=2.0, noise=5.0, signal_edge=0.3, seed=1):
        rng = np.random.default_rng(seed)
        # every SHORT collects `prem`; the "signal" adds a small genuine edge
        true_sig = rng.choice([-1, 1], size=len(sides))
        pnl = np.where(sides < 0, prem, -prem) \
            + signal_edge * (sides == true_sig) \
            + rng.normal(0, noise, len(sides))
        return float(pnl.sum())

    def test_negative_control_catches_it(self):
        n = 4000
        rng = np.random.default_rng(2)
        sides = rng.choice([-1, 1], size=n)              # residual-fade-like: ~50/50
        res = negative_control_sides(lambda s: self._tape(s), sides)
        assert not res["signal_beats_constant"]          # always-short wins
        assert res["always_short"] > res["signal"] > res["random"]

    def test_leg_attribution_shows_the_source(self):
        marks = pd.DataFrame({
            "leg": ["option"] * 3 + ["hedge"] * 3,
            "pnl": [-40.0, -50.0, -39.0, 70.0, 65.0, 66.0]})
        att = attribution_decompose(marks)
        assert att["option"] < 0 < att["hedge"]          # thesis leg LOSES money
