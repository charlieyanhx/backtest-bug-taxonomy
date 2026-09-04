"""opentape — reference implementation of honest options-backtest accounting.

Companion library to "Dollar-Correct, Time-Wrong" (P1). Sixteen named bug classes,
each with a runtime-checkable invariant and a synthetic reproduction in tests/.

Everything here is synthetic-data-safe: no market data ships with the library.
"""
from .marking import mtm_daily_marks, assert_full_coverage
from .accounting import three_lines, full_calendar_sharpe, sign_check
from .walkforward import (
    assert_past_only,
    assert_embargoed,
    assert_clears_selection_bar,
    HoldoutLedger,
    required_ic,
    expected_max_sharpe,
    decay_ratio,
)
from .checks import (
    assert_calendar_exit,
    assert_trade_count_ties,
    attribution_decompose,
    negative_control_sides,
    best_day_check,
)

__all__ = [
    "mtm_daily_marks", "assert_full_coverage",
    "three_lines", "full_calendar_sharpe", "sign_check",
    "assert_calendar_exit", "assert_trade_count_ties",
    "attribution_decompose", "negative_control_sides", "best_day_check",
    "assert_past_only", "assert_embargoed", "assert_clears_selection_bar",
    "HoldoutLedger", "required_ic", "expected_max_sharpe", "decay_ratio",
]
__version__ = "0.3.0"
