"""Walk-forward invariants (#13-#16) and the statistics that price a search."""
import math

import numpy as np
import pytest

from opentape.walkforward import (
    HoldoutLedger, assert_clears_selection_bar, assert_embargoed, assert_past_only,
    decay_ratio, expected_max_sharpe, required_ic,
)


# --- #13 future-fitted selection -------------------------------------------

def test_clean_split_passes():
    assert_past_only(range(0, 100), range(110, 150))


def test_overlapping_folds_are_rejected():
    with pytest.raises(AssertionError, match="appear in both"):
        assert_past_only(range(0, 100), range(90, 150))


def test_test_before_train_is_rejected():
    with pytest.raises(AssertionError, match="fit rather than a forecast"):
        assert_past_only(range(100, 200), range(0, 50))


def test_empty_fold_is_rejected():
    with pytest.raises(AssertionError, match="nothing was validated"):
        assert_past_only([], range(10))


# --- #14 unembargoed adjacency ---------------------------------------------

def test_sufficient_embargo_passes():
    assert_embargoed(range(0, 100), range(110, 150), embargo=10)


def test_adjacent_folds_fail_a_ten_day_embargo():
    """The defect this exists for: train ends at 99, test starts at 100, and the
    labels span ten days - so the last training label overlaps the test window."""
    with pytest.raises(AssertionError, match="information crosses the split"):
        assert_embargoed(range(0, 100), range(100, 150), embargo=10)


def test_gap_exactly_equal_to_the_embargo_is_allowed():
    assert_embargoed(range(0, 100), range(110, 150), embargo=10)   # gap == 10


def test_gap_one_short_of_the_embargo_fails():
    with pytest.raises(AssertionError, match="memory is 10"):
        assert_embargoed(range(0, 100), range(109, 150), embargo=10)


def test_zero_embargo_still_requires_past_only():
    assert_embargoed(range(0, 100), range(100, 150), embargo=0)
    with pytest.raises(AssertionError):
        assert_embargoed(range(0, 100), range(99, 150), embargo=0)


def test_negative_embargo_is_a_usage_error():
    with pytest.raises(ValueError, match="non-negative"):
        assert_embargoed(range(10), range(20, 30), embargo=-1)


# --- #15 reused test set ----------------------------------------------------

def test_a_single_look_is_clean():
    led = HoldoutLedger()
    led.record("2023")
    led.assert_touched_once()


def test_second_look_is_flagged():
    led = HoldoutLedger()
    led.record("2023"); led.record("2023")
    with pytest.raises(AssertionError, match="more than once"):
        led.assert_touched_once()


def test_reuse_message_names_the_adjusted_trial_count():
    led = HoldoutLedger()
    for _ in range(7):
        led.record("2023")
    with pytest.raises(AssertionError, match="n_trials = 7"):
        led.assert_touched_once()


def test_folds_are_tracked_independently():
    led = HoldoutLedger()
    led.record("2022"); led.record("2023"); led.record("2023")
    led.assert_touched_once("2022")
    with pytest.raises(AssertionError):
        led.assert_touched_once("2023")
    assert led.total_looks() == 3


# --- required IC: reproduces P3 --------------------------------------------

def test_required_ic_reproduces_the_published_table():
    """P3, ten-day overlap: n_eff 116, Bonferroni over 1,450 tests, 80% power."""
    assert required_ic(116, 1450) == pytest.approx(0.460, abs=0.001)
    assert required_ic(233, 1450) == pytest.approx(0.335, abs=0.001)
    assert required_ic(15, 1450) == pytest.approx(0.910, abs=0.001)


def test_nominal_bar_is_far_lower_than_the_corrected_one():
    """The whole point of the accounting: quoting the nominal bar after a
    1,450-test search understates what was needed by roughly half."""
    assert required_ic(116, 1) == pytest.approx(0.272, abs=0.001)
    assert required_ic(116, 1450) > 1.6 * required_ic(116, 1)


def test_more_tests_raise_the_bar():
    bars = [required_ic(116, n) for n in (1, 10, 100, 1450)]
    assert bars == sorted(bars)


def test_more_data_lowers_the_bar():
    assert required_ic(500, 100) < required_ic(100, 100)


def test_required_ic_rejects_degenerate_input():
    with pytest.raises(ValueError, match="must exceed 3"):
        required_ic(3, 10)
    with pytest.raises(ValueError, match="at least 1"):
        required_ic(100, 0)


# --- #16 selection-adjusted Sharpe -----------------------------------------

def test_one_trial_has_no_selection_bar():
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_selection_bar_grows_with_the_search():
    bars = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1450)]
    assert bars == sorted(bars)


def test_selection_bar_scales_with_dispersion():
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2 * expected_max_sharpe(100, 0.5))


def test_a_headline_below_the_bar_is_rejected():
    with pytest.raises(AssertionError, match="from noise alone"):
        assert_clears_selection_bar(1.2, n_trials=1450, sharpe_std=0.5)


def test_a_headline_above_the_bar_passes():
    assert_clears_selection_bar(2.5, n_trials=1450, sharpe_std=0.5)


# --- decay ------------------------------------------------------------------

def test_decay_ratio():
    assert decay_ratio(2.0, 1.0) == 0.5
    assert decay_ratio(1.0, 1.5) == 1.5


def test_zero_in_sample_gives_nan_not_infinity():
    assert math.isnan(decay_ratio(0.0, 1.0))
