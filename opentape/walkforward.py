"""Walk-forward invariants: was the validation honest, not how do I split.

Splitters already exist (purgedcv, mlfinlab). This module does the other half:
given whatever splits you used, it refuses the ones that leak, and it prices the
search you ran to get your headline.

Bug classes addressed:
  #13 future-fitted selection — a fold whose training window is not strictly in
      the past of its test window. Parameters chosen with any knowledge of the
      test period are not a forecast, they are a fit.
  #14 unembargoed adjacency — train ending at t and test starting at t+1 leaks
      whenever the label spans more than one period. The gap must be at least the
      signal's memory (overlap horizon), not one bar.
  #15 reused test set — each additional look at the same test fold selects on it.
      The bar that applies after k looks is not the bar that applied at k=1.
  #16 unadjusted sweep — a configuration chosen as the best of N is quoted at a
      threshold that assumed N=1. The selection-adjusted bar is what it must clear.

The statistics here reproduce the detectability table in "How Much Tail
Prediction Could We Have Detected?" (P3); `required_ic` is the same computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329

# Rank-residualised Spearman ICs have a Fisher standard error inflated over the
# Pearson 1/sqrt(n-3); P3 uses this factor and so does this module.
SPEARMAN_SE_FACTOR = 1.06


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def assert_past_only(train_idx: Sequence[int], test_idx: Sequence[int]) -> None:
    """Every test observation must lie strictly after every training one (#13)."""
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if train_idx.size == 0 or test_idx.size == 0:
        raise AssertionError("empty train or test fold: nothing was validated")
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise AssertionError(
            f"{overlap.size} observation(s) appear in both train and test "
            f"(first: {int(overlap[0])}): the model was fitted on its own test set"
        )
    if int(test_idx.min()) <= int(train_idx.max()):
        raise AssertionError(
            f"test begins at {int(test_idx.min())} but training runs to "
            f"{int(train_idx.max())}: parameters were selected with knowledge of "
            f"the test period, which makes this a fit rather than a forecast"
        )


def assert_embargoed(train_idx: Sequence[int], test_idx: Sequence[int],
                     embargo: int) -> None:
    """Require a gap of at least `embargo` observations between train and test (#14).

    `embargo` is the signal's memory - the label horizon, or the autocorrelation
    length of the target. A one-bar gap embargoes nothing when the label spans
    ten bars: the last training label still overlaps the first test period.
    """
    if embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")
    assert_past_only(train_idx, test_idx)
    gap = int(np.asarray(test_idx).min()) - int(np.asarray(train_idx).max()) - 1
    if gap < embargo:
        raise AssertionError(
            f"gap between train and test is {gap} observation(s) but the signal's "
            f"memory is {embargo}: the final training labels still overlap the test "
            f"period, so information crosses the split"
        )


@dataclass
class HoldoutLedger:
    """Count looks at each held-out fold, so reuse is visible rather than assumed (#15).

    Every evaluation against the same held-out fold is a selection on it. This
    does not forbid a second look - it records how many there were, so the bar
    can be adjusted and the count reported.
    """
    looks: Dict[str, int] = field(default_factory=dict)

    def record(self, fold: str) -> int:
        self.looks[fold] = self.looks.get(fold, 0) + 1
        return self.looks[fold]

    def assert_touched_once(self, fold: Optional[str] = None) -> None:
        folds = self.looks if fold is None else {fold: self.looks.get(fold, 0)}
        reused = {f: n for f, n in folds.items() if n > 1}
        if reused:
            detail = ", ".join(f"{f}: {n} looks" for f, n in sorted(reused.items()))
            raise AssertionError(
                f"test fold(s) evaluated more than once ({detail}). Each look "
                f"selects on the held-out data; quote the selection-adjusted bar "
                f"for n_trials = {max(reused.values())} or hold out fresh data"
            )

    def total_looks(self) -> int:
        return sum(self.looks.values())


# ---------------------------------------------------------------------------
# pricing the search
# ---------------------------------------------------------------------------

def required_ic(n_eff: int, n_tests: int = 1, power: float = 0.80,
                alpha: float = 0.05, spearman: bool = True) -> float:
    """Smallest |IC| detectable at `power`, Bonferroni-corrected over `n_tests`.

    Fisher-z on the correlation, two-sided. This is the computation behind P3's
    detectability table: a null from a large search is uninterpretable unless the
    searcher reports what the search could have detected.

    Args:
        n_eff: effective independent observations. With overlapping h-period
            labels this is roughly n / h, not n.
        n_tests: how many configurations were actually tried.
    """
    if n_eff <= 3:
        raise ValueError(f"n_eff must exceed 3 for the Fisher transform, got {n_eff}")
    if n_tests < 1:
        raise ValueError(f"n_tests must be at least 1, got {n_tests}")
    se = (SPEARMAN_SE_FACTOR if spearman else 1.0) / math.sqrt(n_eff - 3)
    z_alpha = norm.ppf(1 - alpha / n_tests / 2)
    z_power = norm.ppf(power)
    return float(np.tanh((z_alpha + z_power) * se))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected best Sharpe from `n_trials` of a strategy with no edge.

    Bailey and Lopez de Prado's expected-maximum term. `sharpe_std` is the
    dispersion of Sharpe estimates across the trials you ran. Any headline below
    this is what a search of that size produces from noise alone.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if sharpe_std < 0:
        raise ValueError(f"sharpe_std must be non-negative, got {sharpe_std}")
    if n_trials == 1:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sharpe_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def decay_ratio(in_sample: float, out_of_sample: float) -> float:
    """OOS / IS. Reported, never buried - it is the first number a reader wants.

    Returns nan when the in-sample figure is zero, rather than an infinity that
    propagates into a table.
    """
    if in_sample == 0:
        return float("nan")
    return float(out_of_sample) / float(in_sample)


def assert_clears_selection_bar(headline_sharpe: float, n_trials: int,
                                sharpe_std: float) -> None:
    """A config selected from a sweep must beat what the sweep yields on noise (#16)."""
    bar = expected_max_sharpe(n_trials, sharpe_std)
    if headline_sharpe <= bar:
        raise AssertionError(
            f"headline Sharpe {headline_sharpe:.3f} does not clear the "
            f"selection-adjusted bar {bar:.3f} for {n_trials} trials "
            f"(dispersion {sharpe_std:.3f}): a search this size produces that "
            f"much from noise alone"
        )
