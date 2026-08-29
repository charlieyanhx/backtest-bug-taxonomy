# Backtest Bug Taxonomy — `opentape`

Code and data-availability statement for:

> **Dollar-Correct, Time-Wrong: Twelve Backtest Bugs That Survive Reconciliation**
> Charlie Yan, 2026. [`paper/p1_dollar_correct_time_wrong.pdf`](paper/p1_dollar_correct_time_wrong.pdf)

## What the paper claims

A backtest whose P&L reconciles to the cent can still be wrong by 2–12× in risk-adjusted terms, or can be measuring an exposure the strategy never claimed. Twelve bug classes, each with a runtime-checkable invariant and a synthetic reproduction.

**Headline result.** Exit-day lumping inflates Sharpe **only** when concurrent positions share shocks: ratio 0.64 (i.i.d., lumping *deflates*) → 2.14 (fully common), bracketing the 2.1–2.4× observed in production.

## Reproducibility

**FULLY REPRODUCIBLE — no market data needed.** Every bug class ships as a synthetic failing-then-passing test. `pip install -e ".[test]" && pytest -q` reproduces all thirteen, including the paper's headline exhibit (exit-day lumping inflates the Sharpe ratio 0.64× under i.i.d. P&L and 2.14× under fully common shocks).

## What is here

`opentape/` — the library (marking, accounting, structural checks, attribution).
`tests/` — thirteen synthetic bug reproductions.
`paper/` — the paper and its two exhibits.

## Evidence conventions used throughout

Every performance figure in the paper carries its accounting basis inline. Unless labelled
otherwise: **line 3** = full cross-spread fills (buy at ask, sell at bid, every leg both ways),
ex-commission, marked to market daily, padded to the full business calendar. Figures labelled
**screen** are descriptive or information-coefficient statistics and are never annualised into a
Sharpe ratio. Numbers marked **invalid** appear only as invalidated examples, with the corrected
figure alongside.

All tests reported in the paper were pre-registered — horizons, controls, nulls and decision bars
fixed before execution — and deviations are recorded rather than edited away. Where pre-registration
documents exist in this repository they are included verbatim.

## Citation

```bibtex
@techreport{yan2026backtestbugtaxonomy,
  title  = {Dollar-Correct, Time-Wrong: Twelve Backtest Bugs That Survive Reconciliation},
  author = {Yan, Charlie},
  year   = {2026},
  type   = {Working paper}
}
```

## License

Code MIT (see `LICENSE`). The paper PDF is © 2026 Charlie Yan, all rights reserved.
