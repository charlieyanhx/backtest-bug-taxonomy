"""P1 exhibits: (1) identical dollars, different Sharpe; (2) inflation vs shock commonality."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
ANN = np.sqrt(252.0)

def simulate(rho, n_trades=120, hold=10, n_days=252, seed=42, n_crash=5):
    """rho = weight on the COMMON daily factor (0 = iid idiosyncratic, 1 = pure common)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.15, 1.0, n_days)
    factor[rng.choice(n_days, n_crash, replace=False)] -= 12.0
    lump = np.zeros(n_days); mtm = np.zeros(n_days)
    for _ in range(n_trades):
        e = int(rng.integers(0, n_days - hold - 1))
        idio = rng.normal(0.4, 1.0, hold + 1)
        daily = rho * factor[e:e + hold + 1] + (1 - rho) * idio
        lump[e + hold] += daily.sum()
        mtm[e:e + hold + 1] += daily
    sh = lambda x: x.mean() / x.std(ddof=1) * ANN
    return sh(lump), sh(mtm), lump, mtm

# Panel A: the canonical case
sl, sm, lump, mtm = simulate(1.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
a1.plot(np.cumsum(mtm), color="#31567a", lw=1.2, label=f"MTM-daily (Sharpe {sm:.2f})")
a1.plot(np.cumsum(lump), color="#a63d40", lw=1.2, ls="--", label=f"exit-day (Sharpe {sl:.2f})")
a1.set_title("Identical totals, different daily series", fontsize=9)
a1.set_xlabel("trading day"); a1.set_ylabel("cumulative P&L")
a1.legend(frameon=False, fontsize=7.5); a1.spines[["top","right"]].set_visible(False)
a2.hist(mtm, bins=40, color="#31567a", alpha=0.75, label="MTM-daily")
a2.hist(lump, bins=40, color="#a63d40", alpha=0.6, label="exit-day")
a2.set_title("The crash days the lumping hides", fontsize=9)
a2.set_xlabel("daily P&L"); a2.set_ylabel("days")
a2.legend(frameon=False, fontsize=7.5); a2.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_p1_lumping.pdf")

# Panel B: inflation ratio vs shock commonality (the novel claim, computed)
rhos = np.linspace(0.0, 1.0, 11)
ratios = []
for r in rhos:
    vals = [simulate(r, seed=s)[:2] for s in range(40)]
    ratios.append(np.median([a / b for a, b in vals]))
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(rhos, ratios, "o-", color="#31567a")
ax.axhline(1.0, color="black", lw=0.8, ls=":")
ax.text(0.02, 1.02, "no inflation", fontsize=7.5, color="#555")
ax.set_xlabel("weight on the common daily shock  ($\\rho$)")
ax.set_ylabel("exit-day Sharpe / MTM-daily Sharpe")
ax.set_title("Exit-day lumping inflates only when positions share shocks", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_p1_commonality.pdf")
print("ratio at rho=0 (iid):", round(ratios[0], 3), " at rho=1 (common):", round(ratios[-1], 3))

# Panel B result, part 2: the GLM fingerprint. Lag-1 autocorrelation of both series vs
# commonality. The exit-day series' autocorr RISES with rho (manufactured by the convention);
# the MTM-daily series' FALLS (an overlapping-hold artifact that common shocks wash out).
# Both numbers are quoted in the paper; the second is why the diagnostic must be conditional.
def ac1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

print("\nlag-1 autocorrelation (median over the same 40 seeds):")
print(f"{'rho':>5} {'exit-day':>10} {'MTM-daily':>11}")
for r in (0.0, 0.5, 1.0):
    L = [ac1(simulate(r, seed=s)[2]) for s in range(40)]
    M = [ac1(simulate(r, seed=s)[3]) for s in range(40)]
    print(f"{r:>5.1f} {np.median(L):>+10.3f} {np.median(M):>+11.3f}")
