"""Run all four canaries and write charts plus a JSON of the numbers to reports/.

    python studies/canaries.py            # full run (a few minutes)
    python studies/canaries.py --quick    # smaller simulation for CI / make reproduce
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skeptic import canaries

OUT = Path(__file__).resolve().parents[1] / "reports"
BLUE, GREY, RED = "#1f5fa8", "#8a8a8a", "#b23a3a"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def main(quick: bool) -> dict:
    global OUT
    if quick:  # never overwrite the published full-run outputs
        OUT = OUT / "quick"
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    # 1. Noise canary
    n = canaries.noise_canary(n_strategies=1000, n_obs=1260, seed=0)
    res["noise"] = {"n_strategies": n.n_strategies, "n_obs": n.n_obs, "pass_rate": n.pass_rate,
                    "best_sr_annual": n.best_sr * np.sqrt(252), "best_psr": n.best_psr,
                    "best_dsr": n.best_dsr}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = ["No correction", "Bonferroni", "Holm", "Benjamini-Hochberg"]
    vals = [100 * n.pass_rate[k] for k in ("none", "bonferroni", "holm", "bh")]
    bars = ax.bar(labels, vals, color=[RED, BLUE, BLUE, BLUE], width=0.6)
    ax.axhline(5, color=GREY, ls="--", lw=1)
    ax.text(1.5, 5.15, "5% nominal level", color=GREY, fontsize=8, ha="center", va="bottom")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("zero-skill strategies passing (%)")
    ax.set_title("Noise canary: 1,000 zero-skill strategies, 5 years of daily returns", fontsize=10)
    _style(ax)
    fig.savefig(OUT / "canary_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Planted signal canary: power by true Sharpe and sample length
    srs = [0.5, 1.0, 1.5, 2.0, 3.0]
    years = [0.5, 1, 2, 5, 10]
    sims = 150 if quick else 600
    power = {f"N={k}": [[canaries.planted_power(s, y, n_trials=k, n_sims=sims, seed=7) for y in years]
                        for s in srs] for k in (1, 100)}
    res["planted"] = {"sr_annual": srs, "years": years, "power": power}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (k, grid) in zip(axes, power.items(), strict=True):
        for s, row in zip(srs, grid, strict=True):
            ax.plot(years, np.array(row) * 100, marker="o", ms=3, label=f"SR {s:g}")
        ax.set_xscale("log")
        ax.set_xticks(years, [f"{y:g}" for y in years])
        ax.set_xlabel("years of daily data")
        ax.set_title("1 trial (PSR)" if k == "N=1" else "best of 100 trials (deflated)", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("detection rate (%)")
    axes[1].legend(frameon=False, fontsize=8, title="true annual Sharpe", title_fontsize=8)
    fig.suptitle("Planted signal canary: how often a real edge is detected at 5%", fontsize=10)
    fig.savefig(OUT / "canary_power.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Leakage canary (averaged over seeds; a random walk is noisy)
    seeds = range(3 if quick else 10)
    lk = [canaries.leakage_canary(n=4000 if quick else 6000, seed=s) for s in seeds]
    leak = {k: float(np.mean([getattr(x, k) for x in lk])) for k in lk[0].__dataclass_fields__}
    leak_sd = {k: float(np.std([getattr(x, k) for x in lk])) for k in lk[0].__dataclass_fields__}
    res["leakage"] = {"mean_accuracy": leak, "sd": leak_sd, "n_seeds": len(lk)}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = ["Lookahead feature,\nnaive CV", "Same feature,\npoint-in-time",
             "Overlapping labels,\nshuffled K-fold", "Overlapping labels,\npurged K-fold"]
    keys = ["naive_cv_lookahead", "pit_aligned_lookahead", "shuffled_kfold_overlap", "purged_kfold_overlap"]
    v = [100 * leak[k] for k in keys]
    e = [100 * leak_sd[k] for k in keys]
    ax.bar(names, v, yerr=e, color=[RED, BLUE, RED, BLUE], width=0.6, capsize=3)
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.set_ylim(40, max(v) + 8)
    ax.set_ylabel("cross-validated accuracy (%)")
    ax.set_title("Leakage canary on a random walk (true accuracy is 50%)", fontsize=10)
    for i, val in enumerate(v):
        ax.text(i, val + e[i] + 0.6, f"{val:.1f}%", ha="center", fontsize=9)
    _style(ax)
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(OUT / "canary_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Best-of-N
    ns = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    b = canaries.best_of_n(ns, n_obs=252, n_sims=60 if quick else 200)
    res["best_of_n"] = {k: np.asarray(val).tolist() for k, val in b.items()}
    fig, ax1 = plt.subplots(figsize=(6.4, 3.4))
    ax1.plot(b["n"], b["best_sr_mean"], color=RED, marker="o", ms=3, label="best in-sample Sharpe (mean)")
    ax1.plot(b["n"], b["theory"], color=RED, ls=":", lw=1, label="theory (False Strategy Theorem)")
    ax1.set_xscale("log")
    ax1.set_xlabel("number of zero-skill strategies tried (N), 1 year of daily data each")
    ax1.set_ylabel("annualised Sharpe of the winner", color=RED)
    ax2 = ax1.twinx()
    ax2.plot(b["n"], b["psr_of_best"], color=GREY, marker="s", ms=3, label="PSR of winner (ignores N)")
    ax2.plot(b["n"], b["dsr_of_best"], color=BLUE, marker="s", ms=3, label="deflated Sharpe of winner")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("probability skill > 0")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.set_title("Best-of-N: optimising over noise manufactures Sharpe", fontsize=10)
    fig.savefig(OUT / "canary_best_of_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (OUT / "canaries.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    r = main(ap.parse_args().quick)
    print(json.dumps({"noise": r["noise"], "leakage": r["leakage"]["mean_accuracy"]}, indent=2))
