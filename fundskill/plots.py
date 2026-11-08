"""The headline chart: real t(alpha) distribution vs the 'pure luck' one."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bootstrap import BootstrapResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ACTUAL = "#2a78d6"   # blue: the real funds
LUCK = "#eb6834"     # orange: the zero-alpha (luck) world


def plot_skill_vs_luck(res: BootstrapResult, path: str, title: str = "Fund skill vs luck"):
    """Cumulative distribution of t(alpha): actual funds vs bootstrap.

    A real curve sitting to the RIGHT of the luck band in the top tail means
    the best funds beat what luck alone produces.
    """
    p = np.linspace(1, 99, 99)
    actual = np.nanpercentile(res.actual["t_alpha"].to_numpy(), p)
    sim = np.nanpercentile(res.sim_t, p, axis=1)          # (99, n_sims)
    lo, mid, hi = np.percentile(sim, [5, 50, 95], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_betweenx(p, lo, hi, color=LUCK, alpha=0.18, linewidth=0,
                     label="Luck only: 90% range of simulations")
    ax.plot(mid, p, color=LUCK, lw=2, label="Luck only: median simulation")
    ax.plot(actual, p, color=ACTUAL, lw=2, label="Actual funds")

    # Direct labels near the top-right tail, where the story is.
    # Direct labels at the line ends (top-right tail, where the story is).
    ax.annotate("Actual funds", (actual[-1], p[-1]), xytext=(0, -16),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.annotate("Luck only", (mid[-1], p[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=INK_2, fontsize=9)
    ax.set_ylim(0, 106)

    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel("t-statistic of alpha", color=INK_2, fontsize=10)
    ax.set_ylabel("Percentile of funds", color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
