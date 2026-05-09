"""
Generate animated GIF assets for ETL monitoring visualizations.

Produces:
  assets/mw_monitoring.gif     — MetaWorld pick-place-wall dual-predicate trace
  assets/sorting_monitoring.gif — Sorting 2-phase predicate trace

Usage:
    python generate_assets.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

OUT = Path("assets")
OUT.mkdir(exist_ok=True)

ETL_COLOR    = "#2563EB"
PLACE_COLOR  = "#7C3AED"
GT_COLOR     = "#64748B"
FONT         = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── MetaWorld: dual-predicate (grasp → place) monitoring ─────────────────────
def gif_mw_monitoring():
    rng = np.random.default_rng(42)
    T = 101
    t = np.arange(T)

    raw_grasp = 4.0 - 3.5 * (1 / (1 + np.exp(-0.25 * (t - 22))))
    d_grasp   = np.clip(raw_grasp + rng.normal(0, 0.06, T), 0, None)

    raw_place = 4.2 - 3.8 * (1 / (1 + np.exp(-0.20 * (t - 62))))
    d_place   = np.clip(raw_place + rng.normal(0, 0.07, T), 0, None)

    tau_grasp = 2.85
    tau_place = 2.82

    pred_grasp = (d_grasp <= tau_grasp).astype(float)
    pred_place = (d_place <= tau_place).astype(float)
    gt_grasp   = (t >= 22).astype(float)
    gt_place   = (t >= 60).astype(float)

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 6.0),
                             gridspec_kw={"height_ratios": [2.2, 2.2, 0.6, 0.6]})
    fig.suptitle("ETL monitoring — mw-pick-place-wall", fontsize=11, y=1.01)

    # Panel 0: grasp distance
    ax0 = axes[0]
    ax0.set_xlim(0, T - 1); ax0.set_ylim(-0.1, 4.8)
    ax0.axhline(tau_grasp, color="red", ls="--", lw=1.2,
                label=rf"$\tau_A$={tau_grasp}")
    line_g, = ax0.plot([], [], color=ETL_COLOR, lw=1.4,
                       label=r"$\mathrm{dist}(z_t,z_A)$")
    ax0.set_ylabel(r"$\mathrm{dist}(z_t,z_A)$" + "\n(grasp)", fontsize=9)
    ax0.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax0.set_xticklabels([])

    # Panel 1: place distance
    ax1 = axes[1]
    ax1.set_xlim(0, T - 1); ax1.set_ylim(-0.1, 5.0)
    ax1.axhline(tau_place, color="red", ls="--", lw=1.2,
                label=rf"$\tau_B$={tau_place}")
    line_p, = ax1.plot([], [], color=PLACE_COLOR, lw=1.4,
                       label=r"$\mathrm{dist}(z_t,z_B)$")
    ax1.set_ylabel(r"$\mathrm{dist}(z_t,z_B)$" + "\n(place)", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax1.set_xticklabels([])

    # Panels 2-3: predicate bars
    bar_axes = [axes[2], axes[3]]
    data_pairs = [
        (pred_grasp, gt_grasp, "pred A", "GT A"),
        (pred_place, gt_place, "pred B", "GT B"),
    ]
    for ax, (_, _, lp, lg) in zip(bar_axes, data_pairs):
        ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([lg, lp], fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    axes[-1].set_xticks([0, 25, 50, 75, 100])

    # bar containers updated each frame
    bar_collections = [[], []]

    fig.tight_layout(h_pad=0.4)

    def init():
        line_g.set_data([], [])
        line_p.set_data([], [])
        for bc in bar_collections:
            for b in bc:
                b.remove()
            bc.clear()
        return line_g, line_p

    def update(frame):
        f = frame + 1
        line_g.set_data(t[:f], d_grasp[:f])
        line_p.set_data(t[:f], d_place[:f])

        for idx, (pred, gt, ax) in enumerate(zip(
            [pred_grasp, pred_place],
            [gt_grasp,   gt_place],
            [axes[2],    axes[3]],
        )):
            for b in bar_collections[idx]:
                b.remove()
            bar_collections[idx].clear()
            for j in range(f - 1):
                if pred[j]:
                    b = ax.barh(1.5, 1, left=j, height=0.6,
                                color=ETL_COLOR, align="center")
                    bar_collections[idx].append(b[0])
                if gt[j]:
                    b = ax.barh(0.5, 1, left=j, height=0.6,
                                color=GT_COLOR, align="center")
                    bar_collections[idx].append(b[0])
        return line_g, line_p

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        interval=60, blit=False)
    ani.save(OUT / "mw_monitoring.gif", writer=PillowWriter(fps=18))
    plt.close(fig)
    print("  saved assets/mw_monitoring.gif")


# ── Sorting: 2-phase (push block 1 → push block 2) monitoring ────────────────
def gif_sorting_monitoring():
    rng = np.random.default_rng(7)
    T = 121
    t = np.arange(T)

    raw_b1 = 3.8 - 3.3 * (1 / (1 + np.exp(-0.22 * (t - 30))))
    d_b1   = np.clip(raw_b1 + rng.normal(0, 0.07, T), 0, None)

    raw_b2 = 4.0 - 3.5 * (1 / (1 + np.exp(-0.20 * (t - 75))))
    d_b2   = np.clip(raw_b2 + rng.normal(0, 0.07, T), 0, None)

    tau_b1 = 2.50
    tau_b2 = 2.60

    pred_b1 = (d_b1 <= tau_b1).astype(float)
    pred_b2 = (d_b2 <= tau_b2).astype(float)
    gt_b1   = (t >= 30).astype(float)
    gt_b2   = (t >= 75).astype(float)

    BLUE2  = "#3B82F6"

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 6.0),
                             gridspec_kw={"height_ratios": [2.2, 2.2, 0.6, 0.6]})
    fig.suptitle("ETL monitoring — Sorting (2-phase)", fontsize=11, y=1.01)

    ax0 = axes[0]
    ax0.set_xlim(0, T - 1); ax0.set_ylim(-0.1, 4.6)
    ax0.axhline(tau_b1, color="red", ls="--", lw=1.2,
                label=rf"$\tau_1$={tau_b1}")
    line_b1, = ax0.plot([], [], color=ETL_COLOR, lw=1.4,
                        label=r"$\mathrm{dist}(z_t,z_1)$")
    ax0.set_ylabel(r"$\mathrm{dist}(z_t,z_1)$" + "\n(block 1)", fontsize=9)
    ax0.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax0.set_xticklabels([])

    ax1 = axes[1]
    ax1.set_xlim(0, T - 1); ax1.set_ylim(-0.1, 4.8)
    ax1.axhline(tau_b2, color="red", ls="--", lw=1.2,
                label=rf"$\tau_2$={tau_b2}")
    line_b2, = ax1.plot([], [], color=BLUE2, lw=1.4,
                        label=r"$\mathrm{dist}(z_t,z_2)$")
    ax1.set_ylabel(r"$\mathrm{dist}(z_t,z_2)$" + "\n(block 2)", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax1.set_xticklabels([])

    bar_axes = [axes[2], axes[3]]
    data_pairs = [
        (pred_b1, gt_b1, "pred 1", "GT 1"),
        (pred_b2, gt_b2, "pred 2", "GT 2"),
    ]
    bar_colors_pred = [ETL_COLOR, BLUE2]
    for ax, (_, _, lp, lg) in zip(bar_axes, data_pairs):
        ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([lg, lp], fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    axes[-1].set_xticks([0, 30, 60, 90, 120])

    bar_collections = [[], []]
    fig.tight_layout(h_pad=0.4)

    def init():
        line_b1.set_data([], [])
        line_b2.set_data([], [])
        for bc in bar_collections:
            for b in bc:
                b.remove()
            bc.clear()
        return line_b1, line_b2

    def update(frame):
        f = frame + 1
        line_b1.set_data(t[:f], d_b1[:f])
        line_b2.set_data(t[:f], d_b2[:f])

        preds = [pred_b1, pred_b2]
        gts   = [gt_b1,   gt_b2]
        axs   = [axes[2], axes[3]]
        for idx in range(2):
            for b in bar_collections[idx]:
                b.remove()
            bar_collections[idx].clear()
            for j in range(f - 1):
                if preds[idx][j]:
                    b = axs[idx].barh(1.5, 1, left=j, height=0.6,
                                      color=bar_colors_pred[idx], align="center")
                    bar_collections[idx].append(b[0])
                if gts[idx][j]:
                    b = axs[idx].barh(0.5, 1, left=j, height=0.6,
                                      color=GT_COLOR, align="center")
                    bar_collections[idx].append(b[0])
        return line_b1, line_b2

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        interval=60, blit=False)
    ani.save(OUT / "sorting_monitoring.gif", writer=PillowWriter(fps=18))
    plt.close(fig)
    print("  saved assets/sorting_monitoring.gif")


if __name__ == "__main__":
    print("Generating monitoring GIFs...")
    gif_mw_monitoring()
    gif_sorting_monitoring()
    print("Done.")
