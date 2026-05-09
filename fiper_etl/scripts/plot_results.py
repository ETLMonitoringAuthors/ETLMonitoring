"""
plot_results.py
---------------
Generates two figures for the ETL vs FIPER baselines comparison:
  1. AUROC bar chart across all 5 tasks (main result)
  2. Push_chair diagnosis panel (episode-length discrimination)

Usage:
    python scripts/plot_results.py
"""

import json, glob, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── load results ──────────────────────────────────────────────────────────────

with open(ROOT / "data/results/etl_final_results.json") as f:
    results = json.load(f)

TASK_LABELS = {
    "stacking":   "Stacking\n(K=2)",
    "sorting":    "Sorting\n(K=2)",
    "push_t":     "Push-T\n(K=1)",
    "pretzel":    "Pretzel\n(K=2)",
    "push_chair": "Push-Chair\n(K=3)",
}
TASKS = list(TASK_LABELS.keys())

METHODS = ["ETL-or", "PCA-kmeans", "logpZO"]
COLORS  = ["#2166ac", "#d6604d", "#4dac26"]   # blue, red, green
HATCHES = ["", "//", ".."]

# FIPER Table 1 Accuracy (from paper, averaged over seeds) for reference
FIPER_ACC = {
    "stacking":   {"PCA-kmeans": 0.75, "logpZO": 0.69},
    "sorting":    {"PCA-kmeans": 0.56, "logpZO": 0.67},
    "push_t":     {"PCA-kmeans": 0.58, "logpZO": 0.55},
    "pretzel":    {"PCA-kmeans": 0.65, "logpZO": 0.65},
    "push_chair": {"PCA-kmeans": 0.50, "logpZO": 0.92},
}

# ── Figure 1: AUROC bar chart ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                         gridspec_kw={"width_ratios": [3, 2]})

ax = axes[0]
n_tasks   = len(TASKS)
n_methods = len(METHODS)
bar_w     = 0.22
group_gap = 0.08
x = np.arange(n_tasks) * (n_methods * bar_w + group_gap)

for j, (method, color, hatch) in enumerate(zip(METHODS, COLORS, HATCHES)):
    aurocs = [results[t][method]["auroc"] for t in TASKS]
    bars = ax.bar(x + j * bar_w, aurocs, bar_w,
                  color=color, hatch=hatch, edgecolor="white",
                  linewidth=0.8, label=method, alpha=0.88, zorder=3)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=color)

ax.set_xticks(x + bar_w)
ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
ax.set_ylabel("AUROC ↑", fontsize=11)
ax.set_ylim(0.3, 1.12)
ax.set_title("Failure Detection AUROC by Task", fontsize=12, fontweight="bold")
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
ax.legend(fontsize=9.5, framealpha=0.9, loc="upper left")
ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# Annotate push_chair with a star / note
pc_idx = TASKS.index("push_chair")
ax.annotate("†", xy=(x[pc_idx] + bar_w, 1.01), fontsize=13, color="#555",
            ha="center")

# means
means = {m: np.mean([results[t][m]["auroc"] for t in TASKS]) for m in METHODS}
ax.text(0.98, 0.04,
        "\n".join([f"{m}: {means[m]:.3f} mean" for m in METHODS]),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="gray", style="italic")

ax.text(0.98, 0.20, "† episode-length trivially\n  separates push_chair",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        color="#888", style="italic")

# ── Figure 1b: push_chair episode-length diagnosis ───────────────────────────

ax2 = axes[1]

cal_paths  = sorted(glob.glob(str(ROOT / "data/push_chair/rollouts/calibration/*.pkl")))
test_paths = sorted(glob.glob(str(ROOT / "data/push_chair/rollouts/test/*.pkl")))

cal_lens   = [pickle.load(open(p,"rb"))["metadata"]["num_steps"] for p in cal_paths]
test_eps   = [(pickle.load(open(p,"rb"))["metadata"]["num_steps"],
               pickle.load(open(p,"rb"))["metadata"]["successful"])
              for p in test_paths]

suc_lens  = [T for T, s in test_eps if s]
fail_lens = [T for T, s in test_eps if not s]

bdata   = [cal_lens, suc_lens, fail_lens]
blabels = ["Cal.\n(success)", "Test\nsuccess", "Test\nfailure"]
bcolors = ["#4dac26", "#2166ac", "#d6604d"]

bp = ax2.boxplot(bdata, patch_artist=True, widths=0.45,
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker="o", markersize=4, alpha=0.6))

for patch, color in zip(bp["boxes"], bcolors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# overlay jittered points
for i, (data, color) in enumerate(zip(bdata, bcolors), start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(data))
    ax2.scatter(np.full(len(data), i) + jitter, data,
                color=color, s=28, zorder=5, edgecolors="white", linewidth=0.5)

ax2.set_xticklabels(blabels, fontsize=10)
ax2.set_ylabel("Episode length (steps)", fontsize=11)
ax2.set_title("Push-Chair: Failure ↔ Duration†", fontsize=12, fontweight="bold")
ax2.axhline(max(cal_lens) + 0.5, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7, label=f"Max cal. length ({max(cal_lens)} steps)")
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(axis="y", linewidth=0.5, alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

auroc_len = 1.0  # episode length alone gives perfect separation
ax2.text(0.97, 0.06, f"Length-only AUROC = {auroc_len:.2f}",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=9, color="#d6604d", fontweight="bold")

plt.tight_layout(pad=1.5)
out = ROOT / "data/results/etl_comparison.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
out_png = ROOT / "data/results/etl_comparison.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"Saved: {out_png}")


# ── Figure 2: Accuracy comparison vs FIPER paper ─────────────────────────────

fig2, ax3 = plt.subplots(figsize=(10, 4))

bar_w2 = 0.18
x2 = np.arange(n_tasks) * (5 * bar_w2 + group_gap)

our_methods = [
    ("ETL-or (ours)",   COLORS[0],  "",   [results[t]["ETL-or"]["accuracy"]   for t in TASKS]),
    ("PCA-kmeans (ours)", COLORS[1], "//", [results[t]["PCA-kmeans"]["accuracy"] for t in TASKS]),
    ("logpZO (ours)",   COLORS[2],  "..", [results[t]["logpZO"]["accuracy"]   for t in TASKS]),
]
paper_methods = [
    ("PCA-kmeans (paper)", "#e08070", "xx", [FIPER_ACC[t]["PCA-kmeans"] for t in TASKS]),
    ("logpZO (paper)",     "#80c060", "oo", [FIPER_ACC[t]["logpZO"]     for t in TASKS]),
]

all_bars = our_methods + paper_methods
for j, (label, color, hatch, vals) in enumerate(all_bars):
    bars = ax3.bar(x2 + j * bar_w2, vals, bar_w2,
                   color=color, hatch=hatch, edgecolor="white",
                   linewidth=0.7, label=label, alpha=0.85, zorder=3)

ax3.set_xticks(x2 + 2 * bar_w2)
ax3.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
ax3.set_ylabel("Accuracy ↑", fontsize=11)
ax3.set_ylim(0.3, 1.05)
ax3.set_title("Accuracy: Our Implementation vs FIPER Paper (Table 1)", fontsize=12, fontweight="bold")
ax3.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
ax3.legend(fontsize=8.5, ncol=3, framealpha=0.9, loc="upper left")
ax3.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
ax3.spines[["top", "right"]].set_visible(False)
ax3.text(0.99, 0.03, "Paper values from FIPER Table 1 (TWA-aggregated);\nour values use per-episode max-score CP.",
         transform=ax3.transAxes, ha="right", va="bottom", fontsize=7.5, color="gray", style="italic")

plt.tight_layout(pad=1.5)
out2 = ROOT / "data/results/accuracy_vs_paper.pdf"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
out2_png = ROOT / "data/results/accuracy_vs_paper.png"
fig2.savefig(out2_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
print(f"Saved: {out2_png}")
