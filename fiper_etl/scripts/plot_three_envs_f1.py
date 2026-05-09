"""
plot_three_envs_f1.py
---------------------
Single combined F1 plot for ETL vs FIPER baselines (PCA-kmeans, logpZO) on
three envs with sequential structure:

  * stacking            (FIPER, K=2 phases, rollout-level F1)
  * sorting             (FIPER, K=2 phases, rollout-level F1)
  * mw-pick-place-wall  (Newt-WM, K=2 predicates A=grasp + B=place,
                         per-frame F1)

Two panels:
  * F1_opt — best F1 over a τ sweep on test data (capability of the score).
  * F1_CP  — F1 at the calibrated CP threshold (operationally honest).

Inputs:
  fiper_etl/data/results/etl_f1_results.json   (sorting / stacking)
  etl_results/mw_fiper_baselines/fiper_metrics.json   (mw-pick-place-wall)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
NEWT_ROOT = ROOT.parent

FIPER_F1 = ROOT / "data" / "results" / "etl_f1_results.json"
MW_F1 = NEWT_ROOT / "etl_results" / "mw_fiper_baselines" / "fiper_metrics.json"

with open(FIPER_F1) as f:
    fiper = json.load(f)
with open(MW_F1) as f:
    mw = json.load(f)["results"]


def mw_avg(method: str, key: str) -> float:
    return 0.5 * (mw[method]["subtask_A"][key] + mw[method]["subtask_B"][key])


ENVS = [
    ("Stacking",       "stacking",          "ETL-or",   "rollout"),
    ("Sorting",        "sorting",           "ETL-or",   "rollout"),
    ("MW-Pick-Place\n(grasp ∧ place)", "mw_pick_place_wall", "etl", "frame"),
]
METHODS = [
    ("ETL",        "ETL-or",     "etl",        "#2166ac"),
    ("PCA-kmeans", "PCA-kmeans", "pca_kmeans", "#d6604d"),
    ("logpZO",     "logpZO",     "logpzo",     "#4dac26"),
]


def get(env_key: str, method_fiper: str, method_mw: str, metric: str) -> float:
    if env_key == "mw_pick_place_wall":
        if metric == "f1_optimal":
            return mw_avg(method_mw, "f1_f1")
        if metric == "f1_cp":
            return mw_avg(method_mw, "cp_f1")
        if metric == "accuracy":
            # Per-frame agreement at the CP threshold — directly analogous to
            # the rollout-level accuracy FIPER reports for sorting/stacking.
            return mw_avg(method_mw, "cp_agreement")
    return fiper[env_key][method_fiper][metric]


fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
n_envs = len(ENVS)
n_methods = len(METHODS)
bar_w = 0.24
group_gap = 0.10
x = np.arange(n_envs) * (n_methods * bar_w + group_gap)

panels = [
    ("f1_optimal", "F1 at threshold sweep optimum", "Score-separability F1 (upper bound)"),
    ("f1_cp",      "F1 at calibrated threshold",    "Deployed F1 (CP / cal-set quantile)"),
    ("accuracy",   "Accuracy at calibrated threshold", "Deployed accuracy"),
]

for ax, (metric, ylabel, title) in zip(axes, panels):
    for j, (m_label, m_fiper, m_mw, color) in enumerate(METHODS):
        vals = [get(env_key, m_fiper, m_mw, metric) for _, env_key, _, _ in ENVS]
        bars = ax.bar(
            x + j * bar_w, vals, bar_w,
            color=color, edgecolor="white", linewidth=0.8,
            label=m_label, alpha=0.9, zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color,
            )

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([e[0] for e in ENVS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "f1_optimal":
        ax.legend(fontsize=9.5, framealpha=0.9, loc="lower left")

fig.suptitle(
    "ETL vs FIPER baselines — failure / predicate detection F1 across three sequential tasks",
    fontsize=12,
)
fig.text(
    0.5, -0.02,
    "Stacking / Sorting: rollout-level F1 (FIPER protocol; ETL-or aggregates K=3 phase prototypes).  "
    "MW-Pick-Place-Wall: per-frame F1 averaged over predicates A (grasp) and B (place); "
    "scores fit per-predicate on calibration positives.",
    ha="center", va="top", fontsize=8.5, color="#444", style="italic",
)
fig.tight_layout(pad=1.5)

out_pdf = NEWT_ROOT / "figures" / "etl_vs_fiper_three_envs_f1_acc.pdf"
out_png = NEWT_ROOT / "figures" / "etl_vs_fiper_three_envs_f1_acc.png"
out_pdf.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")

print("\n──  Combined numbers ──")
print(f"{'env':<26} {'method':<12} {'F1_opt':>7} {'F1_CP':>7} {'Acc_CP':>7}")
print("-" * 64)
for env_label, env_key, _, _ in ENVS:
    for m_label, m_fiper, m_mw, _ in METHODS:
        f_opt = get(env_key, m_fiper, m_mw, "f1_optimal")
        f_cp  = get(env_key, m_fiper, m_mw, "f1_cp")
        acc   = get(env_key, m_fiper, m_mw, "accuracy")
        print(
            f"{env_label.replace(chr(10),' '):<26} {m_label:<12} "
            f"{f_opt:>7.3f} {f_cp:>7.3f} {acc:>7.3f}"
        )
