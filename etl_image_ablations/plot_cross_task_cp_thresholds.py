#!/usr/bin/env python3
"""
Visualize whether class-conditional split CP helps vs median (cal→test) and legacy
for runs stored under results/<task>/<subdir>/ (e.g. blue_with_green_avoid).

Reads:  dynamic_threshold_results.json (same schema as flat results/)
Writes: cross_task_cp_f1_comparison.png
        cross_task_cp_threshold_scales.png  (optional magnitude of τ)

Important: thresholds are mined on **goal distance** (embedding / latent / state to the
**main-task** goal manifold). Cross-task **avoid / negation** curves are separate;
this plot answers "is CP still sensible for success prediction on blue goals during
that run?" — not "CP on negation" unless you add that signal to the pipeline.

Usage:
  python etl_image_ablations/plot_cross_task_cp_thresholds.py \\
    --task rd-push-blue --subdir blue_with_green_avoid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "etl_image_ablations" / "results"

SPACES = ["embedding", "latent", "state"]
METHODS = [
    ("median_cal_test", "Median (cal∩high-R → test)"),
    ("ccp_class_conditional", "Split CP (class-cond.)"),
    ("legacy_median_all_timesteps", "Legacy median (all t)"),
]
METRICS = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def aggregate(rows: list, space: str, method_key: str, metric: str):
    sub = [r for r in rows if r.get("space") == space]
    if not sub:
        return float("nan"), 0.0
    out = []
    for r in sub:
        block = r.get(method_key)
        if block is None or metric not in block:
            continue
        out.append(float(block[metric]))
    if not out:
        return float("nan"), 0.0
    a = np.array(out)
    return float(np.mean(a)), _sem(a)


def main():
    p = argparse.ArgumentParser(description="Plot CP vs median threshold quality for a run subfolder")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--subdir", type=str, required=True)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = p.parse_args()

    run_dir = args.results_root / args.task / args.subdir
    path = run_dir / "dynamic_threshold_results.json"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("rows", [])
    if not rows:
        raise SystemExit("No rows in dynamic_threshold_results.json")

    cp_alpha = raw.get("cp_alpha", "?")
    cal_frac = raw.get("cal_frac", "?")
    reward_q = raw.get("threshold_reward_quantile", "?")
    ks = sorted({r.get("k_goals") for r in rows})
    k_note = f"k_goals ∈ {ks}" if len(ks) <= 5 else f"{len(ks)} distinct k_goals"

    # --- Figure 1: F1, Precision, Recall (grouped bars per space) ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    x = np.arange(len(SPACES))
    width = 0.25
    for ax, (mkey, mlabel) in zip(axes, METRICS):
        for mi, (method_key, method_label) in enumerate(METHODS):
            means = []
            sems = []
            for sp in SPACES:
                mu, se = aggregate(rows, sp, method_key, mkey)
                means.append(mu)
                sems.append(se)
            ax.bar(
                x + (mi - 1) * width,
                means,
                width,
                yerr=sems,
                capsize=2,
                label=method_label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(SPACES)
        ax.set_ylabel(mlabel)
        ax.set_title(f"Mean ± SEM across demos ({mkey})")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"Threshold mining on goal distance (main-task manifold) — {args.task} / {args.subdir}\n"
        f"α={cp_alpha}, cal_frac={cal_frac}, reward_q={reward_q}; {k_note}\n"
        "(Avoid/negation is not used to set τ; CP vs baselines use the same success labels.)",
        fontsize=9,
        y=1.12,
    )
    plt.tight_layout()
    out1 = run_dir / "cross_task_cp_f1_comparison.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out1}")

    # --- Figure 2: threshold scale (shows CP is often more conservative) ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for mi, (method_key, method_label) in enumerate(METHODS):
        means = []
        for sp in SPACES:
            mu, _ = aggregate(rows, sp, method_key, "threshold")
            means.append(mu)
        ax2.plot(SPACES, means, marker="o", linewidth=2, label=method_label)
    ax2.set_ylabel("Threshold τ (same units as distance in each space)")
    ax2.set_title("Calibrated threshold level by space (not comparable across spaces)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.suptitle(f"{args.task} / {args.subdir} — threshold magnitudes", fontsize=10, y=1.02)
    plt.tight_layout()
    out2 = run_dir / "cross_task_cp_threshold_scales.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
