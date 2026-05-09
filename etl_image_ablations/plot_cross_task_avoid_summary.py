#!/usr/bin/env python3
"""
Summary figures for +avoid_source_task runs (e.g. blue rollout vs green success latents).

Reads:  <results_root>/<task>/<subdir>/cross_task_negation_results.json
Writes: same folder / cross_task_avoid_summary.png
         (and cross_task_avoid_summary_at_done.png if you prefer split — single multi-panel for simplicity)

Usage:
  python etl_image_ablations/plot_cross_task_avoid_summary.py \\
    --task rd-push-blue --subdir blue_with_green_avoid

From repo root (newt/), with conda env that has matplotlib.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "etl_image_ablations" / "results"


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("rows", [])


def main():
    p = argparse.ArgumentParser(description="Plot cross-task avoid / negation summary")
    p.add_argument("--task", type=str, required=True, help="Main task, e.g. rd-push-blue")
    p.add_argument("--subdir", type=str, required=True, help="results_subdir, e.g. blue_with_green_avoid")
    p.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Parent of <task>/<subdir>/",
    )
    args = p.parse_args()

    run_dir = args.results_root / args.task / args.subdir
    json_path = run_dir / "cross_task_negation_results.json"
    if not json_path.is_file():
        raise SystemExit(
            f"Missing {json_path}\nRun ablation with +avoid_source_task=... and +results_subdir={args.subdir}"
        )

    meta, rows = load_rows(json_path)
    if not rows:
        raise SystemExit("No rows in cross_task_negation_results.json")

    # --- aggregate by k_goals ---
    by_k: dict = defaultdict(list)
    for r in rows:
        by_k[r["k_goals"]].append(r)
    ks = sorted(by_k.keys())
    demos = sorted({r["demo"] for r in rows})

    avoid_name = meta.get("avoid_source_task", "?")
    main_name = meta.get("main_task", args.task)
    n_avoid = meta.get("n_avoid_latents", "?")

    # -------- Figure: multi-panel summary --------
    fig = plt.figure(figsize=(12, 10))

    # (1) Negation at success vs k — one line per demo + mean
    ax1 = fig.add_subplot(2, 2, 1)
    for dem in demos:
        ys = []
        xk = []
        for k in ks:
            block = [x for x in by_k[k] if x["demo"] == dem]
            if block:
                xk.append(k)
                ys.append(block[0]["latent_negation_at_done"])
        if xk:
            ax1.plot(xk, ys, marker="o", alpha=0.75, linewidth=1.5, label=f"demo {dem}")
    mean_neg = [np.mean([x["latent_negation_at_done"] for x in by_k[k]]) for k in ks]
    ax1.plot(ks, mean_neg, "k--", linewidth=2.5, marker="s", label="mean", zorder=10)
    ax1.set_xlabel("k (goal manifold size)")
    ax1.set_ylabel("Latent negation at success\n(avoid_dist − goal_dist)")
    ax1.set_title("Higher ⇒ farther from green-success vs blue-goal at done")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=7)

    # (2) Mean ± SEM: goal vs avoid distance at done vs k
    ax2 = fig.add_subplot(2, 2, 2)
    def stats(key):
        m, s = [], []
        for k in ks:
            v = [x[key] for x in by_k[k]]
            m.append(float(np.mean(v)))
            s.append(float(np.std(v) / (len(v) ** 0.5 + 1e-8)))
        return m, s

    g_m, g_e = stats("latent_goal_at_done")
    a_m, a_e = stats("latent_avoid_at_done")
    ax2.errorbar(ks, g_m, yerr=g_e, marker="o", capsize=3, label="goal L2 at done (blue manifold)")
    ax2.errorbar(ks, a_m, yerr=a_e, marker="s", capsize=3, label=f"avoid L2 at done ({avoid_name})")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Latent L2 distance")
    ax2.set_title("Distances at success timestep")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    # (3) Smoothness (TV) vs k — mean across demos
    ax3 = fig.add_subplot(2, 2, 3)
    tv_g = [np.mean([x["latent_goal_tv"] for x in by_k[k]]) for k in ks]
    tv_a = [np.mean([x["latent_avoid_tv"] for x in by_k[k]]) for k in ks]
    tv_n = [np.mean([x["latent_negation_tv"] for x in by_k[k]]) for k in ks]
    ax3.plot(ks, tv_g, marker="o", label="goal dist TV")
    ax3.plot(ks, tv_a, marker="s", label="avoid dist TV")
    ax3.plot(ks, tv_n, marker="^", label="negation TV")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Mean TV (across demos)")
    ax3.set_title("Signal smoothness vs goal-set size")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    # (4) Per-demo bars at largest k (or only k): negation at done
    ax4 = fig.add_subplot(2, 2, 4)
    k_max = max(ks)
    sub = [r for r in rows if r["k_goals"] == k_max]
    sub = sorted(sub, key=lambda x: x["demo"])
    xs = np.arange(len(sub))
    ax4.bar(xs, [r["latent_negation_at_done"] for r in sub], color="steelblue", edgecolor="black")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([f"d{r['demo']}" for r in sub])
    ax4.set_ylabel("Negation at success")
    ax4.set_title(f"Per-demo negation at done (k={k_max})")
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Cross-task avoid summary: {main_name} vs avoid latents from {avoid_name}\n"
        f"(n_avoid_latents={n_avoid}, runs in …/{args.task}/{args.subdir}/)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out = run_dir / "cross_task_avoid_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
