#!/usr/bin/env python3
"""
Check: success threshold τ on *latent goal distance* (from CP / median cal→test) vs
using the *same numeric* τ as a cutoff on *latent avoid distance*.

Questions addressed:
  1) When the trajectory is **absolutely** close to the avoid manifold (small lat_avoid),
     does the rule (lat_avoid < τ) fire?  → recall of "reuse τ on avoid" vs a reference set.
  2) When we are in the **goal-success region** (lat_goal < τ), how small is lat_avoid?
     → overlap / risk that goal-closeness co-occurs with being near the avoid set.

Requires NPZ files from a cross-task avoid run:
  demo_XX_lat_goal_avoid_kYY.npz  (written when +avoid_source_task is set; disable with
  +save_avoid_distance_trajectories=False)

Usage:
  python etl_image_ablations/analyze_tau_transfer_goal_to_avoid.py \\
    --task rd-push-blue --subdir blue_with_green_avoid --k 10 --tau-source ccp
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "etl_image_ablations" / "results"

NPZ_RE = re.compile(r"demo_(\d+)_lat_goal_avoid_k(\d+)\.npz$")


def load_tau(rows: list, demo: int, k: int, space: str, source: str) -> float:
    for r in rows:
        if (
            r.get("demo") == demo
            and r.get("k_goals") == k
            and r.get("space") == space
        ):
            if source == "ccp":
                return float(r["ccp_class_conditional"]["threshold"])
            if source == "median":
                return float(r["median_cal_test"]["threshold"])
            raise ValueError(source)
    raise KeyError(f"No threshold row for demo={demo} k={k} space={space}")


def per_traj_close_mask(lat_avoid: np.ndarray, q: float) -> np.ndarray:
    """Timesteps in the bottom-q fraction of lat_avoid within this trajectory."""
    n = lat_avoid.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(q * n)))
    thr = np.partition(lat_avoid, k - 1)[k - 1]
    return lat_avoid <= thr


def main():
    ap = argparse.ArgumentParser(
        description="Analyze reusing latent goal τ for latent avoid closeness"
    )
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--subdir", type=str, required=True)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--k", type=int, default=None, help="Goal count k (default: infer from NPZ)")
    ap.add_argument(
        "--tau-source",
        type=str,
        choices=("ccp", "median"),
        default="ccp",
        help="Which threshold from dynamic_threshold_results.json (latent space)",
    )
    ap.add_argument(
        "--per-traj-q",
        type=float,
        default=0.05,
        help="Define 'absolutely close to avoid during traj' as bottom this fraction of lat_avoid per episode",
    )
    ap.add_argument(
        "--global-q",
        type=float,
        default=None,
        help="If set, also report metrics vs global pool quantile of lat_avoid (0–1)",
    )
    args = ap.parse_args()

    run_dir = args.results_root / args.task / args.subdir
    thr_path = run_dir / "dynamic_threshold_results.json"
    if not thr_path.is_file():
        raise SystemExit(f"Missing {thr_path}")

    npz_paths = sorted(run_dir.glob("demo_*_lat_goal_avoid_k*.npz"))
    if not npz_paths:
        raise SystemExit(
            f"No demo_*_lat_goal_avoid_k*.npz under {run_dir}\n"
            "Re-run the avoid ablation (trajectories are saved by default when +avoid_source_task is set)."
        )

    by_k: dict[int, list[tuple[int, Path]]] = {}
    for p in npz_paths:
        m = NPZ_RE.search(p.name)
        if not m:
            continue
        demo = int(m.group(1))
        k = int(m.group(2))
        by_k.setdefault(k, []).append((demo, p))

    if args.k is not None:
        if args.k not in by_k:
            raise SystemExit(f"No NPZ for k={args.k}. Found k={sorted(by_k.keys())}")
        ks = [args.k]
    else:
        if len(by_k) != 1:
            raise SystemExit(
                f"Multiple k in folder {sorted(by_k.keys())}; pass --k explicitly"
            )
        ks = list(by_k.keys())

    thr_raw = json.loads(thr_path.read_text(encoding="utf-8"))
    thr_rows = thr_raw.get("rows", [])

    for k in ks:
        pairs = sorted(by_k[k], key=lambda x: x[0])
        stats = []
        all_lg = []
        all_la = []

        for demo, p in pairs:
            z = np.load(p)
            lat_goal = z["lat_goal"].reshape(-1).astype(np.float64)
            lat_avoid = z["lat_avoid"].reshape(-1).astype(np.float64)
            tau = load_tau(thr_rows, demo, k, "latent", args.tau_source)

            close_traj = per_traj_close_mask(lat_avoid, args.per_traj_q)
            n_close = int(close_traj.sum())
            if n_close > 0:
                recall_tau = float((lat_avoid[close_traj] < tau).mean())
            else:
                recall_tau = float("nan")

            in_goal_zone = lat_goal < tau
            n_gz = int(in_goal_zone.sum())
            if n_gz > 0:
                mean_avoid_in_gz = float(lat_avoid[in_goal_zone].mean())
                frac_avoid_below_tau_in_gz = float((lat_avoid[in_goal_zone] < tau).mean())
            else:
                mean_avoid_in_gz = float("nan")
                frac_avoid_below_tau_in_gz = float("nan")

            both = np.logical_and(in_goal_zone, lat_avoid < tau)
            frac_both = float(both.mean())

            stats.append(
                {
                    "demo": demo,
                    "tau": tau,
                    "n_steps": int(lat_goal.size),
                    "n_per_traj_close": n_close,
                    "recall_lat_avoid_lt_tau_given_per_traj_close": recall_tau,
                    "frac_lat_goal_lt_tau": float(in_goal_zone.mean()),
                    "mean_lat_avoid_when_lat_goal_lt_tau": mean_avoid_in_gz,
                    "frac_lat_avoid_lt_tau_when_lat_goal_lt_tau": frac_avoid_below_tau_in_gz,
                    "frac_both_goal_lt_tau_and_avoid_lt_tau": frac_both,
                }
            )
            all_lg.append(lat_goal)
            all_la.append(lat_avoid)

        pooled_goal = np.concatenate(all_lg)
        pooled_avoid = np.concatenate(all_la)

        global_metrics = {}
        if args.global_q is not None:
            gq = float(args.global_q)
            q_thr = float(np.quantile(pooled_avoid, gq))
            global_metrics["global_quantile"] = gq
            global_metrics["lat_avoid_threshold_at_quantile"] = q_thr
            recalls = []
            for (demo, p), s in zip(pairs, stats):
                z = np.load(p)
                la = z["lat_avoid"].reshape(-1)
                tau = s["tau"]
                m = la <= q_thr
                if m.any():
                    recalls.append(float((la[m] < tau).mean()))
            global_metrics["recall_lat_avoid_lt_tau_given_global_q_close"] = (
                float(np.mean(recalls)) if recalls else float("nan")
            )

        summary = {
            "task": args.task,
            "subdir": args.subdir,
            "k_goals": k,
            "tau_source": args.tau_source,
            "space": "latent",
            "per_traj_q": args.per_traj_q,
            "interpretation": {
                "recall_columns": (
                    "Among timesteps in the per-trajectory bottom per_traj_q fraction of "
                    "lat_avoid (closest-to-avoid in that rollout), fraction with lat_avoid < τ. "
                    "High ⇒ the same τ you use for goal 'closeness' also flags those moments as "
                    "'close' on the avoid axis."
                ),
                "overlap_column": (
                    "frac_both_goal_lt_tau_and_avoid_lt_tau: fraction of all timesteps where "
                    "lat_goal < τ AND lat_avoid < τ. High ⇒ goal-success region overlaps the "
                    "same absolute cutoff on avoid distance (manifolds not separated at τ)."
                ),
            },
            "per_demo": stats,
            "mean_across_demos": {
                "tau": float(np.mean([s["tau"] for s in stats])),
                "recall_close_given_per_traj_q": float(
                    np.nanmean([s["recall_lat_avoid_lt_tau_given_per_traj_close"] for s in stats])
                ),
                "frac_both_goal_and_avoid_lt_tau": float(
                    np.mean([s["frac_both_goal_lt_tau_and_avoid_lt_tau"] for s in stats])
                ),
                "mean_frac_avoid_lt_tau_in_goal_zone": float(
                    np.nanmean(
                        [s["frac_lat_avoid_lt_tau_when_lat_goal_lt_tau"] for s in stats]
                    ),
                ),
            },
        }
        if global_metrics:
            summary["global_close"] = global_metrics

        out_json = run_dir / f"tau_transfer_goal_to_avoid_k{k}_{args.tau_source}.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out_json}")

        # --- Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(pooled_goal.size, size=min(8000, pooled_goal.size), replace=False)
        ax.scatter(
            pooled_goal[idx],
            pooled_avoid[idx],
            s=4,
            alpha=0.25,
            c="C0",
            rasterized=True,
        )
        tau_mean = float(np.mean([s["tau"] for s in stats]))
        ax.axvline(tau_mean, color="C1", linestyle="--", linewidth=2, label=f"mean τ ({args.tau_source})")
        ax.axhline(tau_mean, color="C2", linestyle="--", linewidth=2, label="same τ on avoid axis")
        ax.set_xlabel("lat_goal (min L2 to goal set)")
        ax.set_ylabel("lat_avoid (min L2 to avoid set)")
        ax.set_title(
            "Pooled timesteps: vertical/horizontal = reuse τ on both axes\n"
            "(per-demo τ varies slightly; mean τ shown)"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        demos = [s["demo"] for s in stats]
        x = np.arange(len(demos))
        w = 0.35
        r = [s["recall_lat_avoid_lt_tau_given_per_traj_close"] for s in stats]
        b = [s["frac_both_goal_lt_tau_and_avoid_lt_tau"] for s in stats]
        ax2.bar(x - w / 2, r, width=w, label=f"Recall: avoid<τ | per-traj bottom {args.per_traj_q:.0%} lat_avoid")
        ax2.bar(x + w / 2, b, width=w, label="Frac timesteps: goal<τ AND avoid<τ")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"d{d}" for d in demos])
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Rate")
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_title("Per-demo: does τ catch local min-avoid? vs dual-close overlap")

        fig.suptitle(
            f"{args.task} / {args.subdir}  |  k={k}  |  latent τ from {args.tau_source}",
            fontsize=10,
            y=1.02,
        )
        plt.tight_layout()
        out_png = run_dir / f"tau_transfer_goal_to_avoid_k{k}_{args.tau_source}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
