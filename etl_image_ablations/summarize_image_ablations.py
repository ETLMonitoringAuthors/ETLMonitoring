import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "etl_image_ablations" / "results"
TASKS = ["rd-push-green", "rd-push-blue"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_dict(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _threshold_rows(thr_raw):
    if isinstance(thr_raw, dict) and "rows" in thr_raw:
        return thr_raw["rows"]
    return thr_raw


def main():
    summary = {"baseline_k1": {}, "sensitivity_by_k": {}, "thresholds": {}, "thresholds_ccp": {}}

    for task in TASKS:
        task_dir = RESULTS_ROOT / task
        smooth = load_json(task_dir / "smoothness_results.json")
        thr_raw = load_json(task_dir / "dynamic_threshold_results.json")
        thr = _threshold_rows(thr_raw)

        by_k_space = defaultdict(list)
        thr_by_k_space = defaultdict(list)
        for r in smooth:
            by_k_space[(r["k_goals"], r["space"])].append(r)
        for r in thr:
            thr_by_k_space[(r["k_goals"], r["space"])].append(r)

        k_values = sorted({r["k_goals"] for r in smooth})
        summary["sensitivity_by_k"][task] = {}
        for k in k_values:
            summary["sensitivity_by_k"][task][str(k)] = {}
            for space in ["embedding", "latent", "state"]:
                rows = by_k_space.get((k, space), [])
                summary["sensitivity_by_k"][task][str(k)][space] = {
                    "tv": mean_dict(rows, "tv"),
                    "jerk": mean_dict(rows, "jerk"),
                    "std_diff": mean_dict(rows, "std_diff"),
                }

        ks = sorted(int(k) for k in summary["sensitivity_by_k"][task].keys())
        k0 = str(ks[0]) if ks else "1"
        summary["baseline_k1"][task] = summary["sensitivity_by_k"][task].get(
            k0, summary["sensitivity_by_k"][task].get("1", {})
        )

        summary["thresholds"][task] = {}
        summary["thresholds_ccp"][task] = {}
        for space in ["embedding", "latent", "state"]:
            rows = [r for r in thr if r["space"] == space]
            if rows and "median_cal_test" in rows[0]:
                summary["thresholds"][task][space] = {
                    "f1_mean": float(
                        np.mean([r["median_cal_test"]["f1"] for r in rows])
                    ),
                    "precision_mean": float(
                        np.mean([r["median_cal_test"]["precision"] for r in rows])
                    ),
                    "recall_mean": float(
                        np.mean([r["median_cal_test"]["recall"] for r in rows])
                    ),
                    "threshold_mean": float(
                        np.mean([r["median_cal_test"]["threshold"] for r in rows])
                    ),
                }
                summary["thresholds_ccp"][task][space] = {
                    "f1_mean": float(
                        np.mean([r["ccp_class_conditional"]["f1"] for r in rows])
                    ),
                    "precision_mean": float(
                        np.mean([r["ccp_class_conditional"]["precision"] for r in rows])
                    ),
                    "recall_mean": float(
                        np.mean([r["ccp_class_conditional"]["recall"] for r in rows])
                    ),
                    "threshold_mean": float(
                        np.mean([r["ccp_class_conditional"]["threshold"] for r in rows])
                    ),
                }
            else:
                summary["thresholds"][task][space] = {
                    "f1_mean": mean_dict(rows, "f1"),
                    "precision_mean": mean_dict(rows, "precision"),
                    "recall_mean": mean_dict(rows, "recall"),
                    "threshold_mean": mean_dict(rows, "threshold"),
                }
                summary["thresholds_ccp"][task][space] = {
                    "f1_mean": float("nan"),
                    "precision_mean": float("nan"),
                    "recall_mean": float("nan"),
                    "threshold_mean": float("nan"),
                }

    out_path = RESULTS_ROOT / "combined_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot 1: baseline smoothness (k=1) for embedding vs latent vs state.
    plt.figure(figsize=(8, 4))
    x = np.arange(len(TASKS))
    width = 0.25
    for i, space in enumerate(["embedding", "latent", "state"]):
        vals = [summary["baseline_k1"][t][space]["tv"] for t in TASKS]
        plt.bar(x + (i - 1) * width, vals, width=width, label=space)
    plt.xticks(x, TASKS, rotation=15)
    plt.ylabel("TV (lower is smoother)")
    plt.title("Baseline smoothness at k=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "baseline_smoothness_k1.png", dpi=150)
    plt.close()

    # Plot 2: median (cal/test) vs class-conditional CP F1 by space.
    plt.figure(figsize=(9, 4))
    for space in ["embedding", "latent", "state"]:
        vals_med = [summary["thresholds"][t][space]["f1_mean"] for t in TASKS]
        vals_cp = [summary["thresholds_ccp"][t][space]["f1_mean"] for t in TASKS]
        plt.plot(TASKS, vals_med, marker="o", linewidth=2, linestyle="--", label=f"{space} median (cal→test)")
        plt.plot(TASKS, vals_cp, marker="s", linewidth=2, label=f"{space} CP (class-cond.)")
    plt.ylabel("Mean F1 on test suffix")
    plt.title("Threshold mining: median vs class-conditional split CP")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "threshold_f1_by_space.png", dpi=150)
    plt.close()

    print(f"Saved combined summary to {out_path}")


if __name__ == "__main__":
    main()
