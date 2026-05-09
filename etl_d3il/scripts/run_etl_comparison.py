"""
run_etl_comparison.py
---------------------
Standalone script that runs ETL variants alongside PCA-kmeans and logpZO on
one task, prints a comparison table, and saves a CSV.

Usage:
    python scripts/run_etl_comparison.py  [overrides...]

Typical override:
    python scripts/run_etl_comparison.py task=sorting methods='["etl","etl_temporal","etl_seq","similarity","logpzo"]'

This mirrors scripts/run_etl.py but restricts to the methods of interest
and prints a human-readable per-method summary at the end.
"""

import os, sys, json, pickle
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Make etl_d3il importable from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.task_manager import TaskManager
from evaluation.evaluation_manager import EvaluationManager

DEFAULT_METHODS = ["etl", "etl_temporal", "etl_seq", "similarity", "logpzo"]
PRINT_QUANTILE = 0.95
PRINT_WINDOW   = 1


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    methods = list(cfg.get("methods", DEFAULT_METHODS))
    tasks   = list(cfg.get("tasks", ["stacking"]))

    base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config_path    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

    all_results = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"{'='*60}")
        task_data_path = os.path.join(base_data_path, task)

        # Build task manager and dataset
        task_manager = TaskManager(
            cfg=cfg,
            task=task,
            task_data_path=task_data_path,
        )
        dataset = task_manager.get_rollout_dataset()

        # Run evaluation
        eval_manager = EvaluationManager(
            config_path=config_path,
            task_data_path=task_data_path,
            dataset=dataset,
        )
        results = eval_manager.evaluate(methods=methods)
        all_results[task] = results

        # Print summary table
        print(f"\n{'Method':<20} {'AUROC':>8} {'Acc':>8} {'DetTime':>10}")
        print("-" * 50)
        for method, res in results.items():
            try:
                ts = "ct_quantile"
                q  = PRINT_QUANTILE
                ws = PRINT_WINDOW
                metrics = res["test_metrics"][ts][q][ws]
                auroc   = metrics.get("auroc", float("nan"))
                acc     = metrics.get("accuracy", float("nan"))
                det_t   = metrics.get("detection_time", float("nan"))
                print(f"  {method:<18} {auroc:>8.3f} {acc:>8.3f} {det_t:>10.1f}")
            except Exception:
                print(f"  {method:<18}  (no metrics at q={q}, ws={ws})")

    # Save results JSON
    out_path = os.path.join(base_data_path, "results", "etl_comparison.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
