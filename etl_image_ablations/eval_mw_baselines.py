"""
eval_mw_baselines.py
--------------------------
Run **logpZO** and **PCA-kmeans** (embedding baselines) on MetaWorld pick-place-wall,
using the *same* predicate-based eval pipeline as `eval_mw_sequential_spec.py`.
ETL (cosine/L2 to spec-latent prototype) is included as well so all three
methods are evaluated on the **same** cal/test split → apples-to-apples.

For each subtask predicate X ∈ {A=grasp, B=place} we have a per-frame ground
truth label GT_X(t).  We fit each baseline on calibration embeddings WHERE
GT_X(t) is True, score every test-frame embedding, then sweep / CP-calibrate τ
just like in `eval_mw_sequential_spec.py`.  Sequential satisfaction
(F A → F B) is computed exactly as in that script.

Methods
-------
* `etl`       — score = ‖z_t − z_X*‖₂  with  z_X* = mean of cal positives.
* `logpzo`    — score = ‖noise(z_t)‖² where the flow-matching network is trained
                on cal positives (so X-like states have low score).
* `pca_kmeans`— score = min_k ‖PCA(z_t) − c_k‖₂ with KMeans clusters fit on
                cal positives.

Notes:
The prior work introduces logpZO and PCA-kmeans as baselines for failure
prediction.  We previously plugged them into the this framework
(`etl_d3il/`); here we run the same baselines on Newt-WM latents from
MetaWorld so we can directly compare them to ETL on a sequential predicate.

Usage:
  cd /path/to/repo
  MUJOCO_GL=egl python -m etl_image_ablations.eval_mw_baselines \
      --num-demos 40 \
      --out-dir etl_results/mw_baselines

Outputs under --out-dir/:
  eval_metrics.json  — full metrics for all 3 methods (etl/logpzo/pca_kmeans)
  comparison.png      — side-by-side bar chart (F1/precision/recall/seq)
  cache/demos.pt      — cached demos so re-runs are fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
ETL_DIR = ROOT / "etl_d3il"
for p in [str(TDMPC2_DIR), str(ROOT), str(ETL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# eval_mw_sequential_spec already does the hydra cwd workaround.
from etl_image_ablations.eval_mw_sequential_spec import (  # noqa: E402
    TASK,
    OBJ_Z_IDX,
    LIFT_THR,
    load_agent_and_env,
    _gt_grasped,
    _gt_placed,
    build_spec_latent_A,
    build_spec_latent_B,
    calibrate,
    sequential_sat_latent,
    sequential_sat_gt,
)
from etl_image_ablations.run_image_etl_ablations import (  # noqa: E402
    collect_demos,
    encode_latent,
)

# logpZO model (flow-matching density estimator from Xu et al. 2025).
from evaluation.method_eval_classes.logpzo_eval import (  # noqa: E402
    LogpZOModel,
)


# ── caching collected demos ───────────────────────────────────────────────────


def collect_or_load_demos(cache_path: Path, num_demos: int, seed: int) -> List[Dict]:
    """Run the policy and cache the resulting demos to disk on first call."""
    if cache_path.exists():
        print(f"[cache] Loading demos from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"[cache] No cache at {cache_path}, collecting demos …")
    agent, env, tasks_t, cfg = load_agent_and_env(num_demos, seed)
    demos = collect_demos(cfg, agent, env, tasks_t)
    if len(demos) < num_demos:
        raise RuntimeError(f"Only {len(demos)} demos; need {num_demos}")

    for d in demos:
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent, d["obs"], task_tensor)
        d["gt_A"] = _gt_grasped(d["obs"])
        d["gt_B"] = _gt_placed(d["success"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(demos, cache_path)
    print(f"[cache] Saved {len(demos)} demos to {cache_path}")
    return demos


# ── logpZO baseline ───────────────────────────────────────────────────────────


def train_logpzo(
    embeddings: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> LogpZOModel:
    """Train flow-matching model on positive-class cal embeddings."""
    model = LogpZOModel(input_dim=embeddings.shape[-1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = embeddings.shape[0]
    train_n = max(1, int(0.9 * n))
    train_n = min(train_n, n - 1) if n > 1 else n
    perm = torch.randperm(n)
    train_e = embeddings[perm[:train_n]].to(device)
    val_e = embeddings[perm[train_n:]].to(device) if n - train_n > 0 else None

    best_val = float("inf")
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        idx = torch.randperm(train_e.shape[0])
        total = 0.0
        nb = 0
        for s in range(0, train_e.shape[0], batch_size):
            batch = train_e[idx[s : s + batch_size]]
            o0 = torch.randn_like(batch)
            target = o0 - batch
            t = torch.rand((batch.shape[0], 1), device=device)
            ot = (1 - t) * batch + t * o0
            out = model(ot, t)
            loss = nn.MSELoss()(out, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            nb += 1
        train_loss = total / max(1, nb)

        if val_e is None or val_e.shape[0] == 0:
            val_loss = train_loss
        else:
            model.eval()
            with torch.no_grad():
                o0 = torch.randn_like(val_e)
                target = o0 - val_e
                t = torch.rand((val_e.shape[0], 1), device=device)
                ot = (1 - t) * val_e + t * o0
                out = model(ot, t)
                val_loss = nn.MSELoss()(out, target).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(
                f"  [logpzo] epoch {epoch + 1}/{num_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def logpzo_score(model: LogpZOModel, embeddings: torch.Tensor, device: str) -> np.ndarray:
    e = embeddings.to(device)
    t = torch.zeros((e.shape[0], 1), device=device)
    noise = e + model(e, t)
    return (noise.norm(dim=-1) ** 2).cpu().numpy()


# ── PCA-KMeans baseline ───────────────────────────────────────────────────────


class PCAKMeansScorer:
    def __init__(self, embeddings: np.ndarray, n_components: int = 10, n_clusters: int = 64):
        n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
        n_clusters = min(n_clusters, embeddings.shape[0])
        self.pca = PCA(n_components=n_components)
        self.compressed = self.pca.fit_transform(embeddings)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.compressed)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        z = self.pca.transform(embeddings)
        d = np.linalg.norm(self.kmeans.cluster_centers_[None, :, :] - z[:, None, :], axis=-1)
        return d.min(axis=-1)


# ── shared evaluation helpers ─────────────────────────────────────────────────


def per_demo_split(scores_flat: np.ndarray, demo_lengths: List[int]) -> List[np.ndarray]:
    out, off = [], 0
    for L in demo_lengths:
        out.append(scores_flat[off : off + L])
        off += L
    return out


def evaluate_method(
    name: str,
    score_A_cal: np.ndarray,
    score_B_cal: np.ndarray,
    score_A_test_per_demo: List[np.ndarray],
    score_B_test_per_demo: List[np.ndarray],
    gt_A_cal: np.ndarray,
    gt_B_cal: np.ndarray,
    gt_A_test_per_demo: List[np.ndarray],
    gt_B_test_per_demo: List[np.ndarray],
    alpha: float,
) -> Dict[str, Any]:
    score_A_test_flat = np.concatenate(score_A_test_per_demo)
    score_B_test_flat = np.concatenate(score_B_test_per_demo)
    gt_A_test_flat = np.concatenate(gt_A_test_per_demo)
    gt_B_test_flat = np.concatenate(gt_B_test_per_demo)

    res_A = calibrate(score_A_cal, gt_A_cal, score_A_test_flat, gt_A_test_flat, alpha=alpha)
    res_B = calibrate(score_B_cal, gt_B_cal, score_B_test_flat, gt_B_test_flat, alpha=alpha)

    # AUROC of score vs predicate GT (both classes present required).  Score is
    # "distance/anomaly" → lower means predicate holds → use negated score.
    def _safe_auroc(scores: np.ndarray, gt: np.ndarray) -> float:
        gt_b = gt.astype(bool)
        if gt_b.all() or (~gt_b).all():
            return float("nan")
        return float(roc_auc_score(gt_b, -scores))

    res_A["auroc"] = _safe_auroc(score_A_test_flat, gt_A_test_flat)
    res_B["auroc"] = _safe_auroc(score_B_test_flat, gt_B_test_flat)

    seq_lats, seq_gts = [], []
    for sA, sB, gA, gB in zip(
        score_A_test_per_demo, score_B_test_per_demo, gt_A_test_per_demo, gt_B_test_per_demo
    ):
        seq_lats.append(sequential_sat_latent(sA, res_A["tau_f1"], sB, res_B["tau_f1"]))
        seq_gts.append(sequential_sat_gt(gA, gB))

    seq_agree = float(np.mean([s == g for s, g in zip(seq_lats, seq_gts)]))
    seq_tp = sum(s and g for s, g in zip(seq_lats, seq_gts))
    seq_fp = sum(s and not g for s, g in zip(seq_lats, seq_gts))
    seq_fn = sum(not s and g for s, g in zip(seq_lats, seq_gts))
    seq_tn = sum(not s and not g for s, g in zip(seq_lats, seq_gts))

    keep = (
        "tau_f1", "cal_f1",
        "f1_precision", "f1_recall", "f1_f1", "f1_agreement", "f1_lift",
        "f1_gt_positive_rate", "f1_baseline_f1",
        "f1_tp", "f1_fp", "f1_fn", "f1_tn",
        "tau_cp", "cp_alpha",
        "cp_precision", "cp_recall", "cp_f1", "cp_agreement", "cp_lift",
        "auroc",
    )
    res_A_slim = {k: res_A[k] for k in keep if k in res_A}
    res_B_slim = {k: res_B[k] for k in keep if k in res_B}

    print(
        f"\n[{name}] A(grasp): F1={res_A['f1_f1']:.3f} AUROC={res_A['auroc']:.3f} "
        f"p={res_A['f1_precision']:.3f} r={res_A['f1_recall']:.3f} agree={res_A['f1_agreement']:.3f} | "
        f"CP r={res_A['cp_recall']:.3f}\n"
        f"        B(place): F1={res_B['f1_f1']:.3f} AUROC={res_B['auroc']:.3f} "
        f"p={res_B['f1_precision']:.3f} r={res_B['f1_recall']:.3f} agree={res_B['f1_agreement']:.3f} | "
        f"CP r={res_B['cp_recall']:.3f}\n"
        f"        seq agree={seq_agree:.1%} (tp={seq_tp} fp={seq_fp} fn={seq_fn} tn={seq_tn})"
    )

    return {
        "method": name,
        "subtask_A": res_A_slim,
        "subtask_B": res_B_slim,
        "sequential": {
            "agree": seq_agree,
            "tp": seq_tp, "fp": seq_fp, "fn": seq_fn, "tn": seq_tn,
            "gt_sat_rate": float(sum(seq_gts) / max(1, len(seq_gts))),
            "lat_sat_rate": float(sum(seq_lats) / max(1, len(seq_lats))),
        },
    }


def plot_comparison(all_results: Dict[str, Dict], out_path: Path) -> None:
    methods = list(all_results.keys())
    f1A = [all_results[m]["subtask_A"]["f1_f1"] for m in methods]
    f1B = [all_results[m]["subtask_B"]["f1_f1"] for m in methods]
    pA = [all_results[m]["subtask_A"]["f1_precision"] for m in methods]
    pB = [all_results[m]["subtask_B"]["f1_precision"] for m in methods]
    rA = [all_results[m]["subtask_A"]["f1_recall"] for m in methods]
    rB = [all_results[m]["subtask_B"]["f1_recall"] for m in methods]
    seq = [all_results[m]["sequential"]["agree"] for m in methods]
    cpA = [all_results[m]["subtask_A"].get("cp_recall", 0) for m in methods]
    cpB = [all_results[m]["subtask_B"].get("cp_recall", 0) for m in methods]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))

    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, f1A, w, label="A (grasp)", color="steelblue")
    ax.bar(x + w / 2, f1B, w, label="B (place)", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set(ylim=(0, 1.05), ylabel="F1", title="F1-optimal threshold")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(x - w / 2, pA, w, label="A precision", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, pB, w, label="B precision", color="seagreen", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set(ylim=(0, 1.05), ylabel="Precision", title="Precision @ τ_F1")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)

    ax = axes[2]
    ax.bar(x - w / 2, rA, w, label="A recall", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, rB, w, label="B recall", color="seagreen", alpha=0.8)
    ax.bar(x - w / 2, cpA, w, label="A CP recall", color="darkorange", alpha=0.55, hatch="//")
    ax.bar(x + w / 2, cpB, w, label="B CP recall", color="firebrick", alpha=0.55, hatch="\\\\")
    ax.axhline(0.90, color="black", ls="--", lw=0.8, alpha=0.5, label="CP target 90%")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set(ylim=(0, 1.05), ylabel="Recall", title="Recall (F1 vs CP)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, ncol=2)

    ax = axes[3]
    bars = ax.bar(methods, seq, color=["#4daf4a" if s >= 0.8 else "#e41a1c" for s in seq])
    ax.axhline(0.8, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set(ylim=(0, 1.05), ylabel="Agreement", title="Sequential F A → F B")
    for b, s in zip(bars, seq):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.0%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"embedding baselines vs ETL — {TASK}  (Newt WM latent)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="logpZO + PCA-kmeans on MetaWorld pick-place-wall")
    ap.add_argument("--num-demos", type=int, default=40)
    ap.add_argument("--cal-frac", type=float, default=0.40)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logpzo-epochs", type=int, default=200)
    ap.add_argument("--logpzo-batch-size", type=int, default=256)
    ap.add_argument("--logpzo-lr", type=float, default=1e-4)
    ap.add_argument("--pca-components", type=int, default=10)
    ap.add_argument("--n-clusters", type=int, default=64)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.out_dir / "cache" / f"demos_n{args.num_demos}_seed{args.seed}.pt"
    demos = collect_or_load_demos(cache_path, args.num_demos, args.seed)
    print(f"[main] {len(demos)} demos loaded")

    n_cal = max(1, int(args.cal_frac * len(demos)))
    cal_demos = demos[:n_cal]
    test_demos = demos[n_cal:]
    print(f"[main] cal={len(cal_demos)} test={len(test_demos)}")

    cal_lat = torch.cat([d["lat"].float() for d in cal_demos], dim=0)
    cal_gt_A = np.concatenate([d["gt_A"] for d in cal_demos])
    cal_gt_B = np.concatenate([d["gt_B"] for d in cal_demos])
    print(
        f"[main] cal frames={cal_lat.shape[0]}  "
        f"GT_A pos rate={cal_gt_A.mean():.2%}  GT_B pos rate={cal_gt_B.mean():.2%}"
    )

    test_lat_per_demo = [d["lat"].float() for d in test_demos]
    gt_A_test_per_demo = [d["gt_A"] for d in test_demos]
    gt_B_test_per_demo = [d["gt_B"] for d in test_demos]
    test_lengths = [d.shape[0] for d in test_lat_per_demo]
    test_lat_flat = torch.cat(test_lat_per_demo, dim=0)

    device = "cuda"

    # ── ETL (L2 to centroid prototype, same as eval_mw_sequential_spec) ───
    print("\n========== ETL ==========")
    z_A = build_spec_latent_A(cal_demos, window=8)
    z_B = build_spec_latent_B(cal_demos, window=10)
    etl_score_A_cal = (cal_lat - z_A.float().unsqueeze(0)).norm(dim=-1).numpy()
    etl_score_B_cal = (cal_lat - z_B.float().unsqueeze(0)).norm(dim=-1).numpy()
    etl_score_A_test = [
        (lat - z_A.float().unsqueeze(0)).norm(dim=-1).numpy() for lat in test_lat_per_demo
    ]
    etl_score_B_test = [
        (lat - z_B.float().unsqueeze(0)).norm(dim=-1).numpy() for lat in test_lat_per_demo
    ]
    res_etl = evaluate_method(
        "etl",
        etl_score_A_cal, etl_score_B_cal,
        etl_score_A_test, etl_score_B_test,
        cal_gt_A, cal_gt_B,
        gt_A_test_per_demo, gt_B_test_per_demo,
        alpha=args.alpha,
    )

    # ── logpZO (per-predicate flow matching) ───────────────────────────────
    print("\n========== logpZO ==========")
    pos_A_idx = np.where(cal_gt_A)[0]
    pos_B_idx = np.where(cal_gt_B)[0]
    print(f"[logpzo] training on {len(pos_A_idx)} A-positive cal frames")
    t0 = time.time()
    model_A = train_logpzo(
        cal_lat[pos_A_idx], args.logpzo_epochs, args.logpzo_batch_size, args.logpzo_lr, device
    )
    print(f"[logpzo] A trained in {time.time() - t0:.1f}s")
    print(f"[logpzo] training on {len(pos_B_idx)} B-positive cal frames")
    t0 = time.time()
    model_B = train_logpzo(
        cal_lat[pos_B_idx], args.logpzo_epochs, args.logpzo_batch_size, args.logpzo_lr, device
    )
    print(f"[logpzo] B trained in {time.time() - t0:.1f}s")

    logpzo_A_cal = logpzo_score(model_A, cal_lat, device)
    logpzo_B_cal = logpzo_score(model_B, cal_lat, device)
    logpzo_A_test_flat = logpzo_score(model_A, test_lat_flat, device)
    logpzo_B_test_flat = logpzo_score(model_B, test_lat_flat, device)
    logpzo_A_test = per_demo_split(logpzo_A_test_flat, test_lengths)
    logpzo_B_test = per_demo_split(logpzo_B_test_flat, test_lengths)

    res_logpzo = evaluate_method(
        "logpzo",
        logpzo_A_cal, logpzo_B_cal,
        logpzo_A_test, logpzo_B_test,
        cal_gt_A, cal_gt_B,
        gt_A_test_per_demo, gt_B_test_per_demo,
        alpha=args.alpha,
    )

    # ── PCA-KMeans (per-predicate) ─────────────────────────────────────────
    print("\n========== PCA-kmeans ==========")
    cal_lat_np = cal_lat.numpy()
    test_lat_np_flat = test_lat_flat.numpy()
    print(
        f"[pca-kmeans] components={args.pca_components}  clusters={args.n_clusters}  "
        f"A-pos cal frames={len(pos_A_idx)}  B-pos cal frames={len(pos_B_idx)}"
    )
    pk_A = PCAKMeansScorer(cal_lat_np[pos_A_idx], args.pca_components, args.n_clusters)
    pk_B = PCAKMeansScorer(cal_lat_np[pos_B_idx], args.pca_components, args.n_clusters)

    pk_A_cal = pk_A.score(cal_lat_np)
    pk_B_cal = pk_B.score(cal_lat_np)
    pk_A_test_flat = pk_A.score(test_lat_np_flat)
    pk_B_test_flat = pk_B.score(test_lat_np_flat)
    pk_A_test = per_demo_split(pk_A_test_flat, test_lengths)
    pk_B_test = per_demo_split(pk_B_test_flat, test_lengths)

    res_pca = evaluate_method(
        "pca_kmeans",
        pk_A_cal, pk_B_cal,
        pk_A_test, pk_B_test,
        cal_gt_A, cal_gt_B,
        gt_A_test_per_demo, gt_B_test_per_demo,
        alpha=args.alpha,
    )

    # ── save ──────────────────────────────────────────────────────────────
    all_results = {"etl": res_etl, "logpzo": res_logpzo, "pca_kmeans": res_pca}
    metrics_path = args.out_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "task": TASK,
                "n_demos_total": len(demos),
                "n_cal": len(cal_demos),
                "n_test": len(test_demos),
                "lift_threshold": LIFT_THR,
                "config": {
                    "logpzo_epochs": args.logpzo_epochs,
                    "logpzo_batch_size": args.logpzo_batch_size,
                    "logpzo_lr": args.logpzo_lr,
                    "pca_components": args.pca_components,
                    "n_clusters": args.n_clusters,
                    "cal_frac": args.cal_frac,
                    "alpha": args.alpha,
                    "seed": args.seed,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n[main] Wrote {metrics_path}")

    plot_comparison(all_results, args.out_dir / "comparison.png")
    print(f"[main] Wrote {args.out_dir / 'comparison.png'}")

    # Console summary
    print(f"\n{'─' * 70}")
    print(f"{TASK}  —  embedding baselines vs ETL  (n_test={len(test_demos)} demos)")
    print(f"{'─' * 70}")
    print(f"{'method':<14} | {'A_F1':>6} {'A_p':>6} {'A_r':>6} | "
          f"{'B_F1':>6} {'B_p':>6} {'B_r':>6} | {'seq':>6}")
    for m, r in all_results.items():
        A, B, S = r["subtask_A"], r["subtask_B"], r["sequential"]
        print(
            f"{m:<14} | "
            f"{A['f1_f1']:>6.3f} {A['f1_precision']:>6.3f} {A['f1_recall']:>6.3f} | "
            f"{B['f1_f1']:>6.3f} {B['f1_precision']:>6.3f} {B['f1_recall']:>6.3f} | "
            f"{S['agree']:>6.1%}"
        )


if __name__ == "__main__":
    main()
