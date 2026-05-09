"""
eval_newt_spec_predicates.py
----------------------------
Boolean predicate agreement evaluation for the Newt world-model latent space.

Mirrors eval_dubins_spec_predicates.py (AnySafe/Dubins) but for Newt tasks
(e.g. rd-push-green, rd-push-blue).

Spec latent z_spec: centroid of latent vectors in the success-neighborhood
  [done_idx - window, done_idx + window] of calibration demos.
  No synthetic rendering needed — real rollout frames define the spec.

GT predicate: success[t] >= 0.99 directly from the simulator.

Two thresholds are computed on the calibration split:
  τ_F1 — F1-optimal (upper bound; supervised on cal labels).
  τ_CP  — class-conditional split conformal (α=0.10): guarantees recall ≥ 0.90
           on held-out data under exchangeability.

Optional cross-task discrimination:
  Run a second task's policy; check whether the first task's predicate fires.
  Should have low F1 / agreement if the latent is task-specific.

Usage:
  cd /path/to/repo
  python -m etl_image_ablations.eval_newt_spec_predicates \\
      --task rd-push-green \\
      --num-demos 30 \\
      --out-dir etl_results/spec_predicates_rd \\
      --cross-task rd-push-blue

Outputs under --out-dir/<task>/:
  spec_metrics.json   — per-spec metrics (F1, CP, etc.)
  timelines/          — Boolean timeline plots per demo
  roc/                — F1-vs-τ and PR curves
  cross_task/         — discrimination plots (if --cross-task)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
if str(TDMPC2_DIR) not in sys.path:
    sys.path.insert(0, str(TDMPC2_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# parse_cfg calls hydra.utils.get_original_cwd() which requires Hydra to be
# initialized. Monkeypatch it to return cwd so we can use parse_cfg standalone.
import hydra.utils as _hu
if not hasattr(_hu, "_orig_get_original_cwd"):
    _hu._orig_get_original_cwd = _hu.get_original_cwd
    _hu.get_original_cwd = lambda: str(Path.cwd())

from common import set_seed                          # noqa: E402
from common.world_model import WorldModel            # noqa: E402
from common.vision_encoder import PretrainedEncoder  # noqa: E402
from config import Config, parse_cfg                 # noqa: E402
from envs import make_env                            # noqa: E402
from tdmpc2 import TDMPC2                            # noqa: E402

from etl_image_ablations.run_image_etl_ablations import (  # noqa: E402
    collect_demos,
    encode_latent,
    to_td,
    estimate_value,
)
from etl_image_ablations.conformal_threshold import (  # noqa: E402
    threshold_class_conditional_split_cp,
)

CHECKPOINT_PATH = str(
    ROOT / "checkpoints/models--nicklashansen--newt/snapshots"
    "/7eef11eb63c8ed53d61d739693d7140135ea0876"
)


def _build_cfg(task: str, num_demos: int, seed: int, num_envs: Optional[int] = None):
    """Build a parsed cfg without requiring Hydra to be initialized."""
    base = OmegaConf.structured(Config())
    OmegaConf.set_struct(base, False)
    if num_envs is None:
        num_envs = 2 * num_demos
    overrides = OmegaConf.create({
        "task": task,
        "num_demos": num_demos,
        "enable_wandb": False,
        "env_mode": "sync",
        "checkpoint": f"{CHECKPOINT_PATH}/{task}.pt",
        "num_envs": num_envs,
        "model_size": "B",
        "save_video": True,
        "compile": False,
        "seed": seed,
    })
    merged = OmegaConf.merge(base, overrides)
    return parse_cfg(merged)


# ──────────────────────────────────────────────────────────────────────────
# Threshold helpers
# ──────────────────────────────────────────────────────────────────────────

def _sweep_f1(
    distances: np.ndarray,
    gt: np.ndarray,
    n_taus: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    taus = np.linspace(distances.min(), distances.max(), n_taus)
    f1s = np.empty(n_taus)
    precs = np.empty(n_taus)
    recs = np.empty(n_taus)
    gt_b = gt.astype(bool)
    for i, tau in enumerate(taus):
        pred = distances < tau
        tp = (pred & gt_b).sum()
        fp = (pred & ~gt_b).sum()
        fn = (~pred & gt_b).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1s[i] = 2 * p * r / (p + r + 1e-9)
        precs[i] = p
        recs[i] = r
    return taus, f1s, precs, recs


def calibrate(
    d_cal: np.ndarray,
    gt_cal: np.ndarray,
    d_test: np.ndarray,
    gt_test: np.ndarray,
    alpha: float = 0.10,
    n_taus: int = 300,
) -> Dict[str, Any]:
    """Compute τ_F1 and τ_CP on cal; evaluate both on test."""
    # F1-optimal
    taus, f1s, precs, recs = _sweep_f1(d_cal, gt_cal, n_taus)
    best_i = int(np.argmax(f1s))
    tau_f1 = float(taus[best_i])

    def _test_metrics(tau):
        gt_b = gt_test.astype(bool)
        pred = d_test < tau
        tp = (pred & gt_b).sum()
        fp = (pred & ~gt_b).sum()
        fn = (~pred & gt_b).sum()
        tn = (~pred & ~gt_b).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        agree = (pred == gt_b).mean()
        pos_rate = gt_b.mean()
        baseline_f1 = 2 * pos_rate / (1 + pos_rate + 1e-9)
        return dict(precision=float(p), recall=float(r), f1=float(f1),
                    agreement=float(agree), tp=int(tp), fp=int(fp),
                    fn=int(fn), tn=int(tn),
                    gt_positive_rate=float(pos_rate),
                    baseline_f1=float(baseline_f1),
                    lift=float(f1 - baseline_f1))

    f1_metrics = _test_metrics(tau_f1)

    # Class-conditional split CP
    d_cal_t = torch.from_numpy(d_cal.astype(np.float32))
    gt_cal_t = torch.from_numpy(gt_cal.astype(bool))
    cal_mask = torch.ones(len(d_cal_t), dtype=torch.bool)
    tau_cp, n_cp, k_cp = threshold_class_conditional_split_cp(
        d_cal_t, gt_cal_t, cal_mask, alpha
    )
    cp_metrics = _test_metrics(float(tau_cp))

    return {
        "tau_f1": tau_f1,
        "cal_f1": float(f1s[best_i]),
        **{f"f1_{k}": v for k, v in f1_metrics.items()},
        "tau_cp": float(tau_cp),
        "cp_alpha": alpha,
        "cp_target_recall": 1.0 - alpha,
        "cp_n_cal_pos": int(n_cp),
        **{f"cp_{k}": v for k, v in cp_metrics.items()},
        "taus": taus.tolist(),
        "f1_curve": f1s.tolist(),
        "prec_curve": precs.tolist(),
        "rec_curve": recs.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Spec latent: centroid of success-neighborhood frames
# ──────────────────────────────────────────────────────────────────────────

def build_spec_latent(
    cal_demos: List[Dict],
    agent,
    window: int = 10,
) -> torch.Tensor:
    """
    z_spec = mean of latent vectors within [done_idx-window, done_idx+window]
    across all calibration demos that have a valid done_idx.
    """
    vecs: List[torch.Tensor] = []
    for d in cal_demos:
        done_idx = int(d["done_idx"].item())
        T = d["lat"].shape[0]
        t0 = max(0, done_idx - window)
        t1 = min(T - 1, done_idx + window)
        vecs.append(d["lat"][t0 : t1 + 1])
    if not vecs:
        raise ValueError("No cal demos — cannot build spec latent.")
    return torch.cat(vecs, dim=0).mean(dim=0)  # [latent_dim]


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

def _bool_strip(ax, arr: np.ndarray, true_color="#4daf4a", false_color="#e41a1c"):
    T = len(arr)
    for t, v in enumerate(arr):
        ax.barh(0, 1, left=t, height=1,
                color=true_color if v else false_color, linewidth=0)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")


def plot_boolean_timeline(
    distances: np.ndarray,
    gt: np.ndarray,
    tau_f1: float,
    tau_cp: Optional[float],
    frames: Optional[List[np.ndarray]],
    frame_times: Optional[List[int]],
    out_path: Path,
    title: str,
) -> None:
    T = len(distances)
    lat_bool = distances < tau_f1
    agree = (lat_bool == gt.astype(bool)).mean()

    has_imgs = frames is not None and len(frames) > 0
    nrows = 3 + (1 if has_imgs else 0)
    fig = plt.figure(figsize=(14, 3.5 + (2 if has_imgs else 0)))
    gs = gridspec.GridSpec(nrows, 1, hspace=0.05,
                           height_ratios=[2.5, 0.7, 0.7] + ([1.5] if has_imgs else []))
    t = np.arange(T)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, distances, color="steelblue", linewidth=1.5, label="dist(z_t, z_spec)")
    ax0.axhline(tau_f1, color="darkred", linestyle="--", linewidth=1.5,
                label=f"τ_F1={tau_f1:.3f}")
    if tau_cp is not None:
        ax0.axhline(tau_cp, color="darkorange", linestyle="-.", linewidth=1.5,
                    label=f"τ_CP={tau_cp:.3f}")
    ax0.fill_between(t, 0, distances, where=lat_bool, alpha=0.15, color="green")
    ax0.fill_between(t, 0, distances, where=~lat_bool, alpha=0.10, color="red")
    ax0.set_xlim(-1, T)
    ax0.set_ylabel("dist(z, z_spec)", fontsize=8)
    ax0.set_title(f"{title}  [agreement={agree:.1%}]", fontsize=10)
    ax0.legend(fontsize=7, ncol=3, loc="upper right")
    ax0.grid(alpha=0.3)
    ax0.tick_params(labelbottom=False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    _bool_strip(ax1, lat_bool)
    ax1.text(-0.01, 0, "latent", transform=ax1.transAxes,
             ha="right", va="center", fontsize=7)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    _bool_strip(ax2, gt.astype(bool), true_color="#377eb8", false_color="#ff7f00")
    ax2.text(-0.01, 0, "GT", transform=ax2.transAxes,
             ha="right", va="center", fontsize=7)
    ticks = np.linspace(0, T - 1, min(10, T), dtype=int)
    ax2.set_xticks(ticks)
    ax2.axis("on")
    ax2.yaxis.set_visible(False)
    ax2.spines[["top", "left", "right"]].set_visible(False)
    ax2.set_xlabel("timestep", fontsize=8)

    if has_imgs and frame_times is not None:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        ax3 = fig.add_subplot(gs[3])
        ax3.axis("off")
        n = len(frames)
        ax3.set_xlim(0, n)
        ax3.set_ylim(0, 1)
        for i, (img, ts) in enumerate(zip(frames, frame_times)):
            arr = np.asarray(img).astype(np.uint8)
            oi = OffsetImage(arr, zoom=0.45)
            ab = AnnotationBbox(oi, ((i + 0.5) / n, 0.5),
                                xycoords="axes fraction", frameon=False)
            ax3.add_artist(ab)
            ax3.text((i + 0.5) / n, 0.02, str(ts),
                     transform=ax3.transAxes,
                     ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(
    result: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    taus = np.array(result["taus"])
    f1s = np.array(result["f1_curve"])
    precs = np.array(result["prec_curve"])
    recs = np.array(result["rec_curve"])
    tau_f1 = result["tau_f1"]
    tau_cp = result.get("tau_cp")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(taus, f1s, color="steelblue", lw=1.8, label="F1")
    ax.plot(taus, precs, color="C2", lw=1.2, ls="--", alpha=0.8, label="Precision")
    ax.plot(taus, recs, color="C3", lw=1.2, ls=":", alpha=0.8, label="Recall")
    ax.axvline(tau_f1, color="darkred", ls="--", lw=1.6,
               label=f"τ_F1={tau_f1:.3f} (F1={result['f1_f1']:.3f})")
    if tau_cp is not None:
        ax.axvline(tau_cp, color="darkorange", ls="-.", lw=1.6,
                   label=f"τ_CP={tau_cp:.3f} (rec={result['cp_recall']:.2f}, α={result['cp_alpha']:.0%})")
    ax.axhline(result["f1_baseline_f1"], color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"chance F1={result['f1_baseline_f1']:.3f}")
    ax.set_xlabel("threshold τ", fontsize=9)
    ax.set_ylabel("score", fontsize=9)
    ax.set_title(f"{title}\nF1 vs τ", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.plot(recs, precs, color="navy", lw=1.8)
    ax2.scatter([result["f1_recall"]], [result["f1_precision"]],
                color="darkred", s=90, zorder=5,
                label=f"τ_F1 (F1={result['f1_f1']:.3f})")
    if tau_cp is not None:
        ax2.scatter([result["cp_recall"]], [result["cp_precision"]],
                    color="darkorange", s=90, marker="D", zorder=5,
                    label=f"τ_CP (rec={result['cp_recall']:.2f})")
        ax2.axvline(1.0 - result["cp_alpha"], color="darkorange",
                    lw=1.0, ls=":", alpha=0.6,
                    label=f"target recall={1-result['cp_alpha']:.0%}")
    ax2.set_xlabel("Recall", fontsize=9)
    ax2.set_ylabel("Precision", fontsize=9)
    ax2.set_title(f"{title}\nPrecision-Recall", fontsize=9)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(results: Dict[str, Dict], out_path: Path) -> None:
    specs = list(results.keys())
    f1s   = [results[s]["f1_f1"]       for s in specs]
    precs = [results[s]["f1_precision"] for s in specs]
    recs  = [results[s]["f1_recall"]    for s in specs]
    lifts = [results[s]["f1_lift"]      for s in specs]
    bases = [results[s]["f1_baseline_f1"] for s in specs]
    cp_recs  = [results[s].get("cp_recall", 0)    for s in specs]
    cp_precs = [results[s].get("cp_precision", 0) for s in specs]

    x = np.arange(len(specs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    w = 0.22
    ax.bar(x - 1.5*w, f1s,   w, label="F1 (τ_F1)",        color="steelblue")
    ax.bar(x - 0.5*w, precs, w, label="Prec (τ_F1)",      color="C2", alpha=0.8)
    ax.bar(x + 0.5*w, recs,  w, label="Rec (τ_F1)",       color="C3", alpha=0.8)
    ax.bar(x + 1.5*w, bases, w, label="Chance F1",         color="gray", alpha=0.5)
    ax.scatter(x, [1]*len(specs), alpha=0)  # force y to 1
    for xi, lift in zip(x, lifts):
        ax.text(xi - 1.5*w, f1s[specs.index(specs[list(x).index(xi)])] + 0.01,
                f"+{lift:.2f}", ha="center", fontsize=7, color="darkblue")
    ax.set_xticks(x); ax.set_xticklabels(specs, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("F1-optimal threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x - 0.3, cp_recs,  0.28, label="CP Recall",     color="C3",     alpha=0.9)
    ax2.bar(x + 0.0, cp_precs, 0.28, label="CP Precision",  color="C2",     alpha=0.9)
    ax2.axhline(0.90, color="darkorange", ls="--", lw=1.4, label="target recall 90%")
    ax2.set_xticks(x); ax2.set_xticklabels(specs, fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Score")
    ax2.set_title(f"Conformal threshold (α={results[specs[0]]['cp_alpha']:.0%})")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("ETL predicate evaluation — Newt world-model latent", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Agent / env setup helpers
# ──────────────────────────────────────────────────────────────────────────

def load_agent_and_env(task: str, num_demos: int, seed: int, num_envs: Optional[int] = None):
    cfg = _build_cfg(task, num_demos, seed, num_envs=num_envs)
    set_seed(seed)
    env = make_env(cfg)
    tasks_t = torch.arange(len(cfg.tasks), dtype=torch.int32)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    agent.load(cfg.checkpoint)
    return agent, env, tasks_t, cfg


# ──────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate_task(
    task: str,
    num_demos: int,
    cal_frac: float,
    alpha: float,
    goal_window: int,
    n_timeline_demos: int,
    out_dir: Path,
    seed: int,
    no_plots: bool,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Task: {task}   demos={num_demos}  cal={cal_frac:.0%}  α={alpha}")
    print(f"{'='*60}")

    agent, env, tasks_t, cfg = load_agent_and_env(task, num_demos, seed)

    print("[eval] Collecting demos …")
    demos = collect_demos(cfg, agent, env, tasks_t)
    if len(demos) < num_demos:
        raise RuntimeError(f"Only {len(demos)} demos collected (need {num_demos})")
    print(f"[eval] Collected {len(demos)} demos")

    # Encode all demos
    for d in demos:
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent, d["obs"], task_tensor)

    # Cal / test split
    n_cal = max(1, int(cal_frac * len(demos)))
    cal_demos  = demos[:n_cal]
    test_demos = demos[n_cal:]
    print(f"[eval] cal={len(cal_demos)}  test={len(test_demos)}")

    # Build spec latent from cal successes
    z_spec = build_spec_latent(cal_demos, agent, window=goal_window)
    print(f"[eval] z_spec built from cal demos  (dim={z_spec.shape[0]})")

    # Pool cal distances + GT labels
    d_cal_list, gt_cal_list = [], []
    for d in cal_demos:
        dist = torch.norm(d["lat"].float() - z_spec.float().unsqueeze(0), dim=-1).numpy()
        gt   = (d["success"].numpy() >= 0.99).astype(bool)
        d_cal_list.append(dist); gt_cal_list.append(gt)
    d_cal  = np.concatenate(d_cal_list)
    gt_cal = np.concatenate(gt_cal_list)

    # Pool test distances + GT labels
    d_test_list, gt_test_list = [], []
    for d in test_demos:
        dist = torch.norm(d["lat"].float() - z_spec.float().unsqueeze(0), dim=-1).numpy()
        gt   = (d["success"].numpy() >= 0.99).astype(bool)
        d_test_list.append(dist); gt_test_list.append(gt)
    d_test  = np.concatenate(d_test_list)
    gt_test = np.concatenate(gt_test_list)

    result = calibrate(d_cal, gt_cal, d_test, gt_test, alpha=alpha)

    tau_f1 = result["tau_f1"]
    tau_cp = result.get("tau_cp")
    print(
        f"[eval] τ_F1={tau_f1:.3f}  F1={result['f1_f1']:.3f}  "
        f"agree={result['f1_agreement']:.3f}  lift={result['f1_lift']:+.3f}"
        f"\n       τ_CP={tau_cp:.3f}  CP_rec={result['cp_recall']:.3f}  "
        f"CP_prec={result['cp_precision']:.3f}"
    )

    # Plots
    if not no_plots:
        tl_dir = out_dir / "timelines"
        for di, (d, dist, gt_arr) in enumerate(
            zip(test_demos[:n_timeline_demos],
                d_test_list[:n_timeline_demos],
                gt_test_list[:n_timeline_demos])
        ):
            T = len(dist)
            sample_ts = np.linspace(0, T - 1, min(8, T), dtype=int).tolist()
            raw = [np.asarray(d["frame"][t]) for t in sample_ts]
            # frames may be CHW (3,H,W) — convert to HWC for matplotlib
            frames = [
                f.transpose(1, 2, 0).astype(np.uint8) if f.ndim == 3 and f.shape[0] == 3
                else f.astype(np.uint8)
                for f in raw
            ]
            plot_boolean_timeline(
                dist, gt_arr, tau_f1, tau_cp,
                frames=frames, frame_times=sample_ts,
                out_path=tl_dir / f"demo{di:02d}_{task}.png",
                title=f"Demo {di} | {task}",
            )

        plot_roc(result,
                 out_path=out_dir / "roc" / f"{task}.png",
                 title=task)

    return result


def evaluate_cross_task_discrimination(
    main_task: str,
    cross_task: str,
    main_result: Dict[str, Any],
    z_spec: torch.Tensor,
    num_demos: int,
    cal_frac: float,
    alpha: float,
    goal_window: int,
    n_timeline_demos: int,
    out_dir: Path,
    seed: int,
    no_plots: bool,
) -> Dict[str, Any]:
    """
    Load the cross_task agent; collect demos; encode under cross_task's OWN latent
    space; compute distance to z_spec of the main_task.

    If z_spec is specific to main_task, these distances should be high and the
    predicate should fire rarely → low F1 / agreement, showing discrimination.
    """
    print(f"\n[cross-task] Loading {cross_task} to test discrimination …")
    agent_x, env_x, tasks_x, cfg_x = load_agent_and_env(cross_task, num_demos, seed + 1)

    demos_x = collect_demos(cfg_x, agent_x, env_x, tasks_x)
    print(f"[cross-task] {len(demos_x)} demos collected for {cross_task}")

    for d in demos_x:
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent_x, d["obs"], task_tensor)

    # GT labels: success on cross_task is "not the main spec" → all False for main spec
    d_all, gt_all = [], []
    for d in demos_x:
        dist = torch.norm(d["lat"].float() - z_spec.float().unsqueeze(0), dim=-1).numpy()
        # GT for main predicate is always False on cross-task demos
        gt = np.zeros(len(dist), dtype=bool)
        d_all.append(dist); gt_all.append(gt)

    d_arr  = np.concatenate(d_all)
    gt_arr = np.concatenate(gt_all)

    # Evaluate at the main task's calibrated τ_F1
    tau = main_result["tau_f1"]
    pred = d_arr < tau
    fp   = pred.sum()
    tn   = (~pred).sum()
    fpr  = float(fp / (fp + tn + 1e-9))
    spec = float(tn / (fp + tn + 1e-9))

    print(f"[cross-task] FPR={fpr:.3f}  specificity={spec:.3f}  "
          f"(at τ_F1={tau:.3f} from {main_task})")

    result = {
        "main_task": main_task,
        "cross_task": cross_task,
        "tau_used": float(tau),
        "n_cross_timesteps": int(len(d_arr)),
        "fpr": fpr,
        "specificity": spec,
        "fp": int(fp),
        "tn": int(tn),
    }

    if not no_plots and len(demos_x) > 0:
        tl_dir = out_dir / "cross_task"
        for di, (d, dist) in enumerate(zip(demos_x[:n_timeline_demos], d_all[:n_timeline_demos])):
            T = len(dist)
            sample_ts = np.linspace(0, T - 1, min(8, T), dtype=int).tolist()
            raw = [np.asarray(d["frame"][t]) for t in sample_ts]
            frames = [
                f.transpose(1, 2, 0).astype(np.uint8) if f.ndim == 3 and f.shape[0] == 3
                else f.astype(np.uint8) for f in raw
            ]
            gt_demo = np.zeros(T, dtype=bool)
            plot_boolean_timeline(
                dist, gt_demo, tau, main_result.get("tau_cp"),
                frames=frames, frame_times=sample_ts,
                out_path=tl_dir / f"demo{di:02d}_{cross_task}_vs_{main_task}.png",
                title=f"Cross-task: {cross_task} agent | {main_task} predicate  [FPR={fpr:.2f}]",
            )

    return result


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ETL Boolean predicate evaluation for Newt WM")
    ap.add_argument("--task", type=str, required=True,
                    help="Primary task, e.g. rd-push-green")
    ap.add_argument("--cross-task", type=str, default=None,
                    help="Optional second task for discrimination test, e.g. rd-push-blue")
    ap.add_argument("--num-demos", type=int, default=30,
                    help="Total demos to collect per task")
    ap.add_argument("--cal-frac", type=float, default=0.40,
                    help="Fraction of demos used for calibration (rest = test)")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Conformal miscoverage level α (target recall = 1−α)")
    ap.add_argument("--goal-window", type=int, default=10,
                    help="Timesteps around done_idx to include in z_spec centroid")
    ap.add_argument("--n-timeline-demos", type=int, default=6,
                    help="Number of test demos to plot Boolean timelines for")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Root output directory; results go under <out-dir>/<task>/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    task_out = args.out_dir / args.task
    task_out.mkdir(parents=True, exist_ok=True)

    result = evaluate_task(
        task=args.task,
        num_demos=args.num_demos,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        goal_window=args.goal_window,
        n_timeline_demos=args.n_timeline_demos,
        out_dir=task_out,
        seed=args.seed,
        no_plots=args.no_plots,
    )

    all_results = {args.task: result}

    # Cross-task discrimination
    if args.cross_task:
        # Rebuild z_spec from the main task's cal demos (we need to redo since agent was freed)
        # Re-load main agent briefly to get z_spec tensor
        agent_m, _, tasks_m, cfg_m = load_agent_and_env(
            args.task, args.num_demos, args.seed)
        demos_m = collect_demos(cfg_m, agent_m, _, tasks_m)
        for d in demos_m:
            task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
            d["lat"] = encode_latent(agent_m, d["obs"], task_tensor)
        n_cal = max(1, int(args.cal_frac * len(demos_m)))
        z_spec = build_spec_latent(demos_m[:n_cal], agent_m, window=args.goal_window)
        del agent_m

        cross_result = evaluate_cross_task_discrimination(
            main_task=args.task,
            cross_task=args.cross_task,
            main_result=result,
            z_spec=z_spec,
            num_demos=args.num_demos,
            cal_frac=args.cal_frac,
            alpha=args.alpha,
            goal_window=args.goal_window,
            n_timeline_demos=args.n_timeline_demos,
            out_dir=task_out,
            seed=args.seed,
            no_plots=args.no_plots,
        )
        all_results["cross_task_discrimination"] = cross_result

    # Summary bar (single-task for now)
    if not args.no_plots:
        plot_summary_bar({args.task: result},
                         out_path=task_out / "summary_bar.png")

    # Save metrics
    metrics_path = task_out / "spec_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n── SUMMARY ───────────────────────────────────────────────────────")
    print(f"  {args.task}: F1={result['f1_f1']:.3f}  agree={result['f1_agreement']:.3f}"
          f"  lift={result['f1_lift']:+.3f}")
    print(f"             CP rec={result['cp_recall']:.3f}  CP prec={result['cp_precision']:.3f}")
    if args.cross_task and "cross_task_discrimination" in all_results:
        cr = all_results["cross_task_discrimination"]
        print(f"  Cross-task ({args.cross_task}): FPR={cr['fpr']:.3f}  "
              f"specificity={cr['specificity']:.3f}")
    print(f"\n  Metrics → {metrics_path}")
    print(f"  Plots   → {task_out}/")


if __name__ == "__main__":
    main()
