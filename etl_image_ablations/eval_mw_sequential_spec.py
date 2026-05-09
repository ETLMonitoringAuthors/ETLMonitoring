"""
eval_mw_sequential_spec.py
--------------------------
Sequential predicate evaluation for mw-pick-place-wall in the Newt WM latent.

Spec:  F A  then  F B   (A ◯→ B — first grasp the object, then place it)

  A = "object grasped"  — object z-position lifted above table threshold
                          GT: obs[:, OBJ_Z_IDX] > LIFT_THR
  B = "object placed"   — simulator success signal
                          GT: success[t] >= 0.99

Two latent spec anchors are built from calibration demos:
  z_A — centroid of latent vectors during the "lift phase" (obj_z > LIFT_THR,
         before first success) in cal demos.
  z_B — centroid of latent vectors in the success window around done_idx.

For each subtask predicate π_X(z_t) = 1 iff dist(z_t, z_X) < τ_X, we calibrate:
  τ_F1  — F1-optimal on cal split
  τ_CP  — class-conditional split conformal (α=0.10): recall ≥ 0.90 guarantee

Sequential satisfaction:
  seq_latent(traj) = 1 iff ∃ t1 < t2 :  π_A(z_{t1})  ∧  π_B(z_{t2})
  seq_gt(traj)     = 1 iff ∃ t1 < t2 :  GT_A(t1)     ∧  GT_B(t2)

We evaluate agreement(seq_latent, seq_gt) across test demos, along with
per-subtask F1/precision/recall/lift.

Usage:
  cd /path/to/repo
  MUJOCO_GL=egl python -m etl_image_ablations.eval_mw_sequential_spec \\
      --num-demos 40 \\
      --out-dir etl_results/mw_sequential

Outputs under --out-dir/:
  seq_metrics.json        — all metrics
  timelines/              — dual-predicate Boolean timeline per test demo
  roc/                    — F1-vs-τ and PR curves for A and B
  seq_summary.png         — sequential satisfaction bar chart
"""

from __future__ import annotations

import argparse
import json
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

# ── path / hydra setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
for p in [str(TDMPC2_DIR), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import hydra.utils as _hu
if not hasattr(_hu, "_orig_get_original_cwd"):
    _hu._orig_get_original_cwd = _hu.get_original_cwd
    _hu.get_original_cwd = lambda: str(Path.cwd())

from common import set_seed                          # noqa: E402
from common.world_model import WorldModel            # noqa: E402
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

# ── constants ───────────────────────────────────────────────────────────────
CHECKPOINT_BASE = (
    ROOT
    / "checkpoints/models--nicklashansen--newt/snapshots"
    / "7eef11eb63c8ed53d61d739693d7140135ea0876"
)
TASK = "mw-pick-place-wall"
CHECKPOINT = str(CHECKPOINT_BASE / f"{TASK}.pt")

# MetaWorld pick-place-wall-v2 observation layout (128-D in Newt wrapper)
#   obs[0:3]  — end-effector xyz
#   obs[3]    — gripper (1.0=open, -1.0=closed)
#   obs[4:7]  — object xyz
#   obs[6]    — object z  ← used for "grasped" GT
OBJ_Z_IDX = 6
# object rests at z ≈ 0.016; use a generous threshold so a firm grasp + small
# lift is detected, but the object still touching the table is not.
LIFT_THR = 0.05


# ── cfg builder ─────────────────────────────────────────────────────────────

def _build_cfg(num_demos: int, seed: int):
    base = OmegaConf.structured(Config())
    OmegaConf.set_struct(base, False)
    merged = OmegaConf.merge(base, OmegaConf.create({
        "task": TASK,
        "num_demos": num_demos,
        "enable_wandb": False,
        "env_mode": "sync",
        "checkpoint": CHECKPOINT,
        "num_envs": 2 * num_demos,
        "model_size": "B",
        "save_video": True,
        "compile": False,
        "seed": seed,
    }))
    return parse_cfg(merged)


# ── loading (handles 10-task checkpoint → single-task model) ────────────────

def _load_checkpoint_safe(agent: TDMPC2, ckpt_path: str):
    """
    Load the Newt multi-task checkpoint into the current WorldModel.

    The published checkpoints have _action_masks of shape [10, action_dim]
    (trained on 10 tasks).  In eval mode we may create a model whose
    cfg.tasks is duplicated num_envs times (e.g. [80, 16]).
    api_model_conversion's repeat(n_target // n_src) fails for n_target > n_src
    OR produces zeros when n_target < n_src.  We fix it by tiling the first
    row of the source mask to fill n_target rows.
    """
    state_dict = torch.load(ckpt_path, map_location=torch.get_default_device(),
                            weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    target_sd = agent.model.state_dict()

    # Fix _action_masks size mismatch
    if "_action_masks" in state_dict and "_action_masks" in target_sd:
        n_target = target_sd["_action_masks"].shape[0]
        src_am   = state_dict["_action_masks"]
        if src_am.shape[0] != n_target:
            # Tile first row (task 0) to fill all target slots.
            # All envs run the same single task, so row 0 is correct for all.
            row0 = src_am[0:1]  # [1, action_dim]
            state_dict["_action_masks"] = row0.repeat(n_target, 1)

    # Remove _task_emb if not in target
    if "_task_emb.weight" in state_dict and "_task_emb.weight" not in target_sd:
        state_dict.pop("_task_emb.weight", None)

    agent.model.load_state_dict(state_dict, strict=True)


def load_agent_and_env(num_demos: int, seed: int):
    cfg = _build_cfg(num_demos, seed)
    set_seed(seed)
    env = make_env(cfg)
    tasks_t = torch.arange(len(cfg.tasks), dtype=torch.int32)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    _load_checkpoint_safe(agent, CHECKPOINT)
    return agent, env, tasks_t, cfg


# ── GT helpers ───────────────────────────────────────────────────────────────

def _gt_grasped(obs: torch.Tensor) -> np.ndarray:
    """GT_A: object z-coordinate above lift threshold (object is held aloft)."""
    return (obs[:, OBJ_Z_IDX].numpy() > LIFT_THR).astype(bool)


def _gt_placed(success: torch.Tensor) -> np.ndarray:
    """GT_B: environment success signal."""
    return (success.numpy() >= 0.99).astype(bool)


# ── spec latent construction ─────────────────────────────────────────────────

def build_spec_latent_A(cal_demos: List[Dict], window: int = 8) -> torch.Tensor:
    """
    z_A = centroid of latents during the lift phase (obj_z > LIFT_THR, before success).
    For each cal demo: take up to `window` frames around the *first* lifted frame.
    """
    vecs: List[torch.Tensor] = []
    for d in cal_demos:
        gt_a = _gt_grasped(d["obs"])
        gt_b = _gt_placed(d["success"])
        # First frame where object is lifted AND we're not yet at success
        # (captures mid-transit, carrying-over-wall phase)
        lifted_pre_success = gt_a & (~gt_b)
        idx = np.where(lifted_pre_success)[0]
        if len(idx) == 0:
            # Fall back: any lifted frame
            idx = np.where(gt_a)[0]
        if len(idx) == 0:
            continue
        # Use a window centred on the first lift event
        first = int(idx[0])
        T = d["lat"].shape[0]
        t0 = max(0, first - window // 2)
        t1 = min(T - 1, first + window)
        vecs.append(d["lat"][t0: t1 + 1])
    if not vecs:
        raise ValueError("No grasp events found in cal demos — cannot build z_A.")
    return torch.cat(vecs, dim=0).mean(dim=0)


def build_spec_latent_B(cal_demos: List[Dict], window: int = 10) -> torch.Tensor:
    """z_B = centroid of latents in the success neighbourhood."""
    vecs: List[torch.Tensor] = []
    for d in cal_demos:
        done_idx = int(d["done_idx"].item())
        T = d["lat"].shape[0]
        t0 = max(0, done_idx - window)
        t1 = min(T - 1, done_idx + window)
        vecs.append(d["lat"][t0: t1 + 1])
    if not vecs:
        raise ValueError("No cal demos — cannot build z_B.")
    return torch.cat(vecs, dim=0).mean(dim=0)


# ── F1 / CP calibration ──────────────────────────────────────────────────────

def _sweep_f1(
    distances: np.ndarray, gt: np.ndarray, n_taus: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    taus = np.linspace(distances.min(), distances.max(), n_taus)
    f1s = np.empty(n_taus); precs = np.empty(n_taus); recs = np.empty(n_taus)
    gt_b = gt.astype(bool)
    for i, tau in enumerate(taus):
        pred = distances < tau
        tp = (pred & gt_b).sum(); fp = (pred & ~gt_b).sum(); fn = (~pred & gt_b).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1s[i] = 2 * p * r / (p + r + 1e-9)
        precs[i] = p; recs[i] = r
    return taus, f1s, precs, recs


def _test_metrics(d_test: np.ndarray, gt_test: np.ndarray, tau: float) -> Dict[str, Any]:
    gt_b = gt_test.astype(bool); pred = d_test < tau
    tp = (pred & gt_b).sum(); fp = (pred & ~gt_b).sum()
    fn = (~pred & gt_b).sum(); tn = (~pred & ~gt_b).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    pos_rate = gt_b.mean()
    return dict(precision=float(p), recall=float(r), f1=float(f1),
                agreement=float((pred == gt_b).mean()),
                lift=float(f1 - 2 * pos_rate / (1 + pos_rate + 1e-9)),
                gt_positive_rate=float(pos_rate),
                baseline_f1=float(2 * pos_rate / (1 + pos_rate + 1e-9)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def calibrate(
    d_cal: np.ndarray, gt_cal: np.ndarray,
    d_test: np.ndarray, gt_test: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    taus, f1s, precs, recs = _sweep_f1(d_cal, gt_cal)
    best_i = int(np.argmax(f1s))
    tau_f1 = float(taus[best_i])
    f1_m = _test_metrics(d_test, gt_test, tau_f1)

    d_cal_t = torch.from_numpy(d_cal.astype(np.float32))
    gt_cal_t = torch.from_numpy(gt_cal.astype(bool))
    cal_mask = torch.ones(len(d_cal_t), dtype=torch.bool)
    tau_cp, n_cp, _ = threshold_class_conditional_split_cp(d_cal_t, gt_cal_t, cal_mask, alpha)
    cp_m = _test_metrics(d_test, gt_test, float(tau_cp))

    return {
        "tau_f1": tau_f1, "cal_f1": float(f1s[best_i]),
        **{f"f1_{k}": v for k, v in f1_m.items()},
        "tau_cp": float(tau_cp), "cp_alpha": alpha,
        **{f"cp_{k}": v for k, v in cp_m.items()},
        "taus": taus.tolist(), "f1_curve": f1s.tolist(),
        "prec_curve": precs.tolist(), "rec_curve": recs.tolist(),
    }


# ── sequential satisfaction ───────────────────────────────────────────────────

def sequential_sat_latent(
    dist_A: np.ndarray, tau_A: float,
    dist_B: np.ndarray, tau_B: float,
) -> bool:
    """∃ t1 < t2 : π_A(t1) ∧ π_B(t2)"""
    pi_A = dist_A < tau_A
    pi_B = dist_B < tau_B
    if not pi_A.any() or not pi_B.any():
        return False
    first_A = int(np.argmax(pi_A))
    return bool(pi_B[first_A + 1:].any())


def sequential_sat_gt(gt_A: np.ndarray, gt_B: np.ndarray) -> bool:
    """∃ t1 < t2 : GT_A(t1) ∧ GT_B(t2) — always True for successful demos."""
    if not gt_A.any() or not gt_B.any():
        return False
    first_A = int(np.argmax(gt_A))
    return bool(gt_B[first_A + 1:].any())


# ── plotting ─────────────────────────────────────────────────────────────────

def _bool_strip(ax, arr: np.ndarray, true_col="#4daf4a", false_col="#e41a1c"):
    T = len(arr)
    for t, v in enumerate(arr):
        ax.barh(0, 1, left=t, height=1, color=true_col if v else false_col, linewidth=0)
    ax.set_xlim(0, T); ax.set_ylim(-0.5, 0.5); ax.axis("off")


def plot_sequential_timeline(
    dist_A: np.ndarray,
    dist_B: np.ndarray,
    gt_A: np.ndarray,
    gt_B: np.ndarray,
    tau_A_f1: float, tau_A_cp: Optional[float],
    tau_B_f1: float, tau_B_cp: Optional[float],
    frames: Optional[List[np.ndarray]],
    frame_times: Optional[List[int]],
    out_path: Path,
    title: str,
) -> None:
    T = len(dist_A)
    lat_A = dist_A < tau_A_f1
    lat_B = dist_B < tau_B_f1

    seq_lat = sequential_sat_latent(dist_A, tau_A_f1, dist_B, tau_B_f1)
    seq_gt  = sequential_sat_gt(gt_A, gt_B)
    agree_seq = seq_lat == seq_gt
    agree_A = (lat_A == gt_A.astype(bool)).mean()
    agree_B = (lat_B == gt_B.astype(bool)).mean()

    has_imgs = frames is not None and len(frames) > 0
    nrows = 6 + (1 if has_imgs else 0)
    height_ratios = [2.2, 2.2, 0.6, 0.6, 0.6, 0.6] + ([1.5] if has_imgs else [])
    fig = plt.figure(figsize=(14, 8 + (2 if has_imgs else 0)))
    gs  = gridspec.GridSpec(nrows, 1, hspace=0.05, height_ratios=height_ratios)
    t   = np.arange(T)

    color_A = "#1f78b4"  # blue
    color_B = "#33a02c"  # green

    # ── row 0: distance A ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, dist_A, color=color_A, lw=1.5, label="dist(z_t, z_A) — grasp")
    ax0.axhline(tau_A_f1, color="darkred", ls="--", lw=1.4, label=f"τ_A(F1)={tau_A_f1:.3f}")
    if tau_A_cp is not None:
        ax0.axhline(tau_A_cp, color="darkorange", ls="-.", lw=1.4,
                    label=f"τ_A(CP)={tau_A_cp:.3f}")
    ax0.fill_between(t, 0, dist_A, where=lat_A, alpha=0.15, color=color_A)
    ax0.set_ylabel("dist(z, z_A)", fontsize=8)
    ax0.set_title(f"{title}\nagree_A={agree_A:.1%}  agree_B={agree_B:.1%}  "
                  f"seq_lat={'✓' if seq_lat else '✗'}  seq_gt={'✓' if seq_gt else '✗'}  "
                  f"seq_agree={'✓' if agree_seq else '✗'}",
                  fontsize=9)
    ax0.legend(fontsize=7, ncol=3, loc="upper right"); ax0.grid(alpha=0.3)
    ax0.tick_params(labelbottom=False); ax0.set_xlim(-1, T)

    # ── row 1: distance B ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t, dist_B, color=color_B, lw=1.5, label="dist(z_t, z_B) — place")
    ax1.axhline(tau_B_f1, color="darkred", ls="--", lw=1.4, label=f"τ_B(F1)={tau_B_f1:.3f}")
    if tau_B_cp is not None:
        ax1.axhline(tau_B_cp, color="darkorange", ls="-.", lw=1.4,
                    label=f"τ_B(CP)={tau_B_cp:.3f}")
    ax1.fill_between(t, 0, dist_B, where=lat_B, alpha=0.15, color=color_B)
    ax1.set_ylabel("dist(z, z_B)", fontsize=8)
    ax1.legend(fontsize=7, ncol=3, loc="upper right"); ax1.grid(alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # ── rows 2-5: bool strips ─────────────────────────────────────────────
    def _labeled_strip(gs_row, arr, label, true_col, false_col, sharex):
        ax = fig.add_subplot(gs[gs_row], sharex=sharex)
        _bool_strip(ax, arr, true_col, false_col)
        ax.text(-0.01, 0, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=7)
        return ax

    _labeled_strip(2, lat_A,          "lat A",  color_A,  "#ff7f00", ax0)
    _labeled_strip(3, gt_A.astype(bool), "GT A",   "#377eb8", "#ff7f00", ax0)
    _labeled_strip(4, lat_B,          "lat B",  color_B,  "#e41a1c", ax0)
    ax_last = _labeled_strip(5, gt_B.astype(bool), "GT B", "#4daf4a", "#e41a1c", ax0)

    ticks = np.linspace(0, T - 1, min(10, T), dtype=int)
    ax_last.axis("on"); ax_last.yaxis.set_visible(False)
    ax_last.spines[["top", "left", "right"]].set_visible(False)
    ax_last.set_xticks(ticks); ax_last.set_xlabel("timestep", fontsize=8)

    # ── images ────────────────────────────────────────────────────────────
    if has_imgs and frame_times is not None:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        ax_img = fig.add_subplot(gs[6])
        ax_img.axis("off")
        n = len(frames)
        ax_img.set_xlim(0, n); ax_img.set_ylim(0, 1)
        for i, (img, ts) in enumerate(zip(frames, frame_times)):
            arr = np.asarray(img).astype(np.uint8)
            oi = OffsetImage(arr, zoom=0.4)
            ab = AnnotationBbox(oi, ((i + 0.5) / n, 0.5),
                                xycoords="axes fraction", frameon=False)
            ax_img.add_artist(ab)
            ax_img.text((i + 0.5) / n, 0.02, str(ts),
                        transform=ax_img.transAxes,
                        ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_panel(result: Dict, out_path: Path, label: str) -> None:
    taus = np.array(result["taus"])
    f1s  = np.array(result["f1_curve"])
    precs = np.array(result["prec_curve"])
    recs  = np.array(result["rec_curve"])
    tau_f1 = result["tau_f1"]; tau_cp = result.get("tau_cp")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(taus, f1s,   color="steelblue", lw=1.8, label="F1")
    ax.plot(taus, precs, color="C2", lw=1.2, ls="--", alpha=0.8, label="Prec")
    ax.plot(taus, recs,  color="C3", lw=1.2, ls=":",  alpha=0.8, label="Rec")
    ax.axvline(tau_f1, color="darkred", ls="--", lw=1.6,
               label=f"τ_F1={tau_f1:.3f} (F1={result['f1_f1']:.3f})")
    if tau_cp is not None:
        ax.axvline(tau_cp, color="darkorange", ls="-.", lw=1.6,
                   label=f"τ_CP={tau_cp:.3f} (rec={result['cp_recall']:.2f})")
    ax.axhline(result["f1_baseline_f1"], color="gray", ls=":", lw=1.0,
               alpha=0.7, label=f"chance={result['f1_baseline_f1']:.3f}")
    ax.set(xlabel="τ", ylabel="score", title=f"{label}\nF1 vs τ",
           ylim=(0, 1)); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(recs, precs, color="navy", lw=1.8)
    ax2.scatter([result["f1_recall"]], [result["f1_precision"]],
                color="darkred", s=90, zorder=5,
                label=f"τ_F1 (F1={result['f1_f1']:.3f})")
    if tau_cp is not None:
        ax2.scatter([result["cp_recall"]], [result["cp_precision"]],
                    color="darkorange", s=90, marker="D", zorder=5,
                    label=f"τ_CP (rec={result['cp_recall']:.2f})")
        ax2.axvline(1 - result["cp_alpha"], color="darkorange",
                    lw=1.0, ls=":", alpha=0.6)
    ax2.set(xlabel="Recall", ylabel="Precision",
            title=f"{label}\nPrecision-Recall",
            xlim=(0, 1), ylim=(0, 1))
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seq_summary(
    seq_agree: float,
    res_A: Dict, res_B: Dict,
    out_path: Path,
) -> None:
    specs = ["Grasp (A)", "Place (B)"]
    f1s   = [res_A["f1_f1"],  res_B["f1_f1"]]
    precs = [res_A["f1_precision"], res_B["f1_precision"]]
    recs  = [res_A["f1_recall"],    res_B["f1_recall"]]
    bases = [res_A["f1_baseline_f1"], res_B["f1_baseline_f1"]]
    cp_recs  = [res_A.get("cp_recall",  0), res_B.get("cp_recall",  0)]
    cp_precs = [res_A.get("cp_precision",0), res_B.get("cp_precision",0)]

    x = np.arange(2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    w = 0.20
    ax.bar(x - 1.5*w, f1s,   w, label="F1",         color="steelblue")
    ax.bar(x - 0.5*w, precs, w, label="Precision",   color="C2", alpha=0.8)
    ax.bar(x + 0.5*w, recs,  w, label="Recall",      color="C3", alpha=0.8)
    ax.bar(x + 1.5*w, bases, w, label="Chance F1",   color="gray", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(specs)
    ax.set(ylim=(0, 1.05), ylabel="Score", title="F1-optimal threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x - 0.2, cp_recs,  0.35, label="CP Recall",    color="C3",   alpha=0.9)
    ax2.bar(x + 0.2, cp_precs, 0.35, label="CP Precision", color="C2",   alpha=0.9)
    ax2.axhline(0.90, color="darkorange", ls="--", lw=1.4, label="target 90%")
    ax2.set_xticks(x); ax2.set_xticklabels(specs)
    ax2.set(ylim=(0, 1.05), ylabel="Score", title="Conformal threshold (α=10%)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    ax3 = axes[2]
    colors = ["#4daf4a" if seq_agree >= 0.80 else "#e41a1c"]
    ax3.bar(["Sequential\nAgreement"], [seq_agree], color=colors[0], alpha=0.85)
    ax3.axhline(0.8, color="gray", ls="--", lw=1.2, alpha=0.7, label="80% ref")
    ax3.set(ylim=(0, 1.05), ylabel="Agreement",
            title="F A → F B\nsequential satisfaction agreement")
    ax3.text(0, seq_agree + 0.02, f"{seq_agree:.1%}",
             ha="center", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, axis="y")

    fig.suptitle(f"ETL sequential spec — {TASK} (Newt WM)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    num_demos: int,
    cal_frac: float,
    alpha: float,
    goal_window: int,
    grasp_window: int,
    n_timeline_demos: int,
    out_dir: Path,
    seed: int,
    no_plots: bool,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Task: {TASK}   demos={num_demos}  cal={cal_frac:.0%}  α={alpha}")
    print(f"{'='*60}")

    agent, env, tasks_t, cfg = load_agent_and_env(num_demos, seed)

    print("[eval] Collecting demos …")
    demos = collect_demos(cfg, agent, env, tasks_t)
    print(f"[eval] Collected {len(demos)} demos")
    if len(demos) < num_demos:
        raise RuntimeError(f"Only {len(demos)} demos; need {num_demos}")

    # Encode latents
    for d in demos:
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent, d["obs"], task_tensor)
        d["gt_A"] = _gt_grasped(d["obs"])
        d["gt_B"] = _gt_placed(d["success"])

    # Cal / test split
    n_cal = max(1, int(cal_frac * len(demos)))
    cal_demos  = demos[:n_cal]
    test_demos = demos[n_cal:]
    print(f"[eval] cal={len(cal_demos)}  test={len(test_demos)}")

    # Grasp stats from cal demos
    for d in cal_demos:
        lifted = (d["obs"][:, OBJ_Z_IDX].numpy() > LIFT_THR).sum()
        print(f"  cal demo: lifted_frames={lifted}/{d['obs'].shape[0]}  "
              f"done_idx={d['done_idx'].item()}")

    # Build spec latents
    z_A = build_spec_latent_A(cal_demos, window=grasp_window)
    z_B = build_spec_latent_B(cal_demos, window=goal_window)
    print(f"[eval] z_A dim={z_A.shape[0]}  z_B dim={z_B.shape[0]}")

    # Distances for all demos
    def _dist(d, z):
        return torch.norm(d["lat"].float() - z.float().unsqueeze(0), dim=-1).numpy()

    # Calibration arrays
    d_A_cal = np.concatenate([_dist(d, z_A) for d in cal_demos])
    gt_A_cal = np.concatenate([d["gt_A"] for d in cal_demos])
    d_B_cal = np.concatenate([_dist(d, z_B) for d in cal_demos])
    gt_B_cal = np.concatenate([d["gt_B"] for d in cal_demos])

    # Test arrays (per-demo kept for sequential eval)
    dist_A_test = [_dist(d, z_A) for d in test_demos]
    dist_B_test = [_dist(d, z_B) for d in test_demos]
    gt_A_test = [d["gt_A"] for d in test_demos]
    gt_B_test = [d["gt_B"] for d in test_demos]

    d_A_test = np.concatenate(dist_A_test)
    d_B_test = np.concatenate(dist_B_test)
    gt_A_flat = np.concatenate(gt_A_test)
    gt_B_flat = np.concatenate(gt_B_test)

    # Calibrate A and B
    res_A = calibrate(d_A_cal, gt_A_cal, d_A_test, gt_A_flat, alpha=alpha)
    res_B = calibrate(d_B_cal, gt_B_cal, d_B_test, gt_B_flat, alpha=alpha)

    tau_A_f1 = res_A["tau_f1"]; tau_A_cp = res_A.get("tau_cp")
    tau_B_f1 = res_B["tau_f1"]; tau_B_cp = res_B.get("tau_cp")

    print(
        f"[eval] A(grasp): τ_F1={tau_A_f1:.3f}  F1={res_A['f1_f1']:.3f}  "
        f"agree={res_A['f1_agreement']:.3f}  lift={res_A['f1_lift']:+.3f}\n"
        f"                τ_CP={tau_A_cp:.3f}  rec={res_A['cp_recall']:.3f}\n"
        f"       B(place): τ_F1={tau_B_f1:.3f}  F1={res_B['f1_f1']:.3f}  "
        f"agree={res_B['f1_agreement']:.3f}  lift={res_B['f1_lift']:+.3f}\n"
        f"                τ_CP={tau_B_cp:.3f}  rec={res_B['cp_recall']:.3f}"
    )

    # Sequential satisfaction per test demo
    seq_lats = []
    seq_gts  = []
    for dA, dB, gA, gB in zip(dist_A_test, dist_B_test, gt_A_test, gt_B_test):
        seq_lats.append(sequential_sat_latent(dA, tau_A_f1, dB, tau_B_f1))
        seq_gts.append(sequential_sat_gt(gA, gB))

    seq_agree = float(np.mean([s == g for s, g in zip(seq_lats, seq_gts)]))
    seq_tp = sum(s and g for s, g in zip(seq_lats, seq_gts))
    seq_fp = sum(s and not g for s, g in zip(seq_lats, seq_gts))
    seq_fn = sum(not s and g for s, g in zip(seq_lats, seq_gts))
    seq_tn = sum(not s and not g for s, g in zip(seq_lats, seq_gts))
    print(f"\n[eval] Sequential agreement: {seq_agree:.1%}  "
          f"(tp={seq_tp} fp={seq_fp} fn={seq_fn} tn={seq_tn})")
    print(f"       seq_gt  satisfied in {sum(seq_gts)}/{len(seq_gts)} test demos")
    print(f"       seq_lat satisfied in {sum(seq_lats)}/{len(seq_lats)} test demos")

    # Save per-demo arrays for offline re-plotting
    npz_path = out_dir / "timelines_data.npz"
    save_kwargs: dict = {
        "tau_A_f1": np.array(tau_A_f1),
        "tau_B_f1": np.array(tau_B_f1),
    }
    if tau_A_cp is not None:
        save_kwargs["tau_A_cp"] = np.array(tau_A_cp)
    if tau_B_cp is not None:
        save_kwargs["tau_B_cp"] = np.array(tau_B_cp)
    for di, (d, dA, dB, gA, gB) in enumerate(
        zip(test_demos, dist_A_test, dist_B_test, gt_A_test, gt_B_test)
    ):
        save_kwargs[f"dist_A_{di}"] = dA
        save_kwargs[f"dist_B_{di}"] = dB
        save_kwargs[f"gt_A_{di}"]   = gA
        save_kwargs[f"gt_B_{di}"]   = gB
        T = len(dA)
        sample_ts = np.linspace(0, T - 1, min(8, T), dtype=int).tolist()
        raw = [np.asarray(d["frame"][t]) for t in sample_ts]
        frames = [
            f.transpose(1, 2, 0).astype(np.uint8) if f.ndim == 3 and f.shape[0] == 3
            else f.astype(np.uint8)
            for f in raw
        ]
        save_kwargs[f"frames_{di}"]     = np.stack(frames)
        save_kwargs[f"frame_ts_{di}"]   = np.array(sample_ts)
    np.savez(npz_path, **save_kwargs)

    # Plots
    if not no_plots:
        tl_dir = out_dir / "timelines"
        for di, (d, dA, dB, gA, gB) in enumerate(
            zip(test_demos[:n_timeline_demos],
                dist_A_test[:n_timeline_demos],
                dist_B_test[:n_timeline_demos],
                gt_A_test[:n_timeline_demos],
                gt_B_test[:n_timeline_demos])
        ):
            T = len(dA)
            sample_ts = np.linspace(0, T - 1, min(8, T), dtype=int).tolist()
            raw = [np.asarray(d["frame"][t]) for t in sample_ts]
            frames = [
                f.transpose(1, 2, 0).astype(np.uint8) if f.ndim == 3 and f.shape[0] == 3
                else f.astype(np.uint8)
                for f in raw
            ]
            plot_sequential_timeline(
                dA, dB, gA, gB,
                tau_A_f1=tau_A_f1, tau_A_cp=tau_A_cp,
                tau_B_f1=tau_B_f1, tau_B_cp=tau_B_cp,
                frames=frames, frame_times=sample_ts,
                out_path=tl_dir / f"demo{di:02d}.png",
                title=f"Demo {di} | {TASK}",
            )
        plot_roc_panel(res_A, out_path=out_dir / "roc" / "grasp_A.png", label="A: Grasp")
        plot_roc_panel(res_B, out_path=out_dir / "roc" / "place_B.png", label="B: Place")
        plot_seq_summary(seq_agree, res_A, res_B,
                         out_path=out_dir / "seq_summary.png")

    return {
        "task": TASK,
        "n_demos_total": len(demos),
        "n_cal": n_cal,
        "n_test": len(test_demos),
        "lift_threshold": LIFT_THR,
        "subtask_A": res_A,
        "subtask_B": res_B,
        "sequential": {
            "agree": seq_agree,
            "tp": seq_tp, "fp": seq_fp, "fn": seq_fn, "tn": seq_tn,
            "gt_sat_rate": float(sum(seq_gts) / max(1, len(seq_gts))),
            "lat_sat_rate": float(sum(seq_lats) / max(1, len(seq_lats))),
        },
    }


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ETL sequential spec — mw-pick-place-wall")
    ap.add_argument("--num-demos",        type=int,   default=40)
    ap.add_argument("--cal-frac",         type=float, default=0.40)
    ap.add_argument("--alpha",            type=float, default=0.10)
    ap.add_argument("--goal-window",      type=int,   default=10,
                    help="Frames around done_idx for z_B")
    ap.add_argument("--grasp-window",     type=int,   default=8,
                    help="Frames around first lift event for z_A")
    ap.add_argument("--n-timeline-demos", type=int,   default=8)
    ap.add_argument("--out-dir",          type=Path,  required=True)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--no-plots",         action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate(
        num_demos=args.num_demos,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        goal_window=args.goal_window,
        grasp_window=args.grasp_window,
        n_timeline_demos=args.n_timeline_demos,
        out_dir=args.out_dir,
        seed=args.seed,
        no_plots=args.no_plots,
    )

    metrics_path = args.out_dir / "seq_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    A = results["subtask_A"]; B = results["subtask_B"]; S = results["sequential"]
    print(f"\n{'─'*60}")
    print(f"  {TASK}  (sequential spec)")
    print(f"    A (grasp): F1={A['f1_f1']:.3f}  prec={A['f1_precision']:.3f}  "
          f"rec={A['f1_recall']:.3f}  lift={A['f1_lift']:+.3f}")
    print(f"    B (place): F1={B['f1_f1']:.3f}  prec={B['f1_precision']:.3f}  "
          f"rec={B['f1_recall']:.3f}  lift={B['f1_lift']:+.3f}")
    print(f"    seq agree: {S['agree']:.1%}   "
          f"gt_sat={S['gt_sat_rate']:.1%}  lat_sat={S['lat_sat_rate']:.1%}")
    print(f"    CP: A_rec={A['cp_recall']:.3f}  B_rec={B['cp_recall']:.3f}")
    print(f"\n  Metrics → {metrics_path}")
    print(f"  Plots   → {args.out_dir}/")


if __name__ == "__main__":
    main()
