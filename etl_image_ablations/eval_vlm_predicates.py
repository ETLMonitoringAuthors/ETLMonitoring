"""
eval_vlm_predicates.py
----------------------
VLM baseline for Boolean predicate evaluation — reviewer comparison vs latent distance.

Defaults: **Qwen2-VL-2B-Instruct** and **subsample=5** (~20 VLM calls per 100-step demo).
For final paper numbers use ``--model Qwen/Qwen2-VL-7B-Instruct --subsample 1``.
Classifies each frame (or subsampled frames) as task-success / not, then computes the
same F1 / precision / recall / agreement metrics as eval_newt_spec_predicates.py.

Key differences from the latent approach:
  - Zero-shot: no calibration demos, no threshold tuning
  - Requires natural language task description
  - ~100-1000× slower per frame

Requires:
  pip install qwen-vl-utils   (optional — we drive the processor directly via PIL)
  transformers >= 4.45        (Qwen2VLForConditionalGeneration)

Environment:
  MUJOCO_GL=egl (set automatically if unset; RoboDesk needs headless GL)
  If ~/.cache is on a full disk, set HF_HOME to a path on a larger volume, e.g.
  HF_HOME=/path/with/space/hf_cache

  On some GPUs (e.g. Blackwell), HuggingFace's Qwen2-VL uses a CUDA .prod().tolist()
  path that can raise "CUDA driver error: invalid argument"; this script patches
  split_sizes to be computed on CPU after load (see _patch_qwen2vl_split_sizes_cpu).

Usage:
  cd /path/to/repo
  python -m etl_image_ablations.eval_vlm_predicates \\
      --task rd-push-green \\
      --num-demos 30 \\
      --out-dir etl_results/vlm_predicates \\
      --compare-json etl_results/spec_predicates_rd/rd-push-green/spec_metrics.json

  # Paper-quality (all frames, 7B):
  python -m etl_image_ablations.eval_vlm_predicates \\
      --task rd-push-green --num-demos 30 --subsample 1 \\
      --model Qwen/Qwen2-VL-7B-Instruct \\
      --out-dir etl_results/vlm_predicates \\
      --compare-json etl_results/spec_predicates_rd/rd-push-green/spec_metrics.json

Outputs under --out-dir/<task>/:
  vlm_metrics.json      — F1, precision, recall, agreement, speed
  timelines/            — Boolean timeline plots per test demo
  comparison_bar.png    — side-by-side vs latent (if --compare-json given)
"""

from __future__ import annotations

# Headless RoboDesk / dm_control rendering (must run before `envs` imports mujoco).
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from omegaconf import OmegaConf

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
if str(TDMPC2_DIR) not in sys.path:
    sys.path.insert(0, str(TDMPC2_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import hydra.utils as _hu
if not hasattr(_hu, "_orig_get_original_cwd"):
    _hu._orig_get_original_cwd = _hu.get_original_cwd
    _hu.get_original_cwd = lambda: str(Path.cwd())

from common import set_seed                          # noqa: E402
from common.world_model import WorldModel            # noqa: E402
from config import Config, parse_cfg                 # noqa: E402
from envs import make_env                            # noqa: E402
from tdmpc2 import TDMPC2                            # noqa: E402

from etl_image_ablations.run_image_etl_ablations import collect_demos  # noqa: E402

CHECKPOINT_PATH = str(
    ROOT / "checkpoints/models--nicklashansen--newt/snapshots"
    "/7eef11eb63c8ed53d61d739693d7140135ea0876"
)

# ──────────────────────────────────────────────────────────────────────────
# Task → natural language predicate descriptions
# ──────────────────────────────────────────────────────────────────────────

# The prompt asks: "Has the robot successfully completed: {description}?"
# Keep descriptions short and visually verifiable.
TASK_PROMPTS: Dict[str, str] = {
    # RoboDesk
    "rd-push-green":       "push the green block to the target location on the desk",
    "rd-push-blue":        "push the blue block to the target location on the desk",
    "rd-push-red":         "push the red block to the target location on the desk",
    "rd-push-yellow":      "push the yellow block to the target location on the desk",
    "rd-reach":            "reach the target object with the robot end-effector",
    "rd-slide":            "slide the block into the target bin",
    "rd-flat-block":       "place the flat block on the target area",
    "rd-upright-block":    "stand the block upright in the target zone",
    "rd-drawer":           "open the desk drawer fully",
    "rd-button":           "press the button so it lights up",
    "rd-lift":             "lift the block above the desk surface",
    # MetaWorld
    "mw-reach":            "reach the target position with the robot end-effector",
    "mw-push":             "push the puck to the target location",
    "mw-pick-place":       "pick up the object and place it at the target position",
    "mw-door-open":        "open the door fully by pulling the handle",
    "mw-drawer-open":      "open the drawer fully",
    "mw-drawer-close":     "close the drawer completely",
    "mw-button-press":     "press the button down until it clicks",
    "mw-button-press-topdown": "press the button down from above",
    "mw-peg-insert-side":  "insert the peg into the hole from the side",
    "mw-window-open":      "open the window by sliding it to the side",
    "mw-window-close":     "close the window by sliding it shut",
    "mw-hammer":           "use the hammer to hit the nail into the wall",
    "mw-sweep-into":       "sweep the cube into the target region",
    "mw-basketball":       "throw the basketball through the hoop",
    "mw-soccer":           "kick the ball into the goal",
    "mw-hand-insert":      "insert the gripper into the target hole",
    "mw-dial-turn":        "turn the dial to the target angle",
    "mw-coffee-button":    "press the coffee machine button",
    "mw-faucet-open":      "open the faucet by rotating the handle",
    "mw-faucet-close":     "close the faucet by rotating the handle",
    "mw-plate-slide":      "slide the plate into the cabinet",
    "mw-shelf-place":      "place the object on the shelf",
    # ManiSkill
    "ms-cartpole":         "balance the pole on the moving cart",
    "ms-hopper":           "make the hopper jump forward",
    "ms-ant":              "make the ant robot walk forward",
    "ms-pick-cube":        "pick up the cube with the robot arm",
    "ms-stack-cube":       "stack the top cube onto the bottom cube",
    "ms-peg-insertion":    "insert the peg into the slot",
    # DMControl / continuous control
    "walker-walk":         "make the walker robot walk forward upright",
    "walker-run":          "make the walker robot run forward",
    "walker-stand":        "make the walker robot stand upright",
    "cheetah-run":         "make the cheetah robot run as fast as possible",
    "hopper-hop":          "make the hopper robot hop forward",
    "pendulum-swingup":    "swing the pendulum up to the upright position",
    "cartpole-balance":    "balance the pole on the cart in the center",
    "cartpole-swingup":    "swing the pole up and balance it",
    "reacher-easy":        "move the reacher arm to touch the target",
    "reacher-hard":        "move the reacher arm to touch the small target",
}

SYSTEM_PROMPT = (
    "You are evaluating whether a robot has successfully completed a manipulation task. "
    "Look carefully at the image and answer only with 'yes' or 'no'."
)

QUESTION_TEMPLATE = (
    "Has the robot successfully completed the following task: {description}? "
    "Answer yes or no."
)


def get_task_prompt(task: str) -> str:
    """Return the natural language description for a task, with a fallback."""
    if task in TASK_PROMPTS:
        return TASK_PROMPTS[task]
    # Generic fallback: strip domain prefix and humanise
    parts = task.split("-", 1)
    desc = parts[-1].replace("-", " ") if len(parts) > 1 else task.replace("-", " ")
    return f"complete the robot task: {desc}"


# ──────────────────────────────────────────────────────────────────────────
# VLM loading and inference
# ──────────────────────────────────────────────────────────────────────────

def load_vlm(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    attn_implementation: str = "eager",
):
    """Load Qwen2-VL model and processor. Returns (model, processor)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"[vlm] Loading {model_name}  attn={attn_implementation} …")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    # Single-GPU map keeps image_grid_thw and weights aligned (avoids some multi-GPU bugs).
    # Default attn_implementation=eager avoids SDPA/flash CUDA "invalid argument" on some GPUs (e.g. Blackwell).
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attn_implementation,
    )
    model.eval()
    _patch_qwen2vl_split_sizes_cpu(model)
    print(f"[vlm] Loaded in {time.time()-t0:.1f}s  "
          f"({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
    return model, processor


def _patch_qwen2vl_split_sizes_cpu(model) -> None:
    """
    Transformers computes split_sizes via CUDA tensor .prod().tolist(); on some stacks
    (e.g. Blackwell + certain torch builds) that sync throws CUDA driver invalid argument.
    Doing the reduction on CPU matches HF logic and avoids the bad sync.
    """
    inner = model.model
    merge_sq = inner.visual.spatial_merge_size**2

    def get_image_features(pixel_values, image_grid_thw=None):
        pixel_values = pixel_values.type(inner.visual.dtype)
        image_embeds = inner.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.detach().cpu().prod(-1) // merge_sq).tolist()
        return torch.split(image_embeds, split_sizes)

    def get_video_features(pixel_values_videos, video_grid_thw=None):
        pixel_values_videos = pixel_values_videos.type(inner.visual.dtype)
        video_embeds = inner.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.detach().cpu().prod(-1) // merge_sq).tolist()
        return torch.split(video_embeds, split_sizes)

    inner.get_image_features = get_image_features
    inner.get_video_features = get_video_features
    print("[vlm] Patched Qwen2-VL split_sizes to CPU (driver workaround)")


def frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a CHW uint8 tensor to a PIL RGB image."""
    arr = frame.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def vlm_classify_frames(
    frames: torch.Tensor,       # [T, C, H, W] uint8
    model,
    processor,
    task_description: str,
    batch_size: int = 1,
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run VLM binary classification on each frame.

    Returns:
        predictions : bool array of shape [T] (True = task succeeded)
        evaluated   : bool array of shape [T] (True = this frame was evaluated;
                       False = interpolated from neighbour for subsampled frames)
        fps         : frames evaluated per second
    """
    T = frames.shape[0]
    question = QUESTION_TEMPLATE.format(description=task_description)
    predictions = np.zeros(T, dtype=bool)
    evaluated   = np.zeros(T, dtype=bool)

    # Indices actually sent to the VLM
    eval_indices = list(range(0, T, subsample))

    t_start = time.time()
    for batch_start in range(0, len(eval_indices), batch_size):
        batch_idx = eval_indices[batch_start : batch_start + batch_size]
        pil_images = [frame_to_pil(frames[i]) for i in batch_idx]

        # Build one message per image (same prompt for all)
        texts, all_images = [], []
        for pil in pil_images:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
            ]
            texts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            all_images.append(pil)

        inputs = processor(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        # Decode only the newly generated tokens
        gen_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        answers = processor.batch_decode(gen_tokens, skip_special_tokens=True)

        for local_i, (global_i, answer) in enumerate(zip(batch_idx, answers)):
            answer = answer.strip().lower()
            pred = answer.startswith("yes") or (
                "yes" in answer and "no" not in answer[:answer.find("yes")+3]
            )
            predictions[global_i] = pred
            evaluated[global_i] = True

    elapsed = time.time() - t_start
    fps = len(eval_indices) / max(elapsed, 1e-6)

    # Fill in non-evaluated frames by forward-filling from previous evaluated frame
    last_pred = False
    for t in range(T):
        if evaluated[t]:
            last_pred = predictions[t]
        else:
            predictions[t] = last_pred

    return predictions, evaluated, fps


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """Compute precision, recall, F1, agreement between boolean arrays."""
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    tp = (pred_b &  gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b &  gt_b).sum()
    tn = (~pred_b & ~gt_b).sum()
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    agree = (pred_b == gt_b).mean()
    pos_rate = gt_b.mean()
    baseline_f1 = 2 * pos_rate / (1 + pos_rate + 1e-9)
    return dict(
        precision=float(p), recall=float(r), f1=float(f1),
        agreement=float(agree),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        gt_positive_rate=float(pos_rate),
        baseline_f1=float(baseline_f1),
        lift=float(f1 - baseline_f1),
    )


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────

def _bool_strip(ax, arr: np.ndarray, true_color="#4daf4a", false_color="#e41a1c"):
    T = len(arr)
    for t, v in enumerate(arr):
        ax.barh(0, 1, left=t, height=1,
                color=true_color if v else false_color, linewidth=0)
    ax.set_xlim(0, T); ax.set_ylim(-0.5, 0.5); ax.axis("off")


def plot_vlm_timeline(
    predictions: np.ndarray,
    gt: np.ndarray,
    evaluated: np.ndarray,
    frames: Optional[List[np.ndarray]],
    frame_times: Optional[List[int]],
    out_path: Path,
    title: str,
) -> None:
    T = len(predictions)
    agree = (predictions == gt.astype(bool)).mean()

    has_imgs = frames is not None and len(frames) > 0
    nrows = 3 + (1 if has_imgs else 0)
    fig = plt.figure(figsize=(14, 3.5 + (2 if has_imgs else 0)))
    gs = gridspec.GridSpec(nrows, 1, hspace=0.05,
                           height_ratios=[2.5, 0.7, 0.7] + ([1.5] if has_imgs else []))
    t = np.arange(T)

    ax0 = fig.add_subplot(gs[0])
    pred_f = predictions.astype(float)
    ax0.step(t, pred_f, color="steelblue", linewidth=1.5, where="post",
             label="VLM prediction (1=yes)")
    # Mark frames that were actually evaluated vs forward-filled
    eval_ts = t[evaluated]
    ax0.scatter(eval_ts, pred_f[evaluated], color="steelblue", s=15, zorder=4, alpha=0.6)
    ax0.fill_between(t, 0, pred_f, where=predictions.astype(bool),
                     alpha=0.15, color="green", step="post")
    ax0.fill_between(t, 0, 1 - pred_f, where=~predictions.astype(bool),
                     alpha=0.10, color="red", step="post")
    ax0.set_xlim(-1, T); ax0.set_ylim(-0.05, 1.15)
    ax0.set_ylabel("VLM answer", fontsize=8)
    ax0.set_title(f"{title}  [agreement={agree:.1%}]", fontsize=10)
    ax0.legend(fontsize=7, loc="upper left")
    ax0.grid(alpha=0.3); ax0.tick_params(labelbottom=False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    _bool_strip(ax1, predictions)
    ax1.text(-0.01, 0, "VLM", transform=ax1.transAxes,
             ha="right", va="center", fontsize=7)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    _bool_strip(ax2, gt.astype(bool), true_color="#377eb8", false_color="#ff7f00")
    ax2.text(-0.01, 0, "GT", transform=ax2.transAxes,
             ha="right", va="center", fontsize=7)
    ticks = np.linspace(0, T - 1, min(10, T), dtype=int)
    ax2.set_xticks(ticks); ax2.axis("on"); ax2.yaxis.set_visible(False)
    ax2.spines[["top", "left", "right"]].set_visible(False)
    ax2.set_xlabel("timestep", fontsize=8)

    if has_imgs and frame_times is not None:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        ax3 = fig.add_subplot(gs[3])
        ax3.axis("off"); ax3.set_xlim(0, len(frames)); ax3.set_ylim(0, 1)
        for i, (img, ts) in enumerate(zip(frames, frame_times)):
            oi = OffsetImage(np.asarray(img).astype(np.uint8), zoom=0.45)
            ab = AnnotationBbox(oi, ((i + 0.5) / len(frames), 0.5),
                                xycoords="axes fraction", frameon=False)
            ax3.add_artist(ab)
            ax3.text((i + 0.5) / len(frames), 0.02, str(ts),
                     transform=ax3.transAxes, ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    vlm_metrics: Dict[str, Any],
    latent_metrics: Optional[Dict[str, Any]],
    task: str,
    out_path: Path,
) -> None:
    """Side-by-side bar chart comparing VLM vs latent distance predicate."""
    methods = ["VLM\n(zero-shot)"]
    f1s    = [vlm_metrics["f1"]]
    precs  = [vlm_metrics["precision"]]
    recs   = [vlm_metrics["recall"]]
    agrees = [vlm_metrics["agreement"]]
    bases  = [vlm_metrics["baseline_f1"]]

    if latent_metrics is not None:
        methods.append("Latent dist\n(τ_F1, cal'd)")
        f1s.append(latent_metrics.get("f1_f1", float("nan")))
        precs.append(latent_metrics.get("f1_precision", float("nan")))
        recs.append(latent_metrics.get("f1_recall", float("nan")))
        agrees.append(latent_metrics.get("f1_agreement", float("nan")))
        bases.append(latent_metrics.get("f1_baseline_f1", float("nan")))
        # Also add CP row for latent
        methods.append("Latent dist\n(τ_CP, conf.)")
        f1s.append(latent_metrics.get("cp_f1", float("nan")))
        precs.append(latent_metrics.get("cp_precision", float("nan")))
        recs.append(latent_metrics.get("cp_recall", float("nan")))
        agrees.append(latent_metrics.get("cp_agreement", float("nan")))
        bases.append(latent_metrics.get("f1_baseline_f1", float("nan")))

    x = np.arange(len(methods))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(10 + 2 * len(methods), 4))

    ax = axes[0]
    ax.bar(x - 1.5*w, f1s,    w, label="F1",        color="steelblue",  alpha=0.9)
    ax.bar(x - 0.5*w, precs,  w, label="Precision",  color="#2ca02c",    alpha=0.8)
    ax.bar(x + 0.5*w, recs,   w, label="Recall",     color="#d62728",    alpha=0.8)
    ax.bar(x + 1.5*w, bases,  w, label="Chance F1",  color="lightgray",  alpha=0.9)
    for xi, f1 in zip(x, f1s):
        ax.text(xi - 1.5*w, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=8, color="navy")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"{task} — predicate F1 comparison", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x, agrees, 0.5, color=["#e377c2"] + ["steelblue"] * (len(methods) - 1), alpha=0.85)
    for xi, ag in zip(x, agrees):
        ax2.text(xi, ag + 0.01, f"{ag:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("Agreement with GT", fontsize=10)
    ax2.set_title(f"{task} — agreement comparison", fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("VLM vs latent-distance Boolean predicate evaluation", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[vlm] Comparison plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Agent / env helpers
# ──────────────────────────────────────────────────────────────────────────

def _build_cfg(task: str, num_demos: int, seed: int):
    base = OmegaConf.structured(Config())
    OmegaConf.set_struct(base, False)
    overrides = OmegaConf.create({
        "task": task, "num_demos": num_demos,
        "enable_wandb": False, "env_mode": "sync",
        "checkpoint": f"{CHECKPOINT_PATH}/{task}.pt",
        "num_envs": 2 * num_demos, "model_size": "B",
        "save_video": True, "compile": False, "seed": seed,
    })
    return parse_cfg(OmegaConf.merge(base, overrides))


def load_agent_and_env(task: str, num_demos: int, seed: int):
    cfg = _build_cfg(task, num_demos, seed)
    set_seed(seed)
    env = make_env(cfg)
    tasks_t = torch.arange(len(cfg.tasks), dtype=torch.int32)
    model = WorldModel(cfg).to("cuda:0")
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
    n_timeline_demos: int,
    out_dir: Path,
    seed: int,
    model_name: str,
    batch_size: int,
    subsample: int,
    no_plots: bool,
    compare_json: Optional[Path],
    attn_implementation: str,
    prompt_override: Optional[str] = None,
    gt_obs_index: Optional[int] = None,
    gt_obs_threshold: float = 0.05,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Task: {task}   demos={num_demos}   model={model_name}")
    print(f"  subsample=1/{subsample}   batch={batch_size}   attn={attn_implementation}")
    print(f"{'='*60}")

    task_desc = prompt_override if prompt_override is not None else get_task_prompt(task)
    gt_source = f"obs[{gt_obs_index}] > {gt_obs_threshold}" if gt_obs_index is not None else "success >= 0.99"
    print(f"[vlm] Predicate: \"{task_desc}\"")
    print(f"[vlm] GT source: {gt_source}")

    # 1. Collect demos via Newt agent
    agent, env, tasks_t, cfg = load_agent_and_env(task, num_demos, seed)
    print("[vlm] Collecting demos …")
    demos = collect_demos(cfg, agent, env, tasks_t)
    del agent  # free GPU memory before loading VLM
    torch.cuda.empty_cache()
    if len(demos) < num_demos:
        raise RuntimeError(f"Only {len(demos)}/{num_demos} demos collected")
    print(f"[vlm] Collected {len(demos)} demos")

    # Cal / test split (mirrors eval_newt_spec_predicates)
    n_cal = max(1, int(cal_frac * len(demos)))
    test_demos = demos[n_cal:]
    print(f"[vlm] Using {len(test_demos)} test demos (skipping {n_cal} cal demos)")

    # 2. Load VLM
    vlm_model, processor = load_vlm(model_name, attn_implementation=attn_implementation)

    # 3. Evaluate each test demo frame-by-frame
    all_preds, all_gt = [], []
    total_fps_sum = 0.0
    tl_dir = out_dir / "timelines"

    for di, d in enumerate(test_demos):
        frames  = d["frame"]    # [T, C, H, W] uint8

        if gt_obs_index is not None:
            # e.g. grasp predicate: obs[:, 6] > 0.05
            obs_col = d["obs"][:, gt_obs_index].numpy()
            gt = (obs_col > gt_obs_threshold).astype(bool)
        else:
            gt = (d["success"].numpy() >= 0.99).astype(bool)
        print(f"[vlm] Demo {di+1}/{len(test_demos)}  T={len(frames)}  "
              f"GT positives={gt.sum()}/{len(gt)}")

        preds, evaluated, fps = vlm_classify_frames(
            frames, vlm_model, processor, task_desc,
            batch_size=batch_size, subsample=subsample,
        )
        total_fps_sum += fps
        all_preds.append(preds)
        all_gt.append(gt)

        m = compute_metrics(preds, gt)
        print(f"           F1={m['f1']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  agree={m['agreement']:.3f}  fps={fps:.2f}")

        if not no_plots and di < n_timeline_demos:
            T = len(frames)
            sample_ts = np.linspace(0, T - 1, min(8, T), dtype=int).tolist()
            raw = [np.asarray(frames[t]).transpose(1, 2, 0).astype(np.uint8)
                   for t in sample_ts]
            plot_vlm_timeline(
                preds, gt, evaluated,
                frames=raw, frame_times=sample_ts,
                out_path=tl_dir / f"demo{di:02d}_{task}.png",
                title=f"VLM | Demo {di} | {task}",
            )

    # 4. Pool metrics across all test demos
    pred_all = np.concatenate(all_preds)
    gt_all   = np.concatenate(all_gt)
    metrics  = compute_metrics(pred_all, gt_all)
    mean_fps = total_fps_sum / max(len(test_demos), 1)

    metrics["mean_fps"] = mean_fps
    metrics["ms_per_frame"] = 1000.0 / max(mean_fps, 1e-6)
    metrics["subsample"] = subsample
    metrics["model"] = model_name
    metrics["attn_implementation"] = attn_implementation
    metrics["task_description"] = task_desc
    metrics["gt_source"] = gt_source
    metrics["n_test_demos"] = len(test_demos)

    print(f"\n[vlm] POOLED  F1={metrics['f1']:.3f}  prec={metrics['precision']:.3f}  "
          f"rec={metrics['recall']:.3f}  agree={metrics['agreement']:.3f}")
    print(f"[vlm] Speed:  {mean_fps:.2f} frames/s  ({metrics['ms_per_frame']:.0f} ms/frame)")

    # 5. Load latent results for comparison (optional)
    latent_metrics = None
    if compare_json is not None:
        try:
            with open(compare_json) as f:
                raw = json.load(f)
            # The JSON has task as top-level key
            if task in raw:
                latent_metrics = raw[task]
                print(f"[vlm] Loaded latent metrics from {compare_json}")
            else:
                # Try first key
                first_key = next(iter(raw))
                latent_metrics = raw[first_key]
                print(f"[vlm] Loaded latent metrics (key='{first_key}') from {compare_json}")
        except Exception as e:
            print(f"[vlm] Could not load --compare-json: {e}")

    # 6. Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "vlm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[vlm] Metrics → {metrics_path}")

    if not no_plots:
        plot_comparison(
            metrics, latent_metrics, task,
            out_path=out_dir / "comparison_bar.png",
        )

    return metrics


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="VLM Boolean predicate evaluation (Qwen2-VL) for Newt tasks"
    )
    ap.add_argument("--task",       type=str,  required=True,
                    help="Task name, e.g. rd-push-green")
    ap.add_argument("--num-demos",  type=int,  default=30,
                    help="Total demos to collect (same split as latent eval)")
    ap.add_argument("--cal-frac",   type=float, default=0.40,
                    help="Cal fraction (skipped for VLM; kept for consistent test split)")
    ap.add_argument("--out-dir",    type=Path,  required=True,
                    help="Output root; results go under <out-dir>/<task>/")
    ap.add_argument("--compare-json", type=Path, default=None,
                    help="Path to spec_metrics.json from eval_newt_spec_predicates "
                         "for side-by-side comparison plot")
    ap.add_argument("--model",      type=str,
                    default="Qwen/Qwen2-VL-2B-Instruct",
                    help="HuggingFace model ID for the VLM "
                         "(2B for speed; use 7B-Instruct for final paper results)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Frames per VLM forward pass; 1 avoids batched-vision CUDA issues on some GPUs (raise if stable)")
    ap.add_argument("--subsample",  type=int,  default=5,
                    help="Evaluate every Nth frame; others are forward-filled. "
                         "5 = ~20 frames per 100-step demo (fast). "
                         "1 = evaluate all (slow but most accurate for paper)")
    ap.add_argument("--n-timeline-demos", type=int, default=6,
                    help="Number of test demos to plot timelines for")
    ap.add_argument("--prompt",      type=str,  default=None,
                    help="Override the natural language predicate description "
                         "(default: looked up from TASK_PROMPTS dict)")
    ap.add_argument("--gt-obs-index", type=int, default=None,
                    help="Use obs[:, INDEX] > --gt-obs-threshold as GT instead of "
                         "success >= 0.99.  E.g. 6 for mw-pick-place-wall grasp.")
    ap.add_argument("--gt-obs-threshold", type=float, default=0.05,
                    help="Threshold for --gt-obs-index (default 0.05)")
    ap.add_argument("--seed",       type=int,  default=42)
    ap.add_argument("--no-plots",   action="store_true")
    ap.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Transformer attention backend; eager avoids CUDA driver errors on some GPUs",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    task_out = args.out_dir / args.task
    task_out.mkdir(parents=True, exist_ok=True)

    result = evaluate_task(
        task=args.task,
        num_demos=args.num_demos,
        cal_frac=args.cal_frac,
        n_timeline_demos=args.n_timeline_demos,
        out_dir=task_out,
        seed=args.seed,
        model_name=args.model,
        batch_size=args.batch_size,
        subsample=args.subsample,
        no_plots=args.no_plots,
        compare_json=args.compare_json,
        attn_implementation=args.attn_implementation,
        prompt_override=args.prompt,
        gt_obs_index=args.gt_obs_index,
        gt_obs_threshold=args.gt_obs_threshold,
    )

    print(f"\n── SUMMARY ───────────────────────────────────────────────────────")
    print(f"  {args.task}: F1={result['f1']:.3f}  agree={result['agreement']:.3f}  "
          f"lift={result['lift']:+.3f}")
    print(f"  Speed: {result['mean_fps']:.2f} frames/s  ({result['ms_per_frame']:.0f} ms/frame)")
    print(f"  Metrics → {task_out / 'vlm_metrics.json'}")


if __name__ == "__main__":
    main()
