"""
eval_vlm_real_robot.py
----------------------
VLM baseline (Qwen2-VL) for real-robot predicate evaluation on FMB and DROID.
Mirrors eval_vlm_predicates.py but for lerobot datasets (no Newt agent needed).

Datasets:
  fmb   — lerobot/fmb, side camera, pick/insert GT from task_index
  droid — lerobot/droid_100, wrist camera, grasp GT from state[:,6] > 0.5

GT is identical to eval_lerobot_etl.py and eval_droid_etl.py so metrics are
directly comparable.

Usage:
  cd /path/to/repo
  HF_HOME=~/.cache/huggingface \\
  python -m etl_image_ablations.eval_vlm_real_robot \\
      --dataset fmb --num-episodes 60 \\
      --out-dir etl_results/vlm_real_robot/fmb

  HF_HOME=~/.cache/huggingface \\
  python -m etl_image_ablations.eval_vlm_real_robot \\
      --dataset droid --num-episodes 40 \\
      --out-dir etl_results/vlm_real_robot/droid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# Dataset constants (mirrors eval_lerobot_etl.py / eval_droid_etl.py)
# ──────────────────────────────────────────────────────────────────────────

FMB_REPO        = "lerobot/fmb"
FMB_VIDEO_KEY   = "observation.images.image_side_1"
FMB_PICK_IDS    = {0, 4, 6, 10, 16, 20}
FMB_INSERT_IDS  = {3, 5, 9, 11, 17, 21}

DROID_REPO      = "lerobot/droid_100"
DROID_CAM_KEY   = "observation.images.wrist_image_left"
DROID_GRASP_THRESH = 0.5   # state[:,6] > this = gripper closed

# ──────────────────────────────────────────────────────────────────────────
# Data loading (same approach as existing ETL scripts)
# ──────────────────────────────────────────────────────────────────────────

def _hf_download(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_meta(repo_id: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    info  = json.load(open(_hf_download(repo_id, "meta/info.json")))
    data  = pd.read_parquet(_hf_download(repo_id, "data/chunk-000/file-000.parquet"))
    eps   = pd.read_parquet(_hf_download(repo_id, "meta/episodes/chunk-000/file-000.parquet"))
    return info, data, eps


def decode_mp4(path: Path, t0: float, t1: float) -> List[np.ndarray]:
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(t0 * 1_000_000))
        for pkt in c.demux(s):
            for frm in pkt.decode():
                t = float(frm.pts * s.time_base)
                if t < t0 - 0.001: continue
                if t > t1 + 0.001: return frames
                frames.append(frm.to_ndarray(format="rgb24"))
    return frames


def load_episode_frames(repo_id: str, ep_row: pd.Series, video_key: str) -> List[np.ndarray]:
    chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
    fidx  = int(ep_row[f"videos/{video_key}/file_index"])
    t0    = float(ep_row[f"videos/{video_key}/from_timestamp"])
    t1    = float(ep_row[f"videos/{video_key}/to_timestamp"])
    mp4   = f"videos/{video_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    return decode_mp4(_hf_download(repo_id, mp4), t0, t1)


# ──────────────────────────────────────────────────────────────────────────
# VLM loading and inference (mirrors eval_vlm_predicates.py)
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are evaluating whether a robot has successfully completed a manipulation task. "
    "Look carefully at the image and answer only with 'yes' or 'no'."
)


def load_vlm(model_name: str, attn: str = "eager"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"[vlm] Loading {model_name} …")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attn,
    ).eval()
    _patch_qwen2vl_split_sizes_cpu(model)
    print(f"[vlm] Loaded in {time.time()-t0:.1f}s")
    return model, processor


def _patch_qwen2vl_split_sizes_cpu(model) -> None:
    """Same workaround as eval_vlm_predicates.py for CUDA .prod().tolist() on some GPUs."""
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


@torch.no_grad()
def vlm_classify(
    frames: List[np.ndarray],   # list of H×W×3 uint8
    model,
    processor,
    question: str,
    subsample: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Returns predictions [T] bool and fps.
    Non-evaluated frames are forward-filled from the previous evaluated frame.
    """
    T = len(frames)
    preds = np.zeros(T, dtype=bool)
    eval_idx = list(range(0, T, subsample))

    t_start = time.time()
    for i in eval_idx:
        pil = Image.fromarray(frames[i], mode="RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to("cuda:0")
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(gen, skip_special_tokens=True)[0].strip().lower()
        preds[i] = answer.startswith("yes") or (
            "yes" in answer and "no" not in answer[:answer.find("yes") + 3]
        )

    elapsed = time.time() - t_start
    fps = len(eval_idx) / max(elapsed, 1e-6)

    # Forward-fill non-evaluated frames
    last = False
    for t in range(T):
        if t in set(eval_idx):
            last = preds[t]
        else:
            preds[t] = last

    return preds, fps


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = (pred_b & gt_b).sum();  fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum(); tn = (~pred_b & ~gt_b).sum()
    p  = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    pos = gt_b.mean()
    return dict(
        precision=float(p), recall=float(r), f1=float(f1),
        agreement=float((pred_b == gt_b).mean()),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        gt_positive_rate=float(pos),
        baseline_f1=float(2 * pos / (1 + pos + 1e-9)),
        lift=float(f1 - 2 * pos / (1 + pos + 1e-9)),
    )


# ──────────────────────────────────────────────────────────────────────────
# FMB evaluation
# ──────────────────────────────────────────────────────────────────────────

FMB_PICK_Q   = "Is the robot arm currently picking up or grasping an object with its gripper?"
FMB_INSERT_Q = "Is the robot arm currently inserting an object into a slot or socket?"


def evaluate_fmb(
    model, processor,
    num_episodes: int,
    out_dir: Path,
    subsample: int,
    seed: int,
) -> Dict[str, Any]:
    print("\n=== FMB ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, data_df, ep_df = load_meta(FMB_REPO)

    # Keep episodes that contain both pick and insert phases
    valid_eps = []
    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        if len(ep_data) == 0:
            continue
        tids = set(ep_data["task_index"].values)
        if tids & FMB_PICK_IDS and tids & FMB_INSERT_IDS:
            valid_eps.append(ep_idx)

    rng = np.random.default_rng(seed)
    rng.shuffle(valid_eps)
    selected = valid_eps[:min(num_episodes, len(valid_eps))]
    print(f"[fmb] {len(selected)} valid episodes (of {len(valid_eps)} with both pick+insert)")

    all_pick_pred, all_pick_gt   = [], []
    all_ins_pred,  all_ins_gt    = [], []
    seq_correct = 0
    fps_sum = 0.0

    for ep_idx in selected:
        ep_row  = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        ep_data = data_df[data_df["episode_index"] == ep_idx]

        frames = load_episode_frames(FMB_REPO, ep_row, FMB_VIDEO_KEY)
        n_vid = len(frames)
        n_tab = len(ep_data)
        T = min(n_vid, n_tab)
        if T == 0:
            continue
        frames = frames[:T]

        tid_arr   = ep_data["task_index"].values[:T]
        pick_gt   = np.array([int(t) in FMB_PICK_IDS   for t in tid_arr])
        insert_gt = np.array([int(t) in FMB_INSERT_IDS for t in tid_arr])

        print(f"[fmb] ep{ep_idx:04d}  T={T}  pick={pick_gt.sum()}  insert={insert_gt.sum()}")

        pick_pred,   fps1 = vlm_classify(frames, model, processor, FMB_PICK_Q,   subsample)
        insert_pred, fps2 = vlm_classify(frames, model, processor, FMB_INSERT_Q, subsample)
        fps_sum += (fps1 + fps2) / 2

        all_pick_pred.append(pick_pred);   all_pick_gt.append(pick_gt)
        all_ins_pred.append(insert_pred);  all_ins_gt.append(insert_gt)

        # Sequential ordering: first pick fire before first insert fire?
        pick_fires   = np.where(pick_pred)[0]
        insert_fires = np.where(insert_pred)[0]
        if len(pick_fires) > 0 and len(insert_fires) > 0:
            seq_correct += int(pick_fires[0] < insert_fires[0])
        elif len(pick_fires) == 0 or len(insert_fires) == 0:
            seq_correct += 0  # at least one predicate never fired

    n_ep = len(all_pick_pred)
    pick_metrics   = compute_metrics(np.concatenate(all_pick_pred),  np.concatenate(all_pick_gt))
    insert_metrics = compute_metrics(np.concatenate(all_ins_pred),   np.concatenate(all_ins_gt))

    result = {
        "dataset": "fmb",
        "model": model.config.name_or_path if hasattr(model.config, "name_or_path") else "Qwen2-VL",
        "subsample": subsample,
        "n_episodes": n_ep,
        "mean_fps": fps_sum / max(n_ep, 1),
        "pick": pick_metrics,
        "insert": insert_metrics,
        "sequential": {
            "correct": seq_correct,
            "total": n_ep,
            "agreement": seq_correct / max(n_ep, 1),
        },
    }

    metrics_path = out_dir / "vlm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[fmb] Pick   F1={pick_metrics['f1']:.3f}  agree={pick_metrics['agreement']:.3f}  "
          f"fn={pick_metrics['fn']}")
    print(f"[fmb] Insert F1={insert_metrics['f1']:.3f}  agree={insert_metrics['agreement']:.3f}  "
          f"fn={insert_metrics['fn']}")
    print(f"[fmb] Sequential ordering: {seq_correct}/{n_ep} = {seq_correct/max(n_ep,1):.1%}")
    print(f"[fmb] Metrics → {metrics_path}")
    return result


# ──────────────────────────────────────────────────────────────────────────
# DROID evaluation
# ──────────────────────────────────────────────────────────────────────────

DROID_GRASP_Q = (
    "Is the robot gripper currently closed and grasping or holding an object? "
    "Answer yes if the gripper fingers appear closed around something, no if the gripper is open."
)


def evaluate_droid(
    model, processor,
    num_episodes: int,
    out_dir: Path,
    subsample: int,
    seed: int,
) -> Dict[str, Any]:
    print("\n=== DROID ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, data_df, ep_df = load_meta(DROID_REPO)

    rng = np.random.default_rng(seed)
    all_eps = ep_df["episode_index"].values.tolist()
    rng.shuffle(all_eps)
    selected = all_eps[:min(num_episodes, len(all_eps))]
    print(f"[droid] {len(selected)} episodes selected")

    all_pred, all_gt = [], []
    fps_sum = 0.0
    n_valid = 0

    for ep_idx in selected:
        ep_row  = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        if len(ep_data) == 0:
            continue

        # GT: gripper closed
        state_rows = ep_data["observation.state"].values
        if len(state_rows) == 0:
            continue
        state = np.stack(state_rows)  # [T, D]
        grasp_gt = (state[:, 6] > DROID_GRASP_THRESH)

        frames = load_episode_frames(DROID_REPO, ep_row, DROID_CAM_KEY)
        T = min(len(frames), len(grasp_gt))
        if T == 0:
            continue
        frames   = frames[:T]
        grasp_gt = grasp_gt[:T]

        print(f"[droid] ep{ep_idx:04d}  T={T}  grasp_gt={grasp_gt.sum()}")

        preds, fps = vlm_classify(frames, model, processor, DROID_GRASP_Q, subsample)
        fps_sum += fps
        n_valid += 1

        all_pred.append(preds)
        all_gt.append(grasp_gt)

    metrics = compute_metrics(np.concatenate(all_pred), np.concatenate(all_gt))
    result = {
        "dataset": "droid",
        "model": "Qwen2-VL",
        "subsample": subsample,
        "n_episodes": n_valid,
        "mean_fps": fps_sum / max(n_valid, 1),
        "grasp": metrics,
    }

    metrics_path = out_dir / "vlm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[droid] Grasp F1={metrics['f1']:.3f}  agree={metrics['agreement']:.3f}  "
          f"fn={metrics['fn']}")
    print(f"[droid] Metrics → {metrics_path}")
    return result


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VLM predicate evaluation on real-robot datasets")
    ap.add_argument("--dataset", choices=["fmb", "droid"], required=True)
    ap.add_argument("--num-episodes", type=int, default=40,
                    help="Episodes to evaluate (FMB default 60, DROID default 40)")
    ap.add_argument("--out-dir",  type=Path, required=True)
    ap.add_argument("--model",    type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--subsample", type=int, default=5,
                    help="Evaluate every Nth frame (default 5)")
    ap.add_argument("--attn",     type=str, default="eager",
                    choices=("eager", "sdpa", "flash_attention_2"))
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_vlm(args.model, args.attn)

    if args.dataset == "fmb":
        evaluate_fmb(model, processor,
                     num_episodes=args.num_episodes,
                     out_dir=args.out_dir,
                     subsample=args.subsample,
                     seed=args.seed)
    else:
        evaluate_droid(model, processor,
                       num_episodes=args.num_episodes,
                       out_dir=args.out_dir,
                       subsample=args.subsample,
                       seed=args.seed)


if __name__ == "__main__":
    main()
