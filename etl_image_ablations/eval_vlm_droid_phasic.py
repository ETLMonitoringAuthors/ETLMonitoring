"""
eval_vlm_droid_phasic.py
------------------------
VLM baseline (Qwen2-VL) for the phasic DROID episodes used in eval_droid_phasic.py.

Loads the phasic metrics JSON (which has episode IDs and gripper-close windows),
then for each episode/phase runs the VLM with a task- and phase-specific prompt.
GT masks are the same gripper-close windows as the ETL evaluation so metrics are
directly comparable.

Usage:
  cd /path/to/repo
  HF_HOME=~/.cache/huggingface \\
  python -m etl_image_ablations.eval_vlm_droid_phasic \\
      --phasic-json etl_results/droid_phasic/metrics_phasic.json \\
      --out-dir etl_results/vlm_droid_phasic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import av
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import pandas as pd

# ── Constants (mirror eval_droid_phasic.py) ──────────────────────────────────

REPO_ID       = "lerobot/droid_100"
DROID_CAM_KEY = "observation.images.wrist_image_left"

SYSTEM_PROMPT = (
    "You are evaluating whether a robot arm is currently performing a specific "
    "manipulation action. Look carefully at the image and answer only with 'yes' or 'no'."
)

# Per-episode, per-phase VLM prompts.
# Keys are episode indices (int); values are lists of prompts, one per phase.
PHASE_PROMPTS: Dict[int, List[str]] = {
    13: [
        "Is the robot gripper currently closed and holding the black object it picked up from the towel?",
        "Is the robot gripper currently closed and holding the black object while moving it toward or above the white sheet of paper?",
    ],
    52: [
        "Is the robot gripper currently grasping or turning the tap/faucet handle?",
        "Is the robot gripper currently holding the spoon?",
    ],
    56: [
        "Is the robot gripper currently holding the white towel?",
        "Is the robot gripper currently holding the marker?",
        "Is the robot gripper currently holding the yellow towel?",
        "Is the robot gripper currently holding the masking tape?",
    ],
    71: [
        "Is the robot gripper currently holding and moving the white cloth or object from the right heap?",
        "Is the robot gripper currently holding the blue-white cloth or object?",
        "Is the robot gripper currently pressing down or folding the blue-white cloth onto the left heap?",
    ],
    97: [
        "Is the robot gripper currently holding the first bottle from the stove?",
        "Is the robot gripper currently holding the second bottle from the stove?",
        "Is the robot gripper currently holding the cooking stick or spatula?",
    ],
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _hf_download(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_meta() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_p = _hf_download(REPO_ID, "data/chunk-000/file-000.parquet")
    ep_p   = _hf_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet")
    return pd.read_parquet(data_p), pd.read_parquet(ep_p)


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


def load_episode_frames(ep_row: pd.Series) -> List[np.ndarray]:
    cam_key = DROID_CAM_KEY
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    fidx  = int(ep_row[f"videos/{cam_key}/file_index"])
    t0    = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    t1    = float(ep_row[f"videos/{cam_key}/to_timestamp"])
    mp4   = f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    return decode_mp4(_hf_download(REPO_ID, mp4), t0, t1)


# ── VLM ───────────────────────────────────────────────────────────────────────

def load_vlm(model_name: str, attn: str = "eager"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"[vlm] Loading {model_name} …")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attn,
    ).eval()
    _patch_qwen2vl_split_sizes_cpu(model)
    print(f"[vlm] Loaded in {time.time()-t0:.1f}s")
    return model, processor


def _patch_qwen2vl_split_sizes_cpu(model) -> None:
    inner = model.model
    merge_sq = inner.visual.spatial_merge_size ** 2

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
    frames: List[np.ndarray],
    model,
    processor,
    question: str,
    subsample: int = 5,
) -> Tuple[np.ndarray, float]:
    T = len(frames)
    preds = np.zeros(T, dtype=bool)
    eval_idx = list(range(0, T, subsample))

    t_start = time.time()
    for i in eval_idx:
        pil = Image.fromarray(frames[i], mode="RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
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

    # Forward-fill
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
    return dict(
        precision=float(p), recall=float(r), f1=float(f1),
        agreement=float((pred_b == gt_b).mean()),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.phasic_json) as f:
        phasic = json.load(f)

    print("Loading DROID metadata …")
    data_df, ep_df = load_meta()

    model, processor = load_vlm(args.model, args.attn)

    all_ep_results = []

    for ep_data in phasic["episodes"]:
        ep_idx = ep_data["episode"]
        task   = ep_data["task"]
        windows = [tuple(ph["window"]) for ph in ep_data["phases"]]

        if ep_idx not in PHASE_PROMPTS:
            print(f"[warn] No prompts defined for ep{ep_idx}, skipping")
            continue

        prompts = PHASE_PROMPTS[ep_idx]
        if len(prompts) != len(windows):
            print(f"[warn] ep{ep_idx}: {len(prompts)} prompts but {len(windows)} phases — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"ep{ep_idx}  n_phases={len(windows)}")
        print(f"  task: {task}")

        ep_row = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        frames = load_episode_frames(ep_row)

        ep_tab = data_df[data_df["episode_index"] == ep_idx]
        state  = np.stack(ep_tab.reset_index(drop=True)["observation.state"].values)
        T = min(len(frames), len(state))
        frames = frames[:T]

        phase_results = []
        pred_lists    = []

        for k, ((ws, we), prompt) in enumerate(zip(windows, prompts)):
            gt_k = np.zeros(T, bool)
            gt_k[ws:we] = True

            print(f"  phase {k+1} [{ws},{we}]  prompt: {prompt[:70]}")
            preds, fps = vlm_classify(frames, model, processor, prompt, args.subsample)
            metrics = compute_metrics(preds, gt_k)
            pred_lists.append(preds)
            phase_results.append(metrics)
            print(f"    F1={metrics['f1']:.3f}  agree={metrics['agreement']:.3f}  "
                  f"fps={fps:.1f}")

        # Sequential ordering: phase k fires before phase k+1
        seq_ok = True
        for k in range(len(windows) - 1):
            first_k  = np.argmax(pred_lists[k])   if pred_lists[k].any()   else T
            first_k1 = np.argmax(pred_lists[k+1]) if pred_lists[k+1].any() else T
            ok = first_k < first_k1
            print(f"  phase{k+1}→phase{k+2}: first fire {first_k} vs {first_k1}  "
                  f"({'✓' if ok else '✗'})")
            if not ok:
                seq_ok = False

        print(f"  Sequential: {'✓' if seq_ok else '✗'}")

        all_ep_results.append({
            "episode": ep_idx,
            "task": task,
            "n_phases": len(windows),
            "phases": [{"window": list(w), **r}
                       for w, r in zip(windows, phase_results)],
            "sequential_correct": seq_ok,
            "mean_phase_f1": float(np.mean([r["f1"] for r in phase_results])),
            "mean_phase_agreement": float(np.mean([r["agreement"] for r in phase_results])),
        })

    # Summary
    print(f"\n{'='*60}")
    print("VLM PHASIC DROID SUMMARY")
    if all_ep_results:
        seq_rate  = float(np.mean([r["sequential_correct"] for r in all_ep_results]))
        mean_f1   = float(np.mean([r["mean_phase_f1"] for r in all_ep_results]))
        mean_agr  = float(np.mean([r["mean_phase_agreement"] for r in all_ep_results]))
        n_correct = sum(r["sequential_correct"] for r in all_ep_results)
        print(f"  Mean phase F1:        {mean_f1:.3f}")
        print(f"  Mean phase agreement: {mean_agr:.3f}")
        print(f"  Sequential correct:   {n_correct}/{len(all_ep_results)} = {seq_rate:.3f}")

        metrics_out = {
            "model": args.model,
            "subsample": args.subsample,
            "n_episodes": len(all_ep_results),
            "mean_phase_f1": mean_f1,
            "mean_phase_agreement": mean_agr,
            "sequential_rate": seq_rate,
            "episodes": all_ep_results,
        }
    else:
        metrics_out = {"model": args.model, "n_episodes": 0, "episodes": []}

    out_path = out_dir / "metrics_phasic_vlm.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasic-json", type=Path,
                    default="etl_results/droid_phasic/metrics_phasic.json")
    ap.add_argument("--out-dir",     type=Path,
                    default="etl_results/vlm_droid_phasic")
    ap.add_argument("--model",       type=str,
                    default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--subsample",   type=int, default=5)
    ap.add_argument("--attn",        type=str, default="eager",
                    choices=("eager", "sdpa", "flash_attention_2"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    evaluate(args)


if __name__ == "__main__":
    main()
