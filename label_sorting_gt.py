"""
Use Qwen2-VL to label per-timestep GT for D3IL sorting:
  - Is the red block inside its bin?
  - Is the blue block inside its bin?

Extracts one frame per rollout step from the pre-rendered video,
queries Qwen2-VL for each frame, and saves a JSON file with binary
GT labels per timestep.

Usage:
    conda activate newt
    python label_sorting_gt.py \
        --pkl  /path/to/data/sorting/rollouts/test/episode_s_0000.pkl \
        --video /path/to/data/sorting/rollouts/videos/test/episode_s_0000.mp4 \
        --out  sorting_gt.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


SYSTEM_PROMPT = (
    "You are a precise visual question answering assistant. "
    "Answer only 'yes' or 'no'. Do not explain."
)

QUESTION_RED  = ("Is the red block fully or mostly inside the red-marked target bin "
                 "(the red square region on the table)? Answer yes or no.")
QUESTION_BLUE = ("Is the blue block fully or mostly inside the blue-marked target bin "
                 "(the blue square region on the table)? Answer yes or no.")


def _patch_qwen2vl(model) -> None:
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


def load_vlm(model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu"):
    print(f"Loading {model_name} on {device}…")
    # max_pixels caps image token count to avoid OOM on visual encoder
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=256*28*28, max_pixels=256*28*28)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="eager",
    ).eval()
    _patch_qwen2vl(model)
    return model, processor, device


@torch.no_grad()
def ask(model, processor, pil_img: Image.Image, question: str, device: str) -> bool:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(gen, skip_special_tokens=True)[0].strip().lower()
    return answer.startswith("yes") or (
        "yes" in answer and "no" not in answer[:answer.find("yes") + 3]
    )


def extract_frames(pkl_path: Path, video_path: Path) -> list[np.ndarray]:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    T = len(d["rollout"])
    cap = cv2.VideoCapture(str(video_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.round(np.linspace(0, n_vid - 1, T)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                      else np.zeros((96, 192, 3), dtype=np.uint8))
    cap.release()
    return frames


def smooth_gt(labels: list[dict], min_run: int = 2) -> list[dict]:
    """Remove single-frame noise; once confirmed in bin, stays True."""
    T = len(labels)
    for key in ("red_in_bin", "blue_in_bin"):
        arr = np.array([l[key] for l in labels], dtype=bool)
        # remove isolated single-frame positives
        for t in range(1, T - 1):
            if arr[t] and not arr[t-1] and not arr[t+1]:
                arr[t] = False
        # find first sustained run
        first_confirmed, run = None, 0
        for t in range(T):
            if arr[t]:
                run += 1
                if run >= min_run and first_confirmed is None:
                    first_confirmed = t - run + 1
            else:
                run = 0
        if first_confirmed is not None:
            arr[:first_confirmed] = False  # clear pre-confirmation noise
            arr[first_confirmed:] = True
        else:
            arr[:] = False  # no confirmed run → all False
        for t, l in enumerate(labels):
            l[key] = bool(arr[t])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl",   required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   default="sorting_gt.json")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    print(f"Extracting frames…")
    frames = extract_frames(Path(args.pkl), Path(args.video))
    print(f"  {len(frames)} frames")

    model, processor, device = load_vlm(args.model, args.device)

    results = []
    for i, frame in enumerate(frames):
        pil = Image.fromarray(frame)
        red  = ask(model, processor, pil, QUESTION_RED,  device)
        blue = ask(model, processor, pil, QUESTION_BLUE, device)
        results.append({"t": i, "red_in_bin": red, "blue_in_bin": blue})
        print(f"  t={i:3d}  red={red}  blue={blue}")

    if not args.no_smooth:
        results = smooth_gt(results)
        print("  smoothed (irreversible placement)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")

    red_t  = next((l["t"] for l in results if l["red_in_bin"]),  None)
    blue_t = next((l["t"] for l in results if l["blue_in_bin"]), None)
    print(f"  red first in bin:  t={red_t}")
    print(f"  blue first in bin: t={blue_t}")


if __name__ == "__main__":
    main()
