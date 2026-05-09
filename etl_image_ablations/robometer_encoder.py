"""
robometer_encoder.py
--------------------
Extracts per-frame hidden-state features from the Robometer-4B backbone
(fine-tuned Qwen2.5-VL) for use as a drop-in ETL encoder.

These features sit directly upstream of the reward prediction head, making
them semantically richer than R3M/DINOv2 for robot manipulation phases.

Usage (from the robometer uv env):
  cd /path/to/robometer
  uv run python ./etl_image_ablations/robometer_encoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF

ROBOMETER_REPO = "robometer/Robometer-4B"
TARGET_IMG_SIZE = 224   # resize input frames before feeding to model


class RobometerEncoder:
    """
    Loads Robometer-4B (Qwen2.5-VL backbone) and extracts per-frame hidden
    states from just before the progress/success prediction heads.

    The latent for each frame is the mean-pooled vision-token hidden state
    (shape: hidden_dim = 3584 for Qwen2.5-VL-7B base).

    Typical usage:
        enc = RobometerEncoder()
        feats = enc.encode_frames(frames_list)   # (N, 3584)
    """

    def __init__(self, model_path: str = ROBOMETER_REPO, device: str = "cuda"):
        self.device = device
        self._load(model_path)

    def _load(self, model_path: str):
        import os, tempfile
        # Redirect TMPDIR to /tmp so temp writes don't hit the full root partition
        tmp_dir = "/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        os.environ["TMPDIR"] = tmp_dir
        tempfile.tempdir = tmp_dir

        sys.path.insert(0, str(Path(__file__).parents[2] / "robometer"))

        # Load config then disable unsloth (the checkpoint config has use_unsloth=True
        # but unsloth tries to re-download weights to a temp location on root)
        from robometer.utils.save import (
            resolve_checkpoint_path, parse_hf_model_id_and_revision, load_model_from_hf
        )
        from robometer.utils.setup_utils import setup_model_and_processor
        from robometer.configs.experiment_configs import ExperimentConfig
        from dataclasses import fields
        import yaml

        repo_id, revision = parse_hf_model_id_and_revision(model_path, model_name="checkpoint")
        resolved_path = resolve_checkpoint_path(model_path)

        # Read config.yaml
        from pathlib import Path as _Path
        config_path = str(_Path(resolved_path) / "config.yaml")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        valid_keys = {f.name for f in fields(ExperimentConfig)}
        exp_config = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid_keys})

        # Force-disable unsloth so model loads without needing temp disk on root
        exp_config.model.use_unsloth = False

        print(f"Loading Robometer backbone from {resolved_path} (unsloth disabled) ...")
        tokenizer, processor, reward_model = setup_model_and_processor(
            exp_config.model, resolved_path, peft_config=None
        )
        reward_model = reward_model.to(self.device)
        reward_model.eval()

        self.model      = reward_model
        self.processor  = processor
        self.tokenizer  = tokenizer
        self.exp_config = exp_config
        cfg = reward_model.model.config
        hidden_dim = (
            cfg.hidden_size if hasattr(cfg, "hidden_size")
            else cfg.text_config.hidden_size  # Qwen3VL / SmolVLM nested config
        )
        print(f"  Backbone: {reward_model.base_model_id}, hidden_dim={hidden_dim}")
        self.hidden_dim = hidden_dim

    @torch.no_grad()
    def encode_frames(self, frames: List[np.ndarray],
                      task: str = "robot manipulation",
                      chunk_size: int = 8,
                      stride: int = 4,
                      batch_size: int = 4) -> np.ndarray:
        """
        Encode a list of H×W×3 uint8 frames.
        Returns (N, hidden_dim) float32 array.

        Strategy: sliding anchors with stride < chunk_size, batched for GPU efficiency.
          - Every `stride` frames we compute one anchor embedding using a chunk of
            `chunk_size` frames ending at that anchor (so the anchor is always at
            position chunk_size-1 = full context).
          - `batch_size` anchor clips are processed in one forward pass.
          - Each frame is assigned the embedding of its nearest preceding anchor.

        With stride=4, chunk_size=8, batch_size=4:
          - 2× better temporal resolution vs non-overlapping stride-8
          - ~same GPU time (batching compensates for more anchors)
          - 5-frame GT windows always contain ≥1 anchor
        """
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.utils.setup_utils import setup_batch_collator

        N = len(frames)
        if N == 0:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)

        batch_collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True
        )

        # Pad beginning so first anchor (at index chunk_size-1) has full history
        padding = [frames[0]] * (chunk_size - 1)
        padded = padding + list(frames)  # length = N + chunk_size - 1

        # Anchor positions in original frame index space: stride-1, stride*2-1, ...
        # i.e. every `stride` frames starting from stride-1 (0-indexed)
        anchors = list(range(stride - 1, N, stride))
        if not anchors or anchors[-1] < N - 1:
            anchors.append(N - 1)  # always include last frame

        # Map anchor index → embedding (computed in batches)
        anchor_embeds: dict[int, np.ndarray] = {}
        for b_start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[b_start : b_start + batch_size]
            samples = []
            for a in batch_anchors:
                # Window ending at anchor a in padded space
                p_end = a + chunk_size        # exclusive in padded array
                p_start = p_end - chunk_size  # = a (since padding shifts by chunk_size-1)
                window = np.stack(padded[p_start : p_end])  # (chunk_size, H, W, 3)
                traj = Trajectory(
                    frames=window,
                    frames_shape=tuple(window.shape),
                    task=task,
                    id=str(a),
                    metadata={"subsequence_length": chunk_size},
                    video_embeddings=None,
                )
                samples.append(ProgressSample(trajectory=traj, sample_type="progress"))

            batch = batch_collator(samples)
            progress_inputs = batch["progress_inputs"]
            for k, v in progress_inputs.items():
                if hasattr(v, "to"):
                    progress_inputs[k] = v.to(self.device)

            feats = self._forward_backbone(progress_inputs)  # (B*chunk_size, hidden_dim)
            # feats is concatenated across batch items; split back out
            per_clip = np.split(feats, len(batch_anchors), axis=0)
            for a, clip_feats in zip(batch_anchors, per_clip):
                anchor_embeds[a] = clip_feats[-1]  # last frame = anchor embedding

        # Assign each frame the embedding of its nearest preceding anchor
        result = np.empty((N, self.hidden_dim), dtype=np.float32)
        prev_embed = anchor_embeds[anchors[0]]
        ai = 0
        for t in range(N):
            if ai < len(anchors) and t >= anchors[ai]:
                prev_embed = anchor_embeds[anchors[ai]]
                ai += 1
            result[t] = prev_embed
        return result

    def _encode_chunk(self, frames_uint8: np.ndarray, task: str,
                      batch_collator) -> np.ndarray:
        """Encode one chunk of frames; returns (T, hidden_dim)."""
        from robometer.data.dataset_types import ProgressSample, Trajectory

        T = frames_uint8.shape[0]
        traj = Trajectory(
            frames=frames_uint8,
            frames_shape=tuple(frames_uint8.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch  = batch_collator([sample])

        progress_inputs = batch["progress_inputs"]
        for k, v in progress_inputs.items():
            if hasattr(v, "to"):
                progress_inputs[k] = v.to(self.device)

        # Forward through backbone only — hook out hidden states
        feats = self._forward_backbone(progress_inputs)
        return feats  # (T, hidden_dim)

    def _forward_backbone(self, progress_inputs: dict) -> np.ndarray:
        """
        Run the VL backbone directly and extract per-frame embeddings via
        _extract_hidden_states_from_token_pairs — bypassing the reward heads.
        """
        model = self.model
        is_qwen3 = "Qwen3" in model.base_model_id

        # Keys accepted by the underlying Qwen backbone (not the RBM wrapper)
        _BACKBONE_KEYS = {
            "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw", "second_per_grid_ts",
        }
        model_kwargs = {k: v for k, v in progress_inputs.items() if k in _BACKBONE_KEYS}

        with torch.autocast(self.device, dtype=torch.bfloat16):
            if is_qwen3:
                outputs = model.model(**model_kwargs, output_hidden_states=True, return_dict=True)
                hidden_state = outputs.hidden_states[-1]   # [B, seq_len, hidden_dim]
            else:
                outputs = model.model(**model_kwargs)
                hidden_state = outputs.last_hidden_state   # [B, seq_len, hidden_dim]

        input_ids = progress_inputs["input_ids"]           # [B, seq_len]
        all_embeds = []
        for i in range(hidden_state.shape[0]):
            frame_embeds = model._extract_hidden_states_from_token_pairs(
                hidden_state[i], input_ids[i]
            )  # [T_i, hidden_dim]
            all_embeds.append(frame_embeds)

        if not all_embeds:
            raise RuntimeError("No frame embeddings extracted from Robometer backbone")

        result = torch.cat(all_embeds, dim=0).detach().float().cpu()  # [T, hidden_dim]
        return result.numpy().astype(np.float32)


def smoke_test():
    import numpy as np
    enc = RobometerEncoder()
    # 10 random frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    feats = enc.encode_frames(frames, task="Pick up the green block")
    print(f"Output shape: {feats.shape}  (expected (10, {enc.hidden_dim}))")
    print(f"Feature norm range: {np.linalg.norm(feats, axis=1).min():.2f} - {np.linalg.norm(feats, axis=1).max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model-path", default=ROBOMETER_REPO)
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
