"""
eval_lerobot_etl.py
-------------------
ETL (Embedding Temporal Logic) analysis on real robot LeRobot datasets using
frozen DINOv2 ViT-B/14 as vision encoder.

Datasets:
  lerobot/iamlab_cmu_pickup_insert
    - Each episode is a single subtask: "Pick up X" (task_index 0,1,3,5) or
      "Insert X" (task_index 2,4,6).
    - Reach spec per subtask: π fires in the done-window. Cross-task
      discrimination: π_pick should stay low during insert episodes.

  lerobot/fmb (Functional Manipulation Benchmark)
    - Each episode contains a full sequential subtask chain, e.g.:
        Pick up blue → Move up → Above board → Insert blue
      with `task_index` labeling each frame's current subtask.
    - Sequential spec:  ∃ t1 < t2  s.t. π_pick(t1) ∧ π_insert(t2)
      GT from task_index directly.

Ground truth:
  - next.done = True  →  episode complete (last frame GT positive)
  - task_index per frame  →  subtask GT (FMB sequential, iamlab task type)

Threshold mining (calibration split):
  τ_F1  – F1-optimal (supervised upper bound)
  τ_CP  – class-conditional split conformal (α=0.10 → recall ≥ 0.90)

Usage:
  cd /path/to/repo
  MUJOCO_GL=egl python -m etl_image_ablations.eval_lerobot_etl \\
      --dataset iamlab --num-episodes 60 --out-dir etl_results/lerobot_iamlab
  MUJOCO_GL=egl python -m etl_image_ablations.eval_lerobot_etl \\
      --dataset fmb   --num-episodes 60 --out-dir etl_results/lerobot_fmb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# ─── DINOv2 encoder ───────────────────────────────────────────────────────────

class DINOv2Encoder:
    """Frozen DINOv2 ViT-B/14, output 768-D CLS token."""

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading DINOv2 ViT-B/14 …")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False
        ).to(device).eval()
        self.mean = self.MEAN.to(device)
        self.std  = self.STD.to(device)

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        frames: list of H×W×3 uint8 numpy arrays
        returns: (N, 768) float32 numpy array
        """
        import torchvision.transforms.functional as TF
        all_z = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            tensors = []
            for f in batch:
                t = torch.from_numpy(f).permute(2, 0, 1).float().div(255.0).to(self.device)
                t = TF.resize(t, [224, 224], antialias=True)
                t = (t - self.mean) / self.std
                tensors.append(t)
            x = torch.stack(tensors)
            z = self.model(x)  # (B, 768)
            all_z.append(z.cpu().numpy())
        return np.concatenate(all_z, axis=0)


# ─── R3M encoder ──────────────────────────────────────────────────────────────

class R3MEncoder:
    """R3M ResNet50 — trained on Ego4D human video with time-contrastive + language loss.
    Outputs 2048-D features. Designed for robot manipulation reward/goal prediction."""

    def __init__(self, device: str = "cuda", backbone: str = "resnet50"):
        import torchvision.transforms.functional as TF
        self.device = device
        self.TF = TF
        print(f"Loading R3M ({backbone}) ...")
        from r3m import load_r3m
        self.model = load_r3m(backbone)
        self.model.eval().to(device)
        print("  R3M loaded OK")

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """frames: list of H x W x 3 uint8.  Returns (N, 2048) float32."""
        all_z = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            tensors = []
            for f in batch:
                t = torch.from_numpy(f).permute(2, 0, 1).float().to(self.device)
                t = self.TF.resize(t, [224, 224], antialias=True)
                tensors.append(t)
            x = torch.stack(tensors)   # already in [0, 255] — R3M expects this
            z = self.model(x)          # (B, 2048)
            all_z.append(z.cpu().numpy())
        return np.concatenate(all_z, axis=0)


# ─── LeRobot data loading ──────────────────────────────────────────────────────

def download(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_dataset_meta(repo_id: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (info, data_df, episodes_df)."""
    info_path = download(repo_id, "meta/info.json")
    with open(info_path) as f:
        info = json.load(f)
    data_path  = download(repo_id, "data/chunk-000/file-000.parquet")
    data_df    = pd.read_parquet(data_path)
    ep_path    = download(repo_id, "meta/episodes/chunk-000/file-000.parquet")
    ep_df      = pd.read_parquet(ep_path)
    return info, data_df, ep_df


def decode_mp4_frames(mp4_path: Path, from_ts: float, to_ts: float) -> List[np.ndarray]:
    """Decode all frames in [from_ts, to_ts] seconds from an mp4 file.

    NOTE: seek with container (not stream) so the timestamp is in AV_TIME_BASE
    (microseconds = 1e6), which is independent of the stream's time_base.
    """
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        # AV_TIME_BASE = 1_000_000 (microseconds)
        container.seek(int(from_ts * 1_000_000))
        for packet in container.demux(stream):
            for frame in packet.decode():
                t = float(frame.pts * stream.time_base)
                if t < from_ts - 0.001:
                    continue
                if t > to_ts + 0.001:
                    return frames
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)
    return frames


def load_episode_frames(
    repo_id: str,
    ep_row: pd.Series,
    video_key: str,
) -> List[np.ndarray]:
    """Download the mp4 file for a single episode and return its frames."""
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col  = f"videos/{video_key}/file_index"
    ts_from   = f"videos/{video_key}/from_timestamp"
    ts_to     = f"videos/{video_key}/to_timestamp"
    chunk_idx = int(ep_row[chunk_col])
    file_idx  = int(ep_row[file_col])
    from_ts   = float(ep_row[ts_from])
    to_ts     = float(ep_row[ts_to])
    mp4_file  = f"videos/{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
    mp4_path  = download(repo_id, mp4_file)
    return decode_mp4_frames(mp4_path, from_ts, to_ts)


# ─── ETL core ──────────────────────────────────────────────────────────────────

def build_spec_latent(
    embeddings: np.ndarray,
    positive_mask: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Centroid of last `window` positive frames (done-window spec latent)."""
    pos_idx = np.where(positive_mask)[0]
    if len(pos_idx) == 0:
        return embeddings[-min(window, len(embeddings)):].mean(axis=0)
    sel = pos_idx[-min(window, len(pos_idx)):]
    return embeddings[sel].mean(axis=0)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    return embeddings / norms


def l2_distances(embeddings: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    """Per-frame L2 distance to spec latent."""
    return np.linalg.norm(embeddings - z_spec[None], axis=1)


def cosine_distances(embeddings: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    """Per-frame cosine distance (1 - cos_sim) to spec latent."""
    e_norm = l2_normalize(embeddings)
    s_norm = z_spec / max(np.linalg.norm(z_spec), 1e-8)
    return 1.0 - (e_norm @ s_norm)


def phase_done_mask(task_ids: np.ndarray, phase_ids: set, window: int = 5) -> np.ndarray:
    """
    True only for the last `window` frames of each contiguous run of task_ids
    that belongs to phase_ids.  This captures the 'phase completion' moment rather
    than all frames in the phase (which include approach / pre-grasp).
    """
    in_phase = np.array([int(t) in phase_ids for t in task_ids])
    done = np.zeros(len(in_phase), dtype=bool)
    i = 0
    while i < len(in_phase):
        if in_phase[i]:
            j = i
            while j < len(in_phase) and in_phase[j]:
                j += 1
            # [i, j) is one contiguous phase run; mark last `window` frames
            done[max(i, j - window):j] = True
            i = j
        else:
            i += 1
    return done


def sweep_f1(
    distances: np.ndarray,
    labels: np.ndarray,
    n_thresh: int = 200,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Sweep distance thresholds; return (tau_F1, best_F1, best_prec, best_rec, thresholds).
    Predicate fires when distance < tau.
    """
    taus = np.linspace(distances.min(), distances.max(), n_thresh)
    best_f1, best_tau, best_p, best_r = 0.0, taus[0], 0.0, 0.0
    for tau in taus:
        pred = (distances < tau).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau, best_p, best_r = f1, tau, prec, rec
    return best_tau, best_f1, best_p, best_r, taus


def conformal_tau(
    distances: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
) -> Optional[float]:
    """
    Class-conditional split-conformal threshold for recall ≥ 1-alpha.
    Returns the (1-alpha) quantile of distances among GT-positive frames.
    """
    pos_dists = distances[labels == 1]
    if len(pos_dists) < 2:
        return None
    n = len(pos_dists)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(pos_dists, q_level))


def predicate_metrics(
    distances: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> dict:
    pred = (distances < tau).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tau=tau, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn,
                agreement=(tp + tn) / (tp + fp + fn + tn + 1e-9))


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_timeline(
    distances: np.ndarray,
    gt_labels: np.ndarray,
    tau_f1: float,
    tau_cp: Optional[float],
    title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    T = len(distances)
    t = np.arange(T)

    # Top: GT labels
    axes[0].fill_between(t, 0, gt_labels.astype(float), alpha=0.4, color="green", label="GT positive")
    axes[0].set_ylabel("GT label")
    axes[0].set_ylim(-0.05, 1.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title, fontsize=10)

    # Bottom: distance + thresholds
    axes[1].plot(t, distances, color="steelblue", lw=1.2, label="d(z_t, z_spec)")
    axes[1].axhline(tau_f1, color="red",    ls="--", lw=1.5, label=f"τ_F1 = {tau_f1:.3f}")
    if tau_cp is not None:
        axes[1].axhline(tau_cp, color="orange", ls=":",  lw=1.5, label=f"τ_CP  = {tau_cp:.3f}")
    axes[1].set_ylabel("L2 distance")
    axes[1].set_xlabel("Frame index")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_f1_curve(
    distances: np.ndarray,
    labels: np.ndarray,
    tau_f1: float,
    tau_cp: Optional[float],
    title: str,
    out_path: Path,
):
    taus = np.linspace(distances.min(), distances.max(), 300)
    f1s, precs, recs = [], [], []
    for tau in taus:
        m = predicate_metrics(distances, labels, tau)
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(taus, f1s,   label="F1",        color="steelblue")
    ax.plot(taus, precs, label="Precision", color="green",   ls="--")
    ax.plot(taus, recs,  label="Recall",    color="orange",  ls=":")
    ax.axvline(tau_f1, color="red",    ls="--", lw=1.5, label=f"τ_F1={tau_f1:.3f}")
    if tau_cp is not None:
        ax.axvline(tau_cp, color="purple", ls=":",  lw=1.5, label=f"τ_CP={tau_cp:.3f}")
    ax.set_xlabel("τ (distance threshold)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_sequential_timeline(
    dist_A: np.ndarray,
    dist_B: np.ndarray,
    gt_A: np.ndarray,
    gt_B: np.ndarray,
    tau_A: float,
    tau_B: float,
    title: str,
    out_path: Path,
):
    """Three-panel: GT phases, distance to A, distance to B."""
    T = len(dist_A)
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # GT phases
    axes[0].fill_between(t, 0, gt_A.astype(float), alpha=0.4, color="steelblue", label="GT phase A (pick)")
    axes[0].fill_between(t, 0, gt_B.astype(float), alpha=0.4, color="green",     label="GT phase B (insert)")
    axes[0].set_ylabel("GT label"); axes[0].set_ylim(-0.05, 1.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title, fontsize=10)

    # Distance to A spec
    axes[1].plot(t, dist_A, color="steelblue", lw=1.2, label="d(z_t, z_A)")
    axes[1].axhline(tau_A, color="red", ls="--", lw=1.5, label=f"τ_A = {tau_A:.3f}")
    axes[1].set_ylabel("Dist to z_A")
    axes[1].legend(loc="upper right", fontsize=8)

    # Distance to B spec
    axes[2].plot(t, dist_B, color="green", lw=1.2, label="d(z_t, z_B)")
    axes[2].axhline(tau_B, color="red", ls="--", lw=1.5, label=f"τ_B = {tau_B:.3f}")
    axes[2].set_ylabel("Dist to z_B")
    axes[2].set_xlabel("Frame index")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


# ─── Dataset-specific evaluation ───────────────────────────────────────────────

# ── iamlab ────────────────────────────────────────────────────────────────────

IAMLAB_REPO = "lerobot/iamlab_cmu_pickup_insert"
IAMLAB_VIDEO_KEY = "observation.images.image"
IAMLAB_DONE_WINDOW = 10  # last N frames = positive GT for Reach spec


def evaluate_iamlab(
    encoder: DINOv2Encoder,
    num_episodes: int,
    out_dir: Path,
    cal_frac: float = 0.5,
    seed: int = 42,
    dist_fn=None,
    normalize: bool = False,
):
    """
    iamlab has per-subtask episodes: each episode is EITHER "Pick up X" OR
    "Insert X".  Because objects differ (green block, pink flower, …), we must
    group by the specific task string before building spec latents; otherwise
    the centroid averages visually-dissimilar states.

    Strategy:
      • Find the task that has both a pick AND an insert variant (green block).
      • Build z_pick from the pick-green-block done-window, z_insert from the
        insert-green-block done-window.
      • Evaluate discrimination on same-object test episodes AND on other-object
        pick episodes (cross-object discrimination).
    """
    print("\n=== Evaluating iamlab_cmu_pickup_insert (per-object grouping) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, data_df, ep_df = load_dataset_meta(IAMLAB_REPO)

    # Identify all unique tasks and their episode indices
    task_to_eps: Dict[str, List[int]] = {}
    for _, row in ep_df.iterrows():
        task_str = row["tasks"][0] if row["tasks"] else "unknown"
        task_to_eps.setdefault(task_str, []).append(int(row["episode_index"]))

    print("  Available tasks:")
    for k, v in sorted(task_to_eps.items(), key=lambda x: -len(x[1])):
        print(f"    '{k}': {len(v)} episodes")

    # Find a pick-object pair with enough episodes on both sides
    # iamlab has "Pick up green block." and "Insert greeen block." (note typo)
    pick_tasks   = {k: v for k, v in task_to_eps.items() if "Pick" in k or "pick" in k}
    insert_tasks = {k: v for k, v in task_to_eps.items() if "Insert" in k or "insert" in k}

    # For the main analysis: use the most common pick task paired with its insert
    # (pick green block ↔ insert green block)
    main_pick_task   = max(pick_tasks,   key=lambda k: len(pick_tasks[k]))
    # Match insert task containing same object keyword
    pick_object = main_pick_task.lower().replace("pick up ", "").replace(".", "").strip()
    main_insert_task = None
    for k in insert_tasks:
        obj = k.lower().replace("insert ", "").replace(".", "").replace("reen", "reen").strip()
        if any(word in obj for word in pick_object.split()):
            main_insert_task = k; break
    if main_insert_task is None:
        main_insert_task = max(insert_tasks, key=lambda k: len(insert_tasks[k]))

    pick_eps_all   = pick_tasks[main_pick_task]
    insert_eps_all = insert_tasks[main_insert_task]
    print(f"\n  Using:  '{main_pick_task}' ({len(pick_eps_all)} eps)")
    print(f"          '{main_insert_task}' ({len(insert_eps_all)} eps)")

    rng     = np.random.default_rng(seed)
    n_each  = min(num_episodes // 2, len(pick_eps_all), len(insert_eps_all))
    pick_sample   = rng.choice(pick_eps_all,   n_each, replace=False).tolist()
    insert_sample = rng.choice(insert_eps_all, n_each, replace=False).tolist()

    # Also collect a set of OTHER-object pick episodes for cross-object discrimination
    other_pick_eps = []
    for task, eps in pick_tasks.items():
        if task != main_pick_task:
            other_pick_eps.extend(eps)
    n_other = min(20, len(other_pick_eps))
    other_pick_sample = rng.choice(other_pick_eps, n_other, replace=False).tolist() if other_pick_eps else []

    # ── Encode ──
    ep_embeddings: Dict[int, np.ndarray] = {}
    ep_done_masks: Dict[int, np.ndarray] = {}
    ep_kind:       Dict[int, str]        = {}  # "pick" | "insert" | "other_pick"

    if dist_fn is None:
        dist_fn = l2_distances

    def encode_episodes(eps, kind):
        for ep_idx in eps:
            ep_row = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
            frames = load_episode_frames(IAMLAB_REPO, ep_row, IAMLAB_VIDEO_KEY)
            if len(frames) == 0:
                print(f"    ep{ep_idx} ({kind}): no frames, skipping"); continue
            emb  = encoder.encode(frames)
            if normalize:
                emb = l2_normalize(emb)
            T    = len(emb)
            done = np.zeros(T, dtype=bool)
            done[-min(IAMLAB_DONE_WINDOW, T):] = True
            ep_embeddings[ep_idx] = emb
            ep_done_masks[ep_idx] = done
            ep_kind[ep_idx]       = kind
            print(f"    ep{ep_idx} ({kind}): {T} frames")

    print("\n  Encoding pick episodes …")
    encode_episodes(pick_sample, "pick")
    print("  Encoding insert episodes …")
    encode_episodes(insert_sample, "insert")
    if other_pick_sample:
        print("  Encoding other-object pick episodes …")
        encode_episodes(other_pick_sample, "other_pick")

    valid_pick   = [e for e in pick_sample   if e in ep_embeddings]
    valid_insert = [e for e in insert_sample if e in ep_embeddings]
    valid_other  = [e for e in other_pick_sample if e in ep_embeddings]

    # ── Cal / test split ──
    n_cal = max(1, int(len(valid_pick) * cal_frac))
    cal_pick   = valid_pick[:n_cal];   test_pick   = valid_pick[n_cal:]
    cal_insert = valid_insert[:n_cal]; test_insert = valid_insert[n_cal:]
    test_other  = valid_other  # use all for cross-object eval

    # ── Build spec latents ──
    def stack(ep_list, use_done_only=True):
        embs, masks = [], []
        for e in ep_list:
            embs.append(ep_embeddings[e])
            masks.append(ep_done_masks[e])
        return np.concatenate(embs), np.concatenate(masks)

    cal_pk_emb, cal_pk_done   = stack(cal_pick)
    cal_in_emb, cal_in_done   = stack(cal_insert)
    z_pick   = build_spec_latent(cal_pk_emb,  cal_pk_done)
    z_insert = build_spec_latent(cal_in_emb,  cal_in_done)
    if normalize:
        z_pick   = z_pick   / max(np.linalg.norm(z_pick),   1e-8)
        z_insert = z_insert / max(np.linalg.norm(z_insert), 1e-8)
    print(f"\n  z_pick   from {cal_pk_done.sum()} positive frames ({len(cal_pick)} cal eps)")
    print(f"  z_insert from {cal_in_done.sum()} positive frames ({len(cal_insert)} cal eps)")

    # ── Threshold mining ──
    tau_pick_f1,   _, _, _, _ = sweep_f1(dist_fn(cal_pk_emb, z_pick),  cal_pk_done.astype(int))
    tau_insert_f1, _, _, _, _ = sweep_f1(dist_fn(cal_in_emb, z_insert), cal_in_done.astype(int))
    tau_pick_cp   = conformal_tau(dist_fn(cal_pk_emb, z_pick),  cal_pk_done.astype(int))
    tau_insert_cp = conformal_tau(dist_fn(cal_in_emb, z_insert), cal_in_done.astype(int))
    print(f"\n  τ_pick:   F1={tau_pick_f1:.4f}  CP={tau_pick_cp}")
    print(f"  τ_insert: F1={tau_insert_f1:.4f}  CP={tau_insert_cp}")

    # ── Test evaluation ──
    # For π_pick: GT positive = done-window of pick eps; GT negative = insert eps
    # Cross-object: GT negative = other-object pick eps (should also be negative for this spec)
    metrics_out = {}
    for spec_name, z_spec, tau_f1, tau_cp, test_pos, test_neg in [
        ("pick",   z_pick,   tau_pick_f1,   tau_pick_cp,   test_pick,   test_insert),
        ("insert", z_insert, tau_insert_f1, tau_insert_cp, test_insert, test_pick),
    ]:
        all_dist, all_gt = [], []
        for e in test_pos + test_neg:
            d  = dist_fn(ep_embeddings[e], z_spec)
            gt = ep_done_masks[e].astype(int) if e in test_pos else np.zeros(len(d), dtype=int)
            all_dist.append(d); all_gt.append(gt)
        all_dist = np.concatenate(all_dist)
        all_gt   = np.concatenate(all_gt)
        m_f1 = predicate_metrics(all_dist, all_gt, tau_f1)
        m_cp = predicate_metrics(all_dist, all_gt, tau_cp) if tau_cp else None
        print(f"\n  π_{spec_name} τ_F1: F1={m_f1['f1']:.3f}  P={m_f1['precision']:.3f}  R={m_f1['recall']:.3f}")
        if m_cp:
            print(f"  π_{spec_name} τ_CP: F1={m_cp['f1']:.3f}  P={m_cp['precision']:.3f}  R={m_cp['recall']:.3f}")

        # Cross-object discrimination (pick spec vs other-object pick episodes)
        cross_obj_m = None
        if spec_name == "pick" and test_other:
            cd = np.concatenate([dist_fn(ep_embeddings[e], z_spec) for e in test_other])
            cg = np.zeros(len(cd), dtype=int)  # none of these are the specific pick task
            cross_obj_m = predicate_metrics(cd, cg, tau_f1)
            cross_fpr = cross_obj_m["fp"] / (cross_obj_m["fp"] + cross_obj_m["tn"] + 1e-9)
            print(f"  π_pick cross-object FPR (other-pick eps): {cross_fpr:.3f}")

        metrics_out[spec_name] = {
            "tau_F1": tau_f1, "tau_CP": tau_cp,
            "test_tau_F1": m_f1, "test_tau_CP": m_cp,
            "cross_object": cross_obj_m,
        }

    # ── Plots ──
    print("\n  Generating plots …")
    for spec_name, z_spec, tau_f1, tau_cp, test_pos, test_neg in [
        ("pick",   z_pick,   tau_pick_f1,   tau_pick_cp,   test_pick,   test_insert),
        ("insert", z_insert, tau_insert_f1, tau_insert_cp, test_insert, test_pick),
    ]:
        all_dist, all_gt = [], []
        for e in test_pos + test_neg:
            d  = dist_fn(ep_embeddings[e], z_spec)
            gt = ep_done_masks[e].astype(int) if e in test_pos else np.zeros(len(d), dtype=int)
            all_dist.append(d); all_gt.append(gt)
        all_dist = np.concatenate(all_dist); all_gt = np.concatenate(all_gt)
        plot_f1_curve(all_dist, all_gt, tau_f1, tau_cp,
                      f"iamlab ({main_pick_task[:20]}) — π_{spec_name} F1 vs τ",
                      out_dir / f"f1_curve_{spec_name}.png")

    timelines_dir = out_dir / "timelines"
    for spec_name, z_spec, tau_f1, tau_cp, test_pos in [
        ("pick",   z_pick,   tau_pick_f1,   tau_pick_cp,   test_pick),
        ("insert", z_insert, tau_insert_f1, tau_insert_cp, test_insert),
    ]:
        for ep_idx in test_pos[:5]:
            d  = dist_fn(ep_embeddings[ep_idx], z_spec)
            gt = ep_done_masks[ep_idx].astype(int)
            plot_timeline(d, gt, tau_f1, tau_cp,
                          f"iamlab ep{ep_idx} — π_{spec_name}",
                          timelines_dir / f"{spec_name}_ep{ep_idx}.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)
    print(f"\n  Saved metrics → {out_dir/'metrics.json'}")
    return metrics_out


# ── FMB ───────────────────────────────────────────────────────────────────────

FMB_REPO = "lerobot/fmb"
FMB_VIDEO_KEY = "observation.images.image_side_1"

# FMB task_index labels: pick and insert per object colour
FMB_PICK_TASK_IDS   = {0, 4, 6, 10, 16, 20}   # Pick up X
FMB_INSERT_TASK_IDS = {3, 5, 9, 11, 17, 21}   # Insert X
FMB_DONE_WINDOW = 5


def evaluate_fmb(
    encoder: DINOv2Encoder,
    num_episodes: int,
    out_dir: Path,
    cal_frac: float = 0.5,
    seed: int = 42,
    dist_fn=None,
    normalize: bool = False,
):
    if dist_fn is None:
        dist_fn = l2_distances
    print("\n=== Evaluating FMB (Functional Manipulation Benchmark) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, data_df, ep_df = load_dataset_meta(FMB_REPO)

    # Filter episodes that contain both pick and insert phases
    def has_pick_and_insert(task_list):
        tids = set(task_list) if isinstance(task_list, (list, set)) else set()
        # The 'tasks' column in ep_df for FMB is a list of task strings, not ints
        # But task_index in data_df is per-frame; check via stats
        return True  # we'll filter per-frame below

    # Only keep episodes with both pick AND insert task_index in data
    valid_ep_ids = []
    for _, row in ep_df.iterrows():
        tid_min = row.get("stats/task_index/min", [None])
        tid_max = row.get("stats/task_index/max", [None])
        if isinstance(tid_min, list): tid_min = tid_min[0]
        if isinstance(tid_max, list): tid_max = tid_max[0]
        if tid_min is None or tid_max is None:
            continue
        ep_task_ids = set(range(int(tid_min), int(tid_max) + 1))
        if ep_task_ids & FMB_PICK_TASK_IDS and ep_task_ids & FMB_INSERT_TASK_IDS:
            valid_ep_ids.append(int(row["episode_index"]))

    print(f"  Episodes with pick+insert phases: {len(valid_ep_ids)}")
    rng = np.random.default_rng(seed)
    n_use = min(num_episodes, len(valid_ep_ids))
    sampled_ids = rng.choice(valid_ep_ids, n_use, replace=False).tolist()

    # ── Encode all episodes ──
    ep_embeddings: Dict[int, np.ndarray] = {}
    ep_pick_mask:  Dict[int, np.ndarray] = {}
    ep_insert_mask: Dict[int, np.ndarray] = {}
    ep_done_mask:  Dict[int, np.ndarray] = {}

    for ep_idx in sampled_ids:
        print(f"  Encoding FMB episode {ep_idx} …", end=" ")
        ep_row = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        frames = load_episode_frames(FMB_REPO, ep_row, FMB_VIDEO_KEY)
        if len(frames) == 0:
            print("no frames, skipping"); continue

        # Per-frame task_index from data parquet
        ep_data = data_df[data_df["episode_index"] == ep_idx].sort_values("frame_index")
        T_data  = len(ep_data)
        T_vid   = len(frames)
        T       = min(T_data, T_vid)

        tid_arr  = ep_data["task_index"].values[:T]
        done_arr = ep_data["next.done"].values[:T]

        # phase_done_mask: True only for last FMB_DONE_WINDOW frames of each
        # pick/insert phase segment — the completion moment, not the whole phase
        pick_mask   = phase_done_mask(tid_arr, FMB_PICK_TASK_IDS,   window=FMB_DONE_WINDOW)
        insert_mask = phase_done_mask(tid_arr, FMB_INSERT_TASK_IDS, window=FMB_DONE_WINDOW)
        done_mask   = done_arr.astype(bool)

        emb = encoder.encode(frames[:T])
        if normalize:
            emb = l2_normalize(emb)
        ep_embeddings[ep_idx]  = emb
        ep_pick_mask[ep_idx]   = pick_mask
        ep_insert_mask[ep_idx] = insert_mask
        ep_done_mask[ep_idx]   = done_mask
        print(f"{T} frames | pick_done={pick_mask.sum()} insert_done={insert_mask.sum()}")

    valid_ids = list(ep_embeddings.keys())

    # ── Cal / test split ──
    n_cal = max(1, int(len(valid_ids) * cal_frac))
    cal_ids  = valid_ids[:n_cal]
    test_ids = valid_ids[n_cal:]
    if len(test_ids) == 0:
        test_ids = cal_ids  # fallback: evaluate on cal

    # ── Build spec latents from calibration ──
    def cal_spec(ep_list, mask_dict, window=FMB_DONE_WINDOW):
        zs = []
        for e in ep_list:
            m = mask_dict[e]
            if m.any():
                idx = np.where(m)[0][-min(window, m.sum()):]
                zs.append(ep_embeddings[e][idx].mean(axis=0))
        return np.stack(zs).mean(axis=0)

    z_pick   = cal_spec(cal_ids, ep_pick_mask)
    z_insert = cal_spec(cal_ids, ep_insert_mask)
    if normalize:
        z_pick   = z_pick   / max(np.linalg.norm(z_pick),   1e-8)
        z_insert = z_insert / max(np.linalg.norm(z_insert), 1e-8)
    print(f"\n  z_pick built from {n_cal} cal episodes")
    print(f"  z_insert built from {n_cal} cal episodes")

    # ── Calibration threshold mining ──
    def pool_cal(ep_list, z_spec, mask_dict):
        dists, labels = [], []
        for e in ep_list:
            d = dist_fn(ep_embeddings[e], z_spec)
            dists.append(d); labels.append(mask_dict[e].astype(int))
        return np.concatenate(dists), np.concatenate(labels)

    cal_d_pick, cal_l_pick     = pool_cal(cal_ids, z_pick,   ep_pick_mask)
    cal_d_insert, cal_l_insert = pool_cal(cal_ids, z_insert, ep_insert_mask)

    tau_pick_f1,   f1_pk, _, _, _ = sweep_f1(cal_d_pick,   cal_l_pick)
    tau_insert_f1, f1_in, _, _, _ = sweep_f1(cal_d_insert, cal_l_insert)
    tau_pick_cp   = conformal_tau(cal_d_pick,   cal_l_pick)
    tau_insert_cp = conformal_tau(cal_d_insert, cal_l_insert)

    print(f"\n  τ_pick_F1={tau_pick_f1:.4f}  τ_pick_CP={tau_pick_cp}")
    print(f"  τ_insert_F1={tau_insert_f1:.4f}  τ_insert_CP={tau_insert_cp}")

    # ── Test evaluation: frame-level predicate agreement ──
    metrics_out = {}
    for spec_name, z_spec, tau_f1, tau_cp, mask_dict in [
        ("pick",   z_pick,   tau_pick_f1,   tau_pick_cp,   ep_pick_mask),
        ("insert", z_insert, tau_insert_f1, tau_insert_cp, ep_insert_mask),
    ]:
        all_d, all_l = pool_cal(test_ids, z_spec, mask_dict)
        m_f1 = predicate_metrics(all_d, all_l, tau_f1)
        m_cp = predicate_metrics(all_d, all_l, tau_cp) if tau_cp else None
        metrics_out[spec_name] = {
            "tau_F1": tau_f1, "tau_CP": tau_cp,
            "test_tau_F1": m_f1, "test_tau_CP": m_cp,
        }
        print(f"\n  π_{spec_name}  τ_F1: F1={m_f1['f1']:.3f} P={m_f1['precision']:.3f} R={m_f1['recall']:.3f}")
        if m_cp:
            print(f"  π_{spec_name}  τ_CP: F1={m_cp['f1']:.3f} P={m_cp['precision']:.3f} R={m_cp['recall']:.3f}")

    # ── Sequential spec: ∃ t1 < t2 s.t. π_pick(t1) ∧ π_insert(t2) ──
    seq_results = []
    for ep_idx in test_ids:
        da = dist_fn(ep_embeddings[ep_idx], z_pick)
        db = dist_fn(ep_embeddings[ep_idx], z_insert)
        pred_A = da < tau_pick_cp   # high-recall threshold for sequential detection
        pred_B = db < tau_insert_cp
        gt_A   = ep_pick_mask[ep_idx]
        gt_B   = ep_insert_mask[ep_idx]
        # sequential GT: any pick frame precedes any insert frame (by definition in FMB)
        gt_seq = gt_A.any() and gt_B.any() and (np.where(gt_A)[0][0] < np.where(gt_B)[0][-1])
        # sequential pred: first time A fires before last time B fires
        A_times = np.where(pred_A)[0]
        B_times = np.where(pred_B)[0]
        pred_seq = (len(A_times) > 0 and len(B_times) > 0 and A_times[0] < B_times[-1])
        seq_results.append(dict(ep=ep_idx, gt_seq=gt_seq, pred_seq=pred_seq))

    gt_seq_arr   = np.array([r["gt_seq"]   for r in seq_results])
    pred_seq_arr = np.array([r["pred_seq"] for r in seq_results])
    seq_agree    = float((gt_seq_arr == pred_seq_arr).mean())
    metrics_out["sequential"] = dict(
        agreement=seq_agree,
        n_test=len(seq_results),
        gt_positive=int(gt_seq_arr.sum()),
        pred_positive=int(pred_seq_arr.sum()),
    )
    print(f"\n  Sequential spec agreement: {seq_agree:.3f} ({int(gt_seq_arr.sum())}/{len(seq_results)} GT positive)")

    # ── Plots ──
    print("\n  Generating plots …")
    for spec_name, z_spec, tau_f1, tau_cp, mask_dict in [
        ("pick",   z_pick,   tau_pick_f1,   tau_pick_cp,   ep_pick_mask),
        ("insert", z_insert, tau_insert_f1, tau_insert_cp, ep_insert_mask),
    ]:
        all_d, all_l = pool_cal(test_ids, z_spec, mask_dict)
        plot_f1_curve(all_d, all_l, tau_f1, tau_cp,
                      f"FMB — π_{spec_name} F1 vs τ",
                      out_dir / f"f1_curve_{spec_name}.png")

    timelines_dir = out_dir / "timelines"
    for ep_idx in test_ids[:5]:
        da = dist_fn(ep_embeddings[ep_idx], z_pick)
        db = dist_fn(ep_embeddings[ep_idx], z_insert)
        plot_sequential_timeline(
            da, db,
            ep_pick_mask[ep_idx],   ep_insert_mask[ep_idx],
            tau_pick_f1, tau_insert_f1,
            f"FMB ep{ep_idx} — Sequential Pick→Insert",
            timelines_dir / f"seq_ep{ep_idx}.png",
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)
    print(f"\n  Saved metrics → {out_dir/'metrics.json'}")
    return metrics_out


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      choices=["iamlab", "fmb", "both"], default="both")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--out-dir",      type=str, default="etl_results/lerobot")
    parser.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cal-frac",     type=float, default=0.5)
    parser.add_argument("--encoder",  choices=["dino", "r3m", "robometer"], default="dino",
                        help="Vision encoder: dino=DINOv2 (768-D), r3m=R3M ResNet50 (2048-D), robometer=Robometer-4B (2560-D)")
    parser.add_argument("--distance", choices=["l2", "cosine"], default="cosine",
                        help="Distance metric. cosine = L2-normalize then L2 (= cosine up to scale)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.encoder == "r3m":
        encoder = R3MEncoder(device=args.device)
    elif args.encoder == "robometer":
        sys.path.insert(0, str(Path(__file__).parent))
        from robometer_encoder import RobometerEncoder as _RobometerEncoder
        _rbm = _RobometerEncoder(device=args.device)
        # Wrap to match the .encode(frames) interface
        class _RobometerEncoderWrapper:
            def encode(self_, frames, batch_size=8):
                return _rbm.encode_frames(frames, task="robot manipulation",
                                          chunk_size=8, stride=4, batch_size=4)
        encoder = _RobometerEncoderWrapper()
    else:
        encoder = DINOv2Encoder(device=args.device)

    use_cosine = args.distance == "cosine"
    dist_fn    = cosine_distances if use_cosine else l2_distances

    all_results = {}
    if args.dataset in ("iamlab", "both"):
        all_results["iamlab"] = evaluate_iamlab(
            encoder, args.num_episodes, out_dir / "iamlab", cal_frac=args.cal_frac,
            dist_fn=dist_fn, normalize=use_cosine,
        )
    if args.dataset in ("fmb", "both"):
        all_results["fmb"] = evaluate_fmb(
            encoder, args.num_episodes, out_dir / "fmb", cal_frac=args.cal_frac,
            dist_fn=dist_fn, normalize=use_cosine,
        )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        for spec, m in res.items():
            if "test_tau_F1" in m and m["test_tau_F1"]:
                f1 = m["test_tau_F1"]["f1"]
                pr = m["test_tau_F1"]["precision"]
                rc = m["test_tau_F1"]["recall"]
                print(f"  π_{spec} (τ_F1): F1={f1:.3f}  P={pr:.3f}  R={rc:.3f}")
            elif "agreement" in m:
                print(f"  sequential: agreement={m['agreement']:.3f}")


if __name__ == "__main__":
    main()
