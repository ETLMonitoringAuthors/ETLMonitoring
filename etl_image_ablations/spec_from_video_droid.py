"""
spec_from_video_droid.py
------------------------
Video-as-specification for DROID in-the-wild episodes.

Given ONE reference DROID episode (a video clip), this tool extracts TWO
kinds of temporal spec from it:

  WHERE spec  (what is reached):
    z_phase_k = mean embedding of reference phase k.
    Predicate: d(z_t, z_phase_k) < τ_k

  HOW spec  (how motion unfolds within each phase):
    traj_phase_k = reference phase trajectory (subsampled to N_PTS points).
    Predicate: sliding-window DTW distance between test trajectory and
               traj_phase_k is below τ_dtw_k.

Combined sequential spec:
  F( WHERE_1 ∧ HOW_1 ∧ F( WHERE_2 ∧ HOW_2 ∧ ... ) )

Segmentation: purely from the video.
  Δcos(t) = 1 - cos(z_t, z_{t+1}), smoothed and peak-detected.
  No gripper signal is used as input; gripper is only used for GT comparison.

Task matching: object-noun overlap (bottle, towel, cup, …) is prioritised
over generic word overlap, so test episodes share similar *objects* rather
than just similar sentence structure.

Usage
-----
  cd /path/to/repo
  TMPDIR=/tmp HF_HOME=~/.cache/huggingface \\
  python \\
      -m etl_image_ablations.spec_from_video_droid \\
      --ref-ep 13 --encoder svd_vae --cam wrist \\
      --n-phases 2 --out-dir etl_results/spec_from_video_svd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from huggingface_hub import hf_hub_download

REPO_ID = "lerobot/droid_100"
CAM_KEYS = {
    "wrist":     "observation.images.wrist_image_left",
    "exterior1": "observation.images.exterior_image_1_left",
}

# Gripper segmentation (GT comparison only)
CLOSE_THRESH     = 0.45
OPEN_THRESH      = 0.20
MIN_CLOSE_FRAMES = 5

# HOW-spec DTW trajectory subsampling length
DTW_N_PTS = 24

# Object nouns and action verbs used for task similarity scoring
OBJECT_NOUNS = {
    "bottle", "bottles", "cup", "cups", "mug", "mugs",
    "towel", "towels", "cloth", "cloths", "tissue",
    "paper", "papers", "sheet", "sheets",
    "pen", "pens", "pencil", "pencils", "marker",
    "lid", "lids", "cap", "caps",
    "pot", "pots", "pan", "pans",
    "bowl", "bowls", "plate", "plates",
    "basket", "baskets", "box", "boxes", "container",
    "bag", "bags", "tray", "trays",
    "book", "books", "notebook",
    "block", "blocks", "cube", "cubes", "object", "objects",
    "toy", "toys", "ball", "balls",
    "can", "cans", "jar", "jars",
    "fork", "forks", "spoon", "spoons", "knife", "utensil",
    "phone", "remote", "key", "keys",
    "sponge", "brush",
    "white", "black", "red", "blue", "green", "yellow",  # colour-objects
}
ACTION_VERBS = {
    "pick", "pickup", "grab", "grasp", "take", "lift",
    "put", "place", "set", "drop", "lay", "move", "push",
    "fold", "unfold", "open", "close", "stack", "sort",
}


# ── Encoders ─────────────────────────────────────────────────────────────────

class DINOv2Encoder:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, device="cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                                    pretrained=True, verbose=False).to(device).eval()
        self.mean = self.MEAN.to(device)
        self.std  = self.STD.to(device)

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [(TF.resize(torch.from_numpy(f).permute(2, 0, 1).float().div(255.).to(self.device),
                                [224, 224], antialias=True) - self.mean) / self.std
                     for f in frames[i:i + batch_size]]
            out.append(self.model(torch.stack(batch)).cpu().numpy())
        return np.concatenate(out)


class SVDVAEEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        from diffusers.models import AutoencoderKLTemporalDecoder
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae",
            torch_dtype=torch.float16, cache_dir="/tmp",
        ).to(device).eval()
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [TF.resize(torch.from_numpy(f).permute(2, 0, 1).float().div(255.),
                               [192, 320], antialias=True) * 2.0 - 1.0
                     for f in frames[i:i + batch_size]]
            x = torch.stack(batch).to(self.device, dtype=torch.float16)
            z = self.vae.encode(x).latent_dist.sample().float().cpu()
            out.append(z.reshape(len(batch), -1).numpy())
        return np.concatenate(out)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_meta() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_p = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet", repo_type="dataset")
    ep_p   = hf_hub_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    return pd.read_parquet(data_p), pd.read_parquet(ep_p)


def decode_mp4(path: str, t0: float, t1: float) -> List[np.ndarray]:
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


def load_frames(ep_row: pd.Series, cam_key: str) -> List[np.ndarray]:
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    fidx  = int(ep_row[f"videos/{cam_key}/file_index"])
    t0    = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    t1    = float(ep_row[f"videos/{cam_key}/to_timestamp"])
    path  = hf_hub_download(REPO_ID,
        f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4", repo_type="dataset")
    return decode_mp4(path, t0, t1)


def get_state(data_df: pd.DataFrame, ep_idx: int) -> np.ndarray:
    rows = data_df[data_df["episode_index"] == ep_idx]
    return np.stack(rows.reset_index(drop=True)["observation.state"].values)


# ── Task similarity: object + action weighted Jaccard ────────────────────────

def _tokenise(task: str) -> set:
    import re
    return set(re.findall(r"[a-z]+", task.lower()))


def task_similarity(task_a: str, task_b: str) -> float:
    """
    Weighted Jaccard: shared object nouns count 3×, shared action verbs 2×,
    other shared words 1×.  Returns value in [0, 1].
    """
    w_a, w_b = _tokenise(task_a), _tokenise(task_b)
    if not w_a or not w_b:
        return 0.0
    score_num = score_den = 0.0
    for w in w_a | w_b:
        weight = 3.0 if w in OBJECT_NOUNS else (2.0 if w in ACTION_VERBS else 1.0)
        if w in w_a and w in w_b:
            score_num += weight
        score_den += weight
    return score_num / (score_den + 1e-9)


def shared_objects(task_a: str, task_b: str) -> List[str]:
    wa, wb = _tokenise(task_a), _tokenise(task_b)
    return sorted((wa & wb) & OBJECT_NOUNS)


# ── Latent-space change-point segmentation ────────────────────────────────────

def consecutive_cosine_dist(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
    en = embs / norms
    return 1.0 - (en[:-1] * en[1:]).sum(axis=1)


def segment_from_latents(
    embs: np.ndarray,
    n_phases: Optional[int] = None,
    sigma: float = 3.0,
    min_prominence: float = 0.002,
    min_phase_frames: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    T = len(embs)
    delta = consecutive_cosine_dist(embs)
    delta_smooth = gaussian_filter1d(delta.astype(np.float64), sigma=sigma)

    # When n_phases is explicit, use a lower distance bound so we find enough peaks
    min_dist = max(4, min_phase_frames // 2) if n_phases else min_phase_frames
    peaks, props = find_peaks(delta_smooth, prominence=min_prominence, distance=min_dist)

    if n_phases is not None and n_phases > 1:
        n_boundaries = n_phases - 1
        if len(peaks) >= n_boundaries:
            order = np.argsort(props["prominences"])[::-1][:n_boundaries]
            peaks = np.sort(peaks[order])
        else:
            # Not enough peaks: add evenly-spaced fallback boundaries
            extra = n_boundaries - len(peaks)
            fallback = np.linspace(0, T - 1, extra + 2, dtype=int)[1:-1]
            peaks = np.sort(np.concatenate([peaks, fallback]))[:n_boundaries]

        # When n_phases is explicit: do NOT filter by min_phase_frames so that
        # exactly the requested number of phases is returned.
        boundaries = np.concatenate([[0], peaks + 1, [T]])
        phases: List[Tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            phases.append((int(boundaries[i]), int(boundaries[i + 1])))
    else:
        boundaries = np.concatenate([[0], peaks + 1, [T]])
        phases = []
        for i in range(len(boundaries) - 1):
            s, e = int(boundaries[i]), int(boundaries[i + 1])
            if e - s >= min_phase_frames:
                phases.append((s, e))
        if not phases:
            phases = [(0, T)]
    return delta, delta_smooth, phases


# ── Gripper segmentation (GT comparison) ─────────────────────────────────────

def segment_gripper(gripper: np.ndarray) -> List[Tuple[int, int]]:
    T = len(gripper); in_close = False; windows = []
    start = 0
    for t in range(T):
        if not in_close and gripper[t] > CLOSE_THRESH:
            in_close = True; start = t
        elif in_close and gripper[t] < OPEN_THRESH:
            in_close = False
            if t - start >= MIN_CLOSE_FRAMES:
                windows.append((start, t))
    if in_close and T - start >= MIN_CLOSE_FRAMES:
        windows.append((start, T))
    return windows


# ── WHERE spec helpers ────────────────────────────────────────────────────────

def cosine_dist_vec(embs: np.ndarray, z: np.ndarray) -> np.ndarray:
    en = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(np.linalg.norm(z), 1e-8)
    return 1.0 - (en @ zn)


def mine_f1_tau(dists: np.ndarray, gt: np.ndarray, n: int = 300) -> Tuple[float, float]:
    if not gt.any():
        return float(np.median(dists)), 0.0
    taus = np.linspace(dists.min(), dists.max(), n)
    best_f1, best_tau = 0.0, float(np.median(taus))
    for tau in taus:
        pred = dists < tau
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return float(best_tau), float(best_f1)


def eval_pred(dists: np.ndarray, gt: np.ndarray, tau: float) -> dict:
    pred = dists < tau
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"f1": float(f1), "precision": float(p), "recall": float(r),
            "agreement": float((pred == gt).mean())}


def adaptive_tau(dist: np.ndarray, quantile: float = 0.25) -> float:
    return float(np.quantile(dist, quantile))


def argmin_ordering(dist_list: List[np.ndarray]) -> Tuple[List[int], bool]:
    nearest = [int(np.argmin(d)) for d in dist_list]
    ok = all(nearest[k] < nearest[k + 1] for k in range(len(nearest) - 1))
    return nearest, ok


# ── HOW spec: trajectory DTW ──────────────────────────────────────────────────

def subsample_traj(embs: np.ndarray, n_pts: int) -> np.ndarray:
    """Uniformly subsample a [T, D] trajectory to [n_pts, D]."""
    T = len(embs)
    if T <= n_pts:
        idx = np.round(np.linspace(0, T - 1, n_pts)).astype(int)
    else:
        idx = np.round(np.linspace(0, T - 1, n_pts)).astype(int)
    return embs[idx]  # [n_pts, D]


def dtw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    DTW distance between two trajectory arrays [Na, D] and [Nb, D]
    using cosine distance as the local metric.
    Returns the normalised accumulated cost.
    """
    Na, Nb = len(a), len(b)
    an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
    cost = 1.0 - (an @ bn.T)  # [Na, Nb]

    dp = np.full((Na, Nb), np.inf)
    dp[0, 0] = cost[0, 0]
    for i in range(1, Na):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, Nb):
        dp[0, j] = dp[0, j - 1] + cost[0, j]
    for i in range(1, Na):
        for j in range(1, Nb):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1]) / (Na + Nb)


def sliding_dtw(
    embs: np.ndarray,
    ref_traj: np.ndarray,
    window_frames: int,
    step: int = 4,
) -> np.ndarray:
    """
    Slide a window of `window_frames` over `embs` and compute DTW distance
    to `ref_traj` (already subsampled to DTW_N_PTS).
    Returns an array of length T with DTW distance at each center frame;
    frames without a full window are filled with the nearest valid value.
    """
    T = len(embs)
    half = window_frames // 2
    scores = np.full(T, np.nan)
    centers = range(half, T - half, step)
    for c in centers:
        window = embs[c - half: c + half]
        sub = subsample_traj(window, DTW_N_PTS)
        scores[c] = dtw_cosine(sub, ref_traj)
    # Fill NaN by nearest-neighbour interpolation
    valid = np.where(~np.isnan(scores))[0]
    if len(valid) == 0:
        return np.zeros(T)
    scores[:valid[0]] = scores[valid[0]]
    scores[valid[-1]:] = scores[valid[-1]]
    xp = valid; fp = scores[valid]
    all_x = np.arange(T)
    scores = np.interp(all_x, xp, fp)
    return scores


def etl_formula_str(n_phases: int, how: bool = False) -> str:
    def nested(k: int) -> str:
        where = f"near_p{k+1}"
        how_p = f"matches_traj_p{k+1}"
        atom = f"({where} ∧ {how_p})" if how else where
        if k == n_phases - 1:
            return f"F({atom})"
        return f"F({atom} ∧ {nested(k+1)})"
    return nested(0)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_segmentation(
    delta: np.ndarray,
    delta_smooth: np.ndarray,
    phases_latent: List[Tuple[int, int]],
    phases_gripper: Optional[List[Tuple[int, int]]],
    task: str,
    ep_idx: int,
    out_path: Path,
):
    COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#F4A582", "#984EA3"]
    T1 = len(delta)
    t = np.arange(T1)
    n_rows = 2 if phases_gripper else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.5 * n_rows + 0.5), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(t, delta, color="lightsteelblue", lw=0.8, alpha=0.6, label="raw Δcos")
    ax.plot(t, delta_smooth, color="steelblue", lw=1.6, label="smoothed Δcos")
    for k, (s, e) in enumerate(phases_latent):
        col = COLORS[k % len(COLORS)]
        ax.axvspan(s, e, alpha=0.12, color=col)
        ax.axvline(s, color=col, lw=1.0, ls="--", alpha=0.8, label=f"phase {k+1}")
    ax.set_ylabel("Δcos(z_t, z_{t+1})", fontsize=8)
    ax.set_title(f"ep{ep_idx} — video-spec segmentation  |  {task[:70]}", fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc="upper right"); ax.grid(alpha=0.25)

    if phases_gripper:
        ax2 = axes[1]
        ax2.plot(t, delta_smooth, color="steelblue", lw=1.0, alpha=0.4)
        for k, (s, e) in enumerate(phases_gripper):
            col = COLORS[k % len(COLORS)]
            ax2.axvspan(s, e, alpha=0.25, color=col, label=f"gripper {k+1}")
            ax2.axvline(s, color=col, lw=1.0, ls=":")
        ax2.set_ylabel("Δcos (gripper GT)", fontsize=8)
        ax2.legend(fontsize=7, ncol=5, loc="upper right"); ax2.grid(alpha=0.25)

    axes[-1].set_xlabel("Frame", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Segmentation → {out_path}")


def plot_monitoring(
    ep_idx: int,
    task: str,
    where_dists: List[np.ndarray],
    where_taus: List[float],
    how_dists: List[np.ndarray],
    how_taus: List[float],
    phases_gripper: Optional[List[Tuple[int, int]]],
    out_path: Path,
):
    K = len(where_dists)
    T = len(where_dists[0])
    t = np.arange(T)
    COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#F4A582", "#984EA3"]
    n_rows = K * 2 + 1  # per-phase WHERE + HOW + combined ribbon

    heights = ([2.0, 1.2] * K) + [0.6]
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, sum(heights) + 0.5),
                             gridspec_kw={"height_ratios": heights}, sharex=True)

    for k in range(K):
        col = COLORS[k % len(COLORS)]
        ax_w = axes[k * 2]
        ax_h = axes[k * 2 + 1]

        # WHERE
        d_w = where_dists[k]; tau_w = where_taus[k]
        pred_w = d_w < tau_w
        gt_mask = None
        if phases_gripper and k < len(phases_gripper):
            ws, we = phases_gripper[k]
            gt_mask = np.zeros(T, bool); gt_mask[ws:we] = True
        ax_w.plot(t, d_w, color=col, lw=1.3, label=f"WHERE d(z_t, z_ph{k+1})")
        ax_w.axhline(tau_w, color="black", ls="--", lw=1.0, label=f"τ={tau_w:.3f}")
        ax_w.fill_between(t, d_w.min(), d_w, where=pred_w, alpha=0.18, color=col)
        if gt_mask is not None:
            ax_w.fill_between(t, d_w.min(), d_w.max() * 1.02, where=gt_mask,
                              alpha=0.12, color="gray", label="gripper GT")
            r = eval_pred(d_w, gt_mask, tau_w)
            ax_w.set_title(f"ph{k+1} WHERE  F1={r['f1']:.3f}  "
                           f"agree={r['agreement']:.3f}", fontsize=8)
        ax_w.set_ylabel(f"ph{k+1} Δcos", fontsize=7)
        ax_w.legend(fontsize=6, ncol=3, loc="upper right"); ax_w.grid(alpha=0.2)

        # HOW
        d_h = how_dists[k]; tau_h = how_taus[k]
        pred_h = d_h < tau_h
        ax_h.plot(t, d_h, color=col, lw=1.1, ls="--",
                  label=f"HOW DTW(window, traj_ph{k+1})")
        ax_h.axhline(tau_h, color="black", ls=":", lw=1.0, label=f"τ_dtw={tau_h:.4f}")
        ax_h.fill_between(t, d_h.min(), d_h, where=pred_h, alpha=0.20, color=col)
        combined_pred = pred_w & pred_h
        ax_h.fill_between(t, d_h.min(), d_h.max() * 1.02, where=combined_pred,
                          alpha=0.25, color="gold", label="WHERE ∧ HOW")
        ax_h.set_ylabel(f"ph{k+1} DTW", fontsize=7)
        ax_h.legend(fontsize=6, ncol=3, loc="upper right"); ax_h.grid(alpha=0.2)

    # Combined ribbon
    ax_r = axes[-1]
    for k in range(K):
        col = COLORS[k % len(COLORS)]
        combined = (where_dists[k] < where_taus[k]) & (how_dists[k] < how_taus[k])
        for tt in range(T):
            if combined[tt]:
                ax_r.axvspan(tt, tt + 1, ymin=k / K, ymax=(k + 1) / K, color=col, alpha=0.85)
    ax_r.set_yticks([(k + 0.5) / K for k in range(K)])
    ax_r.set_yticklabels([f"ph{k+1}" for k in range(K)], fontsize=7)
    ax_r.set_xlabel("Frame", fontsize=8); ax_r.set_ylabel("WHERE∧HOW", fontsize=7)

    fig.suptitle(f"ep{ep_idx}  WHERE+HOW monitoring  '{task[:70]}'", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Monitoring → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_key = CAM_KEYS[args.cam]

    print("Loading DROID metadata …")
    data_df, ep_df = load_meta()
    ep_df = ep_df.copy()
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else "")

    # ── Reference episode ──────────────────────────────────────────────────────
    ref_row = ep_df[ep_df["episode_index"] == args.ref_ep]
    if ref_row.empty:
        raise ValueError(f"Episode {args.ref_ep} not in metadata.")
    ref_row = ref_row.iloc[0]
    ref_task = ref_row["task_text"]
    print(f"\n{'='*60}")
    print(f"Reference ep{args.ref_ep}: '{ref_task}'")

    ref_frames = load_frames(ref_row, cam_key)
    ref_state  = get_state(data_df, args.ref_ep)
    T_ref = min(len(ref_frames), len(ref_state))
    ref_frames, ref_state = ref_frames[:T_ref], ref_state[:T_ref]
    ref_gripper = ref_state[:, 6]
    print(f"  T={T_ref} frames")

    # Build encoder
    if args.encoder == "dino":
        enc = DINOv2Encoder(args.device)
    else:
        enc = SVDVAEEncoder(args.device)
    print(f"  Encoding with {args.encoder} …")
    ref_embs = enc.encode(ref_frames)
    print(f"  Embedding shape: {ref_embs.shape}")
    np.savez_compressed(out_dir / f"embs_ep{args.ref_ep:04d}.npz",
                        embs=ref_embs)

    # ── Auto-segment from video only ──────────────────────────────────────────
    n_phases_arg = args.n_phases if args.n_phases and args.n_phases > 0 else None
    delta, delta_smooth, phases_latent = segment_from_latents(
        ref_embs, n_phases=n_phases_arg,
        sigma=args.sigma, min_prominence=args.min_prominence,
        min_phase_frames=args.min_phase_frames,
    )
    phases_gripper = segment_gripper(ref_gripper)

    print(f"\n  Video auto-segmentation ({len(phases_latent)} phase(s)):")
    for k, (s, e) in enumerate(phases_latent):
        print(f"    phase {k+1}: [{s}, {e})  len={e-s}")
    print(f"  Gripper GT ({len(phases_gripper)} phase(s)):")
    for k, (s, e) in enumerate(phases_gripper):
        print(f"    phase {k+1}: [{s}, {e})  len={e-s}")

    formula = etl_formula_str(len(phases_latent), how=True)
    print(f"\n  ETL formula: {formula}")

    plot_segmentation(delta, delta_smooth, phases_latent, phases_gripper,
                      ref_task, args.ref_ep,
                      out_dir / f"seg_ep{args.ref_ep:04d}.pdf")

    # ── Build WHERE + HOW spec latents from reference phases ──────────────────
    # Cap sliding-window at HOW_WINDOW_MAX so very long phases don't make
    # DTW computation O(T * window^2).  Subsampling to DTW_N_PTS normalises
    # duration differences anyway.
    HOW_WINDOW_MAX = 64
    spec_means, spec_trajs, ref_where_taus, ref_how_taus = [], [], [], []
    where_dists_ref, how_dists_ref = [], []
    phase_windows = [min(e - s, HOW_WINDOW_MAX) for s, e in phases_latent]

    for k, (s, e) in enumerate(phases_latent):
        win_k = phase_windows[k]

        # WHERE spec
        z_k = ref_embs[s:e].mean(axis=0)
        spec_means.append(z_k)
        d_where = cosine_dist_vec(ref_embs, z_k)
        gt_k = np.zeros(T_ref, bool); gt_k[s:e] = True
        tau_w, f1_w = mine_f1_tau(d_where, gt_k)
        ref_where_taus.append(tau_w)
        where_dists_ref.append(d_where)

        # HOW spec: subsample MIDDLE win_k frames of the reference phase
        mid = (s + e) // 2
        h_s = max(s, mid - win_k // 2); h_e = min(e, h_s + win_k)
        ref_traj_k = subsample_traj(ref_embs[h_s:h_e], DTW_N_PTS)
        spec_trajs.append(ref_traj_k)
        phase_windows[k] = h_e - h_s  # actual window used

        # HOW distances on reference (self-eval)
        d_how = sliding_dtw(ref_embs, ref_traj_k, window_frames=h_e - h_s,
                            step=max(1, (h_e - h_s) // 8))
        tau_h = mine_f1_tau(d_how, gt_k)[0]
        ref_how_taus.append(tau_h)
        how_dists_ref.append(d_how)

        combined_pred = (d_where < tau_w) & (d_how < tau_h)
        f1_comb = float(2 * (combined_pred & gt_k).sum() /
                        (combined_pred.sum() + gt_k.sum() + 1e-9))
        print(f"\n  Phase {k+1}:")
        print(f"    WHERE  τ={tau_w:.4f}  F1={f1_w:.3f}")
        print(f"    HOW    τ_dtw={tau_h:.5f}  window={h_e-h_s}")
        print(f"    WHERE ∧ HOW  F1={f1_comb:.3f} (self-eval on ref)")

    # Cross-phase separation
    print("\n  Cross-phase cosine distance (WHERE spec latents):")
    sep_vals = {}
    for i in range(len(spec_means)):
        for j in range(i + 1, len(spec_means)):
            zi, zj = spec_means[i], spec_means[j]
            d = 1.0 - float(
                (zi / np.linalg.norm(zi).clip(1e-8)) @ (zj / np.linalg.norm(zj).clip(1e-8))
            )
            sep_vals[f"p{i+1}_p{j+1}"] = round(d, 4)
            print(f"    d(z_ph{i+1}, z_ph{j+1}) = {d:.4f}")

    # Argmin ordering on reference
    nearest_ref, order_ref = argmin_ordering(where_dists_ref)
    print(f"\n  Argmin WHERE ordering on ref: {nearest_ref}  {'✓' if order_ref else '✗'}")

    plot_monitoring(args.ref_ep, ref_task,
                    where_dists_ref, ref_where_taus,
                    how_dists_ref, ref_how_taus,
                    phases_gripper, out_dir / f"monitor_ref_ep{args.ref_ep:04d}.pdf")

    # ── Test episodes: object-weighted similarity ─────────────────────────────
    ep_df["sim"] = ep_df["task_text"].apply(
        lambda t: task_similarity(ref_task, t))
    ep_df["shared_objs"] = ep_df["task_text"].apply(
        lambda t: ",".join(shared_objects(ref_task, t)))

    similar = ep_df[
        (ep_df["episode_index"] != args.ref_ep) &
        (ep_df["sim"] >= args.min_task_sim)
    ].sort_values("sim", ascending=False).head(args.n_test_eps)

    print(f"\n  Test episodes (sim ≥ {args.min_task_sim}, object-weighted):")
    for _, r in similar.iterrows():
        print(f"    ep{int(r['episode_index']):3d}  sim={r['sim']:.3f}  "
              f"objs=[{r['shared_objs']}]  '{r['task_text'][:55]}'")

    test_results = []
    for _, row in similar.iterrows():
        test_ep = int(row["episode_index"])
        test_task = row["task_text"]
        sobjs = row["shared_objs"]
        print(f"\n  {'='*50}")
        print(f"  Testing ep{test_ep}  shared_objs=[{sobjs}]  '{test_task[:60]}'")

        try:
            t_frames = load_frames(row, cam_key)
            t_state  = get_state(data_df, test_ep)
        except Exception as e:
            print(f"    SKIP: {e}"); continue

        T_t = min(len(t_frames), len(t_state))
        t_frames, t_state = t_frames[:T_t], t_state[:T_t]
        t_phases_gripper = segment_gripper(t_state[:, 6])

        print(f"    Encoding {T_t} frames …")
        t_embs = enc.encode(t_frames)
        np.savez_compressed(out_dir / f"embs_ep{test_ep:04d}.npz", embs=t_embs)

        # WHERE distances
        t_where_dists = [cosine_dist_vec(t_embs, z) for z in spec_means]

        # HOW distances: sliding DTW using the capped window from the reference
        t_how_dists = []
        for k, ref_traj_k in enumerate(spec_trajs):
            win = phase_windows[k]
            d_dtw = sliding_dtw(t_embs, ref_traj_k,
                                 window_frames=win,
                                 step=max(1, win // 8))
            t_how_dists.append(d_dtw)

        # Adaptive thresholds (adapt_quantile percentile) for both WHERE and HOW
        # so the predicates fire on the relatively-best frames in each episode.
        adapt_where_taus = [adaptive_tau(d, args.adapt_quantile) for d in t_where_dists]
        adapt_how_taus   = [adaptive_tau(d, args.adapt_quantile) for d in t_how_dists]

        # Argmin WHERE ordering (threshold-free)
        nearest_t,     argmin_where_ok  = argmin_ordering(t_where_dists)
        nearest_how_t, argmin_how_ok    = argmin_ordering(t_how_dists)
        # Combined argmin: both WHERE and HOW argmin-nearest must be in order
        nearest_comb_t = [max(nearest_t[k], nearest_how_t[k]) for k in range(len(nearest_t))]
        argmin_comb_ok = all(nearest_comb_t[k] < nearest_comb_t[k+1]
                             for k in range(len(nearest_comb_t) - 1)) if len(nearest_comb_t) > 1 else True
        print(f"    Argmin WHERE:         {nearest_t}  {'✓' if argmin_where_ok else '✗'}")
        print(f"    Argmin HOW (DTW):     {nearest_how_t}  {'✓' if argmin_how_ok else '✗'}")
        print(f"    Argmin WHERE∧HOW:     {nearest_comb_t}  {'✓' if argmin_comb_ok else '✗'}")

        # Adaptive WHERE+HOW combined ordering
        where_preds = [d < tau for d, tau in zip(t_where_dists, adapt_where_taus)]
        how_preds   = [d < tau for d, tau in zip(t_how_dists,   adapt_how_taus)]
        combined    = [w & h for w, h in zip(where_preds, how_preds)]

        seq_where_only = True
        seq_combined   = True
        for k in range(len(phases_latent) - 1):
            tw0  = int(np.argmax(where_preds[k]))  if where_preds[k].any()  else T_t
            tw1  = int(np.argmax(where_preds[k+1])) if where_preds[k+1].any() else T_t
            tc0  = int(np.argmax(combined[k]))   if combined[k].any()   else T_t
            tc1  = int(np.argmax(combined[k+1])) if combined[k+1].any() else T_t
            ok_w = tw0 < tw1; ok_c = tc0 < tc1
            if not ok_w: seq_where_only = False
            if not ok_c: seq_combined   = False
            print(f"    ph{k+1}→ph{k+2}  WHERE: {tw0}<{tw1}? {'✓' if ok_w else '✗'}  "
                  f"WHERE∧HOW: {tc0}<{tc1}? {'✓' if ok_c else '✗'}")

        # Frame-level metrics vs gripper GT
        phase_metrics = []
        if len(t_phases_gripper) >= len(phases_latent):
            for k in range(len(phases_latent)):
                ws, we = t_phases_gripper[k]
                gt_k = np.zeros(T_t, bool); gt_k[ws:we] = True
                r_w = eval_pred(t_where_dists[k], gt_k, adapt_where_taus[k])
                r_h = eval_pred(t_how_dists[k],   gt_k, adapt_how_taus[k])
                comb_pred = (t_where_dists[k] < adapt_where_taus[k]) & \
                            (t_how_dists[k]   < adapt_how_taus[k])
                f1_comb = float(2 * (comb_pred & gt_k).sum() /
                                (comb_pred.sum() + gt_k.sum() + 1e-9))
                phase_metrics.append({"k": k+1,
                    "where_f1": round(r_w["f1"], 3),
                    "how_f1":   round(r_h["f1"], 3),
                    "combined_f1": round(f1_comb, 3)})
                print(f"    ph{k+1} vs gripper GT → WHERE F1={r_w['f1']:.3f}  "
                      f"HOW F1={r_h['f1']:.3f}  combined F1={f1_comb:.3f}")

        plot_monitoring(test_ep, test_task,
                        t_where_dists, adapt_where_taus,
                        t_how_dists, adapt_how_taus,
                        t_phases_gripper,
                        out_dir / f"monitor_ep{test_ep:04d}.pdf")

        test_results.append({
            "episode": test_ep,
            "task": test_task,
            "sim": float(row["sim"]),
            "shared_objects": sobjs,
            "n_frames": T_t,
            "n_gripper_phases": len(t_phases_gripper),
            "argmin_where_frames": nearest_t,
            "argmin_how_frames": nearest_how_t,
            "argmin_where_correct": bool(argmin_where_ok),
            "argmin_how_correct": bool(argmin_how_ok),
            "argmin_combined_correct": bool(argmin_comb_ok),
            "where_sequential_correct": bool(seq_where_only),
            "combined_sequential_correct": bool(seq_combined),
            "phase_metrics": phase_metrics,
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    n_argmin_where = sum(r["argmin_where_correct"]    for r in test_results)
    n_argmin_how   = sum(r["argmin_how_correct"]      for r in test_results)
    n_argmin_comb  = sum(r["argmin_combined_correct"] for r in test_results)
    n_where   = sum(r["where_sequential_correct"]     for r in test_results)
    n_combined = sum(r["combined_sequential_correct"] for r in test_results)
    N = len(test_results)
    print(f"\n  Sequential ordering on {N} test episodes:")
    print(f"    Argmin WHERE (threshold-free):       {n_argmin_where}/{N}")
    print(f"    Argmin HOW/DTW (threshold-free):     {n_argmin_how}/{N}")
    print(f"    Argmin WHERE∧HOW (threshold-free):   {n_argmin_comb}/{N}")
    print(f"    Adaptive WHERE:                      {n_where}/{N}")
    print(f"    Adaptive WHERE ∧ HOW (combined):     {n_combined}/{N}")

    out_data = {
        "reference_episode": args.ref_ep,
        "reference_task": ref_task,
        "encoder": args.encoder,
        "n_phases": len(phases_latent),
        "phases_video": [{"start": int(s), "end": int(e)} for s, e in phases_latent],
        "phases_gripper_ref": [{"start": int(s), "end": int(e)} for s, e in phases_gripper],
        "etl_formula": formula,
        "where_taus_ref": [round(float(t), 6) for t in ref_where_taus],
        "how_taus_ref":   [round(float(t), 6) for t in ref_how_taus],
        "cross_phase_separation": sep_vals,
        "test_episodes": test_results,
        "test_n_argmin_where":    n_argmin_where,
        "test_n_argmin_how":      n_argmin_how,
        "test_n_argmin_combined": n_argmin_comb,
        "test_n_where_correct":   n_where,
        "test_n_combined_correct": n_combined,
    }
    out_json = out_dir / "spec_from_video.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"ETL formula: {formula}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-ep",    type=int,   required=True)
    p.add_argument("--encoder",   choices=["dino", "svd_vae"], default="svd_vae")
    p.add_argument("--cam",       choices=["wrist", "exterior1"], default="wrist")
    p.add_argument("--n-phases",  type=int,   default=None)
    p.add_argument("--sigma",     type=float, default=3.0)
    p.add_argument("--min-prominence", type=float, default=0.002)
    p.add_argument("--min-phase-frames", type=int, default=8)
    p.add_argument("--n-test-eps", type=int, default=6)
    p.add_argument("--min-task-sim", type=float, default=0.10)
    p.add_argument("--adapt-quantile", type=float, default=0.25)
    p.add_argument("--out-dir",   default="etl_results/spec_from_video")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
