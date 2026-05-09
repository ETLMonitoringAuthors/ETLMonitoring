"""
spec_way_droid.py
-----------------
Specify the WAY a behaviour should be done using pure ETL predicates.

Given a short reference clip (one episode, one phase) showing the CORRECT
WAY to perform a sub-task (e.g. "pick up the object slowly and deliberately"),
we:

  1. Encode every frame with SVD VAE  →  z_0, …, z_{T-1}  in R^{3840}
  2. Uniformly subsample to K waypoints  z_w0, …, z_{wK-1}
  3. Generate the sequential ETL formula:
       φ = F(near_0 ∧ F(near_1 ∧ … ∧ F(near_{K-1}) … ))
     where  near_k(t) ≡ cosine_dist(z_t, z_wk) < τ_k

This is pure ETL – no DTW, no WHERE goal.  The predicate sequence forces
the test trajectory to visit the same points in embedding space in the
same order, capturing HOW the motion unfolds.

Evaluation: threshold-free argmin ordering
  nearest[k] = argmin_t  cosine_dist(z_t, z_wk)
  Correct if  nearest[0] < nearest[1] < … < nearest[K-1]

Usage
-----
  cd /path/to/repo
  TMPDIR=/tmp HF_HOME=~/.cache/huggingface \\
  python \\
      -m etl_image_ablations.spec_way_droid \\
      --ref-ep 13 --phase-start 0 --phase-end 90 \\
      --n-waypoints 8 --n-test-eps 8 \\
      --out-dir etl_results/spec_way_svd
"""

from __future__ import annotations

import argparse
import json
import os
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

warnings.filterwarnings("ignore")
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from huggingface_hub import hf_hub_download

REPO_ID = "lerobot/droid_100"
CAM_KEYS = {
    "wrist":     "observation.images.wrist_image_left",
    "exterior1": "observation.images.exterior_image_1_left",
}

# Gripper constants for GT comparison
CLOSE_THRESH = 0.45
OPEN_THRESH  = 0.20
MIN_CLOSE    = 5

OBJECT_NOUNS = {
    "bottle","bottles","cup","cups","mug","mugs","towel","towels","cloth",
    "paper","papers","pen","pens","lid","lids","pot","pots","pan","bowl",
    "bowls","plate","plates","basket","box","boxes","bag","bags","tray",
    "book","books","block","blocks","cube","cubes","object","objects",
    "toy","toys","ball","balls","can","cans","jar","jars","fork","spoon",
    "sponge","brush","white","black","red","blue","green","yellow",
}
ACTION_VERBS = {
    "pick","pickup","grab","grasp","take","lift","put","place","set",
    "drop","lay","move","push","fold","unfold","open","close","stack",
}


# ── Encoder ───────────────────────────────────────────────────────────────────

class SVDVAEEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        from diffusers.models import AutoencoderKLTemporalDecoder
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device).eval()
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [TF.resize(torch.from_numpy(f).permute(2,0,1).float().div(255.),
                               [192, 320], antialias=True) * 2.0 - 1.0
                     for f in frames[i:i+batch_size]]
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


def load_frames(ep_row, cam_key: str) -> List[np.ndarray]:
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    fidx  = int(ep_row[f"videos/{cam_key}/file_index"])
    t0    = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    t1    = float(ep_row[f"videos/{cam_key}/to_timestamp"])
    path  = hf_hub_download(REPO_ID,
        f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4", repo_type="dataset")
    return decode_mp4(path, t0, t1)


def get_state(df: pd.DataFrame, ep: int) -> np.ndarray:
    rows = df[df["episode_index"] == ep]
    return np.stack(rows.reset_index(drop=True)["observation.state"].values)


def gripper_phase(gripper: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (start, end) of first sustained close event, or None."""
    in_c = False; start = 0
    for t, v in enumerate(gripper):
        if not in_c and v > CLOSE_THRESH:
            in_c = True; start = t
        elif in_c and v < OPEN_THRESH:
            if t - start >= MIN_CLOSE:
                return (start, t)
            in_c = False
    if in_c and len(gripper) - start >= MIN_CLOSE:
        return (start, len(gripper))
    return None


# ── Task similarity ───────────────────────────────────────────────────────────

import re as _re

def _tok(s: str) -> set:
    return set(_re.findall(r"[a-z]+", s.lower()))

def task_sim(a: str, b: str) -> float:
    wa, wb = _tok(a), _tok(b)
    if not wa or not wb: return 0.0
    num = den = 0.0
    for w in wa | wb:
        wt = 3.0 if w in OBJECT_NOUNS else (2.0 if w in ACTION_VERBS else 1.0)
        if w in wa and w in wb: num += wt
        den += wt
    return num / (den + 1e-9)


# ── ETL waypoint spec ─────────────────────────────────────────────────────────

def cosine_dist(embs: np.ndarray, z: np.ndarray) -> np.ndarray:
    en = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(np.linalg.norm(z), 1e-8)
    return 1.0 - (en @ zn)


def build_waypoints(embs: np.ndarray, K: int) -> np.ndarray:
    """Uniformly subsample the trajectory to K waypoints. Shape: [K, D]."""
    idx = np.round(np.linspace(0, len(embs) - 1, K)).astype(int)
    return embs[idx]


def etl_formula(K: int) -> str:
    def nest(k):
        if k == K - 1:
            return f"F(near_{k})"
        return f"F(near_{k} ∧ {nest(k+1)})"
    return nest(0)


def adaptive_tau(dist: np.ndarray, q: float = 0.20) -> float:
    return float(np.quantile(dist, q))


def mine_tau_f1(dists: np.ndarray, gt: np.ndarray, n: int = 300) -> Tuple[float, float]:
    if not gt.any():
        return float(np.median(dists)), 0.0
    taus = np.linspace(dists.min(), dists.max(), n)
    best_f1, best_tau = 0.0, float(np.median(taus))
    for tau in taus:
        pred = dists < tau
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        p  = tp / (tp + fp + 1e-9); r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return float(best_tau), float(best_f1)


def argmin_order_ok(
    dists_per_wp: List[np.ndarray],
    search_start: int = 0,
    search_end: Optional[int] = None,
) -> Tuple[List[int], bool]:
    """
    Argmin ordering check, optionally restricted to [search_start, search_end).
    Restricting to the gripper-phase window removes out-of-phase noise and asks
    purely: 'within the behaviour window, does the trajectory visit the
    waypoints in the right order?'
    """
    end = search_end if search_end is not None else len(dists_per_wp[0])
    nearest = [int(search_start + np.argmin(d[search_start:end]))
               for d in dists_per_wp]
    ok = all(nearest[k] < nearest[k+1] for k in range(len(nearest) - 1))
    return nearest, ok


def velocity_profile(embs: np.ndarray) -> np.ndarray:
    """Frame-to-frame cosine distance — the 'speed' in latent space."""
    en = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
    delta = 1.0 - (en[:-1] * en[1:]).sum(axis=1)
    return gaussian_filter1d(delta.astype(np.float64), sigma=2.0)


# ── Visualisation ─────────────────────────────────────────────────────────────

COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#999999"]


def plot_waypoint_distances(
    ep_idx: int,
    task: str,
    waypoint_dists: List[np.ndarray],
    taus: List[float],
    nearest: List[int],
    ref_phase: Tuple[int, int],
    gripper_phase_ep: Optional[Tuple[int, int]],
    out_path: Path,
):
    K = len(waypoint_dists)
    T = len(waypoint_dists[0])
    t_ax = np.arange(T)

    fig, axes = plt.subplots(K, 1, figsize=(14, 1.8 * K + 0.5), sharex=True)
    if K == 1: axes = [axes]

    for k, (d, tau, ax) in enumerate(zip(waypoint_dists, taus, axes)):
        col = COLORS[k % len(COLORS)]
        pred = d < tau
        ax.plot(t_ax, d, color=col, lw=1.2, label=f"near_{k}")
        ax.axhline(tau, color="k", lw=0.9, ls="--", label=f"τ={tau:.3f}")
        ax.fill_between(t_ax, 0, d, where=pred, alpha=0.22, color=col)
        ax.axvline(nearest[k], color=col, lw=1.4, ls=":", label=f"argmin t={nearest[k]}")
        if gripper_phase_ep:
            gs, ge = gripper_phase_ep
            gt = np.zeros(T, bool); gt[gs:ge] = True
            ax.fill_between(t_ax, d.max() * 0.95, d.max(),
                            where=gt, alpha=0.18, color="gray", label="gripper GT")
        ax.set_ylabel(f"d(z_t, w_{k})", fontsize=7)
        ax.legend(fontsize=6, ncol=5, loc="upper right"); ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Frame", fontsize=8)
    fig.suptitle(f"ep{ep_idx} — waypoint distances  |  {task[:72]}", fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distances plot → {out_path.name}")


def plot_velocity_comparison(
    ref_vel: np.ndarray,
    test_vels: List[np.ndarray],
    test_labels: List[str],
    test_ok: List[bool],
    out_path: Path,
):
    """
    Overlay velocity profiles (||Δz_t|| in latent space) of reference and
    test episodes.  This shows whether the test episode moves through
    embedding space at a similar pace to the reference — the HOW signature.
    """
    fig, ax = plt.subplots(figsize=(13, 3.5))
    # Normalise each to [0, 1] for visual comparison
    def norm(v):
        r = v.max() - v.min()
        return (v - v.min()) / (r + 1e-9)

    t_ref = np.linspace(0, 1, len(ref_vel))
    ax.plot(t_ref, norm(ref_vel), color="black", lw=2.2, label="reference", zorder=5)

    for i, (vel, lbl, ok) in enumerate(zip(test_vels, test_labels, test_ok)):
        col = COLORS[i % len(COLORS)]
        ls  = "-" if ok else "--"
        t_t = np.linspace(0, 1, len(vel))
        ax.plot(t_t, norm(vel), color=col, lw=1.1, ls=ls, alpha=0.75,
                label=f"{lbl[:35]} {'✓' if ok else '✗'}")

    ax.set_xlabel("Normalised time (0=start, 1=end of phase)", fontsize=9)
    ax.set_ylabel("Normalised Δcos (speed in latent space)", fontsize=9)
    ax.set_title("Velocity profile: does the test follow the same pace?", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Velocity comparison → {out_path.name}")


def plot_trajectory_pca(
    ref_embs: np.ndarray,
    test_embs_list: List[np.ndarray],
    waypoints: np.ndarray,
    test_labels: List[str],
    test_ok: List[bool],
    out_path: Path,
):
    """PCA projection of all trajectories + waypoint markers."""
    from sklearn.decomposition import PCA
    all_embs = np.vstack([ref_embs] + test_embs_list + [waypoints])
    pca = PCA(n_components=2).fit(all_embs)
    K = len(waypoints)

    fig, ax = plt.subplots(figsize=(9, 7))
    # Reference trajectory
    rp = pca.transform(ref_embs)
    ax.plot(rp[:, 0], rp[:, 1], color="black", lw=2.0, alpha=0.85, label="reference", zorder=4)
    ax.scatter(rp[0, 0], rp[0, 1], color="black", s=60, zorder=6, marker="o")
    ax.scatter(rp[-1, 0], rp[-1, 1], color="black", s=60, zorder=6, marker="s")

    # Waypoints
    wp = pca.transform(waypoints)
    for k in range(K):
        col = COLORS[k % len(COLORS)]
        ax.scatter(wp[k, 0], wp[k, 1], color=col, s=120, zorder=7,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(f"w{k}", (wp[k, 0], wp[k, 1]),
                    fontsize=7, ha="center", va="bottom")

    # Test trajectories
    for i, (te, lbl, ok) in enumerate(zip(test_embs_list, test_labels, test_ok)):
        col = COLORS[i % len(COLORS)]
        ls  = "-" if ok else "--"
        tp  = pca.transform(te)
        ax.plot(tp[:, 0], tp[:, 1], color=col, lw=1.0, ls=ls, alpha=0.55,
                label=f"{lbl[:28]} {'✓' if ok else '✗'}")

    ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
    ax.set_title("PCA of reference + test trajectories in SVD VAE latent space", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA plot → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_key = CAM_KEYS[args.cam]
    K = args.n_waypoints

    print("Loading DROID metadata …")
    data_df, ep_df = load_meta()
    ep_df = ep_df.copy()
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else "")

    # ── Reference episode and phase ───────────────────────────────────────────
    ref_row = ep_df[ep_df["episode_index"] == args.ref_ep].iloc[0]
    ref_task = ref_row["task_text"]
    print(f"\nReference ep{args.ref_ep}: '{ref_task}'")

    ref_frames = load_frames(ref_row, cam_key)
    ref_state  = get_state(data_df, args.ref_ep)
    T_ref = min(len(ref_frames), len(ref_state))
    ref_gripper = ref_state[:T_ref, 6]

    # Determine the reference phase to spec
    if args.phase_start is not None and args.phase_end is not None:
        ps, pe = args.phase_start, min(args.phase_end, T_ref)
    else:
        gp = gripper_phase(ref_gripper)
        if gp:
            # extend window slightly before grip to capture the approach
            ps = max(0, gp[0] - 20)
            pe = gp[1]
        else:
            ps, pe = 0, T_ref
    print(f"  Phase window: [{ps}, {pe})  len={pe-ps}  "
          f"({'auto-gripper' if args.phase_start is None else 'manual'})")

    # Encode reference phase
    enc = SVDVAEEncoder(args.device)
    print(f"  Encoding reference phase ({pe-ps} frames) …")
    phase_frames = ref_frames[ps:pe]
    phase_embs   = enc.encode(phase_frames)   # [L, D]

    # Build K waypoints
    waypoints = build_waypoints(phase_embs, K)   # [K, D]
    print(f"  Built {K} waypoints in R^{waypoints.shape[1]}")

    # Velocity profile of reference
    ref_vel = velocity_profile(phase_embs)

    # ETL formula
    formula = etl_formula(K)
    print(f"\n  ETL formula: {formula}")

    # Self-eval on reference phase
    ref_dists = [cosine_dist(phase_embs, waypoints[k]) for k in range(K)]
    ref_nearest, ref_ok = argmin_order_ok(ref_dists)
    print(f"  Self-check argmin order: {ref_nearest}  {'✓' if ref_ok else '✗'}")
    for k in range(K):
        print(f"    w{k}: argmin at t={ref_nearest[k]}/{len(phase_embs)-1}  "
              f"dist={ref_dists[k][ref_nearest[k]]:.4f}")

    # Save reference npz
    np.savez_compressed(out_dir / f"ref_phase_embs.npz",
                        embs=phase_embs, waypoints=waypoints,
                        ps=ps, pe=pe)

    # ── Test episodes ─────────────────────────────────────────────────────────
    ep_df["sim"] = ep_df["task_text"].apply(lambda t: task_sim(ref_task, t))
    similar = ep_df[
        (ep_df["episode_index"] != args.ref_ep) &
        (ep_df["sim"] >= args.min_sim)
    ].sort_values("sim", ascending=False).head(args.n_test_eps)

    print(f"\n  Test episodes (sim ≥ {args.min_sim}):")
    for _, r in similar.iterrows():
        print(f"    ep{int(r['episode_index']):3d}  sim={r['sim']:.3f}  "
              f"'{r['task_text'][:55]}'")

    results = []
    test_embs_list, test_labels, test_ok_list = [], [], []

    for _, row in similar.iterrows():
        ep  = int(row["episode_index"])
        task = row["task_text"]
        print(f"\n  {'='*50}")
        print(f"  ep{ep}  '{task[:65]}'")

        try:
            t_frames = load_frames(row, cam_key)
            t_state  = get_state(data_df, ep)
        except Exception as e:
            print(f"    SKIP: {e}"); continue

        T_t = min(len(t_frames), len(t_state))
        t_gripper = t_state[:T_t, 6]
        t_gp      = gripper_phase(t_gripper)

        # Encode full episode
        print(f"    Encoding {T_t} frames …")
        t_embs = enc.encode(t_frames[:T_t])
        np.savez_compressed(out_dir / f"embs_ep{ep:04d}.npz", embs=t_embs)

        # Waypoint distances over full episode
        t_dists = [cosine_dist(t_embs, waypoints[k]) for k in range(K)]

        # Argmin ordering: (a) full episode, (b) within gripper-phase window
        nearest_full, ok_full = argmin_order_ok(t_dists)
        if t_gp:
            gs, ge = t_gp
            win_s = max(0, gs - 20)
            nearest_win, ok_win = argmin_order_ok(t_dists, win_s, ge)
        else:
            nearest_win, ok_win = nearest_full, ok_full
        print(f"    Argmin (full episode):    {nearest_full}  {'✓' if ok_full else '✗'}")
        print(f"    Argmin (phase window):    {nearest_win}   {'✓' if ok_win else '✗'}")

        # Adaptive thresholds + F1 vs gripper GT
        adapt_taus = [adaptive_tau(d, args.adapt_quantile) for d in t_dists]
        phase_metrics = []
        if t_gp:
            gs, ge = t_gp
            gt = np.zeros(T_t, bool); gt[gs:ge] = True
            for k in range(K):
                tau_k, f1_k = mine_tau_f1(t_dists[k], gt)
                phase_metrics.append({"k": k, "tau_f1": round(tau_k, 4),
                                      "f1": round(f1_k, 3)})
            avg_f1 = np.mean([m["f1"] for m in phase_metrics])
            print(f"    Gripper GT [{gs},{ge})  avg waypoint F1={avg_f1:.3f}")

        # Velocity profile of the gripper-aligned phase (for comparison)
        if t_gp:
            t_vel = velocity_profile(t_embs[max(0,t_gp[0]-20): t_gp[1]])
        else:
            t_vel = velocity_profile(t_embs)

        # Plot waypoint distances
        plot_waypoint_distances(
            ep, task, t_dists, adapt_taus, nearest_full,
            (ps, pe), t_gp,
            out_dir / f"dist_ep{ep:04d}.pdf")

        test_embs_list.append(t_embs)
        test_labels.append(f"ep{ep}")
        test_ok_list.append(ok_full)

        results.append({
            "episode": ep,
            "task": task,
            "sim": float(row["sim"]),
            "argmin_nearest_full": nearest_full,
            "argmin_nearest_window": nearest_win,
            "argmin_correct_full":   bool(ok_full),
            "argmin_correct_window": bool(ok_win),
            "gripper_phase": list(t_gp) if t_gp else None,
            "phase_metrics": phase_metrics,
        })

    # ── Velocity comparison plot ──────────────────────────────────────────────
    if results:
        test_vels = []
        for r in results:
            ep = r["episode"]
            npz = np.load(out_dir / f"embs_ep{ep:04d}.npz")["embs"]
            gp  = r["gripper_phase"]
            if gp:
                clip = npz[max(0, gp[0]-20): gp[1]]
            else:
                clip = npz
            test_vels.append(velocity_profile(clip))

        plot_velocity_comparison(
            ref_vel, test_vels,
            [r["task"] for r in results],
            [r["argmin_correct_full"] for r in results],
            out_dir / "velocity_comparison.pdf")

        # PCA of trajectories in reference-phase neighbourhood
        # Use only the gripper-aligned slice of each test episode for fair comparison
        phase_clips = []
        for r in results:
            npz = np.load(out_dir / f"embs_ep{r['episode']:04d}.npz")["embs"]
            gp = r["gripper_phase"]
            if gp:
                phase_clips.append(npz[max(0, gp[0]-20): gp[1]])
            else:
                half = len(npz) // 3
                phase_clips.append(npz[:half])

        try:
            from sklearn.decomposition import PCA as _PCA
            plot_trajectory_pca(
                phase_embs, phase_clips, waypoints,
                [r["task"] for r in results],
                [r["argmin_correct_full"] for r in results],
                out_dir / "pca_trajectories.pdf")
        except ImportError:
            print("  (sklearn not available — skipping PCA plot)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_full   = sum(r["argmin_correct_full"]   for r in results)
    n_window = sum(r["argmin_correct_window"] for r in results)
    N        = len(results)
    print(f"\n{'='*60}")
    print(f"HOW-spec argmin ordering:")
    print(f"  Full episode (hard):   {n_full}/{N}")
    print(f"  Phase window (fair):   {n_window}/{N}")
    print(f"ETL formula: {formula}")

    out = {
        "reference_episode": args.ref_ep,
        "reference_task": ref_task,
        "phase_window": [ps, pe],
        "n_waypoints": K,
        "etl_formula": formula,
        "test_n_correct_full":   n_full,
        "test_n_correct_window": n_window,
        "test_n_total":          N,
        "test_episodes":         results,
    }
    with open(out_dir / "spec_way.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_dir}/spec_way.json")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-ep",     type=int, required=True)
    p.add_argument("--phase-start", type=int, default=None,
                   help="start frame of reference phase (default: auto-detect from gripper)")
    p.add_argument("--phase-end",   type=int, default=None)
    p.add_argument("--n-waypoints", type=int, default=8)
    p.add_argument("--cam",         choices=["wrist","exterior1"], default="wrist")
    p.add_argument("--n-test-eps",  type=int, default=8)
    p.add_argument("--min-sim",     type=float, default=0.08)
    p.add_argument("--adapt-quantile", type=float, default=0.20)
    p.add_argument("--out-dir",     default="etl_results/spec_way_svd")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
