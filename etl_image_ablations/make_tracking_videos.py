"""
make_tracking_videos.py
-----------------------
Generate MP4 tracking videos for the LeRobot ETL evaluation.

Each frame of the video shows:
  ┌──────────────────────────────────┐
  │      Robot camera frame          │  (actual observation)
  ├──────────────────────────────────┤
  │  Dist to z_A (pick) ─────────── │  live-updating curve
  │  Dist to z_B (insert) ───────── │  live-updating curve
  ├──────────────────────────────────┤
  │  GT phase bar (colour-coded)     │  task_index label
  └──────────────────────────────────┘

Usage:
  cd /path/to/repo
  python -m etl_image_ablations.make_tracking_videos \
      --dataset fmb   --num-videos 5 --out-dir etl_results/lerobot/videos
  python -m etl_image_ablations.make_tracking_videos \
      --dataset iamlab --num-videos 5 --out-dir etl_results/lerobot/videos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from matplotlib.gridspec import GridSpec

# add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etl_image_ablations.eval_lerobot_etl import (
    DINOv2Encoder, R3MEncoder,
    load_dataset_meta,
    load_episode_frames,
    build_spec_latent,
    l2_distances, cosine_distances, l2_normalize,
    sweep_f1,
    conformal_tau,
    phase_done_mask,
    IAMLAB_REPO, IAMLAB_VIDEO_KEY,
    FMB_REPO, FMB_VIDEO_KEY,
    FMB_PICK_TASK_IDS, FMB_INSERT_TASK_IDS,
)

# ─────────────────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "pick":   (0.20, 0.47, 0.73),   # blue
    "insert": (0.17, 0.63, 0.27),   # green
    "other":  (0.80, 0.80, 0.80),   # grey
    "done":   (0.93, 0.60, 0.13),   # orange
}

# ─────────────────────────────────────────────────────────────────────────────

def render_frame_to_array(
    obs_frame: np.ndarray,          # H×W×3 uint8 robot camera
    dist_A: np.ndarray,             # (T,) full episode distances to spec A
    dist_B: Optional[np.ndarray],   # (T,) full episode distances to spec B (or None)
    gt_phase: np.ndarray,           # (T,) int labels: 0=other, 1=pick, 2=insert, 3=done
    t: int,                         # current timestep
    tau_A: float,
    tau_B: Optional[float],
    title: str,
    fig_width: int = 10,
    obs_size: int = 320,
) -> np.ndarray:
    """Render a single composite frame to a numpy RGB uint8 array."""
    n_rows = 3 if dist_B is not None else 2
    fig = plt.figure(figsize=(fig_width, 1.2 * n_rows + obs_size / 100), dpi=100)
    gs  = GridSpec(n_rows + 1, 1, figure=fig,
                   height_ratios=[obs_size / 100] + [1.2] * n_rows,
                   hspace=0.35)

    # ── row 0: robot camera frame ──
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(obs_frame)
    ax_img.set_title(f"{title}   t={t}", fontsize=9, pad=3)
    ax_img.axis("off")

    T = len(dist_A)
    x = np.arange(T)

    def draw_dist(ax, dist, tau_f1, tau_cp, color, label_str):
        ax.plot(x, dist, color=color, lw=1.2, alpha=0.7)
        ax.axvline(t, color="black", lw=1.0, alpha=0.6)
        ax.axhline(tau_f1, color=color, ls="--", lw=1.2, alpha=0.9,
                   label=f"τ_F1={tau_f1:.3f}")
        ax.axhline(tau_cp, color=color, ls=":", lw=1.2, alpha=0.9,
                   label=f"τ_CP={tau_cp:.3f}")
        # shade region where CP predicate fires
        fires = dist < tau_cp
        ax.fill_between(x, dist.min(), dist.max(),
                         where=fires, alpha=0.18, color=color)
        ax.scatter([t], [dist[t]], color=color, s=40, zorder=5)
        ax.set_ylabel(label_str, fontsize=8)
        ax.set_xlim(0, T - 1)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    # ── row 1: distance to z_A (pick) ──
    ax_A = fig.add_subplot(gs[1])
    tau_A_f1, tau_A_cp = (tau_A if isinstance(tau_A, tuple) else (tau_A, tau_A))
    draw_dist(ax_A, dist_A, tau_A_f1, tau_A_cp, PHASE_COLORS["pick"], "dist→z_pick")

    # ── row 2: distance to z_B (insert) ──
    if dist_B is not None and tau_B is not None:
        ax_B = fig.add_subplot(gs[2])
        tau_B_f1, tau_B_cp = (tau_B if isinstance(tau_B, tuple) else (tau_B, tau_B))
        draw_dist(ax_B, dist_B, tau_B_f1, tau_B_cp, PHASE_COLORS["insert"], "dist→z_insert")
        ax_gt = fig.add_subplot(gs[3])
    else:
        ax_gt = fig.add_subplot(gs[2])

    # ── last row: GT phase bar ──
    phase_map = {0: PHASE_COLORS["other"], 1: PHASE_COLORS["pick"],
                 2: PHASE_COLORS["insert"], 3: PHASE_COLORS["done"]}
    bar_colors = np.array([phase_map.get(int(p), PHASE_COLORS["other"]) for p in gt_phase])
    for xi, c in enumerate(bar_colors):
        ax_gt.axvline(xi, color=c, lw=0.8, alpha=0.9)
    ax_gt.axvline(t, color="black", lw=2.0)
    ax_gt.set_xlim(0, T - 1); ax_gt.set_ylim(0, 1)
    ax_gt.set_yticks([])
    ax_gt.set_xlabel("Frame", fontsize=8)
    ax_gt.set_title("GT phase  (blue=pick  green=insert  grey=other  orange=done)",
                    fontsize=7, pad=2)
    # legend patches
    patches = [
        mpatches.Patch(color=PHASE_COLORS["pick"],   label="pick"),
        mpatches.Patch(color=PHASE_COLORS["insert"], label="insert"),
        mpatches.Patch(color=PHASE_COLORS["other"],  label="other"),
        mpatches.Patch(color=PHASE_COLORS["done"],   label="done"),
    ]
    ax_gt.legend(handles=patches, loc="upper right", fontsize=7,
                 ncol=4, framealpha=0.7)

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # tostring_rgb was removed in newer matplotlib; use buffer_rgba instead
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    arr = buf[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return arr


def write_video(frames: List[np.ndarray], out_path: Path, fps: int = 10):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width  = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for frame_arr in frames:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    print(f"  Wrote {out_path}  ({len(frames)} frames @ {fps}fps)")


# ─────────────────────────────────────────────────────────────────────────────

def make_fmb_videos(
    encoder,
    num_videos: int,
    out_dir: Path,
    dist_fn=None,
    normalize: bool = False,
    seed: int = 42,
):
    if dist_fn is None:
        dist_fn = cosine_distances
        normalize = True

    print("\n=== FMB tracking videos ===")
    info, data_df, ep_df = load_dataset_meta(FMB_REPO)
    rng = np.random.default_rng(seed)

    # pick episodes that contain both pick and insert phases
    valid_ep_ids = []
    for _, row in ep_df.iterrows():
        tmin = row.get("stats/task_index/min", [None])
        tmax = row.get("stats/task_index/max", [None])
        if isinstance(tmin, list): tmin = tmin[0]
        if isinstance(tmax, list): tmax = tmax[0]
        if tmin is None: continue
        ids = set(range(int(tmin), int(tmax)+1))
        if ids & FMB_PICK_TASK_IDS and ids & FMB_INSERT_TASK_IDS:
            valid_ep_ids.append(int(row["episode_index"]))

    cal_n    = min(40, len(valid_ep_ids) // 2)
    cal_ids  = rng.choice(valid_ep_ids, cal_n, replace=False).tolist()
    test_ids = [e for e in valid_ep_ids if e not in set(cal_ids)]
    video_ids = rng.choice(test_ids, min(num_videos, len(test_ids)), replace=False).tolist()

    # Build spec latents from calibration using phase_done_mask
    cal_pk_embs, cal_in_embs = [], []
    cal_d_pick, cal_l_pick   = [], []
    cal_d_insert, cal_l_insert = [], []

    for ep in cal_ids:
        row     = ep_df[ep_df["episode_index"]==ep].iloc[0]
        ep_data = data_df[data_df["episode_index"]==ep].sort_values("frame_index")
        frames  = load_episode_frames(FMB_REPO, row, FMB_VIDEO_KEY)
        if not frames: continue
        T    = min(len(frames), len(ep_data))
        tids = ep_data["task_index"].values[:T]
        emb  = encoder.encode(frames[:T])
        if normalize: emb = l2_normalize(emb)

        pick_full = np.array([int(t) in FMB_PICK_TASK_IDS   for t in tids])
        ins_full  = np.array([int(t) in FMB_INSERT_TASK_IDS for t in tids])
        pick_done = phase_done_mask(tids, FMB_PICK_TASK_IDS)
        ins_done  = phase_done_mask(tids, FMB_INSERT_TASK_IDS)

        if pick_done.any():  cal_pk_embs.append(emb[pick_done].mean(axis=0))
        if ins_done.any():   cal_in_embs.append(emb[ins_done].mean(axis=0))

        if cal_pk_embs:
            z_pk_tmp = np.stack(cal_pk_embs).mean(axis=0)
            if normalize: z_pk_tmp = z_pk_tmp / max(np.linalg.norm(z_pk_tmp), 1e-8)
            cal_d_pick.append(dist_fn(emb, z_pk_tmp))
            cal_l_pick.append(pick_done.astype(int))
        if cal_in_embs:
            z_in_tmp = np.stack(cal_in_embs).mean(axis=0)
            if normalize: z_in_tmp = z_in_tmp / max(np.linalg.norm(z_in_tmp), 1e-8)
            cal_d_insert.append(dist_fn(emb, z_in_tmp))
            cal_l_insert.append(ins_done.astype(int))

    z_pick   = np.stack(cal_pk_embs).mean(axis=0)
    z_insert = np.stack(cal_in_embs).mean(axis=0)
    if normalize:
        z_pick   = z_pick   / max(np.linalg.norm(z_pick),   1e-8)
        z_insert = z_insert / max(np.linalg.norm(z_insert), 1e-8)

    tau_pick_f1,   _, _, _, _ = sweep_f1(np.concatenate(cal_d_pick),   np.concatenate(cal_l_pick))
    tau_insert_f1, _, _, _, _ = sweep_f1(np.concatenate(cal_d_insert), np.concatenate(cal_l_insert))
    tau_pick_cp   = conformal_tau(np.concatenate(cal_d_pick),   np.concatenate(cal_l_pick))
    tau_insert_cp = conformal_tau(np.concatenate(cal_d_insert), np.concatenate(cal_l_insert))
    print(f"  τ_pick   F1={tau_pick_f1:.4f}  CP={tau_pick_cp:.4f}")
    print(f"  τ_insert F1={tau_insert_f1:.4f}  CP={tau_insert_cp:.4f}")

    # Render videos
    for ep_idx in video_ids:
        print(f"  Rendering ep {ep_idx} …")
        row     = ep_df[ep_df["episode_index"]==ep_idx].iloc[0]
        ep_data = data_df[data_df["episode_index"]==ep_idx].sort_values("frame_index")
        obs_frames = load_episode_frames(FMB_REPO, row, FMB_VIDEO_KEY)
        if not obs_frames: continue
        T    = min(len(obs_frames), len(ep_data))
        tids = ep_data["task_index"].values[:T]
        emb  = encoder.encode(obs_frames[:T])
        if normalize: emb = l2_normalize(emb)

        dist_A = dist_fn(emb, z_pick)
        dist_B = dist_fn(emb, z_insert)

        # GT phase bar: 1=pick, 2=insert, 0=other
        gt_phase = np.zeros(T, dtype=int)
        for i, tid in enumerate(tids):
            if int(tid) in FMB_PICK_TASK_IDS:    gt_phase[i] = 1
            elif int(tid) in FMB_INSERT_TASK_IDS: gt_phase[i] = 2
        # highlight done windows
        pick_done = phase_done_mask(tids, FMB_PICK_TASK_IDS)
        ins_done  = phase_done_mask(tids, FMB_INSERT_TASK_IDS)
        gt_phase[pick_done] = 3  # orange = pick completion window
        gt_phase[ins_done]  = 3  # orange = insert completion window

        video_frames = []
        stride = max(1, T // 150)
        for t in range(0, T, stride):
            arr = render_frame_to_array(
                obs_frames[t], dist_A, dist_B, gt_phase, t,
                (tau_pick_f1, tau_pick_cp),
                (tau_insert_f1, tau_insert_cp),
                title=f"FMB ep{ep_idx} [Robometer]",
            )
            video_frames.append(arr)
        write_video(video_frames, out_dir / f"fmb_ep{ep_idx}.mp4", fps=8)


def make_iamlab_videos(
    encoder: DINOv2Encoder,
    num_videos: int,
    out_dir: Path,
    seed: int = 42,
):
    print("\n=== iamlab tracking videos ===")
    info, data_df, ep_df = load_dataset_meta(IAMLAB_REPO)
    rng = np.random.default_rng(seed)

    pick_eps   = ep_df[ep_df["tasks"].apply(lambda t: any("Pick up green" in s for s in t))]["episode_index"].tolist()
    insert_eps = ep_df[ep_df["tasks"].apply(lambda t: any("Insert" in s for s in t))]["episode_index"].tolist()

    n_cal = min(15, len(pick_eps)//2, len(insert_eps)//2)
    cal_p = rng.choice(pick_eps,   n_cal, replace=False).tolist()
    cal_i = rng.choice(insert_eps, n_cal, replace=False).tolist()
    test_p  = [e for e in pick_eps   if e not in set(cal_p)]
    test_i  = [e for e in insert_eps if e not in set(cal_i)]
    video_p = rng.choice(test_p, min(num_videos//2+1, len(test_p)), replace=False).tolist()
    video_i = rng.choice(test_i, min(num_videos//2+1, len(test_i)), replace=False).tolist()

    # Build spec latents
    cal_p_embs, cal_i_embs = [], []
    for ep in cal_p:
        row = ep_df[ep_df["episode_index"]==ep].iloc[0]
        f = load_episode_frames(IAMLAB_REPO, row, IAMLAB_VIDEO_KEY)
        if f: cal_p_embs.append(encoder.encode(f))
    for ep in cal_i:
        row = ep_df[ep_df["episode_index"]==ep].iloc[0]
        f = load_episode_frames(IAMLAB_REPO, row, IAMLAB_VIDEO_KEY)
        if f: cal_i_embs.append(encoder.encode(f))

    z_pick   = np.concatenate(cal_p_embs).mean(axis=0)
    z_insert = np.concatenate(cal_i_embs).mean(axis=0)

    # Rough thresholds: use mean ± 1 std of cal positive dists
    cal_d_pick   = np.concatenate([l2_distances(e, z_pick)   for e in cal_p_embs])
    cal_d_insert = np.concatenate([l2_distances(e, z_insert) for e in cal_i_embs])
    tau_pick   = float(np.percentile(cal_d_pick,   40))
    tau_insert = float(np.percentile(cal_d_insert, 40))
    print(f"  τ_pick={tau_pick:.2f}  τ_insert={tau_insert:.2f}")

    def render_ep(ep_idx, kind):
        row = ep_df[ep_df["episode_index"]==ep_idx].iloc[0]
        ep_data = data_df[data_df["episode_index"]==ep_idx].sort_values("frame_index")
        obs_frames = load_episode_frames(IAMLAB_REPO, row, IAMLAB_VIDEO_KEY)
        if not obs_frames: return
        T   = min(len(obs_frames), len(ep_data))
        emb = encoder.encode(obs_frames[:T])
        dist_A = l2_distances(emb, z_pick)
        dist_B = l2_distances(emb, z_insert)
        done_arr = ep_data["next.done"].values[:T].astype(bool)
        # GT phase: 1=pick episode, 2=insert episode, 3=done
        gt_phase = np.ones(T, dtype=int) * (1 if kind=="pick" else 2)
        gt_phase[done_arr] = 3

        video_frames = []
        stride = max(1, T // 120)
        for t in range(0, T, stride):
            arr = render_frame_to_array(
                obs_frames[t], dist_A, dist_B, gt_phase, t,
                tau_pick, tau_insert,
                title=f"iamlab ep{ep_idx} ({kind})",
            )
            video_frames.append(arr)
        write_video(video_frames, out_dir / f"iamlab_{kind}_ep{ep_idx}.mp4", fps=8)

    for ep in video_p:
        print(f"  Rendering pick ep {ep} …")
        render_ep(ep, "pick")
    for ep in video_i:
        print(f"  Rendering insert ep {ep} …")
        render_ep(ep, "insert")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["fmb", "iamlab", "both"], default="fmb")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--out-dir",    type=str, default="etl_results/lerobot/videos")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder",    choices=["dino", "r3m", "robometer"], default="dino")
    parser.add_argument("--distance",   choices=["l2", "cosine"], default="cosine")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    use_cosine = args.distance == "cosine"
    dist_fn    = cosine_distances if use_cosine else l2_distances

    if args.encoder == "robometer":
        sys.path.insert(0, str(Path(__file__).parent))
        from robometer_encoder import RobometerEncoder as _RobometerEncoder
        _rbm = _RobometerEncoder(device=args.device)
        class _Enc:
            def encode(self_, frames, batch_size=8):
                return _rbm.encode_frames(frames, task="robot manipulation",
                                          chunk_size=8, stride=4, batch_size=4)
        enc = _Enc()
    elif args.encoder == "r3m":
        enc = R3MEncoder(device=args.device)
    else:
        enc = DINOv2Encoder(device=args.device)

    if args.dataset in ("fmb", "both"):
        make_fmb_videos(enc, args.num_videos, out_dir,
                        dist_fn=dist_fn, normalize=use_cosine)
    if args.dataset in ("iamlab", "both"):
        make_iamlab_videos(enc, args.num_videos, out_dir)


if __name__ == "__main__":
    main()
