"""
make_spec_videos.py
-------------------
Render annotated MP4 videos from a spec_from_video_droid.py run.

For each episode (reference + tests) it produces one MP4 where every
frame shows:

  Left panel   — original wrist-camera frame with a coloured phase/
                 predicate border and a status label.

  Right panel  — live-updating line plots of:
                   • WHERE cosine distance to each phase spec latent
                   • HOW sliding-DTW distance to each phase trajectory
                 with horizontal τ lines; shaded when predicate fires.

  Bottom bar   — thin progress bar: filled portions show WHERE ∧ HOW
                 active frames per phase, colour-coded by phase.

Usage
-----
  cd /path/to/repo
  TMPDIR=/tmp HF_HOME=~/.cache/huggingface \\
  python \\
      -m etl_image_ablations.make_spec_videos \\
      --run-dir etl_results/spec_from_video_svd \\
      --ref-ep 13 --cam wrist --fps 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import av
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from huggingface_hub import hf_hub_download

REPO_ID = "lerobot/droid_100"
CAM_KEYS = {
    "wrist":     "observation.images.wrist_image_left",
    "exterior1": "observation.images.exterior_image_1_left",
}
PHASE_COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#F4A582", "#984EA3"]
ALPHA_ACTIVE = 0.45

# ── data helpers (minimal copy from main script) ──────────────────────────────

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


# ── distance computation (reproduced compactly) ───────────────────────────────

def cosine_dist_vec(embs: np.ndarray, z: np.ndarray) -> np.ndarray:
    en = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(np.linalg.norm(z), 1e-8)
    return 1.0 - (en @ zn)


def subsample_traj(embs: np.ndarray, n: int) -> np.ndarray:
    idx = np.round(np.linspace(0, len(embs) - 1, n)).astype(int)
    return embs[idx]


def dtw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    Na, Nb = len(a), len(b)
    an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
    cost = 1.0 - (an @ bn.T)
    dp = np.full((Na, Nb), np.inf)
    dp[0, 0] = cost[0, 0]
    for i in range(1, Na): dp[i, 0] = dp[i-1, 0] + cost[i, 0]
    for j in range(1, Nb): dp[0, j] = dp[0, j-1] + cost[0, j]
    for i in range(1, Na):
        for j in range(1, Nb):
            dp[i, j] = cost[i, j] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[-1, -1]) / (Na + Nb)


def sliding_dtw(embs, ref_traj, window_frames, step=4, n_pts=24):
    T = len(embs)
    half = window_frames // 2
    scores = np.full(T, np.nan)
    for c in range(half, T - half, step):
        sub = subsample_traj(embs[c - half: c + half], n_pts)
        scores[c] = dtw_cosine(sub, ref_traj)
    valid = np.where(~np.isnan(scores))[0]
    if len(valid) == 0:
        return np.zeros(T)
    scores[:valid[0]] = scores[valid[0]]
    scores[valid[-1]:] = scores[valid[-1]]
    return np.interp(np.arange(T), valid, scores[valid])


def adaptive_tau(d: np.ndarray, q: float = 0.25) -> float:
    return float(np.quantile(d, q))


# ── rendering helpers ─────────────────────────────────────────────────────────

SIGNAL_H  = 180    # px height for the right-panel signal plots
SIGNAL_W  = 480    # px width for the right panel
BORDER    = 6      # px coloured border on the video frame
LABEL_H   = 28     # px for text label strip above frame
PROG_H    = 18     # px for bottom progress bar
DTW_N_PTS = 24


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def render_signal_panel(
    t: int,
    where_dists: List[np.ndarray],
    where_taus: List[float],
    how_dists: List[np.ndarray],
    how_taus: List[float],
    height: int,
    width: int,
) -> np.ndarray:
    """Render the right-side signal panel as a [height, width, 3] uint8 array."""
    K = len(where_dists)
    T = len(where_dists[0])
    n_rows = K * 2
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(width / 100, height / 100),
                             dpi=100)
    if n_rows == 1:
        axes = [axes]

    for k in range(K):
        col = PHASE_COLORS[k % len(PHASE_COLORS)]
        ax_w = axes[k * 2]
        ax_h = axes[k * 2 + 1] if k * 2 + 1 < len(axes) else axes[-1]

        # WHERE
        d_w = where_dists[k]; tau_w = where_taus[k]
        ax_w.plot(d_w[:t+1], color=col, lw=1.2)
        ax_w.axhline(tau_w, color="k", lw=0.8, ls="--")
        ax_w.fill_between(np.arange(t+1), tau_w, d_w[:t+1],
                          where=d_w[:t+1] < tau_w, alpha=0.3, color=col)
        ax_w.axvline(t, color="gray", lw=0.7, ls=":")
        ax_w.set_xlim(0, T); ax_w.set_ylim(0, max(d_w.max(), tau_w) * 1.1)
        ax_w.set_ylabel(f"WHERE p{k+1}", fontsize=6)
        ax_w.tick_params(labelsize=5); ax_w.set_xticks([])

        # HOW
        d_h = how_dists[k]; tau_h = how_taus[k]
        ax_h.plot(d_h[:t+1], color=col, lw=1.0, ls="--")
        ax_h.axhline(tau_h, color="k", lw=0.8, ls=":")
        ax_h.fill_between(np.arange(t+1), tau_h, d_h[:t+1],
                          where=d_h[:t+1] < tau_h, alpha=0.3, color=col)
        ax_h.axvline(t, color="gray", lw=0.7, ls=":")
        ax_h.set_xlim(0, T); ax_h.set_ylim(0, max(d_h.max(), tau_h) * 1.1)
        ax_h.set_ylabel(f"HOW p{k+1}", fontsize=6)
        ax_h.tick_params(labelsize=5)
        ax_h.set_xlabel("frame", fontsize=5) if k == K-1 else ax_h.set_xticks([])

    plt.tight_layout(pad=0.3)
    fig.canvas.draw()
    # tostring_rgb was renamed to buffer_rgba in newer matplotlib; handle both
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]
    plt.close(fig)
    # Resize to exactly (height, width)
    img = Image.fromarray(buf).resize((width, height), Image.LANCZOS)
    return np.array(img)


def render_progress_bar(
    t: int,
    T: int,
    where_dists: List[np.ndarray],
    where_taus: List[float],
    how_dists: List[np.ndarray],
    how_taus: List[float],
    width: int,
    height: int,
) -> np.ndarray:
    """Thin colour bar: filled = WHERE∧HOW active at that frame, colour = phase."""
    bar = np.ones((height, width, 3), dtype=np.uint8) * 220  # light gray bg
    K = len(where_dists)
    for k in range(K):
        col = np.array(_hex_to_rgb(PHASE_COLORS[k % len(PHASE_COLORS)]))
        active = (where_dists[k] < where_taus[k]) & (how_dists[k] < how_taus[k])
        row_s = int(k * height / K)
        row_e = int((k + 1) * height / K)
        for fr in range(T):
            x_s = int(fr * width / T)
            x_e = int((fr + 1) * width / T)
            if active[fr]:
                bar[row_s:row_e, x_s:x_e] = col
    # Cursor line
    cx = int(t * width / T)
    bar[:, max(0, cx - 1):cx + 2] = [50, 50, 50]
    return bar


def add_border(frame: np.ndarray, color_hex: str, thickness: int) -> np.ndarray:
    f = frame.copy()
    c = np.array(_hex_to_rgb(color_hex), dtype=np.uint8)
    f[:thickness, :] = c
    f[-thickness:, :] = c
    f[:, :thickness] = c
    f[:, -thickness:] = c
    return f


def label_strip(text: str, width: int, height: int,
                bg_hex: str = "#1a1a2e") -> np.ndarray:
    img = Image.new("RGB", (width, height),
                    color=tuple(_hex_to_rgb(bg_hex)))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(220, 220, 220), font=font)
    return np.array(img)


# ── per-episode video ─────────────────────────────────────────────────────────

def make_episode_video(
    ep_idx: int,
    task: str,
    frames: List[np.ndarray],
    where_dists: List[np.ndarray],
    where_taus: List[float],
    how_dists: List[np.ndarray],
    how_taus: List[float],
    phases_latent: List[Tuple[int, int]],
    out_path: Path,
    fps: int = 10,
    stride: int = 2,
):
    """Render one annotated mp4."""
    T = min(len(frames), len(where_dists[0]))
    K = len(phases_latent)

    # Decide video frame size
    sample = frames[0]
    fh, fw = sample.shape[:2]
    # scale video frame to max 360p
    scale = min(360 / fh, 480 / fw)
    th, tw = int(fh * scale), int(fw * scale)

    sig_w = SIGNAL_W
    sig_h = max(th + LABEL_H, SIGNAL_H)
    total_w = tw + sig_w
    total_h = sig_h + PROG_H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width  = total_w
    stream.height = total_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}

    frame_indices = list(range(0, T, stride))
    print(f"  Rendering {len(frame_indices)} frames → {out_path.name}")

    for fi, t in enumerate(frame_indices):
        # Current phase
        cur_phase = K - 1
        for k, (s, e) in enumerate(phases_latent):
            if s <= t < e:
                cur_phase = k
                break
        col_hex = PHASE_COLORS[cur_phase % len(PHASE_COLORS)]

        # Video frame
        vid_frame = Image.fromarray(frames[t]).resize((tw, th), Image.LANCZOS)
        vid_arr = np.array(vid_frame)

        # Phase border colour: brighten if WHERE∧HOW fires
        w_active = all(where_dists[k][t] < where_taus[k] for k in range(K))
        h_active = all(how_dists[k][t]   < how_taus[k]   for k in range(K))
        border_col = "#FFDD00" if (w_active and h_active) else col_hex
        vid_arr = add_border(vid_arr, border_col, BORDER)

        # Label strip above video
        where_vals = "  ".join(
            f"W{k+1}={'✓' if where_dists[k][t] < where_taus[k] else '✗'}" for k in range(K))
        how_vals   = "  ".join(
            f"H{k+1}={'✓' if how_dists[k][t] < how_taus[k] else '✗'}" for k in range(K))
        lbl = f"t={t:3d}/{T}  phase={cur_phase+1}  {where_vals}  {how_vals}"
        lbl_arr = label_strip(lbl, tw, LABEL_H)
        left_col = np.vstack([lbl_arr, vid_arr])
        # Pad/crop left column to sig_h
        if left_col.shape[0] < sig_h:
            pad = np.zeros((sig_h - left_col.shape[0], tw, 3), dtype=np.uint8)
            left_col = np.vstack([left_col, pad])
        else:
            left_col = left_col[:sig_h]

        # Signal panel (full time axis so user sees full curve, cursor = t)
        sig_arr = render_signal_panel(
            t, where_dists, where_taus, how_dists, how_taus,
            sig_h, sig_w)

        # Compose left + right
        canvas_top = np.hstack([left_col, sig_arr])

        # Progress bar (full width)
        prog_arr = render_progress_bar(
            t, T, where_dists, where_taus, how_dists, how_taus,
            total_w, PROG_H)

        canvas = np.vstack([canvas_top, prog_arr]).astype(np.uint8)

        av_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"    Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_key = CAM_KEYS[args.cam]

    # Load run metadata
    with open(run_dir / "spec_from_video.json") as f:
        meta = json.load(f)

    phases_latent = [(p["start"], p["end"]) for p in meta["phases_video"]]
    K = len(phases_latent)
    ref_ep = meta["reference_episode"]
    ref_task = meta["reference_task"]
    print(f"Loaded run: ref ep{ref_ep}, {K} phases, {len(meta['test_episodes'])} tests")

    # Load DROID metadata once
    import pandas as pd
    data_p = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet", repo_type="dataset")
    ep_p   = hf_hub_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    data_df = pd.read_parquet(data_p)
    ep_df   = pd.read_parquet(ep_p)

    # --- reference episode -------------------------------------------------------
    print(f"\n[Reference ep{ref_ep}]")
    ref_row = ep_df[ep_df["episode_index"] == ref_ep].iloc[0]
    ref_frames = load_frames(ref_row, cam_key)

    embs_path = run_dir / f"embs_ep{ref_ep:04d}.npz"
    if not embs_path.exists():
        print("  Embeddings not cached — re-encoding …")
        # Import encoder from main module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from etl_image_ablations.spec_from_video_droid import SVDVAEEncoder, DINOv2Encoder
        enc = SVDVAEEncoder(args.device) if meta["encoder"] == "svd_vae" else DINOv2Encoder(args.device)
        ref_embs = enc.encode(ref_frames)
        np.savez_compressed(embs_path, embs=ref_embs)
    else:
        ref_embs = np.load(embs_path)["embs"]
        print(f"  Loaded cached embeddings {ref_embs.shape}")

    # Build spec latents + trajectories from reference (identical logic to main script)
    HOW_WINDOW_MAX = 64
    spec_means, spec_trajs, phase_windows = [], [], []
    for k, (s, e) in enumerate(phases_latent):
        spec_means.append(ref_embs[s:e].mean(axis=0))
        win = min(e - s, HOW_WINDOW_MAX)
        mid = (s + e) // 2
        h_s = max(s, mid - win // 2); h_e = min(e, h_s + win)
        spec_trajs.append(subsample_traj(ref_embs[h_s:h_e], DTW_N_PTS))
        phase_windows.append(h_e - h_s)

    # WHERE + HOW distances on reference
    T_ref = len(ref_embs)
    ref_where_dists = [cosine_dist_vec(ref_embs, z) for z in spec_means]
    ref_how_dists   = [sliding_dtw(ref_embs, spec_trajs[k], phase_windows[k],
                                   step=max(1, phase_windows[k] // 8))
                       for k in range(K)]
    ref_where_taus  = [adaptive_tau(d, args.adapt_quantile) for d in ref_where_dists]
    ref_how_taus    = [adaptive_tau(d, args.adapt_quantile) for d in ref_how_dists]

    T_ref_vid = min(len(ref_frames), T_ref)
    make_episode_video(
        ref_ep, f"REF: {ref_task}",
        ref_frames[:T_ref_vid],
        [d[:T_ref_vid] for d in ref_where_dists],
        ref_where_taus,
        [d[:T_ref_vid] for d in ref_how_dists],
        ref_how_taus,
        phases_latent,
        out_dir / f"ref_ep{ref_ep:04d}.mp4",
        fps=args.fps, stride=args.stride,
    )

    # --- test episodes -----------------------------------------------------------
    for ep_info in meta["test_episodes"]:
        test_ep   = ep_info["episode"]
        test_task = ep_info["task"]
        print(f"\n[Test ep{test_ep}]  '{test_task[:60]}'")

        t_row = ep_df[ep_df["episode_index"] == test_ep]
        if t_row.empty:
            print("  SKIP: not in metadata"); continue
        t_row = t_row.iloc[0]

        t_frames = load_frames(t_row, cam_key)

        embs_path_t = run_dir / f"embs_ep{test_ep:04d}.npz"
        if not embs_path_t.exists():
            print("  Embeddings not cached — re-encoding …")
            if "enc" not in dir():
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from etl_image_ablations.spec_from_video_droid import SVDVAEEncoder, DINOv2Encoder
                enc = SVDVAEEncoder(args.device) if meta["encoder"] == "svd_vae" \
                    else DINOv2Encoder(args.device)
            t_embs = enc.encode(t_frames)
            np.savez_compressed(embs_path_t, embs=t_embs)
        else:
            t_embs = np.load(embs_path_t)["embs"]
            print(f"  Loaded cached embeddings {t_embs.shape}")

        T_t = min(len(t_frames), len(t_embs))
        t_where_dists = [cosine_dist_vec(t_embs, z) for z in spec_means]
        t_how_dists   = [sliding_dtw(t_embs, spec_trajs[k], phase_windows[k],
                                     step=max(1, phase_windows[k] // 8))
                         for k in range(K)]
        t_where_taus  = [adaptive_tau(d, args.adapt_quantile) for d in t_where_dists]
        t_how_taus    = [adaptive_tau(d, args.adapt_quantile) for d in t_how_dists]

        where_ok = ep_info.get("argmin_where_correct", False)
        how_ok   = ep_info.get("argmin_how_correct",   False)
        tag = "WHERE✓HOW✓" if (where_ok and how_ok) else \
              ("WHERE✓HOW✗" if where_ok else "WHERE✗")

        make_episode_video(
            test_ep, f"{tag} | {test_task}",
            t_frames[:T_t],
            [d[:T_t] for d in t_where_dists],
            t_where_taus,
            [d[:T_t] for d in t_how_dists],
            t_how_taus,
            phases_latent,
            out_dir / f"test_ep{test_ep:04d}_{tag.replace(' ','')}.mp4",
            fps=args.fps, stride=args.stride,
        )

    print(f"\nAll videos written to {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",  required=True)
    p.add_argument("--ref-ep",   type=int, required=True)
    p.add_argument("--cam",      choices=["wrist", "exterior1"], default="wrist")
    p.add_argument("--fps",      type=int,   default=10)
    p.add_argument("--stride",   type=int,   default=2,
                   help="render every Nth frame (stride=2 → 2× faster playback)")
    p.add_argument("--adapt-quantile", type=float, default=0.25)
    p.add_argument("--device",   default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
