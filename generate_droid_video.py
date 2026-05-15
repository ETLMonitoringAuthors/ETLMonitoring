"""
Generate ETL monitoring GIF for DROID episode 13 (pick-and-place).

Uses exterior camera for display (3rd-person view) and wrist camera
for DINOv2 embedding. Two-phase ETL spec:
  z_0 = mean embedding during gripper-close window 0 (pick up object)
  z_1 = mean embedding during gripper-close window 1 (place on paper)

F(near_0 ∧ F(near_1)) with F1-optimal thresholds.

Usage:
    conda activate newt
    HF_HOME=/usr0/parvk/hf_cache2 python generate_droid_video.py \
        --episode 13 --out-dir assets
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import av
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_HOME", "/usr0/parvk/hf_cache2")
os.environ.setdefault("TMPDIR",  "/usr0/parvk/tmp")

REPO_ID = "lerobot/droid_100"

CLOSE_THRESH     = 0.45
OPEN_THRESH      = 0.20
MIN_CLOSE_FRAMES = 5

BLUE1    = "#2563EB"
BLUE2    = "#7C3AED"
GT_COLOR = "#64748B"
COLORS   = [BLUE1, BLUE2]

# ── data loading ──────────────────────────────────────────────────────────────

def _load_episode_row(ep_idx: int):
    import pandas as pd
    ep_p = hf_hub_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet",
                           repo_type="dataset")
    ep_df = pd.read_parquet(ep_p)
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else "")
    return ep_df[ep_df["episode_index"] == ep_idx].iloc[0]


def _decode_video(mp4_path: str, t0: float, t1: float) -> list[np.ndarray]:
    frames = []
    with av.open(mp4_path) as c:
        c.seek(int(t0 * 1000000))
        for frm in c.decode(video=0):
            ts = float(frm.pts * frm.time_base)
            if ts < t0 - 0.001:
                continue
            if ts > t1 + 0.001:
                break
            frames.append(frm.to_ndarray(format="rgb24"))
    return frames


def _load_frames(row, cam_key: str) -> list[np.ndarray]:
    chunk = int(row[f"videos/{cam_key}/chunk_index"])
    fidx  = int(row[f"videos/{cam_key}/file_index"])
    t0    = float(row[f"videos/{cam_key}/from_timestamp"])
    t1    = float(row[f"videos/{cam_key}/to_timestamp"])
    mp4   = f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    path  = hf_hub_download(REPO_ID, mp4, repo_type="dataset")
    return _decode_video(path, t0, t1)


def _load_state(ep_idx: int) -> np.ndarray:
    import pandas as pd
    data_p = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet",
                             repo_type="dataset")
    df = pd.read_parquet(data_p)
    ep_rows = df[df["episode_index"] == ep_idx].sort_values("frame_index")
    state = np.stack(ep_rows["observation.state"].tolist())
    return state


# ── gripper segmentation ──────────────────────────────────────────────────────

def segment_phases(gripper: np.ndarray) -> list[tuple[int, int]]:
    T = len(gripper)
    in_close = False
    windows: list[tuple[int, int]] = []
    start = 0
    for t in range(T):
        g = gripper[t]
        if not in_close and g > CLOSE_THRESH:
            in_close = True; start = t
        elif in_close and g < OPEN_THRESH:
            in_close = False
            if t - start >= MIN_CLOSE_FRAMES:
                windows.append((start, t))
    if in_close and T - start >= MIN_CLOSE_FRAMES:
        windows.append((start, T))
    return windows


# ── DINOv2 encoder ────────────────────────────────────────────────────────────

class DINOv2Encoder:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
            pretrained=True, verbose=False,
        ).to(device).eval()
        self.mean = self.MEAN.to(device)
        self.std  = self.STD.to(device)

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(frames), batch_size):
            batch = []
            for f in frames[i : i + batch_size]:
                img = Image.fromarray(f).resize((224, 224), Image.LANCZOS)
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                batch.append((t - self.mean.cpu()) / self.std.cpu())
            x = torch.stack(batch).to(self.device)
            out.append(self.model(x).cpu().numpy())
        return np.concatenate(out, axis=0)


# ── ETL helpers ───────────────────────────────────────────────────────────────

def cosine_dist(E: np.ndarray, z: np.ndarray) -> np.ndarray:
    En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(float(np.linalg.norm(z)), 1e-8)
    return 1.0 - (En @ zn)


def window_tau(dists: np.ndarray, gt: np.ndarray, pct: float = 50.0) -> float:
    """tau = pct-th percentile of distances during GT-positive frames (tight threshold)."""
    pos_dists = dists[gt]
    if len(pos_dists) == 0:
        return float(np.median(dists))
    return float(np.percentile(pos_dists, pct))


# ── panel rendering (same style as sorting/MetaWorld) ─────────────────────────

def _panel(t: int, T: int,
           dist_traces: list[np.ndarray],
           taus: list[float],
           pred_traces: list[np.ndarray],
           gt_traces: list[np.ndarray],
           labels: list[str],
           sim_width: int = 480) -> np.ndarray:
    K = len(dist_traces)
    ratios = [2.0] * K + [0.7] * K
    fig, axes = plt.subplots(K + K, 1, figsize=(sim_width / 100, 3.2),
                             gridspec_kw={"height_ratios": ratios})
    ts = np.arange(T)
    d_ylims = [max(tr.max() * 1.15, taus[k] * 1.3) for k, tr in enumerate(dist_traces)]

    for k in range(K):
        ax = axes[k]
        ax.set_xlim(0, T - 1); ax.set_ylim(-0.05, d_ylims[k])
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.plot(ts[:t+1], dist_traces[k][:t+1], color=COLORS[k], lw=1.5)
        ax.axhline(taus[k], color="red", ls="--", lw=1.2,
                   label=rf"$\tau_{k}$={taus[k]:.2f}")
        ax.set_ylabel(rf"dist$(z_t,z_{k})$" + f"\n({labels[k]})", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticklabels([])
        ax.tick_params(labelsize=8)

    for k in range(K):
        ax = axes[K + k]
        ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([f"GT {k+1}", f"pred {k+1}"], fontsize=8)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.grid(False)
        for j in range(t):
            if pred_traces[k][j]:
                ax.barh(1.5, 1, left=j, height=0.55, color=COLORS[k], align="center")
            if gt_traces[k][j]:
                ax.barh(0.5, 1, left=j, height=0.55, color=GT_COLOR, align="center")
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Timestep", fontsize=9)
    axes[-1].tick_params(labelsize=8)
    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    panel = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    panel = np.array(Image.fromarray(panel).resize((sim_width, 320), Image.LANCZOS))
    return panel


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=13)
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--fps",     type=int, default=10)
    ap.add_argument("--step",    type=int, default=2,
                    help="Frame subsampling (show every Nth frame)")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    print(f"Loading episode {args.episode} from DROID…")
    row   = _load_episode_row(args.episode)
    task  = row["task_text"]
    print(f"  Task: {task}")

    wrist_key = "observation.images.wrist_image_left"
    ext_key   = "observation.images.exterior_image_1_left"

    print("  Loading frames…")
    wrist_frames = _load_frames(row, wrist_key)
    ext_frames   = _load_frames(row, ext_key)
    state        = _load_state(args.episode)

    T = min(len(wrist_frames), len(ext_frames), len(state))
    wrist_frames = wrist_frames[:T]
    ext_frames   = ext_frames[:T]
    state        = state[:T]
    print(f"  T={T} frames  state shape={state.shape}")

    print("  Segmenting phases…")
    gripper = state[:, 6]
    windows = segment_phases(gripper)
    print(f"  {len(windows)} gripper-close windows: {windows}")
    if len(windows) < 2:
        print("  Warning: fewer than 2 phases found; using first half / second half split")
        windows = [(0, T // 2), (T // 2, T)]

    K = min(len(windows), 2)
    windows = windows[:K]

    print(f"  Encoding wrist frames with DINOv2 on {args.device}…")
    enc  = DINOv2Encoder(device=args.device)
    embs = enc.encode(wrist_frames)  # [T, D]

    # spec latents: mean of wrist embeddings during each gripper-close window
    spec_latents = []
    for s, e in windows:
        spec_latents.append(embs[s:e].mean(axis=0))
    spec_latents = np.stack(spec_latents)  # [K, D]

    dist_traces = [cosine_dist(embs, spec_latents[k]) for k in range(K)]

    # GT: True during each phase's gripper-close window
    gt_traces = []
    for s, e in windows:
        gt = np.zeros(T, dtype=bool)
        gt[s:e] = True
        gt_traces.append(gt)

    # threshold = 85th pct of distances during GT window (tight, avoids early firing)
    taus = [window_tau(dist_traces[k], gt_traces[k]) for k in range(K)]
    print(f"  taus: {[f'{t:.3f}' for t in taus]}")

    # Point-in-time pred: near spec latent k right now (not accumulated)
    pred_traces = [
        (dist_traces[k] <= taus[k]).astype(float)
        for k in range(K)
    ]

    # verify F1 on episode
    for k in range(K):
        pred = pred_traces[k].astype(bool)
        gt   = gt_traces[k]
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        print(f"  phase {k}: F1={2*p*r/(p+r+1e-9):.3f}  P={p:.3f}  R={r:.3f}")

    labels = ["pick", "place"][:K]

    # exterior frames → resize to 480×270 (16:9) then letterbox to 480×480
    CAM_W = 480
    scaled_h = CAM_W * 180 // 320  # 270
    pad_top  = (CAM_W - scaled_h) // 2   # 105
    pad_bot  = CAM_W - scaled_h - pad_top

    def _letterbox(f: np.ndarray) -> np.ndarray:
        img = np.array(Image.fromarray(f).resize((CAM_W, scaled_h), Image.LANCZOS))
        top = np.zeros((pad_top, CAM_W, 3), dtype=np.uint8)
        bot = np.zeros((pad_bot, CAM_W, 3), dtype=np.uint8)
        return np.vstack([top, img, bot])  # 480×480

    ext_resized = [_letterbox(f) for f in ext_frames]

    print(f"  Compositing {T} frames (step={args.step})…", flush=True)
    display_indices = list(range(0, T, args.step))
    composite_frames = []
    for i, t in enumerate(display_indices):
        sim  = ext_resized[t]
        pan  = _panel(t, T, dist_traces, taus, pred_traces,
                      [gt.astype(float) for gt in gt_traces], labels, CAM_W)
        composite_frames.append(Image.fromarray(np.vstack([sim, pan])))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(display_indices)}", flush=True)

    gif_path = out / "droid_monitoring.gif"
    composite_frames[0].save(
        gif_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=False,
    )
    print(f"  Saved {gif_path}  ({composite_frames[0].size})")
    print(f"  Task: {task}")
    for k in range(K):
        first_t = next((t for t, v in enumerate(pred_traces[k]) if v), None)
        print(f"  phase {k} first pred at t={first_t}  (GT window {windows[k]})")


if __name__ == "__main__":
    main()
