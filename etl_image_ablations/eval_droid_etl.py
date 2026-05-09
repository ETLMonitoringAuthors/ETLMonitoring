"""
eval_droid_etl.py
-----------------
ETL (Embedding Temporal Logic) evaluation on DROID in-the-wild data
(lerobot/droid_100).

Ground truth predicates are derived from proprioception -- no manual labeling:
  π_grasp(t):  state[6] > GRASP_THRESH   (gripper closed)
  π_done(t):   last DONE_WINDOW frames    (near episode completion)

Primary evaluation: **frame-level F1 / precision / recall / agreement**
between ETL monitor output (d(z_t, z_spec) < τ) and GT mask per frame.

Conformal validity: does τ_CP achieve ≥ 90% recall on held-out test frames?

Camera choice:
  --cam wrist     → observation.images.wrist_image_left (best for π_grasp)
  --cam exterior1 → observation.images.exterior_image_1_left (scene-level)

This is intentionally harder than FMB: DROID episodes are in-the-wild,
diverse tasks, and the relationship between embedding distance and gripper
state is not guaranteed. That's the honest challenge.

Usage:
  cd /path/to/repo
  TMPDIR=/tmp HF_HOME=~/.cache/huggingface \\
  python -m etl_image_ablations.eval_droid_etl \\
      --encoder dino --cam wrist --pred grasp \\
      --out-dir etl_results/droid_dino_wrist_grasp

  # Done predicate, exterior camera
  python -m etl_image_ablations.eval_droid_etl \\
      --encoder dino --cam exterior1 --pred done \\
      --out-dir etl_results/droid_dino_ext_done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import av
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

# ─── Constants ────────────────────────────────────────────────────────────────

REPO_ID      = "lerobot/droid_100"
GRASP_THRESH = 0.5    # gripper value > this = closed (grasping)
DONE_WINDOW  = 5      # last N frames = GT-positive for π_done
CAL_FRAC     = 0.40
ALPHA        = 0.10
N_TAU        = 300

CAM_KEYS = {
    "wrist":     "observation.images.wrist_image_left",
    "exterior1": "observation.images.exterior_image_1_left",
    "exterior2": "observation.images.exterior_image_2_left",
}

# ─── Encoders ─────────────────────────────────────────────────────────────────

class DINOv2Encoder:
    """Frozen DINOv2 ViT-B/14 → 768-D CLS token."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, device="cuda"):
        self.device = device
        print("Loading DINOv2 ViT-B/14 …")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
            pretrained=True, verbose=False,
        ).to(device).eval()
        self.mean = self.MEAN.to(device)
        self.std  = self.STD.to(device)

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size=32) -> np.ndarray:
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [
                (TF.resize(torch.from_numpy(f).permute(2,0,1).float().div(255.).to(self.device),
                           [224,224], antialias=True) - self.mean) / self.std
                for f in frames[i:i+batch_size]
            ]
            out.append(self.model(torch.stack(batch)).cpu().numpy())
        return np.concatenate(out, axis=0)


class CLIPEncoder:
    """Frozen CLIP ViT-B/32 image encoder → 512-D embedding (L2-normalised)."""
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading CLIP ViT-B/32 …")
        import clip as openai_clip
        self.model, self.preprocess = openai_clip.load("ViT-B/32", device=device)
        self.model.eval()
        print("  CLIP loaded.")

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size=64) -> np.ndarray:
        from PIL import Image as PILImage
        out = []
        for i in range(0, len(frames), batch_size):
            imgs = [self.preprocess(PILImage.fromarray(f)).to(self.device)
                    for f in frames[i:i + batch_size]]
            feats = self.model.encode_image(torch.stack(imgs)).float().cpu().numpy()
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
            out.append(feats)
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        import clip as openai_clip
        tokens = openai_clip.tokenize([text]).to(self.device)
        feat = self.model.encode_text(tokens).float().cpu().numpy()[0]
        feat = feat / max(np.linalg.norm(feat), 1e-8)
        return feat


class SVDVAEEncoder:
    """
    SVD VAE (AutoencoderKLTemporalDecoder) used in Ctrl-World preprocessing.
    Encodes each frame → 4×24×40 spatial latent → L2-normalised flat vector (3840-D).
    Trained on DROID data as part of the Ctrl-World pipeline.
    """
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading SVD VAE from stabilityai/stable-video-diffusion-img2vid …")
        from diffusers.models import AutoencoderKLTemporalDecoder
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            subfolder="vae",
            torch_dtype=torch.float16,
            cache_dir="/tmp",
        ).to(device).eval()
        self.vae.requires_grad_(False)
        print("  SVD VAE loaded.")

    @torch.no_grad()
    def encode(self, frames: List[np.ndarray], batch_size=16) -> np.ndarray:
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [
                TF.resize(torch.from_numpy(f).permute(2,0,1).float().div(255.),
                          [192,320], antialias=True) * 2.0 - 1.0
                for f in frames[i:i+batch_size]
            ]
            x = torch.stack(batch).to(self.device, dtype=torch.float16)
            z = self.vae.encode(x).latent_dist.sample().float().cpu()  # (B,4,H/8,W/8)
            out.append(z.reshape(len(batch), -1).numpy())              # (B, 3840)
        return np.concatenate(out, axis=0)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_meta():
    info_p = hf_hub_download(REPO_ID, "meta/info.json",    repo_type="dataset")
    data_p = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet", repo_type="dataset")
    ep_p   = hf_hub_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    with open(info_p) as f:
        info = json.load(f)
    return info, pd.read_parquet(data_p), pd.read_parquet(ep_p)


def decode_mp4(path, t0, t1):
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


def load_frames(ep_row, cam_key):
    chunk = int(ep_row[f"videos/{cam_key}/chunk_index"])
    fidx  = int(ep_row[f"videos/{cam_key}/file_index"])
    t0    = float(ep_row[f"videos/{cam_key}/from_timestamp"])
    t1    = float(ep_row[f"videos/{cam_key}/to_timestamp"])
    mp4   = f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    path  = hf_hub_download(REPO_ID, mp4, repo_type="dataset")
    return decode_mp4(path, t0, t1)


def get_state(data_df, ep_idx):
    rows = data_df[data_df["episode_index"] == ep_idx]
    return np.stack(rows.reset_index(drop=True)["observation.state"].values)


# ─── GT masks ─────────────────────────────────────────────────────────────────

OPEN_THRESH = 0.15   # gripper < this after a grasp = release event

def make_grasp_gt(state: np.ndarray) -> np.ndarray:
    """True when gripper is closed (holding). state[:,6] = gripper (0=open, 1=closed)."""
    return state[:, 6] > GRASP_THRESH


def make_release_gt(state: np.ndarray) -> np.ndarray:
    """True when gripper is open AFTER a prior grasp event (i.e., object released)."""
    gripper = state[:, 6]
    had_grasp = np.zeros(len(gripper), dtype=bool)
    seen = False
    for i, g in enumerate(gripper):
        if g > GRASP_THRESH:
            seen = True
        had_grasp[i] = seen
    # Release = open gripper that follows at least one grasping frame
    return (gripper < OPEN_THRESH) & had_grasp


def is_sequential_episode(state: np.ndarray) -> bool:
    """True if episode has a grasp-then-release transition (multi-phase pick-and-place)."""
    gt_hold    = make_grasp_gt(state)
    gt_release = make_release_gt(state)
    if not gt_hold.any() or not gt_release.any():
        return False
    # The first release must come AFTER the first grasp
    first_grasp   = np.argmax(gt_hold)
    first_release = np.argmax(gt_release)
    return first_release > first_grasp


def make_done_gt(T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    m[-DONE_WINDOW:] = True
    return m


# ─── Distance + threshold mining ──────────────────────────────────────────────

def l2_dist(E, z):
    return np.linalg.norm(E - z[None], axis=1)


def cosine_dist(E, z):
    En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(np.linalg.norm(z), 1e-8)
    return 1.0 - (En @ zn)


def mine_f1(dists, gt):
    taus = np.linspace(dists.min(), dists.max(), N_TAU)
    best_f1, best_tau = 0.0, np.median(taus)
    for tau in taus:
        pred = dists < tau
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2*p*r/(p+r+1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return float(best_tau), float(best_f1)


def mine_cp(dists, gt, alpha=ALPHA):
    pos = dists[gt]
    if len(pos) == 0:
        return float(dists.max())
    q = min(1.0, (1-alpha)*(1 + 1/len(pos)))
    return float(np.quantile(pos, q))


def evaluate(dists, gt, tau):
    pred = dists < tau
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum(); tn = (~pred & ~gt).sum()
    p  = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2*p*r/(p+r+1e-9)
    agree = float((pred == gt).mean())
    return {"f1": float(f1), "precision": float(p), "recall": float(r),
            "agreement": agree, "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_key = CAM_KEYS[args.cam]
    pred_name = args.pred  # "grasp" or "done"
    use_cosine = (args.encoder in ("svd_vae", "clip"))

    print(f"Encoder={args.encoder}  Camera={args.cam}  Predicate={pred_name}")
    if args.encoder == "clip":
        print(f"Text query: '{args.text_query}'")

    # Load data
    _, data_df, ep_df = load_meta()

    # Filter episodes with non-empty task text
    ep_df = ep_df.copy()
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else ""
    )
    eps = ep_df[ep_df["task_text"].str.len() > 0].copy().reset_index(drop=True)

    if args.num_episodes:
        eps = eps.iloc[:args.num_episodes].copy()

    n_cal = max(5, int(len(eps) * CAL_FRAC))
    cal_eps  = eps.iloc[:n_cal]
    test_eps = eps.iloc[n_cal:].reset_index(drop=True)
    print(f"Total={len(eps)}  Cal={len(cal_eps)}  Test={len(test_eps)}")

    # Build encoder
    device = args.device
    if args.encoder == "dino":
        enc = DINOv2Encoder(device)
        dist_fn = l2_dist
    elif args.encoder == "clip":
        enc = CLIPEncoder(device)
        dist_fn = cosine_dist
    else:
        enc = SVDVAEEncoder(device)
        dist_fn = cosine_dist

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n=== Calibration ===")
    cal_embs, cal_gts = [], []

    # For CLIP, z_spec comes from the text query; skip positive-frame averaging.
    if args.encoder == "clip":
        z_spec = enc.encode_text(args.text_query)
        print(f"Spec latent (text): shape={z_spec.shape}  query='{args.text_query}'")
    else:
        spec_pos_vecs = []

    for _, row in cal_eps.iterrows():
        ep_idx = int(row["episode_index"])
        task   = row["task_text"]
        try:
            frames = load_frames(row, cam_key)
            state  = get_state(data_df, ep_idx)
        except Exception as e:
            print(f"  ep{ep_idx}: SKIP ({e})")
            continue
        T = min(len(frames), len(state))
        if T < 10:
            continue
        frames, state = frames[:T], state[:T]

        gt = make_grasp_gt(state) if pred_name == "grasp" else make_done_gt(T)
        n_pos = gt.sum()
        print(f"  ep{ep_idx:3d}  T={T}  pos={n_pos}  task='{task[:40]}'")

        embs = enc.encode(frames)
        cal_embs.append(embs)
        cal_gts.append(gt)
        if args.encoder != "clip" and n_pos > 0:
            spec_pos_vecs.append(embs[gt])

    if args.encoder != "clip":
        if not spec_pos_vecs:
            print("No calibration positives found — check GRASP_THRESH or pred choice.")
            sys.exit(1)
        z_spec = np.concatenate(spec_pos_vecs).mean(axis=0)
        print(f"\nSpec latent: shape={z_spec.shape}")

    cal_d  = np.concatenate([dist_fn(e, z_spec) for e in cal_embs])
    cal_gt = np.concatenate(cal_gts)

    tau_f1, cal_f1 = mine_f1(cal_d, cal_gt)
    tau_cp = mine_cp(cal_d, cal_gt)

    cal_metrics_f1 = evaluate(cal_d, cal_gt, tau_f1)
    cal_metrics_cp = evaluate(cal_d, cal_gt, tau_cp)

    print(f"τ_F1={tau_f1:.4f}  cal_F1={cal_metrics_f1['f1']:.3f}  "
          f"P={cal_metrics_f1['precision']:.3f}  R={cal_metrics_f1['recall']:.3f}  "
          f"agree={cal_metrics_f1['agreement']:.3f}")
    print(f"τ_CP={tau_cp:.4f}  cal_F1={cal_metrics_cp['f1']:.3f}  "
          f"P={cal_metrics_cp['precision']:.3f}  R={cal_metrics_cp['recall']:.3f}  "
          f"agree={cal_metrics_cp['agreement']:.3f}")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n=== Test ===")
    results_f1, results_cp = [], []

    for _, row in test_eps.iterrows():
        ep_idx = int(row["episode_index"])
        task   = row["task_text"]
        try:
            frames = load_frames(row, cam_key)
            state  = get_state(data_df, ep_idx)
        except Exception as e:
            print(f"  ep{ep_idx}: SKIP ({e})")
            continue
        T = min(len(frames), len(state))
        if T < 10:
            continue
        frames, state = frames[:T], state[:T]

        gt   = make_grasp_gt(state) if pred_name == "grasp" else make_done_gt(T)
        embs = enc.encode(frames)
        d    = dist_fn(embs, z_spec)

        rf1 = evaluate(d, gt, tau_f1)
        rcp = evaluate(d, gt, tau_cp)
        results_f1.append(rf1)
        results_cp.append(rcp)

        print(f"  ep{ep_idx:3d}  T={T}  pos={gt.sum():3d}  "
              f"F1_f1={rf1['f1']:.3f}  F1_cp={rcp['f1']:.3f}  "
              f"Rec_cp={rcp['recall']:.3f}  agree_f1={rf1['agreement']:.3f}")

        _plot_timeline(out_dir, ep_idx, task, d, gt, tau_f1, tau_cp, pred_name)
        _save_monitoring_video(out_dir, ep_idx, task, frames, d, gt,
                               tau_f1, tau_cp, pred_name)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def avg(lst, k): return float(np.mean([r[k] for r in lst]))

    n_test = len(results_f1)
    print(f"\n{'='*50}")
    print(f"RESULTS  encoder={args.encoder}  cam={args.cam}  pred={pred_name}")
    print(f"N test episodes: {n_test}")
    print(f"F1  (τ_F1): {avg(results_f1,'f1'):.3f}  "
          f"P={avg(results_f1,'precision'):.3f}  R={avg(results_f1,'recall'):.3f}  "
          f"agree={avg(results_f1,'agreement'):.3f}")
    print(f"F1  (τ_CP): {avg(results_cp,'f1'):.3f}  "
          f"P={avg(results_cp,'precision'):.3f}  R={avg(results_cp,'recall'):.3f}  "
          f"agree={avg(results_cp,'agreement'):.3f}")
    # Check conformal validity: did τ_CP hit ≥(1-α) recall on average?
    mean_rec_cp = avg(results_cp, "recall")
    target = 1 - ALPHA
    print(f"Conformal target: recall ≥ {target:.0%}  |  achieved: {mean_rec_cp:.3f}  "
          f"{'✓' if mean_rec_cp >= target else '✗ MISS'}")

    metrics = {
        "encoder": args.encoder,
        "camera": args.cam,
        "predicate": pred_name,
        "n_cal": n_cal,
        "n_test": n_test,
        "grasp_thresh": GRASP_THRESH if pred_name == "grasp" else None,
        "done_window": DONE_WINDOW if pred_name == "done" else None,
        "tau_f1": tau_f1,
        "tau_cp": tau_cp,
        "cal": {"f1": cal_metrics_f1, "cp": cal_metrics_cp},
        "test_tau_F1": {k: avg(results_f1, k) for k in ["f1","precision","recall","agreement"]},
        "test_tau_CP": {k: avg(results_cp, k) for k in ["f1","precision","recall","agreement"]},
        "conformal_recall_achieved": mean_rec_cp,
        "conformal_recall_target": target,
        "conformal_valid": bool(mean_rec_cp >= target),
    }

    mpath = out_dir / "metrics.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {mpath}")


def _save_monitoring_video(out_dir, ep_idx, task, frames, d, gt,
                           tau_f1, tau_cp, pred_name, fps=15, step=2):
    """
    Live-monitoring video: left = camera frame with status border,
    right = running distance plot + GT/pred ribbon.

    Uses PIL pixel drawing (no per-frame matplotlib) so it's fast even for
    long episodes. The GT ribbon background is pre-rendered once; only the
    growing distance curve and cursor update each frame.
    """
    from PIL import Image, ImageDraw

    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    T       = len(d)
    pred_cp = d < tau_cp
    pred_f1 = d < tau_f1

    # ── Layout constants ──────────────────────────────────────────────────────
    CAM_W, CAM_H = 480, 360
    PLT_W, PLT_H = 640, 360
    OUT_W, OUT_H = CAM_W + PLT_W, CAM_H

    # Distance-plot inner area (top 60% of right panel)
    ML, MR, MT, MB = 55, 10, 35, 10   # margins: left, right, top, bottom
    DIST_H = int(PLT_H * 0.58)        # height of distance subplot
    PLOT_X0 = ML
    PLOT_X1 = PLT_W - MR
    PLOT_Y0 = MT
    PLOT_Y1 = DIST_H - MB
    PLOT_IW = PLOT_X1 - PLOT_X0
    PLOT_IH = PLOT_Y1 - PLOT_Y0

    # Ribbon area (bottom 35% of right panel, two rows)
    RIB_Y0   = DIST_H + 8
    RIB_MID  = RIB_Y0 + (PLT_H - RIB_Y0) // 2 - 4
    RIB_Y1   = PLT_H - 4

    # ── Coordinate helpers ────────────────────────────────────────────────────
    d_min = d.min(); d_max = d.max()
    d_rng = max(d_max - d_min, 1e-6)

    # Clip tau lines inside visible range with a small margin
    D_LO = d_min - 0.05 * d_rng
    D_HI = d_max + 0.05 * d_rng

    def tx(t_):  return PLOT_X0 + int(t_ / max(T - 1, 1) * PLOT_IW)
    def dy(d_):  return PLOT_Y1 - int(np.clip((d_ - D_LO) / (D_HI - D_LO), 0, 1) * PLOT_IH)
    def rx(t_):  return PLOT_X0 + int(t_ / T * PLOT_IW)  # ribbon x (0-based)

    y_f1 = int(np.clip(dy(tau_f1), PLOT_Y0, PLOT_Y1))
    y_cp = int(np.clip(dy(tau_cp), PLOT_Y0, PLOT_Y1))

    # ── Palette ───────────────────────────────────────────────────────────────
    C = {
        "bg":         (245, 245, 245),
        "plot_bg":    (255, 255, 255),
        "curve":      (30, 100, 200),
        "tau_f1":     (200, 30, 30),
        "tau_cp":     (210, 130, 0),
        "cursor":     (0, 0, 0),
        "gt_shade":   (173, 216, 230),
        # ribbon
        "pred_pos":   (50, 200, 50),
        "pred_neg":   (200, 50, 50),
        "gt_pos":     (60, 120, 200),
        "gt_neg":     (220, 160, 50),
        # status border
        "border_pos": (0, 220, 0),
        "border_neg": (220, 0, 0),
        "gt_border_pos": (60, 120, 200),
        "gt_border_neg": (180, 100, 0),
    }

    # ── Pre-render static background (GT shading + axes + thresholds + ribbon)
    bg = Image.new("RGB", (PLT_W, PLT_H), C["bg"])
    draw_bg = ImageDraw.Draw(bg)

    # Plot area background
    draw_bg.rectangle([PLOT_X0, PLOT_Y0, PLOT_X1, PLOT_Y1], fill=C["plot_bg"])

    # GT shading stripes
    rb_w = max(1, PLOT_IW // T)
    for i in range(T):
        if gt[i]:
            x0 = tx(i); x1 = min(x0 + rb_w + 1, PLOT_X1)
            draw_bg.rectangle([x0, PLOT_Y0, x1, PLOT_Y1], fill=C["gt_shade"])

    # Axis box
    draw_bg.rectangle([PLOT_X0, PLOT_Y0, PLOT_X1, PLOT_Y1], outline=(150,150,150), width=1)

    # Threshold lines (dashed via short segments)
    for x in range(PLOT_X0, PLOT_X1, 8):
        draw_bg.line([(x, y_f1), (min(x+5, PLOT_X1), y_f1)], fill=C["tau_f1"], width=2)
        draw_bg.line([(x, y_cp), (min(x+5, PLOT_X1), y_cp)], fill=C["tau_cp"], width=2)
    draw_bg.text((PLOT_X1 + 2, y_f1 - 9), "τ_F1", fill=C["tau_f1"])
    draw_bg.text((PLOT_X1 + 2, y_cp - 9), "τ_CP", fill=C["tau_cp"])

    # Ribbon rows: full static GT ribbon (we know all GT upfront)
    for i in range(T):
        rx0 = rx(i); rx1 = min(rx0 + rb_w + 1, PLOT_X1)
        draw_bg.rectangle([rx0, RIB_Y0,  rx1, RIB_MID], fill=C["pred_pos"] if pred_cp[i] else C["pred_neg"])
        draw_bg.rectangle([rx0, RIB_MID+2, rx1, RIB_Y1], fill=C["gt_pos"] if gt[i] else C["gt_neg"])

    # Ribbon labels
    draw_bg.text((2, RIB_Y0  + (RIB_MID - RIB_Y0)//2 - 6),  "τ_CP", fill=(60, 60, 60))
    draw_bg.text((2, RIB_MID + (RIB_Y1 - RIB_MID)//2 - 6),  " GT",  fill=(60, 60, 60))

    # Y-axis tick labels
    for frac in [0.0, 0.5, 1.0]:
        d_val = D_LO + frac * (D_HI - D_LO)
        y_pix = dy(d_val)
        draw_bg.text((2, y_pix - 6), f"{d_val:.2f}", fill=(80, 80, 80))

    bg_arr = np.array(bg)

    # ── Write video ────────────────────────────────────────────────────────────
    out_path = str(vid_dir / f"ep{ep_idx:04d}.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width  = OUT_W
    stream.height = OUT_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    frame_indices = list(range(0, T, step))

    for t in frame_indices:
        # ── Left: camera frame with status border ─────────────────────────────
        cam = np.array(Image.fromarray(frames[t]).resize((CAM_W, CAM_H)))
        bw = 10
        bc = C["border_pos"] if pred_cp[t] else C["border_neg"]
        gc = C["gt_border_pos"] if gt[t] else C["gt_border_neg"]
        cam[:bw, :]  = bc; cam[-bw:, :] = bc
        cam[:, :bw]  = bc; cam[:, -bw:] = bc
        cam[bw:bw*2, bw:-bw] = gc
        cam[-bw*2:-bw, bw:-bw] = gc

        # ── Right: copy pre-rendered background, draw growing curve + cursor ──
        frame_img = Image.fromarray(bg_arr.copy())
        draw = ImageDraw.Draw(frame_img)

        # Distance curve up to t
        pts = [(tx(i), dy(d[i])) for i in range(t + 1)]
        if len(pts) > 1:
            draw.line(pts, fill=C["curve"], width=2)
        elif len(pts) == 1:
            draw.ellipse([pts[0][0]-2, pts[0][1]-2, pts[0][0]+2, pts[0][1]+2], fill=C["curve"])

        # Cursor: vertical line at current t
        cx = tx(t); rx_cur = rx(t)
        draw.line([(cx, PLOT_Y0 - 4), (cx, PLOT_Y1 + 4)], fill=C["cursor"], width=2)
        draw.line([(rx_cur, RIB_Y0 - 4), (rx_cur, RIB_Y1 + 4)], fill=C["cursor"], width=2)

        # Overlay text
        status = "GRASPING" if pred_cp[t] else "open"
        gt_txt = "GT:yes" if gt[t] else "GT:no"
        draw.text((PLOT_X0, 2), f"π_{pred_name}  d={d[t]:.3f}  {status}  {gt_txt}  t={t}", fill=(30,30,30))
        draw.text((PLOT_X0, 14), task[:70], fill=(80, 80, 80))

        plt_arr = np.array(frame_img)

        combined = np.hstack([cam, plt_arr])
        av_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"    Video: {out_path}")


def _plot_timeline(out_dir, ep_idx, task, d, gt, tau_f1, tau_cp, pred_name):
    tl = out_dir / "timelines"
    tl.mkdir(exist_ok=True)
    T = len(d)
    t = np.arange(T)
    pred_f1 = d < tau_f1
    pred_cp = d < tau_cp

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 0.5, 0.5]})
    fig.suptitle(f"ep{ep_idx}  [{pred_name}]  {task[:65]}", fontsize=8)

    ax = axes[0]
    ax.plot(t, d, "b-", lw=1, label=f"dist(z_t, z_{pred_name})")
    ax.axhline(tau_f1, color="red",    ls="--", lw=1.2, label=f"τ_F1={tau_f1:.3f}")
    ax.axhline(tau_cp, color="orange", ls="-.", lw=1.2, label=f"τ_CP={tau_cp:.3f}")
    # GT shading
    ax.fill_between(t, ax.get_ylim()[0], d.max()*1.1, where=gt,
                    color="steelblue", alpha=0.15, label="GT positive")
    ax.set_ylabel("Distance")
    ax.legend(fontsize=7, ncol=2)

    for ax_i, (pred, label) in zip(axes[1:], [(pred_f1, "τ_F1"), (pred_cp, "τ_CP")]):
        for i in range(T):
            c_lat = "green" if pred[i]  else "red"
            c_gt  = "steelblue" if gt[i] else "orange"
            ax_i.axvspan(i, i+1, ymin=0.5, ymax=1.0, color=c_lat, alpha=0.85)
            ax_i.axvspan(i, i+1, ymin=0.0, ymax=0.5, color=c_gt,  alpha=0.85)
        ax_i.set_yticks([0.25, 0.75])
        ax_i.set_yticklabels(["GT", label], fontsize=7)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fig.savefig(tl / f"ep{ep_idx:04d}.png", dpi=100)
    plt.close(fig)


def _save_sequential_video(out_dir, ep_idx, task, frames,
                           d_hold, d_release, gt_hold, gt_release,
                           tau_hold_f1, tau_hold_cp,
                           tau_rel_f1,  tau_rel_cp):
    """
    Two-predicate live-monitoring video for sequential pick-and-place.
    Left: wrist camera with 2-row status strip (hold / release).
    Right: two running distance plots + 4 ribbon rows.
    """
    from PIL import Image, ImageDraw

    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    T = len(d_hold)
    pred_hold_cp    = d_hold    < tau_hold_cp
    pred_release_cp = d_release < tau_rel_cp

    CAM_W, CAM_H = 480, 360
    PLT_W, PLT_H = 700, 360
    OUT_W, OUT_H = CAM_W + PLT_W, CAM_H

    # Each predicate subplot occupies half the right panel height
    HALF = PLT_H // 2
    ML, MR, MT, MB = 55, 8, 18, 6

    def subplot_coords(row):  # row 0=hold, 1=release
        y0 = row * HALF + MT
        y1 = (row + 1) * HALF - MB - 20   # leave room for ribbon
        rib_y0 = y1 + 4
        rib_y1 = (row + 1) * HALF - 2
        return y0, y1, rib_y0, rib_y1

    def make_coords(d, row):
        y0, y1, rib_y0, rib_y1 = subplot_coords(row)
        ih = y1 - y0
        d_lo = d.min() - 0.05 * max(d.max() - d.min(), 1e-6)
        d_hi = d.max() + 0.05 * max(d.max() - d.min(), 1e-6)
        d_rng = max(d_hi - d_lo, 1e-6)
        def tx(t_): return ML + int(t_ / max(T-1, 1) * (PLT_W - ML - MR))
        def dy(d_): return y1 - int(np.clip((d_ - d_lo) / d_rng, 0, 1) * ih)
        def rx(t_): return ML + int(t_ / T * (PLT_W - ML - MR))
        return tx, dy, rx, y0, y1, rib_y0, rib_y1, d_lo, d_hi

    C = {
        "bg":       (245, 245, 245), "plot_bg": (255, 255, 255),
        "hold":     (30, 100, 200),  "release": (180, 50, 180),
        "tau_f1":   (200, 30, 30),   "tau_cp":  (210, 130, 0),
        "cursor":   (0, 0, 0),       "gt_shade": (173, 216, 230),
        "p_pos":    (50, 200, 50),   "p_neg":   (200, 50, 50),
        "gt_pos":   (60, 120, 200),  "gt_neg":  (220, 160, 50),
        "b_hold":   (30, 150, 30),   "b_rel":   (150, 30, 150),
        "b_neg":    (200, 50, 50),   "b_gt_p":  (60, 120, 200),
        "b_gt_n":   (180, 100, 0),
    }

    # Pre-render static background
    bg = Image.new("RGB", (PLT_W, PLT_H), C["bg"])
    draw_bg = ImageDraw.Draw(bg)

    specs = [
        (d_hold,    gt_hold,    pred_hold_cp,    0, "hold",    C["hold"]),
        (d_release, gt_release, pred_release_cp, 1, "release", C["release"]),
    ]

    for d_arr, gt_arr, pred_arr, row, label, color in specs:
        tx, dy, rx, y0, y1, rib_y0, rib_y1, d_lo, d_hi = make_coords(d_arr, row)
        rb_w = max(1, (PLT_W - ML - MR) // T)

        draw_bg.rectangle([ML, y0, PLT_W - MR, y1], fill=C["plot_bg"])
        # GT shading
        for i in range(T):
            if gt_arr[i]:
                x0 = tx(i); x1 = min(x0 + rb_w + 1, PLT_W - MR)
                draw_bg.rectangle([x0, y0, x1, y1], fill=C["gt_shade"])
        draw_bg.rectangle([ML, y0, PLT_W - MR, y1], outline=(150,150,150), width=1)

        # Pre-draw full ribbon
        rib_mid = (rib_y0 + rib_y1) // 2
        for i in range(T):
            rx0 = rx(i); rx1 = min(rx0 + rb_w + 1, PLT_W - MR)
            draw_bg.rectangle([rx0, rib_y0, rx1, rib_mid-1],
                              fill=C["p_pos"] if pred_arr[i] else C["p_neg"])
            draw_bg.rectangle([rx0, rib_mid, rx1, rib_y1],
                              fill=C["gt_pos"] if gt_arr[i] else C["gt_neg"])

        # Threshold lines (dashed)
        tau_f1_val = tau_hold_f1 if row == 0 else tau_rel_f1
        tau_cp_val = tau_hold_cp if row == 0 else tau_rel_cp
        y_f1 = int(np.clip(dy(tau_f1_val), y0, y1))
        y_cp = int(np.clip(dy(tau_cp_val), y0, y1))
        for x in range(ML, PLT_W - MR, 8):
            draw_bg.line([(x, y_f1), (min(x+5, PLT_W-MR), y_f1)], fill=C["tau_f1"], width=1)
            draw_bg.line([(x, y_cp), (min(x+5, PLT_W-MR), y_cp)], fill=C["tau_cp"], width=1)

        # Label
        draw_bg.text((2, y0 + 2), f"π_{label}", fill=color)
        draw_bg.text((2, rib_y0), "Pred", fill=(60, 60, 60))
        draw_bg.text((2, rib_mid), " GT",  fill=(60, 60, 60))

    bg_arr = np.array(bg)

    # Video writer
    out_path = str(vid_dir / f"ep{ep_idx:04d}.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=15)
    stream.width = OUT_W; stream.height = OUT_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    # Sequential satisfaction check at each t
    def seq_satisfied_at(t):
        for t1 in np.where(pred_hold_cp[:t+1])[0]:
            if np.any(pred_release_cp[t1+1:t+1]):
                return True
        return False

    for t in range(0, T, 2):
        # Camera frame with 2-row status border
        cam = np.array(Image.fromarray(frames[t]).resize((CAM_W, CAM_H)))
        bw = 8
        # Top border = hold status, bottom border = release status
        hc = C["b_hold"] if pred_hold_cp[t] else C["b_neg"]
        rc = C["b_rel"]  if pred_release_cp[t] else C["b_neg"]
        cam[:bw,  :] = hc;  cam[bw:bw*2, :] = C["b_gt_p"] if gt_hold[t] else C["b_gt_n"]
        cam[-bw:, :] = rc;  cam[-bw*2:-bw, :] = C["b_gt_p"] if gt_release[t] else C["b_gt_n"]
        # Sequential check indicator: left strip
        seq_ok = seq_satisfied_at(t)
        cam[:, :bw] = (0, 220, 0) if seq_ok else (120, 120, 120)

        # Right panel: copy bg, draw growing curves + cursor
        frame_img = Image.fromarray(bg_arr.copy())
        draw = ImageDraw.Draw(frame_img)

        for d_arr, row, color in [(d_hold, 0, C["hold"]), (d_release, 1, C["release"])]:
            tx, dy, rx, y0, y1, rib_y0, rib_y1, _, _ = make_coords(d_arr, row)
            pts = [(tx(i), dy(d_arr[i])) for i in range(t + 1)]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
            elif pts:
                p = pts[0]
                draw.ellipse([p[0]-2, p[1]-2, p[0]+2, p[1]+2], fill=color)
            # Cursor
            cx = tx(t); rx_c = ML + int(t / T * (PLT_W - ML - MR))
            draw.line([(cx, y0-3), (cx, y1+3)], fill=C["cursor"], width=2)
            draw.line([(rx_c, rib_y0-2), (rx_c, rib_y1+2)], fill=C["cursor"], width=2)

        # Header
        seq_txt = "SEQ ✓" if seq_ok else "SEQ …"
        draw.text((ML, 1), f"t={t}  {seq_txt}  hold={'Y' if pred_hold_cp[t] else 'n'}  "
                           f"rel={'Y' if pred_release_cp[t] else 'n'}", fill=(30,30,30))
        draw.text((ML, 11), task[:75], fill=(80, 80, 80))

        combined = np.hstack([cam, np.array(frame_img)])
        av_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"    Video: {out_path}")


def run_sequential(args):
    """
    Sequential pick-and-place evaluation on DROID.
    Filters to episodes with a genuine grasp→release transition.
    Evaluates π_hold and π_release independently, then checks
    the sequential spec: ∃ t1 < t2 s.t. π_hold(t1) ∧ π_release(t2).
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_key = CAM_KEYS[args.cam]
    _, data_df, ep_df = load_meta()

    ep_df = ep_df.copy()
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else ""
    )
    eps = ep_df[ep_df["task_text"].str.len() > 0].copy().reset_index(drop=True)

    # Filter to multi-phase episodes using proprioception
    print("Filtering to sequential (grasp→release) episodes …")
    seq_eps = []
    for _, row in eps.iterrows():
        try:
            state = get_state(data_df, int(row["episode_index"]))
            if len(state) >= 10 and is_sequential_episode(state):
                seq_eps.append(row)
        except Exception:
            pass
    seq_eps = pd.DataFrame(seq_eps).reset_index(drop=True)
    print(f"Sequential episodes: {len(seq_eps)} / {len(eps)}")

    if len(seq_eps) < 5:
        print("Too few sequential episodes — lower thresholds or use more data.")
        return

    if args.num_episodes:
        seq_eps = seq_eps.iloc[:args.num_episodes].copy()

    n_cal = max(4, int(len(seq_eps) * CAL_FRAC))
    cal_eps  = seq_eps.iloc[:n_cal]
    test_eps = seq_eps.iloc[n_cal:].reset_index(drop=True)
    print(f"Cal={len(cal_eps)}  Test={len(test_eps)}")

    device = args.device
    if args.encoder == "dino":
        enc = DINOv2Encoder(device); dist_fn = l2_dist
    else:
        enc = SVDVAEEncoder(device); dist_fn = cosine_dist

    # ── Calibration ────────────────────────────────────────────────────────────
    print("\n=== Calibration ===")
    cal_embs, cal_hold_gts, cal_rel_gts = [], [], []
    hold_pos_vecs, rel_pos_vecs = [], []

    for _, row in cal_eps.iterrows():
        ep_idx = int(row["episode_index"])
        try:
            frames = load_frames(row, cam_key)
            state  = get_state(data_df, ep_idx)
        except Exception as e:
            print(f"  ep{ep_idx}: SKIP ({e})"); continue
        T = min(len(frames), len(state))
        if T < 10: continue
        frames, state = frames[:T], state[:T]

        gt_hold    = make_grasp_gt(state)
        gt_release = make_release_gt(state)
        embs       = enc.encode(frames)

        print(f"  ep{ep_idx:3d}  T={T}  hold={gt_hold.sum()}  release={gt_release.sum()}  "
              f"'{row['task_text'][:40]}'")

        cal_embs.append(embs)
        cal_hold_gts.append(gt_hold)
        cal_rel_gts.append(gt_release)
        if gt_hold.sum() > 0:    hold_pos_vecs.append(embs[gt_hold])
        if gt_release.sum() > 0: rel_pos_vecs.append(embs[gt_release])

    z_hold    = np.concatenate(hold_pos_vecs).mean(0)
    z_release = np.concatenate(rel_pos_vecs).mean(0)
    print(f"\nz_hold={z_hold.shape}  z_release={z_release.shape}")

    cal_d_hold = np.concatenate([dist_fn(e, z_hold)    for e in cal_embs])
    cal_d_rel  = np.concatenate([dist_fn(e, z_release) for e in cal_embs])
    cal_gt_hold = np.concatenate(cal_hold_gts)
    cal_gt_rel  = np.concatenate(cal_rel_gts)

    tau_hold_f1, _ = mine_f1(cal_d_hold, cal_gt_hold)
    tau_hold_cp    = mine_cp(cal_d_hold, cal_gt_hold)
    tau_rel_f1,  _ = mine_f1(cal_d_rel,  cal_gt_rel)
    tau_rel_cp     = mine_cp(cal_d_rel,  cal_gt_rel)

    print(f"τ_hold:   F1={tau_hold_f1:.4f}  CP={tau_hold_cp:.4f}")
    print(f"τ_release: F1={tau_rel_f1:.4f}  CP={tau_rel_cp:.4f}")

    # ── Test ───────────────────────────────────────────────────────────────────
    print("\n=== Test ===")
    res_hold_f1, res_hold_cp = [], []
    res_rel_f1,  res_rel_cp  = [], []
    seq_agree_list = []

    for _, row in test_eps.iterrows():
        ep_idx = int(row["episode_index"])
        task   = row["task_text"]
        try:
            frames = load_frames(row, cam_key)
            state  = get_state(data_df, ep_idx)
        except Exception as e:
            print(f"  ep{ep_idx}: SKIP ({e})"); continue
        T = min(len(frames), len(state))
        if T < 10: continue
        frames, state = frames[:T], state[:T]

        gt_hold    = make_grasp_gt(state)
        gt_release = make_release_gt(state)
        embs       = enc.encode(frames)
        d_hold     = dist_fn(embs, z_hold)
        d_release  = dist_fn(embs, z_release)

        rh_f1 = evaluate(d_hold,    gt_hold,    tau_hold_f1)
        rh_cp = evaluate(d_hold,    gt_hold,    tau_hold_cp)
        rr_f1 = evaluate(d_release, gt_release, tau_rel_f1)
        rr_cp = evaluate(d_release, gt_release, tau_rel_cp)
        res_hold_f1.append(rh_f1); res_hold_cp.append(rh_cp)
        res_rel_f1.append(rr_f1);  res_rel_cp.append(rr_cp)

        # Sequential spec: ∃ t1 < t2  s.t.  pred_hold(t1) ∧ pred_release(t2)
        pred_hold_cp    = d_hold    < tau_hold_cp
        pred_release_cp = d_release < tau_rel_cp
        gt_seq_sat   = gt_hold.any() and gt_release.any() and \
                       (np.argmax(gt_release) > np.argmax(gt_hold))
        lat_seq_sat  = False
        for t1 in np.where(pred_hold_cp)[0]:
            if np.any(pred_release_cp[t1+1:]):
                lat_seq_sat = True; break
        seq_agree = lat_seq_sat == gt_seq_sat
        seq_agree_list.append(seq_agree)

        print(f"  ep{ep_idx:3d}  T={T}  "
              f"hold_F1={rh_f1['f1']:.3f}  rel_F1={rr_f1['f1']:.3f}  "
              f"hold_agree={rh_f1['agreement']:.3f}  "
              f"seq={'✓' if seq_agree else '✗'}  '{task[:35]}'")

        _plot_timeline(out_dir, ep_idx, task, d_hold, gt_hold,
                       tau_hold_f1, tau_hold_cp, "hold")
        _save_sequential_video(out_dir, ep_idx, task, frames,
                               d_hold, d_release, gt_hold, gt_release,
                               tau_hold_f1, tau_hold_cp,
                               tau_rel_f1,  tau_rel_cp)

    def avg(lst, k): return float(np.mean([r[k] for r in lst]))
    n_test = len(res_hold_f1)
    seq_agreement = float(np.mean(seq_agree_list)) if seq_agree_list else 0.0

    print(f"\n{'='*55}")
    print(f"SEQUENTIAL ETL  encoder={args.encoder}  cam={args.cam}")
    print(f"Episodes:  cal={len(cal_eps)}  test={n_test}")
    print(f"π_hold   F1(τ_F1)={avg(res_hold_f1,'f1'):.3f}  "
          f"P={avg(res_hold_f1,'precision'):.3f}  R={avg(res_hold_f1,'recall'):.3f}  "
          f"agree={avg(res_hold_f1,'agreement'):.3f}")
    print(f"π_hold   F1(τ_CP)={avg(res_hold_cp,'f1'):.3f}  "
          f"R={avg(res_hold_cp,'recall'):.3f}")
    print(f"π_release F1(τ_F1)={avg(res_rel_f1,'f1'):.3f}  "
          f"P={avg(res_rel_f1,'precision'):.3f}  R={avg(res_rel_f1,'recall'):.3f}  "
          f"agree={avg(res_rel_f1,'agreement'):.3f}")
    print(f"π_release F1(τ_CP)={avg(res_rel_cp,'f1'):.3f}  "
          f"R={avg(res_rel_cp,'recall'):.3f}")
    print(f"Sequential spec agree:  {seq_agreement:.3f}  "
          f"({sum(seq_agree_list)}/{len(seq_agree_list)})")
    mean_rec_hold = avg(res_hold_cp, "recall")
    mean_rec_rel  = avg(res_rel_cp,  "recall")
    print(f"Conformal recall  hold={mean_rec_hold:.3f}  release={mean_rec_rel:.3f}  "
          f"(target ≥ {1-ALPHA:.0%})")

    metrics = {
        "encoder": args.encoder, "camera": args.cam, "mode": "sequential",
        "n_cal": len(cal_eps), "n_test": n_test,
        "tau_hold_f1": tau_hold_f1, "tau_hold_cp": tau_hold_cp,
        "tau_rel_f1": tau_rel_f1,   "tau_rel_cp":  tau_rel_cp,
        "hold_test_F1":  {k: avg(res_hold_f1, k) for k in ["f1","precision","recall","agreement"]},
        "hold_test_CP":  {k: avg(res_hold_cp, k) for k in ["f1","precision","recall","agreement"]},
        "rel_test_F1":   {k: avg(res_rel_f1,  k) for k in ["f1","precision","recall","agreement"]},
        "rel_test_CP":   {k: avg(res_rel_cp,  k) for k in ["f1","precision","recall","agreement"]},
        "sequential_agreement": seq_agreement,
        "sequential_n": len(seq_agree_list),
        "sequential_n_agree": sum(seq_agree_list),
    }
    mpath = out_dir / "metrics_sequential.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {mpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",      choices=["dino", "svd_vae", "clip"], default="svd_vae")
    p.add_argument("--text-query",   default="a robot gripper is gripping an object",
                   help="Text description for CLIP-based predicate (only used with --encoder clip)")
    p.add_argument("--cam",          choices=["wrist","exterior1","exterior2"], default="wrist")
    p.add_argument("--pred",         choices=["grasp","done"], default="grasp")
    p.add_argument("--mode",         choices=["single","sequential"], default="single",
                   help="single: one predicate eval; sequential: grasp→release multi-phase")
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--out-dir",      default="etl_results/droid_sequential")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.mode == "sequential":
        run_sequential(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
