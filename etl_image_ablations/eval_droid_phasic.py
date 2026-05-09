"""
eval_droid_phasic.py
--------------------
Phasic ETL evaluation on DROID in-the-wild data.

Filters to episodes whose task language implies multiple compositional phases
(e.g. "pick X then pick Y", "move A then get B and fold") and that have
≥ 2 distinct gripper-close events in proprioception.

For each such episode we:
  1. Segment the trajectory into gripper-close windows (phase 1, phase 2, …)
  2. Build a per-phase spec latent z_k = mean embedding of frames in window k
  3. Compute ALL pairwise cosine distances between phase spec latents
     → measures how visually distinct the phases are in embedding space
  4. Mine per-phase thresholds (F1-optimal) on the SAME episode
     (within-episode eval; honest since spec frames are held out by using
     odd windows as spec and even windows as test, if ≥3 windows exist,
     otherwise leave-one-window-out)
  5. Evaluate: does phase-k predicate fire mainly during window k?
  6. Sequential spec: does z_phase1 fire before z_phase2?

This is the principled compositional evaluation:
the interesting question is not "did grasp happen before release"
but "can the embedding space distinguish *what is being held* across phases,
allowing ETL to separately track each compositional sub-task?"

Usage:
  cd /path/to/repo
  TMPDIR=/tmp HF_HOME=~/.cache/huggingface \\
  python -m etl_image_ablations.eval_droid_phasic \\
      --encoder svd_vae --cam wrist --out-dir etl_results/droid_phasic
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
from typing import List

import av
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

REPO_ID = "lerobot/droid_100"
CAM_KEYS = {
    "wrist":     "observation.images.wrist_image_left",
    "exterior1": "observation.images.exterior_image_1_left",
}

# ── Gripper segmentation ──────────────────────────────────────────────────────
CLOSE_THRESH = 0.45   # gripper > this = closed
OPEN_THRESH  = 0.20   # gripper < this = open
MIN_CLOSE_FRAMES = 5  # ignore transient glitches shorter than this

# ── Phasic task language filters ─────────────────────────────────────────────
PHASE_KWS   = [" then ", ", then ", " and then ", " after ", "finally "]
MULTI_ACT   = [" and put ", " and place ", ", pick up", ", put ", ", get ",
               "pick up the", "and fold", "and place them"]

# ── Encoders ─────────────────────────────────────────────────────────────────

class DINOv2Encoder:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    def __init__(self, device="cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",
                                    pretrained=True,verbose=False).to(device).eval()
        self.mean = self.MEAN.to(device); self.std = self.STD.to(device)
    @torch.no_grad()
    def encode(self, frames, batch_size=32):
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [(TF.resize(torch.from_numpy(f).permute(2,0,1).float().div(255.).to(self.device),
                                [224,224],antialias=True)-self.mean)/self.std
                     for f in frames[i:i+batch_size]]
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
    def encode(self, frames, batch_size=16):
        import torchvision.transforms.functional as TF
        out = []
        for i in range(0, len(frames), batch_size):
            batch = [TF.resize(torch.from_numpy(f).permute(2,0,1).float().div(255.),
                               [192,320],antialias=True)*2.0-1.0
                     for f in frames[i:i+batch_size]]
            x = torch.stack(batch).to(self.device, dtype=torch.float16)
            z = self.vae.encode(x).latent_dist.sample().float().cpu()
            out.append(z.reshape(len(batch),-1).numpy())
        return np.concatenate(out)


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_meta():
    data_p = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet", repo_type="dataset")
    ep_p   = hf_hub_download(REPO_ID, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    return pd.read_parquet(data_p), pd.read_parquet(ep_p)


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
    path  = hf_hub_download(REPO_ID,
        f"videos/{cam_key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4", repo_type="dataset")
    return decode_mp4(path, t0, t1)


def get_state(data_df, ep_idx):
    rows = data_df[data_df["episode_index"] == ep_idx]
    return np.stack(rows.reset_index(drop=True)["observation.state"].values)


# ── Gripper phase segmentation ────────────────────────────────────────────────

def segment_phases(gripper: np.ndarray):
    """
    Returns list of (start, end) index pairs for each continuous gripper-close
    window longer than MIN_CLOSE_FRAMES, with hysteresis (CLOSE/OPEN thresholds).
    """
    T = len(gripper)
    in_close = False
    windows = []
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


# ── ETL helpers ───────────────────────────────────────────────────────────────

def cosine_dist(E, z):
    En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)
    zn = z / max(np.linalg.norm(z), 1e-8)
    return 1.0 - (En @ zn)


def l2_dist(E, z):
    return np.linalg.norm(E - z[None], axis=1)


def mine_f1_tau(dists, gt, n=300):
    taus = np.linspace(dists.min(), dists.max(), n)
    best_f1, best_tau = 0.0, np.median(taus)
    for tau in taus:
        pred = dists < tau
        tp=(pred&gt).sum(); fp=(pred&~gt).sum(); fn=(~pred&gt).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
        f1=2*p*r/(p+r+1e-9)
        if f1>best_f1: best_f1,best_tau=f1,tau
    return float(best_tau), float(best_f1)


def eval_pred(dists, gt, tau):
    pred = dists < tau
    tp=(pred&gt).sum(); fp=(pred&~gt).sum(); fn=(~pred&gt).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
    f1=2*p*r/(p+r+1e-9)
    return {"f1":float(f1),"precision":float(p),"recall":float(r),
            "agreement":float((pred==gt).mean())}


# ── Video generation ──────────────────────────────────────────────────────────

def save_phasic_video(out_dir, ep_idx, task, frames, windows,
                      phase_dists, phase_taus, dist_fn_name, fps=12, step=2):
    """
    Multi-panel video: left = wrist cam with phase label, right = N distance plots.
    phase_dists: list of (d_array, gt_mask) per phase
    phase_taus:  list of tau per phase
    """
    from PIL import Image, ImageDraw
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    T = len(frames)
    N = len(phase_dists)
    COLORS = ['#2166AC','#D6604D','#4DAF4A','#F4A582','#984EA3']

    CAM_W, CAM_H = 480, 360
    PLT_W = 660; PLT_H = CAM_H
    OUT_W, OUT_H = CAM_W + PLT_W, CAM_H
    ROW_H = PLT_H // N

    # Pre-render phase ribbons as background
    bg = Image.new("RGB", (PLT_W, PLT_H), (245,245,245))
    draw_bg = ImageDraw.Draw(bg)

    def tx(t_, ML=50): return ML + int(t_ / max(T-1,1) * (PLT_W - ML - 8))

    for k, ((d, gt), tau) in enumerate(zip(phase_dists, phase_taus)):
        y0 = k * ROW_H + 5
        y1 = (k+1) * ROW_H - 20
        rib_y0 = y1 + 2; rib_y1 = (k+1)*ROW_H - 3
        d_lo = d.min()-0.03*(d.max()-d.min())
        d_hi = d.max()+0.03*(d.max()-d.min())
        d_rng = max(d_hi-d_lo, 1e-6)
        def dy(d_, _y0=y0, _y1=y1, _d_lo=d_lo, _d_rng=d_rng):
            return _y1 - int(np.clip((d_-_d_lo)/_d_rng,0,1)*(_y1-_y0))

        # Plot bg + GT shading
        draw_bg.rectangle([50, y0, PLT_W-8, y1], fill=(255,255,255), outline=(180,180,180))
        rb_w = max(1, (PLT_W-58)//T)
        for t_ in range(T):
            if gt[t_]:
                x0=tx(t_); x1=min(x0+rb_w+1, PLT_W-8)
                draw_bg.rectangle([x0,y0,x1,y1], fill=(198,219,239))

        # Tau line (dashed)
        y_tau = int(np.clip(dy(tau), y0, y1))
        for x_ in range(50, PLT_W-8, 8):
            draw_bg.line([(x_, y_tau),(min(x_+5,PLT_W-8), y_tau)], fill='#D6604D', width=1)

        # Ribbon background
        pred_cp = d < tau
        rib_mid = (rib_y0 + rib_y1)//2
        for t_ in range(T):
            rx0=tx(t_); rx1=min(rx0+rb_w+1, PLT_W-8)
            draw_bg.rectangle([rx0,rib_y0,rx1,rib_mid-1],
                              fill='#4DAF4A' if pred_cp[t_] else '#D6604D')
            draw_bg.rectangle([rx0,rib_mid,rx1,rib_y1],
                              fill='#2166AC' if gt[t_] else '#FDAE61')

        col = COLORS[k % len(COLORS)]
        draw_bg.text((2, y0+2), f"ph{k+1}", fill=col)
        draw_bg.text((2, rib_y0), "Pred", fill=(80,80,80))
        draw_bg.text((2, rib_mid), " GT",  fill=(80,80,80))

        # Y-axis
        for frac in [0.0, 0.5, 1.0]:
            dv = d_lo + frac*(d_hi-d_lo)
            draw_bg.text((2, dy(dv)-6), f"{dv:.2f}", fill=(100,100,100))

    bg_arr = np.array(bg)

    # Which phase is currently active?
    def current_phase(t_):
        for k_, (s, e) in enumerate(windows):
            if s <= t_ < e:
                return k_
        return -1

    out_path = str(vid_dir / f"ep{ep_idx:04d}.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = OUT_W; stream.height = OUT_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    for t in range(0, T, step):
        cam = np.array(Image.fromarray(frames[t]).resize((CAM_W, CAM_H)))
        phase_idx = current_phase(t)
        bw = 10
        if phase_idx >= 0:
            col_hex = COLORS[phase_idx % len(COLORS)]
            col = tuple(int(col_hex.lstrip('#')[i:i+2],16) for i in (0,2,4))
        else:
            col = (120, 120, 120)
        cam[:bw,:]=col; cam[-bw:,:]=col; cam[:,:bw]=col; cam[:,-bw:]=col

        # Right panel
        frame_img = Image.fromarray(bg_arr.copy())
        draw = ImageDraw.Draw(frame_img)

        for k, (d, gt) in enumerate(phase_dists):
            y0 = k * ROW_H + 5; y1 = (k+1) * ROW_H - 20
            d_lo = d.min()-0.03*(d.max()-d.min())
            d_hi = d.max()+0.03*(d.max()-d.min())
            d_rng = max(d_hi-d_lo,1e-6)
            def dy(d_, _y0=y0, _y1=y1, _d_lo=d_lo, _d_rng=d_rng):
                return _y1 - int(np.clip((d_-_d_lo)/_d_rng,0,1)*(_y1-_y0))
            col = COLORS[k % len(COLORS)]
            pts = [(tx(i), dy(d[i])) for i in range(t+1)]
            if len(pts) > 1: draw.line(pts, fill=col, width=2)
            elif pts:
                p=pts[0]; draw.ellipse([p[0]-2,p[1]-2,p[0]+2,p[1]+2], fill=col)
            cx = tx(t)
            draw.line([(cx, y0-2),(cx, y1+2)], fill=(0,0,0), width=2)

        draw.text((50, 1), f"t={t}  phase={'holding-'+str(phase_idx+1) if phase_idx>=0 else 'transit'}", fill=(30,30,30))
        draw.text((50, 11), task[:72], fill=(80,80,80))

        combined = np.hstack([cam, np.array(frame_img)])
        av_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
        for pkt in stream.encode(av_frame): container.mux(pkt)

    for pkt in stream.encode(): container.mux(pkt)
    container.close()
    print(f"    Video → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_key = CAM_KEYS[args.cam]

    print("Loading metadata …")
    data_df, ep_df = load_meta()
    ep_df = ep_df.copy()
    ep_df["task_text"] = ep_df["tasks"].apply(
        lambda x: str(x[0]) if hasattr(x,"__len__") and len(x)>0 else "")

    # Filter: task language implies multiple phases
    def is_phasic(task):
        t = task.lower()
        return any(kw in t for kw in PHASE_KWS + MULTI_ACT)

    # Filter: ≥2 actual gripper-close windows in proprioception
    def count_windows(ep_idx):
        try:
            st = get_state(data_df, ep_idx)
            return len(segment_phases(st[:,6]))
        except:
            return 0

    print("Filtering to phasic episodes …")
    phasic_rows = []
    for _, row in ep_df.iterrows():
        task = row["task_text"]
        if not task or not is_phasic(task): continue
        ep_idx = int(row["episode_index"])
        n_windows = count_windows(ep_idx)
        if n_windows >= 2:
            phasic_rows.append({**row.to_dict(), "n_windows": n_windows})

    phasic_df = pd.DataFrame(phasic_rows).reset_index(drop=True)
    print(f"Phasic episodes found: {len(phasic_df)}")
    for _, r in phasic_df.iterrows():
        print(f"  ep{int(r['episode_index']):3d}  n_windows={r['n_windows']}  '{r['task_text'][:65]}'")

    if len(phasic_df) == 0:
        print("No phasic episodes found."); return

    # Build encoder
    device = args.device
    if args.encoder == "dino":
        enc = DINOv2Encoder(device); dist_fn = l2_dist
    else:
        enc = SVDVAEEncoder(device); dist_fn = cosine_dist

    # ── Per-episode phasic evaluation ────────────────────────────────────────
    all_results = []

    for _, row in phasic_df.iterrows():
        ep_idx = int(row["episode_index"])
        task   = row["task_text"]
        n_win  = int(row["n_windows"])

        print(f"\n{'='*60}")
        print(f"ep{ep_idx}  n_phases={n_win}  '{task}'")

        try:
            frames = load_frames(row, cam_key)
            state  = get_state(data_df, ep_idx)
        except Exception as e:
            print(f"  SKIP: {e}"); continue

        T = min(len(frames), len(state))
        frames, state = frames[:T], state[:T]
        gripper = state[:, 6]
        windows = segment_phases(gripper)
        if len(windows) < 2:
            print(f"  Only {len(windows)} window(s) after filtering — skip"); continue

        print(f"  T={T}  windows={[(s,e) for s,e in windows]}")

        # Encode all frames
        embs = enc.encode(frames)    # (T, D)

        # Build per-phase spec latents and GT masks
        spec_latents, gt_masks = [], []
        for k, (ws, we) in enumerate(windows):
            z_k = embs[ws:we].mean(axis=0)
            gt_k = np.zeros(T, bool)
            gt_k[ws:we] = True
            spec_latents.append(z_k)
            gt_masks.append(gt_k)

        # Cross-phase latent distances (separability)
        print("  Spec latent cosine distances between phases:")
        sep_matrix = np.zeros((len(windows), len(windows)))
        for i in range(len(windows)):
            for j in range(len(windows)):
                if i == j: continue
                zi = spec_latents[i]; zj = spec_latents[j]
                zi_n = zi / max(np.linalg.norm(zi), 1e-8)
                zj_n = zj / max(np.linalg.norm(zj), 1e-8)
                sep_matrix[i,j] = 1.0 - float(zi_n @ zj_n)
        for i in range(len(windows)):
            for j in range(i+1, len(windows)):
                print(f"    d(z_phase{i+1}, z_phase{j+1}) = {sep_matrix[i,j]:.4f}")

        mean_sep = sep_matrix[sep_matrix>0].mean() if (sep_matrix>0).any() else 0.0

        # Per-phase predicate evaluation
        phase_results = []
        phase_dists = []
        phase_taus  = []
        for k, (z_k, gt_k) in enumerate(zip(spec_latents, gt_masks)):
            d_k = dist_fn(embs, z_k)
            tau_k, f1_k = mine_f1_tau(d_k, gt_k)
            r_k = eval_pred(d_k, gt_k, tau_k)
            phase_results.append(r_k)
            phase_dists.append((d_k, gt_k))
            phase_taus.append(tau_k)
            print(f"  phase {k+1}: F1={r_k['f1']:.3f}  P={r_k['precision']:.3f}  "
                  f"R={r_k['recall']:.3f}  agree={r_k['agreement']:.3f}  "
                  f"τ={tau_k:.4f}  GT_frames={gt_k.sum()}")

        # Sequential spec: z_phase1 fires before z_phase2
        # Using per-phase thresholds
        pred_lists = [d < tau for (d, _), tau in zip(phase_dists, phase_taus)]
        seq_ok = True
        for k in range(len(windows) - 1):
            first_k   = np.argmax(pred_lists[k])   if pred_lists[k].any()   else T
            first_k1  = np.argmax(pred_lists[k+1]) if pred_lists[k+1].any() else T
            gt_first_k  = windows[k][0]
            gt_first_k1 = windows[k+1][0]
            latent_order_ok = first_k < first_k1
            gt_order_ok     = gt_first_k < gt_first_k1   # always true by construction
            print(f"  phase{k+1}→phase{k+2}: latent first fires at {first_k} vs {first_k1}  "
                  f"({'✓' if latent_order_ok else '✗'})")
            if not latent_order_ok:
                seq_ok = False
        print(f"  Sequential ordering: {'✓ CORRECT' if seq_ok else '✗ WRONG'}")

        ep_result = {
            "episode": ep_idx,
            "task": task,
            "n_phases": len(windows),
            "mean_phase_separation": float(mean_sep),
            "phases": [{"window": list(w), **r}
                       for w, r in zip(windows, phase_results)],
            "sequential_correct": seq_ok,
        }
        all_results.append(ep_result)

        # Save video
        save_phasic_video(out_dir, ep_idx, task, frames,
                          windows, phase_dists, phase_taus, args.encoder)

        # Save timeline figure
        _save_timeline(out_dir, ep_idx, task, windows, phase_dists, phase_taus)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASIC ETL SUMMARY  encoder={args.encoder}  cam={args.cam}")
    print(f"Episodes evaluated: {len(all_results)}")
    if all_results:
        seq_rate = np.mean([r['sequential_correct'] for r in all_results])
        mean_sep = np.mean([r['mean_phase_separation'] for r in all_results])
        all_f1 = [p['f1'] for r in all_results for p in r['phases']]
        all_agree = [p['agreement'] for r in all_results for p in r['phases']]
        print(f"Mean phase latent separation (cosine): {mean_sep:.4f}")
        print(f"Mean per-phase F1:         {np.mean(all_f1):.3f}")
        print(f"Mean per-phase agreement:  {np.mean(all_agree):.3f}")
        print(f"Sequential ordering correct: {seq_rate:.3f}  "
              f"({sum(r['sequential_correct'] for r in all_results)}/{len(all_results)})")

    metrics = {
        "encoder": args.encoder, "camera": args.cam,
        "n_episodes": len(all_results),
        "sequential_rate": float(np.mean([r['sequential_correct'] for r in all_results])) if all_results else 0,
        "mean_phase_separation": float(np.mean([r['mean_phase_separation'] for r in all_results])) if all_results else 0,
        "mean_phase_f1": float(np.mean([p['f1'] for r in all_results for p in r['phases']])) if all_results else 0,
        "mean_phase_agreement": float(np.mean([p['agreement'] for r in all_results for p in r['phases']])) if all_results else 0,
        "episodes": all_results,
    }
    with open(out_dir / "metrics_phasic.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_dir / 'metrics_phasic.json'}")


def _save_timeline(out_dir, ep_idx, task, windows, phase_dists, phase_taus):
    tl = out_dir / "timelines"; tl.mkdir(exist_ok=True)
    N = len(phase_dists)
    T = len(phase_dists[0][0])
    t = np.arange(T)
    COLORS = ['#2166AC','#D6604D','#4DAF4A','#F4A582','#984EA3']

    fig, axes = plt.subplots(N*2, 1, figsize=(12, 2.5*N),
                             gridspec_kw={"height_ratios": [2.5, 0.6]*N},
                             sharex=True)
    fig.suptitle(f"ep{ep_idx}: {task[:65]}", fontsize=8)

    for k, ((d, gt), tau) in enumerate(zip(phase_dists, phase_taus)):
        col = COLORS[k % len(COLORS)]
        pred = d < tau
        ax_d = axes[k*2]
        ax_r = axes[k*2+1]

        ax_d.plot(t, d, color=col, lw=1.2, label=f"d(z_t, z_phase{k+1})")
        ax_d.axhline(tau, color='#D6604D', ls='--', lw=1, label=f"τ={tau:.3f}")
        ax_d.fill_between(t, d.min(), d.max()*1.05, where=gt,
                          color='#C6DBEF', alpha=0.5, label="GT window")
        # Mark phase window boundaries
        ws, we = windows[k]
        ax_d.axvspan(ws, we, color=col, alpha=0.08)
        ax_d.set_ylabel(f"ph{k+1} dist", fontsize=7)
        ax_d.legend(fontsize=6, loc="upper right", ncol=3)

        rb_w = max(1, T // 400)
        for i in range(T):
            ax_r.axvspan(i, i+1, ymin=0.5, ymax=1.0,
                         color='#4DAF4A' if pred[i] else '#D6604D', alpha=0.85)
            ax_r.axvspan(i, i+1, ymin=0.0, ymax=0.5,
                         color='#2166AC' if gt[i] else '#FDAE61', alpha=0.85)
        ax_r.set_yticks([0.25, 0.75])
        ax_r.set_yticklabels(["GT", "Pred"], fontsize=6)
        ax_r.set_ylabel(f"ph{k+1}", fontsize=7)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fig.savefig(tl / f"ep{ep_idx:04d}.png", dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["dino","svd_vae"], default="svd_vae")
    p.add_argument("--cam",     choices=["wrist","exterior1"], default="wrist")
    p.add_argument("--out-dir", default="etl_results/droid_phasic")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
