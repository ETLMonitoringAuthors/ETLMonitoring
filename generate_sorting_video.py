"""
Generate ETL monitoring GIF for D3IL sorting using pre-rendered rollout videos.

Usage:
    conda activate newt
    python generate_sorting_video.py --data-dir etl_d3il/data/sorting --out-dir assets
"""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BLUE1    = "#2563EB"
BLUE2    = "#3B82F6"
GT_COLOR = "#64748B"
COLORS   = [BLUE1, BLUE2]


def _f1_threshold(dists: np.ndarray, gt: np.ndarray) -> float:
    best_f1, best_tau = 0.0, float(dists.max())
    for tau in np.percentile(dists, np.linspace(2, 98, 80)):
        pred = dists <= tau
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


def run(data_dir: Path, out: Path, fps: int, K: int, sim_width: int):
    cal_dir  = data_dir / "rollouts" / "calibration"
    test_dir = data_dir / "rollouts" / "test"
    vid_dir  = data_dir / "rollouts" / "videos" / "test"

    # ── calibration embeddings ───────────────────────────────────────────────
    per_rollout_embs = []
    for fp in sorted(cal_dir.glob("*.pkl")):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if not d["metadata"].get("successful", False):
            continue
        per_rollout_embs.append(
            np.stack([s["obs_embedding"] for s in d["rollout"]])
        )
    print(f"  loaded {len(per_rollout_embs)} successful cal rollouts")

    # ── K-phase spec latents by equal-time segmentation ──────────────────────
    all_phase_means = []
    for embs in per_rollout_embs:
        T = len(embs)
        phase_means = []
        for k in range(K):
            s = int(k * T / K); e = max(int((k + 1) * T / K), s + 1)
            phase_means.append(embs[s:e].mean(axis=0))
        all_phase_means.append(np.stack(phase_means))
    spec_latents = np.stack(all_phase_means).mean(axis=0)  # [K, D]

    # ── F1 thresholds ────────────────────────────────────────────────────────
    taus = []
    for k in range(K):
        z_k = spec_latents[k]
        all_d, all_gt = [], []
        for embs in per_rollout_embs:
            T = len(embs)
            s = int(k * T / K); e = max(int((k + 1) * T / K), s + 1)
            gt = np.zeros(T, dtype=bool); gt[s:e] = True
            all_d.append(np.linalg.norm(embs - z_k, axis=1))
            all_gt.append(gt)
        taus.append(_f1_threshold(np.concatenate(all_d), np.concatenate(all_gt)))
    print(f"  taus: {[f'{t:.3f}' for t in taus]}")

    # ── pick a successful test rollout ────────────────────────────────────────
    chosen = None
    for fp in sorted(test_dir.glob("*.pkl")):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if d["metadata"].get("successful", False):
            chosen = (fp, d)
            break
    if chosen is None:
        fp = sorted(test_dir.glob("*.pkl"))[0]
        with open(fp, "rb") as f:
            d = pickle.load(f)
        chosen = (fp, d)

    fp, d = chosen
    stem = fp.stem
    print(f"  test rollout: {stem}  (successful={d['metadata'].get('successful')})")

    rollout = d["rollout"]
    T       = len(rollout)
    embs    = np.stack([s["obs_embedding"] for s in rollout])

    dist_traces = [np.linalg.norm(embs - spec_latents[k], axis=1) for k in range(K)]
    pred_traces = [(dist_traces[k] <= taus[k]).astype(float) for k in range(K)]
    gt_traces   = []
    for k in range(K):
        s = int(k * T / K); e = max(int((k + 1) * T / K), s + 1)
        gt = np.zeros(T, dtype=bool); gt[s:e] = True
        gt_traces.append(gt.astype(float))

    # ── extract sim frames ────────────────────────────────────────────────────
    vid_path = vid_dir / f"{stem}.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.round(np.linspace(0, n_vid - 1, T)).astype(int)
    sim_frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((sim_width, sim_width, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2; x0 = (w - side) // 2
            frame = frame[y0:y0+side, x0:x0+side]
            frame = np.array(Image.fromarray(frame).resize((sim_width, sim_width), Image.LANCZOS))
        sim_frames.append(frame)
    cap.release()

    # ── composite ─────────────────────────────────────────────────────────────
    d_ylims = [max(tr.max() * 1.15, taus[k] * 1.3) for k, tr in enumerate(dist_traces)]
    labels  = [f"block {k+1}" for k in range(K)]

    def _panel(t):
        ratios = [2.0] * K + [0.55] * K
        fig, axes = plt.subplots(K + K, 1, figsize=(sim_width / 100, 3.2),
                                 gridspec_kw={"height_ratios": ratios})
        ts = np.arange(T)
        for k in range(K):
            ax = axes[k]
            ax.set_xlim(0, T - 1); ax.set_ylim(-0.05, d_ylims[k])
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.plot(ts[:t+1], dist_traces[k][:t+1], color=COLORS[k], lw=1.3)
            ax.axhline(taus[k], color="red", ls="--", lw=1.0,
                       label=rf"$\tau_{k}$={taus[k]:.2f}")
            ax.set_ylabel(rf"dist$(z_t,z_{k})$" + f"\n({labels[k]})", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
        for k in range(K):
            ax = axes[K + k]
            ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels([f"GT {k+1}", f"pred {k+1}"], fontsize=6)
            ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
            ax.grid(False)
            for j in range(t):
                if pred_traces[k][j]:
                    ax.barh(1.5, 1, left=j, height=0.55, color=COLORS[k], align="center")
                if gt_traces[k][j]:
                    ax.barh(0.5, 1, left=j, height=0.55, color=GT_COLOR, align="center")
            ax.tick_params(labelsize=6)
        axes[-1].set_xlabel("Timestep", fontsize=7)
        axes[-1].tick_params(labelsize=6)
        fig.tight_layout(h_pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        panel = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        if panel.shape[1] != sim_width:
            panel = np.array(Image.fromarray(panel).resize(
                (sim_width, panel.shape[0]), Image.LANCZOS))
        return panel

    print(f"  compositing {T} frames…", flush=True)
    composite = []
    for t in range(T):
        composite.append(np.vstack([sim_frames[t], _panel(t)]))
        if (t + 1) % 5 == 0:
            print(f"    {t+1}/{T}", flush=True)

    gif_path = out / "sorting_monitoring.gif"
    pil_out = [Image.fromarray(f) for f in composite]
    pil_out[0].save(gif_path, save_all=True, append_images=pil_out[1:],
                    duration=int(1000 / fps), loop=0, optimize=False)
    print(f"  saved {gif_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="etl_d3il/data/sorting")
    ap.add_argument("--out-dir",  default="assets")
    ap.add_argument("--fps",      type=int, default=12)
    ap.add_argument("--k",        type=int, default=2)
    ap.add_argument("--width",    type=int, default=480)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    print("Generating sorting monitoring GIF…")
    run(Path(args.data_dir), out, fps=args.fps, K=args.k, sim_width=args.width)
    print("Done.")


if __name__ == "__main__":
    main()
