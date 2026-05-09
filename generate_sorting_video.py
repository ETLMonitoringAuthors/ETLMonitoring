"""
Generate ETL monitoring GIF for D3IL sorting using pre-rendered rollout videos.

Usage:
    conda activate newt
    python generate_sorting_video.py \
        --data-dir etl_d3il/data/sorting \
        --gt-json  /path/to/sorting_gt.json \
        --rollout  episode_s_0275 \
        --out-dir  assets
"""

from __future__ import annotations

import argparse
import io
import json
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


def _cosine_dists(embs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    na = np.linalg.norm(anchor)
    na = na if na > 1e-12 else 1.0
    return 1.0 - (embs / norms) @ (anchor / na)


def _build_spec_latents(per_rollout_embs: list[np.ndarray], K: int) -> np.ndarray:
    """
    Anchor z_k = mean of the frame at the END of phase k across calibration rollouts.
    Phase k ends at frame index int((k+1)/K * T) - 1.
    This represents the goal state at the end of each phase, not the phase mean.
    """
    all_endpoints = []
    for embs in per_rollout_embs:
        T = len(embs)
        pts = [embs[min(T - 1, int((k + 1) * T / K) - 1)] for k in range(K)]
        all_endpoints.append(np.stack(pts))
    return np.stack(all_endpoints).mean(axis=0)  # [K, D]



def _percentile_threshold(per_rollout_embs: list[np.ndarray],
                           spec_latents: np.ndarray, K: int,
                           pct: float = 95.0) -> list[float]:
    """
    tau_k = pct-th percentile of min cosine distance to z_k across cal rollouts.
    Fires when the episode has gotten within pct% of its closest-ever approach to goal k.
    """
    taus = []
    for k in range(K):
        min_dists = [_cosine_dists(e, spec_latents[k]).min() for e in per_rollout_embs]
        taus.append(float(np.percentile(min_dists, pct)))
    return taus


def run(data_dir: Path, out: Path, fps: int, K: int, sim_width: int,
        gt_json: Path | None, rollout_stem: str | None):
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

    # ── spec latents: end-of-phase anchors ──────────────────────────────────
    spec_latents = _build_spec_latents(per_rollout_embs, K)

    # ── thresholds: 95th-pct of min-distances on cal set ────────────────────
    taus = _percentile_threshold(per_rollout_embs, spec_latents, K, pct=95.0)
    print(f"  spec latents built  taus: {[f'{t:.3f}' for t in taus]}")

    # ── pick test rollout ────────────────────────────────────────────────────
    chosen = None
    if rollout_stem:
        fp = test_dir / f"{rollout_stem}.pkl"
        if fp.exists():
            with open(fp, "rb") as f:
                chosen = (fp, pickle.load(f))
    if chosen is None:
        for fp in sorted(test_dir.glob("*.pkl")):
            if not (vid_dir / f"{fp.stem}.mp4").exists():
                continue
            with open(fp, "rb") as f:
                d = pickle.load(f)
            if d["metadata"].get("successful", False):
                chosen = (fp, d)
                break
    if chosen is None:
        raise RuntimeError("No suitable test rollout with matching video found.")

    fp, d = chosen
    stem = fp.stem
    print(f"  test rollout: {stem}  (successful={d['metadata'].get('successful')})")

    rollout = d["rollout"]
    T       = len(rollout)
    embs    = np.stack([s["obs_embedding"] for s in rollout])

    dist_traces = [_cosine_dists(embs, spec_latents[k]) for k in range(K)]
    # F(near_k): satisfied at t iff min_{t'<=t} dist(z_t', z_k) <= tau_k
    pred_traces = [(np.minimum.accumulate(dist_traces[k]) <= taus[k]).astype(float)
                   for k in range(K)]

    # ── GT traces ────────────────────────────────────────────────────────────
    if gt_json is not None and gt_json.exists():
        with open(gt_json) as f:
            gt_data = json.load(f)
        gt_keys = ["red_in_bin", "blue_in_bin"]
        gt_traces = [np.array([g[gt_keys[k]] for g in gt_data], dtype=float)
                     for k in range(K)]
        print(f"  GT loaded from {gt_json}")
        for k, key in enumerate(gt_keys[:K]):
            first = next((i for i, v in enumerate(gt_traces[k]) if v), None)
            print(f"    {key} first True at t={first}")
    else:
        # fallback: equal-time segmentation proxy (no VLM GT available)
        gt_traces = []
        for k in range(K):
            arr = np.zeros(T, dtype=float)
            s = int(k * T / K); e = int((k + 1) * T / K)
            arr[s:e] = 1.0
            gt_traces.append(arr)
        print("  GT: equal-time proxy (no --gt-json provided)")

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
    labels  = ["red block", "blue block"]

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
    ap.add_argument("--data-dir",  default="etl_d3il/data/sorting")
    ap.add_argument("--out-dir",   default="assets")
    ap.add_argument("--fps",       type=int,   default=12)
    ap.add_argument("--k",         type=int,   default=2)
    ap.add_argument("--width",     type=int,   default=480)
    ap.add_argument("--gt-json",   default=None,
                    help="Path to per-timestep GT JSON from label_sorting_gt.py")
    ap.add_argument("--rollout",   default=None,
                    help="Rollout stem to visualize (e.g. episode_s_0275)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    gt_json = Path(args.gt_json) if args.gt_json else None
    print("Generating sorting monitoring GIF…")
    run(Path(args.data_dir), out, fps=args.fps, K=args.k, sim_width=args.width,
        gt_json=gt_json, rollout_stem=args.rollout)
    print("Done.")


if __name__ == "__main__":
    main()
