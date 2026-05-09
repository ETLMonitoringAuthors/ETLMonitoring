"""
Generate ETL monitoring videos for MetaWorld pick-place-wall.

Runs one successful episode with the Newt world model, computes latent
distances to the grasp/place anchors at every timestep, and composites
the simulator frames with the monitoring trace into an animated GIF.

Usage:
    conda activate newt
    cd /path/to/repo
    MUJOCO_GL=egl python generate_videos.py [--out-dir assets]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

# ── repo path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TDMPC2_DIR = ROOT / "tdmpc2"
for p in [str(TDMPC2_DIR), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import hydra.utils as _hu
if not hasattr(_hu, "_orig_get_original_cwd"):
    _hu._orig_get_original_cwd = _hu.get_original_cwd
    _hu.get_original_cwd = lambda: str(Path.cwd())

from common import set_seed
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from etl_image_ablations.run_image_etl_ablations import collect_demos, encode_latent

# ── constants ────────────────────────────────────────────────────────────────
CHECKPOINT_BASE = (
    ROOT / "checkpoints/models--nicklashansen--newt/snapshots"
    / "7eef11eb63c8ed53d61d739693d7140135ea0876"
)
TASK       = "mw-pick-place-wall"
CHECKPOINT = str(CHECKPOINT_BASE / f"{TASK}.pt")
OBJ_Z_IDX  = 6
LIFT_THR   = 0.05

ETL_COLOR   = "#2563EB"
PLACE_COLOR = "#7C3AED"
GT_COLOR    = "#64748B"

# ── model loading ────────────────────────────────────────────────────────────

def _build_cfg(num_demos: int, seed: int):
    base = OmegaConf.structured(Config())
    OmegaConf.set_struct(base, False)
    merged = OmegaConf.merge(base, OmegaConf.create({
        "task":         TASK,
        "num_demos":    num_demos,
        "enable_wandb": False,
        "env_mode":     "sync",
        "checkpoint":   CHECKPOINT,
        "num_envs":     2,
        "model_size":   "B",
        "save_video":   True,
        "compile":      False,
        "seed":         seed,
    }))
    return parse_cfg(merged)


def _load_checkpoint(agent: TDMPC2, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location=torch.get_default_device(),
                            weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    target_sd = agent.model.state_dict()
    if "_action_masks" in state_dict and "_action_masks" in target_sd:
        n_target = target_sd["_action_masks"].shape[0]
        src_am   = state_dict["_action_masks"]
        if src_am.shape[0] != n_target:
            state_dict["_action_masks"] = src_am[0:1].repeat(n_target, 1)
    state_dict.pop("_task_emb.weight", None)
    agent.model.load_state_dict(state_dict, strict=True)


# ── ETL anchor calibration ───────────────────────────────────────────────────

def _gt_grasped(obs: torch.Tensor) -> np.ndarray:
    return (obs[:, OBJ_Z_IDX].numpy() > LIFT_THR).astype(bool)


def _gt_placed(success: torch.Tensor) -> np.ndarray:
    return (success.numpy() >= 0.99).astype(bool)


def build_spec_latent_A(cal_demos: list, window: int = 8) -> torch.Tensor:
    """z_A = centroid of latents during lift phase (lifted AND not yet placed)."""
    vecs = []
    for d in cal_demos:
        gt_a = _gt_grasped(d["obs"])
        gt_b = _gt_placed(d["success"])
        lifted_pre = gt_a & (~gt_b)
        idx = np.where(lifted_pre)[0]
        if len(idx) == 0:
            idx = np.where(gt_a)[0]
        if len(idx) == 0:
            continue
        first = int(idx[0])
        T = d["emb"].shape[0]
        t0 = max(0, first - window // 2)
        t1 = min(T - 1, first + window)
        vecs.append(d["emb"][t0: t1 + 1])
    if not vecs:
        raise ValueError("No grasp events in cal demos.")
    return torch.cat(vecs, dim=0).mean(dim=0)


def build_spec_latent_B(cal_demos: list, window: int = 10) -> torch.Tensor:
    """z_B = centroid of latents around the episode success moment."""
    vecs = []
    for d in cal_demos:
        done_idx = int(d["done_idx"].item())
        T = d["emb"].shape[0]
        t0 = max(0, done_idx - window)
        t1 = min(T - 1, done_idx + window)
        vecs.append(d["emb"][t0: t1 + 1])
    if not vecs:
        raise ValueError("No cal demos for z_B.")
    return torch.cat(vecs, dim=0).mean(dim=0)


def _f1_threshold(dists: np.ndarray, gt: np.ndarray) -> float:
    """F1-optimal distance threshold."""
    best_f1, best_tau = 0.0, float(dists.max())
    for tau in np.percentile(dists, np.linspace(2, 98, 80)):
        pred = dists <= tau
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


# ── composite frame builder ──────────────────────────────────────────────────

def _monitoring_panel(
    t: int, T: int,
    d_grasp: np.ndarray, d_place: np.ndarray,
    tau_grasp: float, tau_place: float,
    pred_grasp: np.ndarray, pred_place: np.ndarray,
    gt_grasp: np.ndarray, gt_place: np.ndarray,
    width_px: int = 480,
) -> np.ndarray:
    """Render the monitoring panel up to timestep t; return as RGB numpy array."""
    fig, axes = plt.subplots(
        4, 1, figsize=(width_px / 100, 3.2),
        gridspec_kw={"height_ratios": [2.0, 2.0, 0.55, 0.55]},
    )
    ts = np.arange(T)

    for ax in axes[:2]:
        ax.set_xlim(0, T - 1)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 0 — grasp distance
    ax = axes[0]
    ax.set_ylim(-0.1, 4.8)
    ax.plot(ts[:t+1], d_grasp[:t+1], color=ETL_COLOR, lw=1.3)
    ax.axhline(tau_grasp, color="red", ls="--", lw=1.0, label=rf"$\tau_A$={tau_grasp:.2f}")
    ax.set_ylabel(r"dist$(z_t,z_A)$" + "\n(grasp)", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6)

    # Panel 1 — place distance
    ax = axes[1]
    ax.set_ylim(-0.1, 5.0)
    ax.plot(ts[:t+1], d_place[:t+1], color=PLACE_COLOR, lw=1.3)
    ax.axhline(tau_place, color="red", ls="--", lw=1.0, label=rf"$\tau_B$={tau_place:.2f}")
    ax.set_ylabel(r"dist$(z_t,z_B)$" + "\n(place)", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6)

    # Predicate bars
    bar_data = [
        (pred_grasp, gt_grasp, "pred A", "GT A", ETL_COLOR),
        (pred_place, gt_place, "pred B", "GT B", PLACE_COLOR),
    ]
    for i, (pred, gt, lp, lg, col) in enumerate(bar_data):
        ax = axes[2 + i]
        ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([lg, lp], fontsize=6)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)
        for j in range(t):
            if pred[j]:
                ax.barh(1.5, 1, left=j, height=0.55, color=col, align="center")
            if gt[j]:
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
    return panel


def composite_frames(
    sim_frames: list[np.ndarray],
    d_grasp: np.ndarray, d_place: np.ndarray,
    tau_grasp: float, tau_place: float,
    pred_grasp: np.ndarray, pred_place: np.ndarray,
    gt_grasp: np.ndarray, gt_place: np.ndarray,
) -> list[np.ndarray]:
    T = len(sim_frames)
    frames_out = []
    W = sim_frames[0].shape[1]   # match sim frame width

    print(f"  compositing {T} frames…", flush=True)
    for t in range(T):
        sim = sim_frames[t]
        panel = _monitoring_panel(
            t, T,
            d_grasp, d_place,
            tau_grasp, tau_place,
            pred_grasp, pred_place,
            gt_grasp, gt_place,
            width_px=W,
        )
        # resize panel width to match sim frame
        if panel.shape[1] != W:
            panel = np.array(
                Image.fromarray(panel).resize((W, panel.shape[0]),
                                              Image.LANCZOS)
            )
        composite = np.vstack([sim, panel])
        frames_out.append(composite)
        if (t + 1) % 20 == 0:
            print(f"    {t+1}/{T}", flush=True)
    return frames_out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--sorting-data", default="etl_d3il/data/sorting",
                    help="Path to D3IL sorting data dir")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    set_seed(args.seed)

    print("Loading model and environment…")
    cfg = _build_cfg(num_demos=6, seed=args.seed)
    env = make_env(cfg)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    _load_checkpoint(agent, CHECKPOINT)
    agent.eval()

    tasks = torch.zeros(cfg.num_envs, dtype=torch.long)

    print("Collecting demos…")
    demos = collect_demos(cfg, agent, env, tasks)
    env.close()

    # encode latents
    print("Encoding latents…")
    for d in demos:
        task_rep = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["emb"] = encode_latent(agent, d["obs"], task_rep)

    # split cal / test
    n_cal = max(2, len(demos) // 2)
    cal_demos  = demos[:n_cal]
    test_demos = demos[n_cal:]
    if not test_demos:
        test_demos = cal_demos[:1]

    z_A = build_spec_latent_A(cal_demos)
    z_B = build_spec_latent_B(cal_demos)

    # calibrate thresholds per-demo then concatenate
    d_A_cal = np.concatenate([
        torch.cdist(d["emb"], z_A.unsqueeze(0)).squeeze(1).numpy()
        for d in cal_demos
    ])
    d_B_cal = np.concatenate([
        torch.cdist(d["emb"], z_B.unsqueeze(0)).squeeze(1).numpy()
        for d in cal_demos
    ])
    gt_A_cal = np.concatenate([_gt_grasped(d["obs"]) for d in cal_demos])
    gt_B_cal = np.concatenate([_gt_placed(d["success"]) for d in cal_demos])

    tau_A = _f1_threshold(d_A_cal, gt_A_cal)
    tau_B = _f1_threshold(d_B_cal, gt_B_cal)
    print(f"  tau_A={tau_A:.3f}  tau_B={tau_B:.3f}")

    # pick one test demo for the video
    d = test_demos[0]
    emb  = d["emb"]
    obs  = d["obs"]
    succ = d["success"]

    d_grasp = torch.cdist(emb, z_A.unsqueeze(0)).squeeze(1).numpy()
    d_place = torch.cdist(emb, z_B.unsqueeze(0)).squeeze(1).numpy()

    gt_grasp = _gt_grasped(obs).astype(float)
    gt_place = _gt_placed(succ).astype(float)
    pred_grasp = (d_grasp <= tau_A).astype(float)
    pred_place = (d_place <= tau_B).astype(float)

    # sim frames: [T, H, W, 3]  (uint8)
    raw_frames = d["frame"]   # torch [T, C, H, W] or [T, H, W, C]
    if raw_frames.shape[1] == 3:   # [T, C, H, W] → [T, H, W, C]
        raw_frames = raw_frames.permute(0, 2, 3, 1)
    sim_frames = [
        np.clip(f.numpy() * 255, 0, 255).astype(np.uint8)
        if raw_frames.dtype != torch.uint8
        else f.numpy()
        for f in raw_frames
    ]

    # composite
    composite = composite_frames(
        sim_frames, d_grasp, d_place,
        tau_A, tau_B,
        pred_grasp, pred_place,
        gt_grasp, gt_place,
    )

    # save GIF
    gif_path = out / "mw_monitoring.gif"
    pil_frames = [Image.fromarray(f) for f in composite]
    duration_ms = int(1000 / args.fps)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  saved {gif_path}")


def sorting_monitoring_gif(
    data_dir: str,
    out: Path,
    fps: int = 12,
    K: int = 2,
    sim_width: int = 480,
):
    """
    Build an ETL monitoring GIF for D3IL sorting using pre-rendered rollout videos.

    Anchors z_0..z_{K-1} are built by equal-time segmentation of calibration
    rollouts (same approach as ETLTemporal in etl_eval.py).
    """
    import pickle, glob, cv2

    data_dir = Path(data_dir)
    cal_dir  = data_dir / "rollouts" / "calibration"
    test_dir = data_dir / "rollouts" / "test"
    vid_dir  = data_dir / "rollouts" / "videos" / "test"

    BLUE1 = "#2563EB"
    BLUE2 = "#3B82F6"
    colors = [BLUE1, BLUE2]

    # ── load calibration embeddings ──────────────────────────────────────────
    cal_files = sorted(cal_dir.glob("*.pkl"))
    per_rollout_embs = []
    for fp in cal_files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if not d["metadata"].get("successful", False):
            continue
        embs = np.stack([s["obs_embedding"] for s in d["rollout"]])
        per_rollout_embs.append(embs)

    # ── build K-phase spec latents by equal-time segmentation ────────────────
    all_phase_means = []
    for embs in per_rollout_embs:
        T = len(embs)
        phase_means = []
        for k in range(K):
            s = int(k * T / K)
            e = int((k + 1) * T / K)
            e = max(e, s + 1)
            phase_means.append(embs[s:e].mean(axis=0))
        all_phase_means.append(np.stack(phase_means))
    spec_latents = np.stack(all_phase_means).mean(axis=0)  # [K, D]

    # ── F1 thresholds from cal rollouts ──────────────────────────────────────
    taus = []
    for k in range(K):
        z_k = spec_latents[k]
        all_dists, all_gt = [], []
        for i, embs in enumerate(per_rollout_embs):
            T = len(embs)
            dists = np.linalg.norm(embs - z_k, axis=1)
            # GT: timesteps in segment k
            s = int(k * T / K); e = int((k + 1) * T / K); e = max(e, s + 1)
            gt = np.zeros(T, dtype=bool)
            gt[s:e] = True
            all_dists.append(dists)
            all_gt.append(gt)
        dists_cat = np.concatenate(all_dists)
        gt_cat    = np.concatenate(all_gt)
        taus.append(_f1_threshold(dists_cat, gt_cat))

    print(f"  sorting taus: {[f'{t:.3f}' for t in taus]}")

    # ── pick a test rollout ───────────────────────────────────────────────────
    test_files = sorted(test_dir.glob("*.pkl"))
    # prefer a successful one
    chosen_pkl = None
    for fp in test_files:
        with open(fp, "rb") as f:
            d = pickle.load(f)
        if d["metadata"].get("successful", False):
            chosen_pkl = fp
            break
    if chosen_pkl is None:
        chosen_pkl = test_files[0]
        with open(chosen_pkl, "rb") as f:
            d = pickle.load(f)

    stem    = chosen_pkl.stem
    vid_path = vid_dir / f"{stem}.mp4"
    print(f"  using rollout: {stem}  (successful={d['metadata'].get('successful')})")

    rollout = d["rollout"]
    T       = len(rollout)
    embs    = np.stack([s["obs_embedding"] for s in rollout])

    # distances per phase
    dist_traces = [np.linalg.norm(embs - spec_latents[k], axis=1) for k in range(K)]
    pred_traces = [(dist_traces[k] <= taus[k]).astype(float) for k in range(K)]
    # GT: each phase active in its time segment
    gt_traces = []
    for k in range(K):
        s = int(k * T / K); e = int((k + 1) * T / K); e = max(e, s + 1)
        gt = np.zeros(T, dtype=bool)
        gt[s:e] = True
        gt_traces.append(gt.astype(float))

    # ── extract sim frames from the pre-rendered video ────────────────────────
    cap = cv2.VideoCapture(str(vid_path))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.round(np.linspace(0, total_vid_frames - 1, T)).astype(int)
    sim_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((sim_width, sim_width, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # crop centre square and resize
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2; x0 = (w - side) // 2
        frame = frame[y0:y0+side, x0:x0+side]
        frame = np.array(Image.fromarray(frame).resize((sim_width, sim_width), Image.LANCZOS))
        sim_frames.append(frame)
    cap.release()

    # ── composite ─────────────────────────────────────────────────────────────
    d_ylims = [max(tr.max() * 1.15, taus[k] * 1.3) for k, tr in enumerate(dist_traces)]

    def _panel(t):
        n_rows = 2 + K
        ratios = [2.0, 2.0] + [0.55] * K
        fig, axes = plt.subplots(n_rows, 1, figsize=(sim_width / 100, 3.2),
                                 gridspec_kw={"height_ratios": ratios})
        ts = np.arange(T)
        labels = [f"block {k+1}" for k in range(K)]
        for k in range(K):
            ax = axes[k]
            ax.set_xlim(0, T - 1); ax.set_ylim(-0.05, d_ylims[k])
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.plot(ts[:t+1], dist_traces[k][:t+1], color=colors[k], lw=1.3)
            ax.axhline(taus[k], color="red", ls="--", lw=1.0,
                       label=rf"$\tau_{k}$={taus[k]:.2f}")
            ax.set_ylabel(rf"dist$(z_t,z_{k})$" + f"\n({labels[k]})", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

        for k in range(K):
            ax = axes[2 + k]
            ax.set_xlim(0, T - 1); ax.set_ylim(0, 2)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels([f"GT {k+1}", f"pred {k+1}"], fontsize=6)
            ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
            ax.grid(False)
            for j in range(t):
                if pred_traces[k][j]:
                    ax.barh(1.5, 1, left=j, height=0.55, color=colors[k], align="center")
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
        # resize panel width to match sim frame
        if panel.shape[1] != sim_width:
            panel = np.array(Image.fromarray(panel).resize(
                (sim_width, panel.shape[0]), Image.LANCZOS))
        return panel

    print(f"  compositing {T} frames…", flush=True)
    composite = []
    for t in range(T):
        panel = _panel(t)
        composite.append(np.vstack([sim_frames[t], panel]))
        if (t + 1) % 10 == 0:
            print(f"    {t+1}/{T}", flush=True)

    gif_path = out / "sorting_monitoring.gif"
    pil_frames_out = [Image.fromarray(f) for f in composite]
    pil_frames_out[0].save(
        gif_path, save_all=True, append_images=pil_frames_out[1:],
        duration=int(1000 / fps), loop=0, optimize=False,
    )
    print(f"  saved {gif_path}")

    # ── Sorting ───────────────────────────────────────────────────────────────
    sorting_data = Path(args.sorting_data)
    if sorting_data.exists():
        print("\nGenerating sorting monitoring GIF…")
        sorting_monitoring_gif(sorting_data, out, fps=args.fps)
    else:
        print(f"\nSkipping sorting (data not found at {sorting_data})")


if __name__ == "__main__":
    main()
