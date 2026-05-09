"""
eval_lerobot_cosmos_etl.py
--------------------------
ETL analysis on real robot LeRobot datasets using Cosmos Wan2pt1 VAE latents.

Uses the FULL spatial latent (16 x 32 x 32 = 16384 dims per latent frame),
PCA-reduced to n_components (default 256) on calibration data before ETL.
This keeps spatial structure rather than collapsing it via mean pooling.

Run inside the cosmos-predict2.5 uv environment:

  export VAE=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/6787e176dce74a101d922174a95dba29fa5f0c55/tokenizer.pth
  export CU13=/path/to/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
  export LD_LIBRARY_PATH=$CU13:$LD_LIBRARY_PATH

  cd /path/to/cosmos-predict2.5
  uv run python ./etl_image_ablations/eval_lerobot_cosmos_etl.py \\
      --dataset both --num-episodes 60 --vae-pth $VAE \\
      --out-dir ./etl_results/lerobot_cosmos_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# ---- Cosmos VAE encoder ------------------------------------------------------

COSMOS_VAE_RES     = 256   # input spatial resolution
COSMOS_VAE_CHUNK_T = 17    # (17-1) % 4 == 0  =>  T_lat = 5 per chunk
COSMOS_VAE_STRIDE  = 4     # latent temporal stride


class CosmosVAEEncoder:
    """
    Encodes RGB frame lists to full spatial Cosmos latents (16 x H_lat x W_lat)
    per latent frame, upsampled back to per-input-frame resolution.

    Workflow:
      embs = enc.encode_frames(frames)          # (N, 16384)  raw latents
      enc.fit_pca(cal_embs, n_components=256)   # fit PCA on calibration data
      embs_pca = enc.pca_transform(embs)        # (N, 256)
    """

    def __init__(self, vae_pth: str, device: str = "cuda"):
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        self.device = device
        self.pca = None
        print(f"Loading Cosmos Wan2pt1 VAE from {vae_pth} ...")
        self.vae = Wan2pt1VAEInterface(vae_pth=vae_pth, device=device)
        self.vae.model.model.eval()
        print("  VAE loaded OK")

    @torch.no_grad()
    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Returns (N, 16*H_lat*W_lat) float32 full spatial latents."""
        N = len(frames)
        if N == 0:
            return np.zeros((0, 16 * 32 * 32), dtype=np.float32)

        processed = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float()
            t = TF.resize(t, [COSMOS_VAE_RES, COSMOS_VAE_RES], antialias=True)
            t = t / 127.5 - 1.0
            processed.append(t)
        video = torch.stack(processed, dim=0)  # (N, 3, H, W)

        pad = (COSMOS_VAE_CHUNK_T - (N % COSMOS_VAE_CHUNK_T)) % COSMOS_VAE_CHUNK_T
        if pad:
            video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
        N_pad = len(video)

        chunks = []
        for s in range(0, N_pad, COSMOS_VAE_CHUNK_T):
            x = video[s:s + COSMOS_VAE_CHUNK_T].permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.vae.encode(x)                        # (1, 16, T_lat, H_lat, W_lat)
            T_lat = z.shape[2]
            # (T_lat, 16 * H_lat * W_lat)
            z_flat = z.squeeze(0).permute(1, 0, 2, 3).reshape(T_lat, -1).float().cpu().numpy()
            chunks.append(z_flat)

        all_lat = np.concatenate(chunks, axis=0)              # (N_chunks*T_lat, D)
        # Upsample: repeat each latent frame STRIDE times
        frame_lat = np.repeat(all_lat, COSMOS_VAE_STRIDE, axis=0)
        if len(frame_lat) < N_pad:
            frame_lat = np.concatenate(
                [frame_lat, frame_lat[-1:].repeat(N_pad - len(frame_lat), axis=0)]
            )
        return frame_lat[:N]

    def fit_pca(self, embeddings: np.ndarray, n_components: int = 256) -> None:
        from sklearn.decomposition import PCA
        print(f"  Fitting PCA({n_components}) on {embeddings.shape[0]} latent frames ...")
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(embeddings)
        var = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA explains {var * 100:.1f}% variance")

    def pca_transform(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "Call fit_pca() first"
        return self.pca.transform(embeddings).astype(np.float32)


# ---- LeRobot data loading ----------------------------------------------------

def download(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_dataset_meta(repo_id: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(download(repo_id, "meta/info.json")) as f:
        info = json.load(f)
    data_df = pd.read_parquet(download(repo_id, "data/chunk-000/file-000.parquet"))
    ep_df   = pd.read_parquet(download(repo_id, "meta/episodes/chunk-000/file-000.parquet"))
    return info, data_df, ep_df


def decode_mp4_frames(mp4_path: Path, from_ts: float, to_ts: float) -> List[np.ndarray]:
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(from_ts * 1_000_000))
        for packet in container.demux(stream):
            for frame in packet.decode():
                t = float(frame.pts * stream.time_base)
                if t < from_ts - 0.001:
                    continue
                if t > to_ts + 0.001:
                    return frames
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_episode_frames(repo_id: str, ep_row, video_key: str) -> List[np.ndarray]:
    chunk_idx = int(ep_row[f"videos/{video_key}/chunk_index"])
    file_idx  = int(ep_row[f"videos/{video_key}/file_index"])
    from_ts   = float(ep_row[f"videos/{video_key}/from_timestamp"])
    to_ts     = float(ep_row[f"videos/{video_key}/to_timestamp"])
    mp4_file  = f"videos/{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
    return decode_mp4_frames(download(repo_id, mp4_file), from_ts, to_ts)


# ---- ETL core ----------------------------------------------------------------

def build_spec_latent(embeddings: np.ndarray, mask: np.ndarray, window: int = 10) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(max(0, len(embeddings) - window), len(embeddings))
    else:
        idx = idx[-min(window, len(idx)):]
    return embeddings[idx].mean(axis=0)


def l2_distances(embeddings: np.ndarray, z_spec: np.ndarray) -> np.ndarray:
    return np.linalg.norm(embeddings - z_spec[None], axis=1)


def sweep_f1(distances: np.ndarray, labels: np.ndarray,
             n_thresh: int = 200) -> Tuple[float, float, float, float]:
    taus = np.linspace(distances.min(), distances.max(), n_thresh)
    best = (0.0, taus[0], 0.0, 0.0)
    for tau in taus:
        pred = distances < tau
        tp = (pred & (labels == 1)).sum()
        fp = (pred & (labels == 0)).sum()
        fn = (~pred & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best[0]:
            best = (f1, tau, prec, rec)
    return best[1], best[0], best[2], best[3]


def conformal_tau(distances: np.ndarray, labels: np.ndarray, alpha: float = 0.10) -> Optional[float]:
    pos = distances[labels == 1]
    if len(pos) < 2:
        return None
    n = len(pos)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(pos, q))


def predicate_metrics(distances: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    pred = (distances < tau).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tau=float(tau), precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn,
                agreement=(tp + tn) / (tp + fp + fn + tn + 1e-9))


# ---- Plotting ----------------------------------------------------------------

def plot_f1_curve(distances, labels, tau_f1, tau_cp, title, out_path):
    taus = np.linspace(distances.min(), distances.max(), 300)
    f1s, precs, recs = [], [], []
    for tau in taus:
        m = predicate_metrics(distances, labels, tau)
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(taus, f1s,   label="F1",        color="steelblue")
    ax.plot(taus, precs, label="Precision", color="green",  ls="--")
    ax.plot(taus, recs,  label="Recall",    color="orange", ls=":")
    ax.axvline(tau_f1, color="red",    ls="--", lw=1.5, label=f"tau_F1={tau_f1:.3f}")
    if tau_cp:
        ax.axvline(tau_cp, color="purple", ls=":",  lw=1.5, label=f"tau_CP={tau_cp:.3f}")
    ax.set_xlabel("tau"); ax.set_ylabel("Score"); ax.set_title(title)
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120); plt.close()


def plot_seq_timeline(dist_A, dist_B, gt_A, gt_B, tau_A, tau_B, title, out_path):
    T = len(dist_A); t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].fill_between(t, 0, gt_A.astype(float), alpha=0.4, color="steelblue", label="GT pick")
    axes[0].fill_between(t, 0, gt_B.astype(float), alpha=0.4, color="green",     label="GT insert")
    axes[0].set_ylabel("GT"); axes[0].set_ylim(-0.05, 1.3)
    axes[0].legend(loc="upper right", fontsize=8); axes[0].set_title(title, fontsize=10)
    axes[1].plot(t, dist_A, color="steelblue", lw=1.2, label="d(z, z_pick)")
    axes[1].axhline(tau_A, color="red", ls="--", lw=1.5, label=f"tau={tau_A:.2f}")
    axes[1].fill_between(t, 0, dist_A.max(), where=dist_A < tau_A, alpha=0.2, color="steelblue")
    axes[1].set_ylabel("Dist to z_pick"); axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, dist_B, color="green", lw=1.2, label="d(z, z_insert)")
    axes[2].axhline(tau_B, color="red", ls="--", lw=1.5, label=f"tau={tau_B:.2f}")
    axes[2].fill_between(t, 0, dist_B.max(), where=dist_B < tau_B, alpha=0.2, color="green")
    axes[2].set_ylabel("Dist to z_insert"); axes[2].set_xlabel("Frame")
    axes[2].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120); plt.close()


# ---- FMB evaluation ----------------------------------------------------------

FMB_REPO      = "lerobot/fmb"
FMB_VIDEO_KEY = "observation.images.image_side_1"
FMB_PICK_IDS   = {0, 4, 6, 10, 16, 20}
FMB_INSERT_IDS = {3, 5, 9, 11, 17, 21}


def evaluate_fmb(enc: CosmosVAEEncoder, num_episodes: int, out_dir: Path,
                 pca_components: int = 256, cal_frac: float = 0.5, seed: int = 42):
    print("\n=== FMB (Cosmos VAE full latent) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    info, data_df, ep_df = load_dataset_meta(FMB_REPO)

    valid_ids = []
    for _, row in ep_df.iterrows():
        tmin = row.get("stats/task_index/min", [None])
        tmax = row.get("stats/task_index/max", [None])
        if isinstance(tmin, list): tmin = tmin[0]
        if isinstance(tmax, list): tmax = tmax[0]
        if tmin is None: continue
        ids = set(range(int(tmin), int(tmax) + 1))
        if ids & FMB_PICK_IDS and ids & FMB_INSERT_IDS:
            valid_ids.append(int(row["episode_index"]))

    rng     = np.random.default_rng(seed)
    sampled = rng.choice(valid_ids, min(num_episodes, len(valid_ids)), replace=False).tolist()

    # Phase 1: encode all episodes (raw 16384-dim)
    ep_raw   : Dict[int, np.ndarray] = {}
    ep_pick_m: Dict[int, np.ndarray] = {}
    ep_ins_m : Dict[int, np.ndarray] = {}

    for ep_idx in sampled:
        print(f"  Encoding FMB ep {ep_idx} ...", end=" ", flush=True)
        ep_row  = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
        frames  = load_episode_frames(FMB_REPO, ep_row, FMB_VIDEO_KEY)
        if not frames: print("no frames"); continue
        ep_data = data_df[data_df["episode_index"] == ep_idx].sort_values("frame_index")
        T       = min(len(frames), len(ep_data))
        tids    = ep_data["task_index"].values[:T]
        raw     = enc.encode_frames(frames[:T])
        ep_raw[ep_idx]    = raw
        ep_pick_m[ep_idx] = np.array([int(t) in FMB_PICK_IDS   for t in tids])
        ep_ins_m[ep_idx]  = np.array([int(t) in FMB_INSERT_IDS for t in tids])
        print(f"{T} frames | pick={ep_pick_m[ep_idx].sum()} insert={ep_ins_m[ep_idx].sum()}")

    valid  = list(ep_raw.keys())
    n_cal  = max(1, int(len(valid) * cal_frac))
    cal_ids, test_ids = valid[:n_cal], valid[n_cal:]
    if not test_ids: test_ids = cal_ids

    # Phase 2: fit PCA on calibration embeddings
    cal_all = np.concatenate([ep_raw[e] for e in cal_ids])
    enc.fit_pca(cal_all, n_components=pca_components)

    # Phase 3: project all episodes
    ep_emb = {e: enc.pca_transform(ep_raw[e]) for e in valid}

    def cal_spec(ep_list, mask_dict, window=5):
        zs = [ep_emb[e][np.where(mask_dict[e])[0][-min(window, mask_dict[e].sum()):]]
              for e in ep_list if mask_dict[e].any()]
        return np.concatenate(zs).mean(axis=0)

    z_pick   = cal_spec(cal_ids, ep_pick_m)
    z_insert = cal_spec(cal_ids, ep_ins_m)

    def pool(ep_list, z_spec, mask_dict):
        return (np.concatenate([l2_distances(ep_emb[e], z_spec) for e in ep_list]),
                np.concatenate([mask_dict[e].astype(int)         for e in ep_list]))

    cal_d_pk, cal_l_pk = pool(cal_ids, z_pick,   ep_pick_m)
    cal_d_in, cal_l_in = pool(cal_ids, z_insert, ep_ins_m)

    tau_pk_f1, f1_pk, _, _ = sweep_f1(cal_d_pk, cal_l_pk)
    tau_in_f1, f1_in, _, _ = sweep_f1(cal_d_in, cal_l_in)
    tau_pk_cp = conformal_tau(cal_d_pk, cal_l_pk)
    tau_in_cp = conformal_tau(cal_d_in, cal_l_in)
    print(f"\n  tau_pick F1={tau_pk_f1:.3f} CP={tau_pk_cp:.3f}  "
          f"tau_insert F1={tau_in_f1:.3f} CP={tau_in_cp:.3f}")

    metrics_out = {}
    for sn, z_spec, tau_f1, tau_cp, mdict in [
        ("pick",   z_pick,   tau_pk_f1, tau_pk_cp, ep_pick_m),
        ("insert", z_insert, tau_in_f1, tau_in_cp, ep_ins_m),
    ]:
        td, tl = pool(test_ids, z_spec, mdict)
        m_f1 = predicate_metrics(td, tl, tau_f1)
        m_cp = predicate_metrics(td, tl, tau_cp) if tau_cp else None
        print(f"  pi_{sn} tau_F1: F1={m_f1['f1']:.3f} P={m_f1['precision']:.3f} R={m_f1['recall']:.3f}")
        if m_cp:
            print(f"  pi_{sn} tau_CP: F1={m_cp['f1']:.3f} P={m_cp['precision']:.3f} R={m_cp['recall']:.3f}")
        plot_f1_curve(td, tl, tau_f1, tau_cp, f"FMB Cosmos full — pi_{sn}", out_dir / f"f1_{sn}.png")
        metrics_out[sn] = {"tau_F1": tau_f1, "tau_CP": tau_cp, "test_tau_F1": m_f1, "test_tau_CP": m_cp}

    # Sequential evaluation
    seq_results = []
    for ep_idx in test_ids:
        da = l2_distances(ep_emb[ep_idx], z_pick)
        db = l2_distances(ep_emb[ep_idx], z_insert)
        gtA, gtB = ep_pick_m[ep_idx], ep_ins_m[ep_idx]
        gt_seq   = gtA.any() and gtB.any() and (np.where(gtA)[0][0] < np.where(gtB)[0][-1])
        A_t = np.where(da < tau_pk_f1)[0]
        B_t = np.where(db < tau_in_f1)[0]
        pred_seq = len(A_t) > 0 and len(B_t) > 0 and A_t[0] < B_t[-1]
        seq_results.append({"gt": bool(gt_seq), "pred": bool(pred_seq)})

    gt_seq_arr   = np.array([r["gt"]   for r in seq_results])
    pred_seq_arr = np.array([r["pred"] for r in seq_results])
    agree = float((gt_seq_arr == pred_seq_arr).mean())
    tp_seq = int((gt_seq_arr & pred_seq_arr).sum())
    fp_seq = int((~gt_seq_arr & pred_seq_arr).sum())
    fn_seq = int((gt_seq_arr & ~pred_seq_arr).sum())
    metrics_out["sequential"] = dict(agreement=agree, n=len(seq_results),
                                     gt_pos=int(gt_seq_arr.sum()),
                                     tp=tp_seq, fp=fp_seq, fn=fn_seq)
    print(f"\n  Sequential: agreement={agree:.3f}  TP={tp_seq} FP={fp_seq} FN={fn_seq}  "
          f"({int(gt_seq_arr.sum())}/{len(seq_results)} GT+)")

    tl_dir = out_dir / "timelines"
    for ep_idx in test_ids[:6]:
        da = l2_distances(ep_emb[ep_idx], z_pick)
        db = l2_distances(ep_emb[ep_idx], z_insert)
        plot_seq_timeline(da, db, ep_pick_m[ep_idx], ep_ins_m[ep_idx],
                          tau_pk_f1, tau_in_f1, f"FMB full Cosmos ep{ep_idx}",
                          tl_dir / f"seq_ep{ep_idx}.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)
    print(f"  Saved -> {out_dir / 'metrics.json'}")
    return metrics_out


# ---- iamlab evaluation -------------------------------------------------------

IAMLAB_REPO      = "lerobot/iamlab_cmu_pickup_insert"
IAMLAB_VIDEO_KEY = "observation.images.image"
IAMLAB_DONE_WIN  = 10


def evaluate_iamlab(enc: CosmosVAEEncoder, num_episodes: int, out_dir: Path,
                    pca_components: int = 256, cal_frac: float = 0.5, seed: int = 42):
    print("\n=== iamlab (Cosmos VAE full latent) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    info, data_df, ep_df = load_dataset_meta(IAMLAB_REPO)

    pick_eps   = ep_df[ep_df["tasks"].apply(
        lambda t: any("Pick up green" in s for s in t))]["episode_index"].tolist()
    insert_eps = ep_df[ep_df["tasks"].apply(
        lambda t: any("Insert" in s for s in t))]["episode_index"].tolist()
    print(f"  Pick green: {len(pick_eps)}, Insert: {len(insert_eps)}")

    rng    = np.random.default_rng(seed)
    n_each = min(num_episodes // 2, len(pick_eps), len(insert_eps))
    samp_pk = rng.choice(pick_eps,   n_each, replace=False).tolist()
    samp_in = rng.choice(insert_eps, n_each, replace=False).tolist()

    # Phase 1: encode raw latents
    ep_raw  : Dict[int, np.ndarray] = {}
    ep_done : Dict[int, np.ndarray] = {}
    ep_kind : Dict[int, str]        = {}

    def encode_set(eps, kind):
        for ep_idx in eps:
            row    = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
            frames = load_episode_frames(IAMLAB_REPO, row, IAMLAB_VIDEO_KEY)
            if not frames: print(f"    ep{ep_idx} no frames"); continue
            raw  = enc.encode_frames(frames)
            T    = len(raw)
            done = np.zeros(T, dtype=bool)
            done[-min(IAMLAB_DONE_WIN, T):] = True
            ep_raw[ep_idx]  = raw
            ep_done[ep_idx] = done
            ep_kind[ep_idx] = kind
            print(f"    ep{ep_idx} ({kind}): {T} frames")

    print("  Encoding pick ..."); encode_set(samp_pk, "pick")
    print("  Encoding insert ..."); encode_set(samp_in, "insert")

    v_pk   = [e for e in samp_pk if e in ep_raw]
    v_in   = [e for e in samp_in if e in ep_raw]
    n_cal  = max(1, int(len(v_pk) * cal_frac))
    cal_pk, test_pk = v_pk[:n_cal], v_pk[n_cal:]
    cal_in, test_in = v_in[:n_cal], v_in[n_cal:]
    if not test_pk: test_pk = v_pk
    if not test_in: test_in = v_in

    # Phase 2: fit PCA on calibration data
    cal_all = np.concatenate([ep_raw[e] for e in cal_pk + cal_in])
    enc.fit_pca(cal_all, n_components=pca_components)

    # Phase 3: project all episodes
    ep_emb = {e: enc.pca_transform(ep_raw[e]) for e in v_pk + v_in}

    def stack(ep_list):
        return (np.concatenate([ep_emb[e] for e in ep_list]),
                np.concatenate([ep_done[e] for e in ep_list]))

    cal_pk_e, cal_pk_d = stack(cal_pk)
    cal_in_e, cal_in_d = stack(cal_in)
    z_pick   = build_spec_latent(cal_pk_e, cal_pk_d)
    z_insert = build_spec_latent(cal_in_e, cal_in_d)
    print(f"\n  z_pick from {cal_pk_d.sum()} pos frames, z_insert from {cal_in_d.sum()}")

    tau_pk_f1, _, _, _ = sweep_f1(l2_distances(cal_pk_e, z_pick),   cal_pk_d.astype(int))
    tau_in_f1, _, _, _ = sweep_f1(l2_distances(cal_in_e, z_insert), cal_in_d.astype(int))
    tau_pk_cp = conformal_tau(l2_distances(cal_pk_e, z_pick),   cal_pk_d.astype(int))
    tau_in_cp = conformal_tau(l2_distances(cal_in_e, z_insert), cal_in_d.astype(int))
    print(f"  tau_pick F1={tau_pk_f1:.3f} CP={tau_pk_cp}  "
          f"tau_insert F1={tau_in_f1:.3f} CP={tau_in_cp}")

    metrics_out = {}
    for sn, z_spec, tau_f1, tau_cp, t_pos, t_neg in [
        ("pick",   z_pick,   tau_pk_f1, tau_pk_cp, test_pk, test_in),
        ("insert", z_insert, tau_in_f1, tau_in_cp, test_in, test_pk),
    ]:
        all_d, all_l = [], []
        for e in t_pos + t_neg:
            d  = l2_distances(ep_emb[e], z_spec)
            gt = ep_done[e].astype(int) if e in t_pos else np.zeros(len(d), dtype=int)
            all_d.append(d); all_l.append(gt)
        all_d = np.concatenate(all_d); all_l = np.concatenate(all_l)
        m_f1 = predicate_metrics(all_d, all_l, tau_f1)
        m_cp = predicate_metrics(all_d, all_l, tau_cp) if tau_cp else None
        print(f"  pi_{sn} tau_F1: F1={m_f1['f1']:.3f} P={m_f1['precision']:.3f} R={m_f1['recall']:.3f}")
        if m_cp:
            print(f"  pi_{sn} tau_CP: F1={m_cp['f1']:.3f} P={m_cp['precision']:.3f} R={m_cp['recall']:.3f}")
        plot_f1_curve(all_d, all_l, tau_f1, tau_cp,
                      f"iamlab Cosmos full -- pi_{sn}", out_dir / f"f1_{sn}.png")
        metrics_out[sn] = {"tau_F1": tau_f1, "tau_CP": tau_cp,
                           "test_tau_F1": m_f1, "test_tau_CP": m_cp}

    # Episode-level AUC
    from sklearn.metrics import roc_auc_score
    scores_pk = [l2_distances(ep_emb[e], z_pick).mean() for e in test_pk]
    scores_in = [l2_distances(ep_emb[e], z_pick).mean() for e in test_in]
    if scores_pk and scores_in:
        sc  = np.array(scores_pk + scores_in)
        lb  = np.array([0] * len(scores_pk) + [1] * len(scores_in))
        auc = roc_auc_score(lb, sc)
        print(f"\n  Episode-level AUC (pick vs insert): {auc:.4f}")
        print(f"  Avg dist: pick={np.mean(scores_pk):.3f}  insert={np.mean(scores_in):.3f}")
        metrics_out["episode_auc"] = float(auc)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)
    print(f"  Saved -> {out_dir / 'metrics.json'}")
    return metrics_out


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        choices=["fmb", "iamlab", "both"], default="both")
    parser.add_argument("--num-episodes",   type=int,   default=60)
    parser.add_argument("--vae-pth",        type=str,   required=True)
    parser.add_argument("--out-dir",        type=str,   default="./etl_results/lerobot_cosmos_full")
    parser.add_argument("--device",         type=str,   default="cuda")
    parser.add_argument("--pca-components", type=int,   default=256)
    parser.add_argument("--cal-frac",       type=float, default=0.5)
    args = parser.parse_args()

    enc = CosmosVAEEncoder(vae_pth=args.vae_pth, device=args.device)
    out = Path(args.out_dir)

    if args.dataset in ("fmb", "both"):
        evaluate_fmb(enc, args.num_episodes, out / "fmb",
                     pca_components=args.pca_components, cal_frac=args.cal_frac)
    if args.dataset in ("iamlab", "both"):
        # PCA is re-fit per-dataset so reset it
        enc.pca = None
        evaluate_iamlab(enc, args.num_episodes, out / "iamlab",
                        pca_components=args.pca_components, cal_frac=args.cal_frac)


if __name__ == "__main__":
    main()
