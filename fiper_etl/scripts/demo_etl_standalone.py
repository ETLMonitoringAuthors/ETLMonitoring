"""
demo_etl_standalone.py
----------------------
Self-contained demo that loads rollout PKL files from a directory,
runs ETL-temporal and PCA-kmeans monitoring, and prints a comparison.

This script does NOT require the full Hydra / FIPER pipeline.
Use it to quickly verify ETL on any set of rollout pkl files.

Expected rollout pkl format (either):
  - List of dicts: [{obs_embedding: np.ndarray[D], success: bool, ...}, ...]
  - Dict: {metadata: {..., success: bool}, rollout: [{obs_embedding: ...}, ...]}

Usage:
    python scripts/demo_etl_standalone.py \
        --rollout_dir data/stacking/rollouts \
        --calib_split 0.5 \
        --n_phases 3 \
        --last_k 10
"""

import argparse, glob, os, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def load_rollout(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "rollout" in data:
        meta = data.get("metadata", {})
        success = meta.get("success", meta.get("successful", False))
        steps = data["rollout"]
    elif isinstance(data, list):
        steps = data
        meta = steps[0] if steps and isinstance(steps[0], dict) and "success" in steps[0] else {}
        success = meta.get("success", meta.get("successful", False))
    else:
        raise ValueError(f"Unknown rollout format in {path}")

    embs = []
    for step in steps:
        if isinstance(step, dict):
            for key in ("obs_embedding", "obs_embeddings", "embedding", "latent"):
                if key in step:
                    embs.append(np.array(step[key]).reshape(-1))
                    break
    return np.stack(embs) if embs else None, bool(success)


def auroc_score(scores_by_ep, labels):
    """Compute AUROC from per-episode max scores and binary labels (1=fail)."""
    from sklearn.metrics import roc_auc_score
    max_scores = [max(s) for s in scores_by_ep]
    try:
        return float(roc_auc_score(labels, max_scores))
    except Exception:
        return float("nan")


def detection_time(scores_by_ep, labels, threshold):
    """Mean step at which the alarm fires for failed episodes."""
    times = []
    for scores, label in zip(scores_by_ep, labels):
        if not label:
            continue
        for t, s in enumerate(scores):
            if s > threshold:
                times.append(t)
                break
        else:
            times.append(len(scores))
    return float(np.mean(times)) if times else float("nan")


# ── ETL monitoring ────────────────────────────────────────────────────────────

def run_etl(calib_embs_list, test_embs_list, test_labels, n_phases=3, last_k=10):
    """ETL-temporal monitor.  Returns per-episode scores for test rollouts."""
    calib_success = [e for e, l in zip(calib_embs_list, [True]*len(calib_embs_list)) if e is not None]

    # Mine K spec latents
    all_phase_means = []
    lengths = []
    for embs in calib_success:
        T = len(embs)
        lengths.append(T)
        phase_means = []
        for k in range(n_phases):
            s = int(k * T / n_phases); e = int((k+1) * T / n_phases)
            if e <= s: e = s + 1
            phase_means.append(embs[s:e].mean(0))
        all_phase_means.append(np.stack(phase_means))
    spec_latents = np.stack(all_phase_means).mean(0)  # [K, D]
    T_mean = float(np.mean(lengths)) if lengths else 50.0

    # Also compute single-phase goal spec
    goal_specs = [embs[-last_k:].mean(0) for embs in calib_success]
    z_goal = np.stack(goal_specs).mean(0) if goal_specs else spec_latents[-1]

    # Score test rollouts
    scores_temporal, scores_single, scores_seq = [], [], []
    for embs in test_embs_list:
        if embs is None:
            scores_temporal.append([1.0]); scores_single.append([1.0]); scores_seq.append([1.0])
            continue
        t_scores, s_scores, seq_scores = [], [], []
        for t, z in enumerate(embs):
            phase = max(0, min(n_phases-1, int(t / T_mean * n_phases)))
            t_scores.append(cosine_dist(z, spec_latents[phase]))
            s_scores.append(cosine_dist(z, z_goal))
            seq_scores.append(min(cosine_dist(z, spec_latents[k]) for k in range(n_phases)))
        scores_temporal.append(t_scores)
        scores_single.append(s_scores)
        scores_seq.append(seq_scores)

    return scores_temporal, scores_single, scores_seq, spec_latents


# ── PCA-kmeans baseline ───────────────────────────────────────────────────────

def run_pca_kmeans(calib_embs_list, test_embs_list, n_components=10, n_clusters=64):
    all_embs = np.concatenate(calib_embs_list, axis=0)
    pca = PCA(n_components=min(n_components, all_embs.shape[1])).fit(all_embs)
    reduced = pca.transform(all_embs)
    km = KMeans(n_clusters=min(n_clusters, len(reduced)), random_state=42).fit(reduced)

    scores = []
    for embs in test_embs_list:
        if embs is None:
            scores.append([1.0]); continue
        ep_scores = []
        for z in embs:
            r = pca.transform(z[None])
            d = np.linalg.norm(km.cluster_centers_ - r, axis=1).min()
            ep_scores.append(float(d))
        scores.append(ep_scores)
    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--calib_split", type=float, default=0.5,
                   help="Fraction of rollouts used for calibration")
    p.add_argument("--n_phases", type=int, default=3)
    p.add_argument("--last_k", type=int, default=10)
    p.add_argument("--n_pca", type=int, default=10)
    p.add_argument("--n_clusters", type=int, default=64)
    args = p.parse_args()

    # Load all rollouts
    paths = sorted(glob.glob(os.path.join(args.rollout_dir, "**/*.pkl"), recursive=True)
                   + glob.glob(os.path.join(args.rollout_dir, "*.pkl")))
    print(f"Found {len(paths)} rollout files in {args.rollout_dir}")

    all_embs, all_labels = [], []
    for path in paths:
        try:
            embs, success = load_rollout(path)
            all_embs.append(embs)
            all_labels.append(0 if success else 1)  # 1=fail
        except Exception as e:
            print(f"  skip {path}: {e}")

    print(f"Loaded {len(all_embs)} rollouts  "
          f"({sum(l==0 for l in all_labels)} success, {sum(l==1 for l in all_labels)} fail)")

    # Split calibration / test
    n = len(all_embs)
    n_calib = max(1, int(n * args.calib_split))
    calib_embs = [e for e in all_embs[:n_calib] if e is not None]
    test_embs  = all_embs[n_calib:]
    test_labels = all_labels[n_calib:]

    calib_success = [e for e, lab in zip(all_embs[:n_calib], all_labels[:n_calib])
                     if lab == 0 and e is not None]
    print(f"Calibration: {len(calib_success)} success rollouts")
    print(f"Test: {len(test_embs)} rollouts")

    # Run methods
    sc_temporal, sc_single, sc_seq, spec_latents = run_etl(
        calib_success, test_embs, test_labels, args.n_phases, args.last_k
    )
    sc_pca = run_pca_kmeans(calib_embs, test_embs, args.n_pca, args.n_clusters)

    # Compute metrics — use 95th percentile of calibration max scores as threshold
    calib_temporal, _, _, _ = run_etl(calib_success, calib_success, [], args.n_phases, args.last_k)
    tau_temporal = float(np.quantile([max(s) for s in calib_temporal], 0.95))
    tau_single   = float(np.quantile([max(s) for s in _], 0.95)) if _ else 0.5
    calib_pca    = run_pca_kmeans(calib_embs, calib_success, args.n_pca, args.n_clusters)
    tau_pca      = float(np.quantile([max(s) for s in calib_pca], 0.95))

    print(f"\n{'Method':<25} {'AUROC':>8} {'DetTime':>10}  (q=0.95 CP threshold)")
    print("-" * 48)
    for name, scores, tau in [
        ("ETL-temporal (ours)", sc_temporal, tau_temporal),
        ("ETL-single (ours)",   sc_single,   tau_temporal),
        ("ETL-seq (ours)",      sc_seq,       tau_temporal),
        ("PCA-kmeans",          sc_pca,       tau_pca),
    ]:
        a  = auroc_score(scores, test_labels)
        dt = detection_time(scores, test_labels, tau)
        print(f"  {name:<23} {a:>8.3f} {dt:>10.1f}")

    print(f"\nSpec latents (K={args.n_phases}) shape: {spec_latents.shape}")
    print(f"ETL formula: " + " ∧ F(".join([f"F(near_{k}" for k in range(args.n_phases)]) + ")"*args.n_phases)


if __name__ == "__main__":
    main()
