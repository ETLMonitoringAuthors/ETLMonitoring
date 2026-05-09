"""
run_etl_fast.py
---------------
Fast standalone evaluation of ETL variants vs PCA-kmeans.
Only loads obs_embeddings (skips RGB / action_preds) — much faster than the
full FIPER pipeline which loads all tensors into memory.

Usage:
    python scripts/run_etl_fast.py --tasks stacking sorting --n_phases 3

Produces a CSV + console table with AUROC and detection-time for each method.
"""

import argparse, glob, os, pickle, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from evaluation.utils import calculate_metrics
from evaluation.method_eval_classes.logpzo_eval import LogpZOModel, LogpZOTrainer


# ── data loading ──────────────────────────────────────────────────────────────

def load_episodes(rollout_dir, split):
    """Load obs_embeddings, agent_pos, and success labels from a split folder.

    Returns:
      episodes : list of [T, D] np.ndarray  (obs_embedding, padded if needed)
      agent_positions : list of [T, 2] np.ndarray  (xy agent pos, or None if absent)
      labels   : list of int  (0=success, 1=fail)
    """
    folder = os.path.join(rollout_dir, split)
    paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    episodes, agent_positions, labels = [], [], []
    for path in paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        meta = d.get("metadata", {}) if isinstance(d, dict) else {}
        success = meta.get("successful", meta.get("success", False))
        steps = d["rollout"] if isinstance(d, dict) and "rollout" in d else d
        raw, pos_raw = [], []
        for s in steps:
            if not isinstance(s, dict):
                continue
            for k in ("obs_embedding", "obs_embeddings", "embedding"):
                if k in s and s[k] is not None:
                    raw.append(np.array(s[k]).reshape(-1))
                    break
            if "agent_pos" in s and s["agent_pos"] is not None:
                pos_raw.append(np.array(s["agent_pos"])[:2])  # keep only xy
        if not raw:
            continue
        # Align to the dominant embedding size (handles history-stacking)
        max_dim = max(v.shape[0] for v in raw)
        embs = []
        for v in raw:
            if v.shape[0] < max_dim:
                v = np.concatenate([np.zeros(max_dim - v.shape[0]), v])
            embs.append(v)
        episodes.append(np.stack(embs))          # [T, D]
        agent_positions.append(
            np.stack(pos_raw) if len(pos_raw) == len(raw) else None
        )
        labels.append(0 if success else 1)       # 1 = fail
    print(f"  [{split}] {len(episodes)} episodes  "
          f"({sum(l==0 for l in labels)} success / {sum(l==1 for l in labels)} fail)")
    return episodes, agent_positions, labels


def detect_reversal_timesteps(pos_seq, n_reversals=1):
    """Return the n_reversals sharpest direction-reversal timesteps in pos_seq.

    Uses the cosine similarity between consecutive velocity vectors: a value
    close to -1 means the agent reversed direction (end of one phase, start
    of next).  Returns sorted list of timestep indices.
    """
    T = len(pos_seq)
    if T < 4 or n_reversals == 0:
        return []
    vel = np.diff(pos_seq, axis=0)          # [T-1, 2]
    speed = np.linalg.norm(vel, axis=1)
    dir_vec = vel / (speed[:, None] + 1e-12)
    cos_sim = (dir_vec[:-1] * dir_vec[1:]).sum(axis=1)  # [T-2]

    # Exclude first and last 10 % of the rollout from candidates
    margin = max(2, T // 10)
    cos_in_range = cos_sim.copy()
    cos_in_range[:margin] = 1.0
    cos_in_range[-(margin):] = 1.0

    found = []
    remaining = cos_in_range.copy()
    for _ in range(n_reversals):
        t = int(np.argmin(remaining))
        found.append(t)
        # Suppress a window of ±5 around this peak so we don't pick duplicates
        lo, hi = max(0, t - 5), min(len(remaining), t + 6)
        remaining[lo:hi] = 1.0
    return sorted(found)


# ── ETL methods ───────────────────────────────────────────────────────────────

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def mine_spec_latents(success_embs, K, success_pos=None):
    """
    K spec latents from calibration success rollouts.

    Strategy (in priority order):
      1. If agent_pos is available (success_pos is not None), detect the K-1
         sharpest direction reversals to find phase-transition frames.
         Phase k → embedding at the k-th reversal (or last frame for phase K-1).
      2. Fallback: proportional sampling — phase k → frame at (k+1)/K * T.

    The final frame of every success rollout anchors phase K-1 because
    FIPER truncates rollouts exactly at task completion.
    """
    use_pos = success_pos is not None and any(p is not None for p in success_pos)
    phase_latents = []
    for i, embs in enumerate(success_embs):
        T = len(embs)
        pm = []

        if use_pos and success_pos[i] is not None and K > 1:
            # Detect K-1 reversal timesteps → K-1 intermediate phase markers
            reversals = detect_reversal_timesteps(success_pos[i], n_reversals=K - 1)
            # Build K phase timestamps: reversals + final frame
            # If we got fewer than K-1 reversals, pad with proportional fallback
            while len(reversals) < K - 1:
                k_missing = len(reversals)
                reversals.append(min(T - 1, int((k_missing + 1) * T / K) - 1))
            reversals = sorted(reversals)
            phase_ts = reversals + [T - 1]
        else:
            # Proportional fallback
            phase_ts = [min(T - 1, int((k + 1) * T / K) - 1) for k in range(K)]

        for t in phase_ts:
            pm.append(embs[t])
        phase_latents.append(np.stack(pm))  # [K, D]

    spec = np.stack(phase_latents).mean(0)  # [K, D]
    mean_T = float(np.mean([len(e) for e in success_embs]))
    return spec, mean_T


def etl_scores(test_embs, spec_latents, mean_T, K, mode="temporal"):
    """Compute episode-level failure scores.

    Modes:
      single   — per-step dist to goal spec; episode score = max over steps.
      temporal — per-step dist to phase-assigned spec (forced ordering);
                 episode score = max over steps.
      seq      — per-step min dist over all specs (OR over targets per step);
                 episode score = max over steps.
      or       — for each spec k, find the closest the episode ever got to it
                 (min_t dist); episode score = max_k of those minimums.
                 Reads: "all K milestones must eventually be reached."
                 No assumed ordering; OR over time, AND over milestones.
    """
    all_scores = []
    for embs in test_embs:
        if mode == "or":
            # For each spec latent k: how close did this episode ever get?
            # min_t dist(z_t, spec_k) — small = phase k was visited
            # Episode failure score = max_k min_t dist → fails if ANY phase missed
            per_phase_min = []
            for k in range(K):
                min_d = min(cosine_dist(z, spec_latents[k]) for z in embs)
                per_phase_min.append(min_d)
            # Return as a length-1 list so compute_metrics (which does max) gets
            # the episode score directly.
            all_scores.append([max(per_phase_min)])
        else:
            ep_scores = []
            for t, z in enumerate(embs):
                if mode == "single":
                    ep_scores.append(cosine_dist(z, spec_latents[-1]))
                elif mode == "temporal":
                    phase = max(0, min(K-1, int(t / mean_T * K)))
                    ep_scores.append(cosine_dist(z, spec_latents[phase]))
                elif mode == "seq":
                    ep_scores.append(min(cosine_dist(z, spec_latents[k]) for k in range(K)))
            all_scores.append(ep_scores)
    return all_scores


# ── logpZO baseline (flow matching density estimator, FIPER's implementation) ─

def logpzo_train(calib_embs_list, num_epochs=300, batch_size=256, lr=1e-3, device=None):
    """Train FIPER's LogpZO flow model on calibration obs_embeddings."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.tensor(np.concatenate(calib_embs_list, axis=0), dtype=torch.float32).to(device)
    dim = data.shape[1]
    model = LogpZOModel(input_dim=dim).to(device)

    from types import SimpleNamespace
    cfg = SimpleNamespace(num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr)
    trainer = LogpZOTrainer(model=model, num_epochs=cfg.num_epochs,
                            batch_size=cfg.batch_size, learning_rate=cfg.learning_rate)
    trainer.train(embeddings=data)
    model.load_state_dict(trainer.best_model)
    model.eval()
    return model, device


def logpzo_scores(model, device, embs_list):
    """Per-episode uncertainty scores using ||noise||² from the flow model."""
    all_scores = []
    with torch.no_grad():
        for embs in embs_list:
            z = torch.tensor(embs, dtype=torch.float32).to(device)
            t = torch.zeros(len(z), 1, device=device)
            noise = z + model(z, t)
            scores = (noise ** 2).sum(dim=1).cpu().numpy().tolist()
            all_scores.append(scores)
    return all_scores


# ── PCA-kmeans baseline ───────────────────────────────────────────────────────

def pca_kmeans_scores(calib_embs, test_embs, n_pc=10, n_cl=64):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    all_embs = np.concatenate(calib_embs, axis=0)
    n_pc = min(n_pc, all_embs.shape[1], len(all_embs))
    pca = PCA(n_components=n_pc).fit(all_embs)
    reduced = pca.transform(all_embs)
    n_cl = min(n_cl, len(reduced))
    km = KMeans(n_clusters=n_cl, random_state=42, n_init=10).fit(reduced)
    scores = []
    for embs in test_embs:
        ep = []
        for z in embs:
            r = pca.transform(z[None])
            ep.append(float(np.linalg.norm(km.cluster_centers_ - r, axis=1).min()))
        scores.append(ep)
    return scores


# ── CP threshold + metrics ────────────────────────────────────────────────────

def cp_threshold(calib_embs, score_fn, quantile=0.95):
    """Max-score over each calibration episode, take quantile as threshold."""
    max_scores = [max(score_fn(ep)) for ep in calib_embs]
    return float(np.quantile(max_scores, quantile))


def compute_metrics(test_scores, test_labels, threshold):
    """AUROC, accuracy + F1 at threshold, and mean detection time for fail eps."""
    from sklearn.metrics import roc_auc_score
    max_scores = [max(s) for s in test_scores]
    try:
        auroc = float(roc_auc_score(test_labels, max_scores))
    except Exception:
        auroc = float("nan")

    preds = [1 if m > threshold else 0 for m in max_scores]
    acc = float(np.mean([p == l for p, l in zip(preds, test_labels)]))

    tp = sum(1 for p, l in zip(preds, test_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, test_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, test_labels) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # F1-optimal threshold (sweep). Track the accuracy at that same threshold
    # so we report the *best F1 operating point's* accuracy — analogous to
    # MW's `f1_agreement` and avoids any test-accuracy overfitting.
    sorted_unique = sorted(set(max_scores))
    best_f1 = 0.0
    acc_at_f1opt = acc
    prec_at_f1opt = precision
    rec_at_f1opt  = recall
    for tau in sorted_unique:
        p_sweep = [1 if m > tau else 0 for m in max_scores]
        tp_s = sum(1 for p, l in zip(p_sweep, test_labels) if p == 1 and l == 1)
        fp_s = sum(1 for p, l in zip(p_sweep, test_labels) if p == 1 and l == 0)
        fn_s = sum(1 for p, l in zip(p_sweep, test_labels) if p == 0 and l == 1)
        ps = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        rs = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        fs = 2 * ps * rs / (ps + rs) if (ps + rs) > 0 else 0.0
        if fs > best_f1:
            best_f1 = fs
            acc_at_f1opt = float(np.mean([p == l for p, l in zip(p_sweep, test_labels)]))
            prec_at_f1opt = ps
            rec_at_f1opt  = rs

    det_times = []
    for scores, label in zip(test_scores, test_labels):
        if label == 0: continue
        for t, s in enumerate(scores):
            if s > threshold:
                det_times.append(t); break
        else:
            det_times.append(len(scores))
    det_time = float(np.mean(det_times)) if det_times else float("nan")

    return {
        "auroc": auroc,
        "accuracy": acc,
        "accuracy_at_f1opt": float(acc_at_f1opt),
        "f1_cp": f1,
        "f1_optimal": float(best_f1),
        "precision_cp": precision,
        "recall_cp": recall,
        "precision_at_f1opt": float(prec_at_f1opt),
        "recall_at_f1opt": float(rec_at_f1opt),
        "det_time": det_time,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_task(task, data_root, K=3, n_pc=10, n_cl=64, quantile=0.95, n_logpzo_epochs=300):
    rollout_dir = os.path.join(data_root, task, "rollouts")
    print(f"\n{'='*55}")
    print(f"  Task: {task.upper()}")
    print(f"{'='*55}")

    calib_embs, calib_pos, calib_labels = load_episodes(rollout_dir, "calibration")
    test_embs,  test_pos,  test_labels  = load_episodes(rollout_dir, "test")

    calib_success     = [e for e, l in zip(calib_embs, calib_labels) if l == 0]
    calib_success_pos = [p for p, l in zip(calib_pos,  calib_labels) if l == 0]
    print(f"  Calibration success episodes: {len(calib_success)}")

    if not calib_success:
        print("  No calibration success episodes — skipping.")
        return {}

    # Mine ETL spec latents — use agent_pos reversals when available
    has_pos = any(p is not None for p in calib_success_pos)
    spec_latents, mean_T = mine_spec_latents(calib_success, K,
                                              calib_success_pos if has_pos else None)
    method_note = "agent_pos reversal" if has_pos else "proportional fallback"
    print(f"  ETL spec latents: K={K}, shape={spec_latents.shape}, mean_T={mean_T:.1f}  [{method_note}]")

    results = {}
    print(f"\n  {'Method':<22} {'AUROC':>7} {'F1_CP':>7} {'F1_opt':>7} {'Acc':>6} {'DetTime':>9}  (q={quantile} CP)")
    print(f"  {'-'*70}")

    # ── ETL methods ─────────────────────────────────────────────────────────
    for mode, label in [
        ("or",       "ETL-or"),
        ("single",   "ETL-single"),
        ("temporal", "ETL-temporal"),
        ("seq",      "ETL-seq"),
    ]:
        calib_s = etl_scores(calib_success, spec_latents, mean_T, K, mode)
        tau = float(np.quantile([max(s) for s in calib_s], quantile))
        test_s  = etl_scores(test_embs, spec_latents, mean_T, K, mode)
        m = compute_metrics(test_s, test_labels, tau)
        results[label] = m
        print(f"  {label:<22} {m['auroc']:>7.3f} {m['f1_cp']:>7.3f} {m['f1_optimal']:>7.3f} {m['accuracy']:>6.3f} {m['det_time']:>9.1f}")

    # ── PCA-kmeans baseline ──────────────────────────────────────────────────
    calib_pca = pca_kmeans_scores(calib_success, calib_success, n_pc, n_cl)
    tau_pca   = float(np.quantile([max(s) for s in calib_pca], quantile))
    test_pca  = pca_kmeans_scores(calib_success, test_embs, n_pc, n_cl)
    m = compute_metrics(test_pca, test_labels, tau_pca)
    results["PCA-kmeans"] = m
    print(f"  {'PCA-kmeans':<22} {m['auroc']:>7.3f} {m['f1_cp']:>7.3f} {m['f1_optimal']:>7.3f} {m['accuracy']:>6.3f} {m['det_time']:>9.1f}")

    # ── logpZO baseline ──────────────────────────────────────────────────────
    print(f"  Training logpZO flow model on {len(calib_success)} success episodes...", end=" ", flush=True)
    lp_model, lp_device = logpzo_train(calib_success, num_epochs=n_logpzo_epochs)
    print("done")
    calib_lp = logpzo_scores(lp_model, lp_device, calib_success)
    tau_lp   = float(np.quantile([max(s) for s in calib_lp], quantile))
    test_lp  = logpzo_scores(lp_model, lp_device, test_embs)
    m = compute_metrics(test_lp, test_labels, tau_lp)
    results["logpZO"] = m
    print(f"  {'logpZO':<22} {m['auroc']:>7.3f} {m['f1_cp']:>7.3f} {m['f1_optimal']:>7.3f} {m['accuracy']:>6.3f} {m['det_time']:>9.1f}")

    # ETL advantage on temporal detection
    if "ETL-temporal" in results and "PCA-kmeans" in results:
        dt_etl = results["ETL-temporal"]["det_time"]
        dt_pca = results["PCA-kmeans"]["det_time"]
        if not (np.isnan(dt_etl) or np.isnan(dt_pca)):
            delta = dt_pca - dt_etl
            print(f"\n  ETL-temporal detects failures {delta:.1f} steps EARLIER than PCA-kmeans")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["stacking", "sorting"],
                   choices=["stacking", "sorting", "push_t", "pretzel", "push_chair"])
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--n_phases", type=int, default=3, help="K spec latents for ETL")
    p.add_argument("--n_pc", type=int, default=10, help="PCA components for baseline")
    p.add_argument("--n_clusters", type=int, default=64, help="KMeans clusters for baseline")
    p.add_argument("--quantile", type=float, default=0.95, help="CP quantile for threshold")
    p.add_argument("--logpzo_epochs", type=int, default=300, help="Training epochs for logpZO flow model")
    p.add_argument("--out", type=str, default="etl_fast_results.json",
                   help="Output filename under data/results/")
    args = p.parse_args()

    all_results = {}
    for task in args.tasks:
        all_results[task] = run_task(
            task, args.data_root, args.n_phases, args.n_pc, args.n_clusters, args.quantile,
            args.logpzo_epochs,
        )

    # Summary table across tasks
    print(f"\n{'='*65}")
    print("SUMMARY (mean across tasks)")
    print(f"{'='*65}")
    methods = list(next(iter(all_results.values())).keys())
    print(f"{'Method':<22} {'AUROC':>7} {'F1_CP':>7} {'F1_opt':>7} {'Acc':>6} {'DetTime':>9}")
    print("-" * 70)
    for m in methods:
        aurocs = [r[m]["auroc"] for r in all_results.values() if m in r]
        f1s_cp = [r[m]["f1_cp"] for r in all_results.values() if m in r]
        f1s_opt = [r[m]["f1_optimal"] for r in all_results.values() if m in r]
        accs   = [r[m]["accuracy"] for r in all_results.values() if m in r]
        dts    = [r[m]["det_time"] for r in all_results.values() if m in r]
        print(
            f"{m:<22} {np.nanmean(aurocs):>7.3f} {np.nanmean(f1s_cp):>7.3f} "
            f"{np.nanmean(f1s_opt):>7.3f} {np.nanmean(accs):>6.3f} {np.nanmean(dts):>9.1f}"
        )

    import json
    out = ROOT / "data" / "results" / args.out
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
