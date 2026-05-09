"""
etl_eval.py
-----------
ETL (Embedding Temporal Logic) failure monitor for the FIPER benchmark.

Two variants are provided, both fitting the FIPER BaseEvalClass interface:

ETLEval  (method name: "etl")
  Single-phase WHERE spec.  Mines one goal spec latent z* from the LAST
  `last_k_frames` of every calibration-success rollout and monitors
  cosine distance d(z_t, z*) at each step.

  This already differs from PCA-kmeans and logpZO because it explicitly
  encodes "what success looks like" rather than "what any training
  observation looks like."

ETLTemporalEval  (method name: "etl_temporal")
  K-phase sequential WHERE spec.  Divides each calibration-success
  rollout into K equal-length temporal segments, averages embeddings
  per segment across rollouts to obtain K spec latents z_0, ..., z_{K-1}.
  At test time the current phase is estimated as

      k(t) = floor(t / T_mean * K)   clipped to [0, K-1]

  and the score is cosine distance to z_{k(t)}.

  Key advantage over all FIPER baselines:
    PCA-kmeans / logpZO / RND-OE treat the full success distribution as
    a single unordered set.  They cannot detect "temporal mis-ordering"
    failures such as the robot reaching the goal too early (before grasping)
    or getting stuck in phase 1 while time runs out.

    ETL-temporal fires high whenever the robot is at the WRONG PART of
    the success manifold for its current stage of the task — a strictly
    more informative signal for sequential manipulation tasks (SORTING,
    STACKING, PRETZEL, PUSH-CHAIR).

Both classes expose the ETL formula string for logging / reporting.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from .base_eval_class import BaseEvalClass


# ── helpers ───────────────────────────────────────────────────────────────────

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical, 2 = antipodal."""
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def _mine_spec_latents(
    per_rollout_embs: List[np.ndarray],
    K: int,
) -> np.ndarray:
    """
    For each rollout, divide into K equal-length segments and compute
    the mean embedding per segment.  Average across rollouts.

    Returns spec_latents of shape [K, D].
    """
    all_phase_means = []  # list of [K, D] arrays, one per rollout
    for embs in per_rollout_embs:
        T = len(embs)
        phase_means = []
        for k in range(K):
            s = int(k * T / K)
            e = int((k + 1) * T / K)
            if e <= s:
                e = s + 1
            phase_means.append(embs[s:e].mean(axis=0))
        all_phase_means.append(np.stack(phase_means))  # [K, D]
    return np.stack(all_phase_means).mean(axis=0)  # [K, D]


def _etl_formula_str(K: int) -> str:
    def nest(k):
        if k == K - 1:
            return f"F(near_{k})"
        return f"F(near_{k} ∧ {nest(k + 1)})"
    return nest(0)


# ── single-phase ETL ──────────────────────────────────────────────────────────

class ETLEval(BaseEvalClass):
    """
    Single-phase ETL monitor.

    Mines goal spec latent z* = mean of last `last_k_frames` embeddings
    of each calibration success rollout.  Score = cosine_dist(z_t, z*).

    ETL formula:  F(near_goal)  where  near_goal(t) ≡ d(z_t, z*) < τ
    """

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        self.spec_latent: Optional[np.ndarray] = None
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def _execute_preprocessing(self):
        last_k = int(getattr(self.cfg, "last_k_frames", 10))

        # Collect per-rollout goal embeddings from calibration success rollouts
        goal_embs = []
        for rollout in self.dataset.iterate_episodes(
            subset="calibration",
            required_tensors=["obs_embeddings"],
            with_success_labels=True,
        ):
            if not rollout["successful"]:
                continue
            embs = np.array(rollout["obs_embeddings"])  # [T, D]
            goal_embs.append(embs[-last_k:].mean(axis=0))

        if not goal_embs:
            # Fallback: use all calibration embeddings
            all_embs = self.dataset.get_subset(
                subset="calibration", required_tensors="obs_embeddings"
            )["obs_embeddings"]
            self.spec_latent = np.array(all_embs).mean(axis=0)
        else:
            self.spec_latent = np.stack(goal_embs).mean(axis=0)  # [D]

        print(f"[ETL] Goal spec latent shape: {self.spec_latent.shape}  "
              f"(mined from {len(goal_embs)} success rollouts, last {last_k} frames)")
        print(f"[ETL] ETL formula: F(near_goal)")

    def calculate_uncertainty_score(self, rollout_tensor_dict: dict, **kwargs) -> float:
        emb = np.array(rollout_tensor_dict["obs_embeddings"])
        if emb.ndim > 1:
            emb = emb.reshape(-1)[-self.spec_latent.shape[0]:]
        return _cosine_dist(emb, self.spec_latent)


# ── multi-phase temporal ETL ──────────────────────────────────────────────────

class ETLTEMPORALEval(BaseEvalClass):
    """
    K-phase temporal ETL monitor.

    Mines K spec latents z_0, ..., z_{K-1} by dividing calibration
    success rollouts into K equal temporal segments.  At test time,
    the current phase k(t) = floor(t / T_mean * K) selects which spec
    latent to compare against.

    Score = cosine_dist(z_t, z_{k(t)})

    ETL formula: F(near_0 ∧ F(near_1 ∧ … ∧ F(near_{K-1})))

    Why this beats baselines on sequential manipulation:
      • PCA-kmeans / logpZO score = distance to nearest success cluster
        (temporally unordered) → cannot detect "stuck in phase 1".
      • ETL-temporal score rises when the robot is at the WRONG PART of
        the success manifold for its current time-slot, giving an earlier
        and more precise failure signal on tasks like SORTING / STACKING.
    """

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        self.spec_latents: Optional[np.ndarray] = None  # [K, D]
        self.mean_success_length: float = 1.0
        self._step_idx: int = 0          # reset each rollout in _process_one_rollout
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def _execute_preprocessing(self):
        K = int(getattr(self.cfg, "n_phases", 3))

        # Collect per-rollout embedding sequences from calibration success rollouts
        per_rollout_embs: List[np.ndarray] = []
        rollout_lengths: List[int] = []
        for rollout in self.dataset.iterate_episodes(
            subset="calibration",
            required_tensors=["obs_embeddings"],
            with_success_labels=True,
        ):
            if not rollout["successful"]:
                continue
            embs = np.array(rollout["obs_embeddings"])  # [T, D]
            per_rollout_embs.append(embs)
            rollout_lengths.append(len(embs))

        if not per_rollout_embs:
            # Fallback: flat mean as single spec
            all_embs = self.dataset.get_subset(
                subset="calibration", required_tensors="obs_embeddings"
            )["obs_embeddings"]
            flat = np.array(all_embs).mean(axis=0)
            self.spec_latents = np.tile(flat[None], (K, 1))
            self.mean_success_length = 50.0
        else:
            self.spec_latents = _mine_spec_latents(per_rollout_embs, K)
            self.mean_success_length = float(np.mean(rollout_lengths))

        self.n_phases = K
        print(f"[ETL-temporal] K={K} spec latents {self.spec_latents.shape}  "
              f"mean_success_length={self.mean_success_length:.1f}")
        print(f"[ETL-temporal] ETL formula: {_etl_formula_str(K)}")

    # ── override _process_one_rollout to inject step index ──────────────────

    def _process_one_rollout(self, rollout_dict: dict):
        """Override to pass the step index into calculate_uncertainty_score."""
        rollout_length = rollout_dict[self.required_tensors[0]].shape[0]
        import time
        inference_times = []
        uncertainty_scores_one_rollout = []
        for i in range(rollout_length):
            new_dict = {k: rollout_dict[k][i] for k in rollout_dict if k != "successful"}
            t0 = time.time()
            score = self.calculate_uncertainty_score(new_dict, step_idx=i)
            inference_times.append(time.time() - t0)
            if self.cfg.get("handle_zero_thresholds", {"style": "noop"}).get("style") == "add_small_score":
                score = score + 1e-6
            uncertainty_scores_one_rollout.append(score)
        return uncertainty_scores_one_rollout, float(np.mean(inference_times))

    def calculate_uncertainty_score(
        self, rollout_tensor_dict: dict, step_idx: int = 0, **kwargs
    ) -> float:
        emb = np.array(rollout_tensor_dict["obs_embeddings"])
        D = self.spec_latents.shape[1]
        if emb.ndim > 1:
            emb = emb.reshape(-1)[-D:]

        # Current phase estimated from elapsed time
        phase = int(step_idx / self.mean_success_length * self.n_phases)
        phase = max(0, min(self.n_phases - 1, phase))

        return _cosine_dist(emb, self.spec_latents[phase])


# ── sequential robustness variant ─────────────────────────────────────────────

class ETLSEQEval(BaseEvalClass):
    """
    Sequential-robustness ETL monitor.

    Score = min_k  cosine_dist(z_t, z_k)   (minimum over ALL K phases).

    This is a smooth proxy for the sequential formula: if the robot is
    close to ANY spec latent, the score is low (alarm suppressed).  If
    it wanders away from all spec latents, the score rises.

    Unlike ETLTEMPORALEval, this variant does NOT require knowing the
    current phase — it fires whenever the robot is globally off-manifold.
    It is therefore more similar to logpZO/PCA-kmeans in spirit, but
    uses cosine distance to interpretable ETL spec latents rather than
    a learned density or cluster.
    """

    def __init__(self, cfg, method_name, device, task_data_path, dataset, **kwargs):
        self.spec_latents: Optional[np.ndarray] = None
        super().__init__(cfg, method_name, device, task_data_path, dataset, **kwargs)

    def _execute_preprocessing(self):
        K = int(getattr(self.cfg, "n_phases", 3))
        per_rollout_embs: List[np.ndarray] = []
        for rollout in self.dataset.iterate_episodes(
            subset="calibration",
            required_tensors=["obs_embeddings"],
            with_success_labels=True,
        ):
            if not rollout["successful"]:
                continue
            per_rollout_embs.append(np.array(rollout["obs_embeddings"]))

        if not per_rollout_embs:
            all_embs = np.array(self.dataset.get_subset(
                subset="calibration", required_tensors="obs_embeddings"
            )["obs_embeddings"])
            self.spec_latents = np.tile(all_embs.mean(0)[None], (K, 1))
        else:
            self.spec_latents = _mine_spec_latents(per_rollout_embs, K)

        self.n_phases = K
        print(f"[ETL-seq] K={K} spec latents {self.spec_latents.shape}")
        print(f"[ETL-seq] ETL formula: {_etl_formula_str(K)}  (score = min-distance)")

    def calculate_uncertainty_score(self, rollout_tensor_dict: dict, **kwargs) -> float:
        emb = np.array(rollout_tensor_dict["obs_embeddings"])
        D = self.spec_latents.shape[1]
        if emb.ndim > 1:
            emb = emb.reshape(-1)[-D:]
        dists = [_cosine_dist(emb, self.spec_latents[k]) for k in range(self.n_phases)]
        return float(np.min(dists))
