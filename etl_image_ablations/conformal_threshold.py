"""
Class-conditional split conformal thresholds for distance-to-goal signals.

Motivation (AnySafeReachability / sweeper_cp-style percentile mining, formalized):
- Pool calibration scores from *labeled* timesteps split by class.
- For the success / high-reward class, use one-sided split conformal quantiles on
  the nonconformity score (here: L2 distance to goal manifold — smaller is more
  "conformal" with success).

References:
- Split conformal prediction (finite-sample): use order statistic
  k = ceil((n+1)(1-alpha)) on sorted calibration scores.
- Class-conditional: calibrate the success threshold using only calibration
  timesteps in the positive class (success or high reward), not pooled with negatives.

This module does NOT depend on hydra/torch env; it only needs torch tensors.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def build_positive_mask(
    reward: torch.Tensor,
    success: torch.Tensor,
    reward_quantile: float = 0.9,
) -> torch.Tensor:
    """Positive class: simulator success or top-quantile reward (same mining idea as before)."""
    reward = reward.float()
    success = success.float()
    rq = torch.quantile(reward, reward_quantile)
    pos = (success >= 0.99) | (reward >= rq)
    if pos.sum() == 0:
        pos = reward >= reward.mean()
    return pos


def cal_test_masks(T: int, cal_frac: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prefix calibration: first T_cal steps, remainder is test.
    Ensures at least one calibration step and one test step when T >= 2.
    """
    if T < 2:
        cal = torch.ones(T, dtype=torch.bool)
        test = torch.zeros(T, dtype=torch.bool)
        return cal, test, T
    T_cal = int(cal_frac * T)
    T_cal = max(1, min(T_cal, T - 1))
    cal = torch.zeros(T, dtype=torch.bool)
    cal[:T_cal] = True
    test = ~cal
    return cal, test, T_cal


def _f1_from_confusion(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def metrics_on_mask(
    pred: torch.Tensor,
    gt_success: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Precision / recall / F1 on timesteps where mask is True."""
    p = pred[mask]
    g = gt_success[mask]
    tp = (p & g).sum().item()
    fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item()
    prec, rec, f1 = _f1_from_confusion(tp, fp, fn)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_eval": float(mask.sum().item()),
    }


def threshold_median_positive_cal(
    distance: torch.Tensor,
    positive_mask: torch.Tensor,
    cal_mask: torch.Tensor,
) -> Tuple[float, int]:
    """Median distance on calibration ∩ positive (legacy-style mining, but cal-only)."""
    m = cal_mask & positive_mask
    d = distance[m]
    n = int(d.numel())
    if n == 0:
        d = distance[positive_mask]
        n = int(d.numel())
    if n == 0:
        return float(torch.median(distance).item()), 0
    return float(torch.median(d).item()), n


def threshold_class_conditional_split_cp(
    distance: torch.Tensor,
    positive_mask: torch.Tensor,
    cal_mask: torch.Tensor,
    alpha: float,
) -> Tuple[float, int, int]:
    """
    One-sided split CP on the success class only (class-conditional).

    Nonconformity score = distance (smaller aligns with success).
    Calibration set: indices in cal_mask with positive_mask.

    Finite-sample quantile index (exchangeable scores):
      k = ceil((n + 1) * (1 - alpha)), clamp to [1, n].
    Threshold = k-th smallest distance among n calibration positives.

    Returns (threshold, n_cal_pos, k_used).
    """
    m = cal_mask & positive_mask
    d = distance[m].float()
    n = int(d.numel())
    if n == 0:
        # Fallback: any positive in full trajectory
        d = distance[positive_mask].float()
        n = int(d.numel())
    if n == 0:
        return float(torch.median(distance).item()), 0, 0
    sorted_d, _ = torch.sort(d)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = max(1, min(k, n))
    return float(sorted_d[k - 1].item()), n, k


def dynamic_thresholds_cal_test(
    distance: torch.Tensor,
    reward: torch.Tensor,
    success: torch.Tensor,
    *,
    alpha: float = 0.1,
    cal_frac: float = 0.4,
    reward_quantile: float = 0.9,
) -> Dict[str, object]:
    """
    Same *analysis* as before (predict success when distance <= threshold), but:
    - Thresholds are mined only on the calibration prefix.
    - Metrics are reported on the test suffix (no threshold leakage).

    Returns dict with median_*, ccp_*, and meta fields.
    """
    distance = distance.float()
    reward = reward.float()
    success = success.float()
    T = distance.shape[0]
    pos = build_positive_mask(reward, success, reward_quantile)
    gt = success >= 0.99
    cal_mask, test_mask, T_cal = cal_test_masks(T, cal_frac)

    tau_med, n_med_cal = threshold_median_positive_cal(distance, pos, cal_mask)
    tau_cp, n_cp_cal, k_cp = threshold_class_conditional_split_cp(
        distance, pos, cal_mask, alpha
    )

    pred_med = distance <= tau_med
    pred_cp = distance <= tau_cp

    out_med = metrics_on_mask(pred_med, gt, test_mask)
    out_cp = metrics_on_mask(pred_cp, gt, test_mask)

    return {
        "median": {
            "threshold": tau_med,
            **{k: out_med[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_eval")},
            "n_cal_positive": float(n_med_cal),
        },
        "ccp_class_conditional": {
            "threshold": tau_cp,
            **{k: out_cp[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_eval")},
            "alpha": float(alpha),
            "n_cal_positive": float(n_cp_cal),
            "cp_quantile_index_k": float(k_cp),
        },
        "meta": {
            "T": int(T),
            "T_cal": int(T_cal),
            "cal_frac": float(cal_frac),
            "reward_quantile": float(reward_quantile),
        },
    }


# Backward-compatible single dict (median on all timesteps — legacy, can leak)
def dynamic_threshold_from_reward_legacy(
    distance: torch.Tensor,
    reward: torch.Tensor,
    success: torch.Tensor,
) -> Dict[str, float]:
    reward = reward.float()
    success = success.float()
    distance = distance.float()
    reward_q = torch.quantile(reward, 0.9)
    positive_mask = (success >= 0.99) | (reward >= reward_q)
    if positive_mask.sum() == 0:
        positive_mask = reward >= reward.mean()
    threshold = torch.median(distance[positive_mask]).item()
    pred = distance <= threshold
    gt = success >= 0.99
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
