"""
Empirical local Lipschitz estimates for the world-model encoder η and the
spec score δ(z) = ||z - z_spec||_2 on RoboDesk-style *state* observations.

We report ratios
  L_η   = ||η(s') - η(s)||_2 / ||s' - s||_2
  L_δ   = |δ(η(s')) - δ(η(s))| / ||s' - s||_2
for (i) random small perturbations s' = s + ε v with ||v||_2 = 1, and
(ii) consecutive policy states (along-rollout finite differences).

Near-threshold stratification uses τ_F1 from the same F1 sweep as
eval_newt_spec_predicates.py on calibration distances to z_spec.

Headless MuJoCo: set ``MUJOCO_GL=egl`` (or ``osmesa``) if frame rendering fails.

For the AnySafe / Dubins semantic encoder (Dreamer RSSM + spec image), use the
sibling repo script ``AnySafeReachability/dino_wm/etl/empirical_lipschitz_dubins.py``.

Usage (from repo root, CUDA required):
  MUJOCO_GL=egl python -m etl_image_ablations.empirical_lipschitz_eta \\
      --task rd-push-green --num-demos 10 --num-envs 10 --cal-frac 0.5 \\
      --goal-window 10 --eps-obs 0.02 0.05 0.1 \\
      --n-random-pairs 2000 --boundary-gamma 0.05 \\
      --out-json etl_results/lipschitz_rd/lipschitz.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
if str(TDMPC2_DIR) not in sys.path:
    sys.path.insert(0, str(TDMPC2_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import hydra.utils as _hu

if not hasattr(_hu, "_orig_get_original_cwd"):
    _hu._orig_get_original_cwd = _hu.get_original_cwd
    _hu.get_original_cwd = lambda: str(Path.cwd())

from etl_image_ablations.eval_newt_spec_predicates import (  # noqa: E402
    build_spec_latent,
    load_agent_and_env,
    _sweep_f1,
)
from etl_image_ablations.run_image_etl_ablations import collect_demos, encode_latent  # noqa: E402


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def _summarize(name: str, lz: np.ndarray, ld: np.ndarray) -> Dict[str, Any]:
    return {
        name: {
            "L_eta": _quantiles(lz),
            "L_delta": _quantiles(ld),
            "n": int(len(lz)),
        }
    }


@torch.no_grad()
def _random_direction_pairs(
    obs_flat: torch.Tensor,
    task_idx: torch.Tensor,
    z_spec: torch.Tensor,
    agent,
    eps: float,
    n_pairs: int,
    rng: np.random.Generator,
    batch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    obs_flat: [N, D], task_idx: [N] — rows are reference states.
    Returns L_eta, L_delta, and base_dist ||η(s)-z_spec|| before perturbation.
    """
    n, d = obs_flat.shape
    idx = rng.integers(0, n, size=n_pairs)
    s = obs_flat[idx].clone()
    t = task_idx[idx].long()

    v = torch.randn(n_pairs, d, device=s.device, dtype=s.dtype)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    sp = s + eps * v

    lz_list, ld_list, d0_list = [], [], []
    device = next(agent.model.parameters()).device
    z_spec_d = z_spec.to(device=device, dtype=torch.float32)

    for start in range(0, n_pairs, batch):
        end = min(start + batch, n_pairs)
        sb = s[start:end].to(device)
        spb = sp[start:end].to(device)
        tb = t[start:end].long()

        z0 = agent.model.encode(sb, tb)
        z1 = agent.model.encode(spb, tb)
        ds = (spb - sb).norm(dim=-1).clamp_min(1e-12)
        dz = (z1 - z0).norm(dim=-1)
        d0 = (z0 - z_spec_d.unsqueeze(0)).norm(dim=-1)
        d1 = (z1 - z_spec_d.unsqueeze(0)).norm(dim=-1)
        dd = (d1 - d0).abs()

        lz_list.append((dz / ds).cpu().numpy())
        ld_list.append((dd / ds).cpu().numpy())
        d0_list.append(d0.cpu().numpy())

    return (
        np.concatenate(lz_list),
        np.concatenate(ld_list),
        np.concatenate(d0_list),
    )


@torch.no_grad()
def _consecutive_pairs(
    demos: List[Dict[str, torch.Tensor]],
    z_spec: torch.Tensor,
    agent,
    batch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(agent.model.parameters()).device
    z_spec_d = z_spec.to(device=device, dtype=torch.float32)
    lz_list, ld_list, d0_list = [], [], []

    for d in demos:
        obs = d["obs"]
        if not isinstance(obs, torch.Tensor) or obs.ndim != 2:
            continue
        task_tensor = d["task_idx"].reshape(1).expand(obs.shape[0]).long()
        t_max = obs.shape[0] - 1
        if t_max < 1:
            continue
        for start in range(0, t_max, batch):
            end = min(start + batch, t_max)
            s0 = obs[start:end].to(device)
            s1 = obs[start + 1 : end + 1].to(device)
            tk = task_tensor[start:end].to(device)
            ds = (s1 - s0).norm(dim=-1)
            mask = ds > 1e-8
            if not mask.any():
                continue
            z0 = agent.model.encode(s0, tk)
            z1 = agent.model.encode(s1, tk)
            dz = (z1 - z0).norm(dim=-1)
            d0 = (z0 - z_spec_d.unsqueeze(0)).norm(dim=-1)
            d1 = (z1 - z_spec_d.unsqueeze(0)).norm(dim=-1)
            dd = (d1 - d0).abs()
            ds_safe = ds.clamp_min(1e-12)
            lz_list.append((dz / ds_safe)[mask].cpu().numpy())
            ld_list.append((dd / ds_safe)[mask].cpu().numpy())
            d0_list.append(d0[mask].cpu().numpy())

    if not lz_list:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(lz_list), np.concatenate(ld_list), np.concatenate(d0_list)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Pretrained Newt checkpoints use num_envs=10; override if load fails otherwise.
    agent, env, tasks_t, cfg = load_agent_and_env(
        args.task, args.num_demos, args.seed, num_envs=args.num_envs
    )
    if cfg.num_envs < args.num_demos:
        raise ValueError(
            f"num_envs={cfg.num_envs} < num_demos={args.num_demos}: "
            "collect_demos only harvests one parallel batch, so need num_envs >= num_demos."
        )
    if cfg.obs != "state":
        raise ValueError(
            f"This script only supports cfg.obs=='state' (got {cfg.obs!r}). "
            "Random Lipschitz probes are defined in raw observation space."
        )

    rng = np.random.default_rng(args.seed)
    print("[lipschitz] Collecting demos …")
    demos = collect_demos(cfg, agent, env, tasks_t)
    if len(demos) < args.num_demos:
        raise RuntimeError(f"Only {len(demos)} demos (need {args.num_demos})")

    for d in demos:
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent, d["obs"], task_tensor)

    n_cal = max(1, int(args.cal_frac * len(demos)))
    cal_demos = demos[:n_cal]
    test_demos = demos[n_cal:]
    z_spec = build_spec_latent(cal_demos, agent, window=args.goal_window)

    d_cal_list, gt_cal_list = [], []
    for d in cal_demos:
        dist = torch.norm(d["lat"].float() - z_spec.float().unsqueeze(0), dim=-1).numpy()
        gt = (d["success"].numpy() >= 0.99).astype(np.float32)
        d_cal_list.append(dist)
        gt_cal_list.append(gt)
    d_cal = np.concatenate(d_cal_list)
    gt_cal = np.concatenate(gt_cal_list)
    taus, f1s, _, _ = _sweep_f1(d_cal, gt_cal, n_taus=300)
    tau_f1 = float(taus[int(np.argmax(f1s))])
    print(f"[lipschitz] τ_F1 (cal) = {tau_f1:.6f}")

    obs_rows: List[torch.Tensor] = []
    task_rows: List[torch.Tensor] = []
    for d in demos:
        o = d["obs"]
        if not isinstance(o, torch.Tensor) or o.ndim != 2:
            raise TypeError(f"Expected demo obs Tensor [T,D], got {type(o)} shape {getattr(o, 'shape', None)}")
        obs_rows.append(o.float().cpu())
        task_rows.append(d["task_idx"].reshape(1).expand(o.shape[0]).float().cpu())
    obs_flat = torch.cat(obs_rows, dim=0)
    task_flat = torch.cat(task_rows, dim=0).squeeze(-1).long()

    out: Dict[str, Any] = {
        "task": args.task,
        "tau_f1_cal": tau_f1,
        "boundary_gamma": args.boundary_gamma,
        "n_demos": len(demos),
        "obs_dim": int(obs_flat.shape[1]),
        "random_probes": {},
        "consecutive": {},
    }

    # --- random probes per eps ---
    for eps in args.eps_obs:
        lz, ld, d0 = _random_direction_pairs(
            obs_flat,
            task_flat,
            z_spec,
            agent,
            eps=float(eps),
            n_pairs=args.n_random_pairs,
            rng=rng,
            batch=args.batch,
        )
        near = np.abs(d0 - tau_f1) <= args.boundary_gamma
        bulk = ~near
        key = f"eps_{eps}"
        out["random_probes"][key] = {
            "eps_obs": float(eps),
            "all": _summarize("all", lz, ld)["all"],
            "near_threshold": _summarize("near_threshold", lz[near], ld[near])["near_threshold"],
            "bulk": _summarize("bulk", lz[bulk], ld[bulk])["bulk"],
            "frac_near_threshold": float(near.mean()),
        }
        print(
            f"[lipschitz] random ε={eps}: L_eta p99={out['random_probes'][key]['all']['L_eta']['p99']:.4f}  "
            f"L_delta p99={out['random_probes'][key]['all']['L_delta']['p99']:.4f}  "
            f"near-B_γ frac={near.mean():.3f}"
        )

    # --- consecutive (all demos) ---
    lz_c, ld_c, d0_c = _consecutive_pairs(demos, z_spec, agent, batch=args.batch)
    if lz_c.size > 0:
        near_c = np.abs(d0_c - tau_f1) <= args.boundary_gamma
        out["consecutive"] = {
            "all": _summarize("all", lz_c, ld_c)["all"],
            "near_threshold": _summarize("near_threshold", lz_c[near_c], ld_c[near_c])["near_threshold"],
            "bulk": _summarize("bulk", lz_c[~near_c], ld_c[~near_c])["bulk"],
            "frac_near_threshold": float(near_c.mean()),
        }
        print(
            f"[lipschitz] consecutive: L_eta p99={out['consecutive']['all']['L_eta']['p99']:.4f}  "
            f"L_delta p99={out['consecutive']['all']['L_delta']['p99']:.4f}"
        )
    else:
        out["consecutive"] = {"error": "no consecutive pairs"}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[lipschitz] Wrote {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="rd-push-green")
    p.add_argument("--num-demos", type=int, default=20)
    p.add_argument("--cal-frac", type=float, default=0.5)
    p.add_argument("--goal-window", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps-obs", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--n-random-pairs", type=int, default=2000)
    p.add_argument("--boundary-gamma", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="Parallel env slots (must match checkpoint; default 10 for released Newt weights).",
    )
    p.add_argument("--out-json", type=str, default="etl_results/lipschitz/lipschitz.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
