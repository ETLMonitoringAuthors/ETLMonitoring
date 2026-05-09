import os
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
TDMPC2_DIR = ROOT / "tdmpc2"
# tdmpc2 agent lives in tdmpc2/tdmpc2.py; `import tdmpc2` must resolve to that module.
# If repo ROOT is *first* on sys.path, Python loads the `tdmpc2/` *package* and breaks
# `from tdmpc2 import TDMPC2`. So put the inner tdmpc2 code dir first, then ROOT for
# `etl_image_ablations.*` imports.
if str(TDMPC2_DIR) not in sys.path:
    sys.path.insert(0, str(TDMPC2_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common import set_seed  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
from common.vision_encoder import PretrainedEncoder  # noqa: E402
from config import Config, parse_cfg  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402

from etl_image_ablations.conformal_threshold import (  # noqa: E402
    dynamic_threshold_from_reward_legacy,
    dynamic_thresholds_cal_test,
)


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

CHECKPOINT_PATH = "./checkpoints/models--nicklashansen--newt/snapshots/7eef11eb63c8ed53d61d739693d7140135ea0876"


def resolve_results_out_dir(task: str, results_subdir: object) -> Path:
    """
    Default: etl_image_ablations/results/<task>/
    With +results_subdir=my_run: etl_image_ablations/results/<task>/my_run/
    (single path segment only — avoids clobbering prior runs / path traversal).
    """
    base = ROOT / "etl_image_ablations" / "results" / task
    if results_subdir is None:
        return base
    s = str(results_subdir).strip()
    if not s:
        return base
    if ".." in s or "/" in s or "\\" in s:
        raise ValueError(
            "results_subdir must be a single folder name (no /, \\, or ..). "
            f"Got: {results_subdir!r}"
        )
    return base / s


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def build_parsed_cfg_for_task(task: str, num_demos: int, seed: int):
    """Hydra-free cfg for a single task (same pattern as @hydra.main body)."""
    base = OmegaConf.structured(Config())
    # `num_demos` is not on the Config dataclass (Hydra adds it via +num_demos at runtime).
    # Allow merge to add extra keys like the main script does after parse_cfg(hydra_cfg).
    OmegaConf.set_struct(base, False)
    overrides = OmegaConf.create(
        {
            "task": task,
            "num_demos": num_demos,
            "enable_wandb": False,
            "env_mode": "sync",
            "checkpoint": f"{CHECKPOINT_PATH}/{task}.pt",
            "num_envs": 2 * num_demos,
            "model_size": "B",
            "save_video": True,
            "compile": False,
            "seed": seed,
        }
    )
    merged = OmegaConf.merge(base, overrides)
    return parse_cfg(merged)


def collect_cross_task_avoid_manifold(
    avoid_task: str,
    num_avoid_demos: int,
    seed: int,
    goal_pool_window: int,
    avoid_max_points: int,
    vision_encoder: PretrainedEncoder,
) -> Dict[str, torch.Tensor]:
    """
    Collect successful rollouts on `avoid_task`, encode latents around success,
    and return stacked tensors for use as a failure / negation reference on the main task.
    """
    assert os.path.exists(f"{CHECKPOINT_PATH}/{avoid_task}.pt"), (
        f"Missing checkpoint for avoid task {avoid_task}"
    )
    cfg_a = build_parsed_cfg_for_task(avoid_task, num_avoid_demos, seed)
    set_seed(seed)
    env_a = make_env(cfg_a)
    tasks_a = torch.arange(len(cfg_a.tasks), dtype=torch.int32)
    model_a = WorldModel(cfg_a).to(f"cuda:{cfg_a.rank}")
    agent_a = TDMPC2(model_a, cfg_a)
    agent_a.load(cfg_a.checkpoint)
    demos_a = collect_demos(cfg_a, agent_a, env_a, tasks_a)
    if len(demos_a) < num_avoid_demos:
        del agent_a, model_a, env_a
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Avoid task {avoid_task}: collected only {len(demos_a)} demos "
            f"(need {num_avoid_demos})"
        )
    lat_list: List[torch.Tensor] = []
    emb_list: List[torch.Tensor] = []
    state_list: List[torch.Tensor] = []
    for d in demos_a:
        emb = vision_encoder(d["frame"].to("cuda")).cpu()
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        lat = encode_latent(agent_a, d["obs"], task_tensor)
        done_idx = d["done_idx"].item()
        t0 = max(0, done_idx - goal_pool_window)
        t1 = min(lat.shape[0] - 1, done_idx + goal_pool_window)
        for t in range(t0, t1 + 1):
            lat_list.append(lat[t])
            emb_list.append(emb[t])
            state_list.append(d["obs"][t])
    del agent_a, model_a, env_a
    torch.cuda.empty_cache()

    avoid_lat = torch.stack(lat_list, dim=0)
    avoid_emb = torch.stack(emb_list, dim=0)
    avoid_state = torch.stack(state_list, dim=0)
    if avoid_max_points > 0 and avoid_lat.shape[0] > avoid_max_points:
        rng = np.random.default_rng(seed + 17)
        idx = rng.choice(avoid_lat.shape[0], size=avoid_max_points, replace=False)
        idx = np.sort(idx)
        avoid_lat = avoid_lat[idx]
        avoid_emb = avoid_emb[idx]
        avoid_state = avoid_state[idx]
    return {
        "avoid_lat": avoid_lat,
        "avoid_emb": avoid_emb,
        "avoid_state": avoid_state,
        "avoid_task": avoid_task,
        "n_points": int(avoid_lat.shape[0]),
    }


def to_td(cfg, env, obs, action=None, reward=None, value=None, terminated=None, frame=None, success=None):
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device="cpu")
    else:
        obs = obs.cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan")).repeat(cfg.num_envs)
    if value is None:
        value = torch.tensor(float("nan")).repeat(cfg.num_envs)
    if terminated is None:
        terminated = torch.tensor(False).repeat(cfg.num_envs)
    elif not isinstance(terminated, torch.Tensor):
        terminated = torch.stack(terminated.tolist())
    if success is None:
        success = torch.tensor(float("nan")).repeat(cfg.num_envs)
    assert frame is not None
    td = TensorDict(
        obs=obs,
        action=action,
        reward=reward,
        value=value,
        terminated=terminated,
        success=success,
        frame=frame,
        batch_size=(cfg.num_envs,),
    )
    return td


@torch.no_grad()
def estimate_value(agent, obs, action, task):
    obs = obs.to(device="cuda", non_blocking=True)
    action = action.to(device="cuda", non_blocking=True)
    task = task.to(device="cuda", non_blocking=True)
    z = agent.model.encode(obs, task)
    value = agent.model.Q(z, action, task, return_type="avg")
    return value.cpu().squeeze(-1)


@torch.no_grad()
def encode_latent(agent, obs, task):
    obs = obs.to(device="cuda", non_blocking=True)
    task = task.to(device="cuda", non_blocking=True)
    z = agent.model.encode(obs, task)
    return z.cpu()


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        x_n = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        y_n = y / (y.norm(dim=-1, keepdim=True) + 1e-8)
        return 1 - (x_n @ y_n.T)
    if metric == "l2":
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1, p=2)
    if metric == "l1":
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1, p=1)
    raise ValueError(f"Unsupported metric: {metric}")


def smoothness_metrics(signal: torch.Tensor) -> Dict[str, float]:
    x = signal.float().cpu()
    if len(x) < 3:
        return {"tv": float("nan"), "jerk": float("nan"), "std_diff": float("nan")}
    d1 = x[1:] - x[:-1]
    d2 = d1[1:] - d1[:-1]
    return {
        "tv": d1.abs().mean().item(),
        "jerk": d2.abs().mean().item(),
        "std_diff": d1.std().item(),
    }


def normalize_curve(curve: torch.Tensor) -> torch.Tensor:
    c = curve.float()
    if c.numel() == 0:
        return c
    c0 = c[0].abs() + 1e-8
    return c / c0


def choose_goal_indices_from_pool(pool_size: int, k: int, seed: int) -> List[int]:
    if pool_size <= 0:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool_size).tolist()
    return order[: min(k, pool_size)]


def plot_goal_vs_avoid_latent(
    lat_goal_dist: torch.Tensor,
    lat_avoid_dist: torch.Tensor,
    done_idx: int,
    out_path: Path,
    title: str,
    avoid_task_label: str,
):
    """Goal distance (main-task manifold) vs cross-task avoid distance; negation = avoid - goal."""
    neg = lat_avoid_dist.float() - lat_goal_dist.float()
    t = np.arange(len(lat_goal_dist))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0 = axes[0]
    ax0.plot(t, normalize_curve(lat_goal_dist).numpy(), label="Latent L2 → goal (main)", linewidth=2)
    ax0.plot(
        t,
        normalize_curve(lat_avoid_dist).numpy(),
        label=f"Latent L2 → avoid ({avoid_task_label})",
        linewidth=2,
    )
    ax0.axvline(done_idx, color="green", linestyle="--", alpha=0.7, label=f"success t={done_idx}")
    ax0.set_ylabel("Normalized (÷ d0)")
    ax0.set_title(title)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best", fontsize=8)
    ax1 = axes[1]
    ax1.plot(t, neg.numpy(), color="purple", linewidth=2, label="avoid_dist − goal_dist (↑ better)")
    ax1.axhline(0, color="k", linewidth=0.8, alpha=0.45)
    ax1.axvline(done_idx, color="green", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Negation score")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_triplet_curves(
    emb: torch.Tensor,
    lat: torch.Tensor,
    state: torch.Tensor,
    done_idx: int,
    out_path: Path,
    title: str,
):
    t = np.arange(len(emb))
    plt.figure(figsize=(10, 4))
    plt.plot(t, emb.numpy(), label="DINOv2 embedding distance", linewidth=2)
    plt.plot(t, lat.numpy(), label="World-model latent distance", linewidth=2)
    plt.plot(t, state.numpy(), label="Simulator state distance", linewidth=2)
    plt.axvline(done_idx, color="green", linestyle="--", alpha=0.7, label=f"success t={done_idx}")
    plt.xlabel("Timestep")
    plt.ylabel("Normalized distance (d / d0)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_demos(cfg, agent, env, tasks) -> List[Dict[str, torch.Tensor]]:
    obs, info = env.reset()
    frame = info["frame"]
    ep_reward = torch.zeros((cfg.num_envs,))
    ep_len = torch.ones((cfg.num_envs,), dtype=torch.int32)
    done = torch.full((cfg.num_envs,), True, dtype=torch.bool)
    tds = TensorDict({}, batch_size=(cfg.episode_length + 1, cfg.num_envs), device="cpu")
    tds[0] = to_td(cfg, env, obs, frame=frame)
    demos: List[Dict[str, torch.Tensor]] = []
    reward_threshold = -float("inf")

    while len(demos) < cfg.num_demos:
        action = agent(obs, t0=done, task=tasks, eval_mode=True)
        value = estimate_value(agent, obs, action, tasks)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated | truncated

        _obs = obs.clone()
        _frame = info["frame"].clone()
        _success = info["success"].clone()
        if "final_observation" in info:
            _obs[done] = info["final_observation"]
            _frame[done] = info["final_frame"]
        td = to_td(cfg, env, _obs, action, reward, value, terminated, _frame, _success)
        tds[ep_len] = td

        if done.any():
            assert done.all()
            median_reward = ep_reward.median()
            reward_threshold = max(reward_threshold, (0.75 if median_reward > 0 else 1.25) * median_reward)
            ep_success = info["final_info"]["success"]
            for i in range(cfg.num_envs):
                accept = (ep_reward[i] > reward_threshold) and (
                    (not cfg.task.startswith("mw") or ep_success[i] == 1.0)
                    and (not cfg.task.startswith("rd") or ep_success[i] == 1.0)
                    and (
                        not cfg.task.startswith("ms")
                        or ep_success[i] == 1.0
                        or cfg.task.startswith("ms-cartpole")
                        or cfg.task.startswith("ms-hopper")
                        or cfg.task.startswith("ms-ant")
                    )
                )
                if len(demos) >= cfg.num_demos:
                    break
                if not accept:
                    continue
                ep_td = tds[:, i].clone()
                success = ep_td["success"]
                done_idx = torch.where(success >= 0.99)[0]
                done_idx = done_idx[0].item() if len(done_idx) > 0 else len(success) - 1
                demos.append(
                    {
                        "obs": ep_td["obs"],
                        "frame": ep_td["frame"],
                        "reward": ep_td["reward"],
                        "success": ep_td["success"],
                        "done_idx": torch.tensor(done_idx),
                        "task_idx": tasks[i].clone(),
                    }
                )
            break
        else:
            ep_len += 1
    return demos


@hydra.main(version_base=None, config_name="config")
def main(cfg):
    assert torch.cuda.is_available(), "CUDA is required"
    assert hasattr(cfg, "num_demos") and cfg.num_demos > 1
    assert os.path.exists(CHECKPOINT_PATH)

    cfg.enable_wandb = False
    cfg.env_mode = "sync"
    cfg.checkpoint = f"{CHECKPOINT_PATH}/{cfg.task}.pt"
    cfg.num_envs = 2 * cfg.num_demos
    cfg.model_size = "B"
    cfg.save_video = True
    cfg.compile = False
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Optional: collect success-region latents from another task (e.g. green) as avoid set for main (e.g. blue).
    avoid_source_task = getattr(cfg, "avoid_source_task", None)
    avoid_pack = None
    vision_encoder = PretrainedEncoder()
    if avoid_source_task is not None and str(avoid_source_task).strip() != "":
        avoid_source_task = str(avoid_source_task)
        if avoid_source_task == str(cfg.task):
            raise ValueError(
                "avoid_source_task must differ from task (e.g. task=rd-push-blue, "
                "+avoid_source_task=rd-push-green)"
            )
        num_avoid_demos = int(getattr(cfg, "num_avoid_demos", cfg.num_demos))
        avoid_pool_window = int(
            getattr(cfg, "avoid_pool_window", getattr(cfg, "goal_pool_window", 20))
        )
        avoid_max_points = int(getattr(cfg, "avoid_max_points", 0))
        avoid_seed = int(getattr(cfg, "avoid_seed", cfg.seed))
        avoid_pack = collect_cross_task_avoid_manifold(
            avoid_source_task,
            num_avoid_demos,
            avoid_seed,
            avoid_pool_window,
            avoid_max_points,
            vision_encoder,
        )
        print(
            f"Cross-task avoid manifold: task={avoid_source_task}, "
            f"n_latents={avoid_pack['n_points']}"
        )

    env = make_env(cfg)
    tasks = torch.arange(len(cfg.tasks), dtype=torch.int32)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    agent.load(cfg.checkpoint)

    demos = collect_demos(cfg, agent, env, tasks)
    if len(demos) < cfg.num_demos:
        raise RuntimeError(f"Collected only {len(demos)} demos")

    goal_counts = list(getattr(cfg, "goal_counts", [1, 2, 3, 4]))
    goal_pool_window = int(getattr(cfg, "goal_pool_window", 20))
    out_dir = resolve_results_out_dir(str(cfg.task), getattr(cfg, "results_subdir", None))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-encode per-demo representations.
    for d in demos:
        d["emb"] = vision_encoder(d["frame"].to("cuda")).cpu()
        task_tensor = d["task_idx"].reshape(1).expand(d["obs"].shape[0])
        d["lat"] = encode_latent(agent, d["obs"], task_tensor)

    # Build large goal manifold pools from neighborhoods around successful timesteps.
    # This supports high-cardinality goal sets (e.g., 100 images).
    manifold_emb: List[List[torch.Tensor]] = []
    manifold_lat: List[List[torch.Tensor]] = []
    manifold_state: List[List[torch.Tensor]] = []
    for d in demos:
        done_idx = d["done_idx"].item()
        t0 = max(0, done_idx - goal_pool_window)
        t1 = min(d["emb"].shape[0] - 1, done_idx + goal_pool_window)
        manifold_emb.append([d["emb"][t] for t in range(t0, t1 + 1)])
        manifold_lat.append([d["lat"][t] for t in range(t0, t1 + 1)])
        manifold_state.append([d["obs"][t] for t in range(t0, t1 + 1)])

    rows = []
    threshold_rows = []
    cross_task_rows: List[Dict[str, Any]] = []
    cp_alpha = float(getattr(cfg, "cp_alpha", 0.1))
    cal_frac = float(getattr(cfg, "cal_frac", 0.4))
    reward_q_thr = float(getattr(cfg, "threshold_reward_quantile", 0.9))

    for k in goal_counts:
        for i, d in enumerate(demos):
            # Leave-one-demo-out manifold: exclude demo i to avoid trivial self-matching.
            pool_emb = [x for j, m in enumerate(manifold_emb) if j != i for x in m]
            pool_lat = [x for j, m in enumerate(manifold_lat) if j != i for x in m]
            pool_state = [x for j, m in enumerate(manifold_state) if j != i for x in m]
            pool_size = len(pool_emb)
            goal_idx = choose_goal_indices_from_pool(pool_size, k, seed=cfg.seed + 1000 * i + k)
            if len(goal_idx) == 0:
                continue
            g_emb = torch.stack([pool_emb[idx] for idx in goal_idx], dim=0)
            g_lat = torch.stack([pool_lat[idx] for idx in goal_idx], dim=0)
            g_state = torch.stack([pool_state[idx] for idx in goal_idx], dim=0)

            emb_dist = pairwise_distance(d["emb"], g_emb, "l2").min(dim=1).values
            lat_dist = pairwise_distance(d["lat"], g_lat, "l2").min(dim=1).values
            state_dist = pairwise_distance(d["obs"], g_state, "l2").min(dim=1).values

            if avoid_pack is not None:
                lat_avoid = pairwise_distance(
                    d["lat"], avoid_pack["avoid_lat"], "l2"
                ).min(dim=1).values
                emb_avoid = pairwise_distance(
                    d["emb"], avoid_pack["avoid_emb"], "l2"
                ).min(dim=1).values
                state_avoid = pairwise_distance(
                    d["obs"], avoid_pack["avoid_state"], "l2"
                ).min(dim=1).values
                neg_lat = lat_avoid - lat_dist
                di = d["done_idx"].item()
                cross_task_rows.append(
                    {
                        "demo": i,
                        "k_goals": k,
                        "avoid_source_task": avoid_pack["avoid_task"],
                        "latent_goal_tv": smoothness_metrics(lat_dist)["tv"],
                        "latent_avoid_tv": smoothness_metrics(lat_avoid)["tv"],
                        "latent_negation_tv": smoothness_metrics(neg_lat)["tv"],
                        "latent_goal_at_done": float(lat_dist[di].item()),
                        "latent_avoid_at_done": float(lat_avoid[di].item()),
                        "latent_negation_at_done": float(neg_lat[di].item()),
                        "embedding_avoid_at_done": float(emb_avoid[di].item()),
                        "state_avoid_at_done": float(state_avoid[di].item()),
                    }
                )
                # Per-timestep curves for offline checks: reusing goal-success τ on lat_avoid, etc.
                if getattr(cfg, "save_avoid_distance_trajectories", True):
                    np.savez(
                        out_dir / f"demo_{i:02d}_lat_goal_avoid_k{k}.npz",
                        lat_goal=lat_dist.detach().cpu().numpy().astype(np.float32),
                        lat_avoid=lat_avoid.detach().cpu().numpy().astype(np.float32),
                        neg_lat=neg_lat.detach().cpu().numpy().astype(np.float32),
                        reward=d["reward"].detach().cpu().numpy().astype(np.float32),
                        success=d["success"].detach().cpu().numpy().astype(np.float32),
                        done_idx=np.int32(d["done_idx"].item()),
                    )

            emb_s = smoothness_metrics(emb_dist)
            lat_s = smoothness_metrics(lat_dist)
            state_s = smoothness_metrics(state_dist)

            thr_emb = dynamic_thresholds_cal_test(
                emb_dist,
                d["reward"],
                d["success"],
                alpha=cp_alpha,
                cal_frac=cal_frac,
                reward_quantile=reward_q_thr,
            )
            thr_lat = dynamic_thresholds_cal_test(
                lat_dist,
                d["reward"],
                d["success"],
                alpha=cp_alpha,
                cal_frac=cal_frac,
                reward_quantile=reward_q_thr,
            )
            thr_state = dynamic_thresholds_cal_test(
                state_dist,
                d["reward"],
                d["success"],
                alpha=cp_alpha,
                cal_frac=cal_frac,
                reward_quantile=reward_q_thr,
            )
            # Legacy: median on all timesteps (old analysis; can inflate F1)
            leg_emb = dynamic_threshold_from_reward_legacy(emb_dist, d["reward"], d["success"])
            leg_lat = dynamic_threshold_from_reward_legacy(lat_dist, d["reward"], d["success"])
            leg_state = dynamic_threshold_from_reward_legacy(state_dist, d["reward"], d["success"])

            rows.extend(
                [
                    {"demo": i, "k_goals": k, "space": "embedding", **emb_s},
                    {"demo": i, "k_goals": k, "space": "latent", **lat_s},
                    {"demo": i, "k_goals": k, "space": "state", **state_s},
                ]
            )
            threshold_rows.extend(
                [
                    {
                        "demo": i,
                        "k_goals": k,
                        "space": "embedding",
                        "median_cal_test": thr_emb["median"],
                        "ccp_class_conditional": thr_emb["ccp_class_conditional"],
                        "meta_cal_test": thr_emb["meta"],
                        "legacy_median_all_timesteps": leg_emb,
                    },
                    {
                        "demo": i,
                        "k_goals": k,
                        "space": "latent",
                        "median_cal_test": thr_lat["median"],
                        "ccp_class_conditional": thr_lat["ccp_class_conditional"],
                        "meta_cal_test": thr_lat["meta"],
                        "legacy_median_all_timesteps": leg_lat,
                    },
                    {
                        "demo": i,
                        "k_goals": k,
                        "space": "state",
                        "median_cal_test": thr_state["median"],
                        "ccp_class_conditional": thr_state["ccp_class_conditional"],
                        "meta_cal_test": thr_state["meta"],
                        "legacy_median_all_timesteps": leg_state,
                    },
                ]
            )

            if k == goal_counts[-1]:
                plot_triplet_curves(
                    normalize_curve(emb_dist),
                    normalize_curve(lat_dist),
                    normalize_curve(state_dist),
                    d["done_idx"].item(),
                    out_dir / f"demo_{i:02d}_distance_triplet_k{k}.png",
                    f"{cfg.task} demo {i} | k={k} goal images",
                )
                if avoid_pack is not None:
                    plot_goal_vs_avoid_latent(
                        lat_dist,
                        lat_avoid,
                        d["done_idx"].item(),
                        out_dir / f"demo_{i:02d}_goal_vs_avoid_latent_k{k}.png",
                        f"{cfg.task} demo {i} | goal k={k} vs avoid={avoid_pack['avoid_task']}",
                        avoid_pack["avoid_task"],
                    )

    # Save JSON outputs for analysis.
    with open(out_dir / "smoothness_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "dynamic_threshold_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": (
                    "median_cal_test: median distance on calibration∩positive; "
                    "F1 on test suffix. ccp_class_conditional: split CP threshold on "
                    "calibration positives only (finite-sample k=ceil((n+1)(1-alpha))); "
                    "F1 on test. legacy_median_all_timesteps: old pooled median (leakage)."
                ),
                "cp_alpha": cp_alpha,
                "cal_frac": cal_frac,
                "threshold_reward_quantile": reward_q_thr,
                "rows": threshold_rows,
            },
            f,
            indent=2,
        )

    # Aggregate sensitivity trends by k and space.
    agg: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
    for r in rows:
        agg.setdefault((r["k_goals"], r["space"]), []).append(r)
    trend_rows = []
    for (k, space), vals in agg.items():
        trend_rows.append(
            {
                "k_goals": k,
                "space": space,
                "tv_mean": float(np.mean([v["tv"] for v in vals])),
                "jerk_mean": float(np.mean([v["jerk"] for v in vals])),
                "std_diff_mean": float(np.mean([v["std_diff"] for v in vals])),
            }
        )
    trend_rows = sorted(trend_rows, key=lambda x: (x["space"], x["k_goals"]))
    with open(out_dir / "sensitivity_trends.json", "w", encoding="utf-8") as f:
        json.dump(trend_rows, f, indent=2)

    if cross_task_rows:
        with open(out_dir / "cross_task_negation_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": (
                        "Cross-task avoid: success-region latents from avoid_source_task "
                        "(task-conditioned encode). Main rollout distances min L2 to that set; "
                        "negation = avoid_dist - goal_dist on latent L2."
                    ),
                    "main_task": str(cfg.task),
                    "avoid_source_task": avoid_pack["avoid_task"],
                    "n_avoid_latents": avoid_pack["n_points"],
                    "rows": cross_task_rows,
                },
                f,
                indent=2,
            )

    # Plot sensitivity (shake/noise proxy) using TV.
    plt.figure(figsize=(8, 4))
    for space in ["embedding", "latent", "state"]:
        points = [r for r in trend_rows if r["space"] == space]
        points = sorted(points, key=lambda x: x["k_goals"])
        plt.plot(
            [p["k_goals"] for p in points],
            [p["tv_mean"] for p in points],
            marker="o",
            linewidth=2,
            label=f"{space} TV",
        )
    plt.xlabel("Number of goal images (k)")
    plt.ylabel("Mean total variation (lower = smoother)")
    plt.title(f"{cfg.task} sensitivity to goal-image count")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_tv_vs_goal_count.png", dpi=150)
    plt.close()

    summary = {
        "task": cfg.task,
        "num_demos": len(demos),
        "goal_counts": goal_counts,
        "goal_pool_window": goal_pool_window,
        "cp_alpha": cp_alpha,
        "cal_frac": cal_frac,
        "output_dir": str(out_dir),
        "results_subdir": getattr(cfg, "results_subdir", None),
        "avoid_source_task": getattr(cfg, "avoid_source_task", None),
        "cross_task_negation_json": str(out_dir / "cross_task_negation_results.json")
        if cross_task_rows
        else None,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
