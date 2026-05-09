"""
Minimal script to re-render hero figure frames at high resolution (480×480).
Skips embedding analysis — only saves PNG frames.
Same seed (2) as original → same rollouts, higher-quality renders.

Usage (from repo root):
  MUJOCO_GL=egl conda run -n newt python tdmpc2/generate_hires_hero_frames.py \
    task=mw-pick-place-wall +num_demos=5 ++data_dir=/tmp/hires_mw \
    render_size=480 seed=2 env_mode=sync model_size=B \
    enable_wandb=false compile=false
"""
import os
from pathlib import Path

import torch
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
from torchvision.utils import save_image

from common import set_seed
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

CHECKPOINT_PATH = "./checkpoints/models--nicklashansen--newt/snapshots/7eef11eb63c8ed53d61d739693d7140135ea0876"

# Which frames to save from each demo (same as hero figure)
GOAL_FRAMES = {0: [100], 1: [90], 2: [95], 3: [85]}
TEST_DEMO   = 4   # save all frames for this demo

OUT_DIR = Path(__file__).resolve().parent.parent / "paper_files/figures/hero_assets"


def to_td(cfg, env, obs, action=None, reward=None, value=None,
          terminated=None, frame=None, success=None):
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device="cpu")
    else:
        obs = obs.cpu()
    if action    is None: action    = torch.full_like(env.rand_act(), float("nan"))
    if reward    is None: reward    = torch.tensor(float("nan")).repeat(cfg.num_envs)
    if value     is None: value     = torch.tensor(float("nan")).repeat(cfg.num_envs)
    if terminated is None: terminated = torch.tensor(False).repeat(cfg.num_envs)
    elif not isinstance(terminated, torch.Tensor):
        terminated = torch.stack(terminated.tolist())
    if success   is None: success   = torch.tensor(float("nan")).repeat(cfg.num_envs)
    assert frame is not None
    return TensorDict(obs=obs, action=action, reward=reward, value=value,
                      terminated=terminated, success=success, frame=frame,
                      batch_size=(cfg.num_envs,))


@torch.no_grad()
def estimate_value(agent, obs, action, task):
    obs    = obs.to("cuda", non_blocking=True)
    action = action.to("cuda", non_blocking=True)
    task   = task.to("cuda", non_blocking=True)
    z      = agent.model.encode(obs, task)
    return agent.model.Q(z, action, task, return_type="avg").cpu().squeeze(-1)


@hydra.main(version_base=None, config_name="config")
def generate(cfg):
    assert torch.cuda.is_available()
    assert hasattr(cfg, "num_demos") and cfg.num_demos > 0
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"

    cfg.enable_wandb = False
    cfg.env_mode     = "sync"
    cfg.checkpoint   = f"{CHECKPOINT_PATH}/{cfg.task}.pt"
    cfg.num_envs     = 2 * cfg.num_demos
    cfg.model_size   = "B"
    cfg.save_video   = True
    cfg.compile      = False
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env   = make_env(cfg)
    tasks = torch.arange(len(cfg.tasks), dtype=torch.int32)

    model = WorldModel(cfg).to("cuda:0")
    agent = TDMPC2(model, cfg)
    agent.load(cfg.checkpoint)

    obs, info = env.reset()
    frame = info["frame"]
    ep_reward = torch.zeros(cfg.num_envs)
    ep_len    = torch.ones(cfg.num_envs, dtype=torch.int32)
    done      = torch.full((cfg.num_envs,), True, dtype=torch.bool)

    tds = TensorDict({}, batch_size=(cfg.episode_length + 1, cfg.num_envs), device="cpu")
    tds[0] = to_td(cfg, env, obs, frame=frame)

    accepted_frames: list[torch.Tensor] = []   # list of [T+1, C, H, W] per accepted demo
    demos_collected  = 0
    reward_threshold = -float("inf")

    print(f"Collecting {cfg.num_demos} demos at {cfg.render_size}×{cfg.render_size}…")

    while demos_collected < cfg.num_demos:
        action = agent(obs, t0=done, task=tasks, eval_mode=True)
        value  = estimate_value(agent, obs, action, tasks)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated | truncated

        _obs     = obs.clone()
        _frame   = info["frame"].clone()
        _success = info["success"].clone()
        if "final_observation" in info:
            _obs[done]   = info["final_observation"]
            _frame[done] = info["final_frame"]

        tds[ep_len] = to_td(cfg, env, _obs, action, reward, value, terminated, _frame, _success)

        if done.any():
            assert done.all()
            ep_success       = info["final_info"]["success"]
            median_rw        = ep_reward.median()
            reward_threshold = max(reward_threshold,
                                   (0.75 if median_rw > 0 else 1.25) * median_rw)

            for i in range(cfg.num_envs):
                if demos_collected >= cfg.num_demos:
                    break
                accept = (ep_reward[i] > reward_threshold and
                          (not cfg.task.startswith("mw") or ep_success[i] == 1.))
                if accept:
                    ep_td  = tds[:, i].unsqueeze(0).clone()
                    frames = ep_td["frame"].squeeze(0)   # [T+1, C, H, W]
                    accepted_frames.append(frames)
                    demos_collected += 1
                    print(f"  Demo {demos_collected}/{cfg.num_demos} accepted "
                          f"(env {i}, reward={ep_reward[i]:.1f}, len={ep_len[i]})")

            ep_reward.zero_()
            ep_len.fill_(1)
            obs, info = env.reset()
            frame = info["frame"]
            tds[0] = to_td(cfg, env, obs, frame=frame)
            done = torch.full((cfg.num_envs,), True, dtype=torch.bool)
        else:
            ep_len += 1

    if demos_collected < cfg.num_demos:
        print(f"Only collected {demos_collected}/{cfg.num_demos} — exiting.")
        return

    def save_png(tensor_chw: torch.Tensor, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # save_image normalises float → use raw uint8 via PIL instead
        arr = tensor_chw.permute(1, 2, 0).numpy().astype(np.uint8)
        from PIL import Image
        Image.fromarray(arr).save(str(path), format="PNG", compress_level=0)

    print(f"\nSaving frames to {OUT_DIR} …")

    # Goal set frames
    for demo_idx, frame_indices in GOAL_FRAMES.items():
        ep = accepted_frames[demo_idx]
        for fi in frame_indices:
            fi = min(fi, ep.shape[0] - 1)
            fname = f"demo{demo_idx}_frame{fi:03d}.png"
            save_png(ep[fi], OUT_DIR / "goal_set" / fname)
            print(f"  goal_set/{fname}  {ep[fi].shape[-1]}×{ep[fi].shape[-2]}")

    # Full test trajectory
    ep_test = accepted_frames[TEST_DEMO]
    for t in range(ep_test.shape[0]):
        save_png(ep_test[t], OUT_DIR / "test_trajectory" / f"frame_{t:03d}.png")
    print(f"  test_trajectory/  {ep_test.shape[0]} frames "
          f"({ep_test.shape[-1]}×{ep_test.shape[-2]})")

    print("Done.")


if __name__ == "__main__":
    generate()
