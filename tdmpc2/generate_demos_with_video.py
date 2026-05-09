import os

import torch
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
import imageio

from common import set_seed
from common.buffer import Buffer
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

CHECKPOINT_PATH = "./checkpoints/models--nicklashansen--newt/snapshots/7eef11eb63c8ed53d61d739693d7140135ea0876"


def to_td(cfg, env, obs, action=None, reward=None, value=None, terminated=None, frame=None, success=None):
	"""Creates a TensorDict for a new episode."""
	if isinstance(obs, dict):
		obs = TensorDict(obs, batch_size=(), device='cpu')
	else:
		obs = obs.cpu()
	if action is None:
		action = torch.full_like(env.rand_act(), float('nan'))
	if reward is None:
		reward = torch.tensor(float('nan')).repeat(cfg.num_envs)
	if value is None:
		value = torch.tensor(float('nan')).repeat(cfg.num_envs)
	if terminated is None:
		terminated = torch.tensor(False).repeat(cfg.num_envs)
	elif not isinstance(terminated, torch.Tensor):
		terminated = torch.stack(terminated.tolist())
	if success is None:
		success = torch.tensor(float('nan')).repeat(cfg.num_envs)
	assert frame is not None, \
		'Missing frame in to_td but it is needed in demo generation.'
	td = TensorDict(
		obs=obs,
		action=action,
		reward=reward,
		value=value,
		terminated=terminated,
		success=success,
		frame=frame,
		batch_size=(cfg.num_envs,))
	return td


@torch.no_grad()
def estimate_value(agent, obs, action, task):
	"""Estimates the value of the current observation."""
	obs = obs.to(device='cuda', non_blocking=True)
	action = action.to(device='cuda', non_blocking=True)
	task = task.to(device='cuda', non_blocking=True)
	z = agent.model.encode(obs, task)
	value = agent.model.Q(z, action, task, return_type='avg')
	return value.cpu().squeeze(-1)


def save_video(frames, save_path, fps=30):
	"""
	Save frames as a video file.

	Args:
		frames: Tensor of shape [T, C, H, W] in uint8 format
		save_path: Path to save the video
		fps: Frames per second for the video
	"""
	# Convert from [T, C, H, W] to [T, H, W, C] and to numpy
	frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()

	# Ensure frames are in uint8 format
	if frames_np.dtype != np.uint8:
		frames_np = frames_np.astype(np.uint8)

	# Save as video using imageio
	imageio.mimsave(save_path, frames_np, fps=fps)
	print(f'Saved video to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""Generates demonstrations and saves them as videos."""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Configuration for video saving
	video_fps = getattr(cfg, 'video_fps', 30)

	cfg.enable_wandb = False
	cfg.env_mode = 'sync'
	cfg.checkpoint = f'{CHECKPOINT_PATH}/{cfg.task}.pt'
	cfg.num_envs = 2*cfg.num_demos  # Some episodes may be rejected
	cfg.model_size = 'B'
	cfg.save_video = True
	cfg.compile = False
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	assert len(cfg.tasks) == cfg.num_envs, \
		'Number of tasks must match number of environments for finetuning.'

	# Define environment
	env = make_env(cfg)
	tasks = torch.arange(len(cfg.tasks), dtype=torch.int32)

	# Define agent
	model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
	agent = TDMPC2(model, cfg)

	# Load checkpoint
	assert cfg.obs == 'state', \
		'Checkpoint loading only works with state observations.'

	if os.path.exists(cfg.get('checkpoint', None)):
		agent.load(cfg.checkpoint)
	else:
		raise ValueError(f'Checkpoint {cfg.checkpoint} does not exist.')

	# Prepare environment and metrics
	obs, info = env.reset()
	frame = info['frame']
	ep_reward = torch.zeros((cfg.num_envs,))
	ep_len = torch.ones((cfg.num_envs,), dtype=torch.int32)
	done = torch.full((cfg.num_envs,), True, dtype=torch.bool)
	tds = TensorDict({}, batch_size=(cfg.episode_length+1, cfg.num_envs), device='cpu')
	tds[0] = to_td(cfg, env, obs, frame=frame)
	frames = []

	# Prepare buffer
	cfg.buffer_size = (cfg.episode_length + 1) * cfg.num_demos
	buffer = Buffer(
		capacity=cfg.buffer_size,
		batch_size=cfg.batch_size,
		horizon=cfg.horizon,
		multiproc=False,
	)

	# Storage for video metadata
	accepted_task_names = []
	done_indices_list = []

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations as videos...')
	demos_collected = 0
	reward_threshold = -float('inf')

	while demos_collected < cfg.num_demos:

		# Collect experience
		action = agent(obs, t0=done, task=tasks, eval_mode=True)
		value = estimate_value(agent, obs, action, tasks)
		obs, reward, terminated, truncated, info = env.step(action)
		assert not terminated.any(), \
			'Unexpected termination signal received.'
		ep_reward += reward
		done = terminated | truncated

		# Store experience
		_obs = obs.clone()
		_frame = info['frame'].clone()
		_success = info['success'].clone()

		if 'final_observation' in info:
			_obs[done] = info['final_observation']
			_frame[done] = info['final_frame']
		td = to_td(cfg, env, _obs, action, reward, value, terminated, _frame, _success)
		tds[ep_len] = td

		# Add to buffer if done and above threshold
		if done.any():
			assert done.all(), \
				'All environments must be done before adding to buffer.'
			median_reward = ep_reward.median()
			reward_threshold = max(reward_threshold, (0.75 if median_reward > 0 else 1.25) * median_reward)
			ep_success = info['final_info']['success']
			print(f'\nMean reward: {ep_reward.mean():.2f}, ')
			print(f'Median reward: {ep_reward.median():.2f}, ')
			print(f'Mean success: {ep_success.mean():.2f}, ')
			print(f'Reward threshold: {reward_threshold:.2f}')

			for i in range(cfg.num_envs):
				accept = (ep_reward[i] > reward_threshold) and \
						 (not cfg.task.startswith('mw') or ep_success[i] == 1.) and \
						 (not cfg.task.startswith('rd') or ep_success[i] == 1.) and \
						 (not cfg.task.startswith('ms') or ep_success[i] == 1. or \
							(cfg.task.startswith('ms-cartpole') or cfg.task.startswith('ms-hopper') \
							 or cfg.task.startswith('ms-ant')))
				if demos_collected >= cfg.num_demos:
					break
				elif accept:  # Accept demo
					# Add to buffer
					ep_td = tds[:, i].unsqueeze(0).clone()
					ep_frames = ep_td['frame'].squeeze(0)  # [T+1, C, H, W]
					frames.append(ep_frames)

					# Find the done index (when task was actually completed successfully)
					success_tensor = ep_td['success'].squeeze(0)  # [T+1]
					done_idx = torch.where(success_tensor >= 0.99)[0]
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')

					accepted_task_names.append(cfg.tasks[i])
					done_indices_list.append(done_idx)

					del ep_td['frame']
					demos_collected = buffer.add(ep_td)
					print(f'Added demo {demos_collected}/{cfg.num_demos} '
						  f'with reward {ep_reward[i]:.2f}, success {ep_success[i]:.2f}, and length {ep_len[i]} '
						  f'for task {cfg.tasks[i]}.')

					# Reset episode metrics
					ep_reward[i] = 0.0
					ep_len[i] = 0

				else:  # Reject demo
					print(f'Rejected demo for task {cfg.tasks[i]} '
						  f'with reward {ep_reward[i]:.2f} and success {ep_success[i]:.2f}.')

			break  # Exit regardless of number of demos collected

		else:
			ep_len += 1

	# Raise an error if not enough demos were collected
	if demos_collected < cfg.num_demos:
		print(f'[Demo collection failed] Only {demos_collected} demos collected, expected {cfg.num_demos}.')
		exit(0)

	# Create data directory if it doesn't exist
	os.makedirs(cfg.data_dir, exist_ok=True)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames, dim=0)

	# Save videos
	print('\nSaving demo videos...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]  # [T+1, C, H, W]
		done_idx = done_indices_list[demo_idx]
		task_name = accepted_task_names[demo_idx]

		# Save full episode as video
		video_path = f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}.mp4'
		save_video(demo_frames, video_path, fps=video_fps)

		print(f'Saved demo {demo_idx+1}/{demos_collected}: {task_name} '
			  f'({demo_frames.shape[0]} frames, success at t={done_idx})')

	print(f'\nSaved {demos_collected} demo videos to {cfg.data_dir}.')


if __name__ == '__main__':
	generate_demos()
