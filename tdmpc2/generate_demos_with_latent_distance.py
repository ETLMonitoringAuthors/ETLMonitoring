import os
import glob as glob_module

import torch
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
from torchvision.utils import make_grid, save_image

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


@torch.no_grad()
def encode_obs_to_latent(agent, obs, task):
	"""
	Encode observations through the world model encoder to get latent states z.

	Args:
		agent: TDMPC2 agent
		obs: Tensor of shape [T, obs_dim] or [B, T, obs_dim]
		task: Tensor of task indices

	Returns:
		z: Tensor of shape [T, latent_dim] or [B, T, latent_dim]
	"""
	obs = obs.to(device='cuda', non_blocking=True)
	task = task.to(device='cuda', non_blocking=True)
	z = agent.model.encode(obs, task)
	return z.cpu()


def load_target_obs(target_obs_path):
	"""
	Load target observations (state vectors) from a .pt file.

	Args:
		target_obs_path: Path to .pt file containing goal state observations

	Returns:
		target_obs: Tensor of shape [N, obs_dim]
	"""
	assert os.path.exists(target_obs_path), f"Target obs file not found at {target_obs_path}"
	target_obs = torch.load(target_obs_path, weights_only=True)
	if isinstance(target_obs, dict):
		target_obs = target_obs['obs']
	if target_obs.dim() == 1:
		target_obs = target_obs.unsqueeze(0)
	print(f"Loaded {target_obs.shape[0]} target observations from {target_obs_path}")
	print(f"Target obs shape: {target_obs.shape}")
	return target_obs


def compute_set_latent_distances(traj_latents, target_latents, distance_metric='cosine'):
	"""
	Compute set-based distances between trajectory latent states and goal latent states.

	Args:
		traj_latents: Tensor of shape [T, latent_dim] — encoded trajectory observations
		target_latents: Tensor of shape [N, latent_dim] — encoded goal observations
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		min_distances: Tensor of shape [T] — min distance to any goal at each timestep
		mean_distances: Tensor of shape [T] — mean distance to all goals at each timestep
		chamfer_distance: Scalar — Chamfer distance between trajectory and goal set
	"""
	T = traj_latents.shape[0]
	N = target_latents.shape[0]

	# Compute pairwise distance matrix [T, N]
	if distance_metric == 'cosine':
		traj_norm = traj_latents / (traj_latents.norm(dim=-1, keepdim=True) + 1e-8)
		target_norm = target_latents / (target_latents.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim = traj_norm @ target_norm.T
		dist_matrix = 1 - cosine_sim
	elif distance_metric == 'l2':
		dist_matrix = torch.norm(traj_latents.unsqueeze(1) - target_latents.unsqueeze(0), dim=-1, p=2)
	elif distance_metric == 'l1':
		dist_matrix = torch.norm(traj_latents.unsqueeze(1) - target_latents.unsqueeze(0), dim=-1, p=1)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	# Per-timestep metrics
	min_distances = dist_matrix.min(dim=1).values  # [T]
	mean_distances = dist_matrix.mean(dim=1)  # [T]

	# Chamfer distance
	forward_chamfer = dist_matrix.min(dim=1).values.mean()
	backward_chamfer = dist_matrix.min(dim=0).values.mean()
	chamfer_distance = (forward_chamfer + backward_chamfer).item()

	return min_distances, mean_distances, chamfer_distance


def plot_latent_distances(min_dist_list, mean_dist_list, chamfer_list, task_names,
						  done_indices_list, save_path, distance_metric='cosine', num_goals=0,
						  avoid_min_dist_list=None, num_avoids=0):
	"""
	Plot latent space distances for multiple episodes.
	Without avoid: 2 columns (min dist to goal, mean dist to goal).
	With avoid: 3 columns (goal+avoid overlaid, mean dist to goal, negation score).
	"""
	num_demos = len(min_dist_list)
	has_avoid = avoid_min_dist_list is not None

	num_cols = 3 if has_avoid else 2
	fig, axes = plt.subplots(num_demos, num_cols, figsize=(7*num_cols, 4*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx, (min_dist, mean_dist, chamfer, task_name, done_idx) in enumerate(
		zip(min_dist_list, mean_dist_list, chamfer_list, task_names, done_indices_list)):

		timesteps = np.arange(len(min_dist))

		if has_avoid:
			avoid_min_dist = avoid_min_dist_list[idx]

			# Column 1: Goal and avoid min distances overlaid
			ax1 = axes[idx, 0]
			ax1.plot(timesteps, min_dist.numpy(), linewidth=2, color='steelblue',
					 label=f'Goal min dist ({num_goals})')
			ax1.plot(timesteps, avoid_min_dist.numpy(), linewidth=2, color='tomato', linestyle='--',
					 label=f'Avoid min dist ({num_avoids})')
			ax1.axvline(x=done_idx, color='black', linestyle=':', linewidth=1.5,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
			ax1.set_xlabel('Timestep', fontsize=12)
			ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
			ax1.set_title(f'Task: {task_name} | Goal vs Avoid Distance', fontsize=13, fontweight='bold')
			ax1.set_ylim(bottom=0)
			ax1.grid(True, alpha=0.3)
			ax1.legend(fontsize=10, loc='best')

			# Column 2: Mean distance to goal
			ax2 = axes[idx, 1]
			ax2.plot(timesteps, mean_dist.numpy(), linewidth=2, color='darkorange', label='Goal mean dist')
			ax2.axvline(x=done_idx, color='black', linestyle=':', linewidth=1.5,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
			ax2.set_xlabel('Timestep', fontsize=12)
			ax2.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
			ax2.set_title(f'Task: {task_name} | Mean Latent Distance to Goal ({num_goals})', fontsize=13, fontweight='bold')
			ax2.set_ylim(bottom=0)
			ax2.grid(True, alpha=0.3)
			ax2.legend(fontsize=10, loc='best')

			# Column 3: Negation score = avoid_dist - goal_dist (higher = better)
			ax3 = axes[idx, 2]
			negation_score = avoid_min_dist.numpy() - min_dist.numpy()
			ax3.plot(timesteps, negation_score, linewidth=2, color='purple', label='Negation score')
			ax3.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
			ax3.fill_between(timesteps, negation_score, 0,
							where=(negation_score > 0), alpha=0.3, color='green', label='Positive (good)')
			ax3.fill_between(timesteps, negation_score, 0,
							where=(negation_score <= 0), alpha=0.3, color='red', label='Negative (bad)')
			ax3.axvline(x=done_idx, color='black', linestyle=':', linewidth=1.5,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
			ax3.set_xlabel('Timestep', fontsize=12)
			ax3.set_ylabel('Avoid dist - Goal dist', fontsize=12)
			ax3.set_title(f'Task: {task_name} | Negation Score (higher=better)', fontsize=13, fontweight='bold')
			ax3.grid(True, alpha=0.3)
			ax3.legend(fontsize=10, loc='best')

		else:
			# Column 1: Min latent distance
			ax1 = axes[idx, 0]
			ax1.plot(timesteps, min_dist.numpy(), linewidth=2, color='steelblue', label='Min distance')

			min_of_min_idx = min_dist.argmin().item()
			ax1.axvline(x=min_of_min_idx, color='green', linestyle='--', linewidth=2,
					   label=f'Closest (t={min_of_min_idx})', alpha=0.7)
			ax1.scatter([min_of_min_idx], [min_dist[min_of_min_idx]], color='green', s=150, zorder=5, marker='o')

			ax1.axvline(x=done_idx, color='red', linestyle='--', linewidth=2,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
			ax1.scatter([done_idx], [min_dist[done_idx]], color='red', s=150, zorder=5, marker='o')

			ax1.set_xlabel('Timestep', fontsize=12)
			ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
			ax1.set_title(f'Task: {task_name} | Min Latent Distance to Goal Set ({num_goals} goals)', fontsize=14, fontweight='bold')
			ax1.set_ylim(bottom=0)
			ax1.grid(True, alpha=0.3)
			ax1.legend(fontsize=10, loc='best')

			stats_text = (f'Min: {min_dist.min().item():.4f}\n'
						  f'Mean: {min_dist.mean().item():.4f}\n'
						  f'Std: {min_dist.std().item():.4f}\n'
						  f'Chamfer: {chamfer:.4f}')
			ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
					verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

			# Column 2: Mean latent distance
			ax2 = axes[idx, 1]
			ax2.plot(timesteps, mean_dist.numpy(), linewidth=2, color='darkorange', label='Mean distance')

			min_mean_idx = mean_dist.argmin().item()
			ax2.axvline(x=min_mean_idx, color='green', linestyle='--', linewidth=2,
					   label=f'Closest (t={min_mean_idx})', alpha=0.7)
			ax2.scatter([min_mean_idx], [mean_dist[min_mean_idx]], color='green', s=150, zorder=5, marker='o')

			ax2.axvline(x=done_idx, color='red', linestyle='--', linewidth=2,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
			ax2.scatter([done_idx], [mean_dist[done_idx]], color='red', s=150, zorder=5, marker='o')

			ax2.set_xlabel('Timestep', fontsize=12)
			ax2.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
			ax2.set_title(f'Task: {task_name} | Mean Latent Distance to Goal Set ({num_goals} goals)', fontsize=14, fontweight='bold')
			ax2.set_ylim(bottom=0)
			ax2.grid(True, alpha=0.3)
			ax2.legend(fontsize=10, loc='best')

			stats_text_2 = (f'Min: {mean_dist.min().item():.4f}\n'
							f'Mean: {mean_dist.mean().item():.4f}\n'
							f'Std: {mean_dist.std().item():.4f}')
			ax2.text(0.02, 0.98, stats_text_2, transform=ax2.transAxes, fontsize=10,
					verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved latent distance plot to {save_path}')


def plot_latent_trajectories_2d(latents_list, task_names, target_latents, save_path, avoid_latents=None):
	"""
	Plot 2D trajectory of latent states using PCA projection with goal set.

	Args:
		latents_list: List of latent tensors, each of shape [T, latent_dim]
		task_names: List of task names
		target_latents: Goal set latents of shape [N, latent_dim]
		save_path: Path to save the plot
		avoid_latents: Optional avoid set latents of shape [M, latent_dim]
	"""
	from sklearn.decomposition import PCA

	num_demos = len(latents_list)

	all_for_pca = latents_list + [target_latents]
	if avoid_latents is not None:
		all_for_pca = all_for_pca + [avoid_latents]
	all_latents = torch.cat(all_for_pca, dim=0).numpy()

	pca = PCA(n_components=2)
	all_latents_2d = pca.fit_transform(all_latents)

	latents_2d_list = []
	start_idx = 0
	for latents in latents_list:
		end_idx = start_idx + len(latents)
		latents_2d_list.append(all_latents_2d[start_idx:end_idx])
		start_idx = end_idx
	target_latents_2d = all_latents_2d[start_idx:start_idx + len(target_latents)]
	start_idx += len(target_latents)
	avoid_latents_2d = all_latents_2d[start_idx:] if avoid_latents is not None else None

	fig, axes = plt.subplots(1, num_demos, figsize=(6*num_demos, 5))
	if num_demos == 1:
		axes = [axes]

	for idx, (latents_2d, task_name) in enumerate(zip(latents_2d_list, task_names)):
		ax = axes[idx]

		ax.plot(latents_2d[:, 0], latents_2d[:, 1], 'o-',
				linewidth=2, markersize=4, alpha=0.6, color='steelblue')

		ax.scatter(latents_2d[0, 0], latents_2d[0, 1],
				  color='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)

		ax.scatter(target_latents_2d[:, 0], target_latents_2d[:, 1],
				  color='green', s=200, marker='*', label=f'Goals ({len(target_latents_2d)})', zorder=5,
				  edgecolors='black', linewidths=1.5, alpha=0.8)

		if avoid_latents_2d is not None:
			ax.scatter(avoid_latents_2d[:, 0], avoid_latents_2d[:, 1],
					  color='red', s=200, marker='X', label=f'Avoid ({len(avoid_latents_2d)})', zorder=5,
					  edgecolors='black', linewidths=1.5, alpha=0.8)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		ax.set_title(f'Task: {task_name}\nLatent Trajectory (PCA)', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved latent trajectory plot to {save_path}')


def plot_latent_trajectories_3d_umap(latents_list, task_names, target_latents, save_path, avoid_latents=None):
	"""
	Plot 3D trajectory of latent states using UMAP projection with goal set.

	Args:
		latents_list: List of latent tensors, each of shape [T, latent_dim]
		task_names: List of task names
		target_latents: Goal set latents of shape [N, latent_dim]
		save_path: Path to save the plot
		avoid_latents: Optional avoid set latents of shape [M, latent_dim]
	"""
	try:
		import umap
	except ImportError:
		print("UMAP not installed. Skipping UMAP visualization. Install with: pip install umap-learn")
		return

	num_demos = len(latents_list)

	all_for_umap = latents_list + [target_latents]
	if avoid_latents is not None:
		all_for_umap = all_for_umap + [avoid_latents]
	all_latents = torch.cat(all_for_umap, dim=0).numpy()

	print("Computing UMAP projection to 3D...")
	reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
	all_latents_3d = reducer.fit_transform(all_latents)

	latents_3d_list = []
	start_idx = 0
	for latents in latents_list:
		end_idx = start_idx + len(latents)
		latents_3d_list.append(all_latents_3d[start_idx:end_idx])
		start_idx = end_idx
	target_latents_3d = all_latents_3d[start_idx:start_idx + len(target_latents)]
	start_idx += len(target_latents)
	avoid_latents_3d = all_latents_3d[start_idx:] if avoid_latents is not None else None

	fig = plt.figure(figsize=(8*num_demos, 7))

	for idx, (latents_3d, task_name) in enumerate(zip(latents_3d_list, task_names)):
		ax = fig.add_subplot(1, num_demos, idx+1, projection='3d')

		timesteps = np.arange(len(latents_3d))
		scatter = ax.scatter(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2],
							c=timesteps, cmap='viridis', s=30, alpha=0.6)

		ax.plot(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2],
			   linewidth=1.5, alpha=0.4, color='steelblue')

		ax.scatter(latents_3d[0, 0], latents_3d[0, 1], latents_3d[0, 2],
				  color='green', s=300, marker='o', label='Start', zorder=10,
				  edgecolors='black', linewidths=2)

		ax.scatter(target_latents_3d[:, 0], target_latents_3d[:, 1], target_latents_3d[:, 2],
				  color='green', s=400, marker='*', label=f'Goals ({len(target_latents_3d)})', zorder=10,
				  edgecolors='black', linewidths=1.5, alpha=0.8)

		if avoid_latents_3d is not None:
			ax.scatter(avoid_latents_3d[:, 0], avoid_latents_3d[:, 1], avoid_latents_3d[:, 2],
					  color='red', s=400, marker='X', label=f'Avoid ({len(avoid_latents_3d)})', zorder=10,
					  edgecolors='black', linewidths=1.5, alpha=0.8)

		ax.set_xlabel('UMAP 1', fontsize=11)
		ax.set_ylabel('UMAP 2', fontsize=11)
		ax.set_zlabel('UMAP 3', fontsize=11)
		ax.set_title(f'Task: {task_name}\nLatent Trajectory (UMAP 3D)', fontsize=13, fontweight='bold')
		ax.legend(fontsize=9, loc='upper left')

		cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
		cbar.set_label('Timestep', fontsize=10)

		ax.view_init(elev=20, azim=45)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved UMAP 3D latent trajectory plot to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""Generates demonstrations with latent space distance analysis using the world model encoder."""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Check for target obs path (optional — if not provided, uses demo success states as goals)
	has_target_obs = hasattr(cfg, 'target_obs_path')
	if has_target_obs:
		target_obs_path = cfg.target_obs_path
	else:
		print('No +target_obs_path provided. Will use successful demo states as goal observations.')

	# Check for avoid obs path (optional)
	has_avoid_obs = hasattr(cfg, 'avoid_obs_path')
	if has_avoid_obs:
		print(f'Avoid obs path provided: {cfg.avoid_obs_path}')

	cfg.enable_wandb = False
	cfg.env_mode = 'sync'
	cfg.checkpoint = f'{CHECKPOINT_PATH}/{cfg.task}.pt'
	cfg.num_envs = 2*cfg.num_demos
	cfg.model_size = 'B'
	cfg.save_video = True
	cfg.compile = False
	cfg = parse_cfg(cfg)
	seed = 2
	set_seed(seed)
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

	# Load target observations if provided
	if has_target_obs:
		target_obs = load_target_obs(target_obs_path)
		num_goals = target_obs.shape[0]
		print(f'Loaded {num_goals} goal observations from {target_obs_path}')
	else:
		target_obs = None
		num_goals = 0  # Will be set after collecting demos

	# Load avoid observations if provided
	if has_avoid_obs:
		avoid_obs = load_target_obs(cfg.avoid_obs_path)
		num_avoids = avoid_obs.shape[0]
		print(f'Loaded {num_avoids} avoid observations from {cfg.avoid_obs_path}')
	else:
		avoid_obs = None
		num_avoids = 0

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

	# Storage for latent analysis
	min_dist_list = []
	mean_dist_list = []
	chamfer_list = []
	latents_list = []
	accepted_task_names = []
	done_indices_list = []
	goal_obs_collected = []  # Collect goal state vectors from successful demos
	target_latents_stored = None
	avoid_min_dist_list = []  # Per-demo avoid distances (populated after collection)

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations with latent space distance analysis...')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = getattr(cfg, 'distance_metric', 'l2')
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
				elif accept:
					ep_td = tds[:, i].unsqueeze(0).clone()
					ep_frames = ep_td['frame'].squeeze(0)  # [T+1, C, H, W]
					frames.append(ep_frames)

					# Get observations for this episode: [T+1, obs_dim]
					ep_obs = ep_td['obs'].squeeze(0)  # [T+1, obs_dim]

					# Find the done index
					success_tensor = ep_td['success'].squeeze(0)  # [T+1]
					done_idx = torch.where(success_tensor >= 0.99)[0]
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')
					done_indices_list.append(done_idx)

					# Save the goal observation (state at done_idx) for later use
					goal_obs_collected.append(ep_obs[done_idx])

					# Encode trajectory observations through world model
					task_tensor = tasks[i:i+1].expand(ep_obs.shape[0])  # [T+1]
					traj_latents = encode_obs_to_latent(agent, ep_obs, task_tensor)  # [T+1, latent_dim]

					# If we have target obs, encode them and compute distances now
					if target_obs is not None:
						target_task = tasks[i:i+1].expand(target_obs.shape[0])
						tgt_latents = encode_obs_to_latent(agent, target_obs, target_task)

						print(f'Computing latent distances for demo {demos_collected+1} ({num_goals} goals)...')
						min_distances, mean_distances, chamfer_distance = compute_set_latent_distances(
							traj_latents, tgt_latents, distance_metric=distance_metric)

						min_dist_list.append(min_distances)
						mean_dist_list.append(mean_distances)
						chamfer_list.append(chamfer_distance)
						latents_list.append(traj_latents)

						if target_latents_stored is None:
							target_latents_stored = tgt_latents

						min_of_min_idx = min_distances.argmin().item()
						print(f'  Closest to any goal: t={min_of_min_idx} (min_dist={min_distances[min_of_min_idx]:.4f})')
						print(f'  Chamfer distance: {chamfer_distance:.4f}')
					else:
						# Store latents for later (will compute distances after all demos collected)
						latents_list.append(traj_latents)

					accepted_task_names.append(cfg.tasks[i])

					del ep_td['frame']
					demos_collected = buffer.add(ep_td)
					print(f'Added demo {demos_collected}/{cfg.num_demos} '
						  f'with reward {ep_reward[i]:.2f}, success {ep_success[i]:.2f}, and length {ep_len[i]} '
						  f'for task {cfg.tasks[i]}.')

					ep_reward[i] = 0.0
					ep_len[i] = 0

				else:
					print(f'Rejected demo for task {cfg.tasks[i]} '
						  f'with reward {ep_reward[i]:.2f} and success {ep_success[i]:.2f}.')

			break

		else:
			ep_len += 1

	if demos_collected < cfg.num_demos:
		print(f'[Demo collection failed] Only {demos_collected} demos collected, expected {cfg.num_demos}.')
		exit(0)

	# If no external target obs were provided, use leave-one-out goal observations
	if target_obs is None:
		all_goal_obs = torch.stack(goal_obs_collected, dim=0)  # [N, obs_dim]
		num_goals = len(goal_obs_collected) - 1  # Each demo uses all others as goals
		print(f'\nUsing leave-one-out: each demo compared against {num_goals} other goal observations...')

		# Encode all goal obs once
		all_goal_task = tasks[0:1].expand(all_goal_obs.shape[0])
		all_goal_latents = encode_obs_to_latent(agent, all_goal_obs, all_goal_task)  # [N, latent_dim]

		# For visualization, use all goal latents
		target_latents_stored = all_goal_latents

		# Leave-one-out: for demo i, use goals from all demos except i
		for idx, traj_latents in enumerate(latents_list):
			loo_mask = torch.ones(len(goal_obs_collected), dtype=torch.bool)
			loo_mask[idx] = False
			loo_latents = all_goal_latents[loo_mask]  # [N-1, latent_dim]

			print(f'Computing latent distances for demo {idx+1} (leave-one-out, {loo_latents.shape[0]} goals)...')
			min_distances, mean_distances, chamfer_distance = compute_set_latent_distances(
				traj_latents, loo_latents, distance_metric=distance_metric)

			min_dist_list.append(min_distances)
			mean_dist_list.append(mean_distances)
			chamfer_list.append(chamfer_distance)

			min_of_min_idx = min_distances.argmin().item()
			print(f'  Closest to any goal: t={min_of_min_idx} (min_dist={min_distances[min_of_min_idx]:.4f})')
			print(f'  Chamfer distance: {chamfer_distance:.4f}')

	# Compute avoid distances if avoid_obs provided
	avoid_latents_stored = None
	if avoid_obs is not None:
		print(f'\nEncoding {num_avoids} avoid observations through world model...')
		avoid_task = tasks[0:1].expand(avoid_obs.shape[0])
		avoid_latents_stored = encode_obs_to_latent(agent, avoid_obs, avoid_task)  # [M, latent_dim]

		for idx, traj_latents in enumerate(latents_list):
			print(f'Computing avoid distances for demo {idx+1} ({num_avoids} avoid states)...')
			avoid_min, _, _ = compute_set_latent_distances(
				traj_latents, avoid_latents_stored, distance_metric=distance_metric)
			avoid_min_dist_list.append(avoid_min)
			worst_idx = avoid_min.argmin().item()
			print(f'  Closest to any avoid: t={worst_idx} (min_dist={avoid_min[worst_idx]:.4f})')

	# Create data directory
	os.makedirs(cfg.data_dir, exist_ok=True)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames, dim=0)

	# Save goal observations for reuse
	goal_obs_to_save = target_obs if target_obs is not None else all_goal_obs
	goal_obs_save_path = f'{cfg.data_dir}/{cfg.task}_goal_obs.pt'
	torch.save({'obs': goal_obs_to_save, 'num_goals': goal_obs_to_save.shape[0]}, goal_obs_save_path)
	print(f'Saved goal observations to {goal_obs_save_path}')

	# Plot latent distances
	print('\nGenerating latent distance plots...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_latent_distances_{distance_metric}.png'
	plot_latent_distances(
		min_dist_list, mean_dist_list, chamfer_list, accepted_task_names,
		done_indices_list, plot_save_path, distance_metric=distance_metric, num_goals=num_goals,
		avoid_min_dist_list=avoid_min_dist_list if avoid_obs is not None else None,
		num_avoids=num_avoids)

	# Plot latent trajectories (PCA 2D)
	print('Generating latent trajectory plots (PCA 2D)...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_latent_trajectories_{distance_metric}.png'
	plot_latent_trajectories_2d(latents_list, accepted_task_names, target_latents_stored, trajectory_save_path,
								avoid_latents=avoid_latents_stored)

	# Plot latent trajectories (UMAP 3D)
	print('Generating latent trajectory plots (UMAP 3D)...')
	trajectory_save_path_umap = f'{cfg.data_dir}/{cfg.task}_latent_trajectories_umap3d_{distance_metric}.png'
	plot_latent_trajectories_3d_umap(latents_list, accepted_task_names, target_latents_stored, trajectory_save_path_umap,
									 avoid_latents=avoid_latents_stored)

	# Save grid image for each demo
	print('\nSaving grid visualizations for each demo...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]
		demo_frames_normalized = demo_frames.float() / 255.0
		grid = make_grid(demo_frames_normalized, nrow=int(np.ceil(np.sqrt(demo_frames.shape[0]))))
		save_image(grid, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_grid.png')
		print(f'Saved grid for demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

	print(f'\nCompleted! Saved {demos_collected} demos to {cfg.data_dir}.')
	print(f'Number of goal observations: {num_goals}')
	print(f'Latent dim: {latents_list[0].shape[-1]}')
	print(f'Chamfer distances: {chamfer_list}')


if __name__ == '__main__':
	generate_demos()
