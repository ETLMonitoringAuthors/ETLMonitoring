import os
import glob as glob_module

import torch
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image

from common import set_seed
from common.buffer import Buffer
from common.world_model import WorldModel
from common.vision_encoder import PretrainedEncoder
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


def load_avoid_frames(avoid_dir):
	"""
	Load all frames to avoid from a directory.
	"""
	assert os.path.isdir(avoid_dir), f"Avoid frames directory not found at {avoid_dir}"

	extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
	image_paths = []
	for ext in extensions:
		image_paths.extend(sorted(glob_module.glob(os.path.join(avoid_dir, ext))))

	assert len(image_paths) > 0, f"No image files found in {avoid_dir}"

	frames = []
	for path in image_paths:
		frame = read_image(path)  # [C, H, W]
		frames.append(frame)
		print(f"  Loaded avoid frame: {os.path.basename(path)} (shape: {frame.shape})")

	avoid_frames = torch.stack(frames, dim=0)  # [N, C, H, W]
	print(f"Loaded {len(frames)} avoid frames from {avoid_dir}")
	return avoid_frames


def compute_pairwise_distances(embeddings, target_embeddings, distance_metric='mse'):
	"""
	Compute pairwise distance matrix between embeddings and target embeddings.

	Args:
		embeddings: [T, D]
		target_embeddings: [N, D]
		distance_metric: 'cosine', 'l2', 'l1', or 'mse'

	Returns:
		dist_matrix: [T, N]
	"""
	if distance_metric == 'cosine':
		emb_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		tgt_norm = target_embeddings / (target_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim = emb_norm @ tgt_norm.T
		dist_matrix = 1 - cosine_sim
	elif distance_metric == 'l2':
		dist_matrix = torch.norm(embeddings.unsqueeze(1) - target_embeddings.unsqueeze(0), dim=-1, p=2)
	elif distance_metric == 'l1':
		dist_matrix = torch.norm(embeddings.unsqueeze(1) - target_embeddings.unsqueeze(0), dim=-1, p=1)
	elif distance_metric == 'mse':
		dist_matrix = ((embeddings.unsqueeze(1) - target_embeddings.unsqueeze(0)) ** 2).mean(dim=-1)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")
	return dist_matrix


def compute_distances_to_set(traj_embeddings, target_embeddings, distance_metric='mse'):
	"""
	Compute per-timestep distances from trajectory embeddings to a set of target embeddings.

	Returns:
		min_distances: [T]
		mean_distances: [T]
		nearest_idx: [T] — which target is closest at each timestep
	"""
	dist_matrix = compute_pairwise_distances(traj_embeddings, target_embeddings, distance_metric)
	min_distances = dist_matrix.min(dim=1).values
	nearest_idx = dist_matrix.argmin(dim=1)
	mean_distances = dist_matrix.mean(dim=1)
	return min_distances, mean_distances, nearest_idx


def plot_negation_and_goal_distances(avoid_min_list, goal_min_list_loo, task_names,
									  done_indices_list, nearest_avoid_idx_list,
									  save_path, distance_metric='mse',
									  num_avoid=0, num_goals=0):
	"""
	Plot negation distance (to avoid set) and goal distance (to success set)
	on the SAME plot per demo.

	2 columns per demo:
	  - Col 1: Min distances (avoid + goal overlaid)
	  - Col 2: Mean distances (avoid + goal overlaid)
	"""
	num_demos = len(avoid_min_list)

	fig, axes = plt.subplots(num_demos, 2, figsize=(16, 5*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx in range(num_demos):
		avoid_min = avoid_min_list[idx]
		goal_min = goal_min_list_loo[idx]
		task_name = task_names[idx]
		done_idx = done_indices_list[idx]
		nearest_avoid_idx = nearest_avoid_idx_list[idx]

		timesteps = np.arange(len(avoid_min))

		# ---- Column 1: Min distances overlaid ----
		ax1 = axes[idx, 0]

		# Avoid distance (want HIGH — red when low)
		ax1.plot(timesteps, avoid_min.numpy(), linewidth=2, color='red', alpha=0.8,
				label=f'Min dist to AVOID ({num_avoid} frames)')

		# Goal distance (want LOW — blue when high)
		ax1.plot(timesteps, goal_min.numpy(), linewidth=2, color='blue', alpha=0.8,
				label=f'Min dist to GOAL ({num_goals} frames)')

		# Mark worst avoidance
		worst_avoid_t = avoid_min.argmin().item()
		worst_avoid_img = nearest_avoid_idx[worst_avoid_t].item()
		ax1.scatter([worst_avoid_t], [avoid_min[worst_avoid_t]], color='red', s=150, zorder=5,
				   marker='v', edgecolors='black', linewidths=1)
		ax1.annotate(f'worst avoid t={worst_avoid_t}\nimg #{worst_avoid_img}',
					 xy=(worst_avoid_t, avoid_min[worst_avoid_t].item()),
					 xytext=(15, -25), textcoords='offset points',
					 fontsize=9, color='red', fontweight='bold',
					 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

		# Mark best goal (closest to success)
		best_goal_t = goal_min.argmin().item()
		ax1.scatter([best_goal_t], [goal_min[best_goal_t]], color='blue', s=150, zorder=5,
				   marker='v', edgecolors='black', linewidths=1)
		ax1.annotate(f'best goal t={best_goal_t}',
					 xy=(best_goal_t, goal_min[best_goal_t].item()),
					 xytext=(15, 15), textcoords='offset points',
					 fontsize=9, color='blue', fontweight='bold',
					 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

		# Mark task done
		ax1.axvline(x=done_idx, color='green', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.8)

		ax1.set_xlabel('Timestep', fontsize=12)
		ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax1.set_title(f'Task: {task_name} | Min Distances\n'
					  f'Red=Avoid (want HIGH) | Blue=Goal (want LOW)',
					  fontsize=13, fontweight='bold')
		ax1.set_ylim(bottom=0)
		ax1.grid(True, alpha=0.3)
		ax1.legend(fontsize=9, loc='best')

		stats = (f'Avoid min: {avoid_min.min().item():.4f}\n'
				 f'Avoid at done: {avoid_min[done_idx].item():.4f}\n'
				 f'Goal min: {goal_min.min().item():.4f}\n'
				 f'Goal at done: {goal_min[done_idx].item():.4f}')
		ax1.text(0.02, 0.98, stats, transform=ax1.transAxes, fontsize=9,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# ---- Column 2: Negation score ----
		# Negation score: avoid_min - goal_min (higher = better: far from avoid, close to goal)
		ax2 = axes[idx, 1]
		negation_score = avoid_min - goal_min
		ax2.plot(timesteps, negation_score.numpy(), linewidth=2, color='purple', alpha=0.8,
				label='Negation score (avoid_dist - goal_dist)')
		ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
		ax2.fill_between(timesteps, 0, negation_score.numpy(),
						where=negation_score.numpy() > 0, alpha=0.15, color='green', label='Good (avoid > goal)')
		ax2.fill_between(timesteps, 0, negation_score.numpy(),
						where=negation_score.numpy() <= 0, alpha=0.15, color='red', label='Bad (avoid <= goal)')

		# Mark task done
		ax2.axvline(x=done_idx, color='green', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.8)

		best_score_t = negation_score.argmax().item()
		ax2.scatter([best_score_t], [negation_score[best_score_t]], color='green', s=150, zorder=5,
				   marker='^', edgecolors='black', linewidths=1)

		ax2.set_xlabel('Timestep', fontsize=12)
		ax2.set_ylabel('Negation Score', fontsize=12)
		ax2.set_title(f'Task: {task_name} | Negation Score\n'
					  f'(avoid_dist - goal_dist) | Higher = Better',
					  fontsize=13, fontweight='bold')
		ax2.grid(True, alpha=0.3)
		ax2.legend(fontsize=9, loc='best')

		score_stats = (f'Max score: {negation_score.max().item():.4f} (t={best_score_t})\n'
					   f'At done: {negation_score[done_idx].item():.4f}\n'
					   f'Mean: {negation_score.mean().item():.4f}')
		ax2.text(0.02, 0.98, score_stats, transform=ax2.transAxes, fontsize=9,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved negation+goal distance plot to {save_path}')


def plot_trajectories_2d(embeddings_list, task_names, avoid_embeddings, goal_embeddings_list, save_path):
	"""
	Plot 2D PCA trajectory with both avoid frames (red X) and goal frames (green *).
	"""
	from sklearn.decomposition import PCA

	num_demos = len(embeddings_list)

	# Collect all goal embeddings into one tensor for PCA
	all_goal_embeddings = torch.cat(goal_embeddings_list, dim=0) if goal_embeddings_list else torch.zeros(0, 768)

	all_embeddings = torch.cat(
		embeddings_list + [avoid_embeddings, all_goal_embeddings], dim=0).numpy()

	pca = PCA(n_components=2)
	all_2d = pca.fit_transform(all_embeddings)

	# Split back
	emb_2d_list = []
	start = 0
	for emb in embeddings_list:
		end = start + len(emb)
		emb_2d_list.append(all_2d[start:end])
		start = end
	avoid_2d = all_2d[start:start + len(avoid_embeddings)]
	start += len(avoid_embeddings)
	goal_2d = all_2d[start:]

	fig, axes = plt.subplots(1, num_demos, figsize=(6*num_demos, 5))
	if num_demos == 1:
		axes = [axes]

	for idx, (emb_2d, task_name) in enumerate(zip(emb_2d_list, task_names)):
		ax = axes[idx]

		ax.plot(emb_2d[:, 0], emb_2d[:, 1], 'o-',
				linewidth=2, markersize=4, alpha=0.6, color='steelblue')

		ax.scatter(emb_2d[0, 0], emb_2d[0, 1],
				  color='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
		ax.scatter(emb_2d[-1, 0], emb_2d[-1, 1],
				  color='blue', s=200, marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)

		ax.scatter(avoid_2d[:, 0], avoid_2d[:, 1],
				  color='red', s=250, marker='X', label=f'AVOID ({len(avoid_2d)})', zorder=5,
				  edgecolors='black', linewidths=1.5, alpha=0.8)

		if len(goal_2d) > 0:
			ax.scatter(goal_2d[:, 0], goal_2d[:, 1],
					  color='limegreen', s=250, marker='*', label=f'GOAL ({len(goal_2d)})', zorder=5,
					  edgecolors='black', linewidths=1.5, alpha=0.8)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		ax.set_title(f'Task: {task_name}\nTrajectory (PCA) — Avoid Red, Reach Green', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved trajectory plot to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""
	Generates demonstrations and measures BOTH:
	  - Negation distance to avoid frames (loaded from directory, e.g., push_blue)
	  - Goal distance to success frames from other demos (leave-one-out)

	Both are plotted on the same graph.

	Usage:
		python generate_demos_with_negation_and_goal.py task=rd-push_green +num_demos=4 \
			+avoid_frames_dir=<path_to_push_blue_frames> +distance_metric=mse
	"""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	assert hasattr(cfg, 'avoid_frames_dir'), \
		'Please specify path to avoid frames directory via +avoid_frames_dir=<path_to_directory>'
	avoid_frames_dir = cfg.avoid_frames_dir

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

	env = make_env(cfg)
	tasks = torch.arange(len(cfg.tasks), dtype=torch.int32)

	model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
	agent = TDMPC2(model, cfg)

	assert cfg.obs == 'state', \
		'Checkpoint loading only works with state observations.'
	if os.path.exists(cfg.get('checkpoint', None)):
		agent.load(cfg.checkpoint)
	else:
		raise ValueError(f'Checkpoint {cfg.checkpoint} does not exist.')

	# Initialize vision encoder
	print('Initializing vision encoder for embedding analysis...')
	vision_encoder = PretrainedEncoder()

	# Load avoid frames
	print(f'Loading avoid frames from {avoid_frames_dir}...')
	avoid_frames = load_avoid_frames(avoid_frames_dir)  # [N, C, H, W]
	num_avoid = avoid_frames.shape[0]

	# Encode avoid frames once
	avoid_embeddings = vision_encoder(avoid_frames.to('cuda')).cpu()  # [N, 768]
	print(f'Avoid embeddings shape: {avoid_embeddings.shape}')

	# Prepare environment and metrics
	obs, info = env.reset()
	frame = info['frame']
	ep_reward = torch.zeros((cfg.num_envs,))
	ep_len = torch.ones((cfg.num_envs,), dtype=torch.int32)
	done = torch.full((cfg.num_envs,), True, dtype=torch.bool)
	tds = TensorDict({}, batch_size=(cfg.episode_length+1, cfg.num_envs), device='cpu')
	tds[0] = to_td(cfg, env, obs, frame=frame)
	frames_list = []

	# Prepare buffer
	cfg.buffer_size = (cfg.episode_length + 1) * cfg.num_demos
	buffer = Buffer(
		capacity=cfg.buffer_size,
		batch_size=cfg.batch_size,
		horizon=cfg.horizon,
		multiproc=False,
	)

	# Storage — we collect everything first, then compute goal distances after (leave-one-out)
	embeddings_list = []  # per-demo trajectory embeddings [T, 768]
	done_indices_list = []
	success_frame_indices = []  # (demo_idx, timestep) of success frame
	accepted_task_names = []

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations...')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = getattr(cfg, 'distance_metric', 'mse')
	while demos_collected < cfg.num_demos:

		action = agent(obs, t0=done, task=tasks, eval_mode=True)
		value = estimate_value(agent, obs, action, tasks)
		obs, reward, terminated, truncated, info = env.step(action)
		assert not terminated.any(), \
			'Unexpected termination signal received.'
		ep_reward += reward
		done = terminated | truncated

		_obs = obs.clone()
		_frame = info['frame'].clone()
		_success = info['success'].clone()

		if 'final_observation' in info:
			_obs[done] = info['final_observation']
			_frame[done] = info['final_frame']
		td = to_td(cfg, env, _obs, action, reward, value, terminated, _frame, _success)
		tds[ep_len] = td

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
					frames_list.append(ep_frames)

					# Find done index
					success_tensor = ep_td['success'].squeeze(0)
					done_idx = torch.where(success_tensor >= 0.99)[0]
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')
					done_indices_list.append(done_idx)
					success_frame_indices.append(done_idx)

					# Encode trajectory frames
					print(f'Encoding trajectory for demo {demos_collected+1}...')
					traj_embeddings = vision_encoder(ep_frames.to('cuda')).cpu()  # [T+1, 768]
					embeddings_list.append(traj_embeddings)
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

	# ---- Post-collection: compute distances ----
	print(f'\n--- Computing distances for {demos_collected} demos ---')

	# Collect success frame embeddings from each demo
	success_embeddings = []
	for demo_idx in range(demos_collected):
		done_t = success_frame_indices[demo_idx]
		success_embeddings.append(embeddings_list[demo_idx][done_t])  # [768]
	success_embeddings = torch.stack(success_embeddings, dim=0)  # [num_demos, 768]
	print(f'Collected {success_embeddings.shape[0]} success frame embeddings for leave-one-out goal set')

	# Per-demo: compute avoid distances + leave-one-out goal distances
	avoid_min_list = []
	avoid_mean_list = []
	nearest_avoid_idx_list = []
	goal_min_list_loo = []
	goal_mean_list_loo = []
	goal_embeddings_loo_list = []  # for trajectory plots

	for demo_idx in range(demos_collected):
		traj_emb = embeddings_list[demo_idx]  # [T, 768]

		# Avoid distances
		avoid_min, avoid_mean, nearest_avoid = compute_distances_to_set(
			traj_emb, avoid_embeddings, distance_metric)
		avoid_min_list.append(avoid_min)
		avoid_mean_list.append(avoid_mean)
		nearest_avoid_idx_list.append(nearest_avoid)

		# Leave-one-out goal distances
		loo_mask = torch.ones(demos_collected, dtype=torch.bool)
		loo_mask[demo_idx] = False
		goal_emb_loo = success_embeddings[loo_mask]  # [num_demos-1, 768]
		goal_embeddings_loo_list.append(goal_emb_loo)

		goal_min, goal_mean, _ = compute_distances_to_set(
			traj_emb, goal_emb_loo, distance_metric)
		goal_min_list_loo.append(goal_min)
		goal_mean_list_loo.append(goal_mean)

		done_idx = done_indices_list[demo_idx]
		worst_avoid_t = avoid_min.argmin().item()
		best_goal_t = goal_min.argmin().item()
		print(f'Demo {demo_idx+1}:')
		print(f'  Avoid — worst t={worst_avoid_t} (dist={avoid_min[worst_avoid_t]:.4f}), at done={avoid_min[done_idx]:.4f}')
		print(f'  Goal  — best t={best_goal_t} (dist={goal_min[best_goal_t]:.4f}), at done={goal_min[done_idx]:.4f}')
		print(f'  Negation score at done: {(avoid_min[done_idx] - goal_min[done_idx]).item():.4f}')

	num_goals = demos_collected - 1

	# ---- Save outputs ----
	os.makedirs(cfg.data_dir, exist_ok=True)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames_list, dim=0)

	# Plot combined negation + goal distances
	print('\nGenerating combined negation+goal distance plots...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_negation_and_goal_{distance_metric}.png'
	plot_negation_and_goal_distances(
		avoid_min_list, goal_min_list_loo, accepted_task_names,
		done_indices_list, nearest_avoid_idx_list,
		plot_save_path, distance_metric=distance_metric,
		num_avoid=num_avoid, num_goals=num_goals)

	# Plot trajectories (PCA 2D) with both avoid and goal markers
	print('Generating trajectory plots (PCA 2D)...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_negation_goal_trajectories_{distance_metric}.png'
	plot_trajectories_2d(embeddings_list, accepted_task_names,
						 avoid_embeddings, goal_embeddings_loo_list,
						 trajectory_save_path)

	# Save grid image for each demo
	print('\nSaving grid visualizations for each demo...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]
		demo_frames_normalized = demo_frames.float() / 255.0
		nrow = int(np.ceil(np.sqrt(demo_frames.shape[0])))
		grid = make_grid(demo_frames_normalized, nrow=nrow)
		save_image(grid, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_grid.png')
		print(f'Saved grid for demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

	# Save side-by-side: worst avoidance frame vs its nearest avoid frame
	print('\nSaving worst-avoidance side-by-side comparisons...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]
		avoid_min = avoid_min_list[demo_idx]
		nearest_avoid_idx = nearest_avoid_idx_list[demo_idx]

		worst_t = avoid_min.argmin().item()
		worst_avoid_img = nearest_avoid_idx[worst_t].item()

		traj_frame = demo_frames[worst_t].float() / 255.0
		avoid_frame = avoid_frames[worst_avoid_img].float() / 255.0

		pair = torch.stack([traj_frame, avoid_frame], dim=0)
		pair_grid = make_grid(pair, nrow=2, padding=4, pad_value=1.0)
		pair_path = f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_worst_avoidance_t{worst_t}_avoid{worst_avoid_img}.png'
		save_image(pair_grid, pair_path)
		print(f'  Demo {demo_idx}: worst at t={worst_t} vs avoid img #{worst_avoid_img} (dist={avoid_min[worst_t]:.4f})')

	print(f'\nCompleted! Saved {demos_collected} demos to {cfg.data_dir}.')
	print(f'Avoid frames: {avoid_frames_dir} ({num_avoid} frames)')
	print(f'Goal frames: leave-one-out from {demos_collected} success frames ({num_goals} per demo)')
	print(f'Distance metric: {distance_metric}')


if __name__ == '__main__':
	generate_demos()
