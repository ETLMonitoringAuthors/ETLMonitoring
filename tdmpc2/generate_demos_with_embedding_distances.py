import os

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


def compute_embedding_distances(frames, encoder, target_idx=None, distance_metric='cosine'):
	"""
	Compute distances between target frame embedding and all frame embeddings.

	Args:
		frames: Tensor of shape [T, C, H, W] containing episode frames
		encoder: PretrainedEncoder instance
		target_idx: Index of the target frame (goal frame). If None, uses last frame.
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		distances: Tensor of shape [T] containing distances
		embeddings: Tensor of shape [T, 768] containing all embeddings
		pixel_mse: Tensor of shape [T] containing pixel MSE to goal frame
	"""
	# Encode all frames
	# frames should be [T, C, H, W] in uint8
	embeddings = encoder(frames)  # [T, 768]

	# Get target frame embedding (goal/completion frame)
	if target_idx is None:
		target_idx = -1
	target_embedding = embeddings[target_idx:target_idx+1].clone()  # [1, 768]

	# Compute embedding distances
	if distance_metric == 'cosine':
		# Cosine distance = 1 - cosine_similarity
		# Normalize embeddings
		target_norm = target_embedding / (target_embedding.norm(dim=-1, keepdim=True) + 1e-8)
		embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim = (embeddings_norm * target_norm).sum(dim=-1)
		distances = 1 - cosine_sim
	elif distance_metric == 'l2':
		# Euclidean distance
		distances = torch.norm(embeddings - target_embedding, dim=-1, p=2)
	elif distance_metric == 'l1':
		# Manhattan distance
		distances = torch.norm(embeddings - target_embedding, dim=-1, p=1)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	# Compute pixel MSE between each frame and goal frame
	target_frame = frames[target_idx:target_idx+1].float()  # [1, C, H, W]
	frames_float = frames.float()  # [T, C, H, W]
	# MSE over all pixels (C, H, W dimensions)
	pixel_mse = ((frames_float - target_frame) ** 2).mean(dim=(1, 2, 3))  # [T]

	return distances.cpu(), embeddings.cpu(), pixel_mse.cpu()


def plot_embedding_distances(distances_list, task_names, done_indices_list, pixel_mse_list, save_path, distance_metric='cosine'):
	"""
	Plot embedding distances for multiple episodes.

	Args:
		distances_list: List of distance tensors, each of shape [T]
		task_names: List of task names corresponding to each episode
		done_indices_list: List of done indices (when task was completed)
		pixel_mse_list: List of pixel MSE tensors, each of shape [T]
		save_path: Path to save the plot
		distance_metric: Type of distance metric used
	"""
	num_demos = len(distances_list)

	# Create figure with subplots - 2 rows per demo (embedding distance and pixel MSE)
	fig, axes = plt.subplots(num_demos, 2, figsize=(16, 4*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx, (distances, pixel_mse, task_name, done_idx) in enumerate(zip(distances_list, pixel_mse_list, task_names, done_indices_list)):
		# Plot embedding distance
		ax1 = axes[idx, 0]
		timesteps = np.arange(len(distances))

		ax1.plot(timesteps, distances.numpy(), linewidth=2, color='steelblue', label='Embedding distance')

		# Mark the done index (when task was actually completed)
		ax1.axvline(x=done_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)
		ax1.scatter([done_idx], [distances[done_idx]], color='green', s=150, zorder=5, marker='o')

		# Mark the final frame (end of episode)
		ax1.axvline(x=len(distances)-1, color='red', linestyle='--', linewidth=2,
				   label=f'Episode End (t={len(distances)-1})', alpha=0.7)
		ax1.scatter([len(distances)-1], [distances[-1]], color='red', s=100, zorder=5)

		ax1.set_xlabel('Timestep', fontsize=12)
		ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax1.set_title(f'Task: {task_name} | Embedding Distance to Goal Frame (t={done_idx})', fontsize=14, fontweight='bold')
		ax1.grid(True, alpha=0.3)
		ax1.legend(fontsize=10, loc='best')

		# Add statistics text box
		distances_up_to_done = distances[:done_idx+1]
		mean_dist = distances_up_to_done.mean().item()
		std_dist = distances_up_to_done.std().item()
		min_dist = distances_up_to_done.min().item()
		max_dist = distances_up_to_done.max().item()

		if done_idx > 0:
			start_dist = distances[0].item()
			end_dist = distances[done_idx].item()
			is_decreasing = end_dist < start_dist
			trend = "↓ Decreasing" if is_decreasing else "↑ Increasing"
			change_pct = ((end_dist - start_dist) / (start_dist + 1e-8)) * 100
		else:
			trend = "N/A"
			change_pct = 0.0

		stats_text = f'Mean: {mean_dist:.4f}\nStd: {std_dist:.4f}\nMin: {min_dist:.4f}\nMax: {max_dist:.4f}\nTrend: {trend}\nChange: {change_pct:+.1f}%'
		ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# Plot pixel MSE
		ax2 = axes[idx, 1]
		ax2.plot(timesteps, pixel_mse.numpy(), linewidth=2, color='orange', label='Pixel MSE')

		# Mark the done index
		ax2.axvline(x=done_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)
		ax2.scatter([done_idx], [pixel_mse[done_idx]], color='green', s=150, zorder=5, marker='o')

		# Mark the final frame
		ax2.axvline(x=len(pixel_mse)-1, color='red', linestyle='--', linewidth=2,
				   label=f'Episode End (t={len(pixel_mse)-1})', alpha=0.7)
		ax2.scatter([len(pixel_mse)-1], [pixel_mse[-1]], color='red', s=100, zorder=5)

		ax2.set_xlabel('Timestep', fontsize=12)
		ax2.set_ylabel('Pixel MSE', fontsize=12)
		ax2.set_title(f'Task: {task_name} | Pixel MSE to Goal Frame (t={done_idx})', fontsize=14, fontweight='bold')
		ax2.grid(True, alpha=0.3)
		ax2.legend(fontsize=10, loc='best')

		# Add statistics for pixel MSE
		mse_up_to_done = pixel_mse[:done_idx+1]
		mean_mse = mse_up_to_done.mean().item()
		std_mse = mse_up_to_done.std().item()
		min_mse = mse_up_to_done.min().item()
		max_mse = mse_up_to_done.max().item()

		if done_idx > 0:
			start_mse = pixel_mse[0].item()
			end_mse = pixel_mse[done_idx].item()
			is_decreasing_mse = end_mse < start_mse
			trend_mse = "↓ Decreasing" if is_decreasing_mse else "↑ Increasing"
			change_pct_mse = ((end_mse - start_mse) / (start_mse + 1e-8)) * 100
		else:
			trend_mse = "N/A"
			change_pct_mse = 0.0

		stats_text_mse = f'Mean: {mean_mse:.2f}\nStd: {std_mse:.2f}\nMin: {min_mse:.2f}\nMax: {max_mse:.2f}\nTrend: {trend_mse}\nChange: {change_pct_mse:+.1f}%'
		ax2.text(0.02, 0.98, stats_text_mse, transform=ax2.transAxes, fontsize=10,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved embedding distance plot to {save_path}')


def plot_embedding_trajectories_2d(embeddings_list, task_names, done_indices_list, save_path):
	"""
	Plot 2D trajectory of embeddings using PCA projection.

	Args:
		embeddings_list: List of embedding tensors, each of shape [T, 768]
		task_names: List of task names
		done_indices_list: List of done indices (when task was completed)
		save_path: Path to save the plot
	"""
	from sklearn.decomposition import PCA

	num_demos = len(embeddings_list)

	# Concatenate all embeddings for PCA
	all_embeddings = torch.cat(embeddings_list, dim=0).numpy()

	# Apply PCA to reduce to 2D
	pca = PCA(n_components=2)
	all_embeddings_2d = pca.fit_transform(all_embeddings)

	# Split back into individual episodes
	embeddings_2d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_2d_list.append(all_embeddings_2d[start_idx:end_idx])
		start_idx = end_idx

	# Plot
	fig, axes = plt.subplots(1, num_demos, figsize=(6*num_demos, 5))
	if num_demos == 1:
		axes = [axes]

	for idx, (embeddings_2d, task_name, done_idx) in enumerate(zip(embeddings_2d_list, task_names, done_indices_list)):
		ax = axes[idx]

		# Plot trajectory
		ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'o-',
				linewidth=2, markersize=4, alpha=0.6, color='steelblue')

		# Highlight start and done index (goal/completion)
		ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
				  color='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
		ax.scatter(embeddings_2d[done_idx, 0], embeddings_2d[done_idx, 1],
				  color='red', s=200, marker='*', label=f'Completion (t={done_idx})', zorder=5, edgecolors='black', linewidths=2)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		ax.set_title(f'Task: {task_name}\nEmbedding Trajectory (PCA)', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved embedding trajectory plot to {save_path}')


def plot_embedding_trajectories_3d_umap(embeddings_list, task_names, done_indices_list, save_path):
	"""
	Plot 3D trajectory of embeddings using UMAP projection.
	UMAP preserves local neighborhoods and reveals nonlinear manifold structure.

	Args:
		embeddings_list: List of embedding tensors, each of shape [T, 768]
		task_names: List of task names
		done_indices_list: List of done indices (when task was completed)
		save_path: Path to save the plot
	"""
	try:
		import umap
	except ImportError:
		print("UMAP not installed. Skipping UMAP visualization. Install with: pip install umap-learn")
		return

	num_demos = len(embeddings_list)

	# Concatenate all embeddings for UMAP
	all_embeddings = torch.cat(embeddings_list, dim=0).numpy()

	# Apply UMAP to reduce to 3D
	print("Computing UMAP projection to 3D...")
	reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
	all_embeddings_3d = reducer.fit_transform(all_embeddings)

	# Split back into individual episodes
	embeddings_3d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_3d_list.append(all_embeddings_3d[start_idx:end_idx])
		start_idx = end_idx

	# Plot in 3D
	fig = plt.figure(figsize=(8*num_demos, 7))

	for idx, (embeddings_3d, task_name, done_idx) in enumerate(zip(embeddings_3d_list, task_names, done_indices_list)):
		ax = fig.add_subplot(1, num_demos, idx+1, projection='3d')

		# Plot trajectory with color gradient (time)
		timesteps = np.arange(len(embeddings_3d))
		scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
							c=timesteps, cmap='viridis', s=30, alpha=0.6)

		# Plot line connecting trajectory
		ax.plot(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
			   linewidth=1.5, alpha=0.4, color='steelblue')

		# Highlight start (green sphere)
		ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2],
				  color='green', s=300, marker='o', label='Start', zorder=10,
				  edgecolors='black', linewidths=2)

		# Highlight done index (red star)
		ax.scatter(embeddings_3d[done_idx, 0], embeddings_3d[done_idx, 1], embeddings_3d[done_idx, 2],
				  color='red', s=400, marker='*', label=f'Completion (t={done_idx})', zorder=10,
				  edgecolors='black', linewidths=2)

		ax.set_xlabel('UMAP 1', fontsize=11)
		ax.set_ylabel('UMAP 2', fontsize=11)
		ax.set_zlabel('UMAP 3', fontsize=11)
		ax.set_title(f'Task: {task_name}\nEmbedding Trajectory (UMAP 3D)', fontsize=13, fontweight='bold')
		ax.legend(fontsize=9, loc='upper left')

		# Add colorbar for timesteps
		cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
		cbar.set_label('Timestep', fontsize=10)

		# Set viewing angle
		ax.view_init(elev=20, azim=45)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved UMAP 3D embedding trajectory plot to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""Generates demonstrations with embedding distance analysis."""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Check if we should save only success frames
	save_only_success_frame = getattr(cfg, 'save_only_success_frame', True)
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

	# Initialize vision encoder for embedding analysis
	print('Initializing vision encoder for embedding analysis...')
	vision_encoder = PretrainedEncoder()

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

	# Storage for embedding analysis
	distances_list = []
	embeddings_list = []
	pixel_mse_list = []
	accepted_task_names = []
	done_indices_list = []  # Track when task was completed

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations with embedding analysis...')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = 'l1'
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
		_success = info['success'].clone()  # Store success at each timestep
		

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
					frames.append(ep_td['frame'])

					# Find the done index (when task was actually completed successfully)
					# Use the success tensor to find when the task reached the goal
					success_tensor = ep_td['success'].squeeze(0)  # [T+1]
					# Find first timestep where success == 1.0
					done_idx = torch.where(success_tensor >= 0.99)[0]  # Use 0.99 threshold for floating point
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						# If task never succeeded, use the last index (will show increasing distance)
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')

					# Compute embedding distances to the goal frame (done frame)
					print(f'Computing embedding distances for demo {demos_collected+1}...')
					print(f'Task completed at timestep {done_idx}/{len(ep_frames)-1}')
					distances, embeddings, pixel_mse = compute_embedding_distances(
						ep_frames.to('cuda'), vision_encoder, target_idx=done_idx, distance_metric=distance_metric)
					distances_list.append(distances)
					embeddings_list.append(embeddings)
					pixel_mse_list.append(pixel_mse)
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

	# Plot embedding distances
	print('\nGenerating embedding distance plots...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_embedding_distances_{distance_metric}.png'
	plot_embedding_distances(distances_list, accepted_task_names, done_indices_list, pixel_mse_list, plot_save_path, distance_metric='l2')

	# Plot embedding trajectories (PCA 2D)
	print('Generating embedding trajectory plots (PCA 2D)...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_embedding_trajectories_{distance_metric}.png'
	plot_embedding_trajectories_2d(embeddings_list, accepted_task_names, done_indices_list, trajectory_save_path)

	# Plot embedding trajectories (UMAP 3D)
	print('Generating embedding trajectory plots (UMAP 3D)...')
	trajectory_save_path_umap = f'{cfg.data_dir}/{cfg.task}_embedding_trajectories_umap3d_{distance_metric}.png'
	plot_embedding_trajectories_3d_umap(embeddings_list, accepted_task_names, done_indices_list, trajectory_save_path_umap)

	# # Save embedding distances as numpy arrays
	# for idx, (distances, task_name) in enumerate(zip(distances_list, accepted_task_names)):
	# 	np.save(f'{cfg.data_dir}/{cfg.task}_demo_{idx}_distances.npy', distances.numpy())
	# print(f'Saved embedding distances to {cfg.data_dir}/')

	# Save frames
	# if save_only_success_frame:
	# 	print('\nSaving success frames only...')
	# 	for demo_idx in range(frames.shape[0]):
	# 		demo_frames = frames[demo_idx]  # [T+1, C, H, W]
	# 		done_idx = done_indices_list[demo_idx]

	# 		# Save only the success frame (done frame)
	# 		success_frame = demo_frames[done_idx].float() / 255.0  # [C, H, W]
	# 		save_image(success_frame, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_success_frame.png')

	# 	print(f'Saved success frames for {demos_collected} demos to {cfg.data_dir}.')
	# else:
		# Save each demo as a separate image sequence
	print('\nSaving individual demo frame sequences...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]  # [T+1, C, H, W]

		# Save each frame individually with frame index
		for frame_idx in range(demo_frames.shape[0]):
			frame = demo_frames[frame_idx].float() / 255.0  # [C, H, W]
			save_image(frame, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_frame_{frame_idx:03d}.png')

		print(f'Saved demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

		print(f'Saved {demos_collected} demos to {cfg.data_dir}.')


if __name__ == '__main__':
	generate_demos()
