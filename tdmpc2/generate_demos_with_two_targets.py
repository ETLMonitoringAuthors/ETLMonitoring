import os

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


def load_target_frame(target_frame_path):
	"""
	Load a target frame from disk.

	Args:
		target_frame_path: Path to the target frame image file

	Returns:
		target_frame: Tensor of shape [C, H, W] in uint8 (0-255 range)
	"""
	assert os.path.exists(target_frame_path), f"Target frame not found at {target_frame_path}"

	# Read image (returns uint8 tensor in range [0, 255])
	target_frame = read_image(target_frame_path)  # [C, H, W]

	print(f"Loaded target frame from {target_frame_path}")
	print(f"Target frame shape: {target_frame.shape}")

	return target_frame


def compute_embedding_distances_two_targets(frames, encoder, target_frame_1, target_frame_2, distance_metric='cosine'):
	"""
	Compute distances between two external target frame embeddings and all frame embeddings.

	Args:
		frames: Tensor of shape [T, C, H, W] containing episode frames
		encoder: PretrainedEncoder instance
		target_frame_1: First external target frame of shape [C, H, W]
		target_frame_2: Second external target frame of shape [C, H, W]
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		distances_1: Tensor of shape [T] containing distances to target 1
		distances_2: Tensor of shape [T] containing distances to target 2
		embeddings: Tensor of shape [T, 768] containing all embeddings
		pixel_mse_1: Tensor of shape [T] containing pixel MSE to target 1
		pixel_mse_2: Tensor of shape [T] containing pixel MSE to target 2
	"""
	# Encode all frames from trajectory
	embeddings = encoder(frames)  # [T, 768]

	# Encode both external target frames
	target_frame_1_batched = target_frame_1.unsqueeze(0)  # [1, C, H, W]
	target_embedding_1 = encoder(target_frame_1_batched)  # [1, 768]

	target_frame_2_batched = target_frame_2.unsqueeze(0)  # [1, C, H, W]
	target_embedding_2 = encoder(target_frame_2_batched)  # [1, 768]

	# Compute embedding distances to target 1
	if distance_metric == 'cosine':
		# Cosine distance = 1 - cosine_similarity
		target_norm_1 = target_embedding_1 / (target_embedding_1.norm(dim=-1, keepdim=True) + 1e-8)
		embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim_1 = (embeddings_norm * target_norm_1).sum(dim=-1)
		distances_1 = 1 - cosine_sim_1

		target_norm_2 = target_embedding_2 / (target_embedding_2.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim_2 = (embeddings_norm * target_norm_2).sum(dim=-1)
		distances_2 = 1 - cosine_sim_2
	elif distance_metric == 'l2':
		# Euclidean distance
		distances_1 = torch.norm(embeddings - target_embedding_1, dim=-1, p=2)
		distances_2 = torch.norm(embeddings - target_embedding_2, dim=-1, p=2)
	elif distance_metric == 'l1':
		# Manhattan distance
		distances_1 = torch.norm(embeddings - target_embedding_1, dim=-1, p=1)
		distances_2 = torch.norm(embeddings - target_embedding_2, dim=-1, p=1)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	# Compute pixel MSE between each frame and both target frames
	target_frame_1_float = target_frame_1.unsqueeze(0).float()  # [1, C, H, W]
	target_frame_2_float = target_frame_2.unsqueeze(0).float()  # [1, C, H, W]
	frames_float = frames.float()  # [T, C, H, W]
	pixel_mse_1 = ((frames_float - target_frame_1_float) ** 2).mean(dim=(1, 2, 3))  # [T]
	pixel_mse_2 = ((frames_float - target_frame_2_float) ** 2).mean(dim=(1, 2, 3))  # [T]

	return distances_1.cpu(), distances_2.cpu(), embeddings.cpu(), pixel_mse_1.cpu(), pixel_mse_2.cpu()


def plot_embedding_distances_two_targets(distances_1_list, distances_2_list, task_names, pixel_mse_1_list, pixel_mse_2_list, save_path, distance_metric='cosine'):
	"""
	Plot embedding distances to two targets for multiple episodes.

	Args:
		distances_1_list: List of distance tensors to target 1, each of shape [T]
		distances_2_list: List of distance tensors to target 2, each of shape [T]
		task_names: List of task names corresponding to each episode
		pixel_mse_1_list: List of pixel MSE tensors to target 1, each of shape [T]
		pixel_mse_2_list: List of pixel MSE tensors to target 2, each of shape [T]
		save_path: Path to save the plot
		distance_metric: Type of distance metric used
	"""
	num_demos = len(distances_1_list)

	# Create figure with subplots - 4 columns per demo (embedding dist 1, embedding dist 2, pixel MSE 1, pixel MSE 2)
	fig, axes = plt.subplots(num_demos, 4, figsize=(24, 4*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx, (distances_1, distances_2, pixel_mse_1, pixel_mse_2, task_name) in enumerate(
		zip(distances_1_list, distances_2_list, pixel_mse_1_list, pixel_mse_2_list, task_names)):

		timesteps = np.arange(len(distances_1))

		# Plot embedding distance to target 1
		ax1 = axes[idx, 0]
		ax1.plot(timesteps, distances_1.numpy(), linewidth=2, color='steelblue', label='Embedding distance')

		min_dist_idx_1 = distances_1.argmin().item()
		ax1.axvline(x=min_dist_idx_1, color='green', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_dist_idx_1})', alpha=0.7)
		ax1.scatter([min_dist_idx_1], [distances_1[min_dist_idx_1]], color='green', s=150, zorder=5, marker='o')

		ax1.axvline(x=len(distances_1)-1, color='red', linestyle='--', linewidth=2,
				   label=f'End (t={len(distances_1)-1})', alpha=0.7)
		ax1.scatter([len(distances_1)-1], [distances_1[-1]], color='red', s=100, zorder=5)

		ax1.set_xlabel('Timestep', fontsize=12)
		ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax1.set_title(f'Task: {task_name} | Distance to Target 1', fontsize=14, fontweight='bold')
		ax1.set_ylim(bottom=0)
		ax1.grid(True, alpha=0.3)
		ax1.legend(fontsize=10, loc='best')

		# Plot embedding distance to target 2
		ax2 = axes[idx, 1]
		ax2.plot(timesteps, distances_2.numpy(), linewidth=2, color='darkorange', label='Embedding distance')

		min_dist_idx_2 = distances_2.argmin().item()
		ax2.axvline(x=min_dist_idx_2, color='green', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_dist_idx_2})', alpha=0.7)
		ax2.scatter([min_dist_idx_2], [distances_2[min_dist_idx_2]], color='green', s=150, zorder=5, marker='o')

		ax2.axvline(x=len(distances_2)-1, color='red', linestyle='--', linewidth=2,
				   label=f'End (t={len(distances_2)-1})', alpha=0.7)
		ax2.scatter([len(distances_2)-1], [distances_2[-1]], color='red', s=100, zorder=5)

		ax2.set_xlabel('Timestep', fontsize=12)
		ax2.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax2.set_title(f'Task: {task_name} | Distance to Target 2', fontsize=14, fontweight='bold')
		ax2.set_ylim(bottom=0)
		ax2.grid(True, alpha=0.3)
		ax2.legend(fontsize=10, loc='best')

		# Plot pixel MSE to target 1
		ax3 = axes[idx, 2]
		ax3.plot(timesteps, pixel_mse_1.numpy(), linewidth=2, color='purple', label='Pixel MSE')

		min_mse_idx_1 = pixel_mse_1.argmin().item()
		ax3.axvline(x=min_mse_idx_1, color='green', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_mse_idx_1})', alpha=0.7)
		ax3.scatter([min_mse_idx_1], [pixel_mse_1[min_mse_idx_1]], color='green', s=150, zorder=5, marker='o')

		ax3.axvline(x=len(pixel_mse_1)-1, color='red', linestyle='--', linewidth=2,
				   label=f'End (t={len(pixel_mse_1)-1})', alpha=0.7)
		ax3.scatter([len(pixel_mse_1)-1], [pixel_mse_1[-1]], color='red', s=100, zorder=5)

		ax3.set_xlabel('Timestep', fontsize=12)
		ax3.set_ylabel('Pixel MSE', fontsize=12)
		ax3.set_title(f'Task: {task_name} | Pixel MSE to Target 1', fontsize=14, fontweight='bold')
		ax3.set_ylim(bottom=0)
		ax3.grid(True, alpha=0.3)
		ax3.legend(fontsize=10, loc='best')

		# Plot pixel MSE to target 2
		ax4 = axes[idx, 3]
		ax4.plot(timesteps, pixel_mse_2.numpy(), linewidth=2, color='brown', label='Pixel MSE')

		min_mse_idx_2 = pixel_mse_2.argmin().item()
		ax4.axvline(x=min_mse_idx_2, color='green', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_mse_idx_2})', alpha=0.7)
		ax4.scatter([min_mse_idx_2], [pixel_mse_2[min_mse_idx_2]], color='green', s=150, zorder=5, marker='o')

		ax4.axvline(x=len(pixel_mse_2)-1, color='red', linestyle='--', linewidth=2,
				   label=f'End (t={len(pixel_mse_2)-1})', alpha=0.7)
		ax4.scatter([len(pixel_mse_2)-1], [pixel_mse_2[-1]], color='red', s=100, zorder=5)

		ax4.set_xlabel('Timestep', fontsize=12)
		ax4.set_ylabel('Pixel MSE', fontsize=12)
		ax4.set_title(f'Task: {task_name} | Pixel MSE to Target 2', fontsize=14, fontweight='bold')
		ax4.set_ylim(bottom=0)
		ax4.grid(True, alpha=0.3)
		ax4.legend(fontsize=10, loc='best')

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved embedding distance plot to {save_path}')


def plot_embedding_trajectories_2d_two_targets(embeddings_list, task_names, target_embedding_1, target_embedding_2, save_path):
	"""
	Plot 2D trajectory of embeddings using PCA projection with two targets.

	Args:
		embeddings_list: List of embedding tensors, each of shape [T, 768]
		task_names: List of task names
		target_embedding_1: First external target embedding of shape [768]
		target_embedding_2: Second external target embedding of shape [768]
		save_path: Path to save the plot
	"""
	from sklearn.decomposition import PCA

	num_demos = len(embeddings_list)

	# Concatenate all embeddings + both target embeddings for PCA
	all_embeddings = torch.cat(embeddings_list + [target_embedding_1.unsqueeze(0), target_embedding_2.unsqueeze(0)], dim=0).numpy()

	# Apply PCA to reduce to 2D
	pca = PCA(n_components=2)
	all_embeddings_2d = pca.fit_transform(all_embeddings)

	# Split back into individual episodes + targets
	embeddings_2d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_2d_list.append(all_embeddings_2d[start_idx:end_idx])
		start_idx = end_idx
	target_embedding_1_2d = all_embeddings_2d[-2]  # Second to last is target 1
	target_embedding_2_2d = all_embeddings_2d[-1]  # Last is target 2

	# Plot
	fig, axes = plt.subplots(1, num_demos, figsize=(6*num_demos, 5))
	if num_demos == 1:
		axes = [axes]

	for idx, (embeddings_2d, task_name) in enumerate(zip(embeddings_2d_list, task_names)):
		ax = axes[idx]

		# Plot trajectory
		ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'o-',
				linewidth=2, markersize=4, alpha=0.6, color='steelblue')

		# Highlight start
		ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
				  color='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)

		# Highlight both external targets
		ax.scatter(target_embedding_1_2d[0], target_embedding_1_2d[1],
				  color='red', s=300, marker='*', label='Target 1', zorder=5, edgecolors='black', linewidths=2)
		ax.scatter(target_embedding_2_2d[0], target_embedding_2_2d[1],
				  color='orange', s=300, marker='*', label='Target 2', zorder=5, edgecolors='black', linewidths=2)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		ax.set_title(f'Task: {task_name}\nEmbedding Trajectory (PCA)', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved embedding trajectory plot to {save_path}')


def plot_embedding_trajectories_3d_umap_two_targets(embeddings_list, task_names, target_embedding_1, target_embedding_2, save_path):
	"""
	Plot 3D trajectory of embeddings using UMAP projection with two targets.

	Args:
		embeddings_list: List of embedding tensors, each of shape [T, 768]
		task_names: List of task names
		target_embedding_1: First external target embedding of shape [768]
		target_embedding_2: Second external target embedding of shape [768]
		save_path: Path to save the plot
	"""
	try:
		import umap
	except ImportError:
		print("UMAP not installed. Skipping UMAP visualization. Install with: pip install umap-learn")
		return

	num_demos = len(embeddings_list)

	# Concatenate all embeddings + both targets for UMAP
	all_embeddings = torch.cat(embeddings_list + [target_embedding_1.unsqueeze(0), target_embedding_2.unsqueeze(0)], dim=0).numpy()

	# Apply UMAP to reduce to 3D
	print("Computing UMAP projection to 3D...")
	reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
	all_embeddings_3d = reducer.fit_transform(all_embeddings)

	# Split back into individual episodes + targets
	embeddings_3d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_3d_list.append(all_embeddings_3d[start_idx:end_idx])
		start_idx = end_idx
	target_embedding_1_3d = all_embeddings_3d[-2]  # Second to last is target 1
	target_embedding_2_3d = all_embeddings_3d[-1]  # Last is target 2

	# Plot in 3D
	fig = plt.figure(figsize=(8*num_demos, 7))

	for idx, (embeddings_3d, task_name) in enumerate(zip(embeddings_3d_list, task_names)):
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

		# Highlight both external targets
		ax.scatter(target_embedding_1_3d[0], target_embedding_1_3d[1], target_embedding_1_3d[2],
				  color='red', s=400, marker='*', label='Target 1', zorder=10,
				  edgecolors='black', linewidths=2)
		ax.scatter(target_embedding_2_3d[0], target_embedding_2_3d[1], target_embedding_2_3d[2],
				  color='orange', s=400, marker='*', label='Target 2', zorder=10,
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
	"""Generates demonstrations with embedding distance analysis using two external target frames."""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Check for both target frame paths
	assert hasattr(cfg, 'target_frame_path_1'), \
		'Please specify path to first target frame via +target_frame_path_1=<path_to_image>'
	assert hasattr(cfg, 'target_frame_path_2'), \
		'Please specify path to second target frame via +target_frame_path_2=<path_to_image>'
	target_frame_path_1 = cfg.target_frame_path_1
	target_frame_path_2 = cfg.target_frame_path_2

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

	# Load both external target frames
	print(f'Loading first external target frame from {target_frame_path_1}...')
	target_frame_1 = load_target_frame(target_frame_path_1)  # [C, H, W]

	print(f'Loading second external target frame from {target_frame_path_2}...')
	target_frame_2 = load_target_frame(target_frame_path_2)  # [C, H, W]

	# Encode both target frames once
	target_frame_1_cuda = target_frame_1.unsqueeze(0).to('cuda')  # [1, C, H, W]
	target_embedding_1 = vision_encoder(target_frame_1_cuda).squeeze(0).cpu()  # [768]
	print(f'Target 1 embedding computed: shape {target_embedding_1.shape}')

	target_frame_2_cuda = target_frame_2.unsqueeze(0).to('cuda')  # [1, C, H, W]
	target_embedding_2 = vision_encoder(target_frame_2_cuda).squeeze(0).cpu()  # [768]
	print(f'Target 2 embedding computed: shape {target_embedding_2.shape}')

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
	distances_1_list = []
	distances_2_list = []
	embeddings_list = []
	pixel_mse_1_list = []
	pixel_mse_2_list = []
	accepted_task_names = []

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations with two-target embedding analysis...')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = getattr(cfg, 'distance_metric', 'l1')
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
					frames.append(ep_frames)

					# Compute embedding distances to both external target frames
					print(f'Computing embedding distances to both targets for demo {demos_collected+1}...')
					distances_1, distances_2, embeddings, pixel_mse_1, pixel_mse_2 = compute_embedding_distances_two_targets(
						ep_frames.to('cuda'), vision_encoder, target_frame_1.to('cuda'), target_frame_2.to('cuda'), distance_metric=distance_metric)
					distances_1_list.append(distances_1)
					distances_2_list.append(distances_2)
					embeddings_list.append(embeddings)
					pixel_mse_1_list.append(pixel_mse_1)
					pixel_mse_2_list.append(pixel_mse_2)
					accepted_task_names.append(cfg.tasks[i])

					# Print closest frames to both targets
					min_dist_idx_1 = distances_1.argmin().item()
					min_dist_idx_2 = distances_2.argmin().item()
					print(f'Closest frame to target 1: t={min_dist_idx_1} (distance={distances_1[min_dist_idx_1]:.4f})')
					print(f'Closest frame to target 2: t={min_dist_idx_2} (distance={distances_2[min_dist_idx_2]:.4f})')

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

	# Plot embedding distances to both targets
	print('\nGenerating embedding distance plots for both targets...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_embedding_distances_two_targets_{distance_metric}.png'
	plot_embedding_distances_two_targets(distances_1_list, distances_2_list, accepted_task_names,
										 pixel_mse_1_list, pixel_mse_2_list, plot_save_path, distance_metric=distance_metric)

	# Plot embedding trajectories (PCA 2D)
	print('Generating embedding trajectory plots (PCA 2D)...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_embedding_trajectories_two_targets_{distance_metric}.png'
	plot_embedding_trajectories_2d_two_targets(embeddings_list, accepted_task_names, target_embedding_1, target_embedding_2, trajectory_save_path)

	# Plot embedding trajectories (UMAP 3D)
	print('Generating embedding trajectory plots (UMAP 3D)...')
	trajectory_save_path_umap = f'{cfg.data_dir}/{cfg.task}_embedding_trajectories_umap3d_two_targets_{distance_metric}.png'
	plot_embedding_trajectories_3d_umap_two_targets(embeddings_list, accepted_task_names, target_embedding_1, target_embedding_2, trajectory_save_path_umap)

	# Save grid image for each demo
	print('\nSaving grid visualizations for each demo...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]  # [T+1, C, H, W]
		demo_frames_normalized = demo_frames.float() / 255.0  # Normalize to [0, 1]

		# Create grid with all frames from this demo in one row
		grid = make_grid(demo_frames_normalized, nrow=demo_frames.shape[0])
		save_image(grid, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_grid.png')
		print(f'Saved grid for demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

	print(f'\nCompleted! Saved {demos_collected} demos to {cfg.data_dir}.')
	print(f'Target 1 frame used: {target_frame_path_1}')
	print(f'Target 2 frame used: {target_frame_path_2}')


if __name__ == '__main__':
	generate_demos()
