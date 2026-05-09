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

	Args:
		avoid_dir: Path to directory containing frames to avoid (e.g., push_blue completion frames)

	Returns:
		avoid_frames: Tensor of shape [N, C, H, W] in uint8 (0-255 range)
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
	print(f"Avoid frames tensor shape: {avoid_frames.shape}")

	return avoid_frames


def compute_negation_distances(frames, encoder, avoid_frames, distance_metric='cosine'):
	"""
	Compute distances between trajectory frames and a set of frames to AVOID.
	Higher distance = better (farther from what we want to avoid).

	Args:
		frames: Tensor of shape [T, C, H, W] containing episode frames
		encoder: PretrainedEncoder instance
		avoid_frames: Tensor of shape [N, C, H, W] containing frames to avoid
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		min_distances: Tensor of shape [T] — min distance to any avoid frame at each timestep
		mean_distances: Tensor of shape [T] — mean distance to all avoid frames at each timestep
		max_distances: Tensor of shape [T] — max distance to any avoid frame at each timestep
		embeddings: Tensor of shape [T, 768] — trajectory embeddings
		avoid_embeddings: Tensor of shape [N, 768] — avoid frame embeddings
		min_pixel_mse: Tensor of shape [T] — min pixel MSE to any avoid frame
		mean_pixel_mse: Tensor of shape [T] — mean pixel MSE to all avoid frames
	"""
	# Encode all trajectory frames
	embeddings = encoder(frames)  # [T, 768]

	# Encode all avoid frames
	avoid_embeddings = encoder(avoid_frames)  # [N, 768]

	T = embeddings.shape[0]
	N = avoid_embeddings.shape[0]

	# Compute pairwise distance matrix [T, N]
	if distance_metric == 'cosine':
		embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		avoid_norm = avoid_embeddings / (avoid_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
		cosine_sim = embeddings_norm @ avoid_norm.T
		dist_matrix = 1 - cosine_sim
	elif distance_metric == 'l2':
		dist_matrix = torch.norm(embeddings.unsqueeze(1) - avoid_embeddings.unsqueeze(0), dim=-1, p=2)
	elif distance_metric == 'l1':
		dist_matrix = torch.norm(embeddings.unsqueeze(1) - avoid_embeddings.unsqueeze(0), dim=-1, p=1)
	elif distance_metric == 'mse':
		dist_matrix = ((embeddings.unsqueeze(1) - avoid_embeddings.unsqueeze(0)) ** 2).mean(dim=-1)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	# Per-timestep metrics
	min_distances = dist_matrix.min(dim=1).values  # [T]
	nearest_avoid_idx = dist_matrix.argmin(dim=1)  # [T] — which avoid frame is closest at each timestep
	mean_distances = dist_matrix.mean(dim=1)  # [T]
	max_distances = dist_matrix.max(dim=1).values  # [T]

	# Compute pixel MSE
	frames_float = frames.float()  # [T, C, H, W]
	avoid_float = avoid_frames.float()  # [N, C, H, W]
	pixel_mse_matrix = ((frames_float.unsqueeze(1) - avoid_float.unsqueeze(0)) ** 2).mean(dim=(2, 3, 4))

	min_pixel_mse = pixel_mse_matrix.min(dim=1).values  # [T]
	mean_pixel_mse = pixel_mse_matrix.mean(dim=1)  # [T]

	return (min_distances.cpu(), mean_distances.cpu(), max_distances.cpu(),
			nearest_avoid_idx.cpu(),
			embeddings.cpu(), avoid_embeddings.cpu(),
			min_pixel_mse.cpu(), mean_pixel_mse.cpu())


def plot_negation_distances(min_dist_list, mean_dist_list, nearest_avoid_idx_list, task_names,
							min_mse_list, mean_mse_list, done_indices_list,
							save_path, distance_metric='cosine', num_avoid=0):
	"""
	Plot negation (avoidance) distances for multiple episodes.
	For negation: we WANT high distance (far from avoid frames).
	Highlights the frame with HIGHEST min-distance (best avoidance).

	4 columns: min embed dist, mean embed dist, min pixel MSE, mean pixel MSE.
	"""
	num_demos = len(min_dist_list)

	fig, axes = plt.subplots(num_demos, 4, figsize=(24, 4*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx, (min_dist, mean_dist, nearest_avoid_idx, min_mse, mean_mse, task_name, done_idx) in enumerate(
		zip(min_dist_list, mean_dist_list, nearest_avoid_idx_list, min_mse_list, mean_mse_list, task_names, done_indices_list)):

		timesteps = np.arange(len(min_dist))

		# Column 1: Min embedding distance to avoid set (higher = better avoidance)
		ax1 = axes[idx, 0]
		ax1.plot(timesteps, min_dist.numpy(), linewidth=2, color='steelblue', label='Min distance to avoid')

		# Highlight CLOSEST frame (worst avoidance — danger zone)
		min_of_min_idx = min_dist.argmin().item()
		worst_avoid_img_idx = nearest_avoid_idx[min_of_min_idx].item()
		ax1.axvline(x=min_of_min_idx, color='red', linestyle='--', linewidth=2,
				   label=f'Closest to avoid (t={min_of_min_idx}, img={worst_avoid_img_idx})', alpha=0.7)
		ax1.scatter([min_of_min_idx], [min_dist[min_of_min_idx]], color='red', s=150, zorder=5, marker='v')
		ax1.annotate(f'avoid img #{worst_avoid_img_idx}',
					 xy=(min_of_min_idx, min_dist[min_of_min_idx].item()),
					 xytext=(15, 15), textcoords='offset points',
					 fontsize=10, color='red', fontweight='bold',
					 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

		# Highlight FARTHEST frame (best avoidance)
		max_of_min_idx = min_dist.argmax().item()
		ax1.axvline(x=max_of_min_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Farthest from avoid (t={max_of_min_idx})', alpha=0.7)
		ax1.scatter([max_of_min_idx], [min_dist[max_of_min_idx]], color='green', s=150, zorder=5, marker='^')

		# Mark task done
		ax1.axvline(x=done_idx, color='orange', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)
		ax1.scatter([done_idx], [min_dist[done_idx]], color='orange', s=100, zorder=5, marker='D')

		ax1.set_xlabel('Timestep', fontsize=12)
		ax1.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax1.set_title(f'Task: {task_name} | Min Dist to Avoid Set ({num_avoid} frames)\n(Higher = Better Avoidance)',
					  fontsize=13, fontweight='bold')
		ax1.set_ylim(bottom=0)
		ax1.grid(True, alpha=0.3)
		ax1.legend(fontsize=9, loc='best')

		stats_text = (f'Min (worst): {min_dist.min().item():.4f}\n'
					  f'Max (best): {min_dist.max().item():.4f}\n'
					  f'Mean: {min_dist.mean().item():.4f}\n'
					  f'At done: {min_dist[done_idx].item():.4f}')
		ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# Column 2: Mean embedding distance to avoid set
		ax2 = axes[idx, 1]
		ax2.plot(timesteps, mean_dist.numpy(), linewidth=2, color='darkorange', label='Mean distance to avoid')

		min_mean_idx = mean_dist.argmin().item()
		ax2.axvline(x=min_mean_idx, color='red', linestyle='--', linewidth=2,
				   label=f'Closest to avoid (t={min_mean_idx})', alpha=0.7)
		ax2.scatter([min_mean_idx], [mean_dist[min_mean_idx]], color='red', s=150, zorder=5, marker='v')

		max_mean_idx = mean_dist.argmax().item()
		ax2.axvline(x=max_mean_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Farthest from avoid (t={max_mean_idx})', alpha=0.7)
		ax2.scatter([max_mean_idx], [mean_dist[max_mean_idx]], color='green', s=150, zorder=5, marker='^')

		ax2.axvline(x=done_idx, color='orange', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)
		ax2.scatter([done_idx], [mean_dist[done_idx]], color='orange', s=100, zorder=5, marker='D')

		ax2.set_xlabel('Timestep', fontsize=12)
		ax2.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		ax2.set_title(f'Task: {task_name} | Mean Dist to Avoid Set ({num_avoid} frames)\n(Higher = Better Avoidance)',
					  fontsize=13, fontweight='bold')
		ax2.set_ylim(bottom=0)
		ax2.grid(True, alpha=0.3)
		ax2.legend(fontsize=9, loc='best')

		stats_text_2 = (f'Min (worst): {mean_dist.min().item():.4f}\n'
						f'Max (best): {mean_dist.max().item():.4f}\n'
						f'Mean: {mean_dist.mean().item():.4f}\n'
						f'At done: {mean_dist[done_idx].item():.4f}')
		ax2.text(0.02, 0.98, stats_text_2, transform=ax2.transAxes, fontsize=10,
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# Column 3: Min pixel MSE to avoid set
		ax3 = axes[idx, 2]
		ax3.plot(timesteps, min_mse.numpy(), linewidth=2, color='purple', label='Min pixel MSE to avoid')

		min_of_min_mse_idx = min_mse.argmin().item()
		ax3.axvline(x=min_of_min_mse_idx, color='red', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_of_min_mse_idx})', alpha=0.7)
		ax3.scatter([min_of_min_mse_idx], [min_mse[min_of_min_mse_idx]], color='red', s=150, zorder=5, marker='v')

		max_of_min_mse_idx = min_mse.argmax().item()
		ax3.axvline(x=max_of_min_mse_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Farthest (t={max_of_min_mse_idx})', alpha=0.7)
		ax3.scatter([max_of_min_mse_idx], [min_mse[max_of_min_mse_idx]], color='green', s=150, zorder=5, marker='^')

		ax3.axvline(x=done_idx, color='orange', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)

		ax3.set_xlabel('Timestep', fontsize=12)
		ax3.set_ylabel('Pixel MSE', fontsize=12)
		ax3.set_title(f'Task: {task_name} | Min Pixel MSE to Avoid Set\n(Higher = Better Avoidance)',
					  fontsize=13, fontweight='bold')
		ax3.set_ylim(bottom=0)
		ax3.grid(True, alpha=0.3)
		ax3.legend(fontsize=9, loc='best')

		# Column 4: Mean pixel MSE to avoid set
		ax4 = axes[idx, 3]
		ax4.plot(timesteps, mean_mse.numpy(), linewidth=2, color='brown', label='Mean pixel MSE to avoid')

		min_mean_mse_idx = mean_mse.argmin().item()
		ax4.axvline(x=min_mean_mse_idx, color='red', linestyle='--', linewidth=2,
				   label=f'Closest (t={min_mean_mse_idx})', alpha=0.7)
		ax4.scatter([min_mean_mse_idx], [mean_mse[min_mean_mse_idx]], color='red', s=150, zorder=5, marker='v')

		max_mean_mse_idx = mean_mse.argmax().item()
		ax4.axvline(x=max_mean_mse_idx, color='green', linestyle='--', linewidth=2,
				   label=f'Farthest (t={max_mean_mse_idx})', alpha=0.7)
		ax4.scatter([max_mean_mse_idx], [mean_mse[max_mean_mse_idx]], color='green', s=150, zorder=5, marker='^')

		ax4.axvline(x=done_idx, color='orange', linestyle=':', linewidth=2,
				   label=f'Task Done (t={done_idx})', alpha=0.7)

		ax4.set_xlabel('Timestep', fontsize=12)
		ax4.set_ylabel('Pixel MSE', fontsize=12)
		ax4.set_title(f'Task: {task_name} | Mean Pixel MSE to Avoid Set\n(Higher = Better Avoidance)',
					  fontsize=13, fontweight='bold')
		ax4.set_ylim(bottom=0)
		ax4.grid(True, alpha=0.3)
		ax4.legend(fontsize=9, loc='best')

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved negation distance plot to {save_path}')


def plot_negation_trajectories_2d(embeddings_list, task_names, avoid_embeddings, save_path):
	"""
	Plot 2D trajectory with avoid frames marked as danger zones.
	"""
	from sklearn.decomposition import PCA

	num_demos = len(embeddings_list)

	all_embeddings = torch.cat(embeddings_list + [avoid_embeddings], dim=0).numpy()

	pca = PCA(n_components=2)
	all_embeddings_2d = pca.fit_transform(all_embeddings)

	embeddings_2d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_2d_list.append(all_embeddings_2d[start_idx:end_idx])
		start_idx = end_idx
	avoid_embeddings_2d = all_embeddings_2d[start_idx:]

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

		# Highlight end
		ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1],
				  color='blue', s=200, marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)

		# Highlight avoid frames as red Xs (danger zone)
		ax.scatter(avoid_embeddings_2d[:, 0], avoid_embeddings_2d[:, 1],
				  color='red', s=250, marker='X', label=f'AVOID ({len(avoid_embeddings_2d)})', zorder=5,
				  edgecolors='black', linewidths=1.5, alpha=0.8)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		ax.set_title(f'Task: {task_name}\nNegation Trajectory (PCA) — Stay Away from Red', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved negation trajectory plot to {save_path}')


def plot_negation_trajectories_3d_umap(embeddings_list, task_names, avoid_embeddings, save_path):
	"""
	Plot 3D trajectory with avoid frames marked as danger zones using UMAP.
	"""
	try:
		import umap
	except ImportError:
		print("UMAP not installed. Skipping UMAP visualization. Install with: pip install umap-learn")
		return

	num_demos = len(embeddings_list)

	all_embeddings = torch.cat(embeddings_list + [avoid_embeddings], dim=0).numpy()

	print("Computing UMAP projection to 3D...")
	reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
	all_embeddings_3d = reducer.fit_transform(all_embeddings)

	embeddings_3d_list = []
	start_idx = 0
	for embeddings in embeddings_list:
		end_idx = start_idx + len(embeddings)
		embeddings_3d_list.append(all_embeddings_3d[start_idx:end_idx])
		start_idx = end_idx
	avoid_embeddings_3d = all_embeddings_3d[start_idx:]

	fig = plt.figure(figsize=(8*num_demos, 7))

	for idx, (embeddings_3d, task_name) in enumerate(zip(embeddings_3d_list, task_names)):
		ax = fig.add_subplot(1, num_demos, idx+1, projection='3d')

		timesteps = np.arange(len(embeddings_3d))
		scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
							c=timesteps, cmap='viridis', s=30, alpha=0.6)

		ax.plot(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
			   linewidth=1.5, alpha=0.4, color='steelblue')

		# Start
		ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2],
				  color='green', s=300, marker='o', label='Start', zorder=10,
				  edgecolors='black', linewidths=2)

		# Avoid frames (danger zone)
		ax.scatter(avoid_embeddings_3d[:, 0], avoid_embeddings_3d[:, 1], avoid_embeddings_3d[:, 2],
				  color='red', s=400, marker='X', label=f'AVOID ({len(avoid_embeddings_3d)})', zorder=10,
				  edgecolors='black', linewidths=1.5, alpha=0.8)

		ax.set_xlabel('UMAP 1', fontsize=11)
		ax.set_ylabel('UMAP 2', fontsize=11)
		ax.set_zlabel('UMAP 3', fontsize=11)
		ax.set_title(f'Task: {task_name}\nNegation Trajectory (UMAP 3D)', fontsize=13, fontweight='bold')
		ax.legend(fontsize=9, loc='upper left')

		cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
		cbar.set_label('Timestep', fontsize=10)

		ax.view_init(elev=20, azim=45)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved UMAP 3D negation trajectory plot to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""
	Generates demonstrations and measures negation/avoidance distance.

	Runs a task (e.g., push_green) and computes how FAR the trajectory stays
	from a set of "avoid" frames (e.g., push_blue completion frames).
	Higher distance to avoid frames = better avoidance.

	Usage:
		python generate_demos_with_negation.py task=rd-push_green +num_demos=2 \
			+avoid_frames_dir=<path_to_push_blue_frames> +distance_metric=cosine
	"""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Check for avoid frames directory
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

	# Load avoid frames
	print(f'Loading avoid frames from {avoid_frames_dir}...')
	avoid_frames = load_avoid_frames(avoid_frames_dir)  # [N, C, H, W]
	num_avoid = avoid_frames.shape[0]

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

	# Storage for negation analysis
	min_dist_list = []
	mean_dist_list = []
	nearest_avoid_idx_list = []
	embeddings_list = []
	min_mse_list = []
	mean_mse_list = []
	accepted_task_names = []
	done_indices_list = []
	avoid_embeddings_stored = None

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations with negation distance analysis ({num_avoid} avoid frames)...')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = getattr(cfg, 'distance_metric', 'mse')
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

					# Find the done index
					success_tensor = ep_td['success'].squeeze(0)  # [T+1]
					done_idx = torch.where(success_tensor >= 0.99)[0]
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')
					done_indices_list.append(done_idx)

					# Compute negation distances
					print(f'Computing negation distances for demo {demos_collected+1} ({num_avoid} avoid frames)...')
					(min_distances, mean_distances, max_distances,
					 nearest_avoid_idx,
					 embeddings, avoid_embs,
					 min_pixel_mse, mean_pixel_mse) = compute_negation_distances(
						ep_frames.to('cuda'), vision_encoder, avoid_frames.to('cuda'), distance_metric=distance_metric)

					min_dist_list.append(min_distances)
					mean_dist_list.append(mean_distances)
					nearest_avoid_idx_list.append(nearest_avoid_idx)
					embeddings_list.append(embeddings)
					min_mse_list.append(min_pixel_mse)
					mean_mse_list.append(mean_pixel_mse)
					accepted_task_names.append(cfg.tasks[i])

					if avoid_embeddings_stored is None:
						avoid_embeddings_stored = avoid_embs

					# Print negation summary
					# For negation: closest to avoid = worst, farthest = best
					closest_idx = min_distances.argmin().item()
					farthest_idx = min_distances.argmax().item()
					worst_avoid_img = nearest_avoid_idx[closest_idx].item()
					print(f'  CLOSEST to avoid (worst): t={closest_idx} -> avoid img #{worst_avoid_img} (min_dist={min_distances[closest_idx]:.4f})')
					print(f'  FARTHEST from avoid (best): t={farthest_idx} (min_dist={min_distances[farthest_idx]:.4f})')
					print(f'  At task done (t={done_idx}): min_dist={min_distances[done_idx]:.4f}')

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

	# Create data directory
	os.makedirs(cfg.data_dir, exist_ok=True)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames, dim=0)

	# Plot negation distances
	print('\nGenerating negation distance plots...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_negation_distances_{distance_metric}.png'
	plot_negation_distances(min_dist_list, mean_dist_list, nearest_avoid_idx_list, accepted_task_names,
							min_mse_list, mean_mse_list, done_indices_list,
							plot_save_path, distance_metric=distance_metric, num_avoid=num_avoid)

	# Plot negation trajectories (PCA 2D)
	print('Generating negation trajectory plots (PCA 2D)...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_negation_trajectories_{distance_metric}.png'
	plot_negation_trajectories_2d(embeddings_list, accepted_task_names, avoid_embeddings_stored, trajectory_save_path)

	# Plot negation trajectories (UMAP 3D)
	print('Generating negation trajectory plots (UMAP 3D)...')
	trajectory_save_path_umap = f'{cfg.data_dir}/{cfg.task}_negation_trajectories_umap3d_{distance_metric}.png'
	plot_negation_trajectories_3d_umap(embeddings_list, accepted_task_names, avoid_embeddings_stored, trajectory_save_path_umap)

	# Save grid image for each demo
	print('\nSaving grid visualizations for each demo...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]
		demo_frames_normalized = demo_frames.float() / 255.0
		nrow = int(np.ceil(np.sqrt(demo_frames.shape[0])))
		grid = make_grid(demo_frames_normalized, nrow=nrow)
		save_image(grid, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_grid.png')
		print(f'Saved grid for demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

	# Save side-by-side of closest trajectory frame and its nearest avoid frame per demo
	print('\nSaving worst-avoidance side-by-side comparisons...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]  # [T+1, C, H, W]
		min_dist = min_dist_list[demo_idx]
		nearest_avoid_idx = nearest_avoid_idx_list[demo_idx]

		# Worst avoidance: trajectory timestep closest to any avoid frame
		worst_t = min_dist.argmin().item()
		worst_avoid_img = nearest_avoid_idx[worst_t].item()

		traj_frame = demo_frames[worst_t].float() / 255.0  # [C, H, W]
		avoid_frame = avoid_frames[worst_avoid_img].float() / 255.0  # [C, H, W]

		pair = torch.stack([traj_frame, avoid_frame], dim=0)  # [2, C, H, W]
		pair_grid = make_grid(pair, nrow=2, padding=4, pad_value=1.0)
		pair_path = f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_worst_avoidance_t{worst_t}_avoid{worst_avoid_img}.png'
		save_image(pair_grid, pair_path)
		print(f'  Demo {demo_idx}: worst at t={worst_t} vs avoid img #{worst_avoid_img} (dist={min_dist[worst_t]:.4f}) -> {pair_path}')

	print(f'\nCompleted! Saved {demos_collected} demos to {cfg.data_dir}.')
	print(f'Avoid frames directory: {avoid_frames_dir} ({num_avoid} frames)')
	print(f'Distance metric: {distance_metric}')
	print(f'Negation analysis: higher distance to avoid frames = better avoidance')


if __name__ == '__main__':
	generate_demos()
