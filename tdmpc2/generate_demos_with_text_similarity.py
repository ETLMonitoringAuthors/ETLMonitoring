import os

import torch
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
from torchvision.utils import make_grid, save_image
from transformers import CLIPProcessor, CLIPModel

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


class CLIPEncoder:
	"""CLIP encoder for both images and text."""

	def __init__(self, model_name="openai/clip-vit-base-patch32"):
		print(f'Initializing CLIP encoder: {model_name}...')
		self.processor = CLIPProcessor.from_pretrained(model_name)
		self.model = CLIPModel.from_pretrained(model_name).to('cuda')
		self.model.eval()
		print(f'CLIP encoder initialized successfully')

	@torch.no_grad()
	def encode_images(self, frames):
		"""
		Encode images using CLIP vision encoder.

		Args:
			frames: Tensor of shape [T, C, H, W] containing episode frames (uint8, 0-255)

		Returns:
			embeddings: Tensor of shape [T, D] containing image embeddings
		"""
		# Convert from [T, C, H, W] uint8 to [T, H, W, C] numpy for PIL processing
		frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()

		# Process images in batches
		batch_size = 32
		embeddings_list = []

		for i in range(0, len(frames_np), batch_size):
			batch = frames_np[i:i+batch_size]
			# CLIP processor expects list of PIL images or numpy arrays in [H, W, C] format
			inputs = self.processor(images=list(batch), return_tensors="pt", padding=True)
			inputs = {k: v.to('cuda') for k, v in inputs.items()}

			# Get image embeddings
			image_features = self.model.get_image_features(**inputs)
			# Normalize embeddings (CLIP embeddings are typically normalized)
			image_features = image_features / image_features.norm(dim=-1, keepdim=True)
			embeddings_list.append(image_features)

		embeddings = torch.cat(embeddings_list, dim=0)
		return embeddings.cpu()

	@torch.no_grad()
	def encode_text(self, text_prompts):
		"""
		Encode text prompts using CLIP text encoder.

		Args:
			text_prompts: str or list of str containing text descriptions

		Returns:
			embeddings: Tensor of shape [N, D] containing text embeddings
		"""
		if isinstance(text_prompts, str):
			text_prompts = [text_prompts]

		inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
		inputs = {k: v.to('cuda') for k, v in inputs.items()}

		# Get text embeddings
		text_features = self.model.get_text_features(**inputs)
		# Normalize embeddings
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)

		return text_features.cpu()


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


def compute_text_similarities(frames, encoder, text_prompt, distance_metric='cosine'):
	"""
	Compute similarities between frames and a text prompt using CLIP.

	Args:
		frames: Tensor of shape [T, C, H, W] containing episode frames
		encoder: CLIPEncoder instance
		text_prompt: str describing the target state (e.g., "robot holding a block")
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		similarities: Tensor of shape [T] containing similarity scores (higher = more similar)
		distances: Tensor of shape [T] containing distance scores (lower = more similar)
		image_embeddings: Tensor of shape [T, D] containing image embeddings
		text_embedding: Tensor of shape [1, D] containing text embedding
	"""
	# Encode all frames
	image_embeddings = encoder.encode_images(frames)  # [T, D]

	# Encode text prompt
	text_embedding = encoder.encode_text(text_prompt)  # [1, D]

	# Compute similarities/distances
	if distance_metric == 'cosine':
		similarities = torch.nn.functional.cosine_similarity(image_embeddings, text_embedding, dim=-1)
		distances = 1 - similarities
	elif distance_metric == 'l2':
		# Euclidean distance
		distances = torch.norm(image_embeddings - text_embedding, dim=-1, p=2)
		# Convert to similarity (inverse distance)
		similarities = 1 / (1 + distances)
	elif distance_metric == 'l1':
		# Manhattan distance
		distances = torch.norm(image_embeddings - text_embedding, dim=-1, p=1)
		# Convert to similarity (inverse distance)
		similarities = 1 / (1 + distances)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	return similarities, distances, image_embeddings, text_embedding


def compute_text_similarities_with_avoid(image_embeddings, encoder, avoid_prompt, distance_metric='cosine'):
	"""
	Compute similarities between already-encoded frames and an avoid text prompt.

	Args:
		image_embeddings: Tensor of shape [T, D] — already encoded image embeddings
		encoder: CLIPEncoder instance
		avoid_prompt: str describing what to avoid
		distance_metric: 'cosine', 'l2', or 'l1'

	Returns:
		similarities: Tensor of shape [T] (lower = better avoidance)
		distances: Tensor of shape [T] (higher = better avoidance)
		avoid_embedding: Tensor of shape [1, D]
	"""
	avoid_embedding = encoder.encode_text(avoid_prompt)  # [1, D]

	if distance_metric == 'cosine':
		similarities = torch.nn.functional.cosine_similarity(image_embeddings, avoid_embedding, dim=-1)
		distances = 1 - similarities
	elif distance_metric == 'l2':
		distances = torch.norm(image_embeddings - avoid_embedding, dim=-1, p=2)
		similarities = 1 / (1 + distances)
	elif distance_metric == 'l1':
		distances = torch.norm(image_embeddings - avoid_embedding, dim=-1, p=1)
		similarities = 1 / (1 + distances)
	else:
		raise ValueError(f"Unknown distance metric: {distance_metric}")

	return similarities, distances, avoid_embedding


def plot_text_similarities(similarities_list, distances_list, task_names, text_prompts,
						   done_indices_list, save_path, distance_metric='cosine',
						   avoid_similarities_list=None, avoid_distances_list=None, avoid_prompt=None):
	"""
	Plot text-to-image similarities for multiple episodes.
	If avoid data is provided, overlays avoid similarity/distance on the same plots.
	"""
	num_demos = len(similarities_list)
	has_avoid = avoid_similarities_list is not None

	num_cols = 3 if has_avoid else 2
	fig, axes = plt.subplots(num_demos, num_cols, figsize=(8*num_cols, 4*num_demos))
	if num_demos == 1:
		axes = axes.reshape(1, -1)

	for idx in range(num_demos):
		similarities = similarities_list[idx]
		distances = distances_list[idx]
		task_name = task_names[idx]
		text_prompt = text_prompts[idx]
		done_idx = done_indices_list[idx]
		timesteps = np.arange(len(similarities))

		avoid_sim = avoid_similarities_list[idx] if has_avoid else None
		avoid_dist = avoid_distances_list[idx] if has_avoid else None

		# ---- Column 1: Similarity (higher = more similar to prompt) ----
		ax_sim = axes[idx, 0]
		ax_sim.plot(timesteps, similarities.numpy(), linewidth=2, color='blue', alpha=0.8,
				   label=f'Goal: "{text_prompt}"')

		if has_avoid:
			ax_sim.plot(timesteps, avoid_sim.numpy(), linewidth=2, color='red', alpha=0.8,
					   label=f'Avoid: "{avoid_prompt}"')

		ax_sim.axvline(x=done_idx, color='green', linestyle='--', linewidth=2,
					   label=f'Task Done (t={done_idx})', alpha=0.7)
		ax_sim.scatter([done_idx], [similarities[done_idx]], color='green', s=150, zorder=5, marker='o')

		ax_sim.set_xlabel('Timestep', fontsize=12)
		ax_sim.set_ylabel('Similarity Score', fontsize=12)
		title = f'Task: {task_name} | Similarity\nGoal (want HIGH)'
		if has_avoid:
			title += ' | Avoid (want LOW)'
		ax_sim.set_title(title, fontsize=13, fontweight='bold')
		ax_sim.grid(True, alpha=0.3)
		ax_sim.legend(fontsize=8, loc='best')

		stats = f'Goal at done: {similarities[done_idx].item():.4f}'
		if has_avoid:
			stats += f'\nAvoid at done: {avoid_sim[done_idx].item():.4f}'
			stats += f'\nGap at done: {(similarities[done_idx] - avoid_sim[done_idx]).item():.4f}'
		ax_sim.text(0.02, 0.98, stats, transform=ax_sim.transAxes, fontsize=9,
					verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		# ---- Column 2: Distance (lower = closer to prompt) ----
		ax_dist = axes[idx, 1]
		ax_dist.plot(timesteps, distances.numpy(), linewidth=2, color='blue', alpha=0.8,
					label=f'Goal dist (want LOW)')

		if has_avoid:
			ax_dist.plot(timesteps, avoid_dist.numpy(), linewidth=2, color='red', alpha=0.8,
						label=f'Avoid dist (want HIGH)')

		ax_dist.axvline(x=done_idx, color='green', linestyle='--', linewidth=2,
						label=f'Task Done (t={done_idx})', alpha=0.7)
		ax_dist.scatter([done_idx], [distances[done_idx]], color='green', s=150, zorder=5, marker='o')

		ax_dist.set_xlabel('Timestep', fontsize=12)
		ax_dist.set_ylabel(f'{distance_metric.capitalize()} Distance', fontsize=12)
		title = f'Task: {task_name} | Distance\nGoal (want LOW)'
		if has_avoid:
			title += ' | Avoid (want HIGH)'
		ax_dist.set_title(title, fontsize=13, fontweight='bold')
		ax_dist.set_ylim(bottom=0)
		ax_dist.grid(True, alpha=0.3)
		ax_dist.legend(fontsize=8, loc='best')

		stats2 = f'Goal dist at done: {distances[done_idx].item():.4f}'
		if has_avoid:
			stats2 += f'\nAvoid dist at done: {avoid_dist[done_idx].item():.4f}'
		ax_dist.text(0.02, 0.98, stats2, transform=ax_dist.transAxes, fontsize=9,
					 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

		# ---- Column 3 (only if avoid): Negation score ----
		if has_avoid:
			ax_neg = axes[idx, 2]
			# Negation score: goal_similarity - avoid_similarity (higher = better)
			neg_score = similarities - avoid_sim
			ax_neg.plot(timesteps, neg_score.numpy(), linewidth=2, color='purple', alpha=0.8,
					   label='goal_sim - avoid_sim')
			ax_neg.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
			ax_neg.fill_between(timesteps, 0, neg_score.numpy(),
							   where=neg_score.numpy() > 0, alpha=0.15, color='green', label='Good')
			ax_neg.fill_between(timesteps, 0, neg_score.numpy(),
							   where=neg_score.numpy() <= 0, alpha=0.15, color='red', label='Bad')

			ax_neg.axvline(x=done_idx, color='green', linestyle='--', linewidth=2,
						   label=f'Task Done (t={done_idx})', alpha=0.7)

			best_t = neg_score.argmax().item()
			ax_neg.scatter([best_t], [neg_score[best_t]], color='green', s=150, zorder=5, marker='^',
						  edgecolors='black', linewidths=1)

			ax_neg.set_xlabel('Timestep', fontsize=12)
			ax_neg.set_ylabel('Negation Score', fontsize=12)
			ax_neg.set_title(f'Task: {task_name} | Negation Score\n(goal_sim - avoid_sim) | Higher = Better',
							fontsize=13, fontweight='bold')
			ax_neg.grid(True, alpha=0.3)
			ax_neg.legend(fontsize=8, loc='best')

			neg_stats = (f'At done: {neg_score[done_idx].item():.4f}\n'
						 f'Max: {neg_score.max().item():.4f} (t={best_t})\n'
						 f'Mean: {neg_score.mean().item():.4f}')
			ax_neg.text(0.02, 0.98, neg_stats, transform=ax_neg.transAxes, fontsize=9,
						verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved text similarity plot to {save_path}')


def plot_embedding_trajectories_2d(image_embeddings_list, text_embeddings_list, task_names,
								   text_prompts, save_path,
								   avoid_embeddings_list=None, avoid_prompt=None):
	"""
	Plot 2D trajectory of image embeddings using PCA projection, with text embedding shown.
	If avoid_embeddings_list is provided, also shows the avoid text embedding.
	"""
	from sklearn.decomposition import PCA

	num_demos = len(image_embeddings_list)
	has_avoid = avoid_embeddings_list is not None

	# Concatenate all embeddings for PCA
	all_embeddings_list = []
	for img_emb, txt_emb in zip(image_embeddings_list, text_embeddings_list):
		all_embeddings_list.append(img_emb)
		all_embeddings_list.append(txt_emb)
	if has_avoid:
		for avoid_emb in avoid_embeddings_list:
			all_embeddings_list.append(avoid_emb)

	all_embeddings = torch.cat(all_embeddings_list, dim=0).numpy()

	pca = PCA(n_components=2)
	all_2d = pca.fit_transform(all_embeddings)

	# Split back
	img_2d_list = []
	txt_2d_list = []
	start = 0
	for img_emb, txt_emb in zip(image_embeddings_list, text_embeddings_list):
		img_end = start + len(img_emb)
		img_2d_list.append(all_2d[start:img_end])
		txt_end = img_end + len(txt_emb)
		txt_2d_list.append(all_2d[img_end:txt_end])
		start = txt_end

	avoid_2d_list = []
	if has_avoid:
		for avoid_emb in avoid_embeddings_list:
			avoid_end = start + len(avoid_emb)
			avoid_2d_list.append(all_2d[start:avoid_end])
			start = avoid_end

	fig, axes = plt.subplots(1, num_demos, figsize=(7*num_demos, 6))
	if num_demos == 1:
		axes = [axes]

	for idx, (img_2d, txt_2d, task_name, text_prompt) in enumerate(
			zip(img_2d_list, txt_2d_list, task_names, text_prompts)):
		ax = axes[idx]

		ax.plot(img_2d[:, 0], img_2d[:, 1], 'o-',
				linewidth=2, markersize=4, alpha=0.6, color='steelblue', label='Trajectory')

		ax.scatter(img_2d[0, 0], img_2d[0, 1],
				  color='green', s=200, marker='o', label='Start', zorder=5,
				  edgecolors='black', linewidths=2)
		ax.scatter(img_2d[-1, 0], img_2d[-1, 1],
				  color='blue', s=200, marker='s', label='End', zorder=5,
				  edgecolors='black', linewidths=2)

		# Goal text embedding
		ax.scatter(txt_2d[0, 0], txt_2d[0, 1],
				  color='limegreen', s=300, marker='*', label=f'Goal: "{text_prompt}"',
				  zorder=5, edgecolors='black', linewidths=2)

		# Avoid text embedding
		if has_avoid and idx < len(avoid_2d_list):
			avoid_2d = avoid_2d_list[idx]
			ax.scatter(avoid_2d[0, 0], avoid_2d[0, 1],
					  color='red', s=300, marker='X', label=f'Avoid: "{avoid_prompt}"',
					  zorder=5, edgecolors='black', linewidths=2)

		ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
		ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
		title = f'Task: {task_name}\nTrajectory vs. Text Prompts'
		if has_avoid:
			title += ' (Goal + Avoid)'
		ax.set_title(title, fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=8, loc='best')

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'Saved embedding trajectory plot to {save_path}')


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""Generates demonstrations with text-to-image similarity analysis using CLIP."""
	assert torch.cuda.is_available()
	assert hasattr(cfg, 'num_demos') and isinstance(cfg.num_demos, int) and cfg.num_demos > 0, \
		'Please specificy number of demos to generate via +num_demos=<int> (must be a positive integer).'
	assert os.path.exists(CHECKPOINT_PATH), \
		f'Checkpoint path {CHECKPOINT_PATH} does not exist.'

	# Text prompt configuration via hydra config
	assert hasattr(cfg, 'text_prompt'), \
		'Please specify goal text prompt via +text_prompt="<your prompt>"'
	TEXT_PROMPT = cfg.text_prompt

	# Optional avoid prompt
	has_avoid = hasattr(cfg, 'avoid_prompt')
	AVOID_PROMPT = cfg.avoid_prompt if has_avoid else None
	if has_avoid:
		print(f'Avoid prompt: "{AVOID_PROMPT}"')

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

	# Initialize CLIP encoder
	print('Initializing CLIP encoder for text-image similarity analysis...')
	clip_encoder = CLIPEncoder()

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

	# Storage for text similarity analysis
	similarities_list = []
	distances_list = []
	image_embeddings_list = []
	text_embeddings_list = []
	accepted_task_names = []
	text_prompts_list = []
	done_indices_list = []
	# Avoid storage
	avoid_similarities_list = []
	avoid_distances_list = []
	avoid_embeddings_list = []

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations with text similarity analysis...')
	print(f'Using text prompt: "{TEXT_PROMPT}"')
	demos_collected = 0
	reward_threshold = -float('inf')
	distance_metric = 'cosine'

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
					# Add to buffer
					ep_td = tds[:, i].unsqueeze(0).clone()
					ep_frames = ep_td['frame'].squeeze(0)  # [T+1, C, H, W]
					frames.append(ep_frames)

					# Find the done index
					success_tensor = ep_td['success'].squeeze(0)
					done_idx = torch.where(success_tensor >= 0.99)[0]
					if len(done_idx) > 0:
						done_idx = done_idx[0].item()
					else:
						done_idx = len(success_tensor) - 1
						print(f'Warning: Task never succeeded (max success: {success_tensor.max().item():.2f})')

					# Compute text-to-image similarities using CLIP
					print(f'Computing text similarities for demo {demos_collected+1}...')
					print(f'Task completed at timestep {done_idx}/{len(ep_frames)-1}')
					similarities, distances, img_embeddings, txt_embedding = compute_text_similarities(
						ep_frames.to('cuda'), clip_encoder, TEXT_PROMPT, distance_metric=distance_metric)

					similarities_list.append(similarities)
					distances_list.append(distances)
					image_embeddings_list.append(img_embeddings)
					text_embeddings_list.append(txt_embedding)
					accepted_task_names.append(cfg.tasks[i])
					text_prompts_list.append(TEXT_PROMPT)
					done_indices_list.append(done_idx)

					# Compute avoid similarities if avoid prompt is provided
					if has_avoid:
						print(f'Computing avoid text similarities...')
						avoid_sim, avoid_dist, avoid_emb = compute_text_similarities_with_avoid(
							img_embeddings, clip_encoder, AVOID_PROMPT, distance_metric=distance_metric)
						avoid_similarities_list.append(avoid_sim)
						avoid_distances_list.append(avoid_dist)
						avoid_embeddings_list.append(avoid_emb)

						neg_score_at_done = (similarities[done_idx] - avoid_sim[done_idx]).item()
						print(f'  Goal sim at done: {similarities[done_idx].item():.4f}')
						print(f'  Avoid sim at done: {avoid_sim[done_idx].item():.4f}')
						print(f'  Negation score at done: {neg_score_at_done:.4f}')

					del ep_td['frame']
					demos_collected = buffer.add(ep_td)
					print(f'Added demo {demos_collected}/{cfg.num_demos} '
						  f'with reward {ep_reward[i]:.2f}, success {ep_success[i]:.2f}, and length {ep_len[i]} '
						  f'for task {cfg.tasks[i]}.')

					# Reset episode metrics
					ep_reward[i] = 0.0
					ep_len[i] = 0

				else:
					print(f'Rejected demo for task {cfg.tasks[i]} '
						  f'with reward {ep_reward[i]:.2f} and success {ep_success[i]:.2f}.')

			break

		else:
			ep_len += 1

	# Raise an error if not enough demos were collected
	if demos_collected < cfg.num_demos:
		print(f'[Demo collection failed] Only {demos_collected} demos collected, expected {cfg.num_demos}.')
		exit(0)

	# Create data directory
	os.makedirs(cfg.data_dir, exist_ok=True)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames, dim=0)

	# Plot text similarities
	print('\nGenerating text similarity plots...')
	plot_save_path = f'{cfg.data_dir}/{cfg.task}_text_similarity_{distance_metric}.png'
	plot_text_similarities(similarities_list, distances_list, accepted_task_names,
						   text_prompts_list, done_indices_list, plot_save_path,
						   distance_metric=distance_metric,
						   avoid_similarities_list=avoid_similarities_list if has_avoid else None,
						   avoid_distances_list=avoid_distances_list if has_avoid else None,
						   avoid_prompt=AVOID_PROMPT)

	# Plot embedding trajectories
	print('Generating embedding trajectory plots...')
	trajectory_save_path = f'{cfg.data_dir}/{cfg.task}_text_embedding_trajectories_{distance_metric}.png'
	plot_embedding_trajectories_2d(image_embeddings_list, text_embeddings_list,
								   accepted_task_names, text_prompts_list, trajectory_save_path,
								   avoid_embeddings_list=avoid_embeddings_list if has_avoid else None,
								   avoid_prompt=AVOID_PROMPT)

	# Save grid image for each demo
	print('\nSaving grid visualizations for each demo...')
	for demo_idx in range(frames.shape[0]):
		demo_frames = frames[demo_idx]
		demo_frames_normalized = demo_frames.float() / 255.0
		nrow = int(np.ceil(np.sqrt(demo_frames.shape[0])))
		grid = make_grid(demo_frames_normalized, nrow=nrow)
		save_image(grid, f'{cfg.data_dir}/{cfg.task}_demo_{demo_idx:03d}_grid.png')
		print(f'Saved grid for demo {demo_idx+1}/{demos_collected} with {demo_frames.shape[0]} frames')

	print(f'\nCompleted! Saved {demos_collected} demos to {cfg.data_dir}.')
	print(f'Goal prompt: "{TEXT_PROMPT}"')
	if has_avoid:
		print(f'Avoid prompt: "{AVOID_PROMPT}"')


if __name__ == '__main__':
	generate_demos()
