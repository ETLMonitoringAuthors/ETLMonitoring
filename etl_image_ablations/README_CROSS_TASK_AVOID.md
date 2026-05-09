# Cross-task avoid / negation (e.g. green success latents while running blue)

## Idea

For **main** task `rd-push-blue`, treat **success-region latents** from **another** task `rd-push-green` as an **avoid manifold**: “don’t get latent-close to states that look like *successful green* when your job is blue.”

- Avoid latents are encoded with the **avoid task’s** task id (task-conditioned world model).
- Main rollout latents use the **main task** id.
- Per timestep:
  - `latent_goal_dist` = min L2 to main-task goal manifold (unchanged).
  - `latent_avoid_dist` = min L2 to the stacked avoid-task success-window latents.
  - **Negation score** = `latent_avoid_dist - latent_goal_dist` (higher ⇒ farther from green success relative to blue goal).

## Usage

From repo root (`newt/`), with EGL/MuJoCo as usual:

```bash
python etl_image_ablations/run_image_etl_ablations.py \
  task=rd-push-blue \
  +num_demos=5 \
  +goal_counts='[10]' \
  +avoid_source_task=rd-push-green \
  +results_subdir=blue_with_green_avoid
```

Outputs then go to `etl_image_ablations/results/rd-push-blue/blue_with_green_avoid/` so default `results/rd-push-blue/` plots are **not** overwritten.

### Optional Hydra knobs

| Override | Default | Meaning |
|----------|---------|--------|
| `+num_avoid_demos` | `num_demos` | How many successful green demos to build the avoid pool |
| `+avoid_pool_window` | `goal_pool_window` | ±timesteps around green success to add latents |
| `+avoid_max_points` | `0` (no cap) | Subsample avoid set for speed |
| `+avoid_seed` | `seed` | RNG for subsampling |
| `+results_subdir` | *(unset)* | Subfolder under `results/<task>/` — single name only (no `/`, `\\`, `..`) |

## Outputs

- `cross_task_negation_results.json` — per `(demo, k_goals)`: TVs, distances and negation at success time.
- `demo_XX_goal_vs_avoid_latent_kYY.png` — two panels: normalized goal vs avoid latent distances, and negation curve.

### Summary figure (avoid-specific)

`plot_cross_task_avoid_summary.py` reads `cross_task_negation_results.json` and writes **`cross_task_avoid_summary.png`** in the same run folder (negation vs k, goal/avoid at success, TV smoothness, per-demo bars).

```bash
python etl_image_ablations/plot_cross_task_avoid_summary.py \
  --task rd-push-blue --subdir blue_with_green_avoid
```

`summarize_image_ablations.py` is for generic goal/threshold ablations under `results/<task>/`; it does **not** merge avoid metrics. Use the script above for blue-with-green (or any `+results_subdir` avoid run).

### Is CP thresholding “good” for the cross-task run?

**What is actually thresholded:** `dynamic_threshold_results.json` still uses **min distance to the main-task goal manifold** (embedding / latent / state). The green **avoid** set and **negation** curves are diagnostics only unless you extend the code to mine τ on `avoid_dist` or `negation` with their own labels.

**How to read it:**

| Plot / check | What it tells you |
|--------------|-------------------|
| **`plot_cross_task_cp_thresholds.py`** → `cross_task_cp_f1_comparison.png` | Mean ± SEM over demos of **F1 / precision / recall** on the **test suffix** for **median (cal∩high-R)** vs **split CP** vs **legacy (all timesteps)**. If CP ≥ median and legacy is lower, CP is doing its job (finite-sample + less leakage than legacy). |
| Same script → `cross_task_cp_threshold_scales.png` | **τ** per space and method. CP often moves τ **up** vs median (more conservative one-sided bound on positives). |
| **`demo_*_goal_vs_avoid_latent_*.png`** | Overlay **mental check**: at timesteps classified “close to goal” by τ, is the trajectory also where you want on **avoid**? That’s **not** guaranteed by goal-only CP. |
| **Future:** CP on **negation** | Define a binary label (e.g. “negation above margin at success”) and run the same cal/test split on the **negation time series** — then you’d visualize that JSON the same way. |

```bash
python etl_image_ablations/plot_cross_task_cp_thresholds.py \
  --task rd-push-blue --subdir blue_with_green_avoid
```

### Reusing the **latent success** threshold τ on **avoid** distance

You can ask: *when the rollout is genuinely near the avoid manifold (small `lat_avoid`), would flagging “close to avoid” with the **same numeric** τ as for goal distance (`lat_avoid < τ`) behave sensibly?* And *when `lat_goal < τ`, how often is `lat_avoid` also below τ?*

1. **Re-run** the avoid ablation once so per-timestep arrays are saved:  
   `demo_XX_lat_goal_avoid_kYY.npz` (written by default; disable with `+save_avoid_distance_trajectories=False`).

2. Run:

```bash
python etl_image_ablations/analyze_tau_transfer_goal_to_avoid.py \
  --task rd-push-blue --subdir blue_with_green_avoid --k 10 --tau-source ccp
```

Outputs in the same folder:

- **`tau_transfer_goal_to_avoid_k{K}_{ccp|median}.json`** — per demo: recall of `(lat_avoid < τ)` on timesteps in the **per-trajectory** bottom `per_traj_q` of `lat_avoid` (local “closest to avoid” moments); `frac_both_goal_lt_tau_and_avoid_lt_tau` (overlap of the two half-spaces).
- **`tau_transfer_goal_to_avoid_k{K}_....png`** — scatter `lat_goal` vs `lat_avoid` with **mean τ** as both a vertical and horizontal line (visual for “same cutoff on both axes”); bar chart of per-demo rates.

Optional: `--global-q 0.05` adds a **pooled** quantile definition of “globally very low `lat_avoid`” and reports an extra recall line in JSON.

## Caveats

- Uses **two checkpoints** (main + avoid); both must exist under `CHECKPOINT_PATH`.
- Cross-task latent distances are only as meaningful as your shared encoder + task embeddings allow (OOD behavior may stretch geometry).
