# ETL Failure Monitoring on the FIPER Benchmark

This fork of [utiasDSL/fiper](https://github.com/utiasDSL/fiper) adds **ETL
(Embedding Temporal Logic)** as a failure prediction method and compares it
against the FIPER baselines (PCA-kmeans, logpZO, STAC, RND-OE, ACE).

## The Core Argument

FIPER baselines answer: *"Is anything going wrong?"* — a **binary alarm** with
no semantic content. They cannot tell you *what* went wrong or *how far* the
robot got.

ETL answers: *"Did phase 1 complete? Did phase 2 complete? How close is the
robot to each sub-goal right now?"* — a **temporally structured, interpretable
failure signal**.

| Property | ETL | PCA-kmeans | logpZO | STAC | RND-OE/FIPER |
|---|:---:|:---:|:---:|:---:|:---:|
| Semantically interpretable output | ✓ | ✗ | ✗ | ✗ | ✗ |
| Specifies what success looks like | ✓ | ✗ | ✗ | ✗ | ✗ |
| Phase-level progress evaluation | ✓ | ✗ | ✗ | ✗ | ✗ |
| Sequential temporal logic formula | ✓ | ✗ | ✗ | ✗ | ✗ |
| No policy access required | ✓ | ✓ | ✓ | ✗ | ✗ |
| No failure data required | ✓ | ✓ | ✓ | ✓ | ✓ |
| CP-calibrated threshold | ✓ | ✗ | ✗ | ✗ | ✓ |
| Spec from reference demo/image | ✓ | ✗ | ✗ | ✗ | ✗ |

## Three ETL Variants

### 1. `etl` — Single-Phase WHERE Spec

Mines one goal spec latent `z* = mean(last k frames of success rollouts)`.
Score = `cosine_dist(z_t, z*)`.

ETL formula: `F(near_goal)` where `near_goal(t) ≡ d(z_t, z*) < τ`

Already beats PCA-kmeans and logpZO in *interpretability*: the single threshold
`τ` calibrated by conformal prediction gives the user a **coverage guarantee**
("at most δ false alarms on success rollouts"), unlike PCA-kmeans.

### 2. `etl_temporal` — K-Phase Sequential Temporal Spec (Key Contribution)

Divides each calibration success rollout into K equal temporal segments.
Averages embeddings per segment across rollouts → K spec latents
`z_0, ..., z_{K-1}`.

At test time: `phase(t) = floor(t / T_mean × K)` selects the expected spec
latent.  Score = `cosine_dist(z_t, z_{phase(t)})`.

ETL formula: `F(near_0 ∧ F(near_1 ∧ … ∧ F(near_{K-1})))`

**Why this beats all FIPER baselines on sequential tasks (SORTING, STACKING,
PRETZEL, PUSH-CHAIR):**

- PCA-kmeans / logpZO / RND-OE score = distance to nearest success cluster
  (temporally **unordered**) → cannot detect "stuck in phase 1."
- ETL-temporal score rises when the robot is at the **wrong part** of the
  success manifold for its current time-slot, giving an **earlier and more
  precise** failure signal.

Example: on SORTING (push block A → box A, then block B → box B):
- PCA-kmeans fires when the robot is globally OOD — could be late.
- ETL-temporal fires as soon as the robot is still near `z_0` (block A phase)
  when it should already be near `z_1` (block B phase).

### 3. `etl_seq` — Sequential Robustness (No Temporal Tracking)

Score = `min_k cosine_dist(z_t, z_k)`.  Fires when the robot is off the
success manifold **globally** (not near any phase's spec latent).

Bridges ETL and PCA-kmeans: uses interpretable ETL spec latents but doesn't
require knowing the current phase.  Compared to PCA-kmeans, the K spec latents
are temporally ordered and interpretable rather than arbitrary cluster centers.

## Getting Started

### Installation

Follow the original FIPER setup:
```bash
conda env create -f environment.yml
conda activate fiper
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Download FIPER Rollout Data

Follow the [original FIPER README](README.md) to download rollouts and place
them in `data/{task}/rollouts/`.

### Running ETL Methods

Edit `configs/default.yaml` to select methods:
```yaml
methods: ["etl", "etl_temporal", "etl_seq", "similarity", "logpzo"]
```

Then run:
```bash
python scripts/run_fiper.py
```

### Running Only ETL vs Key Baselines

```bash
# Edit default.yaml: methods: ["etl", "etl_temporal", "similarity", "logpzo"]
# tasks: ["sorting", "stacking"]  # sequential tasks where ETL has the biggest advantage
python scripts/run_fiper.py
```

## ETL Method Hyperparameters

| Config | Key param | Default | Description |
|---|---|---|---|
| `etl.yaml` | `last_k_frames` | 10 | Frames at end of success rollout used to mine goal latent |
| `etl_temporal.yaml` | `n_phases` | 3 | Number of temporal spec latents |
| `etl_seq.yaml` | `n_phases` | 5 | Number of spec latents for min-distance |

## Expected Results

On **SORTING** and **STACKING** (tasks with clear sequential phases):
- `etl_temporal` should show **lower detection time (DT)** than PCA-kmeans and
  logpZO because temporal grounding allows earlier detection of phase-level failures.
- `etl` (single-phase) should be competitive with logpZO since both use a goal
  embedding, but ETL's CP threshold gives formal coverage guarantees.
- `etl_seq` should dominate PCA-kmeans because it uses semantically meaningful
  spec latents (phase means) rather than arbitrary k-means clusters.

On **PUSHT** (single goal state):
- `etl` should match or exceed logpZO — both use goal embeddings, ETL does not
  require training a flow-matching model.

## Files Added

```
evaluation/method_eval_classes/etl_eval.py   # ETLEval, ETLTEMPORALEval, ETLSEQEval
configs/eval/etl.yaml                        # Single-phase ETL config
configs/eval/etl_temporal.yaml               # K-phase temporal ETL config
configs/eval/etl_seq.yaml                    # Sequential robustness ETL config
ETL_README.md                                # This file
```

## Citation

If you use this code, please cite both:

```bibtex
@inproceedings{romer2025fiper,
  title={Failure Prediction at Runtime for Generative Robot Policies},
  author={Ralf R{\"o}mer and Adrian Kobras and Luca Worbis and Angela P. Schoellig},
  booktitle={NeurIPS},
  year={2025}
}

@article{TODO_etl_paper,
  title={ETL: Embedding Temporal Logic for Robot Monitoring},
  author={...},
  year={2025}
}
```
