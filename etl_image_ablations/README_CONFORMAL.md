# Class-conditional conformal threshold mining

## Relation to AnySafeReachability / `sweeper_cp.py`

The Franka script mines a **percentile** of pairwise scores (cosine similarity filtered by state-space radius) to get a threshold at target miscoverage `alpha`. That is **distribution-free** threshold mining on empirical scores.

Here we use a **split, class-conditional conformal** rule that is in the same spirit (quantile / order statistic on calibration scores) but tailored to **single-rollout latent (or embedding/state) distances**:

- **Class definition** (same as before): “positive” timesteps = simulator success **or** reward above the 90% quantile (fallback to above-mean reward if empty).
- **Calibration / test split**: first `cal_frac` fraction of the trajectory is **calibration**; the remainder is **test**. Thresholds are fit on calibration only; **F1 / precision / recall** are on the test suffix (no leakage).
- **Class-conditional**: the conformal threshold uses **only** calibration timesteps in the **positive** class. Nonconformity score = **distance to goal manifold** (smaller ⇔ more consistent with success).
- **Finite-sample CP quantile**: with `n` calibration positive scores, sort distances ascending and take index  
  `k = ceil((n + 1) * (1 - alpha))` (clamped to `[1, n]`). Threshold = `d_(k)`. This matches standard split conformal one-sided calibration.

## Hydra overrides

- `+cp_alpha=0.1` — target miscoverage (1 − α is the quantile level in the usual finite-sample construction).
- `+cal_frac=0.4` — fraction of timesteps used for calibration.
- `+threshold_reward_quantile=0.9` — reward quantile for the positive class mask.

## Outputs

`dynamic_threshold_results.json` contains per `(demo, k_goals, space)`:

| Field | Meaning |
|--------|--------|
| `median_cal_test` | Median distance on **cal ∩ positive**; metrics on **test** |
| `ccp_class_conditional` | CP threshold on **cal positives**; metrics on **test** |
| `legacy_median_all_timesteps` | Old pooled median over all steps (inflated F1; kept for comparison) |

## Code

- `conformal_threshold.py` — core CP + median (cal/test) helpers.
- `run_image_etl_ablations.py` — rolls trajectories and writes JSON.
- `summarize_image_ablations.py` — `threshold_f1_by_space.png` now overlays **median (cal→test)** vs **CP (class-cond.)** per space.
