# ETL across modeling paradigms

## Conda environment

For **NewT** you likely already use **`conda activate newt`**. That same environment is a
reasonable choice for running **`AnySafeReachability/dino_wm/etl/run_etl_dubins.py`**
as long as `torch` (CUDA), `matplotlib`, `pillow`, and `h5py` are installed. AnySafe’s
own docs recommend **`conda activate anysafe`** from their `environment.yaml`; use
whichever env is already working on your machine.

---

This doc ties together **three** latent / world-model setups for the same style of
**embedding–trajectory–learning (ETL)** analyses (goal manifolds, smoothness, conformal
thresholds on distance-to-goal).

| Paradigm | Repo / path | Latent for ETL | Training notes |
|----------|-------------|----------------|----------------|
| **1. NewT (TD-MPC2)** | `newt/` — `etl_image_ablations/run_image_etl_ablations.py` | `agent.model.encode(obs, task)` (world model) + DINOv2 patches for embedding space | Default checkpoints under `checkpoints/models--nicklashansen--newt/...` |
| **2. AnySafe DinoWM + projector** | `AnySafeReachability/dino_wm/etl/` — `python -m etl.run_etl_dubins` | **512-D semantic** from `VideoTransformer.semantic_encoder` (sliding window over Dubins frames); plus DINO mean-patch and state | Load `--wm-ckpt` from `train_dino_wm.py` (+ optional semantic training). See `dino_wm/etl/README.md`. |
| **3. DinoWM + temporal straightening** | Same AnySafe `dino_wm` training with `DINO_WM_STRAIGHTEN=cos<scale>` | Same encoder stack as row 2 after training | Curvature regularizer lives in `AnySafeReachability/dino_wm/straightening.py` (adapted from `temporal-straightening`). |

**temporal-straightening** (`/path/to/temporal-straightening`) remains the **reference**
implementation of the straightening objective on `VWorldModel`; the AnySafe port keeps
the **same math** on DinoWM transformer latents so you can ablate **with vs without**
straightening without leaving the Franka/Dubins WM codebase.

## Suggested evaluation workflow

1. **NewT**: run existing Hydra / `run_image_etl_ablations.py` on `rd-push-*` (or your task).
2. **AnySafe (no straightening)**: train or reuse `best_testing.pth`, run `run_etl_dubins`.
3. **AnySafe + straightening**: retrain with `DINO_WM_STRAIGHTEN=cos0.01` (tune scale), same data, then rerun `run_etl_dubins` with the new checkpoint.
4. Compare `dynamic_threshold_results.json` and `smoothness_results.json` **per paradigm**
   (metrics are comparable **within** a paradigm; absolute thresholds differ across stacks).

## Dubins / AnySafe analogue (multi-goal + GT + avoid panel)

On **`AnySafeReachability`** (dubins branch), **`dino_wm/etl/run_dubins_spec_analysis.py`**
mirrors much of this script’s structure: **k** image-derived goal latents (512-D semantic),
**privileged** min-L2 in **(x,y,cosθ,sinθ)**, optional **failure-pool** latent distance +
**obstacle signed margin**, optional **sequential** goal split, and a **single figure** with
distance curves + **image strip** (goal thumbnails + key frames). See **`dino_wm/etl/README.md`**.

## Caveats

- Dubins ETL in AnySafe uses a **synthetic renderer** + optional **Franka-pretrained**
  weights; numbers are only interpretable after **domain-matched** training.
- Class-conditional CP logic is **vendored** in `dino_wm/etl/conformal_threshold.py`;
  keep in sync with `newt/etl_image_ablations/conformal_threshold.py` if the algorithm changes.
