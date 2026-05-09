# ETL Evaluation on D3IL (Sorting & Stacking)

Evaluates Embedding Temporal Logic (ETL) and baselines on the D3IL sequential manipulation benchmark.

## Tasks

- **Sorting**: push two blocks into their color-matching target bins (2-phase)
- **Stacking**: arrange colored blocks in a target region (multi-phase)

## Setup

```bash
conda env create -f environment.yml
conda activate etl_d3il
```

## Data

Download rollout data from [D3IL](https://drive.google.com/drive/folders/1VuI3eQmFHT2QKCSYZGQ2pAuuhwNJkGCu) and place as:

```
etl_d3il/data/{task}/rollouts/
```

## Running experiments

```bash
# ETL vs baselines — comparison table
python scripts/run_etl_comparison.py task=sorting \
    methods='["etl","etl_temporal","etl_seq","similarity","logpzo"]'

python scripts/run_etl_comparison.py task=stacking \
    methods='["etl","etl_temporal","etl_seq","similarity","logpzo"]'

# Fast mode (loads only obs_embeddings)
python scripts/run_etl_fast.py --tasks stacking sorting --n_phases 3
```

## Methods

| Method | Description |
|--------|-------------|
| `etl` | Single-phase ETL: `F(near_goal)` |
| `etl_temporal` | K-phase sequential ETL: `F(near_0 ∧ F(near_1 ∧ ...))` |
| `etl_seq` | Sequential robustness via min-k distance |
| `similarity` | PCA-kmeans baseline (Liu et al., 2024) |
| `logpzo` | logpZO flow-matching density baseline (Xu et al., 2025) |
