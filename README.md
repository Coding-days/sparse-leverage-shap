[README.md](https://github.com/user-attachments/files/26805697/README.md)
# Sparse Leverage SHAP — Experiments

Code accompanying the paper on **Sparse Leverage SHAP**, a Shapley-value
estimator that exploits sparsity in the underlying feature attributions to
achieve an `O(s · log n)` sample complexity, compared to `O(n · log n)` for
the dense Leverage SHAP estimator of Musco & Witter (ICLR 2025, *Provably
Accurate Shapley Value Estimation via Leverage Score Sampling*).

This repository contains two experiments. They test **different** claims and
should be read as complementary rather than overlapping.

---

## The three experiments at a glance

| File | What it tests | Data | Ground truth | Primary metric |
|---|---|---|---|---|
| `experiment_real_data_benchmark.py` | End-to-end accuracy of four estimators across a wide range of sample budgets | 8 real datasets via `shap.datasets` + XGBoost | Tree SHAP | Relative ℓ₂ error |
| `experiment_real_sparse_data.py` | Does the sparse sample-complexity advantage appear on real data when the Shapley vector is empirically sparse? | Selected real datasets filtered / measured for sparse Shapley vectors | Tree SHAP | Relative ℓ₂ error, with measured `s_eff`/`n` reported per dataset |
| `experiment_support_recovery.py` | Sample-complexity gap `O(s·log n)` vs `O(n·log n)` on a controlled sparse game | Synthetic sigmoid game with a known active set `R` | Closed-form Shapley via active-player reduction | Relative ℓ₂ error **and** exact support recovery rate |

The three experiments answer three different questions:

1. *Does Sparse Leverage SHAP match general-purpose methods on real data?* — benchmark
2. *When real data is sparse, does the sample-complexity advantage show up?* — real-sparse
3. *Does the method actually recover the true support in a controlled setting?* — synthetic

---

## `experiment_real_data_benchmark.py`

Replicates the plot style of Figure 1 and Figure 3 from Musco & Witter
(ICLR 2025) and adds Sparse Leverage SHAP as a fourth baseline.

**Methods compared**

1. Kernel SHAP (unoptimized baseline, with replacement, no paired sampling)
2. Optimized Kernel SHAP (`shap.KernelExplainer`)
3. Leverage SHAP (uniform size sampling, paired, weighted OLS)
4. Sparse Leverage SHAP (size probability `∝ s`, paired, weighted LASSO)

**Datasets (all from `shap.datasets`)**

IRIS, California, Diabetes, Adult, Correlated, Independent, NHANES,
Communities.

**Sample-budget sweep**

By default the budget `m` ranges over multiples of the feature count `n`:

```
0.2n, 0.5n, n, 2n, 5n, 10n, 20n, 40n, 80n, 160n
```

This spans both regimes:

- **Data-poor** (`m < n`): OLS is underdetermined; standard methods fail;
  sparse methods can still recover an `s`-sparse Shapley vector.
- **Data-rich** (`m >> n`): all methods converge; the sparse advantage
  shrinks.

The vertical dash-dot line on the Figure-3-style plot marks `m = n`, the
boundary between the two regimes.

**Outputs** (written to `--output-dir`, default `paper_style_plots/`)

- `figure1_predicted_vs_true.png` — scatter of predicted vs. true Shapley
  values at `m = 5n`, grid of datasets × methods
- `figure3_error_vs_samples.png` — log–log error vs. sample size, median
  line with Q1–Q3 bands, per dataset
- `all_curve_stats.csv` — per-dataset per-method per-budget statistics
- `table_m10n.csv` — summary at `m = 10n` (the paper's headline budget)

**Run**

```bash
python experiment_real_data_benchmark.py                # defaults
python experiment_real_data_benchmark.py \
    --curve-trials 100 --scatter-trials 25 \
    --output-dir paper_style_plots
```

Key flags: `--curve-trials`, `--scatter-trials`, `--curve-mults`,
`--n-bg`, `--seed`, `--output-dir`.

---

## `experiment_real_sparse_data.py`

Bridges the synthetic and real-data stories: tests whether the
`O(s·log n)` sample-complexity advantage of Sparse Leverage SHAP shows up
on real datasets where the Shapley vector is empirically (approximately)
sparse.

**Pipeline per dataset**

1. Fit XGBoost.
2. **Measure** effective sparsity `s_eff = #{i : |φᵢ| > 0.05 · ||φ||∞}`
   averaged over random test points, using Tree SHAP as ground truth.
   Also measure `tail_mass = ||φ − top_{s_eff}(φ)||² / ||φ||²` — the
   fraction of ℓ₂² energy *outside* the top coordinates. Low tail mass
   means the vector is genuinely close to `s_eff`-sparse.
3. Run the four estimators across a sample-budget sweep.
4. Plot log–log error-vs-`m` curves with Q1–Q3 bands. Each panel title
   reports the measured `s_eff`, `s/n`, and tail mass so the sparsity
   claim is visible alongside the curves. Vertical reference lines mark
   the theoretical thresholds `s·log n` (sparse, green) and `n·log n`
   (dense, red).

**Datasets (defaults)**

Configured with three candidates from `shap.datasets` — but the script
**measures** sparsity rather than assuming it, so if a dataset comes out
dense you can see that in the output and swap it out (edit
`get_dataset_specs()`):

- `independentlinear60` — `n=60`, empirically `s/n ≈ 0.16`, tail mass
  ≈ 0.001 (strongly sparse)
- `corrgroups60` — `n=60`, empirically `s/n ≈ 0.43` (moderate; signal is
  concentrated but spread across correlated groups)
- `communitiesandcrime` — larger `n`, sparsity depends on model fit

If you have access to higher-dimensional sparse-by-construction data (text
bag-of-words, genomics), adding it here is one line in
`get_dataset_specs()`.

**Outputs** (written to `--output-dir`, default `real_sparse_plots/`)

- `error_vs_samples_sparse.png` — log-log error curves, one panel per
  dataset
- `sparsity_measurements.csv` — `s_eff`, `s/n`, tail mass per dataset
- `error_stats.csv` — per-method per-`m` error statistics

**Run**

```bash
python experiment_real_sparse_data.py                       # defaults
python experiment_real_sparse_data.py --trials 100 \
    --curve-mults 0.2 0.5 1 2 5 10 20 40
```

Key flags: `--trials`, `--curve-mults`, `--threshold-frac` (default 0.05),
`--n-bg`, `--seed`, `--output-dir`.

---

## `experiment_support_recovery.py`

Synthetic, controlled-sparsity experiment that directly tests the
sample-complexity claim of the paper.

**Game.** Sigmoid sparse game `v(S) = σ(Σ_{i ∈ S ∩ R} aᵢ) − σ(0)` where
`R` is a randomly chosen active set of size `s`; all features outside `R`
are exact dummies. The non-linearity ensures a non-zero regression residual
so the comparison is not trivial.

**Ground truth.** Active-player reduction: by the Null Player axiom, the
Shapley value of any `i ∉ R` is exactly zero, and for `i ∈ R` the value is
computed from the `s`-player Shapley formula. This requires only `2^s` oracle
calls per active player regardless of `n` — that is why `n = 30` is feasible.

**Two settings**

| | `n` | `s` | `s/n` | Sparse theory `s·log n` | Dense theory `n·log n` |
|---|---|---|---|---|---|
| A — very sparse | 30 | 3 | 10 % | ≈ 10 | ≈ 102 |
| B — less sparse | 30 | 12 | 40 % | ≈ 41 | ≈ 102 |

**Methods compared**

1. Kernel SHAP (size probability `∝ 1/[k(n−k)]`)
2. Dense Leverage SHAP (size probability uniform)
3. Sparse Leverage SHAP (size probability `∝ k`)

All three use the same importance-sampling-corrected estimator; they differ
only in which subsets are sampled. Dense and Sparse Leverage end up with the
same effective regression weight `1/[k(n−k)]` — all the behavioral
difference comes from the sampling distribution, not from the weighting.

**Metrics**

- Normalized ℓ₂ error `‖φ̂ − φ‖² / ‖φ‖²`
- **Exact support recovery rate**: the fraction of trials where
  `{i : |φ̂ᵢ| > ε}` exactly equals the true active set `R`, with
  `ε = β_min / 3` (β_min = smallest magnitude among true active Shapley
  values). This is the experiment's distinctive contribution — it tests
  whether the method identifies the right features, not just whether its
  numbers are close.

**Why `n = 30`?** At `n = 50`, `C(50,25) ≈ 10^14` destabilizes the
importance-sampling weights; at `n = 30`, `C(30,15) ≈ 1.55·10^8` and the
max/min effective weight ratio is ≈ 8, which the ridge-regularized solver
handles cleanly. Meanwhile `2^30 ≈ 10^9` keeps us firmly in the
`m ≪ 2^n` regime the theory is about.

**Sampling.** Without-replacement paired draws: every sampled subset `S`
is paired with its complement `[n] \ S`, and repeats are rejected to
prevent rank deficiency at small `m`.

**Outputs**

- `outputs/sparse_shapley_comparison.png` — 2×2 grid: top row = ℓ₂ error
  (log scale), bottom row = support recovery %; left column = Setting A,
  right column = Setting B. Vertical reference lines mark the theoretical
  `O(s·log n)` and `O(n·log n)` thresholds.
- Console tables of median error and recovery rate per `m`.

**Run**

```bash
mkdir -p outputs
python experiment_support_recovery.py
```

Knobs are set in `__main__`: `N = 30`, `S_A = 3`, `S_B = 12`,
`SAMPLE_SIZES = [N*k for k in [1,2,3,4,6,8,10]]`, 300 trials per setting.

---

## How the three experiments relate

Together the experiments argue the case in three steps:

- The **synthetic support-recovery** experiment isolates the mechanism:
  when the Shapley vector is exactly `s`-sparse, Sparse Leverage SHAP
  crosses into the "recovered" regime at roughly `m ≈ s · log n`, well
  before dense methods reach `m ≈ n · log n`. When sparsity is weak, the
  gap narrows as predicted.

- The **real-sparse** experiment checks that the mechanism carries over
  when sparsity is approximate rather than exact. On datasets with low
  measured `s/n` and low tail mass, the sparse advantage should show up
  as lower error at small `m`, between the two theoretical thresholds.

- The **real-data benchmark** confirms the method is competitive on real
  problems generally, not just the sparse subset — the theoretical
  advantage does not come at the cost of accuracy when sparsity is absent.

Read together: *the theoretical gain is real, it shows up on both
controlled and real-world sparse problems, and it does not cost anything
on real data in general.*

---

## Installation

```bash
pip install numpy scipy pandas matplotlib scikit-learn shap xgboost
```

Python ≥ 3.9.

---

## Repository layout

```
.
├── experiment_real_data_benchmark.py       # real-data, all 8 datasets, Fig 1 + Fig 3 style
├── experiment_real_sparse_data.py          # real-data, sparse-subset, sample-complexity focus
├── experiment_support_recovery.py          # synthetic, controlled sparsity + support recovery
├── paper_style_plots/                      # generated by experiment 1
│   ├── figure1_predicted_vs_true.png
│   ├── figure3_error_vs_samples.png
│   ├── all_curve_stats.csv
│   └── table_m10n.csv
├── real_sparse_plots/                      # generated by experiment 2
│   ├── error_vs_samples_sparse.png
│   ├── sparsity_measurements.csv
│   └── error_stats.csv
├── outputs/                                # generated by experiment 3
│   └── sparse_shapley_comparison.png
└── README.md
```

---

## Citation

If you use this code, please cite our paper alongside the dense
leverage-score baseline:

> Musco, C. and Witter, R.T. *Provably Accurate Shapley Value Estimation
> via Leverage Score Sampling.* ICLR 2025.
