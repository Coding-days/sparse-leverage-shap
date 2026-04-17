#!/usr/bin/env python3
"""
experiment_real_sparse_data.py
═════════════════════════════════════════════════════════════════════════════
Real-data experiment targeting sparse Shapley vectors.

Tests whether the O(s log n) sample-complexity advantage of Sparse Leverage
SHAP shows up on real problems where the Shapley vector is empirically
(approximately) sparse — i.e., where the model's prediction at a given test
point depends strongly on only a few features.

Pipeline per dataset:
  1. Fit XGBoost on the data.
  2. Measure effective sparsity s_eff = #{i : |phi_i| > 0.05 * ||phi||_inf},
     averaged over test points, using Tree SHAP as ground truth.
  3. Run the four estimators (Kernel SHAP, Optimized Kernel SHAP,
     Leverage SHAP, Sparse Leverage SHAP) across a sample-budget sweep.
  4. Plot log-log error-vs-m with Q1-Q3 bands. Each panel reports the
     measured s_eff and s/n ratio in its title so the sparsity claim
     is visible alongside the error curves.

Companion experiments:
  - experiment_real_data_benchmark.py  (general real-data accuracy, 8 datasets)
  - experiment_support_recovery.py     (controlled synthetic sparsity)

This file is the bridge between them: it asks whether the synthetic story
carries over to real data when real data happens to be sparse.

DATASETS
--------
Configured by default with three datasets from `shap.datasets` that are
expected to yield approximately-sparse Shapley vectors on XGBoost. The list
is easy to edit — see DATASETS below. The script MEASURES sparsity rather
than assuming it, so if a dataset turns out dense you will see that in the
output and can swap it out.

In a sandbox test with limited network access, measured s/n ratios on the
datasets we could load were:
    Independent (n=60):  s/n ≈ 0.16   ← strongly sparse, tail mass ≈ 0.001
    Correlated  (n=60):  s/n ≈ 0.43   ← moderate; signal is concentrated
                                        but spread across correlated groups
If `Communities` is not sparse on your machine, swap in another dataset.

OUTPUTS
-------
  real_sparse_plots/error_vs_samples_sparse.png   # log-log error curves
  real_sparse_plots/sparsity_measurements.csv     # s_eff per dataset
  real_sparse_plots/error_stats.csv               # per-method per-m stats
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")


# =============================================================================
#  Shared utilities (mirrors experiment_real_data_benchmark.py)
# =============================================================================

def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rel_l2(phi_hat: np.ndarray, phi_true: np.ndarray) -> float:
    denom = np.linalg.norm(phi_true) ** 2
    if denom < 1e-30:
        return float("nan")
    return np.linalg.norm(phi_hat - phi_true) ** 2 / denom


def ensure_even(m: int) -> int:
    return int(m) if int(m) % 2 == 0 else int(m) + 1


def _extract_single_output_shap(phi_obj, n_features: int, class_index: int = -1) -> np.ndarray:
    values = getattr(phi_obj, "values", phi_obj)
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape[-1] == n_features:
            return arr[0].astype(float)
        if arr.shape[0] == n_features and arr.shape[1] == 1:
            return arr[:, 0].astype(float)
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            return arr[0, :, class_index].astype(float)
        if arr.shape[0] == 1 and arr.shape[2] == n_features:
            return arr[0, class_index, :].astype(float)
        if arr.shape[1] == 1 and arr.shape[2] == n_features:
            return arr[class_index, 0, :].astype(float)
    raise ValueError(f"Cannot coerce SHAP array of shape {arr.shape} to length {n_features}")


def make_shap_value_fn(predict, x_test, bg: np.ndarray):
    """Interventional SHAP value function: v(z) = E_bg[ f(z*x + (1-z)*bg) ]."""
    x_test = np.asarray(x_test).ravel()
    bg = np.asarray(bg)
    B = len(bg)

    def v(z):
        z = np.atleast_2d(z).astype(float)
        batch = z.shape[0]
        z3 = z[:, None, :]
        masked = z3 * x_test + (1.0 - z3) * bg[None, :, :]
        out = predict(masked.reshape(batch * B, -1))
        return np.asarray(out).reshape(batch, B).mean(axis=1)

    return v


# =============================================================================
#  Estimators (identical to experiment_real_data_benchmark.py)
# =============================================================================

def _solve_weighted_projected_ols(rows, szs, vals, v0, v1, n, log_ws):
    ws = np.exp(log_ws - np.max(log_ws))
    ZP = rows - szs[:, None] / n
    b = vals - v0 - (szs / n) * (v1 - v0)
    sqW = np.sqrt(ws)
    phi, *_ = np.linalg.lstsq(ZP * sqW[:, None], b * sqW, rcond=None)
    return phi + (v1 - v0) / n


def _solve_weighted_projected_lasso(rows, szs, vals, v0, v1, n, log_ws, alpha_scale=0.005):
    ws = np.exp(log_ws - np.max(log_ws))
    ZP = rows - szs[:, None] / n
    b = vals - v0 - (szs / n) * (v1 - v0)
    sqW = np.sqrt(ws)
    Aw = ZP * sqW[:, None]
    bw = b * sqW
    alpha = alpha_scale * np.std(bw) / max(np.sqrt(len(bw)), 1.0)
    alpha = max(alpha, 1e-10)
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-6, fit_intercept=False)
    lasso.fit(Aw, bw)
    return lasso.coef_ + (v1 - v0) / n


def kernel_shap_baseline(n, v, m, seed=0):
    rng = np.random.RandomState(seed)
    m = max(2, int(m))
    sizes = np.arange(1, n)
    size_probs = 1.0 / (sizes * (n - sizes))
    size_probs = size_probs / size_probs.sum()
    rows = np.zeros((m, n)); szs = np.zeros(m)
    for i in range(m):
        s = int(rng.choice(sizes, p=size_probs))
        idx = rng.choice(n, size=s, replace=False)
        rows[i, idx] = 1.0; szs[i] = s
    v0 = v(np.zeros((1, n)))[0]; v1 = v(np.ones((1, n)))[0]
    vals = v(rows)
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(s * (n - s), 1)) for s in szs])
    return _solve_weighted_projected_ols(rows, szs, vals, v0, v1, n, log_ws)


def leverage_shap(n, v, m, seed=0):
    rng = np.random.RandomState(seed)
    m = ensure_even(m)
    rows, szs = [], []
    for _ in range(max(1, m // 2)):
        s = rng.randint(1, n)
        idx = rng.choice(n, size=s, replace=False)
        z = np.zeros(n); z[idx] = 1.0
        rows.extend([z, 1.0 - z]); szs.extend([s, n - s])
    rows = np.asarray(rows)[:m]; szs = np.asarray(szs, dtype=float)[:m]
    v0 = v(np.zeros((1, n)))[0]; v1 = v(np.ones((1, n)))[0]
    vals = v(rows)
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(s * (n - s), 1)) for s in szs])
    return _solve_weighted_projected_ols(rows, szs, vals, v0, v1, n, log_ws)


def sparse_leverage_shap(n, v, m, seed=0, alpha_scale=0.005):
    rng = np.random.RandomState(seed)
    m = ensure_even(m)
    sizes = np.arange(1, n)
    probs = sizes.astype(float); probs = probs / probs.sum()
    rows, szs = [], []
    for _ in range(max(1, m // 2)):
        s = int(rng.choice(sizes, p=probs))
        idx = rng.choice(n, size=s, replace=False)
        z = np.zeros(n); z[idx] = 1.0
        rows.extend([z, 1.0 - z]); szs.extend([s, n - s])
    rows = np.asarray(rows)[:m]; szs = np.asarray(szs, dtype=float)[:m]
    v0 = v(np.zeros((1, n)))[0]; v1 = v(np.ones((1, n)))[0]
    vals = v(rows)
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(n - s, 1)) for s in szs])
    return _solve_weighted_projected_lasso(rows, szs, vals, v0, v1, n, log_ws, alpha_scale=alpha_scale)


# =============================================================================
#  Dataset configuration
# =============================================================================

@dataclass
class DatasetSpec:
    name: str
    loader: callable
    n_estimators: int = 50
    max_depth: int = 3


def get_dataset_specs():
    """
    Datasets expected to produce approximately-sparse Shapley vectors.
    Edit this list based on the sparsity measurements printed by this script
    (rerun after changing and pick datasets with low s/n).
    """
    import shap
    return [
        # Highest-priority candidates: large n, known or expected sparse structure.
        DatasetSpec("Independent", shap.datasets.independentlinear60, n_estimators=50, max_depth=3),
        DatasetSpec("Correlated",  shap.datasets.corrgroups60,        n_estimators=50, max_depth=3),
        DatasetSpec("Communities", shap.datasets.communitiesandcrime, n_estimators=80, max_depth=4),
    ]


def load_dataset(spec):
    X, y = spec.loader()
    if hasattr(X, "values"): X = X.values
    if hasattr(y, "values"): y = y.values
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def fit_xgb_model(X, y, seed, n_estimators, max_depth):
    import xgboost as xgb
    if len(np.unique(y)) <= 20 and np.allclose(y, np.round(y)):
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  verbosity=0, random_state=seed, eval_metric="mlogloss")
    else:
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 verbosity=0, random_state=seed)
    model.fit(X, y)
    return model


def make_predict_fn(model):
    if hasattr(model, "predict_proba"):
        def predict(x):
            x = np.asarray(x, dtype=float)
            probs = model.predict_proba(x)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, -1]
            return probs.ravel()
        return predict
    return lambda x: model.predict(np.asarray(x, dtype=float))


def tree_shap_values(model, x_row):
    import shap
    explainer = shap.TreeExplainer(model)
    phi = explainer.shap_values(x_row[None, :])
    if isinstance(phi, list): phi = phi[-1]
    return _extract_single_output_shap(phi, n_features=len(x_row), class_index=-1)


# =============================================================================
#  Sparsity measurement
# =============================================================================

def measure_sparsity(model, X, n_points, threshold_frac, seed):
    """
    For each of n_points random test points, compute the Tree SHAP vector and
    report:
      s_eff     = #{i : |phi_i| > threshold_frac * ||phi||_inf}
      tail_mass = ||phi - top_{s_eff}(phi)||_2^2 / ||phi||_2^2
                  (fraction of l2^2 energy OUTSIDE the top-s_eff coords)
    Averaged/medianed across test points.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_points, len(X)), replace=False)
    n = X.shape[1]

    s_effs, tail_masses = [], []
    for i in idx:
        phi = tree_shap_values(model, X[i])
        absphi = np.abs(phi)
        if absphi.max() < 1e-12:
            continue
        thresh = threshold_frac * absphi.max()
        s_eff = int((absphi > thresh).sum())
        s_effs.append(s_eff)

        sorted_sq = np.sort(phi**2)[::-1]
        total = sorted_sq.sum()
        if total > 0:
            tail_masses.append(sorted_sq[s_eff:].sum() / total)

    return {
        "n": n,
        "s_eff_mean": float(np.mean(s_effs)) if s_effs else np.nan,
        "s_eff_median": float(np.median(s_effs)) if s_effs else np.nan,
        "s_over_n": float(np.mean(s_effs) / n) if s_effs else np.nan,
        "tail_mass_median": float(np.median(tail_masses)) if tail_masses else np.nan,
        "n_points_used": len(s_effs),
    }


# =============================================================================
#  Experiment runner
# =============================================================================

METHOD_ORDER = ["Kernel SHAP", "Optimized Kernel SHAP", "Leverage SHAP", "Sparse Leverage SHAP"]


def estimate_all_methods(n, v, x_row, bg, predict, m, seed):
    import shap
    out = {}
    out["Kernel SHAP"] = kernel_shap_baseline(n, v, m, seed=seed)
    ke = shap.KernelExplainer(predict, bg)
    ks = ke.shap_values(x_row[None, :], nsamples=int(m), silent=True)
    if isinstance(ks, list): ks = ks[-1]
    out["Optimized Kernel SHAP"] = _extract_single_output_shap(ks, n_features=n, class_index=-1)
    out["Leverage SHAP"] = leverage_shap(n, v, m, seed=seed)
    out["Sparse Leverage SHAP"] = sparse_leverage_shap(n, v, m, seed=seed)
    return out


def run_dataset(spec, curve_mults, n_trials, n_bg, threshold_frac, seed):
    X, y = load_dataset(spec)
    n = X.shape[1]
    print(f"\n=== {spec.name} (n={n}) ===")
    model = fit_xgb_model(X, y, seed=seed,
                          n_estimators=spec.n_estimators, max_depth=spec.max_depth)
    predict = make_predict_fn(model)

    # --- sparsity measurement ---
    sparsity = measure_sparsity(model, X, n_points=min(30, len(X)),
                                threshold_frac=threshold_frac, seed=seed)
    sparsity["dataset"] = spec.name
    print(f"  s_eff = {sparsity['s_eff_mean']:.1f},  "
          f"s/n = {sparsity['s_over_n']:.3f},  "
          f"tail mass (l2^2) = {sparsity['tail_mass_median']:.4f}")

    # --- error sweep ---
    rng = np.random.RandomState(seed)
    bg = X[rng.choice(len(X), size=min(n_bg, len(X)), replace=False)]
    curve_errors = {method: {mult: [] for mult in curve_mults} for method in METHOD_ORDER}

    t0 = time.time()
    for trial in range(n_trials):
        xi = rng.randint(0, len(X))
        x_row = X[xi]
        phi_true = tree_shap_values(model, x_row)
        v = make_shap_value_fn(predict, x_row, bg)
        for mult in curve_mults:
            m = max(4, int(mult * n))
            estimates = estimate_all_methods(n, v, x_row, bg, predict, m,
                                             seed=seed + 10000 * trial + int(mult * 100))
            for method in METHOD_ORDER:
                curve_errors[method][mult].append(
                    rel_l2(np.asarray(estimates[method]), phi_true))
    print(f"  Completed {n_trials} trials in {time.time()-t0:.1f}s")

    # --- summary stats ---
    stats_rows = []
    for method in METHOD_ORDER:
        for mult in curve_mults:
            vals = np.asarray(curve_errors[method][mult], dtype=float)
            vals = vals[~np.isnan(vals)]
            stats_rows.append({
                "dataset": spec.name, "method": method,
                "mult": mult, "m": max(4, int(mult * n)),
                "q1":     np.percentile(vals, 25) if len(vals) else np.nan,
                "median": np.median(vals)         if len(vals) else np.nan,
                "q3":     np.percentile(vals, 75) if len(vals) else np.nan,
                "mean":   np.mean(vals)           if len(vals) else np.nan,
            })

    return {
        "name": spec.name, "n": n,
        "sparsity": sparsity,
        "curve_errors": curve_errors,
        "stats": pd.DataFrame(stats_rows),
    }


# =============================================================================
#  Plotting
# =============================================================================

def plot_curves(results, curve_mults, output_path):
    n_cols = len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.0), squeeze=False)
    axes = axes[0]

    for idx, res in enumerate(results):
        ax = axes[idx]
        n = res["n"]
        sp = res["sparsity"]
        xs = np.array([max(4, int(mult * n)) for mult in curve_mults], dtype=float)

        for method in METHOD_ORDER:
            med, q1, q3 = [], [], []
            for mult in curve_mults:
                vals = np.asarray(res["curve_errors"][method][mult], dtype=float)
                vals = vals[~np.isnan(vals)]
                med.append(np.median(vals))
                q1.append(np.percentile(vals, 25))
                q3.append(np.percentile(vals, 75))
            ax.plot(xs, med, linewidth=2, label=method)
            ax.fill_between(xs, q1, q3, alpha=0.18)

        ax.set_xscale("log"); ax.set_yscale("log")
        # Reference lines: s log n  and  n log n  (sample-complexity thresholds)
        s_eff = sp["s_eff_mean"]
        if np.isfinite(s_eff):
            ax.axvline(s_eff * np.log(n), linestyle=":", color="#1a9850",
                       alpha=0.75, linewidth=1.8,
                       label=f"s·log n ≈ {s_eff*np.log(n):.0f}")
        ax.axvline(n * np.log(n), linestyle=":", color="#d6604d",
                   alpha=0.75, linewidth=1.8,
                   label=f"n·log n ≈ {n*np.log(n):.0f}")

        ax.set_title(
            f"{res['name']}  (n={n})\n"
            f"s_eff={sp['s_eff_mean']:.1f},  s/n={sp['s_over_n']:.2f},  "
            f"tail={sp['tail_mass_median']:.3f}",
            fontsize=10)
        ax.set_xlabel("Sample Size (m)")
        if idx == 0:
            ax.set_ylabel(r"Relative error  $\|\hat\phi-\phi\|^2/\|\phi\|^2$")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "Error vs sample size on real sparse datasets\n"
        "(green dotted = sparse-theory threshold s·log n,  "
        "red dotted = dense-theory threshold n·log n)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-data sparse Shapley experiment.")
    parser.add_argument("--output-dir", type=str, default="real_sparse_plots")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of repeated trials per (dataset, m).")
    parser.add_argument("--n-bg", type=int, default=20,
                        help="Background sample size for interventional SHAP.")
    parser.add_argument("--threshold-frac", type=float, default=0.05,
                        help="|phi_i| > threshold_frac * ||phi||_inf => 'active'.")
    parser.add_argument("--curve-mults", type=float, nargs="+",
                        default=[0.2, 0.5, 1, 2, 5, 10, 20, 40],
                        help="Sample budgets as multiples of n.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = get_dataset_specs()
    start = time.time()

    results, sparsity_rows = [], []
    for spec in specs:
        try:
            res = run_dataset(spec,
                              curve_mults=args.curve_mults,
                              n_trials=args.trials,
                              n_bg=args.n_bg,
                              threshold_frac=args.threshold_frac,
                              seed=args.seed)
            results.append(res)
            sparsity_rows.append(res["sparsity"])
        except Exception as e:
            print(f"  SKIPPED {spec.name}: {type(e).__name__}: {e}")

    if not results:
        print("No datasets ran successfully — nothing to plot.")
        return

    pd.DataFrame(sparsity_rows).to_csv(
        outdir / "sparsity_measurements.csv", index=False)
    pd.concat([r["stats"] for r in results], ignore_index=True).to_csv(
        outdir / "error_stats.csv", index=False)
    plot_curves(results, args.curve_mults,
                outdir / "error_vs_samples_sparse.png")

    print(f"\nDone. Outputs in: {outdir.resolve()}")
    print(f"Total elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
