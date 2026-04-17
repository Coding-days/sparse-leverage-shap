
"""
Replicate the plotting style of Figure 1 and Figure 3 from
"Provably Accurate Shapley Value Estimation via Leverage Score Sampling",
while adding a sparse-baseline estimator from our sparse leverage paper.

KEY REGIME DISTINCTION (cf. Hao et al., NeurIPS 2020):
  Data-poor:  m < n   — OLS is underdetermined (fewer samples than features).
                         Standard methods fail; sparse methods can still recover
                         the s-sparse Shapley vector from O(s log n) samples.
  Data-rich:  m >> n  — all methods converge; sparse advantage shrinks.

  The default --curve-mults spans BOTH regimes (0.2n to 160n) to show
  the transition. The vertical dash-dot line on Figure 3 marks m = n.

Outputs
-------
- figure1_predicted_vs_true.png
- figure3_error_vs_samples.png
- table_m10n.csv

Methods compared
----------------
1. Kernel SHAP              : custom baseline, with-replacement, no pairing
2. Optimized Kernel SHAP    : shap.KernelExplainer
3. Leverage SHAP            : leverage-score size sampling + weighted OLS
4. Sparse Leverage SHAP     : sparse-biased size sampling + weighted LASSO

Notes
-----
- The paper's experiments use XGBoost with Tree SHAP for ground truth.
- The paper's main figures use 8 datasets from shap.datasets.
- Figure 1 uses m = 5n samples.
- Figure 3 reports median with Q1-Q3 bands over repeated runs as m varies.
"""

from __future__ import annotations

import argparse
import math
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
#  Utilities
# =============================================================================

def _log_comb(n: int, k: int) -> float:
    """log C(n,k) via log-gamma — avoids overflow for large n."""
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rel_l2(phi_hat: np.ndarray, phi_true: np.ndarray) -> float:
    """||phi_hat - phi_true||^2 / ||phi_true||^2  — the ICLR paper's primary metric."""
    denom = np.linalg.norm(phi_true) ** 2
    if denom < 1e-30:
        return float("nan")
    return np.linalg.norm(phi_hat - phi_true) ** 2 / denom


def ensure_even(m: int) -> int:
    """Paired sampling needs even m (each z is paired with complement 1-z)."""
    return int(m) if int(m) % 2 == 0 else int(m) + 1


def clip_budget_for_exact_methods(n: int, m: int) -> int:
    # The regression design has 2^n - 2 rows, but the paper notes the optimized
    # methods become exact by ~2n in the small-n regime due to without-replacement
    # coverage over subset sizes. For safety, only cap the custom paired samplers.
    return ensure_even(m)


# =============================================================================
#  Core value-function construction
# =============================================================================


def _extract_single_output_shap(phi_obj, n_features: int, class_index: int = -1) -> np.ndarray:
    """
    Normalize SHAP outputs into a flat (n_features,) vector.
    Different shap/xgboost versions return different shapes
    (1D, 2D, 3D, Explanation objects) — this handles all of them.
    """
    values = getattr(phi_obj, "values", phi_obj)
    arr = np.asarray(values)

    # Common cases:
    #   (n_features,)
    #   (1, n_features)
    #   (1, n_features, n_outputs)   <- newer TreeExplainer multiclass
    #   (1, n_outputs, n_features)
    #   (n_outputs, 1, n_features)   <- some older multiclass conventions
    #   list[n_outputs] of (1, n_features)
    if arr.ndim == 1:
        if arr.shape[0] != n_features:
            raise ValueError(f"Expected {n_features} SHAP values, got shape {arr.shape}")
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[-1] == n_features:
            return arr[0].astype(float)
        if arr.shape[0] == n_features and arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        raise ValueError(f"Cannot coerce SHAP array of shape {arr.shape} to length {n_features}")

    if arr.ndim == 3:
        # sample x feature x output
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            return arr[0, :, class_index].astype(float)
        # sample x output x feature
        if arr.shape[0] == 1 and arr.shape[2] == n_features:
            return arr[0, class_index, :].astype(float)
        # output x sample x feature
        if arr.shape[1] == 1 and arr.shape[2] == n_features:
            return arr[class_index, 0, :].astype(float)
        # output x feature x sample
        if arr.shape[0] != 1 and arr.shape[1] == n_features and arr.shape[2] == 1:
            return arr[class_index, :, 0].astype(float)
        raise ValueError(f"Cannot coerce 3D SHAP array of shape {arr.shape} to length {n_features}")

    raise ValueError(f"Unsupported SHAP output shape {arr.shape}")

def make_shap_value_fn(predict, x_test, bg: np.ndarray):
    """
    Interventional SHAP value function:
      v(z) = E_bg[f(z * x_test + (1-z) * bg)]
    """
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
#  Sampling-based estimators
# =============================================================================

def _solve_weighted_projected_ols(rows: np.ndarray, szs: np.ndarray, vals: np.ndarray,
                                  v0: float, v1: float, n: int, log_ws: np.ndarray) -> np.ndarray:
    """
    Shared OLS solver for the Shapley regression problem.

    The Shapley constraint <phi, 1> = v([n]) - v(0) is removed via projection
    P = I - (1/n)11^T, turning it into unconstrained weighted least squares:
        min_x || sqrt(W) * (ZP x - b) ||^2
    Then phi = x + [v([n]) - v(0)] / n  adds the constraint back.
    """
    ws = np.exp(log_ws - np.max(log_ws))          # numerical stability
    ZP = rows - szs[:, None] / n                   # each row projected: z - (s/n)*1
    b = vals - v0 - (szs / n) * (v1 - v0)         # projected target vector
    sqW = np.sqrt(ws)
    phi, *_ = np.linalg.lstsq(ZP * sqW[:, None], b * sqW, rcond=None)
    return phi + (v1 - v0) / n


def _solve_weighted_projected_lasso(rows: np.ndarray, szs: np.ndarray, vals: np.ndarray,
                                    v0: float, v1: float, n: int, log_ws: np.ndarray,
                                    alpha_scale: float = 0.005) -> np.ndarray:
    """
    Same projected regression as OLS, but solved with LASSO.
    The L1 penalty encourages sparse solutions — most coordinates shrink to 0.
    This is what lets Sparse Leverage SHAP work with O(s log n) samples
    instead of the O(n) that OLS requires.
    """
    ws = np.exp(log_ws - np.max(log_ws))
    ZP = rows - szs[:, None] / n
    b = vals - v0 - (szs / n) * (v1 - v0)
    sqW = np.sqrt(ws)
    Aw = ZP * sqW[:, None]
    bw = b * sqW
    # alpha scales with noise level / sqrt(m): more samples → less regularization
    alpha = alpha_scale * np.std(bw) / max(np.sqrt(len(bw)), 1.0)
    alpha = max(alpha, 1e-10)
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-6, fit_intercept=False)
    lasso.fit(Aw, bw)
    return lasso.coef_ + (v1 - v0) / n


def kernel_shap_baseline(n: int, v, m: int, seed: int = 0) -> np.ndarray:
    """
    Unoptimized Kernel SHAP-style baseline.

    Sampling:
        - with replacement
        - no paired sampling
        - subset-size probability proportional to 1 / [s (n-s)]
    Solver:
        - weighted projected OLS using Kernel SHAP regression weights
    """
    rng = np.random.RandomState(seed)
    m = max(2, int(m))
    sizes = np.arange(1, n)
    # Kernel SHAP weight: 1/[s(n-s)].  Overweights extreme sizes (s≈1 and s≈n-1).
    size_probs = 1.0 / (sizes * (n - sizes))
    size_probs = size_probs / size_probs.sum()

    rows = np.zeros((m, n), dtype=float)
    szs = np.zeros(m, dtype=float)
    for i in range(m):
        s = int(rng.choice(sizes, p=size_probs))
        idx = rng.choice(n, size=s, replace=False)
        rows[i, idx] = 1.0
        szs[i] = s

    v0 = v(np.zeros((1, n)))[0]
    v1 = v(np.ones((1, n)))[0]
    vals = v(rows)
    # w(s) = C(n,s)/[s(n-s)] — standard Kernel SHAP regression weight
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(s * (n - s), 1)) for s in szs])
    return _solve_weighted_projected_ols(rows, szs, vals, v0, v1, n, log_ws)


def leverage_shap(n: int, v, m: int, seed: int = 0) -> np.ndarray:
    """
    Leverage SHAP with paired sampling.

    Sampling:
        - subset size uniform over {1,...,n-1}
        - paired with complement
        - with replacement over pairs
    Solver:
        - weighted projected OLS
    """
    rng = np.random.RandomState(seed)
    m = clip_budget_for_exact_methods(n, m)
    rows, szs = [], []
    for _ in range(max(1, m // 2)):
        s = rng.randint(1, n)               # <<< KEY: size UNIFORM on {1,...,n-1}
        idx = rng.choice(n, size=s, replace=False)
        z = np.zeros(n, dtype=float)
        z[idx] = 1.0
        rows.extend([z, 1.0 - z])
        szs.extend([s, n - s])

    rows = np.asarray(rows, dtype=float)[:m]
    szs = np.asarray(szs, dtype=float)[:m]
    v0 = v(np.zeros((1, n)))[0]
    v1 = v(np.ones((1, n)))[0]
    vals = v(rows)

    # Same w(s) = C(n,s)/[s(n-s)] as Kernel SHAP, but different sampling distribution
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(s * (n - s), 1)) for s in szs])
    return _solve_weighted_projected_ols(rows, szs, vals, v0, v1, n, log_ws)


def sparse_leverage_shap(n: int, v, m: int, seed: int = 0, alpha_scale: float = 0.005) -> np.ndarray:
    """
    Sparse leverage baseline.

    Sampling:
        - subset size probability proportional to s
        - paired with complement
    Solver:
        - weighted projected LASSO
    """
    rng = np.random.RandomState(seed)
    m = clip_budget_for_exact_methods(n, m)
    sizes = np.arange(1, n)
    probs = sizes.astype(float)               # <<< KEY: P(size=s) proportional to s
    probs = probs / probs.sum()                #     larger subsets sampled more often

    rows, szs = [], []
    for _ in range(max(1, m // 2)):
        s = int(rng.choice(sizes, p=probs))    # biased draw (vs uniform in Leverage SHAP)
        idx = rng.choice(n, size=s, replace=False)
        z = np.zeros(n, dtype=float)
        z[idx] = 1.0
        rows.extend([z, 1.0 - z])
        szs.extend([s, n - s])

    rows = np.asarray(rows, dtype=float)[:m]
    szs = np.asarray(szs, dtype=float)[:m]
    v0 = v(np.zeros((1, n)))[0]
    v1 = v(np.ones((1, n)))[0]
    vals = v(rows)

    # Sparse weight: w(s) = C(n,s)/(n-s)  — drops the 1/s factor vs standard
    log_ws = np.array([_log_comb(n, int(s)) - np.log(max(n - s, 1)) for s in szs])
    return _solve_weighted_projected_lasso(rows, szs, vals, v0, v1, n, log_ws, alpha_scale=alpha_scale)


# =============================================================================
#  Datasets and models
# =============================================================================

@dataclass
class DatasetSpec:
    """Config for one dataset: name, loader function, and XGBoost hyperparams."""
    name: str
    short_name: str
    loader: callable
    n_estimators: int = 100
    max_depth: int = 4


def get_dataset_specs():
    import shap

    return [
        DatasetSpec("IRIS", "IRIS", shap.datasets.iris),
        DatasetSpec("California", "California", shap.datasets.california),
        DatasetSpec("Diabetes", "Diabetes", shap.datasets.diabetes),
        DatasetSpec("Adult", "Adult", shap.datasets.adult),
        DatasetSpec("Correlated", "Correlated", shap.datasets.corrgroups60, n_estimators=50, max_depth=3),
        DatasetSpec("Independent", "Independent", shap.datasets.independentlinear60, n_estimators=50, max_depth=3),
        DatasetSpec("NHANES", "NHANES", shap.datasets.nhanesi, n_estimators=80, max_depth=4),
        DatasetSpec("Communities", "Communities", shap.datasets.communitiesandcrime, n_estimators=80, max_depth=4),
    ]


def load_dataset(spec: DatasetSpec):
    X, y = spec.loader()
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y


def fit_xgb_model(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int, max_depth: int):
    """Train XGBoost. Auto-detects classification vs regression from target values."""
    import xgboost as xgb

    if len(np.unique(y)) <= 20 and np.allclose(y, np.round(y)):
        # Treat as classification when target is low-cardinality integer labels.
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbosity=0,
            random_state=seed,
            eval_metric="mlogloss",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbosity=0,
            random_state=seed,
        )
    model.fit(X, y)
    return model


def make_predict_fn(model):
    """Wrap model into a single-output predict function.
    For classifiers: returns P(last class). For regressors: returns prediction."""
    if hasattr(model, "predict_proba"):
        def predict(x):
            x = np.asarray(x, dtype=float)
            probs = model.predict_proba(x)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, -1]
            return probs.ravel()
        return predict
    return lambda x: model.predict(np.asarray(x, dtype=float))


def tree_shap_values(model, x_row: np.ndarray) -> np.ndarray:
    """Exact Shapley values via Tree SHAP — this is the ground truth."""
    import shap

    explainer = shap.TreeExplainer(model)
    phi = explainer.shap_values(x_row[None, :])
    if isinstance(phi, list):
        phi = phi[-1]
    return _extract_single_output_shap(phi, n_features=len(x_row), class_index=-1)


# =============================================================================
#  Experiment runner
# =============================================================================

METHOD_ORDER = ["Kernel SHAP", "Optimized Kernel SHAP", "Leverage SHAP", "Sparse Leverage SHAP"]


def estimate_all_methods(n: int, v, x_row: np.ndarray, bg: np.ndarray, predict, m: int, seed: int):
    """Run all 4 estimators on one test point with budget m. Returns dict of estimates."""
    import shap

    out = {}
    out["Kernel SHAP"] = kernel_shap_baseline(n, v, m, seed=seed)           # our unoptimized baseline
    ke = shap.KernelExplainer(predict, bg)                                   # shap library's optimized version
    ks = ke.shap_values(x_row[None, :], nsamples=int(m), silent=True)
    if isinstance(ks, list):
        ks = ks[-1]
    out["Optimized Kernel SHAP"] = _extract_single_output_shap(ks, n_features=n, class_index=-1)
    out["Leverage SHAP"] = leverage_shap(n, v, m, seed=seed)
    out["Sparse Leverage SHAP"] = sparse_leverage_shap(n, v, m, seed=seed)
    return out


def run_dataset(spec: DatasetSpec,
                output_dir: Path,
                curve_mults: list[int],
                scatter_mult: int,
                trials_curve: int,
                trials_scatter: int,
                n_bg: int,
                seed: int):
    """
    Full experiment for one dataset. Two sub-experiments:
      1. Scatter (Figure 1): predicted vs true Shapley at m = scatter_mult * n
      2. Curves  (Figure 3): error vs sample size across curve_mults
    """
    X, y = load_dataset(spec)
    n = X.shape[1]
    print(f"\n=== {spec.name} (n={n}) ===")
    model = fit_xgb_model(X, y, seed=seed, n_estimators=spec.n_estimators, max_depth=spec.max_depth)
    predict = make_predict_fn(model)

    rng = np.random.RandomState(seed)
    bg = X[rng.choice(len(X), size=min(n_bg, len(X)), replace=False)]

    scatter_points = {method: {"true": [], "pred": []} for method in METHOD_ORDER}
    curve_errors = {method: {mult: [] for mult in curve_mults} for method in METHOD_ORDER}
    m10n_rows = []

    # ── Figure 1 experiment: predicted vs true at fixed m = scatter_mult * n ──
    for trial in range(trials_scatter):
        xi = rng.randint(0, len(X))
        x_row = X[xi]
        phi_true = tree_shap_values(model, x_row)
        v = make_shap_value_fn(predict, x_row, bg)
        estimates = estimate_all_methods(n, v, x_row, bg, predict, max(4, int(scatter_mult * n)), seed=seed + 1000 * trial)
        for method in METHOD_ORDER:
            scatter_points[method]["true"].extend(phi_true.tolist())
            scatter_points[method]["pred"].extend(np.asarray(estimates[method]).tolist())

    # ── Figure 3 experiment: error vs sample size (median + Q1-Q3 bands) ──
    t0 = time.time()
    for trial in range(trials_curve):
        xi = rng.randint(0, len(X))
        x_row = X[xi]
        phi_true = tree_shap_values(model, x_row)
        v = make_shap_value_fn(predict, x_row, bg)
        for mult in curve_mults:
            m = max(4, int(mult * n))       # floor to int, minimum 4 samples
            estimates = estimate_all_methods(n, v, x_row, bg, predict, m, seed=seed + 10000 * trial + int(mult * 100))
            for method in METHOD_ORDER:
                err = rel_l2(np.asarray(estimates[method]), phi_true)
                curve_errors[method][mult].append(err)

        if 10 in curve_mults:
            mult = 10
            for method in METHOD_ORDER:
                vals = np.asarray(curve_errors[method][mult], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0 and len(vals) >= trial + 1:
                    cur = vals[-1]
                    m10n_rows.append({
                        "dataset": spec.name,
                        "method": method,
                        "trial": trial,
                        "m": 10 * n,
                        "error": cur,
                    })
    print(f"Completed curve trials in {time.time() - t0:.1f}s")

    # Summary stats
    stats_rows = []
    for method in METHOD_ORDER:
        for mult in curve_mults:
            vals = np.asarray(curve_errors[method][mult], dtype=float)
            vals = vals[~np.isnan(vals)]
            stats_rows.append({
                "dataset": spec.name,
                "method": method,
                "mult": mult,
                "m": max(4, int(mult * n)),
                "mean": np.mean(vals) if len(vals) else np.nan,
                "q1": np.percentile(vals, 25) if len(vals) else np.nan,
                "median": np.median(vals) if len(vals) else np.nan,
                "q3": np.percentile(vals, 75) if len(vals) else np.nan,
            })

    return {
        "name": spec.name,
        "n": n,
        "scatter_points": scatter_points,
        "curve_errors": curve_errors,
        "stats": pd.DataFrame(stats_rows),
        "m10n_rows": pd.DataFrame(m10n_rows),
    }


# =============================================================================
#  Plotting
# =============================================================================

def _identity_limits(x, y):
    vals = np.concatenate([np.asarray(x), np.asarray(y)])
    lo = np.nanpercentile(vals, 1)
    hi = np.nanpercentile(vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0
    pad = 0.05 * (hi - lo + 1e-12)
    return lo - pad, hi + pad


def plot_figure1(results: list[dict], output_path: Path):
    """Scatter plot grid: rows = datasets, columns = methods. Points near diagonal = accurate."""
    n_rows = len(results)
    n_cols = len(METHOD_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 2.5 * n_rows), squeeze=False)

    for r, res in enumerate(results):
        for c, method in enumerate(METHOD_ORDER):
            ax = axes[r, c]
            xt = np.asarray(res["scatter_points"][method]["true"])
            yp = np.asarray(res["scatter_points"][method]["pred"])
            lo, hi = _identity_limits(xt, yp)
            ax.scatter(xt, yp, s=10, alpha=0.45)
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(method, fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{res["name"]}\nPredicted', fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("True Shapley Values", fontsize=10)
            ax.grid(alpha=0.2)

    fig.suptitle("Figure 1-style replication: Predicted vs true Shapley values at m = 5n", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(results: list[dict], curve_mults: list[int], output_path: Path):
    """Log-log error vs sample size plot. Lines = median, shaded bands = Q1-Q3."""
    n_rows = math.ceil(len(results) / 4)
    n_cols = min(4, len(results))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.3 * n_rows), squeeze=False)
    axes = axes.ravel()

    for idx, res in enumerate(results):
        ax = axes[idx]
        n = res["n"]
        xs = np.array([max(4, int(mult * n)) for mult in curve_mults], dtype=float)

        for method in METHOD_ORDER:
            med, q1, q3 = [], [], []
            for mult in curve_mults:
                vals = np.asarray(res["curve_errors"][method][mult], dtype=float)
                vals = vals[~np.isnan(vals)]
                med.append(np.median(vals))
                q1.append(np.percentile(vals, 25))
                q3.append(np.percentile(vals, 75))

            med = np.asarray(med)
            q1 = np.asarray(q1)
            q3 = np.asarray(q3)
            ax.plot(xs, med, linewidth=2, label=method)
            ax.fill_between(xs, q1, q3, alpha=0.18)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f'{res["name"]} (n = {n})')
        ax.set_xlabel("Sample Size (m)")
        ax.set_ylabel(r"Error in $\ell_2$-norm")
        ax.grid(alpha=0.25, which="both")
        # Mark m = n: the boundary between data-poor (left) and data-rich (right)
        if n >= xs.min() and n <= xs.max():
            ax.axvline(n, linestyle="-.", color="gray", linewidth=1.2, alpha=0.6)
        if 2 * n >= xs.min() and 2 * n <= xs.max():
            ax.axvline(2 * n, linestyle="--", linewidth=1, alpha=0.7)  # 2n mark: exact recovery threshold

    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHOD_ORDER), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Error vs sample size (dash-dot line = m=n boundary: poor ← | → rich)", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.975])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Figure 1 / Figure 3 style plots and add Sparse Leverage SHAP.")
    parser.add_argument("--output-dir", type=str, default="paper_style_plots")
    parser.add_argument("--curve-trials", type=int, default=100, help="Number of repeated runs for Figure 3-style curves.")
    parser.add_argument("--scatter-trials", type=int, default=25, help="Number of repeated runs for Figure 1-style scatter aggregation.")
    parser.add_argument("--n-bg", type=int, default=20, help="Background sample size for interventional SHAP value function.")
    parser.add_argument("--seed", type=int, default=0)
    # Data-poor regime (m < n) is where sparse methods shine most.
    # Data-rich regime (m >> n) is where the ICLR paper's experiments live.
    # We include both to show the full picture.
    parser.add_argument("--curve-mults", type=float, nargs="+",
                        default=[0.2, 0.5, 1, 2, 5, 10, 20, 40, 80, 160],
                        help="Sample budget as multiples of n. Values < 1 = data-poor regime.")
    parser.add_argument("--scatter-mult", type=float, default=5)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = get_dataset_specs()
    all_results = []
    all_stats = []
    all_m10n = []

    start = time.time()
    for spec in specs:
        res = run_dataset(
            spec=spec,
            output_dir=outdir,
            curve_mults=args.curve_mults,
            scatter_mult=args.scatter_mult,
            trials_curve=args.curve_trials,
            trials_scatter=args.scatter_trials,
            n_bg=args.n_bg,
            seed=args.seed,
        )
        all_results.append(res)
        all_stats.append(res["stats"])
        all_m10n.append(res["m10n_rows"])

    stats_df = pd.concat(all_stats, ignore_index=True)
    m10n_df = pd.concat(all_m10n, ignore_index=True)

    plot_figure1(all_results, outdir / "figure1_predicted_vs_true.png")
    plot_figure3(all_results, args.curve_mults, outdir / "figure3_error_vs_samples.png")

    stats_df.to_csv(outdir / "all_curve_stats.csv", index=False)

    if not m10n_df.empty:
        summary = (
            m10n_df.groupby(["dataset", "method"])["error"]
            .agg(mean="mean",
                 q1=lambda x: np.percentile(x, 25),
                 median="median",
                 q3=lambda x: np.percentile(x, 75))
            .reset_index()
        )
        summary.to_csv(outdir / "table_m10n.csv", index=False)

    with open(outdir / "README.txt", "w") as f:
        f.write(
            "Generated outputs:\n"
            " - figure1_predicted_vs_true.png\n"
            " - figure3_error_vs_samples.png\n"
            " - all_curve_stats.csv\n"
            " - table_m10n.csv (if 10n is in --curve-mults)\n\n"
            "Defaults match the paper's plot types, but Sparse Leverage SHAP is added as an extra baseline.\n"
        )

    print(f"\nDone. Outputs saved to: {outdir.resolve()}")
    print(f"Total elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
