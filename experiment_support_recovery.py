#!/usr/bin/env python3
"""
experiment_support_recovery.py
═════════════════════════════════════════════════════════════════════════════
Synthetic support-recovery experiment for Sparse / Dense Leverage / Kernel SHAP.

Tests the O(s log n) vs O(n log n) sample-complexity gap directly on a
controlled sparse game with a known active set R. Reports both estimation
error and EXACT SUPPORT RECOVERY rate — the fraction of trials where
{i : |phi_hat_i| > eps} equals the true active set R. Support recovery is
the distinctive metric here: it tests whether the method identifies the
right features, not just whether its numbers are close.

Companion experiment: `experiment_real_data_benchmark.py` measures accuracy
on real datasets across sample budgets spanning both regimes.

Two settings:
  Setting A — Very sparse:  n=30, s=3   (s/n = 10%)
  Setting B — Less sparse:  n=30, s=12  (s/n = 40%)

X-AXIS: raw m (number of value-function evaluations)
  The paper's central claim is a sample-complexity advantage:
    Dense Leverage SHAP  needs O(n log n) ≈ 100 evaluations
    Sparse Leverage SHAP needs O(s log n) ≈ 10–40 evaluations
  Plotting against m directly shows where each method starts to converge.

CORRECT IS WEIGHTS (key fix from previous version)
──────────────────────────────────────────────────
Shapley regression weight: w(k) = 1 / [C(n,k) · k · (n−k)]
  C(n,k) is in the DENOMINATOR — opposite of what was used before.

IS-corrected effective weight (w(k) / p_S):
  Kernel SHAP   (p_S ∝ w(k))      : eff_w = 1           [constant]
  Dense Lev.    (p_S ∝ 1/C(n,k))  : eff_w = 1/[k(n−k)]  [stable, small]
  Sparse Lev.   (p_S ∝ k/C(n,k))  : eff_w = 1/[k(n−k)]  [SAME as dense!]

Both leverage methods use identical effective weights. Their ONLY difference
is which subsets are sampled. For n=30, condition ratio max/min eff_w ≈ 8.

WHY n=30?
  n=50 → C(50,25)≈10^14 → catastrophic conditioning even after IS correction.
  n=30 → C(30,15)≈155M  → max/min effective weight ≈ 8  → numerically stable.
  2^30 ≈ 10^9 → impossible to enumerate, so m << 2^n throughout.

EXACT SHAPLEY VALUES: active-player reduction
  By the Null Player axiom, ϕᵢ = 0 for all i ∉ R.
  For i ∈ R, use the s-player Shapley formula (dummy players don't change ratios).
  Requires only 2^s oracle calls per player regardless of n.
"""

import numpy as np
from math import factorial, log
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# §1. GAME
# ═══════════════════════════════════════════════════════════════════════════

def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
SIGMA0 = _sig(np.zeros(1))[0]

def make_game(n, active_players, coefficients):
    """
    Sigmoid sparse game:  v(S) = σ(Σ_{i∈S∩R} aᵢ) − σ(0).
    Active set R = active_players.  All others are exact dummies.
    Non-linear → non-zero regression residual → meaningful comparison.
    """
    def vbatch(Z): return _sig(Z[:, active_players] @ coefficients) - SIGMA0
    return vbatch


# ═══════════════════════════════════════════════════════════════════════════
# §2. EXACT SHAPLEY (active-player reduction)
# ═══════════════════════════════════════════════════════════════════════════

def exact_shapley(n, active_players, vbatch):
    """
    ϕᵢ = Σ_{T⊆R\{i}} ws(|T|) · [v(T∪{i}) − v(T)]    (i ∈ R)
    ϕᵢ = 0                                             (i ∉ R)
    ws(t) = t!(s−t−1)!/s! — standard s-player Shapley weight.
    Uses 2^s oracle calls per active player.
    """
    s, phi = len(active_players), np.zeros(n)
    for i in active_players:
        others = [j for j in active_players if j != i]
        ZT, ZTi, wts = [], [], []
        for r in range(s):
            ws   = factorial(r) * factorial(s-r-1) / factorial(s)
            subs = list(combinations(others, r)) if r > 0 else [()]
            for sub in subs:
                T = list(sub)
                zT = np.zeros(n); zT[T] = 1.0
                zTi = zT.copy(); zTi[i] = 1.0
                ZT.append(zT); ZTi.append(zTi); wts.append(ws)
        ZT, ZTi, wts = np.stack(ZT), np.stack(ZTi), np.array(wts)
        phi[i] = (wts * (vbatch(ZTi) - vbatch(ZT))).sum()
    return phi


# ═══════════════════════════════════════════════════════════════════════════
# §3. SAMPLING AND IS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

def size_probabilities(n, method):
    """
    P(size = k) for k ∈ {1,…,n−1}.  A specific subset of that size is then
    drawn uniformly: p_S = P(size=k)/C(n,k).

    Kernel SHAP:    p_S ∝ w(k) = 1/[C(n,k)·k(n−k)]
                    → P(size=k) ∝ 1/[k(n−k)]   (extreme sizes)
    Dense Leverage: p_S ∝ 1/C(n,k)
                    → P(size=k) = 1/(n−1)       (uniform)
    Sparse Leverage:p_S ∝ k/C(n,k)
                    → P(size=k) ∝ k              (large subsets preferred)
      Rationale: P(size-k ⊂ random ∩ R ≠ ∅) = 1−C(n−s,k)/C(n,k) ≥ 1−(1−s/n)^k.
      This probability increases with k, so large subsets yield more signal
      about the unknown active set when s is small relative to n.
    """
    sizes = np.arange(1, n)
    if   method == "kernel":   raw = 1.0 / (sizes * (n - sizes))
    elif method == "leverage": raw = np.ones(n-1, float)
    elif method == "sparse":   raw = sizes.astype(float)
    return raw / raw.sum()


def eff_weight(k, n, method):
    """
    IS-corrected effective regression weight:  eff_w = w(k) / p_S.

    Crucially, Dense and Sparse Leverage have the SAME eff_w = 1/[k(n−k)].
    Their difference is entirely in WHICH subsets are sampled, not how
    sampled rows are weighted in the regression.

    Derivation (w(k) = 1/[C(n,k)·k·(n−k)] is the Shapley weight):
      Kernel:  p_S ∝ w(k)        → eff_w = w(k)/w(k) = constant = 1
      Dense:   p_S ∝ 1/C(n,k)   → eff_w = [1/C·k(n-k)] / [1/C] = 1/[k(n-k)]
      Sparse:  p_S ∝ k/C(n,k)   → eff_w = [1/C·k(n-k)] / [k/C] = 1/[k(n-k)]
                                           (same as dense — uses modified w̃, §2.4)
    """
    if k <= 0 or k >= n: return 0.0
    if   method == "kernel":   return 1.0
    elif method == "leverage": return 1.0 / (k * (n - k))
    elif method == "sparse":   return 1.0 / (k * (n - k))   # same as dense


def draw_paired_wor(n, n_pairs, probs, rng):
    """Without-replacement paired sampling: prevents repeated rows → rank-deficiency."""
    sizes = np.arange(1, n)
    used, rows = set(), []
    for _ in range(n_pairs * 40):
        if len(rows) >= 2 * n_pairs: break
        k   = int(rng.choice(sizes, p=probs))
        sub = frozenset(rng.choice(n, k, replace=False).tolist())
        if sub not in used:
            used.add(sub)
            z = np.zeros(n); z[list(sub)] = 1.0
            rows.extend([z, 1.0 - z])
    return np.array(rows) if rows else np.zeros((0, n))


# ═══════════════════════════════════════════════════════════════════════════
# §4. ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

def estimate_shapley(sampled_z, method, n, v_empty, v_full, vbatch,
                     ridge=1e-6):
    """
    IS-corrected Shapley estimator (Lemma 3.1 [1]).
    Efficiency component c·1 is set exactly; residual ϕ⊥ solved by ridge LS.
    Ridge (λ=1e-6 after weight normalisation) stabilises near-rank-deficient
    systems at small m without materially biasing well-conditioned ones.
    """
    m = len(sampled_z)
    if m == 0: return np.zeros(n)
    c     = (v_full - v_empty) / n
    k_vec = sampled_z.sum(axis=1).astype(int)
    v_vals = vbatch(sampled_z)
    Pz = sampled_z - (k_vec[:, None] / n)
    ew = np.array([eff_weight(k, n, method) for k in k_vec])
    ew /= ew.mean() if ew.mean() > 0 else 1.0
    sw = np.sqrt(np.maximum(ew, 0))
    A  = sw[:, None] * Pz
    b  = sw * (v_vals - v_empty - k_vec * c)
    phi_perp = np.linalg.solve(A.T @ A + ridge * np.eye(n), A.T @ b)
    return phi_perp + c * np.ones(n)


# ═══════════════════════════════════════════════════════════════════════════
# §5. SINGLE SETTING RUNNER
# ═══════════════════════════════════════════════════════════════════════════

METHODS = {
    "Kernel SHAP":          "kernel",
    "Dense Leverage SHAP":  "leverage",
    "Sparse Leverage SHAP": "sparse",
}

def run_setting(n, s, sample_sizes, n_trials=300, seed=0):
    rng_s          = np.random.default_rng(seed)
    active_players = rng_s.choice(n, size=s, replace=False)
    mags           = rng_s.uniform(1.5, 3.0, s)
    signs          = rng_s.choice([-1,1], s)
    coefficients   = mags * signs

    vbatch  = make_game(n, active_players, coefficients)
    v_full  = float(vbatch(np.ones((1, n)))[0])
    v_empty = 0.0

    print(f"\n  Exact Shapley (n={n}, s={s}) …", end=" ", flush=True)
    phi_true = exact_shapley(n, active_players, vbatch)
    print("done.")

    inactive = np.ones(n, bool); inactive[active_players] = False
    assert np.abs(phi_true[inactive]).max() < 1e-7
    assert abs(phi_true.sum() - (v_full - v_empty)) < 1e-7

    beta_min = float(np.abs(phi_true[active_players]).min())
    eps      = beta_min / 3
    true_set = set(active_players.tolist())
    phi_sq   = float(phi_true @ phi_true)

    print(f"  Active={sorted(active_players.tolist())}")
    print(f"  ϕ(active)={phi_true[active_players].round(4).tolist()}")
    print(f"  β_min={beta_min:.4f},  ε={eps:.4f}")

    rng    = np.random.default_rng(seed + 1)
    probs  = {lb: size_probabilities(n, k) for lb, k in METHODS.items()}
    errors = {lb: [] for lb in METHODS}
    recovery = {lb: [] for lb in METHODS}

    for m in sample_sizes:
        n_pairs  = m // 2
        per_err  = {lb: [] for lb in METHODS}
        per_rec  = {lb: 0  for lb in METHODS}
        for _ in range(n_trials):
            for lb, key in METHODS.items():
                z   = draw_paired_wor(n, n_pairs, probs[lb], rng)
                phi = estimate_shapley(z, key, n, v_empty, v_full, vbatch)
                err = float(np.sum((phi - phi_true)**2)) / phi_sq
                per_err[lb].append(err)
                if set(np.where(np.abs(phi) > eps)[0].tolist()) == true_set:
                    per_rec[lb] += 1
        for lb in METHODS:
            arr = np.array(per_err[lb])
            errors[lb].append(arr)
            recovery[lb].append(per_rec[lb] / n_trials)
            print(f"  m={m:4d} ({m//n:2d}n)  {lb:<25s}  "
                  f"med={np.median(arr):.4f}  "
                  f"rec={100*per_rec[lb]/n_trials:.0f}%")

    return phi_true, errors, recovery, beta_min, eps, active_players


# ═══════════════════════════════════════════════════════════════════════════
# §6. PLOT
# ═══════════════════════════════════════════════════════════════════════════

STYLE = {
    "Kernel SHAP":          dict(color="#2171b5", ls="--", marker="o", lw=2, ms=6),
    "Dense Leverage SHAP":  dict(color="#d6604d", ls="-.", marker="s", lw=2, ms=6),
    "Sparse Leverage SHAP": dict(color="#1a9850", ls="-",  marker="^", lw=2, ms=7),
}

def plot_comparison(results_A, results_B, sample_sizes, n, sA, sB):
    _, errs_A, recs_A, bmin_A, eps_A, _ = results_A
    _, errs_B, recs_B, bmin_B, eps_B, _ = results_B

    ref_dense = n * log(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Sample Complexity: Sparse vs Less-Sparse  (n={n})\n"
        r"$\bf{X-axis: }$" + "raw number of value-function evaluations m "
        r"(the central sample-complexity metric)",
        fontsize=12)

    for col, (errs, recs, ss, bmin, eps) in enumerate([
        (errs_A, recs_A, sA, bmin_A, eps_A),
        (errs_B, recs_B, sB, bmin_B, eps_B),
    ]):
        ref_sparse = ss * log(n)
        ax_e, ax_r = axes[0][col], axes[1][col]

        # Complexity reference lines
        for ax in (ax_e, ax_r):
            ax.axvline(ref_sparse, color="#1a9850", ls=":", lw=2.0, alpha=0.75,
                       label=f"Sparse theory: O(s·log n) ≈ {ref_sparse:.0f}")
            ax.axvline(ref_dense,  color="#d6604d", ls=":", lw=2.0, alpha=0.75,
                       label=f"Dense theory: O(n·log n) ≈ {ref_dense:.0f}")

        # Error (log scale)
        for lb, err_list in errs.items():
            med = [np.median(e)         for e in err_list]
            q1  = [np.percentile(e, 25) for e in err_list]
            q3  = [np.percentile(e, 75) for e in err_list]
            ax_e.semilogy(sample_sizes, med, label=lb, **STYLE[lb])
            ax_e.fill_between(sample_sizes, q1, q3,
                              alpha=0.14, color=STYLE[lb]["color"])

        ax_e.set_ylabel("Normalised ℓ₂ error  ‖ϕ̃−ϕ‖²/‖ϕ‖²", fontsize=10)
        ax_e.legend(fontsize=8.5, loc="upper right")
        ax_e.grid(True, which="both", alpha=0.3)

        # Recovery
        for lb, rate_list in recs.items():
            ax_r.plot(sample_sizes, [100*r for r in rate_list],
                      label=lb, **STYLE[lb])
        ax_r.axhline(100, color="gray", ls=":", lw=1.2, label="100% (perfect)")
        ax_r.set_ylim(-5, 108)
        ax_r.set_ylabel("Exact support recovery (%)", fontsize=10)
        ax_r.set_xlabel("m — number of value-function evaluations", fontsize=10)
        ax_r.legend(fontsize=8.5, loc="lower right")
        ax_r.grid(alpha=0.3)

        spct = round(100 * ss / n)
        ax_e.set_title(
            f"{'Very sparse' if col==0 else 'Less sparse'}:  "
            f"s={ss},  s/n={spct}%\n"
            f"β_min={bmin:.3f},  threshold ε={eps:.3f}",
            fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = "outputs/sparse_shapley_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")


def print_table(label, errs, recs, sample_sizes, n):
    cols = sample_sizes
    sep  = "─" * (28 + 9 * len(cols))
    print(f"\n{label}")
    hdr  = f"{'Method':<28}" + "".join(f"  m={m//n}n" for m in cols)
    print(sep); print(hdr + "  [median error]"); print(sep)
    for lb in METHODS:
        print(f"{lb:<28}" + "".join(
            f"  {np.median(errs[lb][i]):.4f}" for i in range(len(cols))))
    print(sep); print(hdr + "  [recovery %]"); print(sep)
    for lb in METHODS:
        print(f"{lb:<28}" + "".join(
            f"  {100*recs[lb][i]:5.1f}%" for i in range(len(cols))))
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# §7. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N   = 30
    S_A = 3    # very sparse:  10%
    S_B = 12   # less sparse:  40%

    # Start at n=30 (overdetermined) → up to 10n
    SAMPLE_SIZES = [N * k for k in [1, 2, 3, 4, 6, 8, 10]]

    print("═"*65)
    print(f"  n={N}")
    print(f"  Setting A: s={S_A}  (s/n={100*S_A//N}%,  O(s·log n) ≈ {S_A*log(N):.0f})")
    print(f"  Setting B: s={S_B}  (s/n={100*S_B//N}%,  O(s·log n) ≈ {S_B*log(N):.0f})")
    print(f"  Dense threshold: O(n·log n) ≈ {N*log(N):.0f}")
    print(f"  m values: {SAMPLE_SIZES}")
    print("═"*65)

    print("\n── Setting A: Very sparse (n=30, s=3) ──")
    res_A = run_setting(N, S_A, SAMPLE_SIZES, n_trials=300, seed=42)
    print_table(f"Setting A (n={N}, s={S_A})", res_A[1], res_A[2], SAMPLE_SIZES, N)

    print("\n── Setting B: Less sparse (n=30, s=12) ──")
    res_B = run_setting(N, S_B, SAMPLE_SIZES, n_trials=300, seed=99)
    print_table(f"Setting B (n={N}, s={S_B})", res_B[1], res_B[2], SAMPLE_SIZES, N)

    plot_comparison(res_A, res_B, SAMPLE_SIZES, N, S_A, S_B)
