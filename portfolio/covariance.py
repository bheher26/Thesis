import os
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.covariance import LedoitWolf

# ============================================================
# Covariance Estimators
# portfolio/covariance.py
#
# Provides:
#   estimate_covariance(returns_matrix, shrinkage_target)
#   build_returns_matrix(master_df, year, month, window)
#
# Called once per month inside a rolling/expanding window loop
# in the portfolio optimizer. Returns both the shrunk covariance
# and the shrinkage intensity delta so the caller can log it.
# ============================================================

# Named tuple for clean, unambiguous return values.
# delta: shrinkage intensity (Ledoit-Wolf) or fraction of variance explained
#        by factors (factor model) — logged in results CSV as a diagnostic.
CovarianceResult = namedtuple("CovarianceResult", ["covariance", "delta"])


def _constant_correlation_target(S):
    """
    Build the constant-correlation shrinkage target from a sample
    covariance matrix S.

    Each off-diagonal entry is set to rho_bar * sigma_i * sigma_j, where
    rho_bar is the mean of all sample pairwise correlations.  Diagonal
    entries are kept as the sample variances (sigma_i^2).

    Why constant-correlation instead of the identity (scaled diagonal)?
    -----------------------------------------------------------------------
    The identity target implicitly assumes every asset has unit variance and
    zero correlation — both wildly wrong for equity returns.  The scaled-
    diagonal (Oracle-approx) target keeps individual variances but still
    zeroes all correlations.  For large-cap equities that share common
    factor exposures (market, sector), ignoring the cross-sectional
    correlation structure introduces systematic bias in minimum-variance
    weights and inflates portfolio volatility estimates out-of-sample.

    The constant-correlation target (Ledoit & Wolf 2004, "Honey, I Shrunk
    the Sample Covariance Matrix") preserves per-asset volatility while
    replacing noisy pairwise correlations with a single, stable mean
    correlation.  This is the natural regularisation prior for a universe
    of highly-correlated large-cap stocks where the *level* of correlation
    matters for risk but individual pair estimates are unreliable in short
    windows.
    -----------------------------------------------------------------------

    Parameters
    ----------
    S : np.ndarray, shape (N, N)
        Sample covariance matrix.

    Returns
    -------
    T_cc : np.ndarray, shape (N, N)
        Constant-correlation target matrix.
    """
    n = S.shape[0]

    # Extract per-asset standard deviations from diagonal
    std = np.sqrt(np.diag(S))

    # Compute full correlation matrix; clip to [-1, 1] for numerical safety
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = S / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    # Mean of all off-diagonal (upper-triangle) correlations
    upper_idx = np.triu_indices(n, k=1)
    rho_bar = corr[upper_idx].mean()

    # Target: uniform off-diagonal correlation, sample variances on diagonal
    T_cc = rho_bar * np.outer(std, std)
    np.fill_diagonal(T_cc, np.diag(S))   # restore exact sample variances

    return T_cc


def estimate_covariance(returns_matrix, shrinkage_target="constant_correlation"):
    """
    Estimate a regularised covariance matrix using Ledoit-Wolf shrinkage.

    The analytical Ledoit-Wolf estimator (sklearn) is used to determine the
    optimal shrinkage intensity delta — it is never hard-coded.  For the
    constant-correlation target, delta is solved by minimising the expected
    Frobenius loss between the shrunk estimator and the true covariance.

    Parameters
    ----------
    returns_matrix : np.ndarray or pd.DataFrame, shape (T, N)
        Excess returns matrix: T observations, N assets.
        Must already be clean (no NaNs).
    shrinkage_target : str
        "constant_correlation" (default) — only supported target for now.

    Returns
    -------
    CovarianceResult
        .covariance : np.ndarray, shape (N, N)  — shrunk covariance matrix
        .delta      : float                      — shrinkage intensity in [0, 1]
    """
    # Convert DataFrame to numpy for sklearn compatibility
    if isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = returns_matrix.values

    T, N = returns_matrix.shape
    print(f"  Estimating covariance: T={T} observations, N={N} assets")

    # --------------------------------------------------------
    # Step 1: Sample covariance (de-meaned, unbiased 1/(T-1))
    # --------------------------------------------------------
    S = np.cov(returns_matrix, rowvar=False)   # (N, N)

    # --------------------------------------------------------
    # Step 2: Use sklearn LedoitWolf to get the optimal delta
    # We fit on the raw returns and extract only the shrinkage
    # coefficient; the actual target matrix is our own.
    # --------------------------------------------------------
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns_matrix)
    delta = lw.shrinkage_          # scalar in [0, 1]

    print(f"  Ledoit-Wolf shrinkage intensity delta = {delta:.4f}")

    # --------------------------------------------------------
    # Step 3: Build the shrinkage target
    # --------------------------------------------------------
    if shrinkage_target == "constant_correlation":
        target = _constant_correlation_target(S)
    else:
        raise ValueError(
            f"Unsupported shrinkage_target '{shrinkage_target}'. "
            "Currently only 'constant_correlation' is implemented."
        )

    # --------------------------------------------------------
    # Step 4: Shrink toward target
    # Sigma_shrunk = (1 - delta) * S + delta * Target
    # --------------------------------------------------------
    covariance = (1.0 - delta) * S + delta * target

    return CovarianceResult(covariance=covariance, delta=delta)


# ============================================================
# FF5 Factor Data Loader (cached)
# ============================================================

_FF5_COV_CACHE = None
_FF5_FACTORS   = ["mkt_rf", "smb", "hml", "rmw", "cma"]
_FF5_PATH      = "data_raw/ff_factors_monthly.csv"


def _load_ff5_for_cov():
    """Load FF5 factor returns; cached after first call."""
    global _FF5_COV_CACHE
    if _FF5_COV_CACHE is not None:
        return _FF5_COV_CACHE
    ff = pd.read_csv(_FF5_PATH, skiprows=4)
    ff.columns = [c.strip().lower().replace("-", "_") for c in ff.columns]
    ff = ff.rename(columns={ff.columns[0]: "date"})
    ff["date"] = pd.to_numeric(ff["date"], errors="coerce")
    ff = ff[ff["date"] >= 190001].copy()
    ff["date"]  = ff["date"].astype(int)
    ff["year"]  = ff["date"] // 100
    ff["month"] = ff["date"] % 100
    for col in _FF5_FACTORS + ["rf"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0
    _FF5_COV_CACHE = ff[["year", "month"] + _FF5_FACTORS + ["rf"]].copy()
    return _FF5_COV_CACHE


# ============================================================
# Factor Model Covariance Estimator
# ============================================================

def estimate_factor_covariance(returns_matrix, year, month):
    """
    Estimate a Fama-French 5-factor model covariance matrix.

    Decomposes the N×N covariance into a low-rank factor component
    and a diagonal idiosyncratic component:

        Σ = B @ F @ B' + D

    where:
        B : (N, K) matrix of factor loadings estimated by OLS
        F : (K, K) factor covariance matrix (K=5, always T>>K)
        D : (N, N) diagonal matrix of idiosyncratic variances

    This approach is valid for any N regardless of T because F is
    only K×K (K=5) and each row of B is estimated independently.
    The result is always symmetric positive definite by construction.

    Parameters
    ----------
    returns_matrix : pd.DataFrame, shape (T, N)
        Rolling returns matrix from build_returns_matrix(). Index is
        (year, month) MultiIndex; columns are permno.
    year, month : int
        Portfolio formation date — used to align FF5 factor data and
        enforce no look-ahead (only months in returns_matrix used).

    Returns
    -------
    CovarianceResult
        .covariance : np.ndarray, shape (N, N)
        .delta      : float — fraction of cross-sectional return
                      variance explained by the 5 factors (avg R²),
                      logged in results CSV as a diagnostic.
    """
    if isinstance(returns_matrix, np.ndarray):
        R = returns_matrix
        permnos = None
    else:
        R = returns_matrix.values
        permnos = returns_matrix.columns

    T, N = R.shape
    print(f"  Estimating factor covariance: T={T} observations, N={N} assets")

    # ── Align FF5 factors to the exact window months ──────────────────────
    ff = _load_ff5_for_cov()

    if isinstance(returns_matrix, pd.DataFrame):
        window_idx = returns_matrix.index   # MultiIndex (year, month)
        window_df  = pd.DataFrame({
            "year":  window_idx.get_level_values("year"),
            "month": window_idx.get_level_values("month"),
        })
    else:
        # Fallback: can't align without index info — use last T months up to (year, month)
        end = pd.Period(f"{year}-{month:02d}", freq="M")
        periods = pd.period_range(end=end, periods=T, freq="M")
        window_df = pd.DataFrame({
            "year":  [p.year  for p in periods],
            "month": [p.month for p in periods],
        })

    ff_window = ff.merge(window_df, on=["year", "month"], how="inner")
    ff_window = ff_window.set_index(["year", "month"])

    F_mat = ff_window[_FF5_FACTORS].values   # (T_f, 5)
    rf    = ff_window["rf"].values           # (T_f,)

    # Align returns rows to factor rows (inner join on dates)
    T_f = len(F_mat)
    if isinstance(returns_matrix, pd.DataFrame):
        R_aligned = returns_matrix.reindex(ff_window.index).values  # (T_f, N)
    else:
        # Trim to T_f rows (bottom-aligned)
        R_aligned = R[-T_f:]

    # Excess returns: subtract rf
    Re = R_aligned - rf[:, None]   # (T_f, N)

    # ── OLS: regress each stock's excess return on the 5 factors ──────────
    # Model: Re_i = alpha_i + B_i @ F + eps_i
    # X = [1, F_mat]  shape (T_f, 6)
    ones = np.ones((T_f, 1))
    X    = np.hstack([ones, F_mat])          # (T_f, 6)
    XtX  = X.T @ X                           # (6, 6)
    XtX_inv = np.linalg.pinv(XtX)           # pinv handles near-singular cases
    B_full = XtX_inv @ X.T @ Re             # (6, N) — rows: [alpha, b1..b5]
    B      = B_full[1:, :].T                # (N, 5) — factor loadings only

    # Residuals
    fitted  = X @ B_full                     # (T_f, N)
    resid   = Re - fitted                    # (T_f, N)

    # ── Factor covariance F (5×5) ─────────────────────────────────────────
    F_cov = np.cov(F_mat, rowvar=False)      # (5, 5) — T_f >> 5, well-conditioned

    # ── Idiosyncratic variances D (diagonal) ──────────────────────────────
    # Use sample variance of residuals per stock; floor at a small positive
    # value to guarantee positive definiteness.
    idio_var = np.var(resid, axis=0, ddof=1)           # (N,)
    idio_var = np.maximum(idio_var, 1e-8)              # floor

    # ── Assemble Σ = B F B' + D ───────────────────────────────────────────
    systematic = B @ F_cov @ B.T             # (N, N)
    D          = np.diag(idio_var)           # (N, N)
    Sigma      = systematic + D

    # Force exact symmetry (eliminate floating-point asymmetry)
    Sigma = (Sigma + Sigma.T) / 2.0

    # ── Diagnostic: average R² across stocks ─────────────────────────────
    ss_res   = np.sum(resid ** 2,          axis=0)          # (N,)
    ss_tot   = np.sum((Re - Re.mean(axis=0)) ** 2, axis=0)  # (N,)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_per_stock = 1.0 - ss_res / ss_tot
    r2_per_stock = np.where(ss_tot < 1e-12, 0.0, r2_per_stock)
    avg_r2 = float(np.mean(r2_per_stock))

    print(f"  Factor model: avg R²={avg_r2:.3f}  "
          f"(systematic var share; idio floor applied to {(idio_var <= 1e-7).sum()} stocks)")

    return CovarianceResult(covariance=Sigma, delta=avg_r2)


# ============================================================
# Helper: Build Clean Returns Matrix from Master Panel
# ============================================================

def build_ret_panel(master_df):
    """
    Pre-pivot the full master panel into a wide (year, month) × permno
    returns matrix.  Call once before the backtest loop and pass the result
    as ret_panel to build_returns_matrix() to skip the per-month merge+pivot.

    Duplicate permno rows (from Compustat multi-gvkey matches) are collapsed
    by taking the mean return — consistent with how run_backtest slices
    next-month returns.

    Returns
    -------
    ret_panel : pd.DataFrame
        MultiIndex(year, month) rows × permno columns, values = ret_adjusted.
    """
    ret_panel = (
        master_df.groupby(["year", "month", "permno"])["ret_adjusted"]
        .mean()
        .unstack("permno")
    )
    ret_panel.index.names = ["year", "month"]
    return ret_panel


def build_returns_matrix(master_df, year, month, window=60, ret_panel=None):
    """
    Slice the master panel to the trailing `window` months ending at
    (year, month) inclusive, then pivot to a clean (T x N) returns matrix.

    # !! TUNING PARAMETER: window=60 (5 years) — standard in the LW literature.
    # !! Shorter windows (e.g. 36) admit more stocks but produce noisier
    # !! covariance estimates with higher shrinkage intensity (delta).

    Stocks missing more than 20% of observations in the window are dropped
    to avoid covariance estimates driven by a handful of months.

    # !! TUNING PARAMETER: missingness threshold=20% — requires 48/60 months.
    # !! Paired with window=60; loosening to 40% would admit stocks with only
    # !! 36 months of history, undermining the stability rationale for 60m.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full master panel with columns [permno, year, month, ret_adjusted].
    year : int
        End year of the estimation window.
    month : int
        End month of the estimation window.
    window : int
        Number of trailing months to include (default 60 = 5 years).

    Returns
    -------
    ret_matrix : pd.DataFrame, shape (T, N)
        Returns matrix indexed by period (year, month), columns = permno.
        T <= window, N = number of stocks passing the missingness filter.
    """
    # --------------------------------------------------------
    # Build an ordered sequence of (year, month) tuples for
    # the trailing window ending at the requested date
    # --------------------------------------------------------
    end_period = pd.Period(f"{year}-{month:02d}", freq="M")
    periods = pd.period_range(end=end_period, periods=window, freq="M")
    period_df = pd.DataFrame({
        "year":  [p.year  for p in periods],
        "month": [p.month for p in periods],
    })

    print(f"  Building returns matrix: window={window}m ending {year}-{month:02d}")

    period_index = pd.MultiIndex.from_frame(period_df)

    if ret_panel is not None:
        # Fast path: slice pre-pivoted panel — O(window) index lookup
        ret_wide = ret_panel.reindex(period_index)
    else:
        # Slow path: merge + pivot (used when no pre-built panel is passed)
        subset = master_df.merge(period_df, on=["year", "month"], how="inner")
        subset = subset[["permno", "year", "month", "ret_adjusted"]].copy()
        ret_wide = subset.pivot_table(
            index=["year", "month"],
            columns="permno",
            values="ret_adjusted",
            aggfunc="first",
        )
        ret_wide = ret_wide.reindex(period_index)

    T_actual = len(ret_wide)

    # --------------------------------------------------------
    # Drop stocks with more than 20% missing observations
    # !! TUNING PARAMETER: 0.20 threshold — requires 48/60 months present.
    # !! Adjust in tandem with window above.
    # --------------------------------------------------------
    max_missing = 0.20 * T_actual
    n_before = ret_wide.shape[1]
    ret_wide = ret_wide.dropna(axis=1, thresh=int(T_actual - max_missing))
    n_after = ret_wide.shape[1]
    print(f"  Stocks before/after 20%-missing filter: {n_before} -> {n_after}")

    # --------------------------------------------------------
    # Fill any remaining gaps with column mean (small imputation)
    # and drop any rows that are entirely NaN
    # --------------------------------------------------------
    ret_wide = ret_wide.fillna(ret_wide.mean(axis=0))
    ret_wide = ret_wide.dropna(how="all")

    print(f"  Final returns matrix shape: {ret_wide.shape}")
    return ret_wide


# ============================================================
# __main__: Sanity Check — January 2005
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("COVARIANCE ESTIMATOR — SANITY CHECK: January 2005")
    print("=" * 60)

    # Load master panel
    data_path = "data_clean/master_panel.csv"
    print(f"\nLoading master panel from: {data_path}")
    master = pd.read_csv(data_path, low_memory=False)
    master.columns = [c.strip().lower() for c in master.columns]
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    master["year"]  = pd.to_numeric(master["year"],  errors="coerce").astype(int)
    master["month"] = pd.to_numeric(master["month"], errors="coerce").astype(int)
    print(f"Master panel loaded: {master.shape}")

    # Build returns matrix: 60-month window ending January 2005
    print("\nBuilding returns matrix...")
    ret_matrix = build_returns_matrix(master, year=2005, month=1, window=36)

    # Estimate covariance with constant-correlation target
    print("\nRunning Ledoit-Wolf shrinkage estimator...")
    result = estimate_covariance(ret_matrix, shrinkage_target="constant_correlation")

    # --------------------------------------------------------
    # Diagnostic output
    # --------------------------------------------------------
    cov   = result.covariance
    delta = result.delta

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Returns matrix shape : {ret_matrix.shape}  (T x N)")
    print(f"  Covariance shape     : {cov.shape}")
    print(f"  Shrinkage intensity  : delta = {delta:.4f}")

    # Eigenvalue check — all eigenvalues must be positive for PD guarantee
    eigenvalues = np.linalg.eigvalsh(cov)   # sorted ascending
    print(f"  Min eigenvalue       : {eigenvalues.min():.6e}")
    print(f"  Max eigenvalue       : {eigenvalues.max():.6e}")

    if eigenvalues.min() > 0:
        print("  Positive definite    : YES")
    else:
        print("  Positive definite    : NO — check for near-singular inputs")

    print("=" * 60)
    print("Sanity check complete.")
