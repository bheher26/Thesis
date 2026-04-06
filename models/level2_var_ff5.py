import os
import warnings
import numpy as np
import pandas as pd

# ============================================================
# Level 2: VAR-Conditioned FF5 Expected Return Model
# models/level2_var_ff5.py
#
# ROLE IN THESIS
# --------------
# Extends Level 1.5 (macro-conditioned ridge premia) by replacing the
# per-factor ridge regressions with a fully-specified dynamic system:
#
#   Layer 1 — Macro state dynamics (VAR):
#       x_t = c + A * x_{t-1} + υ_t,   υ_t ~ (0, Ω)
#
#   Layer 2 — Factor pricing:
#       F_t = c_f + P * x_t + ν_t,     ν_t ~ (0, Γ)
#
#   Layer 3 — Asset returns (identical to Level 1 & 1.5):
#       r_t = α + B * F_t + ε_t,       ε_t ~ (0, Λ)
#
# WHAT CHANGES vs. LEVEL 1.5
# ---------------------------
# Level 1.5:  separate ridge regression per factor on lagged macro
#             — no coupling across factors, no explicit macro dynamics
# Level 2:    joint OLS estimation of P (pricing matrix), explicit A
#             (VAR transition matrix), forward iteration of macro state
#             across the forecast horizon τ
#
# The A matrix enables proper k-step-ahead forecasts:
#   Ê[x_{t+s} | x_t] = c*(I + A + ... + A^{s-1}) + A^s * x_t
#   Ê[F̄_{t→t+τ}] = c_f + P * x̄_{t→t+τ}
#   where x̄ = (1/τ) * Σ_{s=1}^τ Ê[x_{t+s} | x_t]
#
# For monthly rebalancing (τ=1) this simplifies to:
#   Ê[F_{t+1} | x_t] = c_f + P * (c + A * x_t)
#
# NO LOOK-AHEAD GUARANTEE
# -----------------------
# At formation month t (i.e., rebalancing at end of month t):
#   - macro x_t is observed (available at end of t)
#   - VAR and P are estimated on the window [t-59 .. t]
#   - A is used only to forecast x_{t+1}, ..., x_{t+τ} (future)
#   - betas: OLS on returns[t-59 .. t] — identical to Level 1
#
# COVARIANCE
# ----------
# Uses the same Ledoit-Wolf shrinkage as all other levels (consistent
# comparison baseline). The full conditional covariance from the lecture
# (BPΩ P'B' + BΓB' + Λ) is derived and logged but not fed to the
# optimizer, preserving apples-to-apples comparison. It is available
# via _conditional_covariance() for offline analysis.
# ============================================================


# ============================================================
# Module-level caches — loaded once per session
# ============================================================
_FF5_CACHE   = None
_MACRO_CACHE = None


# ============================================================
# Constants
# ============================================================
FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
MACRO_COLS  = ["dp_ratio", "term_spread", "real_short_rate",
               "default_spread", "indpro_growth", "volatility"]

MIN_OBS       = 48    # minimum stock-level observations (identical to Level 1 & 1.5)
# !! TUNING PARAMETER: MIN_OBS=48 — see level1_ff5.py for rationale.

VAR_MIN_OBS   = 36    # minimum (macro, macro_lag) training pairs for the VAR
# !! TUNING PARAMETER: VAR_MIN_OBS=36 — with a 60-month window the max is 59 pairs
# !! (one is lost to lagging). 36 requires ~61% coverage; tighten to 48 for
# !! stricter quality control at the cost of earlier periods falling back.

PRICE_MIN_OBS = 24    # minimum (F, x) training pairs for the pricing matrix P
# !! TUNING PARAMETER: PRICE_MIN_OBS=24 — P has K=6 predictors per factor, so
# !! 24 observations gives roughly 4 observations per parameter. Tighten to 36
# !! for more conservative estimates; relax to 18 only in short back-tests.

FORECAST_HORIZON = 1  # τ — months ahead for the average premia forecast
# !! TUNING PARAMETER: FORECAST_HORIZON=1 matches monthly rebalancing.
# !! Increase to 3 or 12 to reflect a quarterly or annual investment horizon;
# !! the VAR's A matrix will propagate macro dynamics over the longer window.


# ============================================================
# Data loaders
# ============================================================

def _load_ff5():
    """
    Load FF5 factor data from data_raw/ff_factors_monthly.csv.
    Returns a DataFrame sorted by (year, month) in decimal form.
    Identical to Level 1 and Level 1.5.
    """
    global _FF5_CACHE
    if _FF5_CACHE is not None:
        return _FF5_CACHE

    fpath = "data_raw/ff_factors_monthly.csv"
    ff = pd.read_csv(fpath, skiprows=4)
    ff.columns = [c.strip().lower().replace("-", "_") for c in ff.columns]
    ff = ff.rename(columns={ff.columns[0]: "date"})

    ff["date"] = pd.to_numeric(ff["date"], errors="coerce")
    ff = ff[ff["date"] >= 190001].copy()
    ff["date"] = ff["date"].astype(int)
    ff["year"]  = ff["date"] // 100
    ff["month"] = ff["date"] % 100

    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0

    ff = ff[["year", "month", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]].copy()
    ff = ff.sort_values(["year", "month"]).reset_index(drop=True)

    _FF5_CACHE = ff
    return ff


def _load_macro():
    """
    Load macro predictors from data_raw/macro_predictors.csv.

    Returns a DataFrame indexed by (year, month) MultiIndex with columns:
        dp_ratio, term_spread, real_short_rate, default_spread,
        indpro_growth, volatility
    Identical to Level 1.5.
    """
    global _MACRO_CACHE
    if _MACRO_CACHE is not None:
        return _MACRO_CACHE

    fpath = "data_raw/macro_predictors.csv"
    macro = pd.read_csv(fpath, index_col=0, parse_dates=True)
    macro.columns = [c.strip().lower() for c in macro.columns]

    macro["year"]  = macro.index.year
    macro["month"] = macro.index.month
    macro = macro.set_index(["year", "month"])
    macro = macro[MACRO_COLS].apply(pd.to_numeric, errors="coerce")

    _MACRO_CACHE = macro
    return macro


# ============================================================
# VAR estimation: x_t = c + A * x_{t-1} + υ_t
# ============================================================

def _estimate_var(macro_window):
    """
    Estimate a VAR(1) on the macro state vector using OLS
    (equation-by-equation, consistent with the lecture's recommendation).

    Model: x_t = c + A * x_{t-1} + υ_t

    Parameters
    ----------
    macro_window : pd.DataFrame, shape (T, K)
        Macro state observations for the rolling estimation window,
        indexed by (year, month). K = len(MACRO_COLS).

    Returns
    -------
    A_hat : np.ndarray, shape (K, K) or None
        Transition matrix. Row i gives the equation for x_{t,i}:
        x_{t,i} = c_i + A_hat[i, :] @ x_{t-1} + υ_{t,i}
    c_hat : np.ndarray, shape (K,) or None
        Intercept vector.
    Omega_hat : np.ndarray, shape (K, K) or None
        Residual covariance matrix (υ_t covariance).
    None, None, None returned if insufficient data.
    """
    T = len(macro_window)
    K = len(MACRO_COLS)

    # Build X_lag (T-1, K) and Y_cur (T-1, K)
    X_lag = macro_window.iloc[:-1].values.astype(float)   # x_{t-1}
    Y_cur = macro_window.iloc[1:].values.astype(float)    # x_t

    # Keep only rows where all macro values are finite in both lag and current
    valid = (np.all(np.isfinite(X_lag), axis=1) &
             np.all(np.isfinite(Y_cur), axis=1))

    if valid.sum() < VAR_MIN_OBS:
        return None, None, None

    X = X_lag[valid]   # (n_valid, K)
    Y = Y_cur[valid]   # (n_valid, K)
    n = len(X)

    # OLS with intercept: design matrix [1 | X], shape (n, K+1)
    ones   = np.ones((n, 1))
    X_full = np.column_stack([ones, X])   # (n, K+1)

    # Solve Y = X_full @ coef, where coef shape is (K+1, K)
    # Row 0 of coef: intercept c; rows 1..K: A' (each row is alpha for one equation)
    coef, _, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)   # (K+1, K)

    c_hat = coef[0]        # (K,)   — VAR intercept vector
    A_hat = coef[1:].T     # (K, K) — transition matrix

    # Residuals and Ω̂
    Y_pred    = X_full @ coef       # (n, K)
    residuals = Y - Y_pred          # (n, K)
    df        = n - (K + 1)         # degrees of freedom (intercept + K lags per equation)
    Omega_hat = (residuals.T @ residuals) / df   # (K, K)

    return A_hat, c_hat, Omega_hat


# ============================================================
# Pricing matrix estimation: F_t = c_f + P * x_t + ν_t
# ============================================================

def _estimate_pricing_matrix(factor_window, macro_window):
    """
    Estimate the pricing matrix P by OLS regression of each factor
    on the contemporaneous macro state (equation-by-equation).

    Model: F_t = c_f + P * x_t + ν_t

    At formation time t the factor return F_t is unobserved; it is
    estimated for prediction using the VAR forecast of x_{t+1} (see
    _forecast_premia). Within the estimation window all {F_s, x_s}
    pairs are available and no look-ahead arises.

    Parameters
    ----------
    factor_window : pd.DataFrame, shape (T_f, 5)
        FF5 factor returns for the estimation window, indexed by
        (year, month) MultiIndex. Columns: FACTOR_COLS.
    macro_window : pd.DataFrame, shape (T_m, K)
        Macro state variables for the same window, indexed by
        (year, month) MultiIndex. Columns: MACRO_COLS.

    Returns
    -------
    P_hat : np.ndarray, shape (5, K) or None
        Pricing matrix. Row j: loadings of factor j on macro state.
    c_f_hat : np.ndarray, shape (5,) or None
        Factor equation intercepts.
    Gamma_hat : np.ndarray, shape (5, 5) or None
        Residual covariance matrix (ν_t covariance).
    None, None, None returned if insufficient data.
    """
    n_factors = len(FACTOR_COLS)
    K         = len(MACRO_COLS)

    # Inner-join factor and macro on (year, month) — both must be present
    common_idx = factor_window.index.intersection(macro_window.index)
    if len(common_idx) < PRICE_MIN_OBS:
        return None, None, None

    F_vals = factor_window.reindex(common_idx).values.astype(float)   # (T, 5)
    X_vals = macro_window.reindex(common_idx).values.astype(float)    # (T, K)

    # Drop rows with any NaN
    valid = (np.all(np.isfinite(F_vals), axis=1) &
             np.all(np.isfinite(X_vals), axis=1))

    if valid.sum() < PRICE_MIN_OBS:
        return None, None, None

    F = F_vals[valid]    # (n, 5)
    X = X_vals[valid]    # (n, K)
    n = len(X)

    # OLS with intercept: design matrix [1 | X], shape (n, K+1)
    ones   = np.ones((n, 1))
    X_full = np.column_stack([ones, X])   # (n, K+1)

    # coef: (K+1, 5) — row 0 is intercept c_f, rows 1..K are P'
    coef, _, _, _ = np.linalg.lstsq(X_full, F, rcond=None)

    c_f_hat = coef[0]      # (5,)
    P_hat   = coef[1:].T   # (5, K) — each row j is pricing loadings for factor j

    # Γ̂ from residuals
    F_pred    = X_full @ coef       # (n, 5)
    residuals = F - F_pred          # (n, 5)
    df        = n - (K + 1)
    Gamma_hat = (residuals.T @ residuals) / df   # (5, 5)

    return P_hat, c_f_hat, Gamma_hat


# ============================================================
# Multi-step macro forecast via VAR iteration
# ============================================================

def _multistep_macro_forecast(A_hat, c_hat, x_current, tau):
    """
    Iterate the VAR forward to compute the time-averaged macro state
    forecast over a τ-period horizon.

    Ê[x_{t+s} | x_t] = c + A * Ê[x_{t+s-1} | x_t], s = 1, ..., τ
    x̄_{t→t+τ} = (1/τ) * Σ_{s=1}^τ Ê[x_{t+s} | x_t]

    Parameters
    ----------
    A_hat : np.ndarray, shape (K, K)
    c_hat : np.ndarray, shape (K,)
    x_current : np.ndarray, shape (K,)
        Current (observed) macro state x_t.
    tau : int
        Forecast horizon (number of steps).

    Returns
    -------
    x_bar : np.ndarray, shape (K,)
        Time-averaged macro state forecast.
    """
    x     = x_current.copy()
    x_sum = np.zeros_like(x)

    for _ in range(tau):
        x      = c_hat + A_hat @ x   # one-step-ahead VAR forecast
        x_sum += x

    return x_sum / tau


# ============================================================
# Conditional covariance matrix (diagnostic / offline use)
# ============================================================

def _conditional_covariance(B_hat, P_hat, A_hat, Omega_hat, Gamma_hat,
                            Lambda_hat, k):
    """
    Compute the theoretical conditional covariance matrix for the
    cumulative k-period return, following the lecture (Slide 15):

        Var[r_{t→t+k}] =
            Σ_{i=0}^k B P (Σ_{j=0}^i A^i) Ω (Σ_{j=0}^i A^i)' P' B'   ← macro risk
            + k * B Γ B'                                                  ← factor risk
            + k * Λ                                                       ← idiosyncratic

    NOTE: This function is provided for offline analysis / comparison.
    It is NOT fed to the optimizer in the main backtest loop (which uses
    Ledoit-Wolf for a consistent cross-level comparison). To use it,
    override the covariance step in run_backtest or run stand-alone.

    Parameters
    ----------
    B_hat : np.ndarray, shape (N, n_factors)
        Asset-level factor loadings (beta matrix).
    P_hat : np.ndarray, shape (n_factors, K)
        Pricing matrix.
    A_hat : np.ndarray, shape (K, K)
        VAR transition matrix.
    Omega_hat : np.ndarray, shape (K, K)
        VAR residual covariance (macro shock covariance).
    Gamma_hat : np.ndarray, shape (n_factors, n_factors)
        Factor pricing residual covariance (factor shock covariance).
    Lambda_hat : np.ndarray, shape (N, N)
        Diagonal idiosyncratic variance matrix.
    k : int
        Forecast horizon (number of periods).

    Returns
    -------
    Sigma_cond : np.ndarray, shape (N, N)
        Conditional covariance matrix for k-period cumulative return.
    """
    K = A_hat.shape[0]
    N = B_hat.shape[0]

    # ── Systematic covariation due to macroeconomy ────────────────────────────
    # Outer sum over i=0..k; inner sum S_i = Σ_{j=0}^i A^j (partial power sums)
    macro_term = np.zeros((N, N))
    S = np.eye(K)                             # S_0 = A^0 = I
    for i in range(k + 1):
        BP_S   = B_hat @ P_hat @ S            # (N, K)
        macro_term += BP_S @ Omega_hat @ BP_S.T
        if i < k:
            S = S + np.linalg.matrix_power(A_hat, i + 1)   # S_{i+1} = S_i + A^{i+1}

    # ── Systematic covariation due to factors ─────────────────────────────────
    factor_term = k * B_hat @ Gamma_hat @ B_hat.T   # (N, N)

    # ── Idiosyncratic variation ───────────────────────────────────────────────
    idio_term = k * Lambda_hat                       # (N, N)

    return macro_term + factor_term + idio_term


# ============================================================
# Core: estimate VAR-conditioned factor premia
# ============================================================

def estimate_var_premia(factor_window, macro_window, rebal_year, rebal_month,
                        tau=FORECAST_HORIZON):
    """
    Predict FF5 factor premia for the next τ periods using the VAR system.

    Steps
    -----
    1. Estimate VAR(1) on macro window → Â, ĉ, Ω̂
    2. Estimate pricing matrix P on (F, x) window → P̂, ĉ_f, Γ̂
    3. Retrieve current macro state x_t
    4. Forecast x̄_{t→t+τ} by iterating the VAR forward τ steps
    5. Return ĉ_f + P̂ * x̄

    Falls back to Level 1 historical mean if either the VAR or pricing
    matrix estimation fails due to insufficient data.

    Parameters
    ----------
    factor_window : pd.DataFrame, shape (T, 5)
        FF5 factor returns in the estimation window, (year, month) indexed.
    macro_window : pd.DataFrame, shape (T_m, K)
        Macro state variables for the estimation window.
    rebal_year, rebal_month : int
        Formation month. x_t at this date is the forecast input.
    tau : int
        Forecast horizon (default FORECAST_HORIZON = 1).

    Returns
    -------
    premia : np.ndarray, shape (5,)
        Predicted per-factor premia. Falls back to historical mean on failure.
    diagnostics : dict
        Keys: var_ok, pricing_ok, fallback, A_spectral_radius.
    """
    diagnostics = {"var_ok": False, "pricing_ok": False,
                   "fallback": False, "A_spectral_radius": np.nan}

    # Historical mean — used as fallback throughout
    historical_mean = factor_window.mean().values   # (5,)

    # ── Step 1: VAR estimation ────────────────────────────────────────────────
    A_hat, c_hat, Omega_hat = _estimate_var(macro_window)

    if A_hat is None:
        diagnostics["fallback"] = True
        return historical_mean, diagnostics

    diagnostics["var_ok"] = True
    # Spectral radius of A — useful for checking stationarity (should be < 1)
    diagnostics["A_spectral_radius"] = float(
        np.max(np.abs(np.linalg.eigvals(A_hat)))
    )

    # ── Step 2: Pricing matrix estimation ────────────────────────────────────
    P_hat, c_f_hat, Gamma_hat = _estimate_pricing_matrix(factor_window,
                                                          macro_window)

    if P_hat is None:
        diagnostics["fallback"] = True
        return historical_mean, diagnostics

    diagnostics["pricing_ok"] = True

    # ── Step 3: Retrieve current macro state x_t ─────────────────────────────
    macro_keys = set(macro_window.index.tolist())
    rebal_key  = (rebal_year, rebal_month)

    if rebal_key not in macro_keys:
        # Forward-fill: use most recent macro observation at or before rebal date
        all_as_int = (macro_window.index.get_level_values("year") * 12
                      + macro_window.index.get_level_values("month"))
        cutoff     = rebal_year * 12 + rebal_month
        candidates = macro_window.index[all_as_int <= cutoff]
        if len(candidates) == 0:
            diagnostics["fallback"] = True
            return historical_mean, diagnostics
        rebal_key = candidates[-1]
        warnings.warn(
            f"  L2 VAR: macro missing at ({rebal_year}-{rebal_month:02d}), "
            f"forward-filled from {rebal_key}."
        )

    x_current = macro_window.loc[rebal_key].values.astype(float)   # (K,)

    if not np.all(np.isfinite(x_current)):
        diagnostics["fallback"] = True
        return historical_mean, diagnostics

    # ── Step 4: Multi-step macro forecast ────────────────────────────────────
    x_bar = _multistep_macro_forecast(A_hat, c_hat, x_current, tau)

    # ── Step 5: Predicted factor premia ──────────────────────────────────────
    premia = c_f_hat + P_hat @ x_bar   # (5,)

    if not np.all(np.isfinite(premia)):
        diagnostics["fallback"] = True
        return historical_mean, diagnostics

    return premia, diagnostics


# ============================================================
# Drop-in hook for run_backtest()
# ============================================================

def var_ff5_expected_returns(_master_df, ret_matrix, year, month):
    """
    VAR-conditioned FF5 expected returns.

    Drop-in replacement for ff5_expected_returns() (Level 1) and
    ff5_macro_expected_returns() (Level 1.5). Plugged into run_backtest()
    as expected_returns_fn.

    Signature matches the expected_returns_fn hook in run_backtest():
        fn(master_df, ret_matrix, year, month) -> pd.Series

    Parameters
    ----------
    _master_df : pd.DataFrame
        Full master panel — not used here (factor data from FF5 CSV,
        macro data from macro_predictors.csv). Kept for API consistency.
    ret_matrix : pd.DataFrame, shape (T, N)
        Rolling returns matrix from build_returns_matrix(). Rows are
        (year, month) MultiIndex, columns are permno.
    year, month : int
        Portfolio formation month.

    Returns
    -------
    mu : pd.Series, length N
        Expected return per stock (permno-indexed).
        NaN for stocks below MIN_OBS threshold.

    Steps
    -----
    1. Load FF5 factor returns for the trailing window [identical to Level 1].
    2. Estimate per-stock betas via OLS [identical to Level 1 & 1.5].
    3. Estimate VAR(1) on macro state + pricing matrix P [new in Level 2].
    4. Forecast factor premia via multi-step VAR iteration [new in Level 2].
    5. E[r_i] = E[rf] + beta_i @ predicted_premia.
    """

    # ── STEP 1: Load and align factor data ───────────────────────────────────
    ff    = _load_ff5()
    macro = _load_macro()

    window_periods = ret_matrix.index
    window_df = pd.DataFrame({
        "year":  window_periods.get_level_values("year"),
        "month": window_periods.get_level_values("month"),
    })
    ff_window = ff.merge(window_df, on=["year", "month"], how="inner")

    if ff_window.empty:
        print(f"  L2 [{year}-{month:02d}]: No factor data in window — returning NaN")
        return pd.Series(np.nan, index=ret_matrix.columns)

    ff_window = ff_window.set_index(["year", "month"])
    F         = ff_window[FACTOR_COLS]    # (T_f, 5)
    rf_series = ff_window["rf"]           # (T_f,)

    # Macro window: same period slice as the factor window
    macro_window = macro.reindex(ff_window.index)
    # Use all macro available up to and including rebal date for VAR estimation
    # (more macro history improves VAR quality; we slice to the rolling window
    #  to maintain consistency with the estimation window used for betas)

    # ── STEP 2: OLS betas — vectorised, identical to Level 1 & 1.5 ──────────
    ret_aligned = ret_matrix.reindex(ff_window.index)   # (T_f, N)

    rf_values  = rf_series.values                        # (T_f,)
    F_vals     = F.values                                # (T_f, 5)
    ones       = np.ones(len(F))
    X_ols      = np.column_stack([ones, F_vals])         # (T_f, 6)

    factor_ok  = np.all(np.isfinite(F_vals), axis=1)
    Y_excess   = ret_aligned.values - rf_values[:, None] # (T_f, N)

    n_obs     = np.isfinite(Y_excess[factor_ok]).sum(axis=0)   # (N,)
    full_mask = (n_obs >= MIN_OBS) & np.all(np.isfinite(Y_excess), axis=0)

    permnos     = ret_aligned.columns.to_numpy()
    beta_matrix = np.full((5, len(permnos)), np.nan)    # (5, N)

    # Fast path: one lstsq call for all complete stocks
    if full_mask.sum() > 0:
        X_fit = X_ols[factor_ok]
        Y_fit = Y_excess[np.ix_(factor_ok, full_mask)]
        coef_all, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
        beta_matrix[:, full_mask] = coef_all[1:]

    # Slow fallback: per-stock for stocks with partial NaN
    partial_mask = (~full_mask) & (n_obs >= MIN_OBS)
    for idx in np.where(partial_mask)[0]:
        y     = Y_excess[:, idx]
        valid = factor_ok & np.isfinite(y)
        if valid.sum() < MIN_OBS:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(X_ols[valid], y[valid], rcond=None)
            beta_matrix[:, idx] = coef[1:]
        except np.linalg.LinAlgError:
            pass

    # ── STEP 3 & 4: VAR-conditioned factor premia ────────────────────────────
    # This is the block that differs from Level 1 (historical mean) and
    # Level 1.5 (per-factor ridge regression). Here we:
    #   a) Fit a VAR(1) on the macro state variables in the window
    #   b) Fit pricing matrix P (OLS, factor ~ macro, contemporaneous)
    #   c) Iterate the VAR forward τ=FORECAST_HORIZON steps from x_t
    #   d) Apply P to the average forecast macro state to get predicted premia

    predicted_premia, diag = estimate_var_premia(
        factor_window=F,
        macro_window=macro_window,
        rebal_year=year,
        rebal_month=month,
        tau=FORECAST_HORIZON,
    )

    mean_rf = rf_series.mean()

    # ── STEP 5: Expected returns ──────────────────────────────────────────────
    mu_arr = mean_rf + beta_matrix.T @ predicted_premia   # (N,)
    mu_arr[np.all(np.isnan(beta_matrix), axis=0)] = np.nan

    mu = pd.Series(mu_arr, index=permnos, dtype=float).reindex(ret_matrix.columns)

    n_valid = mu.notna().sum()
    n_total = len(mu)
    n_nan   = mu.isna().sum()

    fallback_str = " [FALLBACK→hist mean]" if diag["fallback"] else ""
    rho_str      = (f" | ρ(A)={diag['A_spectral_radius']:.3f}"
                    if diag["var_ok"] else "")
    premia_str   = " ".join(f"{p*100:.2f}" for p in predicted_premia)

    print(
        f"  L2 [{year}-{month:02d}]: {n_valid}/{n_total} stocks "
        f"({n_nan} skipped){fallback_str}{rho_str} | premia [%]: {premia_str}"
    )

    return mu


# ============================================================
# __main__: Full backtest 1980–2024, evaluate and save results
# ============================================================

if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from portfolio.optimizer import run_backtest
    from portfolio.metrics   import evaluate_results, print_benchmark_comparison

    print("\n" + "=" * 60)
    print("LEVEL 2: VAR-CONDITIONED FF5 — 1980 to 2024")
    print("=" * 60)
    print(f"  Forecast horizon τ = {FORECAST_HORIZON} month(s)")
    print(f"  Macro state dim  K = {len(MACRO_COLS)}")
    print(f"  Factors            = {FACTOR_COLS}")

    data_path = "data_clean/master_panel.csv"
    print(f"\nLoading master panel from: {data_path}")
    master = pd.read_csv(data_path, low_memory=False)
    master.columns = [c.strip().lower() for c in master.columns]
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    master["year"]         = pd.to_numeric(master["year"],         errors="coerce").astype(int)
    master["month"]        = pd.to_numeric(master["month"],        errors="coerce").astype(int)
    master["permno"]       = pd.to_numeric(master["permno"],       errors="coerce").astype(int)
    print(f"Master panel loaded: {master.shape}")

    # ── Run backtest: Jan 1980 – Dec 2024 ─────────────────────────────────────
    # Same start date as Level 1 and Level 1.5 for direct comparison.
    # The 60-month estimation window requires macro data back to Jan 1975,
    # which is available in macro_predictors.csv.
    results = run_backtest(
        master,
        start_year=1980, start_month=1,
        end_year=2024,   end_month=12,
        risk_aversion=1.0,
        window=60,
        cost_bps=10,
        expected_returns_fn=var_ff5_expected_returns,
    )

    print(f"\nBacktest complete: {len(results)} months computed.")

    # ── VAR diagnostics ───────────────────────────────────────────────────────
    # Re-run a single month to surface the VAR diagnostics (spectral radius etc.)
    # is available via the printed output during the loop above.

    # ── Performance summary ───────────────────────────────────────────────────
    summary = evaluate_results(results, rf_series=None)

    print("\n" + "=" * 60)
    print("LEVEL 2 PERFORMANCE SUMMARY (rf = 0)")
    print("=" * 60)
    print(f"  Annualised net Sharpe    : {summary['annualized_net_sharpe']:.3f}")
    print(f"  Annualised gross Sharpe  : {summary['annualized_gross_sharpe']:.3f}")
    sortino = summary['annualized_sortino']
    print(f"  Annualised Sortino       : {sortino:.3f}" if sortino == sortino else "  Annualised Sortino       : nan")
    print(f"  Annualised net return    : {summary['annualized_net_return']*100:.2f}%")
    print(f"  Annualised volatility    : {summary['annualized_volatility']*100:.2f}%")
    print(f"  Downside deviation (ann.): {summary['annualized_downside_deviation']:.4f}")
    print(f"  Max drawdown             : {summary['max_drawdown']*100:.2f}%")
    print(f"  Avg monthly turnover     : {summary['avg_monthly_turnover']*100:.1f}%")
    print(f"  Avg monthly cost         : {summary['avg_monthly_cost_bps']:.2f} bps")
    print("=" * 60)

    print_benchmark_comparison("level2", summary)

    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/level2_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Level 2 complete.")
