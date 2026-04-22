import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# Level 1.5: Macro-Conditioned FF5 Expected Return Model
# models/level_1_5.py
#
# ROLE IN THESIS
# --------------
# Sits between Level 1 (static FF5) and Level 2 on the complexity
# ladder. The stock-level betas are estimated identically to Level 1
# (OLS on 60-month rolling window). The only change is in how the
# factor premia vector E[f] is constructed: instead of a plain
# historical mean, each premium is predicted by a ridge regression
# on 6 lagged macro predictors.
#
# WHAT CHANGES vs. LEVEL 1
# -------------------------
# Level 1:   E[r_i] = E[rf] + beta_i @ mean(F_window)
# Level 1.5: E[r_i] = E[rf] + beta_i @ ridge_predicted_premia(macro_t)
#
# Everything else — universe construction, covariance estimation,
# portfolio optimisation, transaction costs — is unchanged.
#
# NO LOOK-AHEAD
# -------------
# At formation month t:
#   - Factor betas: OLS on returns[t-59 .. t] (same as Level 1)
#   - Ridge training: X[s] = macro[s-1], y[s] = factor_return[s]
#     for s in [t-59 .. t] — macro is lagged 1 month in training
#   - Ridge prediction: X_pred = macro[t] (current state, available
#     at formation time) → predicts factor return at t+1
# ============================================================


# ============================================================
# Module-level caches — loaded once per session
# ============================================================
_FF5_CACHE   = None
_MACRO_CACHE = None

# Diagnostic log: selected ridge alpha per factor per month.
# Populated during backtesting; inspect after run for tuning.
_ALPHA_LOG = []   # list of dicts: {year, month, factor, alpha}


# ============================================================
# Constants
# ============================================================
FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
MACRO_COLS  = ["dp_ratio", "term_spread", "real_short_rate",
               "default_spread", "indpro_growth", "volatility"]

MIN_OBS = 48   # minimum stock-level observations — identical to Level 1
# !! TUNING PARAMETER: MIN_OBS=48 — see level1_ff5.py for rationale.

# !! TUNING PARAMETER: ALPHA_GRID — ridge regularisation candidates.
# !! Searched by TimeSeriesSplit(5) CV. If the selected alpha is
# !! consistently at the boundary (0.01 or 100), extend the grid.
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# !! TUNING PARAMETER: MACRO_MIN_OBS=60 — minimum number of valid
# !! (macro, factor) training pairs required to fit ridge models.
# !! With window=60 and 1-month lag the maximum is 60 pairs.
# !! Falls back to Level 1 historical mean if threshold is not met.
MACRO_MIN_OBS = 60


# ============================================================
# Data loaders
# ============================================================

def _load_ff5():
    """
    Load FF5 factor data. Identical to Level 1 (_load_ff5 in level1_ff5.py).
    Returns a DataFrame sorted by (year, month) in decimal form.
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
    Values are in the native units produced by load_fred.py.
    """
    global _MACRO_CACHE
    if _MACRO_CACHE is not None:
        return _MACRO_CACHE

    fpath = "data_raw/macro_predictors.csv"
    macro = pd.read_csv(fpath, index_col=0, parse_dates=True)
    macro.columns = [c.strip().lower() for c in macro.columns]

    # Convert DatetimeIndex (end-of-month) to (year, month) MultiIndex
    # so alignment with FF5 data is consistent throughout the pipeline.
    macro["year"]  = macro.index.year
    macro["month"] = macro.index.month
    macro = macro.set_index(["year", "month"])
    macro = macro[MACRO_COLS].apply(pd.to_numeric, errors="coerce")

    _MACRO_CACHE = macro
    return macro


# ============================================================
# Core utility: estimate factor premia via ridge regression
# ============================================================

def estimate_factor_premia(
    factor_returns,
    macro_data,
    rebal_year,
    rebal_month,
    window=60,
):
    """
    Predict next-month FF5 factor premia using macro-conditioned ridge
    regression with time-series cross-validation.

    Parameters
    ----------
    factor_returns : pd.DataFrame, shape (T, 5)
        Realized FF5 factor returns for the trailing estimation window,
        indexed by (year, month) MultiIndex. T is typically 60.
        Columns: mkt_rf, smb, hml, rmw, cma (in decimal, not %).
        This is the same F matrix used for OLS beta estimation in Level 1.
    macro_data : pd.DataFrame, shape (M, 6)
        Full macro predictors series, indexed by (year, month) MultiIndex.
        Columns: dp_ratio, term_spread, real_short_rate, default_spread,
                 indpro_growth, volatility.
        M covers the full history available — the function slices internally.
    rebal_year, rebal_month : int
        Portfolio formation date. Used to retrieve the current macro state
        as the prediction input (macro[rebal_year, rebal_month] is available
        at formation time and predicts the next month's factor returns).
    window : int
        Trailing window length (default 60). Used for MACRO_MIN_OBS check.

    Returns
    -------
    premia : np.ndarray, shape (5,)
        Predicted factor premium for each of [mkt_rf, smb, hml, rmw, cma].
        Returns historical mean (Level 1 fallback) if data is insufficient.

    No look-ahead guarantee
    -----------------------
    Training: X[s] = macro[s-1], y[s] = factor_return[s] for each month s
    in the 60-month estimation window. The 1-month lag means that the most
    recent macro predictor used in training is macro[rebal_month - 1].

    Prediction: X_pred = macro[rebal_month]. This is the current macro
    state, which is available at formation time and represents the lagged
    predictor for month rebal_month+1 (the holding period return).

    Ridge / CV
    ----------
    - One Ridge model per factor (5 fits per rebalancing date).
    - Alpha grid: ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0].
    - CV: TimeSeriesSplit(n_splits=5) — no forward leakage.
    - Predictors standardised using StandardScaler fit on training data only.
    - Selected alphas are appended to _ALPHA_LOG for post-hoc diagnostics.
    """
    # ── Build a fast lookup set for macro index membership ────────────────────
    macro_keys = set(macro_data.index.tolist())

    # ── Construct training pairs: X[s] = macro[s-1], y[s] = factor[s] ────────
    factor_idx = factor_returns.index.tolist()   # list of (year, month) tuples

    X_rows = []
    for y, m in factor_idx:
        # Lagged macro key: one month before the factor observation
        lag_key = (y - 1, 12) if m == 1 else (y, m - 1)
        if lag_key in macro_keys:
            X_rows.append(macro_data.loc[lag_key].values.astype(float))
        else:
            X_rows.append(np.full(len(MACRO_COLS), np.nan))

    X_train_raw = np.array(X_rows, dtype=float)   # (T, 6)
    y_train_raw = factor_returns.values            # (T, 5)

    # Drop rows where any macro predictor is NaN (missing data)
    valid_rows  = np.all(np.isfinite(X_train_raw), axis=1)
    X_train     = X_train_raw[valid_rows]
    y_train     = y_train_raw[valid_rows]

    # Fallback to Level 1 historical mean if insufficient training data
    if len(X_train) < MACRO_MIN_OBS:
        warnings.warn(
            f"  L1.5 [{rebal_year}-{rebal_month:02d}]: only {len(X_train)} macro training "
            f"obs (need {MACRO_MIN_OBS}) — using historical mean premia (Level 1 fallback)."
        )
        return factor_returns.mean().values

    # ── Retrieve prediction input: current macro state ────────────────────────
    rebal_key = (rebal_year, rebal_month)
    if rebal_key not in macro_keys:
        # Forward-fill: use the most recent available macro (up to 1 month lag)
        all_as_int = (macro_data.index.get_level_values("year") * 12
                      + macro_data.index.get_level_values("month"))
        cutoff     = rebal_year * 12 + rebal_month
        candidates = macro_data.index[all_as_int <= cutoff]
        if len(candidates) == 0:
            warnings.warn(f"  L1.5: No macro data at or before {rebal_key} — NaN premia.")
            return np.full(5, np.nan)
        rebal_key = candidates[-1]
        warnings.warn(
            f"  L1.5: Macro missing at ({rebal_year}-{rebal_month:02d}), "
            f"forward-filled from {rebal_key}."
        )

    X_pred_raw = macro_data.loc[rebal_key].values.astype(float).reshape(1, -1)  # (1, 6)

    if not np.all(np.isfinite(X_pred_raw)):
        warnings.warn(
            f"  L1.5: NaN in macro predictors at {rebal_key} — "
            "using historical mean premia (Level 1 fallback)."
        )
        return factor_returns.mean().values

    # ── Standardize predictors (fit scaler on training data only) ────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)    # (T_valid, 6)
    X_pred   = scaler.transform(X_pred_raw)     # (1, 6)

    # ── Fit one Ridge per factor with TimeSeriesSplit CV ─────────────────────
    tscv   = TimeSeriesSplit(n_splits=5)
    premia = np.full(5, np.nan)

    for j, factor in enumerate(FACTOR_COLS):
        y = y_train[:, j]
        if not np.all(np.isfinite(y)):
            # Missing factor data — fall back to historical mean for this factor
            premia[j] = factor_returns[factor].mean()
            continue

        ridge_cv = RidgeCV(
            alphas=ALPHA_GRID,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
        ridge_cv.fit(X_scaled, y)
        premia[j] = ridge_cv.predict(X_pred)[0]

        # Log selected alpha for diagnostics
        _ALPHA_LOG.append({
            "year": rebal_year, "month": rebal_month,
            "factor": factor, "alpha": ridge_cv.alpha_,
        })

    return premia


# ============================================================
# Drop-in hook for run_backtest()
# ============================================================

def ff5_macro_expected_returns(_master_df, ret_matrix, year, month):
    """
    Fama-French 5-factor expected returns with macro-conditioned premia.

    Drop-in replacement for ff5_expected_returns() from Level 1.
    Plugged into run_backtest() as expected_returns_fn.

    Signature matches the expected_returns_fn hook in run_backtest():
        fn(master_df, ret_matrix, year, month) -> pd.Series

    Parameters
    ----------
    _master_df : pd.DataFrame
        Full master panel. Not used here (factor data from FF5 CSV,
        macro data from macro_predictors.csv). Kept for API consistency.
    ret_matrix : pd.DataFrame, shape (T, N)
        Rolling returns matrix from build_returns_matrix(). Rows are
        (year, month) MultiIndex. Columns are permno.
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
    2. Estimate per-stock betas via OLS [identical to Level 1].
    3. Predict factor premia using macro-conditioned ridge regression
       [replaces Level 1's plain historical mean].
    4. Compute E[r_i] = E[rf] + beta_i @ predicted_premia.
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
        print(f"  L1.5 [{year}-{month:02d}]: No factor data in window — returning NaN")
        return pd.Series(np.nan, index=ret_matrix.columns)

    ff_window = ff_window.set_index(["year", "month"])
    F         = ff_window[FACTOR_COLS]    # (T_f, 5)
    rf_series = ff_window["rf"]           # (T_f,)

    # ── STEP 2: OLS betas — vectorised, identical to Level 1 ─────────────────
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
        beta_matrix[:, full_mask] = coef_all[1:]   # rows 1-5 are factor betas

    # Slow fallback: per-stock for stocks with partial NaN
    partial_mask = (~full_mask) & (n_obs >= MIN_OBS)
    for idx in np.where(partial_mask)[0]:
        y = Y_excess[:, idx]
        valid = factor_ok & np.isfinite(y)
        if valid.sum() < MIN_OBS:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(X_ols[valid], y[valid], rcond=None)
            beta_matrix[:, idx] = coef[1:]
        except np.linalg.LinAlgError:
            pass

    # ── STEP 3: Macro-conditioned factor premia ───────────────────────────────
    # This is the only step that differs from Level 1.
    # Level 1 uses: mean_premia = F.mean().values
    # Level 1.5 uses: ridge regression on lagged macro predictors.
    predicted_premia = estimate_factor_premia(
        factor_returns=F,
        macro_data=macro,
        rebal_year=year,
        rebal_month=month,
        window=len(ff_window),
    )

    mean_rf = rf_series.mean()

    # ── STEP 4: Expected returns ──────────────────────────────────────────────
    mu_arr = mean_rf + beta_matrix.T @ predicted_premia   # (N,)
    mu_arr[np.all(np.isnan(beta_matrix), axis=0)] = np.nan

    mu = pd.Series(mu_arr, index=permnos, dtype=float).reindex(ret_matrix.columns)

    n_valid = mu.notna().sum()
    n_total = len(mu)
    n_nan   = mu.isna().sum()
    print(
        f"  L1.5 [{year}-{month:02d}]: {n_valid}/{n_total} stocks "
        f"({n_nan} skipped) | premia [%]: "
        + " ".join(f"{p*100:.2f}" for p in predicted_premia)
    )

    return mu


# ============================================================
# __main__: Full backtest 1980–2024, evaluate and save results
# ============================================================

if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from portfolio.optimizer import run_backtest
    from portfolio.metrics  import evaluate_results, print_benchmark_comparison

    print("\n" + "=" * 60)
    print("LEVEL 1.5: FF5 + MACRO-CONDITIONED PREMIA — 2005 to 2024")
    print("=" * 60)

    try:
        from portfolio.config import PANEL_PATH
    except ImportError:
        PANEL_PATH = "data_clean/master_panel_v2.csv"
    data_path = PANEL_PATH
    print(f"\nLoading master panel from: {data_path}")
    master = pd.read_csv(data_path, low_memory=False)
    master.columns = [c.strip().lower() for c in master.columns]
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    master["year"]         = pd.to_numeric(master["year"],         errors="coerce").astype(int)
    master["month"]        = pd.to_numeric(master["month"],        errors="coerce").astype(int)
    master["permno"]       = pd.to_numeric(master["permno"],       errors="coerce").astype(int)
    print(f"Master panel loaded: {master.shape}")

    results = run_backtest(
        master,
        start_year=2005, start_month=1,
        end_year=2024,   end_month=12,
        risk_aversion=1.0,
        window=60,
        cost_bps=10,
        expected_returns_fn=ff5_macro_expected_returns,
    )

    print(f"\nBacktest complete: {len(results)} months computed.")

    # ── Ridge alpha diagnostics ────────────────────────────────────────────────
    if _ALPHA_LOG:
        alpha_df = pd.DataFrame(_ALPHA_LOG)
        print("\nRidge alpha selection (mean across all months per factor):")
        print(alpha_df.groupby("factor")["alpha"].agg(["mean", "median", "min", "max"])
              .round(3).to_string())
        most_common = alpha_df.groupby("factor")["alpha"].agg(
            lambda s: s.value_counts().idxmax()
        )
        print("\nMost frequently selected alpha per factor:")
        print(most_common.to_string())

    # ── Performance summary ────────────────────────────────────────────────────
    summary = evaluate_results(results, rf_series=None)

    print("\n" + "=" * 60)
    print("LEVEL 1.5 PERFORMANCE SUMMARY (rf = 0)")
    print("=" * 60)
    print(f"  Annualised net Sharpe    : {summary['annualized_net_sharpe']:.3f}")
    print(f"  Annualised gross Sharpe  : {summary['annualized_gross_sharpe']:.3f}")
    sortino = summary['annualized_sortino']
    print(f"  Annualised Sortino       : {sortino:.3f}" if sortino == sortino else "  Annualised Sortino       : nan")
    print(f"  Annualised gross return  : {summary['annualized_gross_return']*100:.2f}%")
    print(f"  Annualised net return    : {summary['annualized_net_return']*100:.2f}%")
    print(f"  Annualised volatility    : {summary['annualized_volatility']*100:.2f}%")
    print(f"  Downside deviation (ann.): {summary['annualized_downside_deviation']:.4f}")
    print(f"  Max drawdown             : {summary['max_drawdown']*100:.2f}%")
    print(f"  Avg monthly turnover     : {summary['avg_monthly_turnover']*100:.1f}%")
    print(f"  Avg monthly cost         : {summary['avg_monthly_cost_bps']:.2f} bps")
    print("=" * 60)

    print_benchmark_comparison("level_1_5", summary, results_df=results)

    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/level_1_5_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Level 1.5 complete.")
