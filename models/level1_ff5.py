import os
import numpy as np
import pandas as pd

# ============================================================
# Level 1: Fama-French 5-Factor Expected Return Model
# models/level1_ff5.py
#
# ROLE IN THESIS
# --------------
# This is the first non-trivial expected return model on the
# complexity ladder. It replaces the 1/N weight assignment with
# a mean-variance portfolio where expected returns are derived
# from cross-sectional factor exposure (CAPM extended to 5 factors).
#
# STATIC FACTOR MODEL vs. LEVEL 1.5
# -----------------------------------
# This is the *static* specification: factor premia (E[MKT-RF],
# E[SMB], ...) are assumed constant within the estimation window.
# They are estimated as simple historical averages over the trailing
# window and applied uniformly across all stocks. There is NO
# time variation in premia — that extension (e.g. conditioning on
# business cycle state or regime-switching premia) is reserved for
# Level 1.5. This distinction matters for interpreting the source
# of any performance improvement over Level 0.
#
# HOW IT FITS THE PIPELINE
# -------------------------
# Passed as expected_returns_fn to run_backtest(). At each month t:
#   1. build_returns_matrix() provides the T×N returns matrix for
#      the trailing window — covariance estimation uses this too.
#   2. ff5_expected_returns() estimates per-stock betas via OLS and
#      projects onto mean factor premia from the same window.
#   3. optimize_portfolio() takes the resulting μ vector and the
#      shrunk Σ from covariance.py to solve the MV problem.
# ============================================================


# ============================================================
# Module-level cache: FF5 factor data loaded once per session
# ============================================================
_FF5_CACHE = None


def _load_ff5():
    """
    Load Fama-French 5 factor data from data_raw/ff_factors_monthly.csv.
    Returns a DataFrame with columns [year, month, mkt_rf, smb, hml, rmw, cma, rf],
    values in decimal (not percentage).

    Called once and cached in _FF5_CACHE to avoid repeated disk reads
    inside the month-by-month backtest loop.
    """
    global _FF5_CACHE
    if _FF5_CACHE is not None:
        return _FF5_CACHE

    fpath = "data_raw/ff_factors_monthly.csv"
    # File has 4 header rows before the data; first data column is YYYYMM date
    ff = pd.read_csv(fpath, skiprows=4)
    ff.columns = [c.strip().lower().replace("-", "_") for c in ff.columns]
    ff = ff.rename(columns={ff.columns[0]: "date"})

    # Drop non-numeric rows (copyright notice etc.) and annual summary rows.
    # Annual summary rows are 4-digit years (e.g. 1963, 2024); monthly rows
    # are 6-digit YYYYMM values (e.g. 196307). Keep only 6-digit dates.
    ff["date"] = pd.to_numeric(ff["date"], errors="coerce")
    ff = ff[ff["date"] >= 190001].copy()   # YYYYMM >= 190001 → all valid months
    ff["date"] = ff["date"].astype(int)
    ff["year"]  = ff["date"] // 100
    ff["month"] = ff["date"] % 100

    # Convert from percentage to decimal
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0

    ff = ff[["year", "month", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]].copy()
    ff = ff.sort_values(["year", "month"]).reset_index(drop=True)

    _FF5_CACHE = ff
    return ff


# ============================================================
# Main expected return function — plugged into run_backtest()
# ============================================================

FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma"]

# !! TUNING PARAMETER: MIN_OBS=48 — minimum number of non-missing
# !! monthly observations required within the estimation window to
# !! fit betas for a given stock. Stocks below this threshold are
# !! excluded (NaN expected return → dropped by optimizer). With a
# !! 60-month window this requires 80% data coverage, consistent
# !! with the 20% missingness threshold in build_returns_matrix.
# !! Lowering to 36 admits more stocks but fits betas on thin data.
MIN_OBS = 48


def ff5_expected_returns(_master_df, ret_matrix, year, month):
    """
    Estimate Fama-French 5-factor expected returns for the current
    portfolio formation date (year, month).

    Signature matches the expected_returns_fn hook in run_backtest():
        fn(master_df, ret_matrix, year, month) -> pd.Series

    Parameters
    ----------
    master_df : pd.DataFrame
        Full master panel (not used directly here — factor data is
        loaded from data_raw/ff_factors_monthly.csv). Included in
        the signature for API consistency with other models that do
        query the panel (e.g. accounting-signal models).
    ret_matrix : pd.DataFrame, shape (T, N)
        Rolling returns matrix from build_returns_matrix(). Rows are
        (year, month) MultiIndex, columns are permno. Contains the
        trailing estimation window ending at (year, month).
    year, month : int
        Portfolio formation month. Used to verify that factor data
        does not extend beyond this date (no look-ahead).

    Returns
    -------
    mu : pd.Series, length N
        Expected total return for each stock, indexed by permno.
        NaN for stocks with fewer than MIN_OBS non-missing observations.
        These NaN entries are handled downstream by optimize_portfolio
        (stocks with NaN mu are dropped before optimisation).

    Steps
    -----
    1. Load FF5 factor returns for the trailing window.
    2. Estimate per-stock betas via OLS (excess returns on 5 factors).
    3. Compute expected returns as mean_rf + betas @ mean_factor_premia.
    """

    # ============================================================
    # STEP 1 — Load factor returns for the trailing window
    # ============================================================

    ff = _load_ff5()

    # Use a merge to filter FF5 to the exact window months from ret_matrix.
    # This is O(T) and avoids the slow row-wise apply() that was used before.
    # Strictly enforces no look-ahead: only months present in ret_matrix's
    # index (which ends at the formation date) are included.
    window_periods = ret_matrix.index   # MultiIndex: (year, month)
    window_df = pd.DataFrame({
        "year":  window_periods.get_level_values("year"),
        "month": window_periods.get_level_values("month"),
    })
    ff_window = ff.merge(window_df, on=["year", "month"], how="inner")

    if ff_window.empty:
        print(f"  FF5 [{year}-{month:02d}]: No factor data found for window — returning NaN")
        return pd.Series(np.nan, index=ret_matrix.columns)

    # Factor matrix: rows aligned to window months, shape (T_f, 5)
    ff_window = ff_window.set_index(["year", "month"])
    F = ff_window[FACTOR_COLS]     # factor excess returns
    rf_series = ff_window["rf"]    # risk-free rate

    # ============================================================
    # STEP 2 — Estimate rolling betas via OLS (vectorised)
    # ============================================================
    # Model: (r_i - rf)_t = alpha_i + beta_i' * f_t + eps_i,t
    #   where f_t = [MKT-RF, SMB, HML, RMW, CMA]
    #
    # build_returns_matrix() already imputes column means and drops
    # stocks with >20% missing, so ret_matrix arrives with no NaN.
    # Factor data is likewise clean over the window.  This means all
    # N stocks share the same valid-row mask, allowing a single matrix
    # lstsq call (X: T×6, Y: T×N) instead of N separate scalar calls.
    #
    # A per-stock fallback loop handles the rare edge case where a stock
    # still has NaN after alignment (e.g. universe-edge months).
    #
    # The intercept (alpha) is estimated but NOT used in the expected
    # return: the model asserts E[alpha]=0 (pure factor pricing).
    # Including it in OLS avoids biasing the beta estimates.

    ret_aligned = ret_matrix.reindex(ff_window.index)   # (T_f, N)

    rf_values  = rf_series.values                        # (T_f,)
    F_vals     = F.values                                # (T_f, 5)
    ones       = np.ones(len(F))
    X          = np.column_stack([ones, F_vals])         # (T_f, 6)

    # Factor rows that are fully valid (should always be all rows)
    factor_ok  = np.all(np.isfinite(F_vals), axis=1)    # (T_f,)

    Y_excess   = ret_aligned.values - rf_values[:, None] # (T_f, N)

    # Identify stocks where every row in the factor-valid subset is finite
    # and the total observation count meets MIN_OBS.
    n_obs      = np.isfinite(Y_excess[factor_ok]).sum(axis=0)   # (N,)
    full_mask  = (n_obs >= MIN_OBS) & np.all(np.isfinite(Y_excess), axis=0)
    # !! TUNING PARAMETER: MIN_OBS=48 — see module-level comment.

    permnos     = ret_aligned.columns.to_numpy()
    beta_matrix = np.full((5, len(permnos)), np.nan)    # 5 rows × N cols

    # --- Fast path: solve all complete stocks in one lstsq call ---
    if full_mask.sum() > 0:
        X_fit = X[factor_ok]                             # (T_ok, 6)
        Y_fit = Y_excess[np.ix_(factor_ok, full_mask)]   # (T_ok, n_full)
        coef_all, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
        # coef_all: (6, n_full) — row 0 is alpha, rows 1-5 are factor betas
        beta_matrix[:, full_mask] = coef_all[1:]

    # --- Slow fallback: per-stock loop for stocks with partial NaN ---
    partial_mask = (~full_mask) & (n_obs >= MIN_OBS)
    for idx in np.where(partial_mask)[0]:
        y = Y_excess[:, idx]
        valid = factor_ok & np.isfinite(y)
        if valid.sum() < MIN_OBS:
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
            beta_matrix[:, idx] = coef[1:]
        except np.linalg.LinAlgError:
            pass

    # ============================================================
    # STEP 3 — Compute expected returns
    # ============================================================
    # Static factor pricing model:
    #   E[r_i] = E[rf] + beta_i' * E[f]
    #   where E[f] = historical mean of each factor premium in the window.
    #
    # This is the static specification: premia are assumed constant
    # within the window. No business-cycle conditioning or regime
    # switching — that is the Level 1.5 extension.

    mean_rf     = rf_series.mean()
    mean_premia = F.mean().values                        # (5,)

    # mu: (N,) — NaN where betas were not estimated
    mu_arr = mean_rf + beta_matrix.T @ mean_premia       # (N,)
    # Restore NaN for stocks that failed the MIN_OBS filter
    mu_arr[np.all(np.isnan(beta_matrix), axis=0)] = np.nan

    mu = pd.Series(mu_arr, index=permnos, dtype=float).reindex(ret_matrix.columns)

    n_valid   = mu.notna().sum()
    n_total   = len(mu)
    n_nan     = mu.isna().sum()
    print(f"  FF5 [{year}-{month:02d}]: {n_valid}/{n_total} stocks with valid betas "
          f"({n_nan} skipped, MIN_OBS={MIN_OBS})")

    return mu


# ============================================================
# __main__: Full backtest 1976–2024, evaluate and save results
# ============================================================

if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from portfolio.optimizer import run_backtest
    from portfolio.metrics  import evaluate_results, print_benchmark_comparison

    print("\n" + "=" * 60)
    print("LEVEL 1: FAMA-FRENCH 5-FACTOR — 2005 to 2024")
    print("=" * 60)

    # --------------------------------------------------------
    # Load master panel — same file used by Level 0 and all higher models.
    # Universe is defined identically across all models: stocks that
    # pass build_returns_matrix's 20% missingness filter on master_panel.
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Run backtest: Jan 2005 – Dec 2024
    # Aligned with the ML model evaluation window.
    # --------------------------------------------------------
    results = run_backtest(
        master,
        start_year=2005, start_month=1,
        end_year=2024,   end_month=12,
        risk_aversion=1.0,
        window=60,
        cost_bps=10,
        expected_returns_fn=ff5_expected_returns,
    )

    print(f"\nBacktest complete: {len(results)} months computed.")

    # --------------------------------------------------------
    # Evaluate performance
    # --------------------------------------------------------
    summary = evaluate_results(results, rf_series=None)

    print("\n" + "=" * 60)
    print("LEVEL 1 PERFORMANCE SUMMARY (rf = 0)")
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

    print_benchmark_comparison("level1", summary, results_df=results)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/level1_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Level 1 complete.")
