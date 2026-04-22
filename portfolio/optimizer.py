import os
import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sp
from tqdm import tqdm


from portfolio.covariance import build_returns_matrix, build_ret_panel, build_mktcap_panel, estimate_factor_covariance

print("\nPortfolio Optimizer Module")
print("Running from directory:", os.getcwd())

# ============================================================
# 1) Mean-Variance Optimizer
# ============================================================

def optimize_portfolio(expected_returns, covariance, risk_aversion=1.0,
                       max_weight=0.05):
    """
    Mean-variance portfolio optimization (long-only, fully invested).

    Solves:
        min  (risk_aversion / 2) * w' * Sigma * w  -  mu' * w
        s.t. sum(w) = 1
             0 <= w_i <= max_weight  (long only, concentration cap)

    Parameters
    ----------
    expected_returns : pd.Series, length N
        Expected return for each asset. Index must be asset identifiers
        (e.g. permno). Typically the trailing-window sample mean.
    covariance : np.ndarray, shape (N, N)
        FF5 factor model covariance matrix from estimate_factor_covariance().
        Must be aligned with expected_returns index order.
    risk_aversion : float
        Risk aversion coefficient lambda. Higher = more conservative.
        # !! TUNING PARAMETER: risk_aversion=1.0 — scale up to penalise
        # !! variance more heavily, scale down to chase return.
    max_weight : float
        Maximum weight allowed in any single asset.
        # !! TUNING PARAMETER: max_weight=0.05 (5%) — ensures at least 20
        # !! stocks are held at all times, providing meaningful diversification.

    Returns
    -------
    weights : pd.Series
        Optimal portfolio weights indexed by asset identifier.
        Sums to 1.0, all entries in [0, max_weight].
    """
    N = len(expected_returns)
    mu = expected_returns.values
    Sigma = covariance

    # --------------------------------------------------------
    # OSQP formulation:
    #   minimize  (1/2) w' P w + q' w
    #   subject to  l <= A w <= u
    #
    # P = risk_aversion * Sigma  (upper triangular sparse)
    # q = -mu
    # A = [1...1 ; I_N]          (sum-to-one + box constraints)
    # l = [1,  0, ..., 0]
    # u = [1,  max_weight, ..., max_weight]
    # --------------------------------------------------------
    P = sp.csc_matrix(risk_aversion * np.triu(Sigma))
    q = -mu

    ones_row = sp.csc_matrix(np.ones((1, N)))
    A = sp.vstack([ones_row, sp.eye(N)], format="csc")

    l = np.concatenate([[1.0], np.zeros(N)])
    u = np.concatenate([[1.0], np.full(N, max_weight)])

    solver = osqp.OSQP()
    solver.setup(
        P, q, A, l, u,
        warm_starting=True,
        verbose=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=10000,
        polish=True,
    )
    result = solver.solve()

    w0 = np.ones(N) / N
    if result.info.status in ("solved", "solved_inaccurate"):
        w = np.clip(result.x, 0.0, max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            print("  WARNING: OSQP returned zero weights — falling back to equal-weight.")
            w = w0
        if result.info.status == "solved_inaccurate":
            print("  WARNING: OSQP solved_inaccurate — solution may be approximate.")
    else:
        print(f"  WARNING: OSQP failed ({result.info.status}) — falling back to equal-weight.")
        w = w0

    return pd.Series(w, index=expected_returns.index)


# ============================================================
# 2) Turnover Calculation
# ============================================================

def calculate_turnover(weights_prev, weights_new):
    """
    Compute one-way portfolio turnover between two weight vectors.

    Turnover = 0.5 * sum(|w_new_i - w_prev_i|)

    Assets present in one period but not the other are treated as
    entering/exiting at zero weight (new positions or full liquidations).

    Parameters
    ----------
    weights_prev : pd.Series or None
        Portfolio weights at end of previous period. Pass None for the
        first period (no prior holdings), which returns turnover = 1.0
        (full portfolio construction from cash).
    weights_new : pd.Series
        Portfolio weights for the current period.

    Returns
    -------
    turnover : float
        One-way turnover in [0, 1]. Multiply by 2 for round-trip.
    """
    if weights_prev is None:
        # First period: entering all positions from cash = full turnover
        return 1.0

    # Align on union of both universes; fill missing with 0
    all_assets = weights_prev.index.union(weights_new.index)
    prev = weights_prev.reindex(all_assets, fill_value=0.0)
    new  = weights_new.reindex(all_assets, fill_value=0.0)

    turnover = 0.5 * np.abs(new - prev).sum()
    return float(turnover)


# ============================================================
# 3) Transaction Cost Adjustment
# ============================================================

def apply_transaction_costs(gross_return, turnover, cost_bps=10):
    """
    Deduct transaction costs from the gross portfolio return.

    Cost model: flat cost per unit of one-way turnover.
    net_return = gross_return - turnover * cost_bps / 10_000

    Parameters
    ----------
    gross_return : float
        Portfolio gross return for the period.
    turnover : float
        One-way turnover for the period (from calculate_turnover).
    cost_bps : float
        Round-trip transaction cost in basis points.
        # !! TUNING PARAMETER: cost_bps=10 (0.10%) — reasonable for
        # !! large-cap US equities. Raise to 20-30 bps for a conservative
        # !! estimate or to stress-test strategy capacity.

    Returns
    -------
    net_return : float
    cost : float
        Cost deducted (for logging).
    """
    cost = turnover * cost_bps / 10_000
    net_return = gross_return - cost
    return net_return, cost


# ============================================================
# 4) Monthly Portfolio Construction Loop
# ============================================================

def run_backtest(master_df, start_year, start_month, end_year, end_month,
                 risk_aversion=1.0, window=60, cost_bps=10,
                 expected_returns_fn=None, weights_fn=None,
                 max_turnover=None):
    """
    Roll the mean-variance optimizer month by month from start to end.

    At each month t:
      1. Build returns matrix from trailing window
      2. If weights_fn is set: assign weights directly, skip steps 3-4
         Else: estimate covariance, compute expected returns, optimise
      3. Compute realised gross return using actual t+1 returns
      4. Apply transaction costs
      5. Log results

    Parameters
    ----------
    master_df : pd.DataFrame
        Full master panel (permno, year, month, ret_adjusted).
    start_year, start_month : int
        First month to construct a portfolio for.
    end_year, end_month : int
        Last month to construct a portfolio for.
    risk_aversion : float
        Passed through to optimize_portfolio. Ignored if weights_fn is set.
    window : int
        Trailing window for covariance and expected return estimation.
    cost_bps : float
        Transaction cost in basis points.
    expected_returns_fn : callable or None
        Function with signature:
            fn(master_df, ret_matrix, year, month) -> pd.Series
        Returns expected return per asset (permno-indexed). Passed to
        optimize_portfolio. If None, defaults to trailing sample mean.
        Ignored if weights_fn is set.

        Example models:
            sample_mean   — ret_matrix.mean(axis=0)
            capm          — beta * E[mkt] using FF mkt_rf factor
            ff5_fitted    — cross-sectional regression on FF5 loadings
            ml_predicted  — any model that outputs a permno-indexed Series

    weights_fn : callable or None
        Function with signature:
            fn(ret_matrix) -> pd.Series
        Returns portfolio weights directly (permno-indexed, sums to 1).
        Bypasses covariance estimation and optimize_portfolio entirely.
        Use only for benchmarks that have no optimisation step (e.g. 1/N).
        Takes priority over expected_returns_fn if both are provided.
    max_turnover : float or None
        If set, caps one-way monthly turnover at this fraction (e.g. 0.35 = 35%).
        When the unconstrained optimal portfolio would exceed the cap, weights are
        blended linearly toward the previous period's weights until the constraint
        binds:  w_final = w_prev + clip * (w_optimal - w_prev)
        where clip = max_turnover / unconstrained_turnover.
        Only applied to the optimised path (not weights_fn benchmarks).
        Defaults to None (no cap).

    Returns
    -------
    results : pd.DataFrame
        Month-by-month log with columns:
        year, month, n_assets, delta, gross_return, turnover, cost, net_return
        delta is NaN for months where weights_fn bypasses covariance estimation.
    """
    # --------------------------------------------------------
    # Build ordered list of (year, month) periods to iterate
    # --------------------------------------------------------
    all_periods = pd.period_range(
        start=f"{start_year}-{start_month:02d}",
        end=f"{end_year}-{end_month:02d}",
        freq="M",
    )

    records = []
    weights_prev = None

    # Pre-pivot the master panel once so each monthly call to
    # build_returns_matrix does an index slice instead of a merge+pivot.
    print("  Pre-building wide returns panel...")
    ret_panel = build_ret_panel(master_df)
    print(f"  Returns panel shape: {ret_panel.shape}")
    print("  Pre-building market cap panel...")
    mktcap_panel = build_mktcap_panel(master_df)
    if mktcap_panel is not None:
        print(f"  Market cap panel shape: {mktcap_panel.shape}")
    else:
        print("  Market cap panel: not available (market_cap column missing)")

    print(f"\nRunning backtest: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    print(f"  risk_aversion={risk_aversion}, window={window}m, cost={cost_bps}bps")
    print("-" * 60)

    pbar = tqdm(all_periods, desc="Backtest", unit="month", ncols=80)
    for period in pbar:
        y, m = period.year, period.month
        pbar.set_postfix({"month": f"{y}-{m:02d}"})
        print(f"\n[{y}-{m:02d}]")

        # --------------------------------------------------------
        # Step 1: Build returns matrix for estimation window
        # --------------------------------------------------------
        try:
            ret_matrix = build_returns_matrix(master_df, y, m, window=window,
                                              ret_panel=ret_panel,
                                              mktcap_panel=mktcap_panel)
        except Exception as e:
            print(f"  SKIP — build_returns_matrix failed: {e}")
            continue

        if ret_matrix.shape[1] < 10:
            print(f"  SKIP — too few assets ({ret_matrix.shape[1]}) after missingness filter")
            continue

        # --------------------------------------------------------
        # Steps 2-4: Either assign weights directly (weights_fn) or run
        # the full covariance → expected returns → optimiser pipeline
        # --------------------------------------------------------
        # Universe at this date = columns of ret_matrix (identical for all
        # models that use this same master_df and window).
        print(f"  Universe: {ret_matrix.shape[1]} stocks (build_returns_matrix)")

        if weights_fn is not None:
            # Bypass covariance estimation and optimizer entirely.
            # Used for benchmarks with no optimisation step (e.g. 1/N).
            weights = weights_fn(ret_matrix)
            delta = float("nan")
            print(f"  Weights assigned directly: {len(weights)} assets (no optimiser)")
        else:
            # Step 2: Estimate factor model covariance (FF5-based)
            cov_result = estimate_factor_covariance(ret_matrix, y, m)
            Sigma = cov_result.covariance
            delta = cov_result.delta   # avg factor R² across stocks

            # Step 3: Expected returns — provided model or trailing sample mean
            if expected_returns_fn is None:
                mu = ret_matrix.mean(axis=0)   # pd.Series indexed by permno
            else:
                mu = expected_returns_fn(master_df, ret_matrix, y, m)

            # Drop stocks with NaN expected returns (e.g. failed MIN_OBS
            # in FF5) before passing to the optimiser. Subset Sigma to match.
            valid = mu.notna()
            if not valid.all():
                n_dropped = int((~valid).sum())
                print(f"  Dropping {n_dropped} stocks with NaN mu before optimisation")
                mu = mu[valid]
                valid_idx = [i for i, c in enumerate(ret_matrix.columns)
                             if c in mu.index]
                Sigma = Sigma[np.ix_(valid_idx, valid_idx)]

            if len(mu) < 10:
                print(f"  SKIP — too few stocks with valid expected returns ({len(mu)})")
                continue

            # Step 4: Optimise weights
            weights = optimize_portfolio(mu, Sigma, risk_aversion=risk_aversion)
            print(f"  Active positions: {(weights > 1e-4).sum()} / {len(weights)}")

            # Step 4b: Turnover cap — blend toward previous weights if needed
            if max_turnover is not None and weights_prev is not None:
                unconstrained_to = calculate_turnover(weights_prev, weights)
                if unconstrained_to > max_turnover:
                    clip = max_turnover / unconstrained_to
                    all_assets = weights_prev.index.union(weights.index)
                    w_prev_a = weights_prev.reindex(all_assets, fill_value=0.0)
                    w_new_a  = weights.reindex(all_assets, fill_value=0.0)
                    w_blended = w_prev_a + clip * (w_new_a - w_prev_a)
                    w_blended = w_blended.clip(lower=0.0)
                    w_sum = w_blended.sum()
                    if w_sum > 0:
                        w_blended = w_blended / w_sum
                    weights = w_blended[w_blended > 1e-8]
                    print(f"  Turnover cap: {unconstrained_to*100:.1f}% → "
                          f"{calculate_turnover(weights_prev, weights)*100:.1f}% "
                          f"(clip={clip:.3f})")

        # --------------------------------------------------------
        # Step 5: Realised gross return in month t+1
        # Slice actual next-month returns from master panel
        # --------------------------------------------------------
        next_period = period + 1
        ny, nm = next_period.year, next_period.month

        # groupby permno to collapse duplicates from the Compustat merge
        # (one permno can map to multiple gvkeys; take the mean return)
        next_ret = (
            master_df[(master_df["year"] == ny) & (master_df["month"] == nm)]
            .groupby("permno")["ret_adjusted"]
            .mean()
        )

        # Align weights to assets with available next-month returns
        common = weights.index.intersection(next_ret.index)
        if len(common) == 0:
            print(f"  SKIP — no overlapping assets with next-month returns")
            continue

        w_aligned = weights.reindex(common)
        w_aligned = w_aligned / w_aligned.sum()   # renormalise after alignment
        r_aligned = next_ret.reindex(common)

        # Drop stocks with NaN next-month return (missing CRSP data / delistings
        # with no recorded return) and renormalise weights over the remainder.
        valid_ret = r_aligned.notna()
        if not valid_ret.all():
            n_nan_ret = int((~valid_ret).sum())
            print(f"  Dropping {n_nan_ret} stock(s) with NaN next-month return — renormalising weights")
            w_aligned = w_aligned[valid_ret]
            r_aligned = r_aligned[valid_ret]
            if w_aligned.sum() == 0:
                print(f"  SKIP — all portfolio weight on NaN-return stocks")
                continue
            w_aligned = w_aligned / w_aligned.sum()

        gross_return = float(w_aligned @ r_aligned)

        # --------------------------------------------------------
        # Step 6: Turnover and transaction costs
        # --------------------------------------------------------
        turnover = calculate_turnover(weights_prev, weights)
        net_return, cost = apply_transaction_costs(gross_return, turnover, cost_bps)

        print(f"  Gross return: {gross_return*100:.2f}%  |  "
              f"Turnover: {turnover*100:.1f}%  |  "
              f"Net return: {net_return*100:.2f}%")

        records.append({
            "year":             y,
            "month":            m,
            "n_assets":         len(weights),
            "n_active":         int((weights > 1e-4).sum()),
            "delta":            None if weights_fn is not None else round(delta, 4),
            "gross_return":     round(gross_return, 6),
            "turnover":         round(turnover, 4),
            "cost":             round(cost, 6),
            "net_return":       round(net_return, 6),
        })

        # Update previous weights for next iteration's turnover calculation
        weights_prev = weights

    pbar.close()
    results = pd.DataFrame(records)
    return results


# ============================================================
# __main__: Test Run — 2005 to 2007
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("OPTIMIZER SANITY CHECK: Jan 2005 — Dec 2007")
    print("=" * 60)

    # Load master panel
    data_path = "data_clean/master_panel.csv"
    print(f"\nLoading master panel from: {data_path}")
    master = pd.read_csv(data_path, low_memory=False)
    master.columns = [c.strip().lower() for c in master.columns]
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    master["year"]  = pd.to_numeric(master["year"],  errors="coerce").astype(int)
    master["month"] = pd.to_numeric(master["month"], errors="coerce").astype(int)
    master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype(int)
    print(f"Master panel loaded: {master.shape}")

    # Run backtest over a short window for testing
    results = run_backtest(
        master,
        start_year=2005, start_month=1,
        end_year=2007,   end_month=12,
        risk_aversion=1.0,
        window=60,
        cost_bps=10,
        expected_returns_fn=None,   # swap in e.g. ff5_fitted or ml_predicted
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Months computed       : {len(results)}")
    print(f"  Avg assets per month  : {results['n_assets'].mean():.0f}")
    print(f"  Avg factor R²         : {results['delta'].mean():.4f}")
    print(f"  Avg gross return      : {results['gross_return'].mean()*100:.2f}% / month")
    print(f"  Avg turnover          : {results['turnover'].mean()*100:.1f}% / month")
    print(f"  Avg transaction cost  : {results['cost'].mean()*10000:.1f} bps / month")
    print(f"  Avg net return        : {results['net_return'].mean()*100:.2f}% / month")

    # Cumulative net return
    cum_net = (1 + results["net_return"]).prod() - 1
    print(f"  Cumulative net return : {cum_net*100:.1f}%")

    # Save results
    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/sample_mean_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("=" * 60)
    print("Optimizer test complete.")
