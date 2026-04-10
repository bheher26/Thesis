import os
import numpy as np
import pandas as pd

# ============================================================
# Level 0: Equal-Weight Benchmark
# models/level0_equal_weight.py
#
# ROLE IN THESIS
# --------------
# This is the naive diversification benchmark from DeMiguel, Garlappi,
# and Uppal (2009, RFS) — "Optimal Versus Naive Diversification: How
# Inefficient Is the 1/N Portfolio Strategy?"
#
# It represents zero forecasting complexity: no expected return model,
# no covariance estimation, no optimisation. Every stock in the current
# universe receives an identical weight of 1/N.
#
# Interpretation on the model complexity ladder:
#   - Any model that cannot beat this benchmark net of transaction costs
#     has no practical value, regardless of its in-sample fit or theoretical
#     motivation. DeMiguel et al. showed that most mean-variance variants
#     fail this test out-of-sample due to estimation error. This benchmark
#     is therefore a high bar, not a low one.
#   - It also serves as the turnover baseline. Because weights are uniform
#     and the universe is relatively stable month to month, turnover arises
#     only from universe reconstitution (stocks entering/exiting the top-500),
#     not from changing forecasts. This gives the lowest possible
#     implementation cost and isolates the cost drag of more complex models.
#
# Usage: pass as expected_returns_fn to run_backtest(). The optimizer still
# runs but with a flat mu, so the solution collapses to equal weight (subject
# to the max_weight cap). Any deviation from 1/N in the weights is due
# solely to the cap binding on small universes — not a forecast.
# ============================================================


def equal_weight(ret_matrix):
    """
    Return uniform 1/N portfolio weights across the current universe.

    Signature matches the weights_fn hook in run_backtest():
        fn(ret_matrix) -> pd.Series

    Parameters
    ----------
    ret_matrix : pd.DataFrame, shape (T, N)
        Returns matrix from build_returns_matrix(). Only the column index
        (asset universe) is used — return history is ignored entirely.

    Returns
    -------
    weights : pd.Series, length N
        Weight 1/N for every asset, indexed by permno. Sums to 1.0.
        Passed directly to the turnover and cost accounting in run_backtest,
        bypassing covariance estimation and optimize_portfolio entirely.
    """
    N = ret_matrix.shape[1]
    weights = pd.Series(1.0 / N, index=ret_matrix.columns)
    return weights


# ============================================================
# __main__: Full backtest 1976–2024, evaluate and save results
# ============================================================

if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from portfolio.optimizer import run_backtest
    from portfolio.metrics  import evaluate_results

    print("\n" + "=" * 60)
    print("LEVEL 0: EQUAL-WEIGHT BENCHMARK — 1980 to 2024")
    print("=" * 60)

    # --------------------------------------------------------
    # Load master panel — same file used by all models above Level 0.
    # Both 1/N and FF5 must draw from identical universes at every
    # rebalancing date so that performance differences reflect model
    # skill, not differences in the investable stock set.
    # --------------------------------------------------------
    try:
        from portfolio.config import PANEL_PATH
    except ImportError:
        PANEL_PATH = "data_clean/master_panel.csv"
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
    # Run backtest: Jan 1980 – Dec 2024
    #
    # Start year is 1980: the 60-month estimation window (Jan 1975–Dec 1979)
    # has solid Compustat/CCM coverage throughout, avoiding the thin early-
    # sample universe present in the 1975–1979 period.
    # --------------------------------------------------------
    results = run_backtest(
        master,
        start_year=1980, start_month=1,
        end_year=2024,   end_month=12,
        risk_aversion=1.0,
        window=60,
        cost_bps=10,
        weights_fn=equal_weight,
    )

    print(f"\nBacktest complete: {len(results)} months computed.")

    # --------------------------------------------------------
    # Evaluate performance
    # --------------------------------------------------------
    summary = evaluate_results(results, rf_series=None)

    print("\n" + "=" * 60)
    print("LEVEL 0 PERFORMANCE SUMMARY (rf = 0)")
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

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/level0_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Level 0 benchmark complete.")
