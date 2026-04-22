import os
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# ============================================================
# Portfolio Performance Metrics
# portfolio/metrics.py
#
# Provides:
#   evaluate_results(results_df, rf_series) -> dict
#
# Designed to consume the DataFrame returned by run_backtest()
# in portfolio/optimizer.py.
# ============================================================


def evaluate_results(results_df, rf_series=None):
    """
    Compute standard performance metrics from a monthly backtest results
    DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_backtest(). Must contain columns:
            year, month, gross_return, net_return, turnover, cost
    rf_series : pd.Series or None
        Monthly risk-free rate indexed by (year, month) MultiIndex or a
        flat Series aligned to results_df row order. If None, rf = 0
        for all periods (excess return = raw return).

    Returns
    -------
    summary : dict
        annualized_net_sharpe    — net return Sharpe ratio (annualised)
        annualized_gross_sharpe  — gross return Sharpe ratio (annualised)
        annualized_net_return    — geometric mean net return, annualised
        annualized_volatility    — std of monthly net returns, annualised
        max_drawdown             — peak-to-trough drawdown on net NAV
        avg_monthly_turnover     — mean one-way turnover per month
        avg_monthly_cost_bps     — mean transaction cost per month in bps
        annualized_downside_deviation — annualised downside deviation (MAR=0)
        annualized_sortino       — annualised Sortino ratio (MAR=0)
    """
    df = results_df.copy()

    # --------------------------------------------------------
    # Risk-free rate alignment
    # If rf_series provided, merge on year+month; otherwise zero
    # --------------------------------------------------------
    if rf_series is not None:
        rf = rf_series.reset_index()
        rf.columns = ["year", "month", "rf"]
        df = df.merge(rf, on=["year", "month"], how="left")
        df["rf"] = df["rf"].fillna(0.0)
    else:
        df["rf"] = 0.0

    # --------------------------------------------------------
    # Excess returns (over risk-free)
    # --------------------------------------------------------
    df["excess_net"]   = df["net_return"]   - df["rf"]
    df["excess_gross"] = df["gross_return"] - df["rf"]

    T = len(df)

    # --------------------------------------------------------
    # Annualised net return — geometric compounding
    # --------------------------------------------------------
    cum_net = (1 + df["net_return"]).prod()
    annualized_net_return = cum_net ** (12 / T) - 1

    cum_gross = (1 + df["gross_return"]).prod()
    annualized_gross_return = cum_gross ** (12 / T) - 1

    # --------------------------------------------------------
    # Annualised volatility — std of monthly net returns * sqrt(12)
    # --------------------------------------------------------
    annualized_volatility = df["net_return"].std() * np.sqrt(12)

    # --------------------------------------------------------
    # Sharpe ratios — mean excess return / std excess return * sqrt(12)
    # --------------------------------------------------------
    net_sharpe_monthly   = df["excess_net"].mean()   / df["excess_net"].std()
    gross_sharpe_monthly = df["excess_gross"].mean() / df["excess_gross"].std()

    annualized_net_sharpe   = net_sharpe_monthly   * np.sqrt(12)
    annualized_gross_sharpe = gross_sharpe_monthly * np.sqrt(12)

    # --------------------------------------------------------
    # Maximum drawdown on net NAV
    # Drawdown = (peak NAV - current NAV) / peak NAV
    # --------------------------------------------------------
    nav = (1 + df["net_return"]).cumprod()
    rolling_peak = nav.cummax()
    drawdowns = (nav - rolling_peak) / rolling_peak
    max_drawdown = drawdowns.min()   # most negative value

    # --------------------------------------------------------
    # Turnover and cost averages
    # Skip first month turnover (=1.0, full construction from cash)
    # --------------------------------------------------------
    avg_monthly_turnover  = df["turnover"].iloc[1:].mean()
    avg_monthly_cost_bps  = df["cost"].mean() * 10_000

    # --------------------------------------------------------
    # Downside deviation and Sortino ratio
    # MAR (Minimum Acceptable Return) = 0, applied to excess net returns.
    # Downside deviation: sqrt of mean squared negative excess returns (monthly),
    # annualised by multiplying by sqrt(12).
    # Sortino = annualised mean excess return / annualised downside deviation.
    # --------------------------------------------------------
    downside         = np.minimum(df["excess_net"].values, 0.0)
    semi_var_monthly = np.mean(downside ** 2)                    # monthly semi-variance

    downside_dev_ann = np.sqrt(semi_var_monthly) * np.sqrt(12)  # annualised downside deviation
    mean_excess_ann  = df["excess_net"].mean() * 12

    if downside_dev_ann > 0:
        annualized_sortino = mean_excess_ann / downside_dev_ann
    else:
        annualized_sortino = np.nan

    summary = {
        "annualized_net_sharpe":        round(annualized_net_sharpe,   4),
        "annualized_gross_sharpe":      round(annualized_gross_sharpe, 4),
        "annualized_gross_return":      round(annualized_gross_return, 4),
        "annualized_net_return":        round(annualized_net_return,   4),
        "annualized_volatility":        round(annualized_volatility,   4),
        "max_drawdown":                 round(max_drawdown,            4),
        "avg_monthly_turnover":         round(avg_monthly_turnover,    4),
        "avg_monthly_cost_bps":         round(avg_monthly_cost_bps,    4),
        "annualized_downside_deviation": round(downside_dev_ann,       6),
        "annualized_sortino":           round(annualized_sortino,      4) if not np.isnan(annualized_sortino) else np.nan,
    }

    return summary


# ============================================================
# Benchmark Comparison
# ============================================================

BENCHMARK_PATH = "data_clean/level0_results.csv"

_MODEL_LABELS = {
    "level0": "Level 0  (1/N equal-weight)",
    "level1": "Level 1  (FF5 static)",
    "level2": "Level 2  (FF5 + macro)",
    "level3": "Level 3  (ML model)",
}


def print_benchmark_comparison(current_label, current_summary,
                                benchmark_path=BENCHMARK_PATH,
                                rf_series=None,
                                results_df=None):
    """
    Print a side-by-side comparison of the current model against the Level 0
    1/N benchmark. Call at the bottom of each model's __main__ block:

        print_benchmark_comparison("level1", summary, results_df=results)

    The benchmark is clipped to the same date range as results_df so that
    all metrics are computed over an identical evaluation window.

    Parameters
    ----------
    current_label : str
        Key into _MODEL_LABELS (e.g. "level1").
    current_summary : dict
        Output of evaluate_results() for the current model.
    benchmark_path : str
        Path to Level 0 results CSV.
    rf_series : pd.Series or None
        Forwarded to evaluate_results() for the benchmark.
    results_df : pd.DataFrame or None
        The raw results DataFrame for the current model (output of
        run_backtest). When provided, the benchmark is clipped to the same
        (year, month) range so the comparison is over identical periods.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON vs. Level 0 (1/N)")
    print("=" * 60)

    if not os.path.exists(benchmark_path):
        print(f"  [Skipped] {benchmark_path} not found.")
        print("  Run models/level0_equal_weight.py first.")
        print("=" * 60)
        return

    try:
        r0 = pd.read_csv(benchmark_path)
        r0.columns = [c.strip().lower() for c in r0.columns]

        # Clip benchmark to the same date window as the current model
        if results_df is not None:
            df_cur = results_df.copy()
            df_cur.columns = [c.strip().lower() for c in df_cur.columns]
            start_year  = int(df_cur["year"].min())
            start_month = int(df_cur.loc[df_cur["year"] == start_year, "month"].min())
            end_year    = int(df_cur["year"].max())
            end_month   = int(df_cur.loc[df_cur["year"] == end_year, "month"].max())

            r0 = r0[
                ((r0["year"] > start_year) |
                 ((r0["year"] == start_year) & (r0["month"] >= start_month))) &
                ((r0["year"] < end_year) |
                 ((r0["year"] == end_year) & (r0["month"] <= end_month)))
            ].copy()

            print(f"  [Evaluation window: {start_year}-{start_month:02d} → "
                  f"{end_year}-{end_month:02d}  ({len(r0)} months)]")

        s0 = evaluate_results(r0, rf_series=rf_series)
    except Exception as e:
        print(f"  [Error] Could not load benchmark: {e}")
        print("=" * 60)
        return

    label_cur = _MODEL_LABELS.get(current_label, current_label)

    rows = [
        ("Net Sharpe (ann.)",    "annualized_net_sharpe",   "{:.3f}", ""),
        ("Gross Sharpe (ann.)",  "annualized_gross_sharpe", "{:.3f}", ""),
        ("Sortino (ann.)",       "annualized_sortino",      "{:.3f}", ""),
        ("Gross return (ann.)",  "annualized_gross_return", "{:.2f}", "%"),
        ("Net return (ann.)",    "annualized_net_return",   "{:.2f}", "%"),
        ("Volatility (ann.)",    "annualized_volatility",   "{:.2f}", "%"),
        ("Downside dev. (ann.)", "annualized_downside_deviation", "{:.4f}", ""),
        ("Max drawdown",         "max_drawdown",            "{:.2f}", "%"),
        ("Avg turnover/month",   "avg_monthly_turnover",    "{:.1f}", "%"),
        ("Avg cost/month",       "avg_monthly_cost_bps",    "{:.2f}", " bps"),
    ]

    col_w = 22
    print(f"  {'Metric':<{col_w}}  {'Level 0':>12}  {label_cur:>20}  {'Δ':>10}")
    print("  " + "-" * (col_w + 50))
    for display, key, fmt, unit in rows:
        v0 = s0[key]
        v1 = current_summary[key]
        if unit == "%":
            v0 *= 100; v1 *= 100
        diff = v1 - v0
        sign = "+" if diff >= 0 else ""
        print(f"  {display:<{col_w}}  "
              f"{fmt.format(v0):>11}{unit}  "
              f"{fmt.format(v1):>19}{unit}  "
              f"{sign}{fmt.format(diff):>9}{unit}")
    print("=" * 60)


# ============================================================
# CAPM Alpha / Beta
# ============================================================

_FF5_PATH = "data_raw/ff_factors_monthly.csv"
_FF5_CAPM_CACHE = None


def _load_ff5_capm():
    """Load FF5 file and return DataFrame with year, month, mkt_rf, rf (decimal)."""
    global _FF5_CAPM_CACHE
    if _FF5_CAPM_CACHE is not None:
        return _FF5_CAPM_CACHE
    ff = pd.read_csv(_FF5_PATH, skiprows=4)
    ff.columns = [c.strip().lower().replace("-", "_") for c in ff.columns]
    ff = ff.rename(columns={ff.columns[0]: "date"})
    ff["date"] = pd.to_numeric(ff["date"], errors="coerce")
    ff = ff[ff["date"] >= 190001].copy()
    ff["date"] = ff["date"].astype(int)
    ff["year"]  = ff["date"] // 100
    ff["month"] = ff["date"] % 100
    for col in ["mkt_rf", "rf"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0
    _FF5_CAPM_CACHE = ff[["year", "month", "mkt_rf", "rf"]].copy()
    return _FF5_CAPM_CACHE


def compute_capm_alpha(results_df):
    """
    Run a CAPM regression of the portfolio's net excess returns on the market
    excess return (Mkt-RF from Ken French's data library).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_backtest(). Must contain year, month, net_return.

    Returns
    -------
    dict with keys:
        capm_alpha        : annualised Jensen's alpha (decimal)
        capm_alpha_pct    : annualised alpha in percent
        capm_beta         : market beta
        capm_alpha_tstat  : t-statistic on the alpha
        capm_alpha_pvalue : two-sided p-value on alpha
        capm_r2           : R-squared of regression
        capm_n            : number of months used
    All NaN if FF5 data unavailable.
    """
    nan_result = {
        "capm_alpha": np.nan, "capm_alpha_pct": np.nan,
        "capm_beta": np.nan, "capm_alpha_tstat": np.nan,
        "capm_alpha_pvalue": np.nan, "capm_r2": np.nan, "capm_n": np.nan,
    }
    try:
        ff = _load_ff5_capm()
    except Exception:
        return nan_result

    df = results_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Portfolio row (year t, month t) holds the return earned in month t+1
    # (weights built at t, realized returns from t+1).  Shift dates forward
    # by one month so factor data aligns with actual return period.
    dates = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    next_dates = dates + pd.DateOffset(months=1)
    df["_ret_year"]  = next_dates.dt.year
    df["_ret_month"] = next_dates.dt.month
    df = df.merge(
        ff.rename(columns={"year": "_ret_year", "month": "_ret_month"}),
        on=["_ret_year", "_ret_month"], how="left"
    )

    df["excess_net"] = df["net_return"] - df["rf"]
    df = df.dropna(subset=["excess_net", "mkt_rf"])

    if len(df) < 12:
        return nan_result

    y = df["excess_net"].values
    x = df["mkt_rf"].values

    slope, intercept, r_value, p_value, se_slope = _scipy_stats.linregress(x, y)

    # t-stat and p-value for alpha (intercept)
    n = len(y)
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    residuals = y - (intercept + slope * x)
    s2 = np.sum(residuals ** 2) / (n - 2)
    se_alpha = np.sqrt(s2 * (1.0 / n + x_mean ** 2 / ss_x))
    t_alpha = intercept / se_alpha
    p_alpha = 2 * _scipy_stats.t.sf(abs(t_alpha), df=n - 2)

    return {
        "capm_alpha":        round(intercept * 12, 6),       # annualised
        "capm_alpha_pct":    round(intercept * 12 * 100, 4), # annualised %
        "capm_beta":         round(slope, 4),
        "capm_alpha_tstat":  round(t_alpha, 4),
        "capm_alpha_pvalue": round(p_alpha, 4),
        "capm_r2":           round(r_value ** 2, 4),
        "capm_n":            int(n),
    }


# ============================================================
# __main__: Load backtest_results.csv and print summary
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PORTFOLIO METRICS SUMMARY")
    print("=" * 60)

    results_path = "data_clean/backtest_results.csv"
    print(f"\nLoading backtest results from: {results_path}")
    results = pd.read_csv(results_path)
    results.columns = [c.strip().lower() for c in results.columns]
    print(f"Loaded {len(results)} months of results.")

    # Run with rf=0 (no risk-free series provided)
    summary = evaluate_results(results, rf_series=None)

    print("\n" + "-" * 40)
    print(f"  Annualised net Sharpe    : {summary['annualized_net_sharpe']:.3f}")
    print(f"  Annualised gross Sharpe  : {summary['annualized_gross_sharpe']:.3f}")
    print(f"  Annualised Sortino       : {summary['annualized_sortino']:.3f}")
    print(f"  Annualised gross return  : {summary['annualized_gross_return']*100:.2f}%")
    print(f"  Annualised net return    : {summary['annualized_net_return']*100:.2f}%")
    print(f"  Annualised volatility    : {summary['annualized_volatility']*100:.2f}%")
    print(f"  Downside deviation (ann.): {summary['annualized_downside_deviation']:.4f}")
    print(f"  Max drawdown             : {summary['max_drawdown']*100:.2f}%")
    print(f"  Avg monthly turnover     : {summary['avg_monthly_turnover']*100:.1f}%")
    print(f"  Avg monthly cost         : {summary['avg_monthly_cost_bps']:.2f} bps")
    print("-" * 40)
    print("Metrics complete.")
