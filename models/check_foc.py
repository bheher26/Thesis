#!/usr/bin/env python3
"""
models/check_foc.py

First-Order Conditions (FOC) diagnostic for the mean-variance portfolio optimizer.

What this checks
----------------
The portfolio optimizer solves:

    min  (λ/2) * w' * Σ * w  -  μ' * w
    s.t. sum(w) = 1,   0 ≤ w_i ≤ max_weight

At the true optimum, the KKT first-order conditions must hold:

    For every stock i, define the "gradient" g_i = (λ * Σ * w)_i - μ_i + ν
    where ν (nu) is the shadow price of the "must be fully invested" constraint.

    Then:
      · If  0 < w_i < max_weight  (interior): g_i ≈ 0  (no reason to trade)
      · If  w_i = 0               (zero weight): g_i ≥ 0  (buying would hurt)
      · If  w_i = max_weight      (at cap):  g_i ≤ 0  (would hold more if allowed)

If these conditions don't hold, the optimizer may not have converged to the true
optimum — either it needs more iterations, the problem is ill-conditioned, or the
model's expected-return vector is numerically messy.

Transaction-cost note
---------------------
TC is deducted from realised returns *after* optimisation — it's not in the
objective function — so TC only enters the FOC check as a diagnostic: a stock
with a large trade (|w_i - w_prev_i|) should have a meaningful expected-return
contribution to justify the round-trip cost.  The "implied alpha net of TC"
column shows  μ_i - cost_bps/10000 * sign(w_i - w_prev_i)  for each stock.

Usage
-----
    python models/check_foc.py                    # checks last available month
    python models/check_foc.py --year 2023 --month 6
    python models/check_foc.py --all-months       # summary over all months

Prerequisites (all produced by running the backtest scripts first):
    data_clean/master_panel.csv
    data_clean/elastic_net/expected_returns_enet_ols.parquet
    data_clean/elastic_net/expected_returns_enet_huber.parquet
    data_clean/random_forest/expected_returns_rf.parquet
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from portfolio.covariance import build_returns_matrix, build_ret_panel, estimate_covariance
from portfolio.optimizer  import optimize_portfolio

# ── Backtest parameters (must match run_enet_backtest / run_rf_backtest) ──────
try:
    from portfolio.config import PANEL_PATH as MASTER_PATH  # type: ignore
except ImportError:
    MASTER_PATH = "data_clean/master_panel.csv"

ENET_DIR       = "data_clean/elastic_net"
RF_DIR         = "data_clean/random_forest"
RISK_AVERSION  = 1.0    # λ — same value used in every backtest
WINDOW         = 60     # trailing months for covariance estimation
MAX_WEIGHT     = 0.10   # per-stock concentration cap (10 %)
COST_BPS       = 10     # transaction cost in basis points (0.10 %)

# Tolerance for declaring an FOC violation
FOC_TOL = 1e-4   # gradient must be this close to 0 for interior stocks


# ── Helper: load expected returns for one month ───────────────────────────────

def _load_mu(path: str, year: int, month: int) -> pd.Series:
    """
    Read a parquet of (permno, year, month, expected_ret) and return
    the expected-return Series for the requested month, indexed by permno.
    """
    df = pd.read_parquet(path)
    df["permno"] = df["permno"].astype(int)
    slice_ = df[(df["year"] == year) & (df["month"] == month)]
    slice_ = slice_.drop_duplicates("permno")
    return slice_.set_index("permno")["expected_ret"]


# ── Core FOC checker ─────────────────────────────────────────────────────────

def check_foc(
    mu: pd.Series,
    Sigma: np.ndarray,
    weights: pd.Series,
    risk_aversion: float = RISK_AVERSION,
    max_weight: float    = MAX_WEIGHT,
    cost_bps: float      = COST_BPS,
    weights_prev: Optional[pd.Series] = None,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Verify the KKT first-order conditions for a solved portfolio.

    Parameters
    ----------
    mu           : expected returns (permno-indexed), length N
    Sigma        : (N, N) covariance matrix, aligned with mu
    weights      : optimal weights returned by optimize_portfolio
    risk_aversion: λ used in the objective
    max_weight   : per-stock upper bound used in the optimisation
    cost_bps     : transaction cost rate (only used for the TC annotation)
    weights_prev : previous period's weights (optional, only for TC annotation)
    model_name   : label for printed output
    verbose      : if True, print a detailed table

    Returns
    -------
    dict with summary statistics for this month
    """
    N = len(mu)
    mu_arr = mu.values
    w      = weights.reindex(mu.index, fill_value=0.0).values

    # ── Step 1: compute the gradient of the objective w.r.t. each weight ──
    # This is:  grad_i = λ * (Σ * w)_i  -  μ_i
    # At the optimum with the "sum = 1" constraint, adding the Lagrange
    # multiplier ν shifts all gradients equally so interior stocks hit zero.
    grad_raw = risk_aversion * (Sigma @ w) - mu_arr   # shape (N,)

    # ── Step 2: estimate ν from the interior stocks ────────────────────────
    # Interior stocks are those that hit neither bound:  0 < w_i < max_weight.
    # At the true optimum their gradient + ν = 0, so ν = –grad_i for each.
    # We use the median (robust to a few boundary mis-classifications).
    interior_mask = (w > 1e-6) & (w < max_weight - 1e-6)
    if interior_mask.sum() == 0:
        # No interior stocks — happens when every stock is at a bound.
        # Fall back: use the mean gradient over all held stocks.
        interior_mask = w > 1e-6
    nu = -np.median(grad_raw[interior_mask]) if interior_mask.sum() > 0 else 0.0

    # ── Step 3: shifted gradient (FOC residual) ───────────────────────────
    # g_i = grad_i + ν  should be:  ≈ 0 for interior,  ≥ 0 for w=0,  ≤ 0 for w=cap
    g = grad_raw + nu   # shape (N,)

    # ── Step 4: classify each stock and check conditions ─────────────────
    at_zero = w < 1e-6
    at_cap  = w > max_weight - 1e-6

    # Violations
    interior_violation = np.abs(g[interior_mask])               # should be ≈ 0
    zero_violation     = np.minimum(g[at_zero],  0.0)           # should be ≥ 0; negatives are bad
    cap_violation      = np.maximum(g[at_cap],   0.0)           # should be ≤ 0; positives are bad

    n_interior_viol = int((np.abs(g[interior_mask]) > FOC_TOL).sum())
    n_zero_viol     = int((g[at_zero]  < -FOC_TOL).sum())
    n_cap_viol      = int((g[at_cap]   >  FOC_TOL).sum())
    n_viol_total    = n_interior_viol + n_zero_viol + n_cap_viol

    # ── Step 5: optional TC annotation ────────────────────────────────────
    if weights_prev is not None:
        w_prev = weights_prev.reindex(mu.index, fill_value=0.0).values
        trade  = w - w_prev                                         # + = buy, – = sell
        tc_per_unit = cost_bps / 10_000                             # one-way cost
        # Net alpha: expected gain minus TC drag
        mu_net = mu_arr - tc_per_unit * np.sign(trade)
    else:
        w_prev    = np.zeros(N)
        trade     = w - w_prev
        mu_net    = mu_arr.copy()

    # ── Step 6: print a summary ───────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*65}")
        print(f"  FOC CHECK  —  {model_name}")
        print(f"{'─'*65}")
        print(f"  Universe: {N} stocks   |   Held (w > 0): {int((w > 1e-6).sum())}")
        print(f"  Interior (0 < w < cap): {int(interior_mask.sum())}")
        print(f"  At lower bound (w = 0): {int(at_zero.sum())}")
        print(f"  At upper bound (w = cap): {int(at_cap.sum())}")
        print(f"  Lagrange multiplier ν  = {nu:.6f}")
        print()
        print(f"  FOC residual |g| on interior stocks:")
        print(f"    mean={np.mean(np.abs(g[interior_mask])):.2e}  "
              f"max={np.max(np.abs(g[interior_mask])):.2e}  "
              f"(tolerance = {FOC_TOL:.0e})")
        print()
        print(f"  Violations (|residual| > {FOC_TOL:.0e}):")
        print(f"    Interior  : {n_interior_viol:3d} / {int(interior_mask.sum())}")
        print(f"    Zero-bound: {n_zero_viol:3d} / {int(at_zero.sum())}  "
              f"(residual < 0 means stock was wrongly excluded)")
        print(f"    Cap-bound : {n_cap_viol:3d} / {int(at_cap.sum())}  "
              f"(residual > 0 means stock was wrongly capped)")
        print(f"    TOTAL violations: {n_viol_total}")

        # Print the worst offenders if any exist
        if n_viol_total > 0:
            idx_sorted = np.argsort(np.abs(g))[::-1][:10]
            permnos    = mu.index.tolist()
            print(f"\n  Worst-offending stocks (by |FOC residual|):")
            print(f"    {'permno':>8}  {'w':>7}  {'w_prev':>7}  "
                  f"{'μ':>9}  {'μ_net':>9}  {'g':>10}  {'status':>10}")
            print(f"    {'─'*8}  {'─'*7}  {'─'*7}  "
                  f"{'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}")
            for i in idx_sorted:
                status = ("INTERIOR" if interior_mask[i]
                          else "ZERO"     if at_zero[i]
                          else "CAP"      if at_cap[i]
                          else "???")
                print(f"    {permnos[i]:>8}  {w[i]:>7.4f}  {w_prev[i]:>7.4f}  "
                      f"{mu_arr[i]:>9.5f}  {mu_net[i]:>9.5f}  {g[i]:>10.2e}  {status:>10}")

        # Overall verdict
        print()
        if n_viol_total == 0:
            print("  ✓  All KKT conditions satisfied — optimizer at a valid optimum.")
        elif n_viol_total <= 5:
            print(f"  ⚠  {n_viol_total} minor violations — likely numerical noise; "
                  "solution is approximately optimal.")
        else:
            print(f"  ✗  {n_viol_total} violations — optimizer may NOT be at the true "
                  "optimum.  Consider increasing max_iter or checking μ/Σ conditioning.")

    return {
        "n_assets":          N,
        "n_held":            int((w > 1e-6).sum()),
        "n_interior":        int(interior_mask.sum()),
        "n_at_zero":         int(at_zero.sum()),
        "n_at_cap":          int(at_cap.sum()),
        "nu":                float(nu),
        "mean_abs_residual": float(np.mean(np.abs(g[interior_mask]))) if interior_mask.sum() > 0 else np.nan,
        "max_abs_residual":  float(np.max(np.abs(g[interior_mask])))  if interior_mask.sum() > 0 else np.nan,
        "n_violations":      n_viol_total,
        "n_interior_viol":   n_interior_viol,
        "n_zero_viol":       n_zero_viol,
        "n_cap_viol":        n_cap_viol,
    }


# ── Per-month runner ──────────────────────────────────────────────────────────

def run_foc_for_month(
    master: pd.DataFrame,
    ret_panel: pd.DataFrame,
    year: int,
    month: int,
    enet_ols_path: str,
    enet_huber_path: str,
    rf_path: str,
    weights_prev: Optional[dict] = None,   # dict of model_name → prev pd.Series
    verbose: bool = True,
) -> dict:
    """
    Run the full FOC diagnostic for a single (year, month).

    Returns a dict of {model_name: foc_stats_dict}.
    """
    print(f"\n{'='*65}")
    print(f"  FOC DIAGNOSTIC  —  {year}-{month:02d}")
    print(f"{'='*65}")

    # ── Build covariance from the 60-month trailing window ────────────────
    try:
        ret_matrix = build_returns_matrix(master, year, month,
                                          window=WINDOW, ret_panel=ret_panel)
    except Exception as e:
        print(f"  Cannot build returns matrix: {e}")
        return {}

    N = ret_matrix.shape[1]
    print(f"  Returns matrix: {ret_matrix.shape[0]} months × {N} stocks")

    cov_result = estimate_covariance(ret_matrix)
    Sigma = cov_result.covariance
    print(f"  Covariance estimated (Ledoit-Wolf shrinkage δ = {cov_result.delta:.3f})")

    universe_permnos = ret_matrix.columns   # permnos in the covariance universe

    results = {}
    models = {
        "enet_ols":   enet_ols_path,
        "enet_huber": enet_huber_path,
        "rf":         rf_path,
    }

    for model_name, path in models.items():
        try:
            mu_full = _load_mu(path, year, month)
        except FileNotFoundError:
            print(f"\n  [{model_name}] Parquet not found at {path}, skipping.")
            continue
        except Exception as e:
            print(f"\n  [{model_name}] Error loading μ: {e}")
            continue

        # Restrict to stocks that are in BOTH the covariance universe and the
        # model forecast — same logic as run_backtest does
        valid = mu_full.reindex(universe_permnos).dropna()
        if len(valid) < 10:
            print(f"\n  [{model_name}] Only {len(valid)} stocks with valid μ, skipping.")
            continue

        # Subset the covariance to those stocks
        idx_map   = [i for i, p in enumerate(universe_permnos) if p in valid.index]
        Sigma_sub = Sigma[np.ix_(idx_map, idx_map)]
        mu_sub    = valid

        # Optimise
        weights = optimize_portfolio(mu_sub, Sigma_sub,
                                     risk_aversion=RISK_AVERSION,
                                     max_weight=MAX_WEIGHT)

        w_prev = weights_prev.get(model_name) if weights_prev else None

        foc_stats = check_foc(
            mu_sub, Sigma_sub, weights,
            risk_aversion=RISK_AVERSION,
            max_weight=MAX_WEIGHT,
            cost_bps=COST_BPS,
            weights_prev=w_prev,
            model_name=f"{model_name.upper()}  [{year}-{month:02d}]",
            verbose=verbose,
        )
        results[model_name] = {**foc_stats, "weights": weights}

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check portfolio optimizer first-order (KKT) conditions.",
    )
    parser.add_argument("--year",       type=int, default=None,
                        help="Year to check (default: last backtest year)")
    parser.add_argument("--month",      type=int, default=None,
                        help="Month to check (default: last backtest month)")
    parser.add_argument("--all-months", action="store_true",
                        help="Run over every month in the backtest window and "
                             "print a summary table")
    parser.add_argument("--start-year",  type=int, default=2005)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year",    type=int, default=2024)
    parser.add_argument("--end-month",   type=int, default=12)
    args = parser.parse_args()

    # ── Load master panel ────────────────────────────────────────────────
    print(f"Loading master panel from {MASTER_PATH} …")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master.columns   = [c.strip().lower() for c in master.columns]
    master["year"]   = pd.to_numeric(master["year"],   errors="coerce").astype(int)
    master["month"]  = pd.to_numeric(master["month"],  errors="coerce").astype(int)
    master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype(int)
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    print(f"  Loaded: {master.shape}")

    print("Pre-building wide returns panel …")
    ret_panel = build_ret_panel(master)

    enet_ols_path   = f"{ENET_DIR}/expected_returns_enet_ols.parquet"
    enet_huber_path = f"{ENET_DIR}/expected_returns_enet_huber.parquet"
    rf_path         = f"{RF_DIR}/expected_returns_rf.parquet"

    # ── Single-month check ───────────────────────────────────────────────
    if not args.all_months:
        year  = args.year  if args.year  else args.end_year
        month = args.month if args.month else args.end_month
        run_foc_for_month(
            master, ret_panel,
            year, month,
            enet_ols_path, enet_huber_path, rf_path,
            verbose=True,
        )
        return

    # ── Multi-month sweep ────────────────────────────────────────────────
    print(f"\nRunning FOC sweep: {args.start_year}-{args.start_month:02d} "
          f"→ {args.end_year}-{args.end_month:02d} …")

    all_periods = pd.period_range(
        start=f"{args.start_year}-{args.start_month:02d}",
        end=f"{args.end_year}-{args.end_month:02d}",
        freq="M",
    )

    summary_rows = []
    weights_prev: dict = {}   # carries previous month's weights for TC annotation

    for period in all_periods:
        y, m = period.year, period.month
        results = run_foc_for_month(
            master, ret_panel, y, m,
            enet_ols_path, enet_huber_path, rf_path,
            weights_prev={k: v["weights"] for k, v in weights_prev.items()},
            verbose=False,   # quiet in sweep mode
        )
        for model_name, stats in results.items():
            summary_rows.append({
                "year":  y, "month": m, "model": model_name,
                **{k: v for k, v in stats.items() if k != "weights"},
            })
        # Carry weights forward for next month's TC annotation
        weights_prev = results

    if not summary_rows:
        print("No results — check that the expected-return parquets exist.")
        return

    summary = pd.DataFrame(summary_rows)

    print(f"\n{'='*65}")
    print("FOC SWEEP SUMMARY")
    print(f"{'='*65}")
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        print(f"\n  {model.upper()}")
        print(f"    Months checked          : {len(sub)}")
        print(f"    Avg universe size       : {sub['n_assets'].mean():.0f} stocks")
        print(f"    Avg held (w > 0)        : {sub['n_held'].mean():.0f} stocks")
        print(f"    Avg interior stocks     : {sub['n_interior'].mean():.0f}")
        print(f"    Mean |FOC residual|     : {sub['mean_abs_residual'].mean():.2e}")
        print(f"    Max  |FOC residual|     : {sub['max_abs_residual'].max():.2e}")
        print(f"    Months with violations  : {(sub['n_violations'] > 0).sum()} / {len(sub)}")
        print(f"    Avg violations / month  : {sub['n_violations'].mean():.1f}")

    out_path = "data_clean/foc_sweep_summary.csv"
    summary.drop(columns=["weights"], errors="ignore").to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
