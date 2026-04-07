#!/usr/bin/env python3
"""
models/run_enet_backtest.py

Runs the pre-computed elastic net expected returns through the shared
portfolio optimizer to produce level3_ols_results.csv and
level3_huber_results.csv in data_clean/.

Must be run from the project root:
    python models/run_enet_backtest.py

Prerequisites:
    - data_clean/master_panel.csv
    - data_clean/elastic_net/expected_returns_enet_ols.parquet
    - data_clean/elastic_net/expected_returns_enet_huber.parquet

These parquets are produced by running level3_elastic_net.py first.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.level3_elastic_net import make_enet_expected_returns_fn
from portfolio.optimizer import run_backtest
from portfolio.metrics import evaluate_results, print_benchmark_comparison

# ── Config ────────────────────────────────────────────────────────────────────
MASTER_PATH = "data_clean/master_panel.csv"
OUTPUT_DIR  = "data_clean/elastic_net"
START_YEAR, START_MONTH = 2005, 1
END_YEAR,   END_MONTH   = 2024, 12
RISK_AVERSION = 1.0
WINDOW        = 60
COST_BPS      = 10
MAX_TURNOVER  = 0.35   # 35% one-way monthly turnover cap

# ── Load master panel ─────────────────────────────────────────────────────────
print(f"\nLoading master panel from {MASTER_PATH} …")
master = pd.read_csv(MASTER_PATH, low_memory=False)
master.columns   = [c.strip().lower() for c in master.columns]
master["year"]   = pd.to_numeric(master["year"],   errors="coerce").astype(int)
master["month"]  = pd.to_numeric(master["month"],  errors="coerce").astype(int)
master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype(int)
master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
print(f"Loaded: {master.shape}")

# ── Run backtest for each loss variant ────────────────────────────────────────
for loss in ("ols", "huber"):
    print(f"\n{'='*60}")
    print(f"ELASTIC NET — {loss.upper()} LOSS  ({START_YEAR}–{END_YEAR})")
    print(f"{'='*60}")

    expected_returns_fn = make_enet_expected_returns_fn(
        loss=loss,
        output_dir=OUTPUT_DIR,
    )

    results = run_backtest(
        master,
        start_year=START_YEAR, start_month=START_MONTH,
        end_year=END_YEAR,     end_month=END_MONTH,
        risk_aversion=RISK_AVERSION,
        window=WINDOW,
        cost_bps=COST_BPS,
        expected_returns_fn=expected_returns_fn,
        max_turnover=MAX_TURNOVER,
    )

    print(f"\nBacktest complete: {len(results)} months.")

    summary = evaluate_results(results, rf_series=None)

    print(f"\n{'='*60}")
    print(f"LEVEL 3 ({loss.upper()}) PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Annualised net Sharpe    : {summary['annualized_net_sharpe']:.3f}")
    print(f"  Annualised gross Sharpe  : {summary['annualized_gross_sharpe']:.3f}")
    print(f"  Annualised net return    : {summary['annualized_net_return']*100:.2f}%")
    print(f"  Annualised volatility    : {summary['annualized_volatility']*100:.2f}%")
    print(f"  Max drawdown             : {summary['max_drawdown']*100:.2f}%")
    avg_to = results["turnover"].mean() * 100
    pct_capped = (results["turnover"] <= MAX_TURNOVER + 0.001).mean() * 100
    print(f"  Avg monthly turnover     : {avg_to:.1f}%  (cap={MAX_TURNOVER*100:.0f}%)")
    print(f"  Months at/below cap      : {pct_capped:.0f}%")
    print(f"  Avg monthly cost         : {summary['avg_monthly_cost_bps']:.2f} bps")
    print(f"{'='*60}")

    print_benchmark_comparison(f"level3_{loss}", summary, results_df=results)

    out_path = f"data_clean/level3_{loss}_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

print("\nDone.")
