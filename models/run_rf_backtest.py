#!/usr/bin/env python3
"""
models/run_rf_backtest.py

Runs the pre-computed random forest expected returns through the shared
portfolio optimizer to produce level4_rf_results.csv in data_clean/.

Must be run from the project root:
    python models/run_rf_backtest.py

Prerequisites:
    - data_clean/master_panel.csv
    - data_clean/random_forest/expected_returns_rf.parquet

The parquet is produced by running level4_random_forest.py first:
    python models/level4_random_forest.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.level4_random_forest import make_rf_expected_returns_fn
from portfolio.optimizer import run_backtest
from portfolio.metrics import evaluate_results, print_benchmark_comparison

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from portfolio.config import PANEL_PATH as MASTER_PATH  # type: ignore
except ImportError:
    MASTER_PATH = "data_clean/master_panel_v2.csv"
OUTPUT_DIR  = "data_clean/random_forest"
START_YEAR, START_MONTH = 2005, 1
END_YEAR,   END_MONTH   = 2024, 12
RISK_AVERSION = 1.0
WINDOW        = 60
COST_BPS      = 10
MAX_TURNOVER  = None   # uncapped — frontier script owns cap sweep

# ── Load master panel ─────────────────────────────────────────────────────────
print(f"\nLoading master panel from {MASTER_PATH} …")
master = pd.read_csv(MASTER_PATH, low_memory=False)
master.columns   = [c.strip().lower() for c in master.columns]
master["year"]   = pd.to_numeric(master["year"],   errors="coerce").astype(int)
master["month"]  = pd.to_numeric(master["month"],  errors="coerce").astype(int)
master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype(int)
master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
print(f"Loaded: {master.shape}")

# ── Run backtest ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RANDOM FOREST — LEVEL 4  ({START_YEAR}–{END_YEAR})")
print(f"{'='*60}")

expected_returns_fn = make_rf_expected_returns_fn(output_dir=OUTPUT_DIR)

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
print(f"LEVEL 4 (RF) PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"  Annualised net Sharpe    : {summary['annualized_net_sharpe']:.3f}")
print(f"  Annualised gross Sharpe  : {summary['annualized_gross_sharpe']:.3f}")
print(f"  Annualised gross return  : {summary['annualized_gross_return']*100:.2f}%")
print(f"  Annualised net return    : {summary['annualized_net_return']*100:.2f}%")
print(f"  Annualised volatility    : {summary['annualized_volatility']*100:.2f}%")
print(f"  Max drawdown             : {summary['max_drawdown']*100:.2f}%")
avg_to = results["turnover"].mean() * 100
cap_str = f"{MAX_TURNOVER*100:.0f}%" if MAX_TURNOVER is not None else "none"
print(f"  Avg monthly turnover     : {avg_to:.1f}%  (cap={cap_str})")
print(f"  Avg monthly cost         : {summary['avg_monthly_cost_bps']:.2f} bps")
print(f"{'='*60}")

print_benchmark_comparison("level4_rf", summary, results_df=results)

out_path = "data_clean/level4_rf_results.csv"
results.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

print("\nDone.")
