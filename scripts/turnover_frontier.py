#!/usr/bin/env python3
"""
scripts/turnover_frontier.py

Turnover-Sharpe frontier for thesis comparison of asset pricing models.

Produces two outputs:
  1. data_clean/frontier/frontier_results.csv
       One row per (model, cap_label). Columns:
         model, cap_label, cap_value, avg_turnover, gross_sharpe,
         net_sharpe, net_return, volatility, max_drawdown, avg_cost_bps

  2. data_clean/frontier/primary_table.csv
       One row per model at natural (uncapped) turnover — the primary
       results table for the thesis.

Usage
-----
    python scripts/turnover_frontier.py [--skip-missing]

    --skip-missing  : silently skip models whose parquet/CSV is not yet
                      available (useful while enet re-run is in progress)

Design
------
ML models (Enet OLS, Enet Huber, RF): expected returns are stored as
  pre-computed parquets; the optimizer is re-run at each cap level.

Simpler models (1/N, FF5, VAR-FF5): their expected_returns_fn is loaded
  from the respective module and re-run through the optimizer, so the
  same cap sweep is applied consistently.  If a module is unavailable the
  model falls back to its pre-computed results CSV (natural turnover only,
  no cap sweep).

Cap levels
----------
  [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
  None = uncapped (natural turnover).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.optimizer import run_backtest
from portfolio.metrics import evaluate_results

try:
    from portfolio.config import PANEL_PATH as MASTER_PATH
except ImportError:
    MASTER_PATH = "data_clean/master_panel_v2.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
START_YEAR,  START_MONTH = 2005, 1
END_YEAR,    END_MONTH   = 2024, 12
RISK_AVERSION = 1.0
WINDOW        = 60      # months of history for covariance estimation
COST_BPS      = 10      # flat one-way transaction cost (bps)

# Cap sweep: None = uncapped.  Values are one-way monthly fractions.
CAP_LEVELS = [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]

OUTPUT_DIR = Path("data_clean/frontier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (display_label, fn_factory_or_None, fallback_csv_or_None)
#   fn_factory : callable() -> expected_returns_fn for run_backtest
#   fallback_csv: path to pre-computed results CSV (used when fn_factory
#                 is None or fails); only natural turnover point is added.

def _enet_huber_fn():
    from models.level3_elastic_net import make_enet_expected_returns_fn
    return make_enet_expected_returns_fn(loss="huber")

def _rf_fn():
    from models.level4_random_forest import make_rf_expected_returns_fn
    return make_rf_expected_returns_fn()

def _ff5_fn():
    from models.level1_ff5 import make_ff5_expected_returns_fn
    return make_ff5_expected_returns_fn()

def _var_ff5_fn():
    from models.level_1_5 import make_var_ff5_expected_returns_fn
    return make_var_ff5_expected_returns_fn()

def _equal_weight_fn():
    # 1/N uses weights_fn, not expected_returns_fn — handled separately below
    return None


MODELS = [
    # (label,          fn_factory,      fallback_csv,                       use_weights_fn)
    ("1/N",            _equal_weight_fn, "data_clean/level0_results.csv",    True),
    ("FF5",            _ff5_fn,          "data_clean/level1_results_v1.csv",  False),
    ("VAR-FF5",        _var_ff5_fn,      "data_clean/level2_results_v1.csv",  False),
    ("Enet Huber",     _enet_huber_fn,   None,                                False),
    ("RF",             _rf_fn,           "data_clean/level4_rf_results.csv",  False),
]


# ── 1/N weights function ──────────────────────────────────────────────────────
def _make_equal_weight_fn() -> Callable:
    def equal_weight(ret_matrix):
        n = ret_matrix.shape[1]
        return pd.Series(np.ones(n) / n, index=ret_matrix.columns)
    return equal_weight


# ── Core sweep ────────────────────────────────────────────────────────────────
def run_cap_sweep(
    master: pd.DataFrame,
    label: str,
    fn_factory: Callable,
    use_weights_fn: bool,
    skip_missing: bool,
) -> list[dict]:
    """Run the cap sweep for one model. Returns list of result-row dicts."""
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"{'='*60}")

    # Build the function once (loads/caches the parquet)
    try:
        if use_weights_fn:
            weights_fn = _make_equal_weight_fn()
            expected_returns_fn = None
        else:
            expected_returns_fn = fn_factory()
            weights_fn = None
    except FileNotFoundError as e:
        if skip_missing:
            print(f"  [SKIP] {e}")
            return []
        raise

    rows = []
    for cap in CAP_LEVELS:
        cap_label = "uncapped" if cap is None else f"{int(cap*100)}%"
        print(f"\n  Cap: {cap_label}")

        results = run_backtest(
            master,
            start_year=START_YEAR,   start_month=START_MONTH,
            end_year=END_YEAR,       end_month=END_MONTH,
            risk_aversion=RISK_AVERSION,
            window=WINDOW,
            cost_bps=COST_BPS,
            expected_returns_fn=expected_returns_fn,
            weights_fn=weights_fn,
            max_turnover=cap,
        )

        if len(results) == 0:
            print(f"    [WARN] No results returned — skipping cap={cap_label}")
            continue

        s = evaluate_results(results, rf_series=None)

        # Realised turnover (skip month 0 full-construction spike)
        avg_to = results["turnover"].iloc[1:].mean()
        pct_months_capped = (
            (results["turnover"].iloc[1:] >= (cap or 999) - 0.001).mean() * 100
            if cap is not None else 0.0
        )

        rows.append({
            "model":              label,
            "cap_label":          cap_label,
            "cap_value":          cap if cap is not None else np.nan,
            "avg_turnover":       round(avg_to, 4),
            "pct_months_at_cap":  round(pct_months_capped, 1),
            "gross_sharpe":       s["annualized_gross_sharpe"],
            "net_sharpe":         s["annualized_net_sharpe"],
            "sortino":            s["annualized_sortino"],
            "gross_return":       s["annualized_gross_return"],
            "net_return":         s["annualized_net_return"],
            "volatility":         s["annualized_volatility"],
            "max_drawdown":       s["max_drawdown"],
            "avg_cost_bps":       s["avg_monthly_cost_bps"],
            "n_months":           len(results),
        })

        print(f"    Avg turnover: {avg_to*100:.1f}%  |  "
              f"Gross Sharpe: {s['annualized_gross_sharpe']:.3f}  |  "
              f"Net Sharpe: {s['annualized_net_sharpe']:.3f}  |  "
              f"Cost: {s['avg_monthly_cost_bps']:.1f} bps/mo")

    return rows


def load_fallback(label: str, fallback_csv: str) -> list[dict]:
    """
    For models where fn_factory failed or returned None, load the pre-computed
    CSV and add a single uncapped data point.
    """
    if fallback_csv is None or not os.path.exists(fallback_csv):
        return []
    print(f"  [Fallback] Loading pre-computed results from {fallback_csv}")
    results = pd.read_csv(fallback_csv)
    results.columns = [c.lower() for c in results.columns]

    # Clip to eval window
    results = results[
        ((results["year"] > START_YEAR) |
         ((results["year"] == START_YEAR) & (results["month"] >= START_MONTH))) &
        ((results["year"] < END_YEAR) |
         ((results["year"] == END_YEAR) & (results["month"] <= END_MONTH)))
    ].copy()

    if len(results) == 0:
        return []

    s = evaluate_results(results, rf_series=None)
    avg_to = results["turnover"].iloc[1:].mean()

    return [{
        "model":              label,
        "cap_label":          "uncapped (pre-computed)",
        "cap_value":          np.nan,
        "avg_turnover":       round(avg_to, 4),
        "pct_months_at_cap":  0.0,
        "gross_sharpe":       s["annualized_gross_sharpe"],
        "net_sharpe":         s["annualized_net_sharpe"],
        "sortino":            s["annualized_sortino"],
        "gross_return":       s["annualized_gross_return"],
        "net_return":         s["annualized_net_return"],
        "volatility":         s["annualized_volatility"],
        "max_drawdown":       s["max_drawdown"],
        "avg_cost_bps":       s["avg_monthly_cost_bps"],
        "n_months":           len(results),
    }]


# ── Primary table (uncapped only) ─────────────────────────────────────────────
def extract_primary_table(frontier_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full frontier, extract the uncapped row for each model.
    This is the main thesis comparison table.
    """
    uncapped = frontier_df[
        frontier_df["cap_label"].str.startswith("uncapped")
    ].copy()
    # One row per model (keep first if duplicates)
    uncapped = uncapped.drop_duplicates(subset=["model"], keep="first")
    # Order matches MODELS list
    order = [label for label, *_ in MODELS]
    uncapped["_sort"] = uncapped["model"].map({m: i for i, m in enumerate(order)})
    uncapped = uncapped.sort_values("_sort").drop(columns=["_sort"])
    return uncapped.reset_index(drop=True)


# ── Pretty-print primary table ────────────────────────────────────────────────
def print_primary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("PRIMARY RESULTS TABLE — Natural (Uncapped) Turnover, Net of 10 bps")
    print("=" * 90)
    cols = ["model", "avg_turnover", "gross_sharpe", "net_sharpe",
            "net_return", "volatility", "max_drawdown", "avg_cost_bps"]
    header = f"  {'Model':<14}  {'Turnover':>9}  {'Gross SR':>9}  {'Net SR':>8}  "
    header += f"{'Net Ret':>8}  {'Vol':>7}  {'MaxDD':>8}  {'Cost bps':>9}"
    print(header)
    print("  " + "-" * 84)
    for _, row in df.iterrows():
        print(
            f"  {row['model']:<14}  "
            f"{row['avg_turnover']*100:>8.1f}%  "
            f"{row['gross_sharpe']:>9.3f}  "
            f"{row['net_sharpe']:>8.3f}  "
            f"{row['net_return']*100:>7.2f}%  "
            f"{row['volatility']*100:>6.2f}%  "
            f"{row['max_drawdown']*100:>7.2f}%  "
            f"{row['avg_cost_bps']:>8.2f}"
        )
    print("=" * 90)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Turnover-Sharpe frontier sweep")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip models whose parquet is not yet available")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run only these model labels (e.g. --models RF 'Enet OLS')")
    args = parser.parse_args()

    print(f"\nLoading master panel from {MASTER_PATH} …")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master.columns   = [c.strip().lower() for c in master.columns]
    master["year"]   = pd.to_numeric(master["year"],   errors="coerce").astype(int)
    master["month"]  = pd.to_numeric(master["month"],  errors="coerce").astype(int)
    master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype(int)
    master["ret_adjusted"] = pd.to_numeric(master["ret_adjusted"], errors="coerce")
    print(f"Loaded: {master.shape}")

    print(f"\nEval window : {START_YEAR}-{START_MONTH:02d} → {END_YEAR}-{END_MONTH:02d}")
    print(f"Cost        : {COST_BPS} bps flat one-way")
    print(f"Cap levels  : {['uncapped' if c is None else f'{int(c*100)}%' for c in CAP_LEVELS]}")

    all_rows: list[dict] = []

    for label, fn_factory, fallback_csv, use_weights_fn in MODELS:
        if args.models and label not in args.models:
            continue

        try:
            rows = run_cap_sweep(
                master, label, fn_factory, use_weights_fn, args.skip_missing,
            )
        except Exception as e:
            warnings.warn(f"[{label}] Sweep failed: {e} — trying fallback CSV")
            rows = load_fallback(label, fallback_csv)

        if not rows and fallback_csv:
            rows = load_fallback(label, fallback_csv)

        all_rows.extend(rows)

    if not all_rows:
        print("\nNo results collected — nothing to save.")
        return

    frontier_df = pd.DataFrame(all_rows)

    # ── Save frontier ─────────────────────────────────────────────────────────
    frontier_path = OUTPUT_DIR / "frontier_results.csv"
    frontier_df.to_csv(frontier_path, index=False)
    print(f"\nFrontier saved → {frontier_path}  ({len(frontier_df)} rows)")

    # ── Primary table ─────────────────────────────────────────────────────────
    primary_df = extract_primary_table(frontier_df)
    primary_path = OUTPUT_DIR / "primary_table.csv"
    primary_df.to_csv(primary_path, index=False)
    print(f"Primary table saved → {primary_path}")

    print_primary_table(primary_df)

    # ── Frontier summary (ML models only) ────────────────────────────────────
    ml_labels = ["Enet Huber", "RF"]
    ml_df = frontier_df[frontier_df["model"].isin(ml_labels)].copy()
    if len(ml_df) > 0:
        print("\n" + "=" * 75)
        print("TURNOVER-SHARPE FRONTIER — ML Models")
        print("=" * 75)
        pivot = ml_df.pivot_table(
            index="cap_label", columns="model",
            values=["avg_turnover", "net_sharpe"],
            aggfunc="first",
        )
        # Reorder cap rows
        cap_order = ["uncapped"] + [f"{int(c*100)}%" for c in CAP_LEVELS if c is not None]
        pivot = pivot.reindex(
            [c for c in cap_order if c in pivot.index]
        )
        print(pivot.to_string())
        print("=" * 75)

    print("\nDone.")


if __name__ == "__main__":
    main()
