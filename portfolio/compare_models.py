import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# Model Comparison Table
# portfolio/compare_models.py
#
# Loads each model's saved results CSV from data_clean/ and
# prints a single side-by-side performance table.
# Missing result files are silently skipped — run any subset
# of models and this script reflects exactly what exists.
#
# Usage:
#   python portfolio/compare_models.py
# ============================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from portfolio.metrics import evaluate_results

# ============================================================
# Model registry — ordered for display
# ============================================================

MODELS = [
    ("level0",   "data_clean/level0_results.csv",     "Level 0",   "1/N Equal-Weight"),
    ("level1",   "data_clean/level1_results.csv",     "Level 1",   "FF5 Static"),
    ("level1_5", "data_clean/level_1_5_results.csv",  "Level 1.5", "FF5 + Macro Ridge"),
    ("level2",   "data_clean/level2_results.csv",     "Level 2",   "VAR-FF5"),
]

# ============================================================
# Metric display spec: (key, label, format, unit, multiplier)
#   multiplier: applied before formatting (e.g. 100 for %)
# ============================================================

METRICS = [
    ("annualized_net_sharpe",        "Net Sharpe (ann.)",      "{:.3f}", "",      1),
    ("annualized_gross_sharpe",      "Gross Sharpe (ann.)",    "{:.3f}", "",      1),
    ("annualized_sortino",           "Sortino (ann.)",         "{:.3f}", "",      1),
    ("annualized_net_return",        "Net Return (ann.)",      "{:.2f}", "%",   100),
    ("annualized_volatility",        "Volatility (ann.)",      "{:.2f}", "%",   100),
    ("annualized_downside_deviation","Downside Dev. (ann.)",   "{:.4f}", "",      1),
    ("max_drawdown",                 "Max Drawdown",           "{:.2f}", "%",   100),
    ("avg_monthly_turnover",         "Avg Turnover/Month",     "{:.1f}", "%",   100),
    ("avg_monthly_cost_bps",         "Avg Cost/Month",         "{:.2f}", " bps",  1),
]


def _fmt(value, fmt, unit, multiplier):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return fmt.format(value * multiplier) + unit


def build_table():
    # Load and evaluate each available model
    available = []
    for key, path, short_label, long_label in MODELS:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            summary = evaluate_results(df, rf_series=None)
            available.append((short_label, long_label, summary))
        except Exception as e:
            print(f"  [error] {path}: {e}")

    if not available:
        print("No model results found. Run at least one model first.")
        return

    # --------------------------------------------------------
    # Column widths
    # --------------------------------------------------------
    label_w  = max(len(m[0]) for m in METRICS) + 2   # metric label column
    col_w    = 14                                      # each model column
    n_models = len(available)
    total_w  = label_w + n_models * (col_w + 2) + 2

    sep  = "=" * total_w
    thin = "-" * total_w

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    print("\n" + sep)
    print("  MODEL PERFORMANCE COMPARISON")
    print(sep)

    # Row 1: short labels (Level 0, Level 1, ...)
    header1 = f"  {'Metric':<{label_w}}"
    for short_label, _, _ in available:
        header1 += f"  {short_label:>{col_w}}"
    print(header1)

    # Row 2: long labels (1/N Equal-Weight, FF5 Static, ...)
    header2 = f"  {'':<{label_w}}"
    for _, long_label, _ in available:
        header2 += f"  {long_label:>{col_w}}"
    print(header2)

    print(thin)

    # --------------------------------------------------------
    # Metric rows
    # --------------------------------------------------------
    for key, label, fmt, unit, multiplier in METRICS:
        row = f"  {label:<{label_w}}"
        for _, _, summary in available:
            val = summary.get(key, None)
            row += f"  {_fmt(val, fmt, unit, multiplier):>{col_w}}"
        print(row)

    print(sep + "\n")


if __name__ == "__main__":
    build_table()
