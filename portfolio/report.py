#!/usr/bin/env python3
"""
portfolio/report.py

Generates performance comparison tables (CSV) and publication-quality
PDF figures for all models in the MODELS registry.

Usage (from project root):
    python portfolio/report.py

Outputs written to reports/
    summary_table.csv
    calendar_year_returns.csv
    cost_drag_table.csv
    fig1_cumulative_nav.pdf
    fig2_rolling_sharpe.pdf
    fig3_drawdown.pdf
    fig4_turnover.pdf
    fig5_calendar_returns.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from portfolio.metrics import evaluate_results, compute_capm_alpha

# ============================================================
# Configuration
# ============================================================

MODELS = [
    ("level0",      "data_clean/level0_results.csv",          "Level 0",   "1/N Equal-Weight"),
    ("level1",      "data_clean/level1_results.csv",           "Level 1",   "FF5 Static"),
    ("level1_5",    "data_clean/level_1_5_results.csv",       "Level 1.5", "FF5 + Macro Ridge"),
    ("level2",      "data_clean/level2_results.csv",          "Level 2",   "VAR-FF5"),
    ("level3_huber","data_clean/level3_huber_results.csv",    "Level 3",   "Elastic Net (Huber)"),
    ("level4",      "data_clean/level4_rf_results.csv",       "Level 4",   "Random Forest"),
]

RECESSION_PATH  = "data_clean/USREC_Dates.csv"
REPORTS_DIR     = "reports"
ROLLING_WINDOW  = 60   # months for rolling Sharpe

# Restrict all models to this common evaluation window so cross-model
# comparisons are over identical periods.
EVAL_START_YEAR = 2005
EVAL_END_YEAR   = 2024

# Colorblind-friendly palette: gray baseline, then blue / orange / teal
COLORS = {
    "level0":       "#888888",
    "level1":       "#0077BB",
    "level1_5":     "#EE7733",
    "level2":       "#009988",
    "level3_huber": "#CC3311",
    "level4":       "#332288",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PCT = FuncFormatter(lambda y, _: f"{y*100:.0f}%")

# ============================================================
# Data loading
# ============================================================

def _to_dates(df):
    """Add a DatetimeIndex (first of each month); year/month kept as columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    )
    return df.set_index("date").sort_index()


def load_models():
    """Load all available model CSVs and compute summary metrics.

    Returns
    -------
    dict keyed by model key, each value:
        df      : date-indexed DataFrame (net_return, gross_return, turnover, cost, …)
        summary : dict from evaluate_results()
        short   : short display label   (e.g. "Level 0")
        long    : long display label    (e.g. "1/N Equal-Weight")
    """
    data = {}
    for key, path, short, long_label in MODELS:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            df = _to_dates(df)
            # Clip to the common evaluation window
            df = df[(df["year"] >= EVAL_START_YEAR) & (df["year"] <= EVAL_END_YEAR)]
            summary = evaluate_results(df, rf_series=None)
            capm    = compute_capm_alpha(df.reset_index())
            data[key] = {"df": df, "summary": summary, "capm": capm, "short": short, "long": long_label}
        except Exception as exc:
            print(f"  [error] {path}: {exc}")
    return data


# ============================================================
# Recession shading
# ============================================================

def load_recession_spans():
    """Parse USREC_Dates.csv into a list of (start, end) datetime pairs."""
    if not os.path.exists(RECESSION_PATH):
        print(f"  [warn] {RECESSION_PATH} not found — recession shading disabled")
        return []
    rec = pd.read_csv(RECESSION_PATH, parse_dates=["observation_date"])
    rec = rec.set_index("observation_date")["USREC"].sort_index()

    spans, in_rec, start = [], False, None
    for date, val in rec.items():
        if val == 1 and not in_rec:
            start, in_rec = date, True
        elif val == 0 and in_rec:
            spans.append((start, date))
            in_rec = False
    if in_rec:
        spans.append((start, rec.index[-1] + pd.offsets.MonthEnd(1)))
    return spans


def _shade(ax, spans, alpha=0.12):
    for start, end in spans:
        ax.axvspan(start, end, alpha=alpha, color="gray", zorder=0)


def _date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))


# ============================================================
# Tables
# ============================================================

_SUMMARY_SPEC = [
    # (key, display label, python format, unit suffix, multiplier, source)
    # source: "summary" or "capm"
    ("annualized_net_sharpe",         "Net Sharpe (ann.)",       "{:.3f}", "",     1,    "summary"),
    ("annualized_gross_sharpe",       "Gross Sharpe (ann.)",     "{:.3f}", "",     1,    "summary"),
    ("annualized_sortino",            "Sortino (ann.)",          "{:.3f}", "",     1,    "summary"),
    ("annualized_net_return",         "Net Return (ann.)",       "{:.2f}", "%",  100,    "summary"),
    ("annualized_volatility",         "Volatility (ann.)",       "{:.2f}", "%",  100,    "summary"),
    ("annualized_downside_deviation", "Downside Dev. (ann.)",    "{:.4f}", "",     1,    "summary"),
    ("max_drawdown",                  "Max Drawdown",            "{:.2f}", "%",  100,    "summary"),
    ("avg_monthly_turnover",          "Avg Turnover/Month",      "{:.1f}", "%",  100,    "summary"),
    ("avg_monthly_cost_bps",          "Avg Cost/Month",          "{:.2f}", "bps",  1,    "summary"),
    ("capm_alpha_pct",                "CAPM Alpha (ann. %)",     "{:.2f}", "%",    1,    "capm"),
    ("capm_alpha_tstat",              "Alpha t-stat",            "{:.2f}", "",     1,    "capm"),
    ("capm_beta",                     "Market Beta",             "{:.3f}", "",     1,    "capm"),
    ("capm_r2",                       "CAPM R²",                 "{:.3f}", "",     1,    "capm"),
]


def _fmt_cell(val, fmt, unit, mult):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return fmt.format(val * mult) + unit


def save_summary_table(models_data):
    """Save summary_table.csv and print the table to stdout."""
    ordered = [models_data[k] for k, *_ in MODELS if k in models_data]
    if not ordered:
        return

    # --- CSV ---
    csv_rows = {}
    for key, label, fmt, unit, mult, src in _SUMMARY_SPEC:
        csv_rows[label] = {
            m["short"]: (
                round(m[src][key] * mult, 4)
                if m[src].get(key) is not None and not np.isnan(m[src].get(key, float("nan")))
                else np.nan
            )
            for m in ordered
        }
    csv_df = pd.DataFrame(csv_rows).T
    csv_df.index.name = "Metric"
    path = os.path.join(REPORTS_DIR, "summary_table.csv")
    csv_df.to_csv(path)
    print(f"  Saved {path}")

    # --- stdout ---
    label_w = max(len(r[1]) for r in _SUMMARY_SPEC) + 2
    col_w   = 15
    total_w = label_w + len(ordered) * (col_w + 2) + 4
    sep, thin = "=" * total_w, "-" * total_w

    print("\n" + sep)
    print("  MODEL PERFORMANCE SUMMARY")
    print(sep)
    print(f"  {'Metric':<{label_w}}" + "".join(f"  {m['short']:>{col_w}}" for m in ordered))
    print(f"  {'':<{label_w}}"       + "".join(f"  {m['long']:>{col_w}}"  for m in ordered))
    print(thin)
    for key, label, fmt, unit, mult, src in _SUMMARY_SPEC:
        row = f"  {label:<{label_w}}"
        for m in ordered:
            row += f"  {_fmt_cell(m[src].get(key), fmt, unit, mult):>{col_w}}"
        print(row)
    print(sep + "\n")


def save_calendar_year_table(models_data):
    """Save calendar_year_returns.csv (year × model)."""
    records = {}
    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df = models_data[key]["df"]
        ann = df.groupby(df.index.year)["net_return"].apply(
            lambda x: (1 + x).prod() - 1
        )
        records[models_data[key]["short"]] = ann

    if not records:
        return None
    cal = pd.DataFrame(records)
    cal.index.name = "Year"
    path = os.path.join(REPORTS_DIR, "calendar_year_returns.csv")
    cal.to_csv(path, float_format="%.4f")
    print(f"  Saved {path}")
    return cal


def save_cost_drag_table(models_data):
    """Save cost_drag_table.csv comparing gross vs. net Sharpe per model."""
    ordered = [models_data[k] for k, *_ in MODELS if k in models_data]
    if not ordered:
        return

    rows = []
    for m in ordered:
        s = m["summary"]
        gross = s.get("annualized_gross_sharpe", np.nan)
        net   = s.get("annualized_net_sharpe",   np.nan)
        drag  = net - gross if not (np.isnan(gross) or np.isnan(net)) else np.nan
        rows.append({
            "Model":              m["short"],
            "Description":        m["long"],
            "Gross Sharpe":       round(gross, 3),
            "Net Sharpe":         round(net,   3),
            "Sharpe Drag":        round(drag,  3),
            "Avg Turnover/Mo (%)": round(s.get("avg_monthly_turnover", np.nan) * 100, 1),
            "Avg Cost/Mo (bps)":  round(s.get("avg_monthly_cost_bps", np.nan), 2),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(REPORTS_DIR, "cost_drag_table.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ============================================================
# Figures
# ============================================================

def fig_cumulative_nav(models_data, rec_spans):
    """Fig 1: Cumulative net NAV (log scale), all models, recession shading."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df  = models_data[key]["df"]
        nav = (1 + df["net_return"]).cumprod()
        ax.plot(nav.index, nav.values,
                color=COLORS[key], linewidth=1.5,
                label=f"{models_data[key]['short']}  {models_data[key]['long']}")

    _shade(ax, rec_spans)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.1f}"))
    ax.set_ylabel("Net Asset Value (log scale, $1 = Jan 2005)")
    ax.set_title("Cumulative Net Returns — All Models (2005–2024)")
    ax.legend(loc="upper left")
    _date_axis(ax)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "fig1_cumulative_nav.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_rolling_sharpe(models_data, rec_spans):
    """Fig 2: Rolling 60-month net Sharpe ratio, recession shading."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df = models_data[key]["df"]
        rs = df["net_return"].rolling(ROLLING_WINDOW).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 1e-10 else np.nan,
            raw=True,
        )
        ax.plot(rs.index, rs.values,
                color=COLORS[key], linewidth=1.5,
                label=f"{models_data[key]['short']}  {models_data[key]['long']}")

    _shade(ax, rec_spans)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylabel(f"Rolling Sharpe Ratio ({ROLLING_WINDOW}-month)")
    ax.set_title(f"Rolling {ROLLING_WINDOW}-Month Net Sharpe Ratio (2005–2024)")
    ax.legend(loc="upper left")
    _date_axis(ax)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "fig2_rolling_sharpe.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_drawdown(models_data, rec_spans):
    """Fig 3: Drawdown from peak (net NAV), all models, recession shading."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df   = models_data[key]["df"]
        nav  = (1 + df["net_return"]).cumprod()
        peak = nav.cummax()
        dd   = (nav - peak) / peak
        ax.plot(dd.index, dd.values,
                color=COLORS[key], linewidth=1.5,
                label=f"{models_data[key]['short']}  {models_data[key]['long']}")

    _shade(ax, rec_spans)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Drawdown from Peak")
    ax.set_title("Drawdown from Peak — All Models (2005–2024)")
    ax.legend(loc="lower left")
    _date_axis(ax)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "fig3_drawdown.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_turnover(models_data, rec_spans):
    """Fig 4: Rolling 12-month average monthly turnover, recession shading."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df = models_data[key]["df"]
        # NaN out the first month (full portfolio construction, turnover = 1.0)
        to = df["turnover"].copy().astype(float)
        to.iloc[0] = np.nan
        rolling_to = to.rolling(12).mean() * 100
        ax.plot(rolling_to.index, rolling_to.values,
                color=COLORS[key], linewidth=1.5,
                label=f"{models_data[key]['short']}  {models_data[key]['long']}")

    _shade(ax, rec_spans)
    ax.set_ylabel("Avg Monthly Turnover, % (12-month rolling)")
    ax.set_title("Portfolio Turnover — Rolling 12-Month Average (2005–2024)")
    ax.legend(loc="upper left")
    _date_axis(ax)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "fig4_turnover.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_calendar_returns(models_data):
    """Fig 5: Grouped bar chart of annual net returns, all models."""
    annual = {}
    for key, *_ in MODELS:
        if key not in models_data:
            continue
        df = models_data[key]["df"]
        annual[key] = df.groupby(df.index.year)["net_return"].apply(
            lambda x: (1 + x).prod() - 1
        )

    if not annual:
        return

    years    = sorted(next(iter(annual.values())).index.tolist())
    keys     = [k for k, *_ in MODELS if k in annual]
    n        = len(keys)
    bar_w    = 0.18
    offsets  = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_w
    x        = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(20, 6))
    for i, key in enumerate(keys):
        vals = [annual[key].get(y, np.nan) * 100 for y in years]
        ax.bar(x + offsets[i], vals, width=bar_w,
               color=COLORS[key], alpha=0.85,
               label=f"{models_data[key]['short']}  {models_data[key]['long']}")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=90, fontsize=7)
    ax.set_ylabel("Annual Net Return (%)")
    ax.set_title("Calendar Year Net Returns — All Models (2005–2024)")
    ax.legend(loc="upper left")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "fig5_calendar_returns.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Entry point
# ============================================================

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading model results...")
    models_data = load_models()
    if not models_data:
        print("No model results found. Run at least one model first.")
        return

    print("Loading recession dates...")
    rec_spans = load_recession_spans()
    print(f"  {len(rec_spans)} recession periods loaded")

    print("\nBuilding tables...")
    save_summary_table(models_data)
    save_calendar_year_table(models_data)
    save_cost_drag_table(models_data)

    print("\nBuilding figures...")
    fig_cumulative_nav(models_data, rec_spans)
    fig_rolling_sharpe(models_data, rec_spans)
    fig_drawdown(models_data, rec_spans)
    fig_turnover(models_data, rec_spans)
    fig_calendar_returns(models_data)

    print(f"\nDone. All outputs written to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
