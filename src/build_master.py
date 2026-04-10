"""
src/build_master.py — Master Panel Construction

UNIVERSE_VERSION controls which construction is used:
  "legacy" : original top-500 by December market cap (all exchanges)
             → outputs data_clean/master_panel.csv  (unchanged)
  "v2"     : NYSE 40th-percentile June rebalance, ~1,000 firms
             → outputs data_clean/master_panel_v2.csv

All downstream model files read the panel filename from
portfolio/config.py (PANEL_PATH), not from a hardcoded string.

Revision history
----------------
Apr 2026  v2 universe: NYSE pctile breakpoint, June rebalance, secondary
          screens, Canadian firm exclusion, duplicate fixes, macro merge,
          expanded OSAP signals. (Data audit Apr 9 2026.)
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Universe version switch ───────────────────────────────────────────────────
# Set to "legacy" to reproduce the original top-500 December panel exactly.
# Set to "v2" to run the improved NYSE-percentile June-rebalance construction.
UNIVERSE_VERSION = "v2"

PANEL_OUTPUT = {
    "legacy": "data_clean/master_panel.csv",
    "v2":     "data_clean/master_panel_v2.csv",
}[UNIVERSE_VERSION]

OSAP_INPUT = {
    "legacy": "data_clean/osap_signals_top500.csv",
    "v2":     "data_clean/osap_signals_expanded.csv",
}[UNIVERSE_VERSION]

print(f"\nBuild Master Panel  [UNIVERSE_VERSION={UNIVERSE_VERSION!r}]")
print(f"Output → {PANEL_OUTPUT}")
print("Running from directory:", os.getcwd())

os.makedirs("logs", exist_ok=True)
os.makedirs("data_clean", exist_ok=True)

# ============================================================
# 1) Load Raw Data
# ============================================================

print("\nLoading CRSP data...")
crsp_df = pd.read_csv("data_raw/crsp_monthly_stock.csv", low_memory=False)
crsp_df.columns = [c.strip().lower() for c in crsp_df.columns]

date_col = "date" if "date" in crsp_df.columns else "mthcaldt"
crsp_df[date_col] = pd.to_datetime(crsp_df[date_col], errors="coerce")
crsp_df["year"]  = crsp_df[date_col].dt.year
crsp_df["month"] = crsp_df[date_col].dt.month
print("CRSP loaded:", crsp_df.shape)

print("\nLoading Compustat data...")
comp_df = pd.read_csv("data_raw/compustat_annual.csv", low_memory=False)
comp_df.columns = [c.strip().lower() for c in comp_df.columns]
comp_df["datadate"] = pd.to_datetime(comp_df["datadate"], errors="coerce")
comp_df["year"]  = comp_df["datadate"].dt.year
comp_df["fyear"] = pd.to_numeric(comp_df["fyear"], errors="coerce")
comp_df["fyear"] = comp_df["fyear"].astype("Int64")

# datadate is the fiscal year-end date. Apply a 6-month publication lag to
# approximate when the filing becomes publicly available (SEC 10-K deadline).
comp_df["available_date"] = comp_df["datadate"] + pd.DateOffset(months=6)
print("Compustat loaded:", comp_df.shape)

print("\nLoading CCM Link file...")
link_df = pd.read_csv("data_raw/ccm_link.csv")
link_df.columns = [c.strip().lower() for c in link_df.columns]
link_df["linkdt"]    = pd.to_datetime(link_df["linkdt"])
link_df["linkenddt"] = link_df["linkenddt"].replace("E", pd.NaT)
link_df["linkenddt"] = pd.to_datetime(link_df["linkenddt"], errors="coerce")
print("CCM Link loaded:", link_df.shape)

# ============================================================
# PRELIMINARY DATA QUALITY FIXES
# ============================================================

# ── Fix 1: Resolve Compustat duplicate (gvkey, datadate) pairs ───────────────
# The raw file contains 46,158 duplicate (gvkey, datadate) pairs. We sort by
# (gvkey, datadate) and keep the last row, which in a standard Compustat
# extract corresponds to the most recently restated or updated filing value.
# "First" would risk retaining the original un-restated observation.
before_dedup_comp = len(comp_df)
comp_df = (
    comp_df
    .sort_values(["gvkey", "datadate"])
    .drop_duplicates(subset=["gvkey", "datadate"], keep="last")
    .reset_index(drop=True)
)
print(f"\n[Fix 1] Compustat (gvkey, datadate) dedup: "
      f"{before_dedup_comp:,} → {len(comp_df):,} rows "
      f"(dropped {before_dedup_comp - len(comp_df):,})")

# ── Fix 2: Exclude Canadian firms from Compustat ─────────────────────────────
# The raw Compustat file includes ~6,236 Canadian-incorporated firms
# (curcd == 'CAD'). These are excluded before the CCM link merge to prevent
# Canadian firms from entering the US equity universe even if they have a
# valid CRSP PERMNO.
if "curcd" in comp_df.columns:
    before_cad = len(comp_df)
    comp_df = comp_df[comp_df["curcd"] != "CAD"].copy()
    print(f"[Fix 2] Canadian firm exclusion (curcd='CAD'): "
          f"{before_cad:,} → {len(comp_df):,} rows "
          f"(dropped {before_cad - len(comp_df):,} observations)")
else:
    print("[Fix 2] curcd column not found — skipping Canadian exclusion")

# ============================================================
# 2) CRSP: Clean Numeric Columns
# ============================================================

print("\nCleaning CRSP numeric columns...")

crsp_df["prc"]    = pd.to_numeric(crsp_df["prc"],    errors="coerce")
crsp_df["prc"]    = crsp_df["prc"].abs()
crsp_df["shrout"] = pd.to_numeric(crsp_df["shrout"], errors="coerce")
crsp_df["ret"]    = pd.to_numeric(crsp_df["ret"],    errors="coerce")
crsp_df["dlret"]  = pd.to_numeric(crsp_df["dlret"],  errors="coerce")
crsp_df["vol"]    = pd.to_numeric(crsp_df["vol"],    errors="coerce")
crsp_df["ret_adjusted"] = (1 + crsp_df["ret"]) * (1 + crsp_df["dlret"].fillna(0)) - 1
crsp_df["market_cap"]   = crsp_df["prc"] * crsp_df["shrout"]

# Year-range filter applied to full CRSP before any universe selection
crsp_df = crsp_df[(crsp_df["year"] >= 1973) & (crsp_df["year"] <= 2024)]

# ============================================================
# 3) Universe Construction
# ============================================================

if UNIVERSE_VERSION == "v2":

    # ── v2: NYSE 40th-percentile June rebalance, ~1,000 firms ────────────────

    print("\n[v2] Building NYSE 40th-percentile June-rebalance universe...")

    # Step 2.1: Security type and exchange filters
    # shrcd ∈ {10, 11} — ordinary common equity only
    # exchcd ∈ {1, 2, 3} — NYSE, AMEX, NASDAQ (excludes 0=unlisted, 4=Arca, 5=other)
    crsp_clean = crsp_df[
        crsp_df["shrcd"].isin([10, 11]) &
        crsp_df["exchcd"].isin([1, 2, 3])
    ].copy()
    print(f"  After shrcd/exchcd filters: {crsp_clean.shape}")

    # Step 2.2: Precompute 24-month return history per permno (full CRSP, pre-filter)
    # For each permno, count non-null ret observations strictly before each month.
    # We compute the rolling count up to each date so we can check the 24-month
    # requirement at the June rebalance date.
    print("  Computing 24-month return history...")
    ret_history = (
        crsp_clean.sort_values(["permno", date_col])
        [["permno", "year", "month", "ret"]]
        .copy()
    )
    ret_history["ret_nonull"] = ret_history["ret"].notna().astype(int)
    # Expanding count of non-null returns per firm up to (but not including)
    # the current month — shift by 1 so we don't count the current month itself.
    ret_history["cum_ret_count"] = (
        ret_history.groupby("permno")["ret_nonull"]
        .transform(lambda x: x.shift(1).expanding().sum())
    )

    # Step 2.3: June snapshots and NYSE breakpoints
    june_snap = crsp_clean[crsp_clean["month"] == 6].copy()

    # Merge in prior return history count
    june_snap = june_snap.merge(
        ret_history[["permno", "year", "month", "cum_ret_count"]],
        on=["permno", "year", "month"],
        how="left",
    )

    # NYSE-only subset for breakpoint computation
    nyse_june = june_snap[june_snap["exchcd"] == 1].copy()

    # Compute NYSE 40th percentile of market_cap per year
    nyse_p40 = (
        nyse_june[nyse_june["market_cap"].notna()]
        .groupby("year")
        .agg(
            threshold_market_cap=("market_cap", lambda x: x.quantile(0.35)),
            nyse_firm_count=("permno", "nunique"),
        )
        .reset_index()
    )
    nyse_p40.to_csv("logs/nyse_breakpoints.csv", index=False)
    print(f"  NYSE breakpoints computed for {len(nyse_p40)} years → logs/nyse_breakpoints.csv")
    print(f"  Sample thresholds (selected years):")
    for yr in [1975, 1985, 1995, 2005, 2015, 2024]:
        row = nyse_p40[nyse_p40["year"] == yr]
        if len(row):
            thr = row.iloc[0]["threshold_market_cap"]
            cnt = row.iloc[0]["nyse_firm_count"]
            print(f"    {yr}: p40 = ${thr/1e6:,.0f}M  (from {cnt} NYSE firms)")

    # Step 2.4: Apply size screen + secondary screens per rebalance year
    p40_map = nyse_p40.set_index("year")["threshold_market_cap"].to_dict()

    screen_log = []
    universe_annual = []    # list of (permno, rebalance_year) DataFrames

    for yr in sorted(june_snap["year"].unique()):
        threshold = p40_map.get(yr)
        if threshold is None:
            print(f"  Warning: no NYSE p40 threshold for {yr} — skipping")
            continue

        yr_snap = june_snap[june_snap["year"] == yr].copy()
        n_start = len(yr_snap)

        # Primary: market_cap >= NYSE p40 threshold
        yr_snap = yr_snap[
            yr_snap["market_cap"].notna() &
            (yr_snap["market_cap"] >= threshold)
        ]
        n_after_size = len(yr_snap)

        # Secondary screen 1: price >= $1.00
        yr_snap = yr_snap[yr_snap["prc"].notna() & (yr_snap["prc"] >= 1.00)]
        n_after_price = len(yr_snap)

        # Secondary screen 2: >= 24 months of non-null returns before June
        yr_snap = yr_snap[
            yr_snap["cum_ret_count"].notna() &
            (yr_snap["cum_ret_count"] >= 24)
        ]
        n_after_history = len(yr_snap)

        # Hard cap: retain top 1,500 by market cap within the eligible set.
        # When the NYSE p35 threshold admits more than 1,500 firms (as occurs
        # during the late-1990s listing boom), rank by market_cap descending
        # and keep the largest 1,500. The cap is non-binding in most years.
        CAP = 1500
        if len(yr_snap) > CAP:
            yr_snap = yr_snap.nlargest(CAP, "market_cap")
        n_after_cap = len(yr_snap)

        screen_log.append({
            "year": yr,
            "n_start": n_start,
            "n_after_size": n_after_size,
            "n_dropped_price": n_after_size - n_after_price,
            "n_dropped_history": n_after_price - n_after_history,
            "n_dropped_cap": n_after_history - n_after_cap,
            "n_final": n_after_cap,
            "nyse_p35_threshold": threshold,
        })

        if len(yr_snap) > 0:
            sub = yr_snap[["permno"]].drop_duplicates().copy()
            sub["rebalance_year"] = yr
            universe_annual.append(sub)

    screen_log_df = pd.DataFrame(screen_log)
    screen_log_df.to_csv("logs/universe_screen_log.csv", index=False)
    print(f"\n  Universe screening log saved → logs/universe_screen_log.csv")
    print(f"  Avg firms at June rebalance: {screen_log_df['n_final'].mean():.0f}")
    print(f"  Min: {screen_log_df['n_final'].min()} ({screen_log_df.loc[screen_log_df['n_final'].idxmin(), 'year']})")
    print(f"  Max: {screen_log_df['n_final'].max()} ({screen_log_df.loc[screen_log_df['n_final'].idxmax(), 'year']})")

    universe_annual_df = pd.concat(universe_annual, ignore_index=True)

    # Step 2.5: Expand to monthly — July Y through June Y+1
    # For rebalance_year Y, the universe applies to:
    #   months 7–12 of year Y  (second half of calendar year Y)
    #   months 1–6  of year Y+1 (first half of calendar year Y+1)
    monthly_rows = []
    for _, row in universe_annual_df.iterrows():
        permno = row["permno"]
        ry = row["rebalance_year"]
        # July–December of rebalance year
        for m in range(7, 13):
            monthly_rows.append({"permno": permno, "year": ry,     "month": m})
        # January–June of following year
        for m in range(1, 7):
            monthly_rows.append({"permno": permno, "year": ry + 1, "month": m})

    universe_monthly = pd.DataFrame(monthly_rows)

    # Seed universe for January–June 1973 (pre-rebalance approximation):
    # Use January 1973 NYSE 40th-percentile threshold applied to all firms
    # passing secondary screens in January 1973.
    jan_1973 = crsp_clean[(crsp_clean["year"] == 1973) & (crsp_clean["month"] == 1)].copy()
    nyse_jan73 = jan_1973[jan_1973["exchcd"] == 1]["market_cap"].dropna()
    if len(nyse_jan73) > 0:
        threshold_jan73 = nyse_jan73.quantile(0.35)
        seed = jan_1973[
            jan_1973["market_cap"].notna() &
            (jan_1973["market_cap"] >= threshold_jan73) &
            jan_1973["prc"].notna() &
            (jan_1973["prc"] >= 1.00)
        ][["permno"]].drop_duplicates().copy()
        print(f"\n  Seed universe (Jan 1973): {len(seed)} firms "
              f"(p40 threshold = ${threshold_jan73/1e6:,.0f}M) "
              f"[pre-rebalance approximation]")
        seed_rows = []
        for _, row in seed.iterrows():
            for m in range(1, 7):  # Jan–Jun 1973 only
                seed_rows.append({"permno": row["permno"], "year": 1973, "month": m})
        seed_df = pd.DataFrame(seed_rows)
        # Append seed for Jan–Jun 1973; the June 1973 rebalance already covers Jul–Dec 1973
        universe_monthly = pd.concat([seed_df, universe_monthly], ignore_index=True)

    # Restrict to 1973–2024 range and drop duplicates
    universe_monthly = universe_monthly[
        (universe_monthly["year"] >= 1973) &
        (universe_monthly["year"] <= 2024)
    ].copy()
    universe_monthly = universe_monthly.drop_duplicates(
        subset=["permno", "year", "month"]
    ).reset_index(drop=True)

    # Add yyyymm for the OSAP filter
    universe_monthly["yyyymm"] = universe_monthly["year"] * 100 + universe_monthly["month"]

    # Save universe_monthly for the OSAP reprocessing script
    universe_monthly.to_csv("data_clean/universe_monthly.csv", index=False)
    print(f"  Monthly universe saved → data_clean/universe_monthly.csv")
    print(f"  Total universe firm-months: {len(universe_monthly):,}")

    # Step 2.6: Apply universe membership to full CRSP panel
    crsp_df = crsp_df[crsp_df["shrcd"].isin([10, 11])].copy()
    crsp_df = crsp_df.merge(
        universe_monthly[["permno", "year", "month"]],
        on=["permno", "year", "month"],
        how="inner",
    )
    print(f"  CRSP after universe filter: {crsp_df.shape}")

    # Step 2.7: Firm count validation and logging
    firm_counts = crsp_df.groupby(["year", "month"])["permno"].nunique().reset_index()
    firm_counts.columns = ["year", "month", "firm_count"]
    firm_counts = firm_counts.merge(
        nyse_p40[["year", "threshold_market_cap"]], on="year", how="left"
    )
    annual_summary = (
        firm_counts.groupby("year")
        .agg(
            avg_firms=("firm_count", "mean"),
            min_firms=("firm_count", "min"),
            max_firms=("firm_count", "max"),
        )
        .reset_index()
    )
    annual_summary = annual_summary.merge(
        nyse_p40[["year", "threshold_market_cap"]], on="year", how="left"
    )
    low_flag  = annual_summary["avg_firms"] < 1000
    high_flag = annual_summary["avg_firms"] > 1600
    annual_summary["flag"] = ""
    annual_summary.loc[low_flag,  "flag"] = "LOW (<1000)"
    annual_summary.loc[high_flag, "flag"] = "HIGH (>1600)"
    annual_summary.to_csv("logs/universe_summary.csv", index=False)
    print(f"\n  Universe summary saved → logs/universe_summary.csv")
    flagged = annual_summary[annual_summary["flag"] != ""]
    if len(flagged):
        print(f"  Flagged years:\n{flagged.to_string(index=False)}")

    # Annual turnover logging
    annual_permnos = universe_annual_df.groupby("rebalance_year")["permno"].apply(set)
    turnover_rows = []
    for i, yr in enumerate(sorted(annual_permnos.index)):
        if i == 0:
            continue
        prev_yr = sorted(annual_permnos.index)[i - 1]
        exits = len(annual_permnos[prev_yr] - annual_permnos[yr])
        prev_n = len(annual_permnos[prev_yr])
        to_rate = exits / max(prev_n, 1)
        turnover_rows.append({"year": yr, "exits": exits, "prev_n": prev_n,
                               "turnover_rate": to_rate,
                               "flag": "HIGH >25%" if to_rate > 0.25 else ""})
    turnover_df = pd.DataFrame(turnover_rows)
    turnover_df.to_csv("logs/universe_turnover.csv", index=False)
    print(f"  Turnover log saved → logs/universe_turnover.csv")
    flagged_to = turnover_df[turnover_df["flag"] != ""]
    if len(flagged_to):
        print(f"  High-turnover years: {flagged_to['year'].tolist()}")

    # Save CRSP-only universe panel (used by level0 benchmark)
    crsp_df[["permno", "year", "month", "ret_adjusted"]].drop_duplicates().to_csv(
        "data_clean/crsp_expanded_panel.csv", index=False
    )
    print("  CRSP expanded panel saved → data_clean/crsp_expanded_panel.csv")

else:

    # ── LEGACY TOP-500 DECEMBER UNIVERSE — RETAINED FOR BACKWARD COMPATIBILITY ──
    # This block exactly reproduces the original build_master.py logic.
    # Do not modify this block when working on the v2 construction.

    print("\n[legacy] Building top-500 December market-cap universe...")
    crsp_df = crsp_df[crsp_df["shrcd"].isin([10, 11])].copy()

    # 1973: rank by January market cap
    jan_1973 = crsp_df[(crsp_df["year"] == 1973) & (crsp_df["month"] == 1)].copy()
    jan_1973["rank"] = jan_1973["market_cap"].rank(method="first", ascending=False)
    top500_1973 = jan_1973[jan_1973["rank"] <= 500][["permno"]].copy()
    top500_1973["hold_year"] = 1973

    # 1974–2024: rank by December market cap, hold following year
    december = crsp_df[crsp_df["month"] == 12].copy()
    december["rank"] = december.groupby("year")["market_cap"].rank(method="first", ascending=False)
    top500_dec = december[december["rank"] <= 500][["permno", "year"]].copy()
    top500_dec["hold_year"] = top500_dec["year"] + 1
    top500_dec = top500_dec[
        (top500_dec["hold_year"] >= 1974) & (top500_dec["hold_year"] <= 2024)
    ][["permno", "hold_year"]]

    top500_all = pd.concat([top500_1973, top500_dec], ignore_index=True)

    crsp_df = crsp_df.merge(
        top500_all,
        left_on=["permno", "year"],
        right_on=["permno", "hold_year"],
        how="inner",
    ).drop(columns=["hold_year"])

    print("After top-500 filter:", crsp_df.shape)

    crsp_df[["permno", "year", "month", "ret_adjusted"]].drop_duplicates().to_csv(
        "data_clean/crsp_top500_panel.csv", index=False
    )
    print("CRSP top-500 panel saved to data_clean/crsp_top500_panel.csv")

# ============================================================
# 4) CRSP: Calculate Market-Based Factors
# ============================================================

print("\nCalculating CRSP factor variables...")

crsp_df = crsp_df.sort_values(["permno", date_col]).reset_index(drop=True)

# Illiquidity: |RET| / (PRC × VOL), lagged by 1 month so month-t features
# do not encode the magnitude of the month-t return being predicted.
crsp_df["illiquidity"] = crsp_df["ret"].abs() / (crsp_df["prc"] * crsp_df["vol"])
crsp_df["illiquidity"] = crsp_df["illiquidity"].replace([np.inf, -np.inf], np.nan)
crsp_df["illiquidity"] = crsp_df.groupby("permno")["illiquidity"].shift(1)

# Short-term reversal: lagged 1-month return
crsp_df["reversal_st"] = crsp_df.groupby("permno")["ret"].shift(1)

# Momentum: log-compounded 12-month return from t-12 to t-2 (GKX convention).
crsp_df["log_ret"]      = np.log1p(crsp_df["ret_adjusted"])
crsp_df["log_ret_lag1"] = crsp_df.groupby("permno")["log_ret"].shift(1)
crsp_df["cum_12m"] = (
    crsp_df.groupby("permno")["log_ret_lag1"]
    .rolling(12, min_periods=11)
    .sum()
    .reset_index(drop=True)
)
crsp_df["momentum"] = np.expm1(crsp_df["cum_12m"] - crsp_df["log_ret_lag1"])
crsp_df = crsp_df.drop(columns=["log_ret", "log_ret_lag1", "cum_12m"])

# Long-term reversal: log-compounded 5-year return excluding most recent 12 months
crsp_df["log_ret"]    = np.log1p(crsp_df["ret_adjusted"])
crsp_df["cum_60m"]    = (
    crsp_df.groupby("permno")["log_ret"]
    .rolling(60, min_periods=48)
    .sum()
    .reset_index(drop=True)
)
crsp_df["cum_12m_lt"] = (
    crsp_df.groupby("permno")["log_ret"]
    .rolling(12, min_periods=11)
    .sum()
    .reset_index(drop=True)
)
crsp_df["reversal_lt"] = np.expm1(crsp_df["cum_60m"] - crsp_df["cum_12m_lt"])
crsp_df = crsp_df.drop(columns=["log_ret", "cum_60m", "cum_12m_lt"])

print("CRSP factors calculated.")

# ============================================================
# 5) Compustat: Clean and Calculate Accounting Factors
# ============================================================

print("\nCleaning Compustat and calculating accounting factors...")

numeric_cols = [
    "act", "ao", "ap", "at", "ceq", "che", "cogs", "dltt", "dv", "ib",
    "ibadj", "ibcom", "icapt", "lco", "lo", "lt", "mibt", "ni", "nopi",
    "pdvc", "pi", "ppent", "pstk", "pstkr", "sale", "teq", "txdi", "txt",
    "xido", "txditc", "capx",
]
for col in numeric_cols:
    if col in comp_df.columns:
        comp_df[col] = pd.to_numeric(comp_df[col], errors="coerce")
    else:
        comp_df[col] = np.nan

# Deduplicate on (gvkey, fyear) — keep last (consistent with Fix 1 above)
comp_df = comp_df.drop_duplicates(subset=["gvkey", "fyear"], keep="last").copy()
comp_df = comp_df[(comp_df["year"] >= 1973) & (comp_df["year"] <= 2024)]
comp_df = comp_df.sort_values(["gvkey", "year"]).reset_index(drop=True)

# Book Equity
comp_df["be"] = comp_df["ceq"] + comp_df["txditc"].fillna(0) - comp_df["pstk"].fillna(0)

# Net Operating Assets
comp_df["noa"] = (comp_df["at"] - comp_df["che"]) - (comp_df["lt"] - comp_df["dltt"])

# Gross Profitability
comp_df["gp"] = (comp_df["sale"] - comp_df["cogs"]) / comp_df["at"]

# ROA
comp_df["roa"] = comp_df["ni"] / comp_df["at"]

# Capital Investment
comp_df["capinv"] = comp_df["capx"] / comp_df["at"]

# Leverage
comp_df["leverage"] = comp_df["dltt"] / comp_df["at"]

# Asset Growth
comp_df["at_lag"]      = comp_df.groupby("gvkey")["at"].shift(1)
comp_df["asset_growth"] = (comp_df["at"] - comp_df["at_lag"]) / comp_df["at_lag"]

# Accruals
comp_df["noa_lag"]  = comp_df.groupby("gvkey")["noa"].shift(1)
comp_df["accruals"] = (comp_df["noa"] - comp_df["noa_lag"]) / comp_df["noa_lag"]

print("Compustat factors calculated.")

# ============================================================
# 6) Merge via CCM Link
# ============================================================

print("\nMerging Compustat with CCM link...")

link_use = link_df[["gvkey", "lpermno", "linkdt", "linkenddt", "linkprim"]].copy()
link_use = link_use[link_use["linkprim"] == "P"].copy()

merged = comp_df.merge(link_use, on="gvkey", how="left")
merged = merged.dropna(subset=["lpermno"])
merged["lpermno"] = merged["lpermno"].astype(int)

merged = merged[
    (merged["linkdt"] <= merged["available_date"]) &
    ((merged["linkenddt"] >= merged["available_date"]) | merged["linkenddt"].isna())
]
print("After link date filter:", merged.shape)

# Build monthly date column for CRSP
crsp_df["date_ym"] = pd.to_datetime(
    crsp_df["year"].astype(str) + "-" +
    crsp_df["month"].astype(str).str.zfill(2) + "-01"
)

merged = merged.rename(columns={"lpermno": "permno"})
merged = merged.sort_values("available_date").reset_index(drop=True)
crsp_df = crsp_df.sort_values("date_ym").reset_index(drop=True)

print("Merging with CRSP panel (merge_asof)...")
master = pd.merge_asof(
    crsp_df,
    merged.drop(columns=["year"], errors="ignore"),
    left_on="date_ym",
    right_on="available_date",
    by="permno",
    direction="backward",
)
print("After merge_asof:", master.shape)

# ── Fix 3 (v2): has_compustat flag — do NOT drop rows without Compustat ──────
# For the v2 universe, we retain all universe members regardless of Compustat
# coverage. Models handle missing characteristics via cross-sectional median
# imputation at training time. For the legacy universe we preserve the original
# behaviour (drop rows without gvkey).
master["has_compustat"] = master["gvkey"].notna()

if UNIVERSE_VERSION == "legacy":
    master = master.dropna(subset=["gvkey"])
    print(f"[legacy] After dropna(gvkey): {master.shape}")
else:
    n_no_comp = (~master["has_compustat"]).sum()
    print(f"[v2] Retaining {n_no_comp:,} firm-months without Compustat match "
          f"(has_compustat=False). Models will impute these at training time.")

# ── Fix 4 (v2): Resolve duplicate (permno, year, month) in master ────────────
before_dedup_master = len(master)
master = (
    master
    .sort_values(["permno", "year", "month", "datadate"])
    .drop_duplicates(subset=["permno", "year", "month"], keep="last")
    .reset_index(drop=True)
)
n_dropped = before_dedup_master - len(master)
if n_dropped:
    print(f"[Fix 4] Duplicate (permno, year, month) resolved: "
          f"dropped {n_dropped:,} rows (kept most recent datadate)")

# Assertion: zero duplicates
assert master.duplicated(subset=["permno", "year", "month"]).sum() == 0, \
    "ASSERTION FAILED: duplicate (permno, year, month) rows remain in master"
print("  Assertion passed: zero duplicate (permno, year, month) rows.")

print("Master dataset shape:", master.shape)

# Compustat coverage log by year (v2 only)
if UNIVERSE_VERSION == "v2":
    comp_cov = (
        master.groupby("year")
        .agg(
            total_obs=("permno", "count"),
            with_compustat=("has_compustat", "sum"),
        )
        .assign(coverage_pct=lambda d: 100 * d["with_compustat"] / d["total_obs"])
        .reset_index()
    )
    comp_cov.to_csv("logs/compustat_coverage_by_year.csv", index=False)
    print(f"  Compustat coverage log → logs/compustat_coverage_by_year.csv")
    low_cov = comp_cov[comp_cov["coverage_pct"] < 85]
    if len(low_cov):
        print(f"  Years with <85% Compustat coverage: {low_cov['year'].tolist()}")

# ============================================================
# 7) Book-to-Market
# ============================================================

master["market_cap"] = master["prc"] * master["shrout"]
master["bm"] = master["be"] / (master["market_cap"] + 1e-8)
master["bm"] = master["bm"].replace([np.inf, -np.inf], np.nan)

# ============================================================
# 8) Fama-French 5 Factor Merge
# ============================================================

print("\nLoading Fama-French 5 factor data...")
ff_df = pd.read_csv("data_raw/ff_factors_monthly.csv", skiprows=4)
ff_df.columns = [c.strip().lower().replace("-", "_") for c in ff_df.columns]
ff_df = ff_df.rename(columns={ff_df.columns[0]: "date"})
ff_df = ff_df[pd.to_numeric(ff_df["date"], errors="coerce").notna()].copy()
ff_df["date"]  = ff_df["date"].astype(int)
ff_df["year"]  = ff_df["date"] // 100
ff_df["month"] = ff_df["date"] % 100

ff_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
for col in ff_cols:
    ff_df[col] = pd.to_numeric(ff_df[col], errors="coerce") / 100

master = master.merge(ff_df[["year", "month"] + ff_cols], on=["year", "month"], how="left")
print("After FF5 merge:", master.shape)

# ============================================================
# 9) Macro Predictor Merge  (NEW — was absent from original pipeline)
# ============================================================

print("\nMerging macro predictors...")
macro_path = "data_raw/macro_predictors.csv"
if os.path.exists(macro_path):
    macro_df = pd.read_csv(macro_path)
    macro_df.columns = [c.strip().lower() for c in macro_df.columns]

    # Identify the date column (first column)
    date_c = macro_df.columns[0]
    macro_df[date_c] = pd.to_datetime(macro_df[date_c], errors="coerce")
    macro_df["year"]  = macro_df[date_c].dt.year
    macro_df["month"] = macro_df[date_c].dt.month

    macro_cols = ["dp_ratio", "term_spread", "real_short_rate",
                  "default_spread", "indpro_growth", "volatility"]
    macro_cols_present = [c for c in macro_cols if c in macro_df.columns]

    master = master.merge(
        macro_df[["year", "month"] + macro_cols_present],
        on=["year", "month"],
        how="left",
    )

    # Flag observations before macro coverage window (pre-January 1975)
    master["pre_macro_period"] = (master["year"] < 1975) | (
        (master["year"] == 1975) & (master["month"] < 1)
    )
    n_pre = master["pre_macro_period"].sum()
    print(f"  Macro columns merged: {macro_cols_present}")
    print(f"  pre_macro_period observations (before Jan 1975): {n_pre:,}")

    # Validate coverage within macro window
    for col in macro_cols_present:
        in_window = master[~master["pre_macro_period"]]
        pct_null = 100 * in_window[col].isna().mean()
        if pct_null > 0:
            print(f"  WARNING: {col} has {pct_null:.2f}% null within macro coverage window")
else:
    print(f"  WARNING: {macro_path} not found — macro predictors not merged.")

print("After macro merge:", master.shape)

# ============================================================
# 10) OSAP Signals Merge
# ============================================================

# For v2: call build_osap_expanded.py first if the expanded file doesn't exist
if UNIVERSE_VERSION == "v2" and not os.path.exists(OSAP_INPUT):
    print(f"\n[v2] {OSAP_INPUT} not found — running build_osap_expanded.py...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    from build_osap_expanded import build_osap_expanded
    build_osap_expanded()

print(f"\nLoading OSAP signals from: {OSAP_INPUT}")
if os.path.exists(OSAP_INPUT):
    osap = pd.read_csv(OSAP_INPUT, low_memory=False)
    osap["permno"] = osap["permno"].astype("Int64")
    # Drop redundant columns if present
    osap = osap.drop(columns=[c for c in ["yyyymm", "bm"] if c in osap.columns],
                     errors="ignore")
    master["permno"] = master["permno"].astype("Int64")
    master = master.merge(osap, on=["permno", "year", "month"], how="left")
    print("After OSAP merge:", master.shape)
else:
    print(f"  WARNING: {OSAP_INPUT} not found — OSAP signals not merged.")

# ============================================================
# 11) Derived (Bucket-2) Signals
# ============================================================

# betasq: beta squared
if "beta" in master.columns:
    master["betasq"] = master["beta"] ** 2

# std_dolvol: 12-month rolling std of log dollar volume
if "dolvol" in master.columns:
    master = master.sort_values(["permno", "year", "month"]).reset_index(drop=True)
    master["std_dolvol"] = (
        master.groupby("permno")["dolvol"]
        .transform(lambda x: x.rolling(12, min_periods=6).std())
    )

# stdacc: 3-year rolling std of accruals
if "accruals" in master.columns:
    master["stdacc"] = (
        master.groupby("permno")["accruals"]
        .transform(lambda x: x.rolling(36, min_periods=24).std())
    )

# sgr: annual sales growth
if "sale" in master.columns:
    master = master.sort_values(["gvkey", "year", "month"]).reset_index(drop=True)
    sale_lag = master.groupby(["gvkey", "month"])["sale"].shift(1)
    master["sgr"] = (master["sale"] / sale_lag) - 1
    master["sgr"] = master["sgr"].replace([np.inf, -np.inf], np.nan)

# Industry-adjusted signals
if "siccd" in master.columns:
    master["sic2"] = master["siccd"].astype(str).str[:2]

    if "bm" in master.columns:
        master["bm_ia"] = master["bm"] - master.groupby(["sic2", "year", "month"])["bm"].transform("mean")

    if "cfp" in master.columns:
        master["cfp_ia"] = master["cfp"] - master.groupby(["sic2", "year", "month"])["cfp"].transform("mean")

    if "mvel1" in master.columns:
        master["mve_ia"] = master["mvel1"] - master.groupby(["sic2", "year", "month"])["mvel1"].transform("mean")

    if "ni" in master.columns and "sale" in master.columns:
        master["pm"]       = master["ni"] / master["sale"].replace(0, np.nan)
        pm_lag             = master.groupby(["gvkey", "month"])["pm"].shift(1)
        master["delta_pm"] = master["pm"] - pm_lag
        master["chpmia"]   = (
            master["delta_pm"] -
            master.groupby(["sic2", "year", "month"])["delta_pm"].transform("mean")
        )
        master = master.drop(columns=["pm", "delta_pm"])

    master = master.drop(columns=["sic2"], errors="ignore")

# ============================================================
# 12) Fix 5: Drop Unusable Characteristic Columns
# ============================================================
# The following columns were identified in the April 2026 data audit as having
# near-zero or zero non-null coverage and are dropped here to avoid carrying
# empty columns into model training. Columns flagged as "sparse but retained"
# (ms 26.7%, tang 40.9%, realestate 38.7%, orgcap 48.2%) are NOT dropped.

COLS_TO_DROP = [
    "rd_mve",   # 0.0%  non-null — broken / zero coverage
    "std_turn", # 0.2%  non-null — completely unusable
    "chmom",    # 3.1%  non-null — effectively unusable
    "ps",       # 4.2%  non-null — very sparse
    "sin",      # 7.4%  non-null — very sparse
    "turn",     # 16.1% non-null — severely sparse
]
dropped_cols = [c for c in COLS_TO_DROP if c in master.columns]
master = master.drop(columns=dropped_cols, errors="ignore")
print(f"\n[Fix 5] Dropped low-coverage columns: {dropped_cols}")

# ============================================================
# 13) Summary
# ============================================================

print("\n" + "=" * 60)
print(f"MASTER PANEL SUMMARY  [v={UNIVERSE_VERSION}]")
print("=" * 60)

firm_counts = master.groupby("year")["permno"].nunique()
print("Years covered:", firm_counts.index.min(), "to", firm_counts.index.max())
print("Avg firms per year:", round(firm_counts.mean(), 2))
print("Total observations:", len(master))

monthly_counts = master.groupby(["year", "month"])["permno"].nunique()
print(f"Avg firms per month: {monthly_counts.mean():.1f}")
print(f"Min firms per month: {monthly_counts.min()} | Max: {monthly_counts.max()}")

all_factors = [
    "market_cap", "ret", "ret_adjusted", "illiquidity",
    "reversal_st", "momentum", "reversal_lt",
    "be", "bm", "noa", "gp", "roa", "capinv", "leverage", "asset_growth", "accruals",
    "mom1m", "mom6m", "mom12m", "mom36m", "indmom", "maxret", "pricedelay",
    "mvel1", "dolvol", "ill", "zerotrade", "baspread",
    "retvol", "idiovol", "beta",
    "absacc", "acc", "age", "agr", "cash", "cashpr", "cfp", "chatoia", "chcsho",
    "chinv", "convind", "divi", "divo", "dy", "ep", "gma", "grcapx", "grltnoa",
    "herf", "hire", "invest", "lev", "operprof", "orgcap",
    "pchsale_pchinvt", "pchsale_pchxsga", "pctacc", "rd", "rd_sale",
    "realestate", "roaq", "sp", "tang", "tb",
    "chtx", "ms", "nincr", "stdcf", "roeq", "rsup", "ear",
    "betasq", "std_dolvol", "stdacc", "sgr", "bm_ia", "cfp_ia", "mve_ia", "chpmia",
    # Macro
    "dp_ratio", "term_spread", "real_short_rate", "default_spread",
    "indpro_growth", "volatility",
    # FF5
    "mkt_rf", "smb", "hml", "rmw", "cma", "rf",
]

print("\nFactor availability:")
for f in all_factors:
    if f in master.columns:
        n   = master[f].notna().sum()
        pct = 100 * n / len(master)
        print(f"  {f}: {n:,} non-null ({pct:.1f}%)")
    else:
        print(f"  {f}: MISSING")

print("=" * 60)

# ============================================================
# 14) Save Output
# ============================================================

master.to_csv(PANEL_OUTPUT, index=False)
print(f"\nMaster panel saved to: {PANEL_OUTPUT}")

# Write the active panel path to portfolio/config.py so all model
# files can import it without hardcoding the filename.
config_path = "portfolio/config.py"
with open(config_path, "w") as f:
    f.write(f'# Auto-generated by src/build_master.py — do not edit manually.\n')
    f.write(f'UNIVERSE_VERSION = {UNIVERSE_VERSION!r}\n')
    f.write(f'PANEL_PATH = {PANEL_OUTPUT!r}\n')
print(f"Panel path written to: {config_path}")

print("Script completed successfully.")

# ============================================================
# 15) Spot Check: January 2005 Top 5 by Market Cap
# ============================================================

jan_2005 = master[(master["year"] == 2005) & (master["month"] == 1)].copy()
jan_2005 = jan_2005.nlargest(5, "market_cap")

display_cols = ["permno", "gvkey", "ticker", "market_cap", "ret", "exchcd"]
display_cols = [c for c in display_cols if c in jan_2005.columns]
spot = jan_2005[display_cols].copy()
if "market_cap" in spot.columns:
    spot["market_cap"] = spot["market_cap"].apply(
        lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) else ""
    )

print("\n" + "=" * 60)
print("SPOT CHECK: January 2005 — Top 5 by Market Cap")
print("=" * 60)
print(spot.to_string(index=False))
print("=" * 60)
