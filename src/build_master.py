import os
import pandas as pd
import numpy as np

print("\nBuilding Master Panel Dataset")
print("Running from directory:", os.getcwd())

# ============================================================
# 1) Load Raw Data
# ============================================================

print("\nLoading CRSP data...")
crsp_df = pd.read_csv("data_raw/crsp_monthly_stock.csv", low_memory=False)
crsp_df.columns = [c.strip().lower() for c in crsp_df.columns]

date_col = "date" if "date" in crsp_df.columns else "mthcaldt"
crsp_df[date_col] = pd.to_datetime(crsp_df[date_col], errors="coerce")
crsp_df["year"] = crsp_df[date_col].dt.year
crsp_df["month"] = crsp_df[date_col].dt.month
print("CRSP loaded:", crsp_df.shape)

print("\nLoading Compustat data...")
comp_df = pd.read_csv("data_raw/compustat_annual.csv", low_memory=False)
comp_df.columns = [c.strip().lower() for c in comp_df.columns]
comp_df["datadate"] = pd.to_datetime(comp_df["datadate"], errors="coerce")
comp_df["year"] = comp_df["datadate"].dt.year
comp_df["fyear"] = pd.to_numeric(comp_df["fyear"], errors="coerce").astype("Int64")

# NOTE: Compustat was downloaded with a 6-month publication lag already applied.
# 'datadate' in this file is therefore the availability date (not the fiscal year-end),
# so no additional DateOffset is needed. available_year == datadate.year directly.
# WARNING — if the download lag is ever removed, re-add pd.DateOffset(months=6) here
# to avoid look-ahead bias.
comp_df["available_date"] = comp_df["datadate"]
comp_df["available_year"] = comp_df["available_date"].dt.year
comp_df["available_month"] = comp_df["available_date"].dt.month
print("Compustat loaded:", comp_df.shape)

print("\nLoading CCM Link file...")
link_df = pd.read_csv("data_raw/ccm_link.csv")
link_df.columns = [c.strip().lower() for c in link_df.columns]
link_df["linkdt"] = pd.to_datetime(link_df["linkdt"])
link_df["linkenddt"] = link_df["linkenddt"].replace("E", pd.NaT)
link_df["linkenddt"] = pd.to_datetime(link_df["linkenddt"], errors="coerce")
print("CCM Link loaded:", link_df.shape)

# ============================================================
# 2) CRSP: Clean and Build Top-500 Universe
# ============================================================

print("\nCleaning CRSP and building top-500 universe...")

crsp_df["prc"] = pd.to_numeric(crsp_df["prc"], errors="coerce").abs()
crsp_df["shrout"] = pd.to_numeric(crsp_df["shrout"], errors="coerce")
crsp_df["ret"] = pd.to_numeric(crsp_df["ret"], errors="coerce")
crsp_df["dlret"] = pd.to_numeric(crsp_df["dlret"], errors="coerce")
crsp_df["vol"] = pd.to_numeric(crsp_df["vol"], errors="coerce")
crsp_df["ret_adjusted"] = (1 + crsp_df["ret"]) * (1 + crsp_df["dlret"].fillna(0)) - 1
crsp_df["market_cap"] = crsp_df["prc"] * crsp_df["shrout"]

crsp_df = crsp_df[(crsp_df["year"] >= 1973) & (crsp_df["year"] <= 2024)]
crsp_df = crsp_df[crsp_df["shrcd"].isin([10, 11])]

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
    how="inner"
).drop(columns=["hold_year"])

print("After top-500 filter:", crsp_df.shape)

# Save CRSP-only top-500 panel before Compustat merge.
# Used by level0_equal_weight.py to avoid the Compustat coverage gap.
crsp_df[["permno", "year", "month", "ret_adjusted"]].drop_duplicates().to_csv(
    "data_clean/crsp_top500_panel.csv", index=False
)
print("CRSP top-500 panel saved to data_clean/crsp_top500_panel.csv")

# ============================================================
# 3) CRSP: Calculate Market-Based Factors
# ============================================================

print("\nCalculating CRSP factor variables...")

crsp_df = crsp_df.sort_values(["permno", date_col]).reset_index(drop=True)

# Illiquidity: |RET| / (PRC × VOL)
crsp_df["illiquidity"] = crsp_df["ret"].abs() / (crsp_df["prc"] * crsp_df["vol"])
crsp_df["illiquidity"] = crsp_df["illiquidity"].replace([np.inf, -np.inf], np.nan)

# Short-term reversal: lagged 1-month return
crsp_df["reversal_st"] = crsp_df.groupby("permno")["ret"].shift(1)

# Momentum: log-compounded 12-month return excluding most recent month (t-2 to t-13)
log_ret = np.log1p(crsp_df["ret_adjusted"])
crsp_df["log_ret"] = log_ret

crsp_df["cum_12m"] = (
    crsp_df.groupby("permno")["log_ret"]
    .rolling(12, min_periods=11)
    .sum()
    .reset_index(drop=True)
)
crsp_df["momentum"] = np.expm1(
    crsp_df["cum_12m"] - np.log1p(crsp_df["ret_adjusted"])
)
crsp_df = crsp_df.drop(columns=["log_ret", "cum_12m"])

# Long-term reversal: log-compounded 5-year return excluding most recent 12 months
crsp_df["log_ret"] = np.log1p(crsp_df["ret_adjusted"])
crsp_df["cum_60m"] = (
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
# 4) Compustat: Clean and Calculate Accounting Factors
# ============================================================

print("\nCleaning Compustat and calculating accounting factors...")

numeric_cols = [
    "act", "ao", "ap", "at", "ceq", "che", "cogs", "dltt", "dv", "ib",
    "ibadj", "ibcom", "icapt", "lco", "lo", "lt", "mibt", "ni", "nopi",
    "pdvc", "pi", "ppent", "pstk", "pstkr", "sale", "teq", "txdi", "txt",
    "xido", "txditc", "capx"
]
for col in numeric_cols:
    if col in comp_df.columns:
        comp_df[col] = pd.to_numeric(comp_df[col], errors="coerce")
    else:
        comp_df[col] = np.nan

comp_df = comp_df.drop_duplicates(subset=["gvkey", "fyear"], keep="first").copy()
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
comp_df["at_lag"] = comp_df.groupby("gvkey")["at"].shift(1)
comp_df["asset_growth"] = (comp_df["at"] - comp_df["at_lag"]) / comp_df["at_lag"]

# Accruals
comp_df["noa_lag"] = comp_df.groupby("gvkey")["noa"].shift(1)
comp_df["accruals"] = (comp_df["noa"] - comp_df["noa_lag"]) / comp_df["noa_lag"]

print("Compustat factors calculated.")

# ============================================================
# 5) Merge via CCM Link
# ============================================================

print("\nMerging Compustat with CCM link...")

link_use = link_df[["gvkey", "lpermno", "linkdt", "linkenddt", "linkprim"]].copy()
link_use = link_use[link_use["linkprim"] == "P"].copy()

merged = comp_df.merge(link_use, on="gvkey", how="left")
merged = merged.dropna(subset=["lpermno"])
merged["lpermno"] = merged["lpermno"].astype(int)

# Restrict to observations that fall within the valid CCM link period
merged["obs_year"] = merged["year"]
merged = merged[
    (merged["linkdt"].dt.year <= merged["obs_year"]) &
    ((merged["linkenddt"].dt.year >= merged["obs_year"]) | merged["linkenddt"].isna())
]
print("After link date filter:", merged.shape)

# NOTE: available_year == datadate year because the download already incorporates the
# 6-month publication lag. No additional offset is applied (see loading block above).
# Merge on year only so all 12 CRSP months in the availability year receive the fundamentals.
# Drop Compustat's 'year' (= datadate year) to avoid year_x/year_y column collision;
# CRSP's 'year' (observation year) is the one used for all downstream merges.
merged = merged.drop(columns=["year"], errors="ignore")
print("Merging with CRSP top-500 panel...")
master = merged.merge(
    crsp_df,
    left_on=["lpermno", "available_year"],
    right_on=["permno", "year"],
    how="inner"
)

print("Master dataset shape:", master.shape)

# ============================================================
# 6) Book-to-Market
# ============================================================

master["market_cap"] = master["prc"] * master["shrout"]
master["bm"] = master["be"] / (master["market_cap"] + 1e-8)
master["bm"] = master["bm"].replace([np.inf, -np.inf], np.nan)

# ============================================================
# 6.6) Load and Merge Fama-French 5 Factor Data
# ============================================================

print("\nLoading Fama-French 5 factor data...")
ff_df = pd.read_csv("data_raw/ff_factors_monthly.csv", skiprows=4)
ff_df.columns = [c.strip().lower().replace("-", "_") for c in ff_df.columns]
ff_df = ff_df.rename(columns={ff_df.columns[0]: "date"})
ff_df = ff_df[pd.to_numeric(ff_df["date"], errors="coerce").notna()].copy()
ff_df["date"] = ff_df["date"].astype(int)
ff_df["year"] = ff_df["date"] // 100
ff_df["month"] = ff_df["date"] % 100

ff_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
for col in ff_cols:
    ff_df[col] = pd.to_numeric(ff_df[col], errors="coerce") / 100

master = master.merge(ff_df[["year", "month"] + ff_cols], on=["year", "month"], how="left")
print("After FF5 merge:", master.shape)

# ============================================================
# 6.7) Merge OSAP Signals and Derive Bucket-2 Characteristics
# ============================================================

print("\nLoading OSAP signals...")
osap = pd.read_csv("data_clean/osap_signals_top500.csv")
osap["permno"] = osap["permno"].astype("Int64")
# Drop yyyymm (redundant with year/month) and bm (master has its own be/market_cap version)
osap = osap.drop(columns=[c for c in ["yyyymm", "bm"] if c in osap.columns])
master["permno"] = master["permno"].astype("Int64")
master = master.merge(osap, on=["permno", "year", "month"], how="left")
print("After OSAP merge:", master.shape)

# --- Bucket 2: Derived signals ---

# betasq: beta squared
if "beta" in master.columns:
    master["betasq"] = master["beta"] ** 2

# std_dolvol: 12-month rolling std of log dollar volume
# dolvol from OSAP is already log-transformed; compute rolling std per firm
if "dolvol" in master.columns:
    master = master.sort_values(["permno", "year", "month"]).reset_index(drop=True)
    master["std_dolvol"] = (
        master.groupby("permno")["dolvol"]
        .transform(lambda x: x.rolling(12, min_periods=6).std())
    )

# stdcf / stdacc: 3-year rolling std of accruals (used as proxy for earnings volatility)
# Note: roavol from OSAP (varcf) is preferred for roavol; stdacc uses our accruals column
if "accruals" in master.columns:
    master["stdacc"] = (
        master.groupby("permno")["accruals"]
        .transform(lambda x: x.rolling(36, min_periods=24).std())
    )

# sgr: annual sales growth = (sale_t / sale_{t-1}) - 1
# sale is annual from Compustat; we compute per gvkey to avoid cross-firm leakage
if "sale" in master.columns:
    master = master.sort_values(["gvkey", "year", "month"]).reset_index(drop=True)
    sale_lag = master.groupby(["gvkey", "month"])["sale"].shift(1)
    master["sgr"] = (master["sale"] / sale_lag) - 1
    master["sgr"] = master["sgr"].replace([np.inf, -np.inf], np.nan)

# bm_ia: industry-adjusted book-to-market (bm - SIC2 industry mean bm per month)
if "bm" in master.columns and "siccd" in master.columns:
    master["sic2"] = master["siccd"].astype(str).str[:2]
    ind_mean_bm = master.groupby(["sic2", "year", "month"])["bm"].transform("mean")
    master["bm_ia"] = master["bm"] - ind_mean_bm

# cfp_ia: industry-adjusted cash flow to price
if "cfp" in master.columns and "siccd" in master.columns:
    master["sic2"] = master["siccd"].astype(str).str[:2]
    ind_mean_cfp = master.groupby(["sic2", "year", "month"])["cfp"].transform("mean")
    master["cfp_ia"] = master["cfp"] - ind_mean_cfp

# mve_ia: industry-adjusted log market equity (mvel1 - SIC2 industry mean mvel1 per month)
if "mvel1" in master.columns and "siccd" in master.columns:
    master["sic2"] = master["siccd"].astype(str).str[:2]
    ind_mean_mve = master.groupby(["sic2", "year", "month"])["mvel1"].transform("mean")
    master["mve_ia"] = master["mvel1"] - ind_mean_mve

# chpmia: industry-adjusted change in profit margin = Δ(ni/sale) - industry mean Δ(ni/sale)
if "ni" in master.columns and "sale" in master.columns and "siccd" in master.columns:
    master["sic2"] = master["siccd"].astype(str).str[:2]
    master["pm"] = master["ni"] / master["sale"].replace(0, np.nan)
    pm_lag = master.groupby(["gvkey", "month"])["pm"].shift(1)
    master["delta_pm"] = master["pm"] - pm_lag
    ind_mean_delta_pm = master.groupby(["sic2", "year", "month"])["delta_pm"].transform("mean")
    master["chpmia"] = master["delta_pm"] - ind_mean_delta_pm
    master = master.drop(columns=["pm", "delta_pm"])

# Drop temporary sic2 helper column
master = master.drop(columns=["sic2"], errors="ignore")

bucket2_signals = ["betasq", "std_dolvol", "stdacc", "sgr", "bm_ia", "cfp_ia", "mve_ia", "chpmia"]
print("Bucket-2 derived signals added:")
for sig in bucket2_signals:
    if sig in master.columns:
        n = master[sig].notna().sum()
        print(f"  {sig}: {n:,} non-null")
    else:
        print(f"  {sig}: skipped (required input column missing)")

# ============================================================
# 7) Summary
# ============================================================

print("\n" + "="*60)
print("MASTER PANEL SUMMARY")
print("="*60)

firm_counts = master.groupby("year")["permno"].nunique()
print("Years covered:", firm_counts.index.min(), "to", firm_counts.index.max())
print("Avg firms per year:", round(firm_counts.mean(), 2))
print("Total observations:", len(master))

all_factors = [
    "market_cap", "ret", "ret_adjusted", "illiquidity",
    "reversal_st", "momentum", "reversal_lt",
    "be", "bm", "noa", "gp", "roa", "capinv", "leverage", "asset_growth", "accruals",
    # OSAP bucket 1 signals (Gu et al. 2020 names)
    "mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom", "maxret", "pricedelay",
    "mvel1", "dolvol", "ill", "turn", "std_turn", "zerotrade", "baspread",
    "retvol", "idiovol", "beta",
    "absacc", "acc", "age", "agr", "cash", "cashpr", "cfp", "chatoia", "chcsho",
    "chinv", "convind", "divi", "divo", "dy", "ep", "gma", "grcapx", "grltnoa",
    "herf", "hire", "invest", "lev", "operprof", "orgcap",
    "pchsale_pchinvt", "pchsale_pchxsga", "pctacc", "ps", "rd", "rd_mve", "rd_sale",
    "realestate", "roaq", "sp", "tang", "sin", "tb",
    "chtx", "ms", "nincr", "stdcf", "roeq", "rsup", "ear",
    # OSAP bucket 2 derived signals
    "betasq", "std_dolvol", "stdacc", "sgr", "bm_ia", "cfp_ia", "mve_ia", "chpmia",
]
print("\nFactor availability:")
for f in all_factors:
    if f in master.columns:
        n = master[f].notna().sum()
        print(f"  {f}: {n:,} non-null")
    else:
        print(f"  {f}: missing")

reversal_lt_pct = 100 * master["reversal_lt"].notna().sum() / len(master)
print(f"\nLong-term reversal sparsity: {reversal_lt_pct:.1f}% non-null (requires 60 months of history)")

print("="*60)

# ============================================================
# 8) Save Output
# ============================================================

os.makedirs("data_clean", exist_ok=True)
out_path = "data_clean/master_panel.csv"
master.to_csv(out_path, index=False)
print(f"\nMaster panel saved to: {out_path}")
print("Script completed successfully.")

# ============================================================
# 9) Spot Check: January 2005 Top 5 by Market Cap
# ============================================================

jan_2005 = master[(master["year"] == 2005) & (master["month"] == 1)].copy()
jan_2005 = jan_2005.nlargest(5, "market_cap")

display_cols = (
    ["permno", "gvkey", "ticker", "date", "market_cap", "ret"]
    + [f for f in all_factors if f in jan_2005.columns and f not in ["market_cap", "ret"]]
)
display_cols = [c for c in display_cols if c in jan_2005.columns]

print("\n" + "="*60)
print("SPOT CHECK: January 2005 — Top 5 by Market Cap")
print("="*60)

spot = jan_2005[display_cols].copy()

# Format columns for readability
if "market_cap" in spot.columns:
    spot["market_cap"] = spot["market_cap"].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) else "")
if "date" in spot.columns:
    spot["date"] = spot["date"].astype(str).str[:10]

float_cols = [c for c in spot.columns if c not in ["permno", "gvkey", "ticker", "date", "market_cap"]]
for c in float_cols:
    spot[c] = spot[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

try:
    from tabulate import tabulate
    print(tabulate(spot, headers="keys", tablefmt="pretty", showindex=False))
except ImportError:
    # Fallback: transpose so each firm is a column, each factor is a row
    spot_t = spot.set_index("ticker").T if "ticker" in spot.columns else spot.T
    print(spot_t.to_string())

print("="*60)
