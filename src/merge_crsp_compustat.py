import os
import pandas as pd
import numpy as np

# ============================================================
# 1) Basic Execution Info
# ============================================================

print("\nStarting CRSP-Compustat Merge Process")
print("Running from directory:", os.getcwd())

# ============================================================
# 2) Load Data
# ============================================================

print("\nLoading CRSP data...")
crsp_path = "data_raw/crsp_monthly_stock.csv"
crsp_df = pd.read_csv(crsp_path, low_memory=False)
crsp_df.columns = [c.strip().lower() for c in crsp_df.columns]
crsp_df["date"] = pd.to_datetime(crsp_df["date"])
crsp_df["year"] = crsp_df["date"].dt.year
print("CRSP loaded:", crsp_df.shape)

print("\nLoading Compustat data...")
comp_path = "data_raw/compustat_annual.csv"
comp_df = pd.read_csv(comp_path, low_memory=False)
comp_df.columns = [c.strip().lower() for c in comp_df.columns]
comp_df["datadate"] = pd.to_datetime(comp_df["datadate"])
comp_df["year"] = comp_df["datadate"].dt.year
print("Compustat loaded:", comp_df.shape)

print("\nLoading CCM Link file...")
link_path = "data_raw/ccm_link.csv"
link_df = pd.read_csv(link_path)
link_df.columns = [c.strip().lower() for c in link_df.columns]
link_df["linkdt"] = pd.to_datetime(link_df["linkdt"])
# Replace 'E' (End) with NaT before converting to datetime
link_df["linkenddt"] = link_df["linkenddt"].replace('E', pd.NaT)
link_df["linkenddt"] = pd.to_datetime(link_df["linkenddt"], errors="coerce")
print("CCM Link loaded:", link_df.shape)

# ============================================================
# 3) Prepare for Merge
# ============================================================

# Filter CRSP data to keep only Top 500 companies (using same logic as load_crsp.py)
print("\nApplying CRSP filters...")
crsp_df["prc"] = pd.to_numeric(crsp_df["prc"], errors="coerce").abs()
crsp_df["shrout"] = pd.to_numeric(crsp_df["shrout"], errors="coerce")
crsp_df["market_cap"] = crsp_df["prc"] * crsp_df["shrout"]
crsp_df["month"] = crsp_df["date"].dt.month

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

# Filter CRSP to top-500 universe
crsp_df = crsp_df.merge(
    top500_all,
    left_on=["permno", "year"],
    right_on=["permno", "hold_year"],
    how="inner"
).drop(columns=["hold_year"])

print("After top-500 filter:", crsp_df.shape)

# Filter Compustat to full historical period (1973-2024)
print("Using Compustat data from 1973-2024...")
comp_df = comp_df[(comp_df["year"] >= 1973) & (comp_df["year"] <= 2024)]

# Rename link columns for clarity
link_df_use = link_df[["gvkey", "lpermno", "linkdt", "linkenddt", "linkprim"]].copy()
link_df_use = link_df_use[link_df_use["linkprim"] == "P"].copy()  # Keep primary links only

print("Primary links:", len(link_df_use))

# ============================================================
# 4) Merge Data
# ============================================================

print("\nMerging Compustat with CCM Link...")
merged = comp_df.merge(
    link_df_use,
    on="gvkey",
    how="left"
)

print("After merging Compustat with link:", merged.shape)

# Filter for valid links
merged = merged.dropna(subset=["lpermno"])
merged["lpermno"] = merged["lpermno"].astype(int)

print("After removing invalid links:", merged.shape)

print("\nMerging with CRSP data...")
# Merge with CRSP data on PERMNO and year
final_merged = merged.merge(
    crsp_df,
    left_on=["lpermno", "year"],
    right_on=["permno", "year"],
    how="inner"
)

print("Final merged dataset shape:", final_merged.shape)

# ============================================================
# 4.1) Calculate Book-to-Market Ratio
# ============================================================

print("\nCalculating Book-to-Market ratio...")
# Book Equity was calculated in load_compustat.py as BE = CEQ + TXDITC - PSTK
# Market Cap was calculated in load_crsp.py as PRC * SHROUT
final_merged["be"] = final_merged["ceq"] + final_merged["txditc"].fillna(0) - final_merged["pstk"].fillna(0)
final_merged["market_cap"] = final_merged["prc"] * final_merged["shrout"]
final_merged["bm"] = final_merged["be"] / (final_merged["market_cap"] + 1e-8)  # Avoid division by zero
final_merged["bm"] = final_merged["bm"].replace([np.inf, -np.inf], np.nan)

print("Book-to-Market calculated.")

# ============================================================
# 5) Data Quality Check
# ============================================================

print("\n" + "="*60)
print("MERGED DATA QUALITY SUMMARY")
print("="*60)

print("Unique companies (GVKEY):", final_merged["gvkey"].nunique())
print("Unique securities (PERMNO):", final_merged["permno"].nunique())
print("Years covered:", final_merged["year"].min(), "to", final_merged["year"].max())
print("Date range (CRSP):", final_merged["date"].min().date(), "to", final_merged["date"].max().date())
print("Total observations:", len(final_merged))

print("\nObservations per year (summary):")
print(final_merged.groupby("year").size().describe())

# Check which factor columns are available
crsp_factors = ['momentum', 'reversal_st', 'reversal_lt', 'illiquidity']
comp_factors = ['be', 'bm', 'gp', 'roa', 'capinv', 'leverage', 'asset_growth', 'accruals']

print("\na) CRSP Factors (Note: calculated separately in load_crsp.py):")
for f in crsp_factors:
    if f in final_merged.columns:
        print(f"  {f}: {final_merged[f].notna().sum()} non-null")
    else:
        print(f"  {f}: Not included in merge")

print("\nb) Compustat Factors:")
for f in comp_factors:
    if f in final_merged.columns:
        cnt = final_merged[f].notna().sum()
        print(f"  {f}: {cnt} non-null")
    else:
        print(f"  {f}: (not yet included)")

print("="*60)

# ============================================================
# 6) Output Sample
# ============================================================

print("\nFirst 20 rows of merged CRSP-Compustat master dataset:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Select key columns for readability - include both CRSP and Compustat factors
key_cols = [
    "permno", "gvkey", "date", "datadate", "year", "ticker",
    "prc", "ret", "market_cap", "be", "bm", "gp", "roa", "leverage", "capinv"
]
available_cols = [col for col in key_cols if col in final_merged.columns]

print(final_merged[available_cols].head(20))

print("\n" + "="*60)
print("Script completed successfully.")
print("Master dataset ready for analysis!")
print("="*60)
