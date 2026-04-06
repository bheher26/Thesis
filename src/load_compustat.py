import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Basic Execution Info
# ============================================================

print("\nStarting Compustat Annual Data Processing")
print("Running from directory:", os.getcwd())

# ============================================================
# 2) Load Data
# ============================================================

csv_path = "data_raw/compustat_annual.csv"

print("\nLoading CSV file...")
df = pd.read_csv(csv_path)

print("Data loaded successfully.")
print("Raw shape:", df.shape)

# ============================================================
# 3) Basic Cleaning
# ============================================================

# Standardize column names to lowercase
df.columns = [c.strip().lower() for c in df.columns]

# Display original column list
print("\nColumns in dataset:", df.columns.tolist())

# Convert date column to datetime
df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

print("Date column converted to datetime format.")
print("Date range:",
      df["datadate"].min().date(),
      "to",
      df["datadate"].max().date())

# Extract year from datadate if not already present
df["year"] = df["datadate"].dt.year

# Convert fyear to integer where possible
df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")

# Identify and convert numeric columns
numeric_cols = [
    "act", "ao", "ap", "at", "ceq", "che", "cogs", "dltt", "dv", "ib",
    "ibadj", "ibcom", "icapt", "lco", "lo", "lt", "mibt", "ni", "nopi",
    "pdvc", "pi", "ppent", "pstk", "pstkr", "sale", "teq", "txdi", "txt", "xido"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicates based on GVKEY and fiscal year
df_clean = df.drop_duplicates(subset=["gvkey", "fyear"], keep="first").copy()

print("\nBasic cleaning complete.")
print("After removing duplicates:", df_clean.shape)

# ============================================================
# 3.1) Calculate Accounting Factor Variables
# ============================================================

print("\nConstructing accounting factor variables...")

# Add necessary numeric columns if missing
numeric_cols_factors = [
    "txditc", "capx", "dltt", "che", "lt"
]

for col in numeric_cols_factors:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    else:
        df_clean[col] = np.nan

# Sort by gvkey and year for lagged calculations
df_clean = df_clean.sort_values(["gvkey", "year"]).reset_index(drop=True)

# 1. Book Equity (BE = CEQ + TXDITC − PSTK)
df_clean["be"] = df_clean["ceq"] + df_clean["txditc"].fillna(0) - df_clean["pstk"].fillna(0)

# 2. Net Operating Assets (NOA = (AT − CHE) − (LT − DLTT))
df_clean["noa"] = (df_clean["at"] - df_clean["che"]) - (df_clean["lt"] - df_clean["dltt"])

# 3. Gross Profitability ((SALE − COGS) / AT)
df_clean["gp"] = (df_clean["sale"] - df_clean["cogs"]) / df_clean["at"]

# 4. ROA (NI / AT)
df_clean["roa"] = df_clean["ni"] / df_clean["at"]

# 5. Capital Investment (CAPX / AT)
df_clean["capinv"] = df_clean["capx"] / df_clean["at"]

# 6. Leverage (DLTT / AT)
df_clean["leverage"] = df_clean["dltt"] / df_clean["at"]

# 7. Asset Growth ((AT_t − AT_{t−1}) / AT_{t−1})
df_clean["at_lag"] = df_clean.groupby("gvkey")["at"].shift(1)
df_clean["asset_growth"] = (df_clean["at"] - df_clean["at_lag"]) / df_clean["at_lag"]

# 8. Accruals ((NOA_t − NOA_{t−1}) / NOA_{t−1})
df_clean["noa_lag"] = df_clean.groupby("gvkey")["noa"].shift(1)
df_clean["accruals"] = (df_clean["noa"] - df_clean["noa_lag"]) / df_clean["noa_lag"]

# Book-to-Market requires market cap from CRSP - will be calculated in merge file
# Creating placeholder column for reference
df_clean["bm"] = np.nan  # To be filled in merge_crsp_compustat.py

print("Accounting factor variables constructed.")
print(f"  Book Equity (BE): {df_clean['be'].notna().sum()} non-null")
print(f"  NOA: {df_clean['noa'].notna().sum()} non-null")
print(f"  Gross Profitability: {df_clean['gp'].notna().sum()} non-null")
print(f"  ROA: {df_clean['roa'].notna().sum()} non-null")
print(f"  Capital Investment: {df_clean['capinv'].notna().sum()} non-null")
print(f"  Leverage: {df_clean['leverage'].notna().sum()} non-null")
print(f"  Asset Growth: {df_clean['asset_growth'].notna().sum()} non-null")
print(f"  Accruals: {df_clean['accruals'].notna().sum()} non-null")


print("\n" + "="*60)
print("COMPUSTAT ANNUAL DATA SUMMARY")
print("="*60)

print("Unique companies (GVKEY):", df_clean["gvkey"].nunique())
print("Years covered:", df_clean["year"].min(), "to", df_clean["year"].max())
print("Total observations:", len(df_clean))
print("Missing values per column:")
print(df_clean.isnull().sum())

print("="*60)

# ============================================================
# 5) Output Sample
# ============================================================

print("\nFirst 20 rows of cleaned Compustat data with factor variables:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Select key columns for display
display_cols = [
    "gvkey", "year", "datadate", "sale", "cogs", "at", "ceq", "ni",
    "be", "noa", "gp", "roa", "capinv", "leverage", "asset_growth", "accruals"
]
available_cols = [col for col in display_cols if col in df_clean.columns]

print(df_clean[available_cols].head(20))

print("\n" + "="*60)
print("Script completed successfully.")
print("="*60)
