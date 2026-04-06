import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Basic Execution Info
# ============================================================

print("\nStarting CRSP Top 500 Universe Construction")
print("Running from directory:", os.getcwd())

# ============================================================
# 2) Load Data
# ============================================================

csv_path = "data_raw/crsp_monthly_stock.csv"

print("\nLoading CSV file...")
df = pd.read_csv(csv_path)

print("Data loaded successfully.")
print("Raw shape:", df.shape)

# ============================================================
# 3) Basic Cleaning
# ============================================================

# Standardize column names
df.columns = [c.strip().lower() for c in df.columns]

# Identify date column
if "date" in df.columns:
    date_col = "date"
elif "mthcaldt" in df.columns:
    date_col = "mthcaldt"
else:
    raise ValueError("No valid date column found.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

print("Date column used:", date_col)
print("Date range:",
      df[date_col].min().date(),
      "to",
      df[date_col].max().date())

# Convert numeric columns
df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce")

# Adjusted return including delisting
df["ret_adjusted"] = (1 + df["ret"]) * (1 + df["dlret"].fillna(0)) - 1

# Create market cap
df["market_cap"] = df["prc"] * df["shrout"]

print("\nBasic cleaning complete.")

# ============================================================
# 4) Restrict Years 1973–2024
# ============================================================

df["year"] = df[date_col].dt.year
df["month"] = df[date_col].dt.month

df = df[(df["year"] >= 1973) & (df["year"] <= 2024)]

print("After year filter:", df.shape)

# Keep common stocks only
df = df[df["shrcd"].isin([10, 11])]

print("After keeping common stocks:", df.shape)

# ============================================================
# 5) Build Top 500 Universe
# ============================================================

# ---- 1973: Use January ranking ----
jan_1973 = df[(df["year"] == 1973) & (df["month"] == 1)].copy()
jan_1973["rank"] = jan_1973["market_cap"].rank(method="first", ascending=False)

top500_1973 = jan_1973[jan_1973["rank"] <= 500][["permno"]].copy()
top500_1973["hold_year"] = 1973

print("Top 500 for 1973:", top500_1973["permno"].nunique())

# ---- 1974–2024: December rebalancing ----
december = df[df["month"] == 12].copy()

december["rank"] = december.groupby("year")["market_cap"] \
    .rank(method="first", ascending=False)

top500_dec = december[december["rank"] <= 500][["permno", "year"]].copy()
top500_dec["hold_year"] = top500_dec["year"] + 1

top500_dec = top500_dec[
    (top500_dec["hold_year"] >= 1974) &
    (top500_dec["hold_year"] <= 2024)
][["permno", "hold_year"]]

print("December-based universes built.")

# Combine all membership
top500_all = pd.concat([top500_1973, top500_dec], ignore_index=True)

# Merge back
df_clean = df.merge(
    top500_all,
    left_on=["permno", "year"],
    right_on=["permno", "hold_year"],
    how="inner"
)

df_clean = df_clean.drop(columns=["hold_year"])

print("Final cleaned dataset shape:", df_clean.shape)

# ============================================================
# 5.1) Calculate Factor Variables
# ============================================================

print("\nCalculating factor variables...")

# Sort by permno and date for time-series calculations
df_clean = df_clean.sort_values(["permno", date_col]).reset_index(drop=True)

# Illiquidity: |RET| / (PRC × VOL)
df_clean["vol"] = pd.to_numeric(df_clean["vol"], errors="coerce")
df_clean["illiquidity"] = (df_clean["ret"].abs()) / (df_clean["prc"] * df_clean["vol"])
df_clean["illiquidity"] = df_clean["illiquidity"].replace([np.inf, -np.inf], np.nan)

# Short-term reversal: lagged 1-month return
df_clean["reversal_st"] = df_clean.groupby("permno")["ret"].shift(1)

# Momentum (12-month return excluding most recent month)
df_clean["momentum"] = df_clean.groupby("permno")["ret_adjusted"].rolling(
    window=12, min_periods=11
).sum().reset_index(drop=True) - df_clean["ret_adjusted"]

# Long-term reversal (5-year return excluding last year)
df_clean["reversal_lt"] = df_clean.groupby("permno")["ret_adjusted"].rolling(
    window=60, min_periods=48
).sum().reset_index(drop=True) - df_clean.groupby("permno")["ret_adjusted"].rolling(
    window=12, min_periods=11
).sum().reset_index(drop=True)

print("\nFactor variables calculated.")

# ============================================================
# 6) Summary Statistics
# ============================================================

print("\n" + "="*60)
print("TOP 500 UNIVERSE SUMMARY")
print("="*60)

firm_counts = df_clean.groupby("year")["permno"].nunique()

print("Years covered:", firm_counts.index.min(), "to", firm_counts.index.max())
print("Average firms per year:", round(firm_counts.mean(), 2))
print("Minimum firms in a year:", firm_counts.min())
print("Maximum firms in a year:", firm_counts.max())
print("Total observations:", len(df_clean))

print("\nVariable Summary:")
print(f"Market Cap: min={df_clean['market_cap'].min():.2f}, max={df_clean['market_cap'].max():.2f}")
print(f"Monthly Return: mean={df_clean['ret'].mean():.4f}, std={df_clean['ret'].std():.4f}")
print(f"Adjusted Return: mean={df_clean['ret_adjusted'].mean():.4f}, std={df_clean['ret_adjusted'].std():.4f}")
print(f"Momentum (12m): {df_clean['momentum'].notna().sum()} non-null observations")
print(f"Short-term Reversal: {df_clean['reversal_st'].notna().sum()} non-null observations")
print(f"Long-term Reversal (5y): {df_clean['reversal_lt'].notna().sum()} non-null observations")
print(f"Illiquidity: mean={df_clean['illiquidity'].mean():.6f}, {df_clean['illiquidity'].notna().sum()} non-null")

print("="*60)

# ============================================================
# 7) Visualization
# ============================================================

plt.figure()
plt.plot(firm_counts.index, firm_counts.values)
plt.title("Number of Firms per Year (Top 500 by Market Cap)")
plt.xlabel("Year")
plt.ylabel("Number of Firms")
plt.show()

# ============================================================
# 8) Output Sample
# ============================================================

print("\n" + "="*80)
print("FIRST 20 ROWS OF CLEANED CRSP DATA WITH FACTOR VARIABLES")
print("="*80)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Select key columns for display
display_cols = [
    "permno", date_col, "ticker", "prc", "shrout", "ret", "dlret", "ret_adjusted",
    "market_cap", "illiquidity", "reversal_st", "momentum", "reversal_lt"
]
available_cols = [col for col in display_cols if col in df_clean.columns]

print(df_clean[available_cols].head(20))
print("="*80)

print("\nScript completed successfully.")




