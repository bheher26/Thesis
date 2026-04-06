"""
download_osap.py
================
Downloads the Open Source Asset Pricing (OSAP) signal panel and filters
it to your top-500 universe. Saves all outputs into your existing
data_raw/ and data_clean/ folder structure.

Usage
-----
    pip install openassetpricing pandas numpy
    python download_osap.py

What this script does
---------------------
1. Downloads the full 209-signal panel from OSAP via their Python package
2. Saves the raw download to data_raw/osap_signals_raw.csv
3. Loads your existing top-500 PERMNO universe from data_clean/crsp_top500_panel.csv
4. Filters the OSAP panel to your universe
5. Selects the 94 Gu et al. characteristics that exist in the OSAP panel
6. Saves the filtered signal panel to data_clean/osap_signals_top500.csv
7. Prints a coverage report so you know which signals are available

Requirements
------------
- pip install openassetpricing
- data_clean/crsp_top500_panel.csv must already exist (created by build_master.py)
- Enough disk space: the raw file is ~1.6 GB unzipped
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# 0) Setup
# ============================================================

os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_clean", exist_ok=True)

RAW_PATH    = "data_raw/osap_signals_raw.csv"
CLEAN_PATH  = "data_clean/osap_signals_top500.csv"
UNIVERSE    = "data_clean/crsp_top500_panel.csv"

# ============================================================
# 1) Download via openassetpricing package
# ============================================================

print("\n" + "="*60)
print("STEP 1: Downloading OSAP signal panel")
print("="*60)
print("This downloads ~1.6 GB. May take several minutes...")

try:
    import openassetpricing as oap

    # The package downloads the wide-format firm-level signal panel.
    # Use the OpenAP class with dl_all_signals (API as of 2024+).
    ap = oap.OpenAP()
    signals = ap.dl_all_signals(df_backend="pandas")

    print(f"Download complete. Shape: {signals.shape}")
    print(f"Columns sample: {list(signals.columns[:10])}")

    # Save raw download immediately
    signals.to_csv(RAW_PATH, index=False)
    print(f"Raw signals saved to: {RAW_PATH}")

except ImportError:
    print("openassetpricing package not found.")
    print("Run: pip install openassetpricing")
    print("Then re-run this script.")
    raise

except Exception as e:
    print(f"Download failed: {e}")
    print("\nFallback: If the package fails, download manually from:")
    print("https://drive.google.com/file/d/1avFIMjz_7LoF3p3nO26eqLW5KdRTOdhW/")
    print("Save to data_raw/osap_signals_raw.csv and re-run from STEP 2.")
    raise

# ============================================================
# 2) Standardize Date and Identifier Columns
# ============================================================

print("\n" + "="*60)
print("STEP 2: Standardizing columns")
print("="*60)

# Normalize column names
signals.columns = [c.strip().lower() for c in signals.columns]

print("Raw columns (first 15):", list(signals.columns[:15]))

# The OSAP panel uses 'permno' and 'yyyymm' or 'date'
# Identify the date column
if "yyyymm" in signals.columns:
    signals["year"]  = signals["yyyymm"] // 100
    signals["month"] = signals["yyyymm"] % 100
    print("Date parsed from 'yyyymm' column.")
elif "date" in signals.columns:
    signals["date"]  = pd.to_datetime(signals["date"].astype(str), errors="coerce")
    signals["year"]  = signals["date"].dt.year
    signals["month"] = signals["date"].dt.month
    print("Date parsed from 'date' column.")
else:
    raise ValueError(
        "Cannot find date column. Available columns: " + str(list(signals.columns))
    )

# Ensure permno is integer
signals["permno"] = pd.to_numeric(signals["permno"], errors="coerce").astype("Int64")

print(f"Date range: {signals['year'].min()} to {signals['year'].max()}")
print(f"Unique PERMNOs: {signals['permno'].nunique():,}")
print(f"Total observations: {len(signals):,}")

# ============================================================
# 3) Filter to Your Top-500 Universe
# ============================================================

print("\n" + "="*60)
print("STEP 3: Filtering to top-500 universe")
print("="*60)

if not os.path.exists(UNIVERSE):
    raise FileNotFoundError(
        f"{UNIVERSE} not found.\n"
        "Run build_master.py first to generate the top-500 PERMNO universe."
    )

universe = pd.read_csv(UNIVERSE)
universe.columns = [c.strip().lower() for c in universe.columns]
universe["permno"] = pd.to_numeric(universe["permno"], errors="coerce").astype("Int64")

print(f"Universe loaded: {universe.shape}")
print(f"Universe date range: {universe['year'].min()} to {universe['year'].max()}")

# Merge: keep only permno-year-month combinations in your top-500 universe
filtered = signals.merge(
    universe[["permno", "year", "month"]].drop_duplicates(),
    on=["permno", "year", "month"],
    how="inner"
)

print(f"After top-500 filter: {filtered.shape}")
print(f"Unique PERMNOs after filter: {filtered['permno'].nunique():,}")

# ============================================================
# 4) Select and Rename to Gu et al. (2020) Characteristic Names
# ============================================================

print("\n" + "="*60)
print("STEP 4: Selecting Gu et al. characteristics")
print("="*60)

# Mapping: OSAP column name -> Gu et al. (2020) Table A.6 name
# Direct matches (OSAP name == Gu name) are listed first, then renames.
# Signals requiring construction from raw data (bucket 3) are omitted here
# and handled in build_master.py.
OSAP_TO_GU = {
    # --- Direct matches (already correct name in OSAP) ---
    "mom6m":     "mom6m",       # 6-month momentum
    "mom12m":    "mom12m",      # 12-month momentum
    "indmom":    "indmom",      # industry momentum
    "maxret":    "maxret",      # maximum daily return
    "dolvol":    "dolvol",      # dollar trading volume
    "std_turn":  "std_turn",    # turnover volatility
    "beta":      "beta",        # market beta
    "cash":      "cash",        # cash holdings
    "cfp":       "cfp",         # cash flow to price
    "chinv":     "chinv",       # change in inventory
    "ep":        "ep",          # earnings to price
    "grcapx":    "grcapx",      # growth in capital expenditures
    "grltnoa":   "grltnoa",     # growth in long-term net operating assets
    "herf":      "herf",        # industry sales concentration
    "hire":      "hire",        # employee growth rate (also covers chempia)
    "operprof":  "operprof",    # operating profitability
    "orgcap":    "orgcap",      # organizational capital
    "pctacc":    "pctacc",      # percent accruals
    "ps":        "ps",          # Piotroski score
    "rd":        "rd",          # R&D increase
    "realestate":"realestate",  # real estate holdings
    "roaq":      "roaq",        # return on assets (quarterly)
    "sp":        "sp",          # sales to price
    "tang":      "tang",        # debt capacity / tangibility
    "ms":        "ms",          # Mohanram score

    # --- Bucket 1: rename OSAP column -> Gu name ---
    "streversal":        "mom1m",          # 1-month reversal
    "lrreversal":        "mom36m",         # 36-month long-term reversal
    "momrev":            "chmom",          # change in 6-month momentum
    "pricedelayrsq":     "pricedelay",     # price delay (R² variant)
    "size":              "mvel1",          # log market equity
    "illiquidity":       "ill",            # Amihud illiquidity
    "sharevol":          "turn",           # share turnover
    "zerotrade1m":       "zerotrade",      # zero-trading days (1-month)
    "bidaskspread":      "baspread",       # bid-ask spread
    "realizedvol":       "retvol",         # return volatility
    "idiovol3f":         "idiovol",        # idiosyncratic vol (FF3)
    "abnormalaccruals":  "absacc",         # absolute accruals
    "accruals":          "acc",            # working capital accruals
    "firmage":           "age",            # years since first Compustat
    "assetgrowth":       "agr",            # asset growth
    "cashprod":          "cashpr",         # cash productivity
    "chassetturnover":   "chatoia",        # industry-adj change in asset turnover
    "compequiss":        "chcsho",         # change in shares outstanding
    "convdebt":          "convind",        # convertible debt indicator
    "divinit":           "divi",           # dividend initiation
    "divomit":           "divo",           # dividend omission
    "divyieldst":        "dy",             # dividend yield
    "gp":                "gma",            # gross profitability (gp/at)
    "investment":        "invest",         # capex + inventory investment
    "leverage":          "lev",            # leverage
    "grsaletogrinv":     "pchsale_pchinvt",  # %Δsales - %Δinventory
    "grsaletogroverhead":"pchsale_pchxsga",  # %Δsales - %ΔSG&A
    "rdcap":             "rd_mve",         # R&D to market cap
    "rds":               "rd_sale",        # R&D to sales
    "sinalgo":           "sin",            # sin stocks indicator
    "tax":               "tb",             # tax income to book income
    "chtax":             "chtx",           # change in tax expense
    "numearnincrease":   "nincr",          # number of earnings increases
    "varcf":             "stdcf",          # cash flow volatility
    "roe":               "roeq",           # return on equity (quarterly)
    "revenuesurprise":   "rsup",           # revenue surprise
    "announcementreturn":"ear",            # earnings announcement return
}

# Note: bm is excluded from OSAP pull — master's own bm (be/market_cap) is used.
# Note: chempia uses hire (same signal, already included above).

filtered.columns = [c.strip().lower() for c in filtered.columns]
osap_cols_present  = {osap_col: gu_name for osap_col, gu_name in OSAP_TO_GU.items()
                      if osap_col in filtered.columns}
osap_cols_missing  = {osap_col: gu_name for osap_col, gu_name in OSAP_TO_GU.items()
                      if osap_col not in filtered.columns}

print(f"\nGu et al. signals mapped from OSAP: {len(osap_cols_present)} / {len(OSAP_TO_GU)}")
if osap_cols_missing:
    print(f"OSAP cols not found ({len(osap_cols_missing)}): {list(osap_cols_missing.keys())}")

# Core identifier columns to always keep
id_cols = ["permno", "year", "month"]
if "yyyymm" in filtered.columns:
    id_cols = ["permno", "yyyymm", "year", "month"]

keep_cols = id_cols + list(osap_cols_present.keys())
filtered_clean = filtered[keep_cols].copy()

# Rename OSAP columns to Gu et al. names
filtered_clean = filtered_clean.rename(columns=osap_cols_present)

# Deduplicate any columns that mapped to the same Gu name (e.g. hire/chempia)
filtered_clean = filtered_clean.loc[:, ~filtered_clean.columns.duplicated()]

print(f"\nFinal signal panel shape: {filtered_clean.shape}")
print(f"Signal columns: {[c for c in filtered_clean.columns if c not in id_cols]}")

# ============================================================
# 5) Coverage Report
# ============================================================

print("\n" + "="*60)
print("STEP 5: Coverage report")
print("="*60)

total_obs = len(filtered_clean)
signal_cols = [c for c in filtered_clean.columns if c not in id_cols]
coverage = {}
for sig in signal_cols:
    pct = 100 * filtered_clean[sig].notna().sum() / total_obs
    coverage[sig] = pct

coverage_df = (
    pd.DataFrame.from_dict(coverage, orient="index", columns=["pct_nonull"])
    .sort_values("pct_nonull", ascending=False)
)

print("\nSignal coverage (% non-null):")
print(coverage_df.to_string())

low_coverage = coverage_df[coverage_df["pct_nonull"] < 50]
if len(low_coverage) > 0:
    print(f"\nWarning: {len(low_coverage)} signals below 50% coverage:")
    print(low_coverage.to_string())

# ============================================================
# 6) Save Output
# ============================================================

print("\n" + "="*60)
print("STEP 6: Saving outputs")
print("="*60)

filtered_clean.to_csv(CLEAN_PATH, index=False)
print(f"Filtered signal panel saved to: {CLEAN_PATH}")

# Save the coverage report as well
coverage_path = "data_clean/osap_signal_coverage.csv"
coverage_df.to_csv(coverage_path)
print(f"Coverage report saved to: {coverage_path}")

# ============================================================
# 7) Integration Note
# ============================================================

print("\n" + "="*60)
print("NEXT STEP: Merge into master panel")
print("="*60)
print("""
In build_master.py, after step 3 (CRSP factor variables), add:

    print("\\nLoading OSAP signals...")
    osap = pd.read_csv("data_clean/osap_signals_top500.csv")
    osap["permno"] = osap["permno"].astype("Int64")
    master = master.merge(
        osap,
        on=["permno", "year", "month"],
        how="left"
    )
    print("After OSAP merge:", master.shape)
""")

print("Script completed successfully.")
