"""
src/build_osap_expanded.py

Filter and trim the full 8.1 GB osap_signals_raw.csv to the expanded universe
produced by build_master.py (v2 construction).

Called by build_master.py after universe_monthly.csv has been written to
data_clean/. Can also be run standalone if universe_monthly.csv already exists.

Design choices
--------------
- Chunked read: the raw file is too large to load entirely into memory, so it
  is processed in chunks of CHUNK_SIZE rows. Only rows matching the expanded
  universe (permno, yyyymm) are retained.
- Signal selection: only signals with >=80% non-null coverage in the existing
  top-500 master_panel are retained (identified in the April 2026 data audit).
  Signals with near-zero or zero coverage are explicitly dropped.
- Output: data_clean/osap_signals_expanded.csv — referenced by build_master.py
  v2 pipeline.
"""

import os
import time

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

RAW_OSAP_PATH       = "data_raw/osap_signals_raw.csv"
UNIVERSE_PATH       = "data_clean/universe_monthly.csv"
OUTPUT_PATH         = "data_clean/osap_signals_expanded.csv"
CHUNK_SIZE          = 200_000   # rows per chunk (~200k rows ≈ manageable RAM)

# Raw OSAP column names to RETAIN from osap_signals_raw.csv.
# These are the OSAP-native names — NOT our model's naming convention.
# The COLUMN_RENAME_MAP below translates them to model names after loading.
#
# Root cause of the v2 characteristic gap (Apr 2026):
# The original list used model names (mom1m, idiovol, baspread …) instead of
# raw OSAP names (streversal, idiovol3f, bidaskspread …). resolve_keep_cols()
# does case-insensitive matching, which can't bridge a name mismatch, so 36
# signals were silently skipped and osap_signals_expanded.csv ended up with
# only 19 signals instead of the intended ~55.
SIGNALS_TO_KEEP_RAW = [
    # Momentum (raw OSAP names)
    "streversal",        # → mom1m  (1-month return)
    "mom6m",             # → mom6m  (exact match)
    "mom12m",            # → mom12m (exact match)
    "lrreversal",        # → mom36m (long-run reversal)
    "indmom",            # → indmom (exact match)
    "momrev",            # → chmom  (change in momentum)
    # Liquidity / trading
    "illiquidity",       # → ill    (Amihud illiquidity — also used directly)
    "dolvol",            # → dolvol (exact match)
    "zerotrade1m",       # → zerotrade
    "bidaskspread",      # → baspread
    "sharevol",          # → turn   (share turnover)
    "std_turn",          # kept as std_turn (already excluded below due to coverage)
    "pricedelayrsq",     # → pricedelay
    # Volatility
    "realizedvol",       # → retvol
    "idiovol3f",         # → idiovol (3-factor idiosyncratic vol)
    "maxret",            # → maxret (exact match)
    "beta",              # → beta   (exact match)
    # Valuation
    "ep",                # → ep     (exact match)
    "cfp",               # → cfp    (exact match)
    "sp",                # → sp     (exact match)
    "divyieldst",        # → dy
    "cashprod",          # → cashpr
    "size",              # → mvel1  (log market equity)
    "bm",                # → bm     (exact match — though computed in build_master too)
    # Profitability
    "roe",               # → roeq
    "roaq",              # → roaq   (exact match)
    "cboperprof",        # → gma
    "chassetturnover",   # → chatoia
    "operprof",          # → operprof (exact match)
    # Investment / accruals
    "assetgrowth",       # → agr
    "investment",        # → invest
    "grcapx",            # → grcapx (exact match)
    "grltnoa",           # → grltnoa (exact match)
    "chinv",             # → chinv  (exact match)
    # Accruals / quality
    "totalaccruals",     # → acc
    "abnormalaccruals",  # → absacc
    "pctacc",            # → pctacc (exact match)
    "varcf",             # → stdcf
    # Other characteristics
    "bookleverage",      # → lev
    "convdebt",          # → convind
    "hire",              # → hire   (exact match)
    "firmage",           # → age
    "cash",              # → cash   (exact match)
    "herf",              # → herf   (exact match)
    "tax",               # → tb
    "numearnincrease",   # → nincr
    "revenuesurprise",   # → rsup
    "earningssurprise",  # → ear
    "chtax",             # → chtx
    "compequiss",        # → chcsho
    "rds",               # → rd_sale
    "realestate",        # → realestate (exact match)
    "tang",              # → tang   (exact match)
    "ms",                # → ms     (exact match)
    "orgcap",            # → orgcap (exact match)
    "ps",                # → ps     (exact match — low coverage but let model drop it)
    "rd",                # → rd     (exact match)
    "noa",               # → noa    (exact match)
    "gp",                # → gp     (exact match)
    "accruals",          # → accruals (exact match)
    "sinalgo",           # → sin
    "grsaletogrinv",     # → pchsale_pchinvt
    "grsaletogroverhead",# → pchsale_pchxsga
    # Exact-match signals already in expanded (retained)
    "indmom", "maxret", "mom12m", "mom6m", "beta", "cash", "cfp", "chinv",
    "dolvol", "ep", "grcapx", "grltnoa", "herf", "hire", "pctacc", "roaq", "sp",
]
# Deduplicate while preserving order
_seen: set = set()
_deduped = []
for _x in SIGNALS_TO_KEEP_RAW:
    if _x not in _seen:
        _seen.add(_x)
        _deduped.append(_x)
SIGNALS_TO_KEEP_RAW = _deduped
del _seen, _deduped, _x

# Map raw OSAP column names → model characteristic names.
# Only entries that differ need to be listed here.
COLUMN_RENAME_MAP = {
    "streversal":        "mom1m",
    "lrreversal":        "mom36m",
    "momrev":            "chmom",
    "illiquidity":       "ill",
    "zerotrade1m":       "zerotrade",
    "bidaskspread":      "baspread",
    "sharevol":          "turn",
    "pricedelayrsq":     "pricedelay",
    "realizedvol":       "retvol",
    "idiovol3f":         "idiovol",
    "divyieldst":        "dy",
    "cashprod":          "cashpr",
    "size":              "mvel1",
    "roe":               "roeq",
    "cboperprof":        "gma",
    "chassetturnover":   "chatoia",
    "assetgrowth":       "agr",
    "investment":        "invest",
    "totalaccruals":     "acc",
    "abnormalaccruals":  "absacc",
    "varcf":             "stdcf",
    "bookleverage":      "lev",
    "convdebt":          "convind",
    "firmage":           "age",
    "tax":               "tb",
    "numearnincrease":   "nincr",
    "revenuesurprise":   "rsup",
    "earningssurprise":  "ear",
    "chtax":             "chtx",
    "compequiss":        "chcsho",
    "rds":               "rd_sale",
    "sinalgo":           "sin",
    "grsaletogrinv":     "pchsale_pchinvt",
    "grsaletogroverhead":"pchsale_pchxsga",
}

# Signals to EXPLICITLY EXCLUDE after loading — near-zero coverage in top-500 audit.
# Note: chmom/momrev IS included above (3.1% non-null in top-500 universe but may
# be higher in the broader v2 universe — let the model's char_missing_threshold drop
# it if truly sparse). std_turn and rd_mve remain excluded.
SIGNALS_TO_DROP_RAW = [
    "std_turn",  # 0.2%  non-null — completely unusable
    # rd_mve has no OSAP equivalent — absent from raw file
]

# Always include identifier columns
ID_COLS = ["permno", "yyyymm"]


def load_universe(path: str) -> set:
    """
    Load the monthly universe membership as a set of (permno, yyyymm) tuples
    for fast O(1) membership testing inside the chunk loop.
    """
    univ = pd.read_csv(path, usecols=["permno", "yyyymm"])
    univ["permno"] = univ["permno"].astype(int)
    univ["yyyymm"] = univ["yyyymm"].astype(int)
    pairs = set(zip(univ["permno"], univ["yyyymm"]))
    print(f"  Universe loaded: {len(pairs):,} (permno, yyyymm) pairs")
    return pairs


def resolve_keep_cols(all_cols: list) -> list:
    """
    Determine the final set of columns to keep from the raw OSAP file.
    Uses case-insensitive matching against SIGNALS_TO_KEEP_RAW (raw OSAP names).
    Returns the actual column names as they appear in the file (original case).
    Columns are lowercased in the chunk loop immediately after usecols selection,
    so the rename map can use lowercase keys safely.
    """
    lower_map = {c.lower(): c for c in all_cols}

    keep = set()
    # Always include ID columns (case-insensitively)
    for col in ID_COLS:
        original = lower_map.get(col.lower(), col)
        keep.add(original)

    for sig in SIGNALS_TO_KEEP_RAW:
        original = lower_map.get(sig.lower())
        if original:
            keep.add(original)

    # Remove explicitly excluded signals
    for sig in SIGNALS_TO_DROP_RAW:
        original = lower_map.get(sig.lower())
        if original and original in keep:
            keep.discard(original)

    return [c for c in all_cols if c in keep]


def build_osap_expanded(
    raw_path: str = RAW_OSAP_PATH,
    universe_path: str = UNIVERSE_PATH,
    output_path: str = OUTPUT_PATH,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Main entry point. Reads osap_signals_raw.csv in chunks, filters to universe
    members, trims columns, and writes osap_signals_expanded.csv.
    """
    print("\n" + "=" * 60)
    print("BUILD OSAP EXPANDED SIGNALS")
    print("=" * 60)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"OSAP raw file not found: {raw_path}")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}\n"
            "Run build_master.py (UNIVERSE_VERSION='v2') first to generate it."
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ── Load universe ────────────────────────────────────────────────────────
    print("\nStep 1: Loading universe membership...")
    universe_pairs = load_universe(universe_path)

    # ── Inspect raw file columns ─────────────────────────────────────────────
    print("\nStep 2: Reading column names from raw OSAP file...")
    header_df = pd.read_csv(raw_path, nrows=0)
    all_cols = list(header_df.columns)
    print(f"  Raw OSAP columns: {len(all_cols)}")

    keep_cols = resolve_keep_cols(all_cols)
    print(f"  Columns to retain: {len(keep_cols)} (including {len(ID_COLS)} ID columns)")
    print(f"  Signal columns: {len(keep_cols) - len(ID_COLS)}")
    print(f"  Dropped (low-coverage): {SIGNALS_TO_DROP_RAW}")

    # ── Chunked read and filter ───────────────────────────────────────────────
    print(f"\nStep 3: Filtering raw file in chunks of {chunk_size:,} rows...")
    t0 = time.time()

    chunks_out = []
    rows_read = 0
    rows_kept = 0
    chunk_num = 0

    reader = pd.read_csv(
        raw_path,
        usecols=keep_cols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        chunk_num += 1
        rows_read += len(chunk)

        # Lowercase all column names so the rename map (lowercase keys) works
        # regardless of the mixed-case naming in the raw OSAP file.
        chunk.columns = chunk.columns.str.lower()

        # Coerce ID columns to int for fast set lookup
        chunk["permno"] = pd.to_numeric(chunk["permno"], errors="coerce").astype("Int64")
        chunk["yyyymm"] = pd.to_numeric(chunk["yyyymm"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["permno", "yyyymm"])
        chunk["permno"] = chunk["permno"].astype(int)
        chunk["yyyymm"] = chunk["yyyymm"].astype(int)

        # Filter to universe members using vectorised isin on a merged index
        mask = pd.Series(
            list(zip(chunk["permno"], chunk["yyyymm"]))
        ).isin(universe_pairs).values
        filtered = chunk[mask].copy()

        rows_kept += len(filtered)
        if len(filtered) > 0:
            chunks_out.append(filtered)

        if chunk_num % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Chunk {chunk_num}: read {rows_read:,} rows, "
                  f"kept {rows_kept:,} ({100*rows_kept/rows_read:.1f}%), "
                  f"{elapsed:.0f}s elapsed")

    elapsed_total = time.time() - t0
    print(f"\n  Done. Total rows read: {rows_read:,} | Kept: {rows_kept:,} "
          f"({100*rows_kept/max(rows_read,1):.1f}%) | Time: {elapsed_total:.0f}s")

    # ── Combine and save ──────────────────────────────────────────────────────
    print("\nStep 4: Combining chunks and saving...")
    if not chunks_out:
        raise ValueError("No rows matched the universe — check permno/yyyymm alignment.")

    result = pd.concat(chunks_out, ignore_index=True)
    result = result.sort_values(["permno", "yyyymm"]).reset_index(drop=True)

    # Deduplicate on (permno, yyyymm) — keep last (most recently read row)
    before_dedup = len(result)
    result = result.drop_duplicates(subset=["permno", "yyyymm"], keep="last")
    after_dedup = len(result)
    if before_dedup > after_dedup:
        print(f"  Deduplication: dropped {before_dedup - after_dedup:,} duplicate (permno, yyyymm) rows")

    # Rename raw OSAP column names → model characteristic names
    rename_actual = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in result.columns}
    result = result.rename(columns=rename_actual)
    if rename_actual:
        print(f"  Renamed {len(rename_actual)} columns to model convention: "
              f"{list(rename_actual.items())[:5]}{'…' if len(rename_actual) > 5 else ''}")

    # Add year and month for convenience
    result["year"]  = result["yyyymm"] // 100
    result["month"] = result["yyyymm"] % 100

    result.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Shape: {result.shape[0]:,} rows × {result.shape[1]} cols")
    print(f"  Date range: {result['yyyymm'].min()} to {result['yyyymm'].max()}")
    print(f"  Unique permnos: {result['permno'].nunique():,}")

    # ── Coverage report ───────────────────────────────────────────────────────
    print("\nStep 5: Signal coverage summary (% non-null):")
    signal_cols = [c for c in result.columns if c not in ID_COLS + ["year", "month"]]
    coverage = {}
    for col in signal_cols:
        pct = 100 * result[col].notna().mean()
        coverage[col] = pct
    coverage_df = pd.DataFrame.from_dict(
        coverage, orient="index", columns=["pct_nonnull"]
    ).sort_values("pct_nonnull", ascending=False)
    print(coverage_df.to_string())

    # Save coverage report
    coverage_path = "data_clean/osap_expanded_coverage.csv"
    coverage_df.to_csv(coverage_path)
    print(f"\n  Coverage report saved to: {coverage_path}")

    print("\n" + "=" * 60)
    print("OSAP EXPANDED BUILD COMPLETE")
    print("=" * 60)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    build_osap_expanded()
