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

# Signals to RETAIN — ≥80% non-null in existing top-500 master_panel
# (sourced from osap_signal_coverage.csv in the April 2026 audit)
SIGNALS_TO_KEEP = [
    # Momentum
    "mom1m", "mom6m", "mom12m", "mom36m", "indmom",
    # Liquidity / trading
    "ill", "dolvol", "zerotrade", "baspread",
    # Volatility
    "retvol", "idiovol", "maxret", "beta",
    # Valuation
    "ep", "cfp", "sp", "dy", "cashpr", "mvel1",
    # Profitability
    "roeq", "roaq", "gma", "chatoia",
    # Investment / accruals
    "agr", "grcapx", "grltnoa", "chinv",
    # Accruals / quality
    "acc", "absacc", "pctacc", "stdcf",
    # Other signals above 80% threshold
    "lev", "cashpr", "convind", "hire", "age",
    "cash", "cfp", "herf", "tb", "nincr", "rsup", "ear",
    "chtx", "chcsho", "invest",
    # Derived signals computed in build_master (not in OSAP raw, included for
    # completeness so the filter doesn't accidentally drop them)
    # betasq, std_dolvol, stdacc, sgr, bm_ia, cfp_ia, mve_ia, chpmia are
    # computed in build_master.py from other columns — not present in OSAP raw.
]

# Signals to EXPLICITLY EXCLUDE — near-zero or zero coverage in audit
SIGNALS_TO_DROP = [
    "chmom",     # 3.1% non-null — effectively unusable
    "turn",      # 16.1% non-null — severely sparse
    "std_turn",  # 0.2%  non-null — completely unusable
    "rd_mve",    # 0.0%  non-null — broken / zero coverage
    "sin",       # 7.4%  non-null — very sparse
    "ps",        # 4.2%  non-null — very sparse
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
    Uses case-insensitive matching because OSAP column names use mixed case.
    Returns the actual column names as they appear in the file.
    """
    # Build a lower-case → original-case mapping
    lower_map = {c.lower(): c for c in all_cols}

    keep = set(ID_COLS)
    for sig in SIGNALS_TO_KEEP:
        original = lower_map.get(sig.lower())
        if original:
            keep.add(original)
        # else: signal not in raw file (may be a derived signal) — skip silently

    # Remove explicitly excluded signals
    for sig in SIGNALS_TO_DROP:
        original = lower_map.get(sig.lower())
        if original and original in keep:
            keep.discard(original)

    # Preserve ID cols even if something went wrong
    for col in ID_COLS:
        keep.add(col)

    # Return in original column order
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
    print(f"  Dropped (low-coverage): {SIGNALS_TO_DROP}")

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
