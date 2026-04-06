# load_fred.py
# Pulls 6 core macro predictors from FRED + Shiller: Jan 1975 – Dec 2024
# Output: data_raw/macro_predictors.csv (monthly, end-of-month index)
#
# Requirements: pip install fredapi pandas numpy xlrd openpyxl
# Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
# Shiller data: download ie_data.xls → data_raw/ie_data.xls

import os
import pandas as pd
import numpy as np
from fredapi import Fred

# ── CONFIG ────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")   # set via: export FRED_API_KEY=your_key
START        = "1974-01-01"   # extra burn-in for log-diff / inflation calc
END          = "2024-12-31"

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHILLER_PATH = os.path.join(_ROOT, "data_raw", "Shiller_Data.xls")
# ─────────────────────────────────────────────────────────────────────────────


def pull(series_id, fred):
    s = fred.get_series(series_id, observation_start=START, observation_end=END)
    s = s.resample("ME").last()
    s.name = series_id
    return s


if __name__ == "__main__":
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable is not set. "
                         "Run: export FRED_API_KEY=your_key")

    fred = Fred(api_key=FRED_API_KEY)

    # ── FRED PULLS ────────────────────────────────────────────────────────────
    gs10   = pull("GS10",     fred)
    tb3ms  = pull("TB3MS",    fred)
    baa    = pull("BAA",      fred)
    aaa    = pull("AAA",      fred)
    cpi    = pull("CPIAUCSL", fred)
    indpro = pull("INDPRO",   fred)
    vix    = pull("VIXCLS",   fred)   # starts Jan 1990

    # ── SHILLER D/P RATIO (local file) ───────────────────────────────────────
    if not os.path.exists(SHILLER_PATH):
        raise FileNotFoundError(
            f"Shiller file not found at {SHILLER_PATH}. "
            "Download ie_data.xls from shiller.econ.yale.edu and place it in data_raw/"
        )

    shiller_raw = pd.read_excel(
        SHILLER_PATH,
        sheet_name="Data",
        skiprows=7,       # rows 0-6 are title/label rows; row 7 is the header
        engine="xlrd",
    )

    # Columns after skiprows=7: Date(0), P(1), D(2), E(3), ...
    shiller = shiller_raw.iloc[:, [0, 1, 2]].copy()
    shiller.columns = ["date_raw", "price", "dividend"]

    # Keep only numeric rows (drops trailing notes/blank rows)
    shiller["date_raw"] = pd.to_numeric(shiller["date_raw"], errors="coerce")
    shiller = shiller[shiller["date_raw"] > 1800].copy()

    # Shiller dates: decimal year — e.g. 1975.01 = Jan 1975, 1975.10 = Oct 1975
    # The fractional part is the month expressed as a 2-digit decimal (01–12)
    shiller["year"]  = shiller["date_raw"].astype(int)
    frac = (shiller["date_raw"] % 1 * 100).round().astype(int).replace(0, 1).clip(1, 12)
    shiller["month"] = frac
    shiller["date"]  = (
        pd.to_datetime(
            shiller["year"].astype(str) + "-" +
            shiller["month"].astype(str).str.zfill(2) + "-01"
        ) + pd.offsets.MonthEnd(0)
    )

    shiller = (
        shiller
        .set_index("date")
        .sort_index()[["price", "dividend"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    dp_ratio = (shiller["dividend"] / shiller["price"]).rename("dp_ratio")
    dp_ratio = dp_ratio.resample("ME").last()
    print(f"  Shiller D/P: {dp_ratio.first_valid_index().date()} → {dp_ratio.last_valid_index().date()}")

    # ── VIX PROXY FOR 1975-1989 ───────────────────────────────────────────────
    # Rolling 12-month realised volatility of S&P 500 (Shiller), scaled to
    # match the VIX level during the 1990+ overlap period.
    sp_price     = shiller["price"].resample("ME").last()
    sp_ret       = np.log(sp_price).diff()
    realised_vol = sp_ret.rolling(12).std() * np.sqrt(12) * 100
    realised_vol.name = "realised_vol"

    overlap = pd.concat([realised_vol, vix], axis=1).dropna()
    scale   = overlap.iloc[:, 1].mean() / overlap.iloc[:, 0].mean() if len(overlap) > 12 else 1.0
    print(f"  Realised vol → VIX scale factor: {scale:.3f}")

    volatility      = vix.combine_first(realised_vol * scale)
    volatility.name = "volatility"

    # ── ASSEMBLE ──────────────────────────────────────────────────────────────
    cpi_yoy = cpi.pct_change(12) * 100

    df = pd.DataFrame({
        "dp_ratio":        dp_ratio,
        "term_spread":     gs10 - tb3ms,
        "real_short_rate": tb3ms - cpi_yoy,
        "default_spread":  baa - aaa,
        "indpro_growth":   np.log(indpro).diff(1) * 100,
        "volatility":      volatility,
    })

    df = df.loc["1975-01-31":"2024-12-31"]

    # ── SAVE ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "data_raw", "macro_predictors.csv")
    df.to_csv(out_path)

    print(f"\nFinal shape: {df.shape}")
    print(f"Date range:  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSaved: {out_path}")
