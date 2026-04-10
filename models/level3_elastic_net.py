#!/usr/bin/env python3
"""
Level 3: Elastic Net — GKX-style pooled panel expected return model
models/level3_elastic_net.py

This module implements Level 3 of the multi-model empirical asset pricing
pipeline. Expected returns are estimated via elastic net penalised regression
on a large GKX-style feature matrix:
  - Firm characteristics (cross-sectionally ranked to [-1, 1])
  - Characteristic × macro-variable interaction terms
  - 2-digit SIC industry dummies

Two loss functions run in parallel:
  - OLS (L2)  : sklearn ElasticNet
  - Huber     : proximal gradient descent (ISTA) with elastic net penalty

The critical design invariant: *no information from month t or later may be
used when generating forecasts for month t*.  Every imputation median, feature
drop decision, and industry encoding is determined from training data only.
An explicit assertion check is performed at each re-estimation date.

References
----------
Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning",
    Review of Financial Studies 33(5), 2223–2273.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Constants
# ============================================================

FIRM_CHARACTERISTICS: List[str] = [
    "market_cap", "illiquidity", "reversal_st",
    "momentum", "reversal_lt", "be", "bm", "noa", "gp", "roa", "capinv",
    "leverage", "asset_growth", "accruals", "mom1m", "mom6m", "mom12m",
    "mom36m", "chmom", "indmom", "maxret", "pricedelay", "mvel1", "dolvol",
    "ill", "turn", "std_turn", "zerotrade", "baspread", "retvol", "idiovol",
    "beta", "absacc", "acc", "age", "agr", "cash", "cashpr", "cfp", "chatoia",
    "chcsho", "chinv", "convind", "divi", "divo", "dy", "ep", "gma", "grcapx",
    "grltnoa", "herf", "hire", "invest", "lev", "operprof", "orgcap",
    "pchsale_pchinvt", "pchsale_pchxsga", "pctacc", "ps", "rd", "rd_mve",
    "rd_sale", "realestate", "roaq", "sp", "tang", "sin", "tb", "chtx", "ms",
    "nincr", "stdcf", "roeq", "rsup", "ear", "betasq", "std_dolvol", "stdacc",
    "sgr", "bm_ia", "cfp_ia", "mve_ia", "chpmia",
]

# Macro variables used as interaction terms (GKX use 8 Welch-Goyal vars).
# We have 6 FRED-sourced equivalents + rf = 7 total. The constant term acts
# as the 8th interactor, giving z_it = c_it ⊗ [macro_1, …, macro_7, 1].
# GKX vars not available in our data: ep (E/P ratio), bm (book-to-market),
# ntis (net equity issuance).  dp_ratio ≈ GKX dp; term_spread ≈ tms;
# real_short_rate ≈ tbl; default_spread ≈ dfy; volatility ≈ svar;
# indpro_growth has no direct GKX equivalent but captures business-cycle risk.
MACRO_VARS: List[str] = [
    "dp_ratio", "term_spread", "real_short_rate", "default_spread",
    "indpro_growth", "volatility", "rf",
]

DEFAULT_CONFIG: Dict = {
    # ── Window parameters ────────────────────────────────────────────────
    "train_start_year": 1973,
    "train_end_year":   2004,   # data through Dec of this year is in training
    "test_start_year":  2005,
    "test_end_year":    2024,
    # Fix 1: 5-year rolling validation window for hyperparameter tuning.
    # 1-year window (~6k obs) is too thin to tune 100+ combos on a 920-feature model.
    # 5 years (~30k obs) gives stable rankings without incurring leakage.
    "val_window_years": 5,

    # ── Hyperparameter grid ──────────────────────────────────────────────
    # alpha  = regularisation strength (λ in the paper)
    # l1_ratio = weight on L1 vs L2 (sklearn convention: 1 = LASSO, 0 = ridge)
    # Alpha grid: logspace -3→1 (20 values).  With PCA pre-projection the
    # effective number of features is ~50-100 orthogonal PCs (vs 920 correlated
    # raw features), so the per-feature penalty scales differently. logspace(-3,1)
    # covers alpha=0.001 (near-unpenalised) through alpha=10 (strong shrinkage).
    # KNS: L2-only ridge already works well; the combined approach needs less
    # alpha range than raw-feature elastic net.
    "alpha_grid":         list(np.logspace(-4, 1, 20)),  # extended lower bound to 0.0001

    # l1_ratio grid: capped at 0.5.  KNS (p.279) is explicit that near-LASSO
    # performs poorly when regressors are correlated.  Even after PCA the
    # PC scores share information from correlated raw characteristics (same
    # macro variables interact with all chars).  Removing values > 0.5 prevents
    # the model from collapsing to a near-empty solution.  Pure ridge (0.0) is
    # included as the KNS-motivated anchor.
    "l1_ratio_grid":      [0.0, 0.01, 0.1, 0.3, 0.5],
    "huber_epsilon":      1.35,       # Huber threshold default (overridden by tuning)
    "huber_epsilon_grid": [0.5, 0.7, 0.9, 1.35],

    # ── n_cv_splits: used only when tune_hyperparameters falls back to
    #    CV-within mode (no external train set).  In the default rolling
    #    validation mode (X_train_ext provided) this setting is ignored.
    "n_cv_splits": 5,

    # ── Feature construction ─────────────────────────────────────────────
    # Loosened to 0.70: elastic net can zero out uninformative sparse features
    # itself via the L1 penalty, so it is better to let more characteristics in
    # and let regularisation do the selection rather than pre-filtering too aggressively.
    "char_missing_threshold": 0.70,  # drop char if > 70 % missing in training

    # ── Quality filters ──────────────────────────────────────────────────
    "min_train_obs":        1000,
    "min_stocks_per_month": 50,

    # ── Paths ────────────────────────────────────────────────────────────
    "data_path":  "data_clean/master_panel.csv",   # overridden below from config
    "macro_path": "data_raw/macro_predictors.csv",
    "output_dir": "data_clean/elastic_net",
    "cache_dir":  "data_clean/elastic_net/cache",

    # ── Run mode ─────────────────────────────────────────────────────────
    "run_mode":      "full",   # 'full' or 'test'
    "test_n_stocks": 50,       # max stocks in test mode (sampled by permno)
    "test_n_years":  2,        # number of re-estimation years in test mode

    # ── Parallel CV ──────────────────────────────────────────────────────
    "n_jobs": -1,

    # ── Solver settings ──────────────────────────────────────────────────
    "max_iter":            5000,   # OLS ElasticNet iters — final model fit
    "cv_ols_max_iter":     500,    # OLS ElasticNet iters — CV tuning only
                                   # (500 is enough to rank combos; saves ~10×)
    "tol":                 1e-4,   # final model convergence tolerance
    "cv_tol":              1e-3,   # CV tuning tolerance (looser is fine for ranking)
    # huber_max_iter / huber_tol used by fit_huber_model_sgd (SGDRegressor max_iter/tol)
    "huber_max_iter":      2000,
    "cv_huber_max_iter":   200,
    "huber_tol":           1e-4,
    "power_iter_n":        20,     # power-iteration steps for Lipschitz constant (legacy FISTA)

    # ── CV training subsample ────────────────────────────────────────────
    # Fix 7: raise cap to 50k for more stable hyperparameter ranking with the
    # 5-year val window.  OLS SGD is cheap enough that 50k vs 20k costs little.
    "cv_max_train_n":      50_000,

    # ── Forecast post-processing ─────────────────────────────────────────
    # Primary winsorisation at 1% per tail (GKX standard).  Raw (unwinsorised)
    # forecasts are also saved as expected_ret_raw_ols / expected_ret_raw_huber
    # so any threshold can be applied downstream without re-running the model.
    "forecast_winsor_pct": 0.01,

    # ── Target variable ──────────────────────────────────────────────────
    # industry_adj_target: subtract the equal-weight industry mean return from
    # each stock's excess return within each (year, month, sic2) cell before
    # fitting.  This focuses the model on within-industry cross-sectional
    # variation and typically improves predictive R².  No leakage: industry means
    # are computed from the contemporaneous cross-section, not future data.
    "industry_adj_target": False,

    # ── Imputation ───────────────────────────────────────────────────────
    # Use SIC peer-group median for missing characteristics instead of the global
    # cross-sectional median.  More accurate imputation for sparse characteristics.
    "industry_imputation": True,

    # ── Variable importance ──────────────────────────────────────────────
    # Fix 8: limit permutation-VI computation to top N features by coefficient
    # magnitude (OLS ∪ Huber).  Full-feature VI on 920 cols × val window is slow.
    "max_vi_features":     50,

    # ── PCA pre-projection ───────────────────────────────────────────────
    # KNS (2020): elastic net on raw correlated characteristics collapses to
    # near-empty solutions (3-8 non-zero coefs); the same model in PC space
    # retains 10-30 orthogonal components and achieves much higher OOS R².
    # L1 regularisation works correctly on orthogonal features — it selects
    # the most informative PCs rather than arbitrarily picking one member of
    # each correlated group.
    #
    # pca_n_components: float → fraction of variance explained (0.95 retains
    # ~50-100 PCs from 920 raw features); int → exact number of components;
    # None → disabled (raw features passed through, original behaviour).
    "pca_n_components":    0.95,
}

# Override data_path from portfolio/config.py if build_master has been run.
# Falls back to the hardcoded default above if the config file doesn't exist.
try:
    from portfolio.config import PANEL_PATH as _PANEL_PATH  # type: ignore
    DEFAULT_CONFIG["data_path"] = _PANEL_PATH
except ImportError:
    pass

# ============================================================
# Logging
# ============================================================

def setup_logging(config: Dict) -> logging.Logger:
    """Configure module-level logger with console + (optional) file handler."""
    logger = logging.getLogger("elastic_net")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    out_dir = Path(config.get("output_dir", "data_clean/elastic_net"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "elastic_net_run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Data Loading
# ============================================================

def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load master panel and macro predictor files.

    Returns
    -------
    df : pd.DataFrame
        Master panel with (permno, year, month) as identifiers.
    macro_df : pd.DataFrame
        Macro predictors with integer (year, month) columns added.
    """
    logger = logging.getLogger("elastic_net")

    logger.info("Loading master panel from %s", config["data_path"])
    df = pd.read_csv(config["data_path"], low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df["permno"]       = pd.to_numeric(df["permno"],       errors="coerce").astype("Int64")
    df["year"]         = pd.to_numeric(df["year"],         errors="coerce").astype(int)
    df["month"]        = pd.to_numeric(df["month"],        errors="coerce").astype(int)
    df["ret_adjusted"] = pd.to_numeric(df["ret_adjusted"], errors="coerce")
    df["rf"]           = pd.to_numeric(df["rf"],           errors="coerce")
    df["siccd"]        = pd.to_numeric(df["siccd"],        errors="coerce")
    df.dropna(subset=["permno", "year", "month"], inplace=True)
    df["permno"] = df["permno"].astype(int)
    logger.info("Master panel loaded: %s rows, %s cols", *df.shape)

    logger.info("Loading macro predictors from %s", config["macro_path"])
    macro_df = pd.read_csv(config["macro_path"], index_col=0, parse_dates=True)
    macro_df.columns = [c.strip().lower() for c in macro_df.columns]
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df["year"]  = macro_df.index.year
    macro_df["month"] = macro_df.index.month
    macro_df = macro_df.reset_index(drop=True)
    # Coerce all macro columns to numeric
    for col in [c for c in macro_df.columns if c not in ("year", "month")]:
        macro_df[col] = pd.to_numeric(macro_df[col], errors="coerce")
    logger.info("Macro predictors loaded: %s rows", len(macro_df))

    return df, macro_df


def merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join macro predictors onto the master panel by (year, month).
    `rf` is already in master_panel; macro_df supplies the 6 FRED variables.

    No look-ahead risk: macro variables are already lagged at source per the
    data documentation. We merge by the same (year, month) key.
    """
    fred_cols = [c for c in MACRO_VARS if c != "rf"]
    available_fred = [c for c in fred_cols if c in macro_df.columns]
    merge_cols = ["year", "month"] + available_fred
    merged = df.merge(macro_df[merge_cols], on=["year", "month"], how="left")
    return merged


# ============================================================
# Feature Matrix Construction
# ============================================================

def _rank_series_pm1(s: pd.Series) -> pd.Series:
    """
    Cross-sectionally rank a series to [-1, 1] using fractional ranking.

    rank_i = 2 * (rank(x_i) / (n + 1)) - 1

    NaNs are preserved (they are not included in the rank denominator).
    """
    n_valid = s.notna().sum()
    if n_valid == 0:
        return s
    ranks = s.rank(method="average", na_option="keep")
    return 2.0 * (ranks / (n_valid + 1)) - 1.0


def select_active_chars(
    df_train: pd.DataFrame,
    candidates: List[str],
    threshold: float,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Identify characteristics usable for modelling.

    A characteristic is dropped if it is missing in more than `threshold`
    fraction of training rows.  Medians are computed on remaining training
    rows only (per characteristic) and are used to impute missing values
    everywhere.

    Parameters
    ----------
    df_train   : training-window rows of master panel
    candidates : full list of characteristic names
    threshold  : drop if fraction missing > this value

    Returns
    -------
    active_chars  : list of characteristic names that pass the filter
    train_medians : dict mapping char_name -> cross-sectional median value
    """
    available = [c for c in candidates if c in df_train.columns]
    active, medians = [], {}
    for col in available:
        col_data = pd.to_numeric(df_train[col], errors="coerce")
        frac_missing = col_data.isna().mean()
        if frac_missing > threshold:
            continue
        medians[col] = col_data.median()
        active.append(col)
    return active, medians


def select_active_chars_industry(
    df_train: pd.DataFrame,
    active_chars: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Compute per-characteristic medians within each 2-digit SIC group.

    Pooling all industries to compute a single median biases imputed values
    for characteristics with strong industry structure (e.g. leverage is high
    for utilities, low for tech). Industry-adjusted medians impute each firm
    with the median of its own 2-digit SIC peer group, falling back to the
    global median (from select_active_chars) for firms in unseen industries.

    Computed strictly on training data — df_window rows must never be passed here.

    Parameters
    ----------
    df_train     : training-window DataFrame (must contain 'siccd' column)
    active_chars : characteristics that passed the missing-value filter

    Returns
    -------
    industry_medians : nested dict  industry_medians[char][sic2_code] = median
    """
    work = pd.DataFrame(df_train).copy()
    sic2_raw = pd.to_numeric(work["siccd"], errors="coerce")
    work["_sic2"] = (sic2_raw.fillna(-10) // 10).astype(int)

    industry_medians: Dict[str, Dict[int, float]] = {}
    for col in active_chars:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        grp_medians: Dict[int, float] = {}
        grouped = work.groupby("_sic2")[col].median()
        counts  = work.groupby("_sic2")[col].count()
        for sic2_code in grouped.index:
            if counts.loc[sic2_code] >= 5:   # require ≥5 obs for a stable median
                val = grouped.loc[sic2_code]
                if pd.notna(val):
                    grp_medians[int(sic2_code)] = float(val)
        industry_medians[col] = grp_medians

    return industry_medians


def get_industry_codes(df_train: pd.DataFrame) -> List[int]:
    """
    Extract unique 2-digit SIC codes observed in training data.

    The encoding (which codes exist) is fixed to training data so that
    a new industry in the test period gets all-zero dummy rows (no leakage).
    """
    sic2 = (df_train["siccd"].dropna() // 10).astype(int)
    return sorted(sic2.unique().tolist())


def fit_macro_scaler(
    df_train: pd.DataFrame,
    macro_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a standardisation transform for macro variables using training data only.

    Macro variables span very different scales (e.g. dp_ratio ≈ 0.04,
    volatility ≈ 15–40). Without standardisation the elastic net penalty shrinks
    coefficients on large-scale interactions far more aggressively than on
    small-scale ones — purely an artefact of units, not signal.  Standardising
    to zero mean / unit variance on the training window removes this distortion.

    Returns
    -------
    macro_means : ndarray, shape (len(macro_cols),)
    macro_stds  : ndarray, shape (len(macro_cols),)
        Standard deviations; any zero std is replaced with 1.0 to avoid division
        by zero (column stays at zero after mean-centering).
    """
    means, stds = [], []
    for mcol in macro_cols:
        vals = pd.to_numeric(df_train[mcol], errors="coerce").dropna()
        means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        std = float(vals.std()) if len(vals) > 1 else 1.0
        stds.append(std if std > 1e-12 else 1.0)
    return np.array(means, dtype=np.float64), np.array(stds, dtype=np.float64)


def _build_single_window(
    df_window: pd.DataFrame,
    active_chars: List[str],
    train_medians: Dict[str, float],
    macro_cols: List[str],
    industry_codes: List[int],
    macro_means: np.ndarray,
    macro_stds: np.ndarray,
    industry_medians: Optional[Dict[str, Dict[int, float]]] = None,
    industry_adj_target: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List]:
    """
    Build the GKX feature matrix and excess-return target for one data window.

    Processing steps (applied in order, using training statistics throughout):
      1. Compute excess return target: ret_adjusted - rf
         (optionally industry-adjusted: subtract equal-weight sic2 mean per month)
      2. Impute each characteristic with its training-window median
      3. Cross-sectionally rank each characteristic to [-1, 1] within each month
      4. Construct interaction terms: char_i × macro_j for j in {1, constant}
      5. Add 2-digit SIC industry dummies (encoding fixed to training codes)
      6. Return (X, y, feature_names) with no-NaN guarantee on y rows

    Parameters
    ----------
    df_window      : panel rows for the window (train, val, or test)
    active_chars   : characteristics passing the missing-value filter
    train_medians  : median values fitted on training data for imputation
    macro_cols     : macro column names present in df_window
    industry_codes : sorted list of 2-digit SIC codes from training
    macro_means    : per-variable means fitted on training data (shape: n_macro)
    macro_stds     : per-variable stds fitted on training data (shape: n_macro)
                     Applied to macro vars before interaction construction so all
                     interaction terms are on a comparable scale.

    Returns
    -------
    X            : float32 ndarray, shape (n_valid, n_features)
    y            : float32 ndarray, shape (n_valid,)  — excess returns
    feature_names: list of n_features column labels
    valid_idx    : integer index into df_window for rows included in X, y
    """
    # ── Target: excess return ────────────────────────────────────────────
    # Compute excess return = ret_adjusted - rf.  Rows where either is NaN
    # are dropped; valid_idx records which df_window rows survive.
    excess_ret = (
        pd.to_numeric(df_window["ret_adjusted"], errors="coerce")
        - pd.to_numeric(df_window["rf"], errors="coerce")
    )
    valid_mask = excess_ret.notna()
    df = df_window[valid_mask]           # view — no copy needed
    y_series = excess_ret[valid_mask]

    if industry_adj_target:
        # Subtract equal-weight sic2 industry mean from each stock's excess return
        # within each (year, month) cross-section.  This is contemporaneous — no
        # future data is used — so there is no leakage.
        sic2_for_target = (
            pd.to_numeric(df["siccd"], errors="coerce").fillna(-1) // 10
        ).astype(int)
        ind_month_key = df["year"].astype(str) + "_" + df["month"].astype(str) + "_" + sic2_for_target.astype(str)
        ind_means = y_series.groupby(ind_month_key).transform("mean")
        y_series  = y_series - ind_means

    y = y_series.values.astype(np.float32)
    valid_idx = df_window.index[valid_mask].tolist()
    n_rows = len(df)

    # ── Impute missing characteristics ───────────────────────────────────
    # Each missing value is filled with the training-window cross-sectional
    # median for that characteristic (GKX footnote 30).  When
    # industry_medians is provided (industry_imputation=True) we use the
    # firm's own 2-digit SIC peer-group median instead, falling back to the
    # global median for unseen industries.
    sic2_col = (pd.to_numeric(df["siccd"], errors="coerce").fillna(-10) // 10).astype(int)
    char_data = pd.DataFrame(index=df.index)
    for col in active_chars:
        series = pd.to_numeric(df[col], errors="coerce")
        if industry_medians is not None and col in industry_medians:
            fill_vals = sic2_col.map(industry_medians[col]).fillna(train_medians.get(col, 0.0))
            series    = series.fillna(fill_vals)
        else:
            series = series.fillna(train_medians.get(col, 0.0))
        char_data[col] = series

    # ── Cross-sectional rank to [-1, 1] within each month ────────────────
    # For each calendar month, rank each characteristic across all stocks in
    # that month, then normalise ranks to [-1, 1].
    #
    # PERFORMANCE NOTE: We iterate over unique (year, month) groups (n_months
    # iterations) and apply scipy.stats.rankdata across ALL characteristics
    # simultaneously in a single C-level call per month.  This avoids the
    # prior pandas groupby.transform approach which made n_months × n_chars
    # separate Python function calls (~51,000 calls for a 30-year window).
    char_arr_raw = char_data.values.astype(np.float64)      # (n, C)
    ranked       = np.zeros((n_rows, len(active_chars)), dtype=np.float32)
    months_key   = df["year"].values * 100 + df["month"].values
    _, month_inv = np.unique(months_key, return_inverse=True)
    for m_idx in range(month_inv.max() + 1):
        row_mask = month_inv == m_idx
        grp      = char_arr_raw[row_mask]                   # (n_m, C)
        n_m      = grp.shape[0]
        if n_m <= 1:
            ranked[row_mask] = 0.0
            continue
        # rankdata(axis=0): ranks 1..n_m per column, handles ties via 'average'
        ranks           = rankdata(grp, method="average", axis=0).astype(np.float32)
        ranked[row_mask] = 2.0 * ranks / (n_m + 1) - 1.0

    # ── Macro variables for this window ──────────────────────────────────
    # Macro vars are already merged onto df_window.  Standardise using
    # training-window mean/std so all interaction terms are on a comparable
    # scale (prevents the penalty from shrinking large-scale interactions
    # like volatility ≈ 20 far more than small-scale ones like dp_ratio ≈ 0.04).
    macro_arr = np.zeros((n_rows, len(macro_cols)), dtype=np.float32)
    for j, mcol in enumerate(macro_cols):
        if mcol in df.columns:
            col_vals = pd.to_numeric(df[mcol], errors="coerce").fillna(macro_means[j]).values
            macro_arr[:, j] = ((col_vals - macro_means[j]) / macro_stds[j]).astype(np.float32)

    # ── Interaction terms: char_i × [1, macro_0, …, macro_M] ─────────────
    # PERFORMANCE NOTE: Single numpy broadcast replaces a double Python loop
    # (previously n_chars × (n_macros+1) = ~680 column-level assignments).
    #
    # Build interactors matrix (n, M+1) = [1, macro_0, …, macro_M]:
    interactors = np.concatenate(
        [np.ones((n_rows, 1), dtype=np.float32), macro_arr], axis=1
    )  # (n, M+1)
    #
    # Outer product char × interactor → (n, C, M+1), then flatten to (n, C*(M+1)).
    # Column order: char_0_const, char_0_macro0, …, char_0_macroM, char_1_const, …
    X_interact = (ranked[:, :, None] * interactors[:, None, :]).reshape(n_rows, -1)
    #
    # Build feature names in matching order (cheap — just strings).
    feat_names: List[str] = []
    interactor_names = ["const"] + list(macro_cols)
    for cname in active_chars:
        for iname in interactor_names:
            feat_names.append(f"{cname}__{iname}")

    # ── Industry dummies ─────────────────────────────────────────────────
    # Encoding is fixed to training-data SIC codes so unseen industries in
    # the test period produce all-zero dummy rows (no leakage).
    #
    # PERFORMANCE NOTE: Vectorized numpy indexing replaces a Python for-loop
    # over n_rows (previously O(n_rows) Python iterations).
    sic2       = (pd.to_numeric(df["siccd"], errors="coerce").fillna(-1) // 10).astype(int)
    ind_lookup = {code: idx for idx, code in enumerate(industry_codes)}
    col_idx_arr = np.fromiter(
        (ind_lookup.get(c, -1) for c in sic2.values),
        dtype=np.int32, count=n_rows,
    )
    X_ind = np.zeros((n_rows, len(industry_codes)), dtype=np.float32)
    valid_ind = col_idx_arr >= 0
    X_ind[np.nonzero(valid_ind)[0], col_idx_arr[valid_ind]] = 1.0
    ind_names = [f"sic2_{code}" for code in industry_codes]

    # ── Assemble final feature matrix ─────────────────────────────────────
    # Final shape: (n_rows, n_chars*(n_macros+1) + n_industry_codes)
    X            = np.concatenate([X_interact, X_ind], axis=1)
    feature_names = feat_names + ind_names

    return X, y, feature_names, valid_idx


# ============================================================
# Leakage Guard
# ============================================================

def check_no_leakage(
    train_end_year: int,
    train_end_month: int,
    test_year: int,
    logger: logging.Logger,
) -> None:
    """
    Assert that the training window ends strictly before the test year.

    This is the central no-leakage invariant for the whole pipeline.  Any
    violation would corrupt all downstream comparisons across model levels.
    """
    train_end_ym = train_end_year * 100 + train_end_month
    test_start_ym = test_year * 100 + 1
    assert train_end_ym < test_start_ym, (
        f"LEAKAGE DETECTED: training ends {train_end_year}-{train_end_month:02d} "
        f"but test year is {test_year}.  "
        "No test-period data may appear in the training window."
    )
    logger.debug(
        "Leakage check passed: train ends %d-%02d, test starts %d-01",
        train_end_year, train_end_month, test_year,
    )


# ============================================================
# GKX Out-of-Sample R²
# ============================================================

def compute_gkx_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    GKX out-of-sample R² with zero as the benchmark forecast.

    R²_oos = 1 - SS_res / SS_tot
    where SS_tot uses zero (not the historical mean) as benchmark,
    following equation (A.1) in Gu et al. (2020).

    A negative value means the model is worse than always predicting zero.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ============================================================
# Huber Elastic Net — ISTA Solver
# ============================================================

def _spectral_norm_power_iter(X: np.ndarray, n_iter: int = 20) -> float:
    """
    Approximate the largest singular value of X via power iteration.

    Returns sigma_max(X), used to compute the Lipschitz constant
    L = sigma_max^2 / n for the Huber gradient w.r.t. w.
    """
    rng = np.random.default_rng(0)
    p = X.shape[1]
    v = rng.standard_normal(p).astype(np.float64)
    v /= np.linalg.norm(v)
    sigma = 1.0
    for _ in range(n_iter):
        u = X @ v
        sigma = np.linalg.norm(u)
        if sigma < 1e-12:
            break
        u /= sigma
        v = X.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            break
        v /= v_norm
    return float(sigma)


def _huber_gradient(residuals: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Gradient of the Huber loss w.r.t. residuals.

    Returns the per-sample gradient: -∂L/∂ŷ_i (negated for residual convention).
    """
    # ∂L/∂r_i where r_i = y_i - ŷ_i:
    #   -r_i          if |r_i| ≤ ε
    #   -ε * sign(r_i) otherwise
    grad = np.where(np.abs(residuals) <= epsilon,
                    -residuals,
                    -epsilon * np.sign(residuals))
    return grad


def fit_huber_enet(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    epsilon: float = 1.35,
    max_iter: int = 1000,
    tol: float = 1e-4,
    power_iter_n: int = 20,
    warm_coef: Optional[np.ndarray] = None,
    precomputed_sigma_sq_n: Optional[float] = None,
) -> np.ndarray:
    """
    Fit Huber loss + elastic net penalty via FISTA (Beck & Teboulle 2009).

    FISTA adds Nesterov momentum to plain ISTA, achieving O(1/k²) convergence
    vs O(1/k) for ISTA — typically 10–100× fewer iterations in practice.

    Objective (minimised over w, with intercept absorbed into X via a
    column of ones):

        (1/n) * Σ huber(y_i - X_i w; ε)
        + alpha * [ l1_ratio * ||w||_1 + 0.5 * (1 - l1_ratio) * ||w||_2^2 ]

    Parameters
    ----------
    X                    : feature matrix, float64, shape (n, p); standardised
    y                    : target vector, float64, shape (n,)
    alpha                : regularisation strength
    l1_ratio             : balance between L1 (=1) and L2 (=0) penalties
    epsilon              : Huber threshold (default 1.35 ≈ 95th pct of N(0,1))
    max_iter             : maximum FISTA iterations
    tol                  : convergence tolerance on max absolute coef change
    power_iter_n         : iterations for spectral-norm approximation
    warm_coef            : initial coefficient vector (warm start across λ path)
    precomputed_sigma_sq_n : if provided, skip the expensive spectral norm
                             computation and use this value as sigma_max²/n.
                             Pre-compute once per fold in tune_hyperparameters
                             and pass here to avoid 400× redundant computations.

    Returns
    -------
    w            : float64 ndarray, shape (p,) — fitted coefficients (including
                   intercept if a bias column was appended to X)
    converged    : bool — True if the tolerance criterion was met before max_iter
    n_iter_taken : int  — number of FISTA iterations actually executed
    final_delta  : float — max absolute coefficient change at termination;
                   should be ≤ tol if converged, larger otherwise
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape

    # ── Lipschitz constant of the smooth part of the objective ────────────
    # L_smooth = sigma_max(X)²/n  +  alpha*(1 - l1_ratio)
    # When pre-computed sigma_sq_n is available (tuning loop), skip the
    # expensive power iteration (20 matrix-vector products on a 200k×700
    # matrix).  This avoids ~400 redundant computations per tuning year.
    if precomputed_sigma_sq_n is not None:
        sigma_sq_n = precomputed_sigma_sq_n
    else:
        sigma_max  = _spectral_norm_power_iter(X, n_iter=power_iter_n)
        sigma_sq_n = sigma_max ** 2 / n
    L_smooth = sigma_sq_n + alpha * (1.0 - l1_ratio)
    if L_smooth < 1e-10:
        L_smooth = 1e-10
    step = 1.0 / L_smooth

    # Coefficient initialisation
    w = np.zeros(p, dtype=np.float64) if warm_coef is None else warm_coef.copy()

    l1_thresh = alpha * l1_ratio * step   # soft-threshold for L1 proximal step

    converged    = False
    final_delta  = np.inf
    n_iter_taken = 0

    # FISTA momentum variables (Beck & Teboulle 2009)
    y_k = w.copy()   # momentum iterate (extrapolation point)
    t_k = 1.0

    for it in range(max_iter):
        n_iter_taken = it + 1

        # Gradient step on y_k (momentum iterate), not w
        residuals   = y - X @ y_k
        huber_g     = _huber_gradient(residuals, epsilon)          # (n,)
        grad_smooth = X.T @ huber_g / n + alpha * (1.0 - l1_ratio) * y_k

        w_half = y_k - step * grad_smooth

        # Proximal operator for L1 (soft thresholding)
        w_new = np.sign(w_half) * np.maximum(np.abs(w_half) - l1_thresh, 0.0)

        # FISTA momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y_k   = w_new + ((t_k - 1.0) / t_new) * (w_new - w)

        final_delta = np.max(np.abs(w_new - w))
        w, t_k = w_new, t_new
        if final_delta < tol:
            converged = True
            break

    return w, converged, n_iter_taken, final_delta


# ============================================================
# Hyperparameter Tuning
# ============================================================

def _cv_score_ols(
    scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> float:
    """Return mean GKX R² across pre-scaled folds for OLS elastic net.

    PERFORMANCE: Accepts pre-scaled (X_tr_s, y_tr, X_vl_s, y_vl) tuples so
    StandardScaler is NOT re-fitted on every alpha/l1_ratio combo call.
    tune_hyperparameters() fits the scaler once per fold before the parallel
    grid search, reducing scaling work from n_combos× to 1×.
    """
    scores    = []
    warm_coef = None
    for X_tr_s, y_tr, X_vl_s, y_vl in scaled_folds:
        # Each alpha/l1_ratio gets its own ElasticNet instance; warm-start
        # across folds within one combo (not across combos).
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter, tol=tol,
            warm_start=True, selection="cyclic",
        )
        if warm_coef is not None:
            try:
                model.coef_      = warm_coef["coef"]
                model.intercept_ = warm_coef["intercept"]
            except Exception:
                pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_s, y_tr)
        warm_coef = {"coef": model.coef_.copy(), "intercept": model.intercept_}
        scores.append(compute_gkx_r2(y_vl, model.predict(X_vl_s)))
    return float(np.mean(scores)) if scores else -np.inf


def _cv_score_huber(
    scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sigma_sq_n_list: List[float],
    alpha: float,
    l1_ratio: float,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> float:
    """Return mean GKX R² across pre-scaled folds for Huber elastic net.

    PERFORMANCE: Accepts pre-scaled fold data AND pre-computed sigma_sq_n
    (= sigma_max²/n of the augmented training matrix) per fold.  This avoids
    running power iteration inside fit_huber_enet on every combo call —
    previously causing ~400 redundant spectral-norm computations per tuning
    year, each taking ~0.5 s on a 200k×700 matrix (≈3 min wasted per year).
    """
    scores    = []
    warm_coef = None
    for (X_tr_s, y_tr, X_vl_s, y_vl), sigma_sq_n in zip(scaled_folds, sigma_sq_n_list):
        # Augment with intercept column (bias absorbed into w)
        X_tr_aug = np.hstack([X_tr_s, np.ones((len(X_tr_s), 1))])
        X_vl_aug = np.hstack([X_vl_s, np.ones((len(X_vl_s), 1))])
        w, _conv, _nit, _delta = fit_huber_enet(
            X_tr_aug, y_tr,
            alpha=alpha, l1_ratio=l1_ratio, epsilon=epsilon,
            max_iter=max_iter, tol=tol,
            warm_coef=warm_coef,
            precomputed_sigma_sq_n=sigma_sq_n,   # skip power iteration
        )
        warm_coef = w.copy()
        scores.append(compute_gkx_r2(y_vl, X_vl_aug @ w))
    return float(np.mean(scores)) if scores else -np.inf


def tune_hyperparameters(
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    loss_type: str,
    logger: logging.Logger,
    X_train_ext: Optional[np.ndarray] = None,
    y_train_ext: Optional[np.ndarray] = None,
) -> Tuple[float, float, pd.DataFrame, float]:
    """
    Grid search over (alpha, l1_ratio) — and epsilon for Huber — to select
    hyperparameters that maximise out-of-sample R² on the validation window.

    GKX rolling-validation mode (X_train_ext provided)
    ---------------------------------------------------
    Each hyperparameter combination is fit on X_train_ext/y_train_ext and
    evaluated on X_val/y_val.  This implements GKX's rolling 12-month
    validation: train on data up to year t-2, validate on year t-1.
    Temporal ordering is strictly preserved; no random shuffling.

    Legacy CV-within mode (X_train_ext = None)
    -------------------------------------------
    TimeSeriesSplit is applied within X_val/y_val itself (n_cv_splits folds).
    Used as a fallback when no external training set is provided.

    For Huber, epsilon is jointly tuned over huber_epsilon_grid alongside
    alpha and l1_ratio, selecting the triple that maximises validation R².
    The 1.35 default is appropriate for N(0,1) but monthly equity returns
    are fat-tailed; smaller epsilon (0.5–0.9) typically performs better.

    Parameters
    ----------
    X_val        : feature matrix for validation window, shape (n_val, p)
    y_val        : excess returns for validation window, shape (n_val,)
    config       : pipeline config dict
    loss_type    : 'ols' or 'huber'
    X_train_ext  : (optional) external training feature matrix, shape (n_tr, p)
    y_train_ext  : (optional) external training labels, shape (n_tr,)

    Returns
    -------
    best_alpha    : chosen regularisation strength
    best_l1_ratio : chosen L1 fraction
    surface_df    : DataFrame recording R² for every grid cell
    best_epsilon  : chosen Huber epsilon (np.nan for OLS)
    """
    alpha_grid    = config["alpha_grid"]
    l1_ratio_grid = config["l1_ratio_grid"]
    n_splits      = config["n_cv_splits"]
    n_jobs        = config["n_jobs"]
    power_iter_n  = config.get("power_iter_n", 20)

    # ── Build CV split indices ────────────────────────────────────────────
    if X_train_ext is not None and y_train_ext is not None:
        # GKX rolling mode: single fold — train on X_train_ext, evaluate on X_val
        n_tr = len(X_train_ext)
        n_vl = len(X_val)
        X_tv = np.vstack([X_train_ext, X_val])
        y_tv = np.concatenate([y_train_ext, y_val])
        split_indices = [(np.arange(n_tr), np.arange(n_tr, n_tr + n_vl))]
    else:
        # Legacy mode: CV within X_val
        X_tv = X_val
        y_tv = y_val
        tscv = TimeSeriesSplit(n_splits=n_splits)
        split_indices = list(tscv.split(X_val))

    if loss_type == "ols":
        epsilon_grid = [np.nan]   # placeholder, not used
    else:
        epsilon_grid = config.get("huber_epsilon_grid", [config["huber_epsilon"]])

    n_folds = len(split_indices)
    logger.debug(
        "  Tuning %s: %d α × %d ρ%s grid, %d fold(s), n_train=%d n_val=%d",
        loss_type.upper(), len(alpha_grid), len(l1_ratio_grid),
        f" × {len(epsilon_grid)} ε" if loss_type == "huber" else "",
        n_folds, len(y_tv) - len(y_val), len(y_val),
    )

    # ── Pre-scale folds ONCE (before the parallel grid search) ───────────
    # PERFORMANCE: StandardScaler is fitted once per fold on the training
    # split, then the pre-scaled arrays are reused across all alpha/l1_ratio
    # combos.  Previously, the scaler was re-fitted inside each parallel
    # worker call — 100× for OLS and 400× for Huber on identical data.
    #
    # Training fold subsample: cap at cv_max_train_n rows so that each
    # combo fit takes O(cv_max_train_n) not O(full_expanding_window).
    # The val fold is NOT subsampled — the R² evaluation is on the full
    # rolling 12-month window so rankings remain reliable.
    cv_max_train_n = config.get("cv_max_train_n", 20_000)
    rng_sub        = np.random.default_rng(42)   # fixed seed → reproducible

    scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in split_indices:
        X_tr_full = X_tv[train_idx].astype(np.float64)
        y_tr_full = y_tv[train_idx]
        # Subsample training fold if it exceeds cv_max_train_n
        if len(X_tr_full) > cv_max_train_n:
            sub_idx  = rng_sub.choice(len(X_tr_full), size=cv_max_train_n, replace=False)
            X_tr_sub = X_tr_full[sub_idx]
            y_tr_sub = y_tr_full[sub_idx]
        else:
            X_tr_sub = X_tr_full
            y_tr_sub = y_tr_full
        X_vl = X_tv[val_idx].astype(np.float64)
        y_vl = y_tv[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_sub)   # scaler fitted on subsampled fold
        X_vl_s = scaler.transform(X_vl)
        scaled_folds.append((X_tr_s, y_tr_sub, X_vl_s, y_vl))

    # ── For Huber: pre-compute spectral norm ONCE per fold ────────────────
    # PERFORMANCE: sigma_max²/n of the augmented training matrix is the same
    # for all (alpha, l1_ratio, epsilon) combos.  Previously computed inside
    # fit_huber_enet → 400 redundant power-iteration runs per tuning year,
    # each ~0.5 s on a 200k×700 matrix (≈3 min/year, 60 min total).
    sigma_sq_n_list: List[float] = []
    if loss_type == "huber":
        for X_tr_s, y_tr, _, _ in scaled_folds:
            X_aug    = np.hstack([X_tr_s, np.ones((len(X_tr_s), 1))])
            sigma    = _spectral_norm_power_iter(X_aug, n_iter=power_iter_n)
            sigma_sq_n_list.append(sigma ** 2 / len(X_tr_s))
        logger.debug(
            "  Pre-computed sigma_sq_n per fold: %s",
            [f"{v:.4f}" for v in sigma_sq_n_list],
        )

    # ── Evaluate every grid cell ─────────────────────────────────────────
    # CV uses reduced max_iter and looser tol vs. the final model fit:
    #   OLS:   cv_ols_max_iter=500,  cv_tol=1e-3  (final: 5000, 1e-4)
    #   Huber: cv_huber_max_iter=200, huber_tol    (final: 1000)
    # This is 5-10× cheaper per combo; ranking quality is unaffected because
    # the coordinate-descent solution is accurate enough after 500 steps to
    # correctly order good vs. bad hyperparameters.
    cv_ols_max_iter   = config.get("cv_ols_max_iter",   500)
    cv_tol            = config.get("cv_tol",             1e-3)
    cv_huber_max_iter = config.get("cv_huber_max_iter",  200)

    if loss_type == "ols":
        _raw_ols = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_cv_score_ols)(
                scaled_folds, a, r, cv_ols_max_iter, cv_tol,
            )
            for a in alpha_grid
            for r in l1_ratio_grid
        )
        cv_results: List[float] = [v if isinstance(v, float) else -np.inf for v in (_raw_ols or [])]
        surface_rows = []
        idx = 0
        best_score, best_alpha, best_l1, best_eps = (
            -np.inf, alpha_grid[0], l1_ratio_grid[0], np.nan,
        )
        for a in alpha_grid:
            for r in l1_ratio_grid:
                score = cv_results[idx]
                surface_rows.append({"alpha": a, "l1_ratio": r, "val_r2": score})
                if score > best_score:
                    best_score, best_alpha, best_l1 = score, a, r
                idx += 1

    else:  # huber — joint search over (alpha, l1_ratio, epsilon)
        _raw_hub = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_cv_score_huber)(
                scaled_folds, sigma_sq_n_list, a, r, eps,
                cv_huber_max_iter, config["huber_tol"],
            )
            for eps in epsilon_grid
            for a in alpha_grid
            for r in l1_ratio_grid
        )
        cv_results = [v if isinstance(v, float) else -np.inf for v in (_raw_hub or [])]
        surface_rows = []
        idx = 0
        best_score, best_alpha, best_l1, best_eps = (
            -np.inf, alpha_grid[0], l1_ratio_grid[0], epsilon_grid[0],
        )
        for eps in epsilon_grid:
            for a in alpha_grid:
                for r in l1_ratio_grid:
                    score = cv_results[idx]
                    surface_rows.append({
                        "alpha": a, "l1_ratio": r, "epsilon": eps, "val_r2": score,
                    })
                    if score > best_score:
                        best_score, best_alpha, best_l1, best_eps = score, a, r, eps
                    idx += 1

    surface_df = pd.DataFrame(surface_rows)

    # Fallback: if all solutions are degenerate (e.g. all-zero coef)
    if best_score < -10.0:
        best_alpha = float(np.median(alpha_grid))
        best_l1    = 0.5
        best_eps   = config.get("huber_epsilon", 1.35) if loss_type == "huber" else np.nan
        logger.warning(
            "  %s tuning: all CV scores < -10 (best=%.4f). "
            "Falling back to α=%.4f, ρ=%.2f%s",
            loss_type.upper(), best_score, best_alpha, best_l1,
            f", ε={best_eps:.2f}" if loss_type == "huber" else "",
        )

    if loss_type == "huber":
        logger.debug(
            "  Huber best: α=%.5f, l1_ratio=%.2f, ε=%.2f, val_R²=%.4f",
            best_alpha, best_l1, best_eps, best_score,
        )
    else:
        logger.debug(
            "  OLS best: α=%.5f, l1_ratio=%.2f, val_R²=%.4f",
            best_alpha, best_l1, best_score,
        )

    return best_alpha, best_l1, surface_df, best_eps


def _select_epsilon_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    l1_ratio: float,
    epsilon_grid: List[float],
    max_n: int = 5_000,
) -> float:
    """
    Select the Huber epsilon parameter via a fast SGDRegressor sweep.

    This replaces the expensive FISTA-based Huber tuning loop.  The
    regularization geometry (alpha, l1_ratio) is identical between OLS and
    Huber elastic net — only the loss function differs — so running the full
    hyperparameter grid under OLS loss and then doing a single cheap epsilon
    sweep here gives equivalent regularization selection at a fraction of the
    cost.

    Steps
    -----
    1. Subsample training data to max_n rows (random, fixed seed).
    2. Fit StandardScaler on the subsampled training data.
    3. For each epsilon in epsilon_grid: fit SGDRegressor(loss='huber') and
       evaluate GKX R² on the full validation fold.
    4. Return the epsilon with the highest validation R².

    SGDRegressor is 10-50× faster than FISTA for this purpose because:
      - No spectral norm computation is needed (step size is adaptive).
      - Early stopping eliminates wasted iterations.
      - The subsample (max_n=5k) keeps each call well under 0.1 s.

    Runtime: 4 epsilon values × ~0.05 s each ≈ 0.2 s per year.
    vs. FISTA tuning: 400 combos × ~1.5 s each ≈ 600 s per year.
    """
    rng = np.random.default_rng(42)

    # ── Subsample training fold ───────────────────────────────────────────
    X_tr = X_train.astype(np.float64)
    y_tr = y_train.astype(np.float64)
    if len(X_tr) > max_n:
        idx  = rng.choice(len(X_tr), size=max_n, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]

    # ── Scale using training statistics ──────────────────────────────────
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_vl_s   = scaler.transform(X_val.astype(np.float64))
    y_vl     = y_val.astype(np.float64)

    # ── Sweep epsilon values ──────────────────────────────────────────────
    best_eps = epsilon_grid[0] if epsilon_grid else 1.35
    best_r2  = -np.inf
    for eps in epsilon_grid:
        model = SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            epsilon=eps,
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            learning_rate="optimal",
            fit_intercept=True,
            n_iter_no_change=5,
            early_stopping=False,    # val set not passed; fit on subsample
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_s, y_tr)
        r2 = compute_gkx_r2(y_vl, model.predict(X_vl_s))
        if r2 > best_r2:
            best_r2  = r2
            best_eps = eps
    return float(best_eps)


# ============================================================
# Final Model Fitting
# ============================================================

def fit_pca_projector(
    X_train: np.ndarray,
    n_components,
) -> Tuple[StandardScaler, PCA]:
    """
    Fit a (StandardScaler → PCA) pipeline on training data.

    PCA requires centred/scaled input so the scaler is fitted first.
    Callers apply this projector to val/test data using the same fitted
    objects — no leakage because the projector is never re-fitted on
    out-of-sample data.

    Parameters
    ----------
    X_train      : raw feature matrix, shape (n, p)
    n_components : passed to PCA — float for variance explained (e.g. 0.95),
                   int for exact number of components, None to skip PCA
                   (returns identity projector with StandardScaler only)

    Returns
    -------
    scaler : fitted StandardScaler
    pca    : fitted PCA  (or None if n_components is None)
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float64))
    if n_components is None:
        return scaler, None
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    pca.fit(X_s)
    return scaler, pca


def apply_pca_projector(
    scaler: StandardScaler,
    pca: Optional[PCA],
    X: np.ndarray,
) -> np.ndarray:
    """
    Apply a fitted (StandardScaler → PCA) projector to X.

    If pca is None, returns standardised X (scaler only).
    """
    X_s = scaler.transform(X.astype(np.float64))
    if pca is None:
        return X_s
    return pca.transform(X_s)


def fit_ols_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    l1_ratio: float,
    config: Dict,
) -> Tuple[ElasticNet, StandardScaler]:
    """
    Fit final OLS elastic net on the full training window.

    Standardises X before fitting so that the elastic net penalty treats all
    features on equal footing — consistent with the Huber path which also
    standardises.  The fitted scaler must be used when predicting.

    Returns
    -------
    model  : fitted ElasticNet (trained on standardised features)
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float64))
    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=config["max_iter"],
        tol=config["tol"],
        warm_start=True,
        selection="cyclic",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_s, y_train.astype(np.float64))
    return model, scaler


def fit_huber_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    l1_ratio: float,
    config: Dict,
    epsilon: Optional[float] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit final Huber elastic net on the full training window via ISTA.

    Returns
    -------
    w       : coefficient vector (includes intercept as last element)
    scaler  : fitted StandardScaler (must be used when predicting)
    """
    logger = logging.getLogger("elastic_net")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float64))
    # Append intercept column (not penalised — handled by ISTA's inclusion)
    ones = np.ones((len(X_s), 1))
    X_aug = np.hstack([X_s, ones])
    effective_epsilon = epsilon if epsilon is not None else config["huber_epsilon"]
    w, converged, n_iter_taken, final_delta = fit_huber_enet(
        X_aug, y_train.astype(np.float64),
        alpha=alpha, l1_ratio=l1_ratio,
        epsilon=effective_epsilon,
        max_iter=config["huber_max_iter"],
        tol=config["huber_tol"],
        power_iter_n=config.get("power_iter_n", 20),
    )
    if not converged:
        logger.warning(
            "  Huber ISTA did NOT converge: %d / %d iterations, "
            "final_delta=%.2e (tol=%.2e). "
            "Coefficients may be imprecise — consider increasing huber_max_iter.",
            n_iter_taken, config["huber_max_iter"], final_delta, config["huber_tol"],
        )
    else:
        logger.debug(
            "  Huber ISTA converged in %d iterations (final_delta=%.2e)",
            n_iter_taken, final_delta,
        )
    return w, scaler


def fit_huber_model_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    l1_ratio: float,
    epsilon: float,
    config: Dict,
) -> Tuple["SGDRegressor", StandardScaler]:
    """
    Fit Huber elastic net using SGDRegressor (replaces FISTA-based fit_huber_model).

    Runtime: ~30–60s on 235k×920 versus 20–40 min for FISTA.
    Across 20 re-estimation years this saves ~7 hours.

    Parameters
    ----------
    X_train  : raw (unscaled) training features
    y_train  : training labels
    alpha    : elastic net regularisation strength
    l1_ratio : L1/(L1+L2) mixing parameter
    epsilon  : Huber loss epsilon (insensitivity band)
    config   : pipeline config dict (uses huber_max_iter, huber_tol)

    Returns
    -------
    model  : fitted SGDRegressor
    scaler : StandardScaler fitted on X_train
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train.astype(np.float64))

    model = SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        epsilon=epsilon,
        max_iter=config.get("huber_max_iter", 2000),
        tol=config.get("huber_tol", 1e-4),
        random_state=42,
        # "adaptive" with fixed eta0 is more stable than "optimal" (which uses
        # 1/(alpha*t) step sizes — problematic when alpha is very small on large N).
        learning_rate="adaptive",
        eta0=0.01,
        fit_intercept=True,
        # Disable early stopping in the final fit: on 170k+ rows the held-out
        # validation fraction has different statistics and causes premature stopping.
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_s, y_train.astype(np.float64))

    return model, scaler


# ============================================================
# Forecast Post-processing
# ============================================================

def _winsorize_forecasts(
    df_valid: pd.DataFrame,
    pred_col: str,
    pct: float = 0.01,
) -> np.ndarray:
    """
    Winsorise cross-sectional forecasts at ``pct`` each tail, per month.

    Fix 4: prevents outlier predictions (common with high-alpha elastic net
    on rare macro-×-char interactions) from dominating portfolio construction.
    Applies symmetrically: lower tail clipped to pct-quantile, upper tail to
    (1-pct)-quantile, computed separately within each (year, month) group.

    Parameters
    ----------
    df_valid : DataFrame with columns [year, month] plus pred_col
    pred_col : name of the forecast column
    pct      : tail fraction to clip (default 0.01 = 1 %)

    Returns
    -------
    np.ndarray of winsorised forecasts aligned with df_valid.index
    """
    result = df_valid[pred_col].copy()
    for _, grp_idx in df_valid.groupby(["year", "month"]).groups.items():
        vals = result.loc[grp_idx]
        lo = vals.quantile(pct)
        hi = vals.quantile(1.0 - pct)
        result.loc[grp_idx] = vals.clip(lo, hi)
    return result.values


# ============================================================
# Variable Importance
# ============================================================

def compute_variable_importance(
    model_ols: ElasticNet,
    scaler_ols: StandardScaler,
    model_huber: "SGDRegressor",
    scaler_huber: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    max_features: int = 50,
) -> pd.DataFrame:
    """
    Estimate variable importance as reduction in out-of-sample R² when
    each predictor is zeroed out (computed on the validation window).

    Only evaluates up to ``max_features`` features by coefficient magnitude
    (union of OLS and Huber non-zero sets, capped for speed).

    Returns
    -------
    DataFrame with columns: [feature, coef_ols, coef_huber,
                              vi_ols, vi_huber]
    where vi_* = baseline_R² - R²_with_feature_zeroed.
    """
    # Baseline R² for OLS
    X_val_ols_s  = scaler_ols.transform(X_val.astype(np.float64))
    y_base_ols   = model_ols.predict(X_val_ols_s)
    base_r2_ols  = compute_gkx_r2(y_val, y_base_ols)

    # Baseline R² for Huber (SGDRegressor)
    X_val_hub_s   = scaler_huber.transform(X_val.astype(np.float64))
    y_base_huber  = model_huber.predict(X_val_hub_s)
    base_r2_huber = compute_gkx_r2(y_val, y_base_huber)

    ols_coefs   = model_ols.coef_
    huber_coefs = model_huber.coef_  # SGDRegressor.coef_ excludes intercept

    nonzero_ols   = np.where(np.abs(ols_coefs)   > 1e-10)[0]
    nonzero_huber = np.where(np.abs(huber_coefs) > 1e-10)[0]
    candidate_idx = np.union1d(nonzero_ols, nonzero_huber)

    # Cap to max_features by combined absolute coefficient magnitude
    if len(candidate_idx) > max_features:
        combined_mag = np.abs(ols_coefs[candidate_idx]) + np.abs(huber_coefs[candidate_idx])
        top_local = np.argsort(combined_mag)[::-1][:max_features]
        candidate_idx = candidate_idx[top_local]

    rows = []
    for i in candidate_idx:
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"

        # OLS: zero out feature i in standardised space
        X_zeroed_ols_s = X_val_ols_s.copy()
        X_zeroed_ols_s[:, i] = 0.0
        vi_ols = base_r2_ols - compute_gkx_r2(y_val, model_ols.predict(X_zeroed_ols_s))

        # Huber: zero out feature i in standardised space
        X_zeroed_hub_s = X_val_hub_s.copy()
        X_zeroed_hub_s[:, i] = 0.0
        vi_huber = base_r2_huber - compute_gkx_r2(y_val, model_huber.predict(X_zeroed_hub_s))

        rows.append({
            "feature":    fname,
            "coef_ols":   float(ols_coefs[i])   if i < len(ols_coefs)   else 0.0,
            "coef_huber": float(huber_coefs[i])  if i < len(huber_coefs) else 0.0,
            "vi_ols":     float(vi_ols),
            "vi_huber":   float(vi_huber),
        })

    return pd.DataFrame(rows).sort_values("vi_ols", ascending=False)


# ============================================================
# Main Pipeline Function
# ============================================================

def run_elastic_net(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Run the full elastic net pipeline.

    Parameters
    ----------
    df        : master_panel DataFrame (firm × month rows)
    macro_df  : macro predictors DataFrame with (year, month) columns
    config    : dict overriding any keys in DEFAULT_CONFIG

    Returns
    -------
    results : dict containing all artefacts (DataFrames ready to save)
    """
    # ── Merge config ─────────────────────────────────────────────────────
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)

    logger = setup_logging(cfg)
    logger.info("=" * 65)
    logger.info("LEVEL 3: ELASTIC NET  —  run_mode=%s", cfg["run_mode"].upper())
    logger.info("=" * 65)

    # ── Optionally restrict to test-mode subsample ────────────────────────
    if cfg["run_mode"] == "test":
        logger.info("TEST MODE: limiting to %d stocks, %d re-estimation years",
                    cfg["test_n_stocks"], cfg["test_n_years"])
        all_permnos = sorted(df["permno"].unique())
        rng = np.random.default_rng(42)
        selected = rng.choice(all_permnos, size=min(cfg["test_n_stocks"], len(all_permnos)), replace=False)
        df = df[df["permno"].isin(selected)].copy()
        # Shrink test window
        cfg["test_end_year"] = cfg["test_start_year"] + cfg["test_n_years"] - 1

    # ── Merge macro onto panel ────────────────────────────────────────────
    logger.info("Merging macro predictors onto master panel …")
    df = merge_macro(df, macro_df)
    # Determine which macro columns are actually available
    available_macro = [m for m in MACRO_VARS if m in df.columns]
    missing_macro = [m for m in MACRO_VARS if m not in df.columns]
    if missing_macro:
        logger.warning("Macro columns not found in merged panel: %s", missing_macro)

    # ── Determine re-estimation years ────────────────────────────────────
    test_years = list(range(cfg["test_start_year"], cfg["test_end_year"] + 1))
    logger.info("Test period: %d – %d  (%d re-estimation points)",
                cfg["test_start_year"], cfg["test_end_year"], len(test_years))

    # ── Output accumulation containers ───────────────────────────────────
    all_expected_ols      : List[pd.DataFrame] = []
    all_expected_huber    : List[pd.DataFrame] = []
    all_expected_raw_ols  : List[pd.DataFrame] = []
    all_expected_raw_huber: List[pd.DataFrame] = []
    all_oos_r2            : List[Dict]         = []
    all_metadata          : List[Dict]         = []
    all_feat_selection    : List[pd.DataFrame] = []
    all_var_importance    : List[pd.DataFrame] = []
    all_val_surfaces      : List[pd.DataFrame] = []
    all_forecast_stability: List[Dict]         = []  # Diagnostic 1

    # Track previous-year OLS/Huber forecasts for rank-IC (Spearman correlation
    # between consecutive years' cross-sectional forecasts on common stocks).
    prev_forecasts_ols   : Optional[pd.Series] = None
    prev_forecasts_huber : Optional[pd.Series] = None

    total_start = time.time()

    # ── Main loop: one iteration per re-estimation year ───────────────────
    # GKX rolling-validation design (Section 3.3):
    #   - Train:      train_start_year … test_year - 1  (expanding window)
    #   - Validation: val_start_year … test_year - 1    (val_window_years rolling)
    #   - Test:       test_year      (held-out forecasts)
    #
    # Fix 1: validation window is val_window_years (default 5) instead of 1 year.
    # Tuning train excludes the full validation window: train_start … val_start - 1.
    # The final model is fit on the full training window (train_start … test_year - 1).
    val_w = cfg.get("val_window_years", 5)
    year_pbar = tqdm(test_years, desc="Elastic Net", unit="year", ncols=90, leave=True)
    for reest_idx, test_year in enumerate(year_pbar):
        loop_start = time.time()
        train_end_year = test_year - 1         # final model uses data through this year
        val_start_year = test_year - val_w     # rolling N-year validation start
        val_end_year   = test_year - 1         # validation ends the year before test
        tune_train_end = val_start_year - 1    # tuning train excludes entire val window

        year_pbar.set_postfix(year=test_year, stage="tuning")
        logger.info("")
        logger.info("─" * 55)
        logger.info(
            "RE-ESTIMATION  %d / %d  →  val=%d–%d (%dy), forecast=%d",
            reest_idx + 1, len(test_years),
            val_start_year, val_end_year, val_w, test_year,
        )
        logger.info("─" * 55)

        # ── Leakage check (central invariant) ────────────────────────────
        # The val window ends at test_year-1, so train_end_year = test_year-1
        # and test data is test_year — no overlap.
        check_no_leakage(train_end_year, 12, test_year, logger)

        # ── Slice tuning training window (train_start … test_year-2) ─────
        # Encodings (char selection, medians, macro scaler, industry codes)
        # are fitted on this window to avoid any look-ahead into val year.
        mask_tune_train = (
            (df["year"] >= cfg["train_start_year"]) &
            (df["year"] <= tune_train_end)
        )
        df_tune_train = pd.DataFrame(df[mask_tune_train])

        if len(df_tune_train) < cfg["min_train_obs"]:
            logger.warning(
                "  Skipping %d: only %d tuning-train obs (min=%d)",
                test_year, len(df_tune_train), cfg["min_train_obs"],
            )
            continue

        tune_chars, tune_medians = select_active_chars(
            df_tune_train, FIRM_CHARACTERISTICS, cfg["char_missing_threshold"],
        )
        tune_industry  = get_industry_codes(df_tune_train)
        tune_macro_means, tune_macro_stds = fit_macro_scaler(df_tune_train, available_macro)

        # ── Build tuning feature matrices ─────────────────────────────────
        ind_adj = cfg.get("industry_adj_target", False)
        X_tune_train, y_tune_train, _, _ = _build_single_window(
            df_tune_train, tune_chars, tune_medians,
            available_macro, tune_industry,
            tune_macro_means, tune_macro_stds,
            industry_adj_target=ind_adj,
        )

        # Fix 1: validation window spans val_start_year … val_end_year (val_w years)
        mask_val = (df["year"] >= val_start_year) & (df["year"] <= val_end_year)
        df_val_roll = pd.DataFrame(df[mask_val])
        X_val_roll, y_val_roll, _, _ = _build_single_window(
            df_val_roll, tune_chars, tune_medians,
            available_macro, tune_industry,
            tune_macro_means, tune_macro_stds,
            industry_adj_target=ind_adj,
        )
        logger.info(
            "  Tuning train: {:,} rows ({} – {})  |  Val: {:,} rows ({} – {})".format(
                X_tune_train.shape[0], cfg["train_start_year"], tune_train_end,
                X_val_roll.shape[0], val_start_year, val_end_year,
            )
        )

        # ── PCA pre-projection (tuning phase) ─────────────────────────────
        # KNS (2020): elastic net in PC space retains 10-30 informative
        # components instead of collapsing to 3-8 raw-feature coefficients.
        # PCA is fitted on tune training data only; val projection uses the
        # same fitted objects — no leakage.
        pca_k = cfg.get("pca_n_components", 0.95)
        pca_scaler_tune, pca_tune = fit_pca_projector(X_tune_train, pca_k)
        X_tune_train_pc = apply_pca_projector(pca_scaler_tune, pca_tune, X_tune_train)
        X_val_roll_pc   = apply_pca_projector(pca_scaler_tune, pca_tune, X_val_roll)
        n_pcs_tune = X_tune_train_pc.shape[1]
        logger.info(
            "  PCA (tuning): %d raw features → %d PCs  (%.1f%% variance)",
            X_tune_train.shape[1], n_pcs_tune,
            100 * (pca_tune.explained_variance_ratio_.sum() if pca_tune is not None else 1.0),
        )

        # ── Tune hyperparameters: fit on tune_train, evaluate on val ──────
        logger.info("  Tuning OLS hyperparameters (rolling val %d–%d) …", val_start_year, val_end_year)
        t0 = time.time()
        best_alpha_ols, best_l1_ols, val_surface_ols, _ = tune_hyperparameters(
            X_val_roll_pc, y_val_roll, cfg, "ols", logger,
            X_train_ext=X_tune_train_pc, y_train_ext=y_tune_train,
        )
        logger.info("  OLS chosen: α=%.5f, l1_ratio=%.2f  (%.1fs)",
                    best_alpha_ols, best_l1_ols, time.time() - t0)

        # Diagnostic 2: warn if chosen alpha is at a grid boundary
        alpha_grid  = cfg["alpha_grid"]
        l1_grid     = cfg["l1_ratio_grid"]
        if best_alpha_ols <= min(alpha_grid) * 1.01:
            logger.warning("  [GRID BOUNDARY] alpha=%.5f at lower bound (%.5f) — consider extending grid",
                           best_alpha_ols, min(alpha_grid))
        if best_alpha_ols >= max(alpha_grid) * 0.99:
            logger.warning("  [GRID BOUNDARY] alpha=%.5f at upper bound (%.5f) — consider extending grid",
                           best_alpha_ols, max(alpha_grid))
        if best_l1_ols <= min(l1_grid) * 1.01:
            logger.warning("  [GRID BOUNDARY] l1_ratio=%.2f at lower bound (%.2f)",
                           best_l1_ols, min(l1_grid))
        if best_l1_ols >= max(l1_grid) * 0.99:
            logger.warning("  [GRID BOUNDARY] l1_ratio=%.2f at upper bound (%.2f)",
                           best_l1_ols, max(l1_grid))

        # Fix 5: reuse OLS hyperparameters for Huber (same α, l1_ratio);
        # only sweep epsilon via a fast SGD sweep — eliminates 400-combo FISTA tuning.
        best_alpha_hub = best_alpha_ols
        # Floor l1_ratio at 0.1 for Huber SGD: pure ridge (l1_ratio=0) with
        # learning_rate="optimal" produces step sizes 1/(alpha*t) that are
        # too large on 170k+ rows, causing SGD divergence.
        best_l1_hub    = max(best_l1_ols, 0.1)
        logger.info("  Selecting Huber epsilon (fast SGD sweep, val %d–%d) …", val_start_year, val_end_year)
        t0 = time.time()
        best_epsilon_hub = _select_epsilon_fast(
            X_tune_train_pc, y_tune_train,
            X_val_roll_pc, y_val_roll,
            alpha=best_alpha_hub,
            l1_ratio=best_l1_hub,
            epsilon_grid=cfg["huber_epsilon_grid"],
        )
        logger.info("  Huber ε=%.2f  (reuses OLS α=%.5f, l1_ratio=%.2f)  (%.1fs)",
                    best_epsilon_hub, best_alpha_hub, best_l1_hub, time.time() - t0)

        val_surface_ols["reest_year"] = test_year
        val_surface_ols["loss"]       = "ols"
        all_val_surfaces.append(val_surface_ols)

        # ── Final training window (train_start … test_year-1) ────────────
        # Re-fit encodings on the full training window (including val year)
        # so the final model benefits from the most recent data.
        mask_train = (
            (df["year"] >= cfg["train_start_year"]) &
            (df["year"] <= train_end_year)
        )
        df_train = pd.DataFrame(df[mask_train])
        logger.info("  Final training rows: {:,}  ({} – {})".format(
            len(df_train), cfg["train_start_year"], train_end_year))

        if len(df_train) < cfg["min_train_obs"]:
            logger.warning(
                "  Skipping %d: only %d training obs (min=%d)",
                test_year, len(df_train), cfg["min_train_obs"],
            )
            continue

        # ── Determine active characteristics, imputation medians, macro scaler
        # and industry encoding — all strictly from training data.
        active_chars, train_medians = select_active_chars(
            df_train, FIRM_CHARACTERISTICS, cfg["char_missing_threshold"],
        )
        industry_codes = get_industry_codes(df_train)
        macro_means, macro_stds = fit_macro_scaler(df_train, available_macro)

        # GKX use simple cross-sectional median imputation (footnote 30).
        # industry_imputation=True is an enhancement available via config;
        # False (the default) matches the paper's specification.
        use_ind_impute = cfg.get("industry_imputation", False)
        ind_medians: Optional[Dict[str, Dict[int, float]]] = (
            select_active_chars_industry(df_train, active_chars)
            if use_ind_impute else None
        )
        logger.info("  Active chars: %d  |  industry codes: %d  |  ind_impute=%s",
                    len(active_chars), len(industry_codes), use_ind_impute)

        # ── Build training feature matrix ─────────────────────────────────
        logger.info("  Building training feature matrix …")
        t0 = time.time()
        X_train, y_train, feature_names, _ = _build_single_window(
            df_train, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
            industry_medians=ind_medians,
            industry_adj_target=ind_adj,
        )
        logger.info("  Training matrix: %s rows × %s features  (%.1fs)",
                    X_train.shape[0], X_train.shape[1], time.time() - t0)

        if X_train.shape[0] < cfg["min_train_obs"]:
            logger.warning("  Skipping %d: too few valid training observations", test_year)
            continue

        # ── PCA pre-projection (final model phase) ────────────────────────
        # Re-fit PCA on the full training window (includes val year); the
        # tuning-phase PCA was fitted on a shorter window and is discarded.
        # All subsequent matrices (val, test) are projected with this projector.
        pca_scaler_final, pca_final = fit_pca_projector(X_train, pca_k)
        X_train_pc = apply_pca_projector(pca_scaler_final, pca_final, X_train)
        n_pcs_final = X_train_pc.shape[1]
        logger.info(
            "  PCA (final):  %d raw features → %d PCs  (%.1f%% variance)",
            X_train.shape[1], n_pcs_final,
            100 * (pca_final.explained_variance_ratio_.sum() if pca_final is not None else 1.0),
        )
        # PC feature names for diagnostics output
        pc_feature_names = [f"PC_{i+1}" for i in range(n_pcs_final)]

        # ── Build validation feature matrix (for variable importance only) ─
        # Use the full rolling val window (val_start_year … val_end_year),
        # re-encoded with final training-window statistics for feature alignment.
        mask_val_vi = (df["year"] >= val_start_year) & (df["year"] <= val_end_year)
        df_val_vi = pd.DataFrame(df[mask_val_vi])
        X_val_raw, y_val, _, _ = _build_single_window(
            df_val_vi, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
            industry_medians=ind_medians,
            industry_adj_target=ind_adj,
        )
        X_val_pc = apply_pca_projector(pca_scaler_final, pca_final, X_val_raw)

        # ── Fit final models on full training window (hyperparams are fixed) ─
        year_pbar.set_postfix(year=test_year, stage="fitting")
        logger.info("  Fitting final OLS model  (α=%.5f, l1_ratio=%.2f) …",
                    best_alpha_ols, best_l1_ols)
        t0 = time.time()
        model_ols, scaler_ols = fit_ols_model(
            X_train_pc, y_train.astype(np.float64),
            best_alpha_ols, best_l1_ols, cfg,
        )
        logger.info("  OLS fit done: %d non-zero PCs  (%.1fs)",
                    int(np.sum(model_ols.coef_ != 0)), time.time() - t0)

        # Fix 6: replace FISTA final fit with SGDRegressor (~30–60s vs 20–40 min)
        logger.info("  Fitting final Huber model [SGD]  (α=%.5f, l1_ratio=%.2f, ε=%.2f) …",
                    best_alpha_hub, best_l1_hub, best_epsilon_hub)
        t0 = time.time()
        model_huber, scaler_huber = fit_huber_model_sgd(
            X_train_pc, y_train.astype(np.float64),
            best_alpha_hub, best_l1_hub, best_epsilon_hub, cfg,
        )
        n_nz_hub = int(np.sum(np.abs(model_huber.coef_) > 1e-10))
        logger.info("  Huber fit done: %d non-zero coefs  (%.1fs)",
                    n_nz_hub, time.time() - t0)

        # ── Build test feature matrix for forecast year ────────────────────
        mask_test = (df["year"] == test_year)
        df_test   = df[mask_test].copy()
        logger.info("  Building test feature matrix (year=%d) …", test_year)
        X_test_raw, y_test, _, valid_test_idx = _build_single_window(
            df_test, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
            industry_adj_target=ind_adj,
        )
        X_test_pc = apply_pca_projector(pca_scaler_final, pca_final, X_test_raw)
        logger.info("  Test matrix: %s rows × %s PCs (from %s raw features)",
                    X_test_pc.shape[0], X_test_pc.shape[1], X_test_raw.shape[1])

        # ── Predict expected returns ───────────────────────────────────────
        X_test_ols_s = scaler_ols.transform(X_test_pc)
        y_pred_ols   = model_ols.predict(X_test_ols_s)

        X_test_hub_s = scaler_huber.transform(X_test_pc)
        y_pred_huber = model_huber.predict(X_test_hub_s)

        # ── OOS R² ────────────────────────────────────────────────────────
        r2_ols   = compute_gkx_r2(y_test, y_pred_ols)
        r2_huber = compute_gkx_r2(y_test, y_pred_huber)
        logger.info("  OOS R²:  OLS=%.4f   Huber=%.4f", r2_ols, r2_huber)

        # ── Attach forecast metadata back to test rows ─────────────────────
        df_test_valid = df_test.loc[valid_test_idx][["permno", "year", "month"]].copy()
        # Save raw forecasts before winsorisation so any threshold can be applied downstream.
        df_test_valid["expected_ret_raw_ols"]   = y_pred_ols.astype(np.float32)
        df_test_valid["expected_ret_raw_huber"] = y_pred_huber.astype(np.float32)
        df_test_valid["expected_ret_ols"]   = y_pred_ols.astype(np.float32)
        df_test_valid["expected_ret_huber"] = y_pred_huber.astype(np.float32)

        # Winsorise cross-sectional forecasts at forecast_winsor_pct per tail per month.
        winsor_pct = cfg.get("forecast_winsor_pct", 0.01)
        df_test_valid["expected_ret_ols"]   = _winsorize_forecasts(
            df_test_valid, "expected_ret_ols",   winsor_pct,
        ).astype(np.float32)
        df_test_valid["expected_ret_huber"] = _winsorize_forecasts(
            df_test_valid, "expected_ret_huber", winsor_pct,
        ).astype(np.float32)

        all_expected_ols.append(
            df_test_valid[["permno", "year", "month", "expected_ret_ols"]]
            .rename(columns={"expected_ret_ols": "expected_ret"})
        )
        all_expected_huber.append(
            df_test_valid[["permno", "year", "month", "expected_ret_huber"]]
            .rename(columns={"expected_ret_huber": "expected_ret"})
        )
        all_expected_raw_ols.append(
            df_test_valid[["permno", "year", "month", "expected_ret_raw_ols"]]
            .rename(columns={"expected_ret_raw_ols": "expected_ret"})
        )
        all_expected_raw_huber.append(
            df_test_valid[["permno", "year", "month", "expected_ret_raw_huber"]]
            .rename(columns={"expected_ret_raw_huber": "expected_ret"})
        )

        # ── Monthly OOS R² breakdown ───────────────────────────────────────
        df_test_valid["y_true"]       = y_test.astype(np.float32)
        df_test_valid["y_pred_ols"]   = y_pred_ols.astype(np.float32)
        df_test_valid["y_pred_huber"] = y_pred_huber.astype(np.float32)
        monthly_r2_rows = []
        for (yr, mo), grp in df_test_valid.groupby(["year", "month"]):
            if len(grp) < 5:
                continue
            r2_mo_ols   = compute_gkx_r2(grp["y_true"].values, grp["y_pred_ols"].values)
            r2_mo_huber = compute_gkx_r2(grp["y_true"].values, grp["y_pred_huber"].values)
            monthly_r2_rows.append({
                "year": yr, "month": mo,
                "r2_ols": r2_mo_ols, "r2_huber": r2_mo_huber,
            })
        all_oos_r2.extend(monthly_r2_rows)

        # ── Diagnostic 1: Forecast stability (rank-IC / cross-sectional std) ─
        # rank_IC: Spearman correlation of this year's forecasts with last year's
        # on the common permno set.  Low rank_IC → high turnover.
        # Aggregate to one forecast per permno (mean across months) so that
        # the rank-IC comparison between years has unique indices on both sides.
        curr_ols_s   = df_test_valid.groupby("permno")["expected_ret_ols"].mean()
        curr_huber_s = df_test_valid.groupby("permno")["expected_ret_huber"].mean()
        rank_ic_ols   = np.nan
        rank_ic_huber = np.nan
        if prev_forecasts_ols is not None:
            common = curr_ols_s.index.intersection(prev_forecasts_ols.index)
            if len(common) >= 20:
                rank_ic_ols, _ = spearmanr(
                    curr_ols_s.loc[common].values,
                    prev_forecasts_ols.loc[common].values,
                )
        if prev_forecasts_huber is not None:
            common = curr_huber_s.index.intersection(prev_forecasts_huber.index)
            if len(common) >= 20:
                rank_ic_huber, _ = spearmanr(
                    curr_huber_s.loc[common].values,
                    prev_forecasts_huber.loc[common].values,
                )
        forecast_std_ols   = float(curr_ols_s.std())
        forecast_std_huber = float(curr_huber_s.std())
        logger.info(
            "  Forecast stability:  rank_IC OLS=%.3f  Huber=%.3f  |  "
            "std OLS=%.4f  Huber=%.4f",
            rank_ic_ols if not np.isnan(rank_ic_ols) else -999,
            rank_ic_huber if not np.isnan(rank_ic_huber) else -999,
            forecast_std_ols, forecast_std_huber,
        )
        all_forecast_stability.append({
            "reest_year":        test_year,
            "rank_ic_ols":       float(rank_ic_ols)   if not np.isnan(rank_ic_ols)   else None,
            "rank_ic_huber":     float(rank_ic_huber) if not np.isnan(rank_ic_huber) else None,
            "forecast_std_ols":  forecast_std_ols,
            "forecast_std_huber": forecast_std_huber,
            "n_stocks":          len(curr_ols_s),
        })
        prev_forecasts_ols   = curr_ols_s
        prev_forecasts_huber = curr_huber_s

        # ── Feature selection diagnostics ──────────────────────────────────
        ols_coefs  = model_ols.coef_
        hub_coefs  = model_huber.coef_  # SGDRegressor.coef_ excludes intercept
        ols_nz_idx = np.where(np.abs(ols_coefs) > 1e-10)[0]
        hub_nz_idx = np.where(np.abs(hub_coefs) > 1e-10)[0]

        feat_rows_ols = [
            {"reest_year": test_year, "loss": "ols",
             "feature": pc_feature_names[i], "coef": float(ols_coefs[i])}
            for i in ols_nz_idx if i < len(pc_feature_names)
        ]
        feat_rows_hub = [
            {"reest_year": test_year, "loss": "huber",
             "feature": pc_feature_names[i], "coef": float(hub_coefs[i])}
            for i in hub_nz_idx if i < len(pc_feature_names)
        ]
        feat_df = pd.DataFrame(feat_rows_ols + feat_rows_hub)
        all_feat_selection.append(feat_df)

        # ── Variable importance (validation window) ────────────────────────
        logger.info("  Computing variable importance on validation window …")
        t0 = time.time()
        vi_df = compute_variable_importance(
            model_ols, scaler_ols, model_huber, scaler_huber, X_val_pc, y_val, pc_feature_names,
            max_features=cfg.get("max_vi_features", 50),
        )
        vi_df.insert(0, "reest_year", test_year)
        all_var_importance.append(vi_df)
        logger.info("  Variable importance done  (%.1fs)", time.time() - t0)

        # ── Metadata ──────────────────────────────────────────────────────
        # val_r2_ols / val_r2_huber: validation R² at the chosen hyperparameters
        # from the rolling val window (val_start_year … val_end_year).
        _ols_match = val_surface_ols.loc[
            (val_surface_ols["alpha"] == best_alpha_ols) &
            (val_surface_ols["l1_ratio"] == best_l1_ols), "val_r2"
        ]
        rolling_val_r2_ols = float(_ols_match.iloc[0]) if len(_ols_match) > 0 else np.nan

        # Huber now reuses OLS hyperparameters; its val_r2 equals the OLS val surface entry
        rolling_val_r2_hub = rolling_val_r2_ols

        all_metadata.append({
            "reest_year":        test_year,
            "train_start":       cfg["train_start_year"],
            "train_end":         train_end_year,
            "val_start":         val_start_year,
            "val_end":           val_end_year,
            "n_train_obs":       len(df_train),
            "n_train_valid_obs": int(X_train.shape[0]),
            "n_features":        int(X_train.shape[1]),
            "n_active_chars":    len(active_chars),
            "n_industry_codes":  len(industry_codes),
            "best_alpha_ols":    best_alpha_ols,
            "best_l1_ols":       best_l1_ols,
            "best_alpha_huber":  best_alpha_hub,
            "best_l1_huber":     best_l1_hub,
            "best_epsilon_huber": best_epsilon_hub,
            "n_nonzero_ols":     int(np.sum(ols_coefs != 0)),
            "n_nonzero_huber":   n_nz_hub,
            "rolling_val_r2_ols":   rolling_val_r2_ols,
            "rolling_val_r2_hub":   rolling_val_r2_hub,
            "test_r2_ols":       r2_ols,
            "test_r2_huber":     r2_huber,
        })

        elapsed = time.time() - loop_start
        year_pbar.set_postfix(
            year=test_year,
            r2_ols=f"{r2_ols:.3f}",
            r2_hub=f"{r2_huber:.3f}",
            min=f"{elapsed/60:.1f}",
        )
        logger.info("  Re-estimation %d complete  (%.1f min elapsed)",
                    test_year, elapsed / 60)

    # ── Assemble final DataFrames ─────────────────────────────────────────
    logger.info("")
    logger.info("Assembling output DataFrames …")

    results = {}

    _empty_ret = pd.DataFrame(columns=["permno","year","month","expected_ret"])
    if all_expected_ols:
        results["expected_returns_ols"]       = pd.concat(all_expected_ols,       ignore_index=True)
        results["expected_returns_huber"]     = pd.concat(all_expected_huber,     ignore_index=True)
        results["expected_returns_raw_ols"]   = pd.concat(all_expected_raw_ols,   ignore_index=True)
        results["expected_returns_raw_huber"] = pd.concat(all_expected_raw_huber, ignore_index=True)
    else:
        results["expected_returns_ols"]       = _empty_ret.copy()
        results["expected_returns_huber"]     = _empty_ret.copy()
        results["expected_returns_raw_ols"]   = _empty_ret.copy()
        results["expected_returns_raw_huber"] = _empty_ret.copy()

    results["oos_r2"]        = pd.DataFrame(all_oos_r2)
    results["metadata"]      = pd.DataFrame(all_metadata)
    results["feat_selection"] = (
        pd.concat(all_feat_selection, ignore_index=True)
        if all_feat_selection else pd.DataFrame()
    )
    results["var_importance"] = (
        pd.concat(all_var_importance, ignore_index=True)
        if all_var_importance else pd.DataFrame()
    )
    results["val_surface"] = (
        pd.concat(all_val_surfaces, ignore_index=True)
        if all_val_surfaces else pd.DataFrame()
    )
    if all_metadata:
        _meta = results["metadata"]
        results["tuned_params"] = _meta[[
            "reest_year",
            "best_alpha_ols", "best_l1_ols",
            "best_alpha_huber", "best_l1_huber", "best_epsilon_huber",
        ]].copy()
    # Diagnostic 1: forecast stability (rank-IC, cross-sectional std)
    results["forecast_stability"] = pd.DataFrame(all_forecast_stability)
    if all_forecast_stability:
        mean_ic_ols   = np.nanmean([r["rank_ic_ols"]   for r in all_forecast_stability if r["rank_ic_ols"]   is not None])
        mean_ic_huber = np.nanmean([r["rank_ic_huber"] for r in all_forecast_stability if r["rank_ic_huber"] is not None])
        logger.info("Mean rank-IC:  OLS=%.3f  Huber=%.3f  (higher = more stable forecasts)",
                    mean_ic_ols, mean_ic_huber)

    logger.info("Total elapsed: %.1f min", (time.time() - total_start) / 60)
    logger.info("Run complete.")

    return results


# ============================================================
# Output Saving
# ============================================================

def save_outputs(results: Dict, config: Dict) -> None:
    """
    Persist all result DataFrames to disk.

    Parquet files: expected returns, OOS R², metadata, feature selection,
                   variable importance, validation surface.
    CSV file:      human-readable diagnostics summary.
    """
    logger = logging.getLogger("elastic_net")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_parquet(df: pd.DataFrame, fname: str) -> None:
        path = out_dir / fname
        df.to_parquet(path, index=False)
        logger.info("  Saved %s  (%d rows)", fname, len(df))

    _save_parquet(results["expected_returns_ols"],   "expected_returns_enet_ols.parquet")
    _save_parquet(results["expected_returns_huber"], "expected_returns_enet_huber.parquet")
    _save_parquet(results["oos_r2"],                 "oos_r2_enet.parquet")
    _save_parquet(results["metadata"],               "model_metadata_enet.parquet")
    _save_parquet(results["feat_selection"],         "feature_selection_enet.parquet")
    _save_parquet(results["var_importance"],         "variable_importance_enet.parquet")
    _save_parquet(results["val_surface"],            "validation_surface_enet.parquet")
    if "forecast_stability" in results and not results["forecast_stability"].empty:
        _save_parquet(results["forecast_stability"], "forecast_stability_enet.parquet")

    # ── Human-readable diagnostics CSV ────────────────────────────────────
    meta = results["metadata"]
    if not meta.empty:
        diag_cols = [
            "reest_year", "train_start", "train_end", "val_start", "val_end",
            "best_alpha_ols", "best_l1_ols", "n_nonzero_ols",
            "rolling_val_r2_ols", "test_r2_ols",
            "best_alpha_huber", "best_l1_huber", "n_nonzero_huber",
            "rolling_val_r2_hub", "test_r2_huber",
            "n_features", "n_train_valid_obs",
        ]
        diag_cols_present = [c for c in diag_cols if c in meta.columns]
        diag_df = meta[diag_cols_present].copy()
        diag_path = out_dir / "diagnostics_summary_enet.csv"
        diag_df.to_csv(diag_path, index=False, float_format="%.6f")
        logger.info("  Saved diagnostics_summary_enet.csv")

    logger.info("All outputs saved to %s", out_dir)


# ============================================================
# Portfolio Adapter — run_backtest()-compatible interface
# ============================================================

# Module-level cache so the parquet is read from disk only once per session,
# not on every call to the adapter function inside the backtest loop.
_ENET_CACHE: Dict[str, Optional[pd.DataFrame]] = {
    "ols":   None,
    "huber": None,
}


def _load_enet_forecasts(loss: str, output_dir: str) -> pd.DataFrame:
    """
    Load pre-computed elastic net expected returns from disk, cached in memory.

    Parameters
    ----------
    loss       : 'ols' or 'huber'
    output_dir : directory where save_outputs() wrote the parquet files

    Returns
    -------
    DataFrame with columns [permno, year, month, expected_ret]
    """
    if _ENET_CACHE[loss] is not None:
        assert _ENET_CACHE[loss] is not None  # narrow Optional for type checker
        return _ENET_CACHE[loss]

    fname = f"expected_returns_enet_{loss}.parquet"
    path  = Path(output_dir) / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Elastic net forecasts not found at {path}. "
            f"Run level3_elastic_net.py (or run_elastic_net()) first to generate them."
        )
    df = pd.read_parquet(path)
    df["permno"] = df["permno"].astype(int)
    df["year"]   = df["year"].astype(int)
    df["month"]  = df["month"].astype(int)
    _ENET_CACHE[loss] = df
    return df


def make_enet_expected_returns_fn(
    loss: str = "ols",
    output_dir: str = DEFAULT_CONFIG["output_dir"],
) -> Callable:
    """
    Return an expected_returns_fn adapter for use with run_backtest().

    The adapter reads from the pre-computed parquet saved by run_elastic_net()
    / save_outputs(), so elastic net expected returns slot into the same
    portfolio optimizer as all other model levels without recomputing.

    Parameters
    ----------
    loss       : 'ols' or 'huber' — which loss variant to use
    output_dir : directory containing the saved parquet files

    Returns
    -------
    fn : callable with signature fn(master_df, ret_matrix, year, month) -> pd.Series
         Returns expected total return indexed by permno, NaN for stocks not
         in the elastic net universe for that month.

    Usage
    -----
    from models.level3_elastic_net import make_enet_expected_returns_fn

    results = run_backtest(
        master,
        start_year=2005, start_month=1,
        end_year=2024,   end_month=12,
        expected_returns_fn=make_enet_expected_returns_fn(loss='ols'),
        ...
    )
    """
    if loss not in ("ols", "huber"):
        raise ValueError(f"loss must be 'ols' or 'huber', got {loss!r}")

    def enet_expected_returns(_master_df, ret_matrix, year, month):  # noqa: ARG001
        """
        Adapter: look up pre-computed elastic net expected returns for (year, month).

        Signature matches the expected_returns_fn hook in run_backtest():
            fn(master_df, ret_matrix, year, month) -> pd.Series

        Parameters
        ----------
        _master_df : pd.DataFrame — not used (forecasts are pre-computed)
        ret_matrix : pd.DataFrame, shape (T, N) — used only to obtain the
                     current universe (column permnos)
        year, month : int — portfolio formation month

        Returns
        -------
        mu : pd.Series indexed by permno
             Expected return for each stock in ret_matrix.columns.
             NaN for stocks not covered by the elastic net forecast.
        """
        forecasts = _load_enet_forecasts(loss, output_dir)
        month_slice = forecasts[
            (forecasts["year"] == year) & (forecasts["month"] == month)
        ]
        # Deduplicate: if the panel had duplicate (permno, year, month) rows,
        # the parquet may carry them through. Keep the first occurrence.
        month_slice = pd.DataFrame(month_slice).drop_duplicates(subset=["permno"])
        month_fc = month_slice.set_index("permno")["expected_ret"]

        # Reindex to the current universe; stocks not in the forecast get NaN
        # (downstream optimizer drops NaN-mu stocks, same as FF5 behaviour).
        mu = month_fc.reindex(ret_matrix.columns)
        n_valid = mu.notna().sum()
        n_total = len(mu)
        logging.getLogger("elastic_net").debug(
            "  ENet-%s [%d-%02d]: %d / %d stocks with forecasts",
            loss.upper(), year, month, n_valid, n_total,
        )
        return mu

    return enet_expected_returns


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Level 3: Elastic Net — empirical asset pricing pipeline",
    )
    parser.add_argument(
        "--mode", choices=["full", "test"], default="full",
        help="'test' runs on a small subsample for debugging (default: full)",
    )
    parser.add_argument(
        "--test-stocks", type=int, default=50,
        help="Number of stocks in test mode (default: 50)",
    )
    parser.add_argument(
        "--test-years", type=int, default=2,
        help="Number of re-estimation years in test mode (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data_clean/elastic_net",
        help="Directory for output files",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel jobs for CV grid search (-1 = all CPUs)",
    )
    args = parser.parse_args()

    config_override = {
        "run_mode":      args.mode,
        "test_n_stocks": args.test_stocks,
        "test_n_years":  args.test_years,
        "output_dir":    args.output_dir,
        "n_jobs":        args.jobs,
    }

    # ── Load data ─────────────────────────────────────────────────────────
    cfg_full = DEFAULT_CONFIG.copy()
    cfg_full.update(config_override)

    logger = setup_logging(cfg_full)
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Run mode: %s", args.mode.upper())

    df, macro_df = load_data(cfg_full)

    # ── Run pipeline ──────────────────────────────────────────────────────
    results = run_elastic_net(df, macro_df, config=config_override)

    # ── Save outputs ──────────────────────────────────────────────────────
    save_outputs(results, cfg_full)

    # ── Print summary ─────────────────────────────────────────────────────
    meta = results["metadata"]
    if not meta.empty:
        print("\n" + "=" * 65)
        print("ELASTIC NET — DIAGNOSTICS SUMMARY")
        print("=" * 65)
        for _, row in meta.iterrows():
            print(
                f"  {int(row['reest_year'])}:  "
                f"OLS  α={row['best_alpha_ols']:.4f}  ρ={row['best_l1_ols']:.2f}  "
                f"nz={int(row['n_nonzero_ols'])}  "
                f"valR²={row['rolling_val_r2_ols']:.4f}  testR²={row['test_r2_ols']:.4f}   |   "
                f"Huber α={row['best_alpha_huber']:.4f}  ρ={row['best_l1_huber']:.2f}  "
                f"nz={int(row['n_nonzero_huber'])}  "
                f"valR²={row['rolling_val_r2_hub']:.4f}  testR²={row['test_r2_huber']:.4f}"
            )
        oos = results["oos_r2"]
        if not oos.empty:
            print(f"\n  Mean OOS R²  —  OLS: {oos['r2_ols'].mean():.4f}   "
                  f"Huber: {oos['r2_huber'].mean():.4f}")
        print("=" * 65)

    logger.info("Done.")
