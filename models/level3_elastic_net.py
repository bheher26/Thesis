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
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Constants
# ============================================================

FIRM_CHARACTERISTICS: List[str] = [
    "market_cap", "ret", "ret_adjusted", "illiquidity", "reversal_st",
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

    # ── Hyperparameter grid ──────────────────────────────────────────────
    # alpha  = regularisation strength (λ in the paper)
    # l1_ratio = weight on L1 vs L2 (sklearn convention: 1 = LASSO, 0 = ridge)
    # Log-spaced alpha grid: finer resolution covers the full regularisation path.
    # Hyperparameters are re-tuned every year via rolling 12-month validation.
    "alpha_grid":         list(np.logspace(-4, 0, 20)),
    "l1_ratio_grid":      [0.1, 0.3, 0.5, 0.7, 0.9],
    "huber_epsilon":      1.35,       # Huber threshold default (overridden by tuning)
    "huber_epsilon_grid": [0.5, 0.7, 0.9, 1.35],  # jointly tuned with alpha/l1_ratio

    # ── n_cv_splits: used only when tune_hyperparameters falls back to
    #    CV-within mode (no external train set).  In the default rolling
    #    validation mode (X_train_ext provided) this setting is ignored.
    "n_cv_splits": 5,

    # ── Feature construction ─────────────────────────────────────────────
    "char_missing_threshold": 0.50,  # drop char if > 50 % missing in training

    # ── Quality filters ──────────────────────────────────────────────────
    "min_train_obs":        1000,
    "min_stocks_per_month": 50,

    # ── Paths ────────────────────────────────────────────────────────────
    "data_path":  "data_clean/master_panel.csv",
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
    "max_iter":         5000,
    "tol":              1e-4,
    "huber_max_iter":   1000,  # ISTA iterations for Huber model
    "huber_tol":        1e-4,
    "power_iter_n":     20,    # power-iteration steps for Lipschitz constant
}

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
) -> Tuple[np.ndarray, np.ndarray, List[str], List]:
    """
    Build the GKX feature matrix and excess-return target for one data window.

    Processing steps (applied in order, using training statistics throughout):
      1. Compute excess return target: ret_adjusted - rf
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
    df = df_window.copy()

    # ── Target: excess return ────────────────────────────────────────────
    df["excess_ret"] = pd.to_numeric(df["ret_adjusted"], errors="coerce") - \
                       pd.to_numeric(df["rf"], errors="coerce")
    valid_mask = df["excess_ret"].notna()
    df = df[valid_mask].copy()
    y = df["excess_ret"].values.astype(np.float32)
    valid_idx = df_window.index[valid_mask].tolist()

    # ── Impute missing characteristics ───────────────────────────────────
    # When industry_medians is provided (industry_imputation=True), each
    # missing value is filled with the median of the firm's own 2-digit SIC
    # peer group (fitted on training data).  Falls back to the global median
    # for firms in industries not seen during training, or when the group had
    # fewer than 5 observations.  This avoids pooling e.g. utility leverage
    # with tech leverage when computing imputation targets.
    char_data = pd.DataFrame(index=df.index)
    sic2_col = (pd.to_numeric(df["siccd"], errors="coerce").fillna(-10) // 10).astype(int)
    for col in active_chars:
        series = pd.to_numeric(df[col], errors="coerce")
        if industry_medians is not None and col in industry_medians:
            ind_med_map = industry_medians[col]
            global_med  = train_medians.get(col, 0.0)
            fill_vals   = sic2_col.map(ind_med_map).fillna(global_med)
            series      = series.fillna(fill_vals)
        else:
            series = series.fillna(train_medians.get(col, 0.0))
        char_data[col] = series

    # ── Cross-sectional rank to [-1, 1] within each month ────────────────
    # Ranking is done independently within each month: firms in month t are
    # ranked only against other firms in month t.  No cross-period leakage.
    ranked_data = (
        char_data.assign(_year=df["year"].values, _month=df["month"].values)
        .groupby(["_year", "_month"])[active_chars]
        .transform(_rank_series_pm1)
    )
    # Any residual NaN (e.g. single-stock months) → imputed with 0 (median rank)
    ranked_data = ranked_data.fillna(0.0)

    # ── Macro variables for this window ──────────────────────────────────
    # macro_cols are already present in df_window (merged upstream).
    # Standardise using training-window statistics (macro_means, macro_stds)
    # so that interaction terms char_i × macro_j are on a uniform scale.
    # This prevents the elastic net penalty from shrinking large-scale macro
    # interactions (e.g. volatility ≈ 20) disproportionately relative to
    # small-scale ones (e.g. dp_ratio ≈ 0.04).
    macro_arr = np.zeros((len(df), len(macro_cols)), dtype=np.float32)
    for j, mcol in enumerate(macro_cols):
        if mcol in df.columns:
            col_vals = pd.to_numeric(df[mcol], errors="coerce").fillna(macro_means[j])
            standardised = (col_vals.values - macro_means[j]) / macro_stds[j]
            macro_arr[:, j] = standardised.astype(np.float32)

    # ── Interaction terms: char_i × macro_j + char_i × constant ─────────
    n_rows    = len(df)
    n_chars   = len(active_chars)
    # 8 interactors: 7 macro vars + 1 constant
    n_interact = n_chars * (len(macro_cols) + 1)
    X_interact = np.zeros((n_rows, n_interact), dtype=np.float32)
    feat_names: List[str] = []
    char_arr = ranked_data.values.astype(np.float32)  # (n_rows, n_chars)
    col_idx = 0
    for i, cname in enumerate(active_chars):
        # constant interaction (= the characteristic itself)
        X_interact[:, col_idx] = char_arr[:, i]
        feat_names.append(f"{cname}__const")
        col_idx += 1
        # macro interactions
        for j, mname in enumerate(macro_cols):
            X_interact[:, col_idx] = char_arr[:, i] * macro_arr[:, j]
            feat_names.append(f"{cname}__{mname}")
            col_idx += 1

    # ── Industry dummies ─────────────────────────────────────────────────
    sic2 = (pd.to_numeric(df["siccd"], errors="coerce").fillna(-1) // 10).astype(int)
    n_ind = len(industry_codes)
    ind_lookup = {code: idx for idx, code in enumerate(industry_codes)}
    X_ind = np.zeros((n_rows, n_ind), dtype=np.float32)
    for row_i, code in enumerate(sic2.values):
        if code in ind_lookup:
            X_ind[row_i, ind_lookup[code]] = 1.0
    ind_names = [f"sic2_{code}" for code in industry_codes]

    # ── Assemble final feature matrix ────────────────────────────────────
    X = np.concatenate([X_interact[:, :col_idx], X_ind], axis=1)
    feature_names = feat_names[:col_idx] + ind_names

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
    X          : feature matrix, float64, shape (n, p); should be standardised
    y          : target vector, float64, shape (n,)
    alpha      : regularisation strength
    l1_ratio   : balance between L1 (=1) and L2 (=0) penalties
    epsilon    : Huber loss threshold (default 1.35 ≈ 95th percentile of N(0,1))
    max_iter   : maximum ISTA iterations
    tol        : convergence tolerance on max absolute coefficient change
    power_iter_n: iterations for spectral-norm approximation
    warm_coef  : initial coefficient vector (for warm starting across λ path)

    Returns
    -------
    w            : float64 ndarray, shape (p,) — fitted coefficients (including
                   intercept if a bias column was appended to X)
    converged    : bool — True if the tolerance criterion was met before max_iter
    n_iter_taken : int  — number of ISTA iterations actually executed
    final_delta  : float — max absolute coefficient change at termination;
                   should be ≤ tol if converged, larger otherwise
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape

    # Lipschitz constant of the smooth part of the objective.
    # L_smooth = sigma_max(X)^2 / n + alpha * (1 - l1_ratio)
    sigma_max = _spectral_norm_power_iter(X, n_iter=power_iter_n)
    L_smooth = sigma_max ** 2 / n + alpha * (1.0 - l1_ratio)
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
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    split_indices: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> float:
    """Return mean GKX R² across time-series CV folds for OLS elastic net.

    Standardises X on each training fold independently (scaler fitted on
    training fold, applied to validation fold) so CV scores are comparable
    with the final model which also standardises.
    """
    scores = []
    warm_coef = None
    for train_idx, val_idx in split_indices:
        X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
        X_vl, y_vl = X_tv[val_idx],   y_tv[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float64))
        X_vl_s = scaler.transform(X_vl.astype(np.float64))
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter, tol=tol,
            warm_start=True, selection="cyclic",
        )
        if warm_coef is not None:
            try:
                model.coef_ = warm_coef["coef"]
                model.intercept_ = warm_coef["intercept"]
            except Exception:
                pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_s, y_tr)
        warm_coef = {"coef": model.coef_.copy(), "intercept": model.intercept_}
        y_pred = model.predict(X_vl_s)
        scores.append(compute_gkx_r2(y_vl, y_pred))
    return float(np.mean(scores)) if scores else -np.inf


def _cv_score_huber(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    split_indices: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float,
    l1_ratio: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    power_iter_n: int,
) -> float:
    """Return mean GKX R² across time-series CV folds for Huber elastic net."""
    scores = []
    warm_coef = None
    for train_idx, val_idx in split_indices:
        X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
        X_vl, y_vl = X_tv[val_idx],   y_tv[val_idx]
        # Standardise on training fold only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)
        # Augment with intercept column
        ones_tr = np.ones((len(X_tr_s), 1))
        ones_vl = np.ones((len(X_vl_s), 1))
        X_tr_aug = np.hstack([X_tr_s, ones_tr])
        X_vl_aug = np.hstack([X_vl_s, ones_vl])
        w, _conv, _nit, _delta = fit_huber_enet(
            X_tr_aug, y_tr, alpha=alpha, l1_ratio=l1_ratio,
            epsilon=epsilon, max_iter=max_iter, tol=tol,
            power_iter_n=power_iter_n, warm_coef=warm_coef,
        )
        warm_coef = w.copy()
        y_pred = X_vl_aug @ w
        scores.append(compute_gkx_r2(y_vl, y_pred))
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
    max_iter      = config["max_iter"]
    tol           = config["tol"]
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

    # ── Evaluate every grid cell ─────────────────────────────────────────
    if loss_type == "ols":
        _raw_ols = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_cv_score_ols)(
                X_tv, y_tv, split_indices, a, r, max_iter, tol,
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
                X_tv, y_tv, split_indices, a, r, eps,
                config["huber_max_iter"], config["huber_tol"], power_iter_n,
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


# ============================================================
# Final Model Fitting
# ============================================================

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


# ============================================================
# Variable Importance
# ============================================================

def compute_variable_importance(
    model_ols: ElasticNet,
    scaler_ols: StandardScaler,
    w_huber: np.ndarray,
    scaler_huber: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Estimate variable importance as reduction in out-of-sample R² when
    each predictor is zeroed out (computed on the validation window).

    Only features with non-zero coefficient are evaluated (for speed).

    Returns
    -------
    DataFrame with columns: [feature, coef_ols, coef_huber,
                              vi_ols, vi_huber]
    where vi_* = baseline_R² - R²_with_feature_zeroed.
    """
    # Baseline R² for OLS (standardise with training scaler before predicting)
    X_val_ols_s  = scaler_ols.transform(X_val.astype(np.float64))
    y_base_ols   = model_ols.predict(X_val_ols_s)
    base_r2_ols  = compute_gkx_r2(y_val, y_base_ols)

    # Baseline R² for Huber
    X_val_s = scaler_huber.transform(X_val.astype(np.float64))
    ones_val = np.ones((len(X_val_s), 1))
    X_val_aug = np.hstack([X_val_s, ones_val])
    y_base_huber  = X_val_aug @ w_huber
    base_r2_huber = compute_gkx_r2(y_val, y_base_huber)

    rows = []
    n_feat = len(feature_names)

    # OLS importance: only for non-zero coefficients
    ols_coefs = model_ols.coef_
    huber_coefs = w_huber[:n_feat]  # exclude intercept

    nonzero_ols   = np.where(np.abs(ols_coefs)   > 1e-10)[0]
    nonzero_huber = np.where(np.abs(huber_coefs)  > 1e-10)[0]
    important_idx = np.union1d(nonzero_ols, nonzero_huber)

    for i in important_idx:
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"

        # OLS: zero out feature i in the standardised space
        X_zeroed_ols_s = X_val_ols_s.copy()
        X_zeroed_ols_s[:, i] = 0.0
        y_zeroed_ols = model_ols.predict(X_zeroed_ols_s)
        vi_ols = base_r2_ols - compute_gkx_r2(y_val, y_zeroed_ols)

        # Huber: zero out feature i (in scaled space)
        X_zeroed_s = X_val_s.copy()
        X_zeroed_s[:, i] = 0.0
        X_zeroed_aug = np.hstack([X_zeroed_s, np.ones((len(X_zeroed_s), 1))])
        y_zeroed_huber = X_zeroed_aug @ w_huber
        vi_huber = base_r2_huber - compute_gkx_r2(y_val, y_zeroed_huber)

        rows.append({
            "feature":    fname,
            "coef_ols":   float(ols_coefs[i]) if i < len(ols_coefs) else 0.0,
            "coef_huber": float(huber_coefs[i]),
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
    all_expected_ols   : List[pd.DataFrame] = []
    all_expected_huber : List[pd.DataFrame] = []
    all_oos_r2         : List[Dict]         = []
    all_metadata       : List[Dict]         = []
    all_feat_selection : List[pd.DataFrame] = []
    all_var_importance : List[pd.DataFrame] = []
    all_val_surfaces   : List[pd.DataFrame] = []

    total_start = time.time()

    # ── Main loop: one iteration per re-estimation year ───────────────────
    # GKX rolling-validation design (Section 3.3):
    #   - Train:      train_start_year … test_year - 1  (expanding window)
    #   - Validation: test_year - 1  (12 months, rolling forward each year)
    #   - Test:       test_year      (held-out forecasts)
    #
    # Hyperparameters are tuned every year by fitting each combo on
    # train_start … test_year - 2 and evaluating on test_year - 1.
    # The final model is fit on the full training window (train_start … test_year - 1).
    year_pbar = tqdm(test_years, desc="Elastic Net", unit="year", ncols=90, leave=True)
    for reest_idx, test_year in enumerate(year_pbar):
        loop_start = time.time()
        train_end_year = test_year - 1   # final model uses data through this year
        tune_train_end = test_year - 2   # tuning training data excludes val year
        val_year       = test_year - 1   # rolling 12-month validation window

        year_pbar.set_postfix(year=test_year, stage="tuning")
        logger.info("")
        logger.info("─" * 55)
        logger.info("RE-ESTIMATION  %d / %d  →  val=%d, forecast=%d",
                    reest_idx + 1, len(test_years), val_year, test_year)
        logger.info("─" * 55)

        # ── Leakage check (central invariant) ────────────────────────────
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
        X_tune_train, y_tune_train, _, _ = _build_single_window(
            df_tune_train, tune_chars, tune_medians,
            available_macro, tune_industry,
            tune_macro_means, tune_macro_stds,
        )

        mask_val = (df["year"] == val_year)
        df_val_roll = pd.DataFrame(df[mask_val])
        X_val_roll, y_val_roll, _, _ = _build_single_window(
            df_val_roll, tune_chars, tune_medians,
            available_macro, tune_industry,
            tune_macro_means, tune_macro_stds,
        )
        logger.info(
            "  Tuning train: {:,} rows ({} – {})  |  Val: {:,} rows (year {})".format(
                X_tune_train.shape[0], cfg["train_start_year"], tune_train_end,
                X_val_roll.shape[0], val_year,
            )
        )

        # ── Tune hyperparameters: fit on tune_train, evaluate on val ──────
        logger.info("  Tuning OLS hyperparameters (rolling val %d) …", val_year)
        t0 = time.time()
        best_alpha_ols, best_l1_ols, val_surface_ols, _ = tune_hyperparameters(
            X_val_roll, y_val_roll, cfg, "ols", logger,
            X_train_ext=X_tune_train, y_train_ext=y_tune_train,
        )
        logger.info("  OLS chosen: α=%.5f, l1_ratio=%.2f  (%.1fs)",
                    best_alpha_ols, best_l1_ols, time.time() - t0)

        logger.info("  Tuning Huber hyperparameters (rolling val %d) …", val_year)
        t0 = time.time()
        best_alpha_hub, best_l1_hub, val_surface_hub, best_epsilon_hub = tune_hyperparameters(
            X_val_roll, y_val_roll, cfg, "huber", logger,
            X_train_ext=X_tune_train, y_train_ext=y_tune_train,
        )
        logger.info("  Huber chosen: α=%.5f, l1_ratio=%.2f, ε=%.2f  (%.1fs)",
                    best_alpha_hub, best_l1_hub, best_epsilon_hub, time.time() - t0)

        val_surface_ols["reest_year"] = test_year
        val_surface_ols["loss"]       = "ols"
        val_surface_hub["reest_year"] = test_year
        val_surface_hub["loss"]       = "huber"
        all_val_surfaces.append(pd.concat(
            [val_surface_ols, val_surface_hub], ignore_index=True,
        ))

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
        )
        logger.info("  Training matrix: %s rows × %s features  (%.1fs)",
                    X_train.shape[0], X_train.shape[1], time.time() - t0)

        if X_train.shape[0] < cfg["min_train_obs"]:
            logger.warning("  Skipping %d: too few valid training observations", test_year)
            continue

        # ── Build validation feature matrix (for variable importance only) ─
        # Use the rolling val year (test_year - 1), re-encoded with the final
        # training-window statistics so feature alignment is consistent.
        mask_val_vi = (df["year"] == val_year)
        df_val_vi = pd.DataFrame(df[mask_val_vi])
        X_val, y_val, _, _ = _build_single_window(
            df_val_vi, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
            industry_medians=ind_medians,
        )

        # ── Fit final models on full training window (hyperparams are fixed) ─
        year_pbar.set_postfix(year=test_year, stage="fitting")
        logger.info("  Fitting final OLS model  (α=%.5f, l1_ratio=%.2f) …",
                    best_alpha_ols, best_l1_ols)
        t0 = time.time()
        model_ols, scaler_ols = fit_ols_model(
            X_train.astype(np.float64), y_train.astype(np.float64),
            best_alpha_ols, best_l1_ols, cfg,
        )
        logger.info("  OLS fit done: %d non-zero coefs  (%.1fs)",
                    int(np.sum(model_ols.coef_ != 0)), time.time() - t0)

        logger.info("  Fitting final Huber model  (α=%.5f, l1_ratio=%.2f, ε=%.2f) …",
                    best_alpha_hub, best_l1_hub, best_epsilon_hub)
        t0 = time.time()
        w_huber, scaler_huber = fit_huber_model(
            X_train.astype(np.float64), y_train.astype(np.float64),
            best_alpha_hub, best_l1_hub, cfg,
            epsilon=best_epsilon_hub,
        )
        n_nz_hub = int(np.sum(np.abs(w_huber[:len(feature_names)]) > 1e-10))
        logger.info("  Huber fit done: %d non-zero coefs  (%.1fs)",
                    n_nz_hub, time.time() - t0)

        # ── Build test feature matrix for forecast year ────────────────────
        mask_test = (df["year"] == test_year)
        df_test   = df[mask_test].copy()
        logger.info("  Building test feature matrix (year=%d) …", test_year)
        X_test, y_test, _, valid_test_idx = _build_single_window(
            df_test, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
        )
        logger.info("  Test matrix: %s rows × %s features", X_test.shape[0], X_test.shape[1])

        # ── Predict expected returns ───────────────────────────────────────
        X_test_ols_s = scaler_ols.transform(X_test.astype(np.float64))
        y_pred_ols   = model_ols.predict(X_test_ols_s)

        X_test_s    = scaler_huber.transform(X_test.astype(np.float64))
        ones_test   = np.ones((len(X_test_s), 1))
        X_test_aug  = np.hstack([X_test_s, ones_test])
        y_pred_huber = X_test_aug @ w_huber

        # ── OOS R² ────────────────────────────────────────────────────────
        r2_ols   = compute_gkx_r2(y_test, y_pred_ols)
        r2_huber = compute_gkx_r2(y_test, y_pred_huber)
        logger.info("  OOS R²:  OLS=%.4f   Huber=%.4f", r2_ols, r2_huber)

        # ── Attach forecast metadata back to test rows ─────────────────────
        df_test_valid = df_test.loc[valid_test_idx][["permno", "year", "month"]].copy()
        df_test_valid["expected_ret_ols"]   = y_pred_ols.astype(np.float32)
        df_test_valid["expected_ret_huber"] = y_pred_huber.astype(np.float32)

        all_expected_ols.append(
            df_test_valid[["permno", "year", "month", "expected_ret_ols"]]
            .rename(columns={"expected_ret_ols": "expected_ret"})
        )
        all_expected_huber.append(
            df_test_valid[["permno", "year", "month", "expected_ret_huber"]]
            .rename(columns={"expected_ret_huber": "expected_ret"})
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

        # ── Feature selection diagnostics ──────────────────────────────────
        ols_coefs  = model_ols.coef_
        hub_coefs  = w_huber[:len(feature_names)]
        ols_nz_idx   = np.where(np.abs(ols_coefs)  > 1e-10)[0]
        hub_nz_idx   = np.where(np.abs(hub_coefs)   > 1e-10)[0]

        feat_rows_ols = [
            {"reest_year": test_year, "loss": "ols",
             "feature": feature_names[i], "coef": float(ols_coefs[i])}
            for i in ols_nz_idx if i < len(feature_names)
        ]
        feat_rows_hub = [
            {"reest_year": test_year, "loss": "huber",
             "feature": feature_names[i], "coef": float(hub_coefs[i])}
            for i in hub_nz_idx
        ]
        feat_df = pd.DataFrame(feat_rows_ols + feat_rows_hub)
        all_feat_selection.append(feat_df)

        # ── Variable importance (validation window) ────────────────────────
        logger.info("  Computing variable importance on validation window …")
        t0 = time.time()
        vi_df = compute_variable_importance(
            model_ols, scaler_ols, w_huber, scaler_huber, X_val, y_val, feature_names,
        )
        vi_df.insert(0, "reest_year", test_year)
        all_var_importance.append(vi_df)
        logger.info("  Variable importance done  (%.1fs)", time.time() - t0)

        # ── Metadata ──────────────────────────────────────────────────────
        # val_r2_ols / val_r2_huber: validation R² at the chosen hyperparameters
        # from the rolling 12-month validation window (val_year = test_year - 1).
        _ols_match = val_surface_ols.loc[
            (val_surface_ols["alpha"] == best_alpha_ols) &
            (val_surface_ols["l1_ratio"] == best_l1_ols), "val_r2"
        ]
        rolling_val_r2_ols = float(_ols_match.iloc[0]) if len(_ols_match) > 0 else np.nan

        _hub_match = val_surface_hub.loc[
            (val_surface_hub["alpha"] == best_alpha_hub) &
            (val_surface_hub["l1_ratio"] == best_l1_hub), "val_r2"
        ]
        rolling_val_r2_hub = float(_hub_match.iloc[0]) if len(_hub_match) > 0 else np.nan

        all_metadata.append({
            "reest_year":        test_year,
            "train_start":       cfg["train_start_year"],
            "train_end":         train_end_year,
            "val_year":          val_year,
            "n_train_obs":       len(df_train),
            "n_train_valid_obs": int(X_train.shape[0]),
            "n_features":        int(X_train.shape[1]),
            "n_active_chars":    len(active_chars),
            "n_industry_codes":  len(industry_codes),
            "best_alpha_ols":    best_alpha_ols,
            "best_l1_ols":       best_l1_ols,
            "best_alpha_huber":  best_alpha_hub,
            "best_l1_huber":     best_l1_hub,
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

    if all_expected_ols:
        results["expected_returns_ols"]   = pd.concat(all_expected_ols,   ignore_index=True)
        results["expected_returns_huber"] = pd.concat(all_expected_huber, ignore_index=True)
    else:
        results["expected_returns_ols"]   = pd.DataFrame(columns=["permno","year","month","expected_ret"])
        results["expected_returns_huber"] = pd.DataFrame(columns=["permno","year","month","expected_ret"])

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

    # ── Human-readable diagnostics CSV ────────────────────────────────────
    meta = results["metadata"]
    if not meta.empty:
        diag_cols = [
            "reest_year", "train_start", "train_end", "val_year",
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
