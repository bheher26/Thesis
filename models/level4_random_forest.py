#!/usr/bin/env python3
"""
Level 4: Random Forest — GKX-style pooled panel expected return model
models/level4_random_forest.py

This module implements Level 4 of the multi-empirical asset pricing pipeline.
Expected returns are estimated via a Random Forest regressor on the same GKX-
style feature matrix used for elastic net (Level 3):
  - Firm characteristics (cross-sectionally ranked to [-1, 1])
  - Characteristic × macro-variable interaction terms
  - 2-digit SIC industry dummies

Unlike elastic net, the RF requires no feature standardisation (tree-based
methods are invariant to monotonic feature transformations).  Variable
importance is reported as MDI (mean decrease in impurity) from sklearn's
`feature_importances_` attribute, consistent with GKX (2020) Section 4.3.

Design invariants shared with all model levels:
  - No information from month t or later may appear in any training window.
  - All feature encodings (imputation medians, industry codes, macro scaler)
    are fitted strictly on the training window.
  - Hyperparameters are re-tuned annually via rolling 12-month validation.
  - The final model is refit on the full expanding training window before
    generating forecasts for the test year.

All feature construction utilities are imported from level3_elastic_net to
avoid code duplication and ensure identical feature matrices across levels.

References
----------
Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning",
    Review of Financial Studies 33(5), 2223–2273.
    Section 3.3 (validation design), Section 4.3 (RF, variable importance).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Import shared feature construction utilities from elastic net module ───────
# This keeps feature matrices identical across all Level 3/4 models.
from models.level3_elastic_net import (
    FIRM_CHARACTERISTICS,
    MACRO_VARS,
    _build_single_window,
    _rank_series_pm1,
    check_no_leakage,
    compute_gkx_r2,
    fit_macro_scaler,
    get_industry_codes,
    load_data,
    merge_macro,
    select_active_chars,
    select_active_chars_industry,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_CONFIG: Dict = {
    # ── Window parameters ────────────────────────────────────────────────
    "train_start_year": 1973,
    "train_end_year":   2004,
    "test_start_year":  2005,
    "test_end_year":    2024,

    # ── Random forest hyperparameter grid ────────────────────────────────
    # GKX (2020) find shallow trees (avg ~6 leaves) work best for return
    # prediction.  max_depth caps tree depth; max_features ≈ P/3 is the GKX
    # default (they also report sqrt as competitive).
    "n_estimators_grid": [100, 300],
    "max_depth_grid":    [2, 3, 4, 5],
    "max_features_grid": ["sqrt", 0.1, 0.3],

    # ── Feature construction ─────────────────────────────────────────────
    "char_missing_threshold": 0.50,

    # ── Quality filters ──────────────────────────────────────────────────
    "min_train_obs":        1000,
    "min_stocks_per_month": 50,

    # ── Paths ────────────────────────────────────────────────────────────
    "data_path":  "data_clean/master_panel.csv",
    "macro_path": "data_raw/macro_predictors.csv",
    "output_dir": "data_clean/random_forest",

    # ── Run mode ─────────────────────────────────────────────────────────
    "run_mode":      "full",
    "test_n_stocks": 50,
    "test_n_years":  2,

    # ── Parallelism ───────────────────────────────────────────────────────
    # n_jobs for the outer grid search; the RF itself uses its own n_jobs.
    # Setting both to -1 can over-subscribe CPUs; default splits by level.
    "n_jobs":    -1,   # joblib for HP search
    "rf_n_jobs": -1,   # sklearn RF internal parallelism

    # ── Random seed ───────────────────────────────────────────────────────
    "random_state": 42,
}


# ============================================================
# Logging
# ============================================================

def setup_logging(config: Dict) -> logging.Logger:
    """Configure module-level logger with console + file handler."""
    logger = logging.getLogger("random_forest")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    out_dir = Path(config.get("output_dir", "data_clean/random_forest"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "random_forest_run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Hyperparameter Tuning
# ============================================================

def _cv_score_rf(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    split_indices: List[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    max_depth: int,
    max_features,
    n_jobs: int,
    random_state: int,
) -> float:
    """
    Evaluate one RF configuration via GKX OOS R² on given split(s).

    Called in parallel by tune_hyperparameters_rf for each grid cell.

    Returns
    -------
    mean GKX OOS R² across all folds (NaN-safe; returns -inf on failure).
    """
    scores = []
    for train_idx, val_idx in split_indices:
        X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
        X_vl, y_vl = X_tv[val_idx],   y_tv[val_idx]
        if len(y_tr) < 50 or len(y_vl) < 5:
            continue
        try:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
                bootstrap=True,
                min_samples_leaf=5,   # GKX: prevents over-fitting to noise
            )
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_vl)
            r2 = compute_gkx_r2(y_vl, y_pred)
            scores.append(r2)
        except Exception:
            scores.append(-np.inf)
    return float(np.mean(scores)) if scores else -np.inf


def tune_hyperparameters_rf(
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    logger: logging.Logger,
    X_train_ext: Optional[np.ndarray] = None,
    y_train_ext: Optional[np.ndarray] = None,
) -> Tuple[int, int, object, pd.DataFrame]:
    """
    Grid-search RF hyperparameters using rolling 12-month validation.

    In GKX rolling mode (X_train_ext provided), a single fold is used:
    train on X_train_ext, evaluate on X_val.  This mirrors the design in
    tune_hyperparameters() from the elastic net module.

    Parameters
    ----------
    X_val, y_val       : validation feature matrix and labels
    config             : DEFAULT_CONFIG or override dict
    logger             : module logger
    X_train_ext        : external training matrix (train_start → test_year-2)
    y_train_ext        : external training labels

    Returns
    -------
    best_n_estimators : int
    best_max_depth    : int
    best_max_features : str or float
    surface_df        : DataFrame recording GKX R² for every grid cell
    """
    n_estimators_grid = config["n_estimators_grid"]
    max_depth_grid    = config["max_depth_grid"]
    max_features_grid = config["max_features_grid"]
    n_jobs            = config["n_jobs"]
    rf_n_jobs         = config.get("rf_n_jobs", 1)
    random_state      = config.get("random_state", 42)

    # Build split indices (single fold for GKX rolling mode)
    if X_train_ext is not None and y_train_ext is not None:
        n_tr = len(X_train_ext)
        n_vl = len(X_val)
        X_tv = np.vstack([X_train_ext, X_val])
        y_tv = np.concatenate([y_train_ext, y_val])
        split_indices = [(np.arange(n_tr), np.arange(n_tr, n_tr + n_vl))]
    else:
        # Fallback: single temporal split within X_val (60/40 train/val)
        n_total = len(X_val)
        split_pt = int(n_total * 0.6)
        X_tv = X_val
        y_tv = y_val
        split_indices = [(np.arange(split_pt), np.arange(split_pt, n_total))]

    n_grid = len(n_estimators_grid) * len(max_depth_grid) * len(max_features_grid)
    logger.debug(
        "  RF tuning: %d n_est × %d depth × %d max_feat = %d combos, "
        "n_train=%d, n_val=%d",
        len(n_estimators_grid), len(max_depth_grid), len(max_features_grid),
        n_grid, len(y_tv) - len(y_val), len(y_val),
    )

    # When n_jobs for outer grid search is set, disable RF internal parallelism
    # to avoid over-subscription.  When outer search is sequential (n_jobs=1),
    # let RF use its own parallelism.
    outer_n_jobs = n_jobs
    inner_n_jobs = 1 if (outer_n_jobs != 1) else rf_n_jobs

    raw_scores = Parallel(n_jobs=outer_n_jobs, prefer="threads")(
        delayed(_cv_score_rf)(
            X_tv, y_tv, split_indices,
            n_est, depth, mf,
            inner_n_jobs, random_state,
        )
        for n_est in n_estimators_grid
        for depth in max_depth_grid
        for mf in max_features_grid
    )

    surface_rows = []
    best_score = -np.inf
    best_n_est, best_depth, best_mf = (
        n_estimators_grid[0], max_depth_grid[0], max_features_grid[0],
    )
    idx = 0
    for n_est in n_estimators_grid:
        for depth in max_depth_grid:
            for mf in max_features_grid:
                score = raw_scores[idx] if raw_scores else -np.inf
                surface_rows.append({
                    "n_estimators": n_est,
                    "max_depth":    depth,
                    "max_features": str(mf),
                    "val_r2":       score,
                })
                if score > best_score:
                    best_score = score
                    best_n_est, best_depth, best_mf = n_est, depth, mf
                idx += 1

    if best_score < -10.0:
        best_n_est  = 300
        best_depth  = 3
        best_mf     = "sqrt"
        logger.warning(
            "  RF tuning: all CV scores < -10 (best=%.4f). "
            "Falling back to n_est=300, depth=3, max_features='sqrt'",
            best_score,
        )

    logger.debug(
        "  RF best: n_est=%d, depth=%d, max_feat=%s, val_R²=%.4f",
        best_n_est, best_depth, best_mf, best_score,
    )

    return best_n_est, best_depth, best_mf, pd.DataFrame(surface_rows)


# ============================================================
# Final Model Fitting
# ============================================================

def fit_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
    max_features,
    config: Dict,
) -> RandomForestRegressor:
    """
    Fit a Random Forest on the full training window.

    No feature standardisation: tree splits are scale-invariant.

    Parameters
    ----------
    X_train, y_train : training feature matrix and labels
    n_estimators     : number of trees (chosen by HP tuning)
    max_depth        : maximum tree depth (chosen by HP tuning)
    max_features     : feature subsetting rule (chosen by HP tuning)
    config           : DEFAULT_CONFIG or override dict

    Returns
    -------
    rf : fitted RandomForestRegressor
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=config.get("rf_n_jobs", -1),
        random_state=config.get("random_state", 42),
        bootstrap=True,
        min_samples_leaf=5,
    )
    rf.fit(X_train, y_train)
    return rf


# ============================================================
# Variable Importance (MDI)
# ============================================================

def compute_variable_importance_rf(
    rf: RandomForestRegressor,
    feature_names: List[str],
    test_year: int,
) -> pd.DataFrame:
    """
    Extract MDI (mean decrease in impurity) feature importance from a fitted RF.

    MDI sums the weighted impurity decrease over all tree nodes where a given
    feature is used as the split criterion, averaged across all trees.
    sklearn normalises so that importances sum to 1.

    Consistent with GKX (2020) Section 4.3, which reports normalised MDI
    variable importance for the RF model.

    Parameters
    ----------
    rf            : fitted RandomForestRegressor
    feature_names : list of feature column labels (length must match n_features)
    test_year     : re-estimation year (appended to output DataFrame)

    Returns
    -------
    DataFrame with columns: [reest_year, feature, importance]
    sorted descending by importance.
    """
    importances = rf.feature_importances_
    n_names = min(len(feature_names), len(importances))
    rows = [
        {"reest_year": test_year,
         "feature":    feature_names[i],
         "importance": float(importances[i])}
        for i in range(n_names)
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# Main Pipeline Function
# ============================================================

def run_random_forest(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Run the full random forest expected-return pipeline.

    Parameters
    ----------
    df        : master_panel DataFrame (firm × month rows)
    macro_df  : macro predictors DataFrame with (year, month) columns
    config    : dict overriding any keys in DEFAULT_CONFIG

    Returns
    -------
    results : dict with DataFrames:
        expected_returns  — (permno, year, month, expected_ret)
        oos_r2            — monthly GKX OOS R²
        metadata          — per-reestimation-year diagnostics
        var_importance    — MDI feature importance per year
        val_surface       — full HP validation grid per year
    """
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)

    logger = setup_logging(cfg)
    logger.info("=" * 65)
    logger.info("LEVEL 4: RANDOM FOREST  —  run_mode=%s", cfg["run_mode"].upper())
    logger.info("=" * 65)

    # ── Test-mode subsample ───────────────────────────────────────────────
    if cfg["run_mode"] == "test":
        logger.info("TEST MODE: limiting to %d stocks, %d re-estimation years",
                    cfg["test_n_stocks"], cfg["test_n_years"])
        all_permnos = sorted(df["permno"].unique())
        rng = np.random.default_rng(42)
        selected = rng.choice(
            all_permnos,
            size=min(cfg["test_n_stocks"], len(all_permnos)),
            replace=False,
        )
        df = df[df["permno"].isin(selected)].copy()
        cfg["test_end_year"] = cfg["test_start_year"] + cfg["test_n_years"] - 1

    # ── Merge macro onto panel ────────────────────────────────────────────
    logger.info("Merging macro predictors onto master panel …")
    df = merge_macro(df, macro_df)
    available_macro = [m for m in MACRO_VARS if m in df.columns]
    missing_macro   = [m for m in MACRO_VARS if m not in df.columns]
    if missing_macro:
        logger.warning("Macro columns not found: %s", missing_macro)

    # ── Re-estimation years ───────────────────────────────────────────────
    test_years = list(range(cfg["test_start_year"], cfg["test_end_year"] + 1))
    logger.info(
        "Test period: %d – %d  (%d re-estimation points)",
        cfg["test_start_year"], cfg["test_end_year"], len(test_years),
    )

    # ── Output containers ─────────────────────────────────────────────────
    all_expected    : List[pd.DataFrame] = []
    all_oos_r2      : List[Dict]         = []
    all_metadata    : List[Dict]         = []
    all_var_imp     : List[pd.DataFrame] = []
    all_val_surfaces: List[pd.DataFrame] = []

    total_start = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    year_pbar = tqdm(test_years, desc="Random Forest", unit="year", ncols=90, leave=True)
    for reest_idx, test_year in enumerate(year_pbar):
        loop_start     = time.time()
        train_end_year = test_year - 1
        tune_train_end = test_year - 2
        val_year       = test_year - 1

        year_pbar.set_postfix(year=test_year, stage="tuning")
        logger.info("")
        logger.info("─" * 55)
        logger.info(
            "RE-ESTIMATION  %d / %d  →  val=%d, forecast=%d",
            reest_idx + 1, len(test_years), val_year, test_year,
        )
        logger.info("─" * 55)

        # ── Leakage check ─────────────────────────────────────────────────
        check_no_leakage(train_end_year, 12, test_year, logger)

        # ── Tuning training window (train_start … test_year-2) ────────────
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
        tune_industry   = get_industry_codes(df_tune_train)
        tune_macro_means, tune_macro_stds = fit_macro_scaler(
            df_tune_train, available_macro,
        )

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

        # ── Tune RF hyperparameters ───────────────────────────────────────
        logger.info("  Tuning RF hyperparameters (rolling val %d) …", val_year)
        t0 = time.time()
        best_n_est, best_depth, best_mf, val_surface = tune_hyperparameters_rf(
            X_val_roll, y_val_roll, cfg, logger,
            X_train_ext=X_tune_train, y_train_ext=y_tune_train,
        )
        logger.info(
            "  RF chosen: n_est=%d, depth=%d, max_feat=%s  (%.1fs)",
            best_n_est, best_depth, best_mf, time.time() - t0,
        )
        val_surface["reest_year"] = test_year
        all_val_surfaces.append(val_surface)

        # ── Final training window (train_start … test_year-1) ────────────
        mask_train = (
            (df["year"] >= cfg["train_start_year"]) &
            (df["year"] <= train_end_year)
        )
        df_train = pd.DataFrame(df[mask_train])
        logger.info(
            "  Final training rows: {:,}  ({} – {})".format(
                len(df_train), cfg["train_start_year"], train_end_year,
            )
        )

        if len(df_train) < cfg["min_train_obs"]:
            logger.warning(
                "  Skipping %d: only %d training obs (min=%d)",
                test_year, len(df_train), cfg["min_train_obs"],
            )
            continue

        # ── Fit encodings on full training window ─────────────────────────
        active_chars, train_medians = select_active_chars(
            df_train, FIRM_CHARACTERISTICS, cfg["char_missing_threshold"],
        )
        industry_codes = get_industry_codes(df_train)
        macro_means, macro_stds = fit_macro_scaler(df_train, available_macro)

        use_ind_impute = cfg.get("industry_imputation", False)
        ind_medians: Optional[Dict] = (
            select_active_chars_industry(df_train, active_chars)
            if use_ind_impute else None
        )
        logger.info(
            "  Active chars: %d  |  industry codes: %d  |  ind_impute=%s",
            len(active_chars), len(industry_codes), use_ind_impute,
        )

        # ── Build training feature matrix ─────────────────────────────────
        logger.info("  Building training feature matrix …")
        t0 = time.time()
        X_train_mat, y_train_arr, feature_names, _ = _build_single_window(
            df_train, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
            industry_medians=ind_medians,
        )
        logger.info(
            "  Training matrix: %s rows × %s features  (%.1fs)",
            X_train_mat.shape[0], X_train_mat.shape[1], time.time() - t0,
        )

        if X_train_mat.shape[0] < cfg["min_train_obs"]:
            logger.warning("  Skipping %d: too few valid training observations", test_year)
            continue

        # ── Fit final RF model ────────────────────────────────────────────
        year_pbar.set_postfix(year=test_year, stage="fitting")
        logger.info(
            "  Fitting RF (n_est=%d, depth=%d, max_feat=%s) …",
            best_n_est, best_depth, best_mf,
        )
        t0 = time.time()
        rf = fit_rf_model(
            X_train_mat.astype(np.float32),
            y_train_arr.astype(np.float32),
            best_n_est, best_depth, best_mf, cfg,
        )
        logger.info("  RF fit done  (%.1fs)", time.time() - t0)

        # ── Build test feature matrix for forecast year ────────────────────
        mask_test = (df["year"] == test_year)
        df_test   = df[mask_test].copy()
        logger.info("  Building test feature matrix (year=%d) …", test_year)
        X_test, y_test, _, valid_test_idx = _build_single_window(
            df_test, active_chars, train_medians,
            available_macro, industry_codes,
            macro_means, macro_stds,
        )
        logger.info(
            "  Test matrix: %s rows × %s features", X_test.shape[0], X_test.shape[1],
        )

        # ── Predict expected returns ───────────────────────────────────────
        y_pred = rf.predict(X_test.astype(np.float32))

        # ── OOS R² ────────────────────────────────────────────────────────
        r2 = compute_gkx_r2(y_test, y_pred)
        logger.info("  OOS R² (RF): %.4f", r2)

        # ── Attach forecasts to test rows ──────────────────────────────────
        df_test_valid = df_test.loc[valid_test_idx][["permno", "year", "month"]].copy()
        df_test_valid["expected_ret"] = y_pred.astype(np.float32)
        all_expected.append(df_test_valid[["permno", "year", "month", "expected_ret"]])

        # ── Monthly OOS R² breakdown ───────────────────────────────────────
        df_test_valid["y_true"] = y_test.astype(np.float32)
        df_test_valid["y_pred"] = y_pred.astype(np.float32)
        for (yr, mo), grp in df_test_valid.groupby(["year", "month"]):
            if len(grp) < 5:
                continue
            r2_mo = compute_gkx_r2(grp["y_true"].values, grp["y_pred"].values)
            all_oos_r2.append({"year": yr, "month": mo, "r2_rf": r2_mo})

        # ── Variable importance (MDI) ──────────────────────────────────────
        vi_df = compute_variable_importance_rf(rf, feature_names, test_year)
        all_var_imp.append(vi_df)

        # ── Metadata ──────────────────────────────────────────────────────
        # Retrieve validation R² at the chosen HP combination
        match = val_surface.loc[
            (val_surface["n_estimators"] == best_n_est) &
            (val_surface["max_depth"]    == best_depth) &
            (val_surface["max_features"] == str(best_mf)),
            "val_r2",
        ]
        rolling_val_r2 = float(match.iloc[0]) if len(match) > 0 else np.nan

        all_metadata.append({
            "reest_year":        test_year,
            "train_start":       cfg["train_start_year"],
            "train_end":         train_end_year,
            "val_year":          val_year,
            "n_train_obs":       len(df_train),
            "n_train_valid_obs": int(X_train_mat.shape[0]),
            "n_features":        int(X_train_mat.shape[1]),
            "n_active_chars":    len(active_chars),
            "n_industry_codes":  len(industry_codes),
            "best_n_estimators": best_n_est,
            "best_max_depth":    best_depth,
            "best_max_features": str(best_mf),
            "rolling_val_r2":    rolling_val_r2,
            "test_r2_rf":        r2,
        })

        elapsed = time.time() - loop_start
        year_pbar.set_postfix(
            year=test_year,
            r2_rf=f"{r2:.3f}",
            min=f"{elapsed/60:.1f}",
        )
        logger.info(
            "  Re-estimation %d complete  (%.1f min elapsed)",
            test_year, elapsed / 60,
        )

    # ── Assemble output DataFrames ────────────────────────────────────────
    logger.info("")
    logger.info("Assembling output DataFrames …")

    results: Dict = {}

    results["expected_returns"] = (
        pd.concat(all_expected, ignore_index=True)
        if all_expected
        else pd.DataFrame(columns=["permno", "year", "month", "expected_ret"])
    )
    results["oos_r2"]       = pd.DataFrame(all_oos_r2)
    results["metadata"]     = pd.DataFrame(all_metadata)
    results["var_importance"] = (
        pd.concat(all_var_imp, ignore_index=True)
        if all_var_imp else pd.DataFrame()
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

def save_outputs_rf(results: Dict, config: Dict) -> None:
    """
    Persist all result DataFrames to parquet + human-readable CSV.
    """
    logger  = logging.getLogger("random_forest")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, fname: str) -> None:
        path = out_dir / fname
        df.to_parquet(path, index=False)
        logger.info("  Saved %s  (%d rows)", fname, len(df))

    _save(results["expected_returns"], "expected_returns_rf.parquet")
    _save(results["oos_r2"],           "oos_r2_rf.parquet")
    _save(results["metadata"],         "model_metadata_rf.parquet")
    _save(results["var_importance"],   "variable_importance_rf.parquet")
    _save(results["val_surface"],      "validation_surface_rf.parquet")

    # ── Human-readable diagnostics CSV ────────────────────────────────────
    meta = results["metadata"]
    if not meta.empty:
        diag_cols = [
            "reest_year", "train_start", "train_end", "val_year",
            "best_n_estimators", "best_max_depth", "best_max_features",
            "rolling_val_r2", "test_r2_rf",
            "n_features", "n_train_valid_obs",
        ]
        diag_cols_present = [c for c in diag_cols if c in meta.columns]
        meta[diag_cols_present].to_csv(
            out_dir / "diagnostics_summary_rf.csv",
            index=False, float_format="%.6f",
        )
        logger.info("  Saved diagnostics_summary_rf.csv")

    logger.info("All RF outputs saved to %s", out_dir)


# ============================================================
# Portfolio Adapter — run_backtest()-compatible interface
# ============================================================

# Module-level cache so the parquet is read only once per session.
_RF_CACHE: Optional[pd.DataFrame] = None


def _load_rf_forecasts(output_dir: str) -> pd.DataFrame:
    """
    Load pre-computed RF expected returns from disk, cached in memory.

    Parameters
    ----------
    output_dir : directory where save_outputs_rf() wrote the parquet files

    Returns
    -------
    DataFrame with columns [permno, year, month, expected_ret]
    """
    global _RF_CACHE
    if _RF_CACHE is not None:
        return _RF_CACHE

    path = Path(output_dir) / "expected_returns_rf.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"RF forecasts not found at {path}. "
            "Run level4_random_forest.py (or run_random_forest()) first."
        )
    df = pd.read_parquet(path)
    df["permno"] = df["permno"].astype(int)
    df["year"]   = df["year"].astype(int)
    df["month"]  = df["month"].astype(int)
    _RF_CACHE = df
    return df


def make_rf_expected_returns_fn(
    output_dir: str = DEFAULT_CONFIG["output_dir"],
) -> Callable:
    """
    Return an expected_returns_fn adapter for use with run_backtest().

    The adapter reads from the pre-computed parquet saved by run_random_forest()
    / save_outputs_rf(), so RF expected returns slot into the same portfolio
    optimizer as all other model levels without recomputing.

    Parameters
    ----------
    output_dir : directory containing expected_returns_rf.parquet

    Returns
    -------
    fn : callable with signature fn(master_df, ret_matrix, year, month) -> pd.Series
         Returns expected total return indexed by permno; NaN for stocks not
         covered by the RF forecast for that month.

    Usage
    -----
    from models.level4_random_forest import make_rf_expected_returns_fn

    results = run_backtest(
        master,
        start_year=2005, start_month=1,
        end_year=2024,   end_month=12,
        expected_returns_fn=make_rf_expected_returns_fn(),
        ...
    )
    """
    def rf_expected_returns(_master_df, ret_matrix, year, month):  # noqa: ARG001
        """
        Adapter: look up pre-computed RF expected returns for (year, month).

        Signature matches the expected_returns_fn hook in run_backtest():
            fn(master_df, ret_matrix, year, month) -> pd.Series

        Parameters
        ----------
        _master_df : pd.DataFrame — not used (forecasts are pre-computed)
        ret_matrix : pd.DataFrame, shape (T, N) — used to obtain current universe
        year, month : int — portfolio formation month

        Returns
        -------
        mu : pd.Series indexed by permno
             Expected return for each stock in ret_matrix.columns.
             NaN for stocks not covered by the RF forecast.
        """
        forecasts = _load_rf_forecasts(output_dir)
        month_slice = forecasts[
            (forecasts["year"] == year) & (forecasts["month"] == month)
        ]
        month_slice = pd.DataFrame(month_slice).drop_duplicates(subset=["permno"])
        month_fc = month_slice.set_index("permno")["expected_ret"]
        mu = month_fc.reindex(ret_matrix.columns)
        n_valid = mu.notna().sum()
        n_total = len(mu)
        logging.getLogger("random_forest").debug(
            "  RF [%d-%02d]: %d / %d stocks with forecasts",
            year, month, n_valid, n_total,
        )
        return mu

    return rf_expected_returns


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Level 4: Random Forest — empirical asset pricing pipeline",
    )
    parser.add_argument(
        "--mode", choices=["full", "test"], default="full",
        help="'test' runs on a small subsample for debugging (default: full)",
    )
    parser.add_argument(
        "--train-start", type=int, default=DEFAULT_CONFIG["train_start_year"],
        help="First year of training data",
    )
    parser.add_argument(
        "--test-start", type=int, default=DEFAULT_CONFIG["test_start_year"],
        help="First year of the held-out test period",
    )
    parser.add_argument(
        "--test-end", type=int, default=DEFAULT_CONFIG["test_end_year"],
        help="Last year of the held-out test period",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
        help="Directory for output parquet files",
    )
    args = parser.parse_args()

    run_config = {
        "run_mode":        args.mode,
        "train_start_year": args.train_start,
        "test_start_year":  args.test_start,
        "test_end_year":    args.test_end,
        "output_dir":       args.output_dir,
    }

    _df, _macro_df = load_data({**DEFAULT_CONFIG, **run_config})
    _results = run_random_forest(_df, _macro_df, config=run_config)
    save_outputs_rf(_results, {**DEFAULT_CONFIG, **run_config})
    print("\nDone. Outputs written to", run_config["output_dir"])
