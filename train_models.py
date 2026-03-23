#!/usr/bin/env python3
"""
Model Training Script

Trains the optimal models:
- Glicko-2 for player ratings and win probability
- VisitLevel180sModel for 180s prediction
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import ParquetStore
from src.models import (
    Glicko2System,
    Glicko2Config,
    VisitLevel180sModel,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def train_visit_level_180s_model(
    data_dir: str = "data/processed",
    output_path: str = "data/models/model_180s_visit_level.json",
    prior_strength: float = 50.0,
    min_visits_for_direct: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Train the VisitLevel180sModel for 180s prediction.

    Uses an expanding window approach:
    1. Collect out-of-sample predictions incrementally (for bias calibration)
    2. Train final model on ALL data (so all players have current rates)
    3. Calibrate bias correction on the out-of-sample predictions

    This ensures recent matches (like MODUS 2026) are included in training.

    Args:
        data_dir: Path to processed data
        output_path: Path to save model
        prior_strength: Effective prior sample size for shrinkage
        min_visits_for_direct: Minimum visits to use direct estimate
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Training VisitLevel180sModel")
    logger.info("=" * 60)

    # Load data
    store = ParquetStore(data_dir)
    matches_df = store.get_matches()
    visits_df = store.get_visits()
    stats_df = store.get_match_stats()

    logger.info(f"Loaded {len(matches_df)} matches")
    logger.info(f"Loaded {len(visits_df)} visits")
    logger.info(f"Loaded {len(stats_df)} stats")

    # Ensure correct types
    matches_df = _ensure_match_types(matches_df)
    visits_df = _ensure_visit_types(visits_df)
    stats_df = _ensure_stats_types(stats_df)

    # Sort by date
    if "match_date" in matches_df.columns:
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    n = len(matches_df)

    # ========================================================================
    # EXPANDING WINDOW: Collect out-of-sample predictions for calibration
    # ========================================================================
    oos_predictions = []  # List of (match_id, predicted, actual) tuples

    logger.info("\nPhase 1: Collecting out-of-sample predictions (expanding window)...")

    # Build actual 180s lookup from stats
    stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
    if stat_180s.empty:
        stat_180s = stats_df[stats_df.get("stat_field_name", "") == "Thrown 180"].copy()
    if not stat_180s.empty:
        stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)
        match_180s_stats = stat_180s.groupby("match_id")["value_num"].sum().to_dict()
    else:
        match_180s_stats = {}

    # Fallback: count from visits
    if not visits_df.empty:
        match_180s_visits = visits_df.groupby("match_id")["is_180"].sum().to_dict()
    else:
        match_180s_visits = {}

    # Start with first 50%, predict on next 10%, expand, repeat
    window_start = int(n * 0.50)
    window_step = int(n * 0.10)

    current_end = window_start
    while current_end < n:
        # Determine prediction window
        pred_start = current_end
        pred_end = min(current_end + window_step, n)

        # Train temporary model on data up to current_end
        train_slice = matches_df.iloc[:current_end]
        train_match_ids = set(train_slice["match_id"].tolist())
        train_visits = visits_df[visits_df["match_id"].isin(train_match_ids)] if not visits_df.empty else visits_df
        train_stats = stats_df[stats_df["match_id"].isin(train_match_ids)] if not stats_df.empty else stats_df

        temp_model = VisitLevel180sModel(
            prior_strength=prior_strength,
            min_visits_for_direct=min_visits_for_direct,
        )
        temp_model.fit(train_visits, train_slice, train_stats)

        # Make predictions on the next window (out-of-sample)
        pred_slice = matches_df.iloc[pred_start:pred_end]
        window_preds = []

        for _, match in pred_slice.iterrows():
            home_id = match["home_player_id"]
            away_id = match["away_player_id"]
            league_id = match.get("league_id", 2)
            match_id = match["match_id"]

            format_params = {
                "best_of_sets": match.get("best_of_sets", 0),
                "best_of_legs": match.get("best_of_legs", 5),
                "is_set_format": match.get("is_set_format", False),
            }

            pred = temp_model.predict(home_id, away_id, league_id, format_params)
            predicted = pred.lambda_total

            # Get actual
            actual = match_180s_stats.get(match_id, match_180s_visits.get(match_id, None))

            if actual is not None:
                window_preds.append({
                    "match_id": match_id,
                    "predicted": predicted,
                    "actual": actual,
                    "expected_visits": pred.expected_visits,
                    "best_of_sets": int(match.get("best_of_sets", 0)),
                    "best_of_legs": int(match.get("best_of_legs", 5)),
                    "is_set_format": bool(match.get("is_set_format", False)),
                    "league_id": league_id,
                })

        oos_predictions.extend(window_preds)
        logger.info(f"  Window {current_end}/{n}: trained on {current_end}, predicted {len(window_preds)}")
        current_end = pred_end

    oos_df = pd.DataFrame(oos_predictions)
    logger.info(f"  Total out-of-sample predictions: {len(oos_df)}")

    # ========================================================================
    # FINAL MODEL: Train on ALL data
    # ========================================================================
    logger.info("\nPhase 2: Training final model on ALL data...")

    model = VisitLevel180sModel(
        prior_strength=prior_strength,
        min_visits_for_direct=min_visits_for_direct,
    )
    model.fit(visits_df, matches_df, stats_df)
    logger.info(f"Total players with rates: {len(model.player_stats)}")

    # Show date range
    if "match_date" in matches_df.columns:
        logger.info(f"Training data: {matches_df['match_date'].min()} to {matches_df['match_date'].max()}")

    # ========================================================================
    # CALIBRATION: Fit bias correction on out-of-sample predictions
    # ========================================================================
    if len(oos_df) >= 50:
        logger.info("\nPhase 3: Calibrating bias correction on out-of-sample predictions...")

        # Calculate bias metrics before calibration
        preds = oos_df["predicted"].values
        actuals = oos_df["actual"].values
        pre_bias = np.mean(preds) - np.mean(actuals)
        pre_rmse = np.sqrt(np.mean((preds - actuals) ** 2))

        logger.info(f"  Pre-calibration: Bias={pre_bias:.3f}, RMSE={pre_rmse:.3f}")

        # Use OOS data to calibrate (compute bias corrections)
        # We need to create a mock matches/stats structure for the calibrate method
        # Or we can compute the corrections directly here

        # Fit separate regressions for LEG vs SET formats (they have different bias patterns)
        residuals = actuals - preds  # Positive = underpredicting
        expected_visits = oos_df["expected_visits"].values
        is_set_format = oos_df["is_set_format"].values

        # Split by format type
        leg_mask = ~is_set_format
        set_mask = is_set_format

        # DISABLE both LEG and SET format calibration
        # Expanding window calibration learns biases from early model states that don't
        # apply to the final model. Analysis showed the raw final model is more accurate:
        # - LEG: raw bias -0.14, calibrated bias -1.77 (calibration makes it WORSE)
        # - SET: raw bias +0.54, calibrated bias much worse
        model.bias_slope_leg = 0.0
        model.bias_intercept_leg = 0.0
        model.bias_slope_set = 0.0
        model.bias_intercept_set = 0.0
        for fmt_name, fmt_mask in [("LEG", leg_mask), ("SET", set_mask)]:
            if fmt_mask.sum() > 0:
                raw_bias = residuals[fmt_mask].mean()
                logger.info(f"  {fmt_name} format ({fmt_mask.sum()} matches): DISABLED calibration")
                logger.info(f"    (OOS bias was {raw_bias:+.2f}, but raw final model is more accurate)")

        # Global bias_correction is now 0 (absorbed into regressions)
        model.bias_correction = 0.0

        # Clear legacy format corrections (no longer needed)
        model.format_bias_corrections = {}

        logger.info(f"  All calibration DISABLED (raw final model is more accurate)")

        # ================================================================
        # Estimate overdispersion via MLE on the final model's raw predictions
        # Uses match stats actuals (covers all formats, not just visit data)
        # ================================================================
        from scipy import stats as sp_stats

        # Build actual 180s lookup from match stats
        all_stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
        if all_stat_180s.empty:
            all_stat_180s = stats_df[stats_df.get("stat_field_name", "") == "Thrown 180"].copy()
        if not all_stat_180s.empty:
            all_stat_180s["value_num"] = pd.to_numeric(all_stat_180s["value"], errors="coerce").fillna(0)
            all_match_180s = all_stat_180s.groupby("match_id")["value_num"].sum().to_dict()
        else:
            all_match_180s = {}
        # Supplement with visit data
        if not visits_df.empty:
            visit_180s_lookup = visits_df.groupby("match_id")["is_180"].sum().to_dict()
        else:
            visit_180s_lookup = {}

        # Collect final model predictions (raw, no calibration) vs actuals, by format
        from collections import defaultdict
        mle_data = defaultdict(lambda: {"preds": [], "actuals": []})

        valid_matches = matches_df[
            ~((matches_df["best_of_sets"] == 0) & (matches_df["best_of_legs"] == 0))
        ]
        for _, row in valid_matches.iterrows():
            h, a = int(row["home_player_id"]), int(row["away_player_id"])
            if h == 0 or a == 0:
                continue
            mid = row["match_id"]
            actual = all_match_180s.get(mid, visit_180s_lookup.get(mid, None))
            if actual is None:
                continue
            fmt = {
                "best_of_sets": int(row["best_of_sets"]),
                "best_of_legs": int(row["best_of_legs"]),
                "is_set_format": bool(row["is_set_format"]),
            }
            pred = model.predict(h, a, int(row.get("league_id", 2)), fmt)
            if pred.lambda_total < 0.01:
                continue
            key = "set" if fmt["is_set_format"] else "leg"
            mle_data[key]["preds"].append(pred.lambda_total)
            mle_data[key]["actuals"].append(int(actual))

        # MLE grid search for optimal phi per format
        def _nb_loglik(preds_arr, actuals_arr, phi):
            """Total NB log-likelihood for a given phi."""
            ll = 0.0
            for mu, k in zip(preds_arr, actuals_arr):
                if phi <= 1.0:
                    ll += sp_stats.poisson.logpmf(k, mu)
                else:
                    n = mu / (phi - 1.0)
                    p = 1.0 / phi
                    ll += sp_stats.nbinom.logpmf(k, n, p)
            return ll

        model.overdispersion_by_format = {}
        all_preds_global = []
        all_actuals_global = []

        for fmt_key in ("leg", "set"):
            if fmt_key not in mle_data or len(mle_data[fmt_key]["preds"]) < 20:
                logger.info(f"  Overdispersion ({fmt_key}): insufficient data, using global")
                continue

            p_arr = np.array(mle_data[fmt_key]["preds"])
            a_arr = np.array(mle_data[fmt_key]["actuals"])
            all_preds_global.extend(p_arr)
            all_actuals_global.extend(a_arr)

            # Coarse grid
            best_phi, best_ll = 1.0, _nb_loglik(p_arr, a_arr, 1.0)
            for phi in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
                ll = _nb_loglik(p_arr, a_arr, phi)
                if ll > best_ll:
                    best_ll = ll
                    best_phi = phi

            # Fine grid around best
            lo = max(1.1, best_phi - 0.8)
            hi = best_phi + 0.8
            for phi in np.arange(lo, hi, 0.1):
                ll = _nb_loglik(p_arr, a_arr, phi)
                if ll > best_ll:
                    best_ll = ll
                    best_phi = round(float(phi), 1)

            poisson_ll = _nb_loglik(p_arr, a_arr, 1.0)
            pct_gain = (best_ll - poisson_ll) / abs(poisson_ll) * 100

            model.overdispersion_by_format[fmt_key] = best_phi
            logger.info(f"  Overdispersion MLE ({fmt_key}, n={len(p_arr)}): "
                       f"phi={best_phi:.1f} (NB vs Poisson: {pct_gain:+.1f}%)")

        # Global overdispersion from all data
        if all_preds_global:
            p_all = np.array(all_preds_global)
            a_all = np.array(all_actuals_global)
            best_phi_g, best_ll_g = 1.0, _nb_loglik(p_all, a_all, 1.0)
            for phi in np.arange(1.1, 10.0, 0.1):
                ll = _nb_loglik(p_all, a_all, phi)
                if ll > best_ll_g:
                    best_ll_g = ll
                    best_phi_g = round(float(phi), 1)
            model.overdispersion = best_phi_g
            logger.info(f"  Overdispersion MLE (global, n={len(p_all)}): phi={best_phi_g:.1f}")

        metrics = {
            "n_oos": len(oos_df),
            "oos_bias": float(pre_bias),
            "oos_rmse": float(pre_rmse),
            "overdispersion": float(model.overdispersion),
            "overdispersion_by_format": {k: float(v) for k, v in model.overdispersion_by_format.items()},
        }
    else:
        logger.warning(f"Skipping calibration: only {len(oos_df)} OOS samples (need 50+)")
        metrics = {}

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info(f"\nModel saved to {output_path}")

    return metrics


def _evaluate_visit_level_model(
    model: VisitLevel180sModel,
    test_matches: pd.DataFrame,
    test_visits: pd.DataFrame,
    test_stats: pd.DataFrame,
    logger: logging.Logger,
) -> dict:
    """Evaluate the visit-level 180s model."""
    predictions = []
    actuals = []

    # Get actual 180s from stats
    stat_180s = test_stats[test_stats["stat_field_id"] == 2].copy()
    if stat_180s.empty:
        stat_180s = test_stats[test_stats["stat_field_name"] == "Thrown 180"].copy()

    if not stat_180s.empty:
        stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)
        match_180s = stat_180s.groupby("match_id")["value_num"].sum().to_dict()
    else:
        match_180s = {}

    # Fallback: count from visits
    if not test_visits.empty:
        visit_180s = test_visits.groupby("match_id")["is_180"].sum().to_dict()
    else:
        visit_180s = {}

    for _, match in test_matches.iterrows():
        home_id = match["home_player_id"]
        away_id = match["away_player_id"]
        league_id = match.get("league_id", 2)

        format_params = {
            "best_of_sets": match.get("best_of_sets", 0),
            "best_of_legs": match.get("best_of_legs", 5),
            "is_set_format": match.get("is_set_format", False),
        }

        pred = model.predict(home_id, away_id, league_id, format_params)
        predictions.append(pred.lambda_total)

        # Get actual
        match_id = match["match_id"]
        actual = match_180s.get(match_id, visit_180s.get(match_id, None))
        if actual is not None:
            actuals.append(actual)
        else:
            actuals.append(np.nan)

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    valid = ~np.isnan(actuals)
    if valid.sum() < 10:
        logger.warning("Not enough test samples with actual 180s counts")
        return {"error": "insufficient_data"}

    preds_valid = predictions[valid]
    actuals_valid = actuals[valid]

    mse = np.mean((preds_valid - actuals_valid) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_valid - actuals_valid))
    mean_actual = np.mean(actuals_valid)
    mean_pred = np.mean(preds_valid)
    ss_tot = np.sum((actuals_valid - mean_actual) ** 2)
    ss_res = np.sum((actuals_valid - preds_valid) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    bias = mean_pred - mean_actual

    metrics = {
        "n_test": int(valid.sum()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "mean_actual": mean_actual,
        "mean_predicted": mean_pred,
    }

    logger.info("Test metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return metrics


def train_modus_180s_model(
    data_dir: str = "data/processed",
    output_path: str = "data/models/model_180s_modus.json",
    prior_strength: float = 50.0,
    min_visits_for_direct: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Train a standalone 180s model exclusively on MODUS data (league_id=38).

    The global model underpredicts MODUS by ~31% because shrinkage toward the
    global population rate (0.050) followed by a league multiplier (0.649) doesn't
    correctly reconstruct the MODUS population rate (0.032). A standalone model
    with its own population prior eliminates this artefact.

    Same pipeline as global model: expanding window OOS → final fit → MLE overdispersion.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Training MODUS-specific 180s Model (league_id=38)")
    logger.info("=" * 60)

    MODUS_LEAGUE_ID = 38

    # Load data
    store = ParquetStore(data_dir)
    matches_df = store.get_matches()
    visits_df = store.get_visits()
    stats_df = store.get_match_stats()

    # Ensure correct types
    matches_df = _ensure_match_types(matches_df)
    visits_df = _ensure_visit_types(visits_df)
    stats_df = _ensure_stats_types(stats_df)

    # Filter to MODUS only
    matches_df = matches_df[matches_df["league_id"] == MODUS_LEAGUE_ID].copy()
    modus_match_ids = set(matches_df["match_id"])
    visits_df = visits_df[visits_df["match_id"].isin(modus_match_ids)].copy()
    stats_df = stats_df[stats_df["match_id"].isin(modus_match_ids)].copy()

    # Fix format: best_of_sets=1 is NOT a set format (it's a single set of legs)
    matches_df.loc[matches_df["best_of_sets"] <= 1, "is_set_format"] = False

    logger.info(f"MODUS matches: {len(matches_df)}")
    logger.info(f"MODUS visits: {len(visits_df)}")
    logger.info(f"MODUS stats: {len(stats_df)}")

    if len(matches_df) < 50:
        logger.warning("Insufficient MODUS data, skipping MODUS model training")
        return {}

    # Sort by date
    if "match_date" in matches_df.columns:
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    n = len(matches_df)

    # ========================================================================
    # EXPANDING WINDOW: Collect out-of-sample predictions
    # ========================================================================
    oos_predictions = []
    logger.info("\nPhase 1: Collecting out-of-sample predictions (expanding window)...")

    # Build actual 180s lookup from stats
    stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
    if stat_180s.empty:
        stat_180s = stats_df[stats_df.get("stat_field_name", "") == "Thrown 180"].copy()
    if not stat_180s.empty:
        stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)
        match_180s_stats = stat_180s.groupby("match_id")["value_num"].sum().to_dict()
    else:
        match_180s_stats = {}

    # Fallback: count from visits
    if not visits_df.empty:
        match_180s_visits = visits_df.groupby("match_id")["is_180"].sum().to_dict()
    else:
        match_180s_visits = {}

    window_start = int(n * 0.50)
    window_step = int(n * 0.10)

    current_end = window_start
    while current_end < n:
        pred_start = current_end
        pred_end = min(current_end + window_step, n)

        train_slice = matches_df.iloc[:current_end]
        train_match_ids = set(train_slice["match_id"].tolist())
        train_visits = visits_df[visits_df["match_id"].isin(train_match_ids)] if not visits_df.empty else visits_df
        train_stats = stats_df[stats_df["match_id"].isin(train_match_ids)] if not stats_df.empty else stats_df

        temp_model = VisitLevel180sModel(
            prior_strength=prior_strength,
            min_visits_for_direct=min_visits_for_direct,
        )
        temp_model.fit(train_visits, train_slice, train_stats)

        pred_slice = matches_df.iloc[pred_start:pred_end]
        window_preds = []

        for _, match in pred_slice.iterrows():
            home_id = match["home_player_id"]
            away_id = match["away_player_id"]
            match_id = match["match_id"]

            format_params = {
                "best_of_sets": match.get("best_of_sets", 0),
                "best_of_legs": match.get("best_of_legs", 7),
                "is_set_format": match.get("is_set_format", False),
            }

            pred = temp_model.predict(home_id, away_id, MODUS_LEAGUE_ID, format_params)
            predicted = pred.lambda_total

            actual = match_180s_stats.get(match_id, match_180s_visits.get(match_id, None))

            if actual is not None:
                window_preds.append({
                    "match_id": match_id,
                    "predicted": predicted,
                    "actual": actual,
                    "expected_visits": pred.expected_visits,
                    "is_set_format": bool(match.get("is_set_format", False)),
                })

        oos_predictions.extend(window_preds)
        logger.info(f"  Window {current_end}/{n}: trained on {current_end}, predicted {len(window_preds)}")
        current_end = pred_end

    oos_df = pd.DataFrame(oos_predictions)
    logger.info(f"  Total out-of-sample predictions: {len(oos_df)}")

    # ========================================================================
    # FINAL MODEL: Train on ALL MODUS data
    # ========================================================================
    logger.info("\nPhase 2: Training final MODUS model on ALL data...")

    model = VisitLevel180sModel(
        prior_strength=prior_strength,
        min_visits_for_direct=min_visits_for_direct,
    )
    model.fit(visits_df, matches_df, stats_df)
    logger.info(f"MODUS players with rates: {len(model.player_stats)}")
    logger.info(f"MODUS population rate: {model.population_rate:.4f}")

    if "match_date" in matches_df.columns:
        logger.info(f"Training data: {matches_df['match_date'].min()} to {matches_df['match_date'].max()}")

    # ========================================================================
    # CALIBRATION + OVERDISPERSION (same approach as global model)
    # ========================================================================
    if len(oos_df) >= 30:
        logger.info("\nPhase 3: Calibration + overdispersion estimation...")

        preds = oos_df["predicted"].values
        actuals = oos_df["actual"].values
        pre_bias = np.mean(preds) - np.mean(actuals)
        pre_rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        logger.info(f"  OOS: Bias={pre_bias:.3f}, RMSE={pre_rmse:.3f}")

        # Disable all calibration (same reasoning as global model)
        model.bias_slope_leg = 0.0
        model.bias_intercept_leg = 0.0
        model.bias_slope_set = 0.0
        model.bias_intercept_set = 0.0
        model.bias_correction = 0.0
        model.format_bias_corrections = {}
        logger.info(f"  All calibration DISABLED")

        # MLE overdispersion on final model predictions vs actuals
        from scipy import stats as sp_stats

        all_stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
        if not all_stat_180s.empty:
            all_stat_180s["value_num"] = pd.to_numeric(all_stat_180s["value"], errors="coerce").fillna(0)
            all_match_180s = all_stat_180s.groupby("match_id")["value_num"].sum().to_dict()
        else:
            all_match_180s = {}
        if not visits_df.empty:
            visit_180s_lookup = visits_df.groupby("match_id")["is_180"].sum().to_dict()
        else:
            visit_180s_lookup = {}

        mle_preds = []
        mle_actuals = []
        valid_matches = matches_df[
            ~((matches_df["best_of_sets"] == 0) & (matches_df["best_of_legs"] == 0))
        ]
        for _, row in valid_matches.iterrows():
            h, a = int(row["home_player_id"]), int(row["away_player_id"])
            if h == 0 or a == 0:
                continue
            mid = row["match_id"]
            actual = all_match_180s.get(mid, visit_180s_lookup.get(mid, None))
            if actual is None:
                continue
            fmt = {
                "best_of_sets": int(row["best_of_sets"]),
                "best_of_legs": int(row["best_of_legs"]),
                "is_set_format": bool(row["is_set_format"]),
            }
            pred = model.predict(h, a, MODUS_LEAGUE_ID, fmt)
            if pred.lambda_total < 0.01:
                continue
            mle_preds.append(pred.lambda_total)
            mle_actuals.append(int(actual))

        def _nb_loglik(preds_arr, actuals_arr, phi):
            ll = 0.0
            for mu, k in zip(preds_arr, actuals_arr):
                if phi <= 1.0:
                    ll += sp_stats.poisson.logpmf(k, mu)
                else:
                    n_param = mu / (phi - 1.0)
                    p_param = 1.0 / phi
                    ll += sp_stats.nbinom.logpmf(k, n_param, p_param)
            return ll

        if len(mle_preds) >= 20:
            p_arr = np.array(mle_preds)
            a_arr = np.array(mle_actuals)

            # Coarse grid
            best_phi, best_ll = 1.0, _nb_loglik(p_arr, a_arr, 1.0)
            for phi in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
                ll = _nb_loglik(p_arr, a_arr, phi)
                if ll > best_ll:
                    best_ll = ll
                    best_phi = phi

            # Fine grid
            lo = max(1.1, best_phi - 0.8)
            hi = best_phi + 0.8
            for phi in np.arange(lo, hi, 0.1):
                ll = _nb_loglik(p_arr, a_arr, phi)
                if ll > best_ll:
                    best_ll = ll
                    best_phi = round(float(phi), 1)

            poisson_ll = _nb_loglik(p_arr, a_arr, 1.0)
            pct_gain = (best_ll - poisson_ll) / abs(poisson_ll) * 100

            # MODUS is all one format (leg), store as both global and leg
            model.overdispersion = best_phi
            model.overdispersion_by_format = {"leg": best_phi}
            logger.info(f"  Overdispersion MLE (n={len(p_arr)}): phi={best_phi:.1f} "
                       f"(NB vs Poisson: {pct_gain:+.1f}%)")

            # Final model bias check
            final_bias = np.mean(p_arr) - np.mean(a_arr)
            final_rmse = np.sqrt(np.mean((p_arr - a_arr) ** 2))
            logger.info(f"  Final model: Bias={final_bias:.3f}, RMSE={final_rmse:.3f}")
            logger.info(f"  Mean predicted={np.mean(p_arr):.2f}, Mean actual={np.mean(a_arr):.2f}")

        metrics = {
            "n_oos": len(oos_df),
            "oos_bias": float(pre_bias),
            "oos_rmse": float(pre_rmse),
            "overdispersion": float(model.overdispersion),
            "population_rate": float(model.population_rate),
            "n_players": len(model.player_stats),
        }
    else:
        logger.warning(f"Skipping calibration: only {len(oos_df)} OOS samples (need 30+)")
        metrics = {}

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info(f"\nMODUS model saved to {output_path}")

    return metrics


def train_glicko2_system(
    data_dir: str = "data/processed",
    output_path: str = "data/models/glicko2_system.json",
    rd_decay_per_day: float = 0.4,  # Tuned value (was 0.5)
    use_match_weighting: bool = True,
    calibrate: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Train the Glicko-2 rating system with optional probability calibration.

    Uses an expanding window approach:
    1. Collect out-of-sample predictions incrementally (for calibration)
    2. Train final model on ALL data (so all players have current ratings)
    3. Fit calibrator on the out-of-sample predictions

    This ensures recent matches (like MODUS 2026) are included in training.

    Args:
        data_dir: Path to processed data
        output_path: Path to save model
        rd_decay_per_day: Rating deviation increase per day inactive
        use_match_weighting: Weight matches by length
        calibrate: Whether to fit isotonic probability calibration
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Training Glicko-2 System")
    logger.info("=" * 60)

    # Load data
    store = ParquetStore(data_dir)
    matches_df = store.get_matches()
    stats_df = store.get_match_stats()

    logger.info(f"Loaded {len(matches_df)} matches")
    logger.info(f"Loaded {len(stats_df)} stats")

    # Ensure correct types
    matches_df = _ensure_match_types(matches_df)
    stats_df = _ensure_stats_types(stats_df)

    # Filter out invalid matches (player_id = 0 from NaN conversion)
    valid_matches = matches_df[
        (matches_df["home_player_id"] > 0) &
        (matches_df["away_player_id"] > 0)
    ]
    logger.info(f"Valid matches (non-zero player IDs): {len(valid_matches)}")

    # Sort by date
    if "match_date" in valid_matches.columns:
        valid_matches = valid_matches.sort_values("match_date").reset_index(drop=True)

    n = len(valid_matches)

    # ========================================================================
    # EXPANDING WINDOW: Collect out-of-sample predictions for calibration
    # ========================================================================
    oos_predictions = []  # Out-of-sample predictions for calibration

    if calibrate:
        logger.info("\nPhase 1: Collecting out-of-sample predictions (expanding window)...")

        # Start with first 50%, predict on next 10%, expand, repeat
        window_start = int(n * 0.50)
        window_step = int(n * 0.10)

        current_end = window_start
        while current_end < n:
            # Determine prediction window
            pred_start = current_end
            pred_end = min(current_end + window_step, n)

            # Train temporary model on data up to current_end
            temp_config = Glicko2Config(
                rd_decay_per_day=rd_decay_per_day,
                use_match_length_weighting=use_match_weighting,
            )
            temp_glicko = Glicko2System(temp_config)

            train_slice = valid_matches.iloc[:current_end]
            train_stats = stats_df[stats_df["match_id"].isin(train_slice["match_id"])]
            temp_glicko.train(train_slice, train_stats)

            # Make predictions on the next window (out-of-sample)
            pred_slice = valid_matches.iloc[pred_start:pred_end]
            window_preds = _make_glicko_predictions(temp_glicko, pred_slice, logger, use_calibrated=False)
            oos_predictions.append(window_preds)

            logger.info(f"  Window {current_end}/{n}: trained on {current_end}, predicted {len(pred_slice)}")
            current_end = pred_end

        # Combine all out-of-sample predictions
        oos_df = pd.concat(oos_predictions, ignore_index=True)
        logger.info(f"  Total out-of-sample predictions: {len(oos_df)}")

    # ========================================================================
    # FINAL MODEL: Train on ALL data
    # ========================================================================
    logger.info("\nPhase 2: Training final model on ALL data...")

    config = Glicko2Config(
        rd_decay_per_day=rd_decay_per_day,
        use_match_length_weighting=use_match_weighting,
    )
    glicko = Glicko2System(config)

    history = glicko.train(valid_matches, stats_df)
    logger.info(f"Processed {len(history)} matches")
    logger.info(f"Total players: {len(glicko.ratings)}")

    # Show date range
    if "match_date" in valid_matches.columns:
        logger.info(f"Training data: {valid_matches['match_date'].min()} to {valid_matches['match_date'].max()}")

    # ========================================================================
    # CALIBRATION: Fit on out-of-sample predictions
    # ========================================================================
    if calibrate and len(oos_df) >= 100:
        logger.info("\nPhase 3: Fitting probability calibrator on out-of-sample predictions...")
        cal_stats = glicko.calibrate_probabilities(oos_df, use_history=False)
        logger.info("Calibration results:")
        for key, value in cal_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        # Evaluate calibration on the OOS predictions
        logger.info("\nOut-of-sample metrics (before calibration):")
        _evaluate_glicko_predictions(oos_df, logger, prefix="oos_raw_")

        # Apply calibration to OOS predictions for comparison
        oos_df_calibrated = oos_df.copy()
        oos_df_calibrated["home_expected"] = glicko.probability_calibrator.predict(
            oos_df["home_expected"].values
        )
        logger.info("\nOut-of-sample metrics (after calibration):")
        metrics = _evaluate_glicko_predictions(oos_df_calibrated, logger, prefix="oos_cal_")
    elif calibrate:
        logger.warning(f"Skipping calibration: only {len(oos_df) if calibrate else 0} OOS samples (need 100+)")
        metrics = {}
    else:
        metrics = {}

    # Show top players
    top_df = glicko.get_top_players(10)
    _display_leaderboard(store, top_df, logger, show_rd=True)

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    glicko.save(output_path)
    logger.info(f"\nGlicko-2 system saved to {output_path}")

    return metrics


def _make_glicko_predictions(
    glicko: Glicko2System,
    matches_df: pd.DataFrame,
    logger: logging.Logger,
    use_calibrated: bool = False,
) -> pd.DataFrame:
    """Make win probability predictions for a set of matches."""
    predictions = []

    for _, match in matches_df.iterrows():
        home_id = int(match["home_player_id"])
        away_id = int(match["away_player_id"])
        match_date = match.get("match_date")
        league_id = match.get("league_id", 0)
        is_tv = glicko.is_tv_event(league_id)

        # Parse match date
        if match_date and isinstance(match_date, str):
            from datetime import date
            try:
                match_date = date.fromisoformat(match_date)
            except:
                match_date = None

        # Get prediction
        if use_calibrated:
            p_home, p_away, meta = glicko.calibrated_win_probability(
                home_id, away_id, match_date, is_tv
            )
        else:
            p_home, p_away, meta = glicko.win_probability(
                home_id, away_id, match_date, is_tv
            )

        # Determine actual result
        is_set_format = match.get("is_set_format", False)
        home_legs = match.get("home_legs", 0)
        away_legs = match.get("away_legs", 0)
        home_sets = match.get("home_sets", 0)
        away_sets = match.get("away_sets", 0)

        # Fallback: if legs are both 0, use sets (handles MODUS data)
        if home_legs == 0 and away_legs == 0:
            home_won = home_sets > away_sets
        elif is_set_format:
            home_won = home_sets > away_sets
        else:
            home_won = home_legs > away_legs

        predictions.append({
            "match_id": match["match_id"],
            "home_expected": p_home,
            "home_result": 1 if home_won else 0,
            "rd_a": meta.get("rd_a", 0),
            "rd_b": meta.get("rd_b", 0),
        })

    return pd.DataFrame(predictions)


def _evaluate_glicko_predictions(
    predictions_df: pd.DataFrame,
    logger: logging.Logger,
    prefix: str = "",
) -> dict:
    """Evaluate Glicko predictions."""
    if predictions_df.empty:
        return {"error": "no_predictions"}

    predicted = predictions_df["home_expected"].values
    actual = predictions_df["home_result"].values

    # Filter NaN
    valid = ~(np.isnan(predicted) | np.isnan(actual))
    predicted = predicted[valid]
    actual = actual[valid]

    if len(predicted) == 0:
        return {"error": "no_valid_predictions"}

    # Log loss
    eps = 1e-15
    predicted_clip = np.clip(predicted, eps, 1 - eps)
    log_loss = -np.mean(
        actual * np.log(predicted_clip) +
        (1 - actual) * np.log(1 - predicted_clip)
    )

    # Brier score
    brier = np.mean((predicted - actual) ** 2)

    # Accuracy
    accuracy = np.mean((predicted > 0.5).astype(int) == actual)

    # Calibration error (ECE)
    ece = 0.0
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (predicted > bin_boundaries[i]) & (predicted <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = predicted[in_bin].mean()
            avg_accuracy = actual[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

    metrics = {
        f"{prefix}log_loss": float(log_loss),
        f"{prefix}brier_score": float(brier),
        f"{prefix}accuracy": float(accuracy),
        f"{prefix}calibration_error": float(ece),
        f"{prefix}n_matches": len(predicted),
    }

    logger.info(f"{'Raw ' if prefix else ''}Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return metrics


def _ensure_match_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for matches DataFrame."""
    if df.empty:
        return df

    int_cols = ["match_id", "league_id", "season_id", "home_player_id", "away_player_id",
                "home_sets", "away_sets", "home_legs", "away_legs",
                "best_of_sets", "best_of_legs", "start_score"]
    bool_cols = ["is_set_format", "has_visit_data"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def _ensure_visit_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for visits DataFrame."""
    if df.empty:
        return df

    int_cols = ["match_id", "league_id", "season_id", "set_no", "leg_no",
                "visit_index", "player_id", "score", "attempts"]
    bool_cols = ["is_180", "is_140_plus", "is_100_plus", "is_checkout"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def _ensure_stats_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for stats DataFrame."""
    if df.empty:
        return df

    int_cols = ["match_id", "player_id", "stat_field_id"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def _display_leaderboard(
    store: ParquetStore,
    top_df: pd.DataFrame,
    logger: logging.Logger,
    show_rd: bool = False,
    skill_col: str = "rating",
) -> None:
    """Display player leaderboard with names."""
    if top_df.empty:
        return

    logger.info("\nTop 10 players:")
    players_df = store.get_players()

    if not players_df.empty:
        top_df = top_df.merge(
            players_df[["player_id", "name"]],
            on="player_id",
            how="left",
        )

    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        name = row.get("name")
        if pd.isna(name) or not name:
            name = f"Player {row['player_id']}"

        rating = row.get(skill_col, row.get("rating", 0))
        matches = row.get("matches", row.get("match_count", 0))

        if show_rd and "rd" in row:
            logger.info(f"  {i}. {name}: {rating:.1f} (±{row['rd']:.0f}) ({matches} matches)")
        else:
            logger.info(f"  {i}. {name}: {rating:.1f} ({matches} matches)")


def main():
    parser = argparse.ArgumentParser(
        description="Train darts prediction models"
    )
    parser.add_argument(
        "--model",
        choices=["180s-visit", "180s-modus", "glicko2", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Data directory (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="Model output directory (default: data/models)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.model in ["180s-visit", "all"]:
        train_visit_level_180s_model(
            data_dir=args.data_dir,
            output_path=str(output_dir / "model_180s_visit_level.json"),
            verbose=args.verbose,
        )

    if args.model in ["180s-modus", "all"]:
        train_modus_180s_model(
            data_dir=args.data_dir,
            output_path=str(output_dir / "model_180s_modus.json"),
            verbose=args.verbose,
        )

    if args.model in ["glicko2", "all"]:
        train_glicko2_system(
            data_dir=args.data_dir,
            output_path=str(output_dir / "glicko2_system.json"),
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
