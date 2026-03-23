#!/usr/bin/env python3
"""
Test script to compare different Glicko-2 configurations.

Compares:
1. Baseline (current settings)
2. With form adjustment enabled
3. With different margin weights
4. Combinations

Uses out-of-sample evaluation with expanding window.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.models.glicko2_system import Glicko2System, Glicko2Config
from src.storage import ParquetStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def evaluate_config(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    config: Glicko2Config,
    config_name: str,
    train_ratio: float = 0.5,
) -> dict:
    """
    Evaluate a Glicko-2 configuration using expanding window OOS evaluation.

    Returns metrics dict.
    """
    # Sort by date
    matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    n_matches = len(matches_df)
    train_end = int(n_matches * train_ratio)

    # Initialize system
    system = Glicko2System(config=config)

    # Train on first half
    train_matches = matches_df.iloc[:train_end]
    system.train(train_matches, stats_df)

    # Evaluate on second half with expanding window
    test_matches = matches_df.iloc[train_end:]

    predictions = []
    actuals = []

    for idx, match in test_matches.iterrows():
        home_id = match["home_player_id"]
        away_id = match["away_player_id"]

        # Skip if missing player IDs
        if pd.isna(home_id) or pd.isna(away_id):
            continue

        home_id = int(home_id)
        away_id = int(away_id)

        # Get prediction BEFORE updating with this match
        try:
            p_home, p_away, meta = system.calibrated_win_probability(
                home_id, away_id,
                match_date=match.get("match_date"),
                use_form=config.use_form_adjustment,
            )
        except Exception as e:
            continue

        # Determine actual result
        is_set_format = match.get("is_set_format", False)
        home_legs = match.get("home_legs", 0)
        away_legs = match.get("away_legs", 0)
        home_sets = match.get("home_sets", 0)
        away_sets = match.get("away_sets", 0)

        if home_legs == 0 and away_legs == 0:
            actual = 1 if home_sets > away_sets else 0
        elif is_set_format:
            actual = 1 if home_sets > away_sets else 0
        else:
            actual = 1 if home_legs > away_legs else 0

        predictions.append(p_home)
        actuals.append(actual)

        # Now update system with this match
        try:
            system.process_match(match)
        except Exception:
            pass

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Accuracy
    accuracy = np.mean((predictions > 0.5).astype(int) == actuals)

    # Brier score
    brier = np.mean((predictions - actuals) ** 2)

    # Log loss
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) +
        (1 - actuals) * np.log(1 - predictions_clipped)
    )

    # Calibration error (binned)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = predictions[in_bin].mean()
            avg_accuracy = actuals[in_bin].mean()
            ece += (in_bin.sum() / len(predictions)) * abs(avg_confidence - avg_accuracy)

    return {
        "config": config_name,
        "n_test": len(predictions),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "calibration_error": ece,
    }


def main():
    logger.info("=" * 60)
    logger.info("GLICKO-2 CONFIGURATION COMPARISON TEST")
    logger.info("=" * 60)

    # Load data
    store = ParquetStore("data/processed")
    matches = store.get_matches()
    stats = store.get_match_stats()

    # Filter to matches with valid dates and player IDs
    matches = matches[
        matches["match_date"].notna() &
        matches["home_player_id"].notna() &
        matches["away_player_id"].notna()
    ].copy()

    logger.info(f"Total matches: {len(matches)}")

    # Define configurations to test
    configs = []

    # 1. Baseline (current defaults)
    baseline_config = Glicko2Config()
    baseline_config.use_form_adjustment = False  # Disable form for baseline
    configs.append((baseline_config, "Baseline (no form)"))

    # 2. With form adjustment (default weight 0.5)
    form_config = Glicko2Config()
    form_config.use_form_adjustment = True
    form_config.form_weight = 0.5
    configs.append((form_config, "Form (weight=0.5)"))

    # 3. Form with lower weight
    form_low_config = Glicko2Config()
    form_low_config.use_form_adjustment = True
    form_low_config.form_weight = 0.3
    configs.append((form_low_config, "Form (weight=0.3)"))

    # 4. Different margin weights
    margin_low_config = Glicko2Config()
    margin_low_config.use_form_adjustment = False
    margin_low_config.margin_weight = 0.3
    configs.append((margin_low_config, "Margin weight=0.3"))

    margin_high_config = Glicko2Config()
    margin_high_config.use_form_adjustment = False
    margin_high_config.margin_weight = 0.7
    configs.append((margin_high_config, "Margin weight=0.7"))

    # 5. Combined: form + optimized margin
    combined_config = Glicko2Config()
    combined_config.use_form_adjustment = True
    combined_config.form_weight = 0.3
    combined_config.margin_weight = 0.5
    configs.append((combined_config, "Form(0.3) + Margin(0.5)"))

    # Run evaluations
    results = []
    for config, name in configs:
        logger.info(f"\nTesting: {name}")
        result = evaluate_config(matches, stats, config, name)
        results.append(result)
        logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  Brier:    {result['brier']:.4f}")
        logger.info(f"  Log Loss: {result['log_loss']:.4f}")
        logger.info(f"  ECE:      {result['calibration_error']:.4f}")

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False)

    logger.info(f"\n{'Config':<30} {'Accuracy':>10} {'Brier':>10} {'Log Loss':>10} {'ECE':>10}")
    logger.info("-" * 72)
    for _, row in results_df.iterrows():
        logger.info(
            f"{row['config']:<30} {row['accuracy']:>10.4f} {row['brier']:>10.4f} "
            f"{row['log_loss']:>10.4f} {row['calibration_error']:>10.4f}"
        )

    # Highlight best
    best_acc = results_df.iloc[0]
    logger.info(f"\nBest by accuracy: {best_acc['config']} ({best_acc['accuracy']:.4f})")

    best_brier = results_df.sort_values("brier").iloc[0]
    logger.info(f"Best by Brier: {best_brier['config']} ({best_brier['brier']:.4f})")


if __name__ == "__main__":
    main()
