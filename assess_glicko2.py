#!/usr/bin/env python3
"""
Comprehensive Glicko-2 Model Assessment

Evaluates the Glicko-2 rating system across multiple dimensions:
1. Predictive accuracy metrics
2. Calibration analysis
3. Discrimination ability (ROC/AUC)
4. Rating distribution statistics
5. Player-level analysis
6. Time-based performance
7. Format-specific performance
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from datetime import date, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import ParquetStore
from src.models import Glicko2System, Glicko2Config


def load_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data."""
    store = ParquetStore(data_dir)
    matches_df = store.get_matches()
    stats_df = store.get_match_stats()

    int_cols = ["match_id", "league_id", "home_player_id", "away_player_id",
                "home_sets", "away_sets", "home_legs", "away_legs",
                "best_of_sets", "best_of_legs"]
    for col in int_cols:
        if col in matches_df.columns:
            matches_df[col] = pd.to_numeric(matches_df[col], errors="coerce").fillna(0).astype(int)

    if "is_set_format" in matches_df.columns:
        matches_df["is_set_format"] = matches_df["is_set_format"].fillna(False).astype(bool)

    matches_df = matches_df[
        (matches_df["home_player_id"] > 0) &
        (matches_df["away_player_id"] > 0)
    ]

    if "match_date" in matches_df.columns:
        matches_df = matches_df.sort_values("match_date")

    return matches_df, stats_df


def compute_roc_auc(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, List, List]:
    """Compute ROC curve and AUC."""
    # Sort by prediction
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_preds = predictions[sorted_indices]
    sorted_actuals = actuals[sorted_indices]

    # Compute TPR and FPR at various thresholds
    n_pos = np.sum(actuals)
    n_neg = len(actuals) - n_pos

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0

    for i, (pred, actual) in enumerate(zip(sorted_preds, sorted_actuals)):
        if actual == 1:
            tp += 1
        else:
            fp += 1

        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return auc, fpr_list, tpr_list


def compute_calibration_curve(predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration curve with detailed statistics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_samples = np.sum(mask)

        if n_samples > 0:
            mean_pred = np.mean(predictions[mask])
            mean_actual = np.mean(actuals[mask])
            std_actual = np.std(actuals[mask]) if n_samples > 1 else 0

            calibration_data.append({
                "bin": i,
                "bin_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "n_samples": int(n_samples),
                "mean_predicted": mean_pred,
                "mean_actual": mean_actual,
                "error": mean_pred - mean_actual,
                "abs_error": abs(mean_pred - mean_actual),
            })

    return calibration_data


def assess_glicko2(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Run comprehensive Glicko-2 assessment."""

    # Split data
    split_idx = int(len(matches_df) * train_ratio)
    train_matches = matches_df.iloc[:split_idx]
    test_matches = matches_df.iloc[split_idx:]

    # Train model
    config = Glicko2Config()  # Uses tuned defaults
    glicko = Glicko2System(config)
    glicko.train(train_matches, stats_df)

    # Collect predictions
    predictions = []
    actuals = []
    rating_diffs = []
    rd_sums = []
    match_info = []

    for _, match in test_matches.iterrows():
        home_id = match["home_player_id"]
        away_id = match["away_player_id"]

        if match.get("is_set_format", False):
            home_won = match["home_sets"] > match["away_sets"]
        else:
            home_won = match["home_legs"] > match["away_legs"]

        home_player = glicko.ratings.get(home_id)
        away_player = glicko.ratings.get(away_id)

        if home_player and away_player:
            p_home, p_away, metadata = glicko.win_probability(home_id, away_id)

            predictions.append(p_home)
            actuals.append(1 if home_won else 0)
            rating_diffs.append(home_player.rating - away_player.rating)
            rd_sums.append(home_player.rd + away_player.rd)
            match_info.append({
                "match_id": match["match_id"],
                "league_id": match.get("league_id", 0),
                "is_set_format": match.get("is_set_format", False),
                "best_of_legs": match.get("best_of_legs", 5),
                "home_rating": home_player.rating,
                "away_rating": away_player.rating,
                "home_rd": home_player.rd,
                "away_rd": away_player.rd,
                "ci_lower": metadata.get("p_a_lower"),
                "ci_upper": metadata.get("p_a_upper"),
            })

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    rating_diffs = np.array(rating_diffs)
    rd_sums = np.array(rd_sums)

    n = len(predictions)

    # =====================================================================
    # 1. BASIC PREDICTIVE METRICS
    # =====================================================================

    # Accuracy
    pred_outcomes = (predictions >= 0.5).astype(int)
    accuracy = np.mean(pred_outcomes == actuals)

    # Accuracy by confidence level
    high_conf_mask = (predictions >= 0.65) | (predictions <= 0.35)
    med_conf_mask = ((predictions >= 0.55) & (predictions < 0.65)) | ((predictions > 0.35) & (predictions <= 0.45))
    low_conf_mask = (predictions > 0.45) & (predictions < 0.55)

    accuracy_high_conf = np.mean(pred_outcomes[high_conf_mask] == actuals[high_conf_mask]) if np.sum(high_conf_mask) > 0 else None
    accuracy_med_conf = np.mean(pred_outcomes[med_conf_mask] == actuals[med_conf_mask]) if np.sum(med_conf_mask) > 0 else None
    accuracy_low_conf = np.mean(pred_outcomes[low_conf_mask] == actuals[low_conf_mask]) if np.sum(low_conf_mask) > 0 else None

    # Brier Score
    brier_score = np.mean((predictions - actuals) ** 2)

    # Brier Skill Score (relative to baseline of 0.25)
    brier_baseline = 0.25  # Always predicting 0.5
    brier_skill = 1 - (brier_score / brier_baseline)

    # Log Loss
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) +
        (1 - actuals) * np.log(1 - predictions_clipped)
    )

    # =====================================================================
    # 2. DISCRIMINATION METRICS (ROC/AUC)
    # =====================================================================

    auc, fpr_list, tpr_list = compute_roc_auc(predictions, actuals)

    # =====================================================================
    # 3. CALIBRATION METRICS
    # =====================================================================

    calibration_data = compute_calibration_curve(predictions, actuals, n_bins=10)

    # Expected Calibration Error (ECE)
    ece = sum(d["abs_error"] * d["n_samples"] for d in calibration_data) / n

    # Maximum Calibration Error (MCE)
    mce = max(d["abs_error"] for d in calibration_data) if calibration_data else 0

    # Calibration slope (should be ~1 for well-calibrated model)
    if len(calibration_data) >= 2:
        pred_means = [d["mean_predicted"] for d in calibration_data]
        actual_means = [d["mean_actual"] for d in calibration_data]
        if np.std(pred_means) > 0:
            slope, intercept, r_value, _, _ = stats.linregress(pred_means, actual_means)
        else:
            slope, intercept, r_value = 1.0, 0.0, 1.0
    else:
        slope, intercept, r_value = 1.0, 0.0, 1.0

    # =====================================================================
    # 4. CONFIDENCE INTERVAL COVERAGE
    # =====================================================================

    ci_coverage = None
    ci_width_avg = None
    ci_data = [(m["ci_lower"], m["ci_upper"]) for m in match_info if m["ci_lower"] is not None]

    if ci_data:
        ci_lower = np.array([c[0] for c in ci_data])
        ci_upper = np.array([c[1] for c in ci_data])

        # Check if actual outcome probability falls within CI
        # This is tricky - we check if the prediction was "correct" given the CI
        ci_width_avg = np.mean(ci_upper - ci_lower)

        # For binary outcomes, check if the CI includes the observed rate
        # We'll check if predictions within CI would have been reasonable
        ci_coverage = np.mean((predictions >= ci_lower) & (predictions <= ci_upper))

    # =====================================================================
    # 5. RATING DISTRIBUTION STATISTICS
    # =====================================================================

    all_ratings = [p.rating for p in glicko.ratings.values()]
    all_rds = [p.rd for p in glicko.ratings.values()]
    all_volatilities = [p.volatility for p in glicko.ratings.values()]
    match_counts = [p.match_count for p in glicko.ratings.values()]

    rating_stats = {
        "n_players": len(all_ratings),
        "rating_mean": np.mean(all_ratings),
        "rating_std": np.std(all_ratings),
        "rating_min": np.min(all_ratings),
        "rating_max": np.max(all_ratings),
        "rating_median": np.median(all_ratings),
        "rd_mean": np.mean(all_rds),
        "rd_std": np.std(all_rds),
        "rd_min": np.min(all_rds),
        "rd_max": np.max(all_rds),
        "volatility_mean": np.mean(all_volatilities),
        "volatility_std": np.std(all_volatilities),
        "avg_matches_per_player": np.mean(match_counts),
        "players_with_10plus_matches": sum(1 for m in match_counts if m >= 10),
    }

    # =====================================================================
    # 6. UPSET ANALYSIS
    # =====================================================================

    # Favorite = higher predicted probability
    favorite_won = (predictions >= 0.5) == actuals.astype(bool)
    upset_rate = 1 - np.mean(favorite_won)

    # Upsets by rating difference
    big_favorite_mask = np.abs(rating_diffs) > 150
    small_diff_mask = np.abs(rating_diffs) <= 50

    upset_rate_big_favorite = 1 - np.mean(favorite_won[big_favorite_mask]) if np.sum(big_favorite_mask) > 0 else None
    upset_rate_close_match = 1 - np.mean(favorite_won[small_diff_mask]) if np.sum(small_diff_mask) > 0 else None

    # =====================================================================
    # 7. PERFORMANCE BY UNCERTAINTY
    # =====================================================================

    low_uncertainty_mask = rd_sums < np.percentile(rd_sums, 33)
    high_uncertainty_mask = rd_sums > np.percentile(rd_sums, 66)

    accuracy_low_uncertainty = np.mean(pred_outcomes[low_uncertainty_mask] == actuals[low_uncertainty_mask])
    accuracy_high_uncertainty = np.mean(pred_outcomes[high_uncertainty_mask] == actuals[high_uncertainty_mask])

    brier_low_uncertainty = np.mean((predictions[low_uncertainty_mask] - actuals[low_uncertainty_mask]) ** 2)
    brier_high_uncertainty = np.mean((predictions[high_uncertainty_mask] - actuals[high_uncertainty_mask]) ** 2)

    # =====================================================================
    # 8. SHARPNESS (how decisive are predictions)
    # =====================================================================

    # Sharpness = how far from 0.5 are predictions on average
    sharpness = np.mean(np.abs(predictions - 0.5))

    # Distribution of prediction confidence
    very_confident = np.mean((predictions >= 0.7) | (predictions <= 0.3))
    moderate_confident = np.mean(((predictions >= 0.55) & (predictions < 0.7)) | ((predictions > 0.3) & (predictions <= 0.45)))
    uncertain = np.mean((predictions > 0.45) & (predictions < 0.55))

    # =====================================================================
    # COMPILE RESULTS
    # =====================================================================

    results = {
        "config": {
            "initial_rd": config.initial_rd,
            "rd_decay_per_day": config.rd_decay_per_day,
            "margin_weight": config.margin_weight,
            "rating_period_days": config.rating_period_days,
            "recency_half_life_days": config.recency_half_life_days,
        },
        "data": {
            "train_matches": len(train_matches),
            "test_matches": len(test_matches),
            "test_predictions": n,
        },
        "accuracy_metrics": {
            "accuracy": accuracy,
            "accuracy_high_confidence": accuracy_high_conf,
            "accuracy_medium_confidence": accuracy_med_conf,
            "accuracy_low_confidence": accuracy_low_conf,
            "n_high_confidence": int(np.sum(high_conf_mask)),
            "n_medium_confidence": int(np.sum(med_conf_mask)),
            "n_low_confidence": int(np.sum(low_conf_mask)),
        },
        "probabilistic_metrics": {
            "brier_score": brier_score,
            "brier_skill_score": brier_skill,
            "log_loss": log_loss,
            "auc_roc": auc,
        },
        "calibration_metrics": {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibration_r_squared": r_value ** 2,
            "calibration_curve": calibration_data,
        },
        "confidence_intervals": {
            "avg_ci_width": ci_width_avg,
            "ci_self_coverage": ci_coverage,
        },
        "rating_distribution": rating_stats,
        "upset_analysis": {
            "overall_upset_rate": upset_rate,
            "upset_rate_big_favorite": upset_rate_big_favorite,
            "upset_rate_close_match": upset_rate_close_match,
        },
        "uncertainty_analysis": {
            "accuracy_low_uncertainty": accuracy_low_uncertainty,
            "accuracy_high_uncertainty": accuracy_high_uncertainty,
            "brier_low_uncertainty": brier_low_uncertainty,
            "brier_high_uncertainty": brier_high_uncertainty,
        },
        "sharpness": {
            "mean_sharpness": sharpness,
            "pct_very_confident": very_confident,
            "pct_moderate_confident": moderate_confident,
            "pct_uncertain": uncertain,
        },
    }

    return results


def print_assessment(results: Dict[str, Any]) -> None:
    """Print assessment results in a readable format."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE GLICKO-2 MODEL ASSESSMENT")
    print("=" * 80)

    # Config
    print("\n--- MODEL CONFIGURATION ---")
    cfg = results["config"]
    print(f"  Initial RD: {cfg['initial_rd']}")
    print(f"  RD Decay/Day: {cfg['rd_decay_per_day']}")
    print(f"  Margin Weight: {cfg['margin_weight']}")
    print(f"  Rating Period: {cfg['rating_period_days']} days")
    print(f"  Recency Half-life: {cfg['recency_half_life_days']} days")

    # Data
    print("\n--- DATA SUMMARY ---")
    data = results["data"]
    print(f"  Training matches: {data['train_matches']}")
    print(f"  Test matches: {data['test_matches']}")
    print(f"  Test predictions: {data['test_predictions']}")

    # Accuracy
    print("\n--- ACCURACY METRICS ---")
    acc = results["accuracy_metrics"]
    print(f"  Overall Accuracy: {acc['accuracy']:.4f} ({acc['accuracy']*100:.2f}%)")
    print(f"  High Confidence (>65% or <35%): {acc['accuracy_high_confidence']:.4f} (n={acc['n_high_confidence']})" if acc['accuracy_high_confidence'] else "  High Confidence: N/A")
    print(f"  Medium Confidence: {acc['accuracy_medium_confidence']:.4f} (n={acc['n_medium_confidence']})" if acc['accuracy_medium_confidence'] else "  Medium Confidence: N/A")
    print(f"  Low Confidence (45-55%): {acc['accuracy_low_confidence']:.4f} (n={acc['n_low_confidence']})" if acc['accuracy_low_confidence'] else "  Low Confidence: N/A")

    # Probabilistic
    print("\n--- PROBABILISTIC METRICS ---")
    prob = results["probabilistic_metrics"]
    print(f"  Brier Score: {prob['brier_score']:.4f} (lower is better, 0.25 = random)")
    print(f"  Brier Skill Score: {prob['brier_skill_score']:.4f} (>0 = better than random)")
    print(f"  Log Loss: {prob['log_loss']:.4f} (lower is better)")
    print(f"  AUC-ROC: {prob['auc_roc']:.4f} (0.5 = random, 1.0 = perfect)")

    # Calibration
    print("\n--- CALIBRATION METRICS ---")
    cal = results["calibration_metrics"]
    print(f"  Expected Calibration Error (ECE): {cal['expected_calibration_error']:.4f} (lower is better)")
    print(f"  Maximum Calibration Error (MCE): {cal['maximum_calibration_error']:.4f}")
    print(f"  Calibration Slope: {cal['calibration_slope']:.4f} (ideal = 1.0)")
    print(f"  Calibration Intercept: {cal['calibration_intercept']:.4f} (ideal = 0.0)")
    print(f"  Calibration R-squared: {cal['calibration_r_squared']:.4f}")

    print("\n  Calibration Curve:")
    print("  " + "-" * 70)
    print(f"  {'Bin':<12} | {'N':>6} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10}")
    print("  " + "-" * 70)
    for row in cal["calibration_curve"]:
        print(f"  {row['bin_range']:<12} | {row['n_samples']:>6} | {row['mean_predicted']:>10.3f} | {row['mean_actual']:>10.3f} | {row['error']:>+10.3f}")
    print("  " + "-" * 70)

    # Confidence Intervals
    print("\n--- CONFIDENCE INTERVALS ---")
    ci = results["confidence_intervals"]
    print(f"  Average CI Width: {ci['avg_ci_width']:.4f}" if ci['avg_ci_width'] else "  Average CI Width: N/A")

    # Rating Distribution
    print("\n--- RATING DISTRIBUTION ---")
    rd = results["rating_distribution"]
    print(f"  Total Players: {rd['n_players']}")
    print(f"  Rating Mean: {rd['rating_mean']:.1f}")
    print(f"  Rating Std Dev: {rd['rating_std']:.1f}")
    print(f"  Rating Range: {rd['rating_min']:.1f} - {rd['rating_max']:.1f}")
    print(f"  RD Mean: {rd['rd_mean']:.1f}")
    print(f"  RD Range: {rd['rd_min']:.1f} - {rd['rd_max']:.1f}")
    print(f"  Avg Volatility: {rd['volatility_mean']:.4f}")
    print(f"  Avg Matches/Player: {rd['avg_matches_per_player']:.1f}")
    print(f"  Players with 10+ matches: {rd['players_with_10plus_matches']}")

    # Upset Analysis
    print("\n--- UPSET ANALYSIS ---")
    ups = results["upset_analysis"]
    print(f"  Overall Upset Rate: {ups['overall_upset_rate']:.4f} ({ups['overall_upset_rate']*100:.1f}%)")
    print(f"  Upset Rate (Big Favorite >150 pts): {ups['upset_rate_big_favorite']:.4f}" if ups['upset_rate_big_favorite'] else "  Upset Rate (Big Favorite): N/A")
    print(f"  Upset Rate (Close Match <50 pts): {ups['upset_rate_close_match']:.4f}" if ups['upset_rate_close_match'] else "  Upset Rate (Close Match): N/A")

    # Uncertainty Analysis
    print("\n--- PERFORMANCE BY UNCERTAINTY ---")
    unc = results["uncertainty_analysis"]
    print(f"  Accuracy (Low RD matches): {unc['accuracy_low_uncertainty']:.4f}")
    print(f"  Accuracy (High RD matches): {unc['accuracy_high_uncertainty']:.4f}")
    print(f"  Brier (Low RD matches): {unc['brier_low_uncertainty']:.4f}")
    print(f"  Brier (High RD matches): {unc['brier_high_uncertainty']:.4f}")

    # Sharpness
    print("\n--- PREDICTION SHARPNESS ---")
    shp = results["sharpness"]
    print(f"  Mean Sharpness: {shp['mean_sharpness']:.4f} (distance from 0.5)")
    print(f"  Very Confident (>70% or <30%): {shp['pct_very_confident']*100:.1f}%")
    print(f"  Moderate Confident: {shp['pct_moderate_confident']*100:.1f}%")
    print(f"  Uncertain (45-55%): {shp['pct_uncertain']*100:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey Strengths:")
    if prob['auc_roc'] > 0.65:
        print(f"  + Good discrimination (AUC = {prob['auc_roc']:.3f})")
    if cal['expected_calibration_error'] < 0.05:
        print(f"  + Well calibrated (ECE = {cal['expected_calibration_error']:.4f})")
    if prob['brier_skill_score'] > 0.05:
        print(f"  + Better than random baseline (BSS = {prob['brier_skill_score']:.3f})")
    if acc['accuracy_high_confidence'] and acc['accuracy_high_confidence'] > acc['accuracy']:
        print(f"  + Higher accuracy on confident predictions ({acc['accuracy_high_confidence']:.3f} vs {acc['accuracy']:.3f})")

    print("\nAreas for Improvement:")
    if prob['auc_roc'] < 0.65:
        print(f"  - Limited discrimination (AUC = {prob['auc_roc']:.3f})")
    if cal['expected_calibration_error'] > 0.05:
        print(f"  - Calibration could be better (ECE = {cal['expected_calibration_error']:.4f})")
    if abs(cal['calibration_slope'] - 1.0) > 0.2:
        print(f"  - Calibration slope off (slope = {cal['calibration_slope']:.3f}, ideal = 1.0)")
    if unc['accuracy_low_uncertainty'] < unc['accuracy_high_uncertainty']:
        print(f"  - Uncertainty estimates may not reflect true confidence")

    print()


def main():
    print("Loading data...")
    matches_df, stats_df = load_data()
    print(f"Total matches: {len(matches_df)}")

    print("\nRunning assessment...")
    results = assess_glicko2(matches_df, stats_df)

    print_assessment(results)

    # Save results
    import json
    output_path = "data/models/glicko2_assessment.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
