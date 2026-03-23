#!/usr/bin/env python3
"""
Glicko-2 Hyperparameter Tuning Script

Uses a two-stage approach:
1. Coarse grid search to find promising regions
2. Fine-tuning around best configurations

Key hyperparameters to tune:
1. initial_rd - Starting uncertainty
2. rd_decay_per_day - Uncertainty increase rate with inactivity
3. margin_weight - How much score margin affects ratings
4. rating_period_days - Length of rating period for batching
5. recency_half_life_days - Recency weighting half-life
"""

import sys
import json
import itertools
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import ParquetStore
from src.models import Glicko2System, Glicko2Config


@dataclass
class TuningResult:
    """Result of a hyperparameter configuration test."""
    params: Dict[str, Any]
    accuracy: float
    brier_score: float
    log_loss: float
    calibration_error: float

    @property
    def composite_score(self) -> float:
        """Composite score balancing accuracy and calibration. Higher is better."""
        accuracy_term = self.accuracy
        brier_term = 1.0 - self.brier_score
        calibration_term = 1.0 - self.calibration_error * 5
        return 0.4 * accuracy_term + 0.3 * brier_term + 0.3 * calibration_term


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


def evaluate_config(
    config: Glicko2Config,
    train_matches: pd.DataFrame,
    test_matches: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> TuningResult:
    """Evaluate a single hyperparameter configuration."""

    glicko = Glicko2System(config)
    glicko.train(train_matches, stats_df)

    predictions = []
    actuals = []

    for _, match in test_matches.iterrows():
        home_id = match["home_player_id"]
        away_id = match["away_player_id"]

        if match.get("is_set_format", False):
            home_won = match["home_sets"] > match["away_sets"]
        else:
            home_won = match["home_legs"] > match["away_legs"]
        actuals.append(1 if home_won else 0)

        home_player = glicko.ratings.get(home_id)
        away_player = glicko.ratings.get(away_id)

        if home_player and away_player:
            p_glicko, _, _ = glicko.win_probability(home_id, away_id)
            predictions.append(p_glicko)
        else:
            predictions.append(0.5)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Accuracy
    pred_outcomes = (predictions >= 0.5).astype(int)
    accuracy = np.mean(pred_outcomes == actuals)

    # Brier Score
    brier_score = np.mean((predictions - actuals) ** 2)

    # Log Loss
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) +
        (1 - actuals) * np.log(1 - predictions_clipped)
    )

    # Calibration Error
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(predictions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)

    calibration_error = 0.0
    bin_counts = 0
    for i in range(10):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_pred = np.mean(predictions[mask])
            bin_actual = np.mean(actuals[mask])
            calibration_error += np.abs(bin_pred - bin_actual) * np.sum(mask)
            bin_counts += np.sum(mask)
    calibration_error /= bin_counts if bin_counts > 0 else 1

    params = {
        "initial_rd": config.initial_rd,
        "rd_decay_per_day": config.rd_decay_per_day,
        "margin_weight": config.margin_weight,
        "rating_period_days": config.rating_period_days,
        "recency_half_life_days": config.recency_half_life_days,
        "use_margin_scoring": config.use_margin_scoring,
        "use_rating_periods": config.use_rating_periods,
        "use_recency_weighting": config.use_recency_weighting,
        "use_surface_ratings": config.use_surface_ratings,
    }

    return TuningResult(
        params=params,
        accuracy=accuracy,
        brier_score=brier_score,
        log_loss=log_loss,
        calibration_error=calibration_error,
    )


def run_focused_search(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> List[TuningResult]:
    """Run a focused search over key hyperparameters."""

    split_idx = int(len(matches_df) * train_ratio)
    train_matches = matches_df.iloc[:split_idx]
    test_matches = matches_df.iloc[split_idx:]

    print(f"Train matches: {len(train_matches)}")
    print(f"Test matches: {len(test_matches)}")

    results = []
    best_result = None
    config_num = 0

    # Stage 1: Test feature combinations with default params
    print("\n--- STAGE 1: Testing Feature Combinations ---")
    feature_sets = [
        {"name": "Full features", "margin": True, "periods": True, "recency": True, "surface": True},
        {"name": "No margin", "margin": False, "periods": True, "recency": True, "surface": True},
        {"name": "No periods", "margin": True, "periods": False, "recency": True, "surface": True},
        {"name": "No recency", "margin": True, "periods": True, "recency": False, "surface": True},
        {"name": "No surface", "margin": True, "periods": True, "recency": True, "surface": False},
        {"name": "Baseline (minimal)", "margin": False, "periods": False, "recency": False, "surface": False},
    ]

    best_features = feature_sets[0]  # Default

    for feat in feature_sets:
        config_num += 1
        config = Glicko2Config(
            initial_rd=350.0,
            rd_decay_per_day=0.5,
            margin_weight=0.3 if feat["margin"] else 0.0,
            rating_period_days=7,
            recency_half_life_days=180.0,
            use_margin_scoring=feat["margin"],
            use_rating_periods=feat["periods"],
            use_recency_weighting=feat["recency"],
            use_surface_ratings=feat["surface"],
        )

        result = evaluate_config(config, train_matches, test_matches, stats_df)
        results.append(result)

        print(f"  [{config_num}] {feat['name']}: Acc={result.accuracy:.4f}, Brier={result.brier_score:.4f}, Calib={result.calibration_error:.4f}")

        if best_result is None or result.composite_score > best_result.composite_score:
            best_result = result
            best_features = feat

    print(f"\n  Best features: {best_features['name']}")

    # Stage 2: Tune numerical parameters with best features
    print("\n--- STAGE 2: Tuning Numerical Parameters ---")

    param_grid = {
        "initial_rd": [250.0, 300.0, 350.0, 400.0],
        "rd_decay_per_day": [0.2, 0.4, 0.6, 0.8, 1.0],
        "margin_weight": [0.1, 0.2, 0.3, 0.4, 0.5] if best_features["margin"] else [0.0],
        "rating_period_days": [5, 7, 10, 14] if best_features["periods"] else [7],
        "recency_half_life_days": [60.0, 120.0, 180.0, 270.0, 365.0] if best_features["recency"] else [180.0],
    }

    # Generate combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))

    print(f"  Testing {len(all_combinations)} parameter combinations...")

    for combo in all_combinations:
        config_num += 1
        params = dict(zip(param_names, combo))

        config = Glicko2Config(
            initial_rd=params["initial_rd"],
            rd_decay_per_day=params["rd_decay_per_day"],
            margin_weight=params["margin_weight"],
            rating_period_days=params["rating_period_days"],
            recency_half_life_days=params["recency_half_life_days"],
            use_margin_scoring=best_features["margin"],
            use_rating_periods=best_features["periods"],
            use_recency_weighting=best_features["recency"],
            use_surface_ratings=best_features["surface"],
        )

        try:
            result = evaluate_config(config, train_matches, test_matches, stats_df)
            results.append(result)

            if result.composite_score > best_result.composite_score:
                best_result = result
                print(f"  [{config_num}] NEW BEST: Acc={result.accuracy:.4f}, Brier={result.brier_score:.4f}, Calib={result.calibration_error:.4f}")
                print(f"        Params: rd={params['initial_rd']}, decay={params['rd_decay_per_day']}, margin={params['margin_weight']}, period={params['rating_period_days']}, recency={params['recency_half_life_days']}")

        except Exception as e:
            print(f"  [{config_num}] Error: {e}")
            continue

        if config_num % 50 == 0:
            print(f"  [{config_num}/{len(all_combinations) + 6}] Progress... (best composite: {best_result.composite_score:.4f})")

    return results


def main():
    start_time = time.time()

    print("=" * 70)
    print("GLICKO-2 HYPERPARAMETER TUNING")
    print("=" * 70)

    print("\nLoading data...")
    matches_df, stats_df = load_data()
    print(f"Total matches: {len(matches_df)}")

    print("\nRunning focused search...")
    results = run_focused_search(matches_df, stats_df)

    # Sort by composite score
    sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
    best = sorted_results[0]

    # Also find best by accuracy
    best_by_accuracy = max(results, key=lambda x: x.accuracy)

    # And best by calibration
    best_by_calibration = min(results, key=lambda x: x.calibration_error)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nConfigurations tested: {len(results)}")
    print(f"Time elapsed: {time.time() - start_time:.1f} seconds")

    print("\n--- BEST OVERALL (Composite Score) ---")
    print(f"  Accuracy:          {best.accuracy:.4f}")
    print(f"  Brier Score:       {best.brier_score:.4f}")
    print(f"  Log Loss:          {best.log_loss:.4f}")
    print(f"  Calibration Error: {best.calibration_error:.4f}")
    print(f"  Composite Score:   {best.composite_score:.4f}")
    print("\n  Optimal Parameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    print("\n--- BEST BY ACCURACY ---")
    print(f"  Accuracy: {best_by_accuracy.accuracy:.4f}")
    print(f"  Key params: rd_decay={best_by_accuracy.params['rd_decay_per_day']}, margin={best_by_accuracy.params['margin_weight']}")

    print("\n--- BEST BY CALIBRATION ---")
    print(f"  Calibration Error: {best_by_calibration.calibration_error:.4f}")
    print(f"  Key params: rd_decay={best_by_calibration.params['rd_decay_per_day']}, margin={best_by_calibration.params['margin_weight']}")

    # Save results
    output_path = "data/models/glicko2_tuning_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    analysis = {
        "n_configs_tested": len(results),
        "best_overall": {
            "params": best.params,
            "accuracy": float(best.accuracy),
            "brier_score": float(best.brier_score),
            "log_loss": float(best.log_loss),
            "calibration_error": float(best.calibration_error),
            "composite_score": float(best.composite_score),
        },
        "best_by_accuracy": {
            "params": best_by_accuracy.params,
            "accuracy": float(best_by_accuracy.accuracy),
        },
        "best_by_calibration": {
            "params": best_by_calibration.params,
            "calibration_error": float(best_by_calibration.calibration_error),
        },
        "top_5": [
            {
                "rank": i + 1,
                "params": r.params,
                "accuracy": float(r.accuracy),
                "brier_score": float(r.brier_score),
                "calibration_error": float(r.calibration_error),
                "composite_score": float(r.composite_score),
            }
            for i, r in enumerate(sorted_results[:5])
        ],
    }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate optimal config code
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    p = best.params
    print(f"""
Glicko2Config(
    initial_rating=1500.0,
    initial_rd={p['initial_rd']},
    initial_volatility=0.06,
    rd_decay_per_day={p['rd_decay_per_day']},
    max_rd=350.0,
    min_rd=30.0,
    use_match_length_weighting=True,
    use_margin_scoring={p['use_margin_scoring']},
    margin_weight={p['margin_weight']},
    use_venue_effects=True,
    use_performance_adjustment=True,
    performance_weight=0.2,
    use_rating_periods={p['use_rating_periods']},
    rating_period_days={p['rating_period_days']},
    use_surface_ratings={p['use_surface_ratings']},
    min_surface_matches=5,
    use_recency_weighting={p['use_recency_weighting']},
    recency_half_life_days={p['recency_half_life_days']},
    min_matches_for_stable=10,
)
""")


if __name__ == "__main__":
    main()
