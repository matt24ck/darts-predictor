#!/usr/bin/env python3
"""
Match Prediction Script

Predict win probabilities and 180s distribution for a match between two players.
Uses the optimal models: Glicko-2 for ratings, VisitLevel180sModel for 180s.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import ParquetStore


def load_glicko2_system(model_path: str = "data/models/glicko2_system.json"):
    """Load the Glicko-2 rating system."""
    from src.models import Glicko2System

    if not Path(model_path).exists():
        return None

    glicko = Glicko2System()
    glicko.load(model_path)
    return glicko


def load_180s_model(model_path: str = "data/models/model_180s_visit_level.json"):
    """Load the VisitLevel180sModel for 180s prediction."""
    from src.models import VisitLevel180sModel

    if not Path(model_path).exists():
        return None

    model = VisitLevel180sModel()
    model.load(model_path)
    return model


def get_player_name(store: ParquetStore, player_id: int) -> str:
    """Get player name from ID."""
    players_df = store.get_players()
    if players_df.empty:
        return f"Player {player_id}"

    match = players_df[players_df["player_id"] == player_id]
    if match.empty:
        return f"Player {player_id}"

    name = match.iloc[0]["name"]
    return name if name else f"Player {player_id}"


def compute_180s_probability_matrix(
    lambda_home: float,
    lambda_away: float,
    overdispersion: float = 1.0,
    max_180s: int = 15,
) -> dict:
    """
    Compute probability matrix for 180s using Negative Binomial.

    Returns probabilities for:
    - P(home 180s >= x) for each x
    - P(away 180s >= x) for each x
    - P(total 180s >= x) for each x
    """
    from src.models.predictions import _nb_cdf

    lambda_total = lambda_home + lambda_away

    x_values = list(range(max_180s + 1))

    # P(X >= x) = 1 - P(X < x) = 1 - P(X <= x-1)
    home_probs = []
    away_probs = []
    total_probs = []

    for x in x_values:
        if x == 0:
            home_probs.append(1.0)
            away_probs.append(1.0)
            total_probs.append(1.0)
        else:
            home_probs.append(1.0 - _nb_cdf(x - 1, lambda_home, overdispersion))
            away_probs.append(1.0 - _nb_cdf(x - 1, lambda_away, overdispersion))
            total_probs.append(1.0 - _nb_cdf(x - 1, lambda_total, overdispersion))

    return {
        "x_values": x_values,
        "home_ge_x": home_probs,
        "away_ge_x": away_probs,
        "total_ge_x": total_probs,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "lambda_total": lambda_total,
        "overdispersion": overdispersion,
    }


def print_180s_matrix(
    matrix: dict,
    home_name: str,
    away_name: str,
    max_display: int = 15,
):
    """Print the 180s probability matrix."""
    print("\n" + "=" * 70)
    print("180s PROBABILITY MATRIX")
    print("=" * 70)
    print(f"\nExpected 180s:")
    print(f"  {home_name}: {matrix['lambda_home']:.2f}")
    print(f"  {away_name}: {matrix['lambda_away']:.2f}")
    print(f"  Total: {matrix['lambda_total']:.2f}")

    print(f"\nP(180s >= X):")
    print("-" * 70)

    # Header
    home_short = home_name[:12] if len(home_name) > 12 else home_name
    away_short = away_name[:12] if len(away_name) > 12 else away_name

    print(f"{'X':>4} | {home_short:>12} | {away_short:>12} | {'Total':>12}")
    print("-" * 70)

    for i, x in enumerate(matrix["x_values"]):
        if x > max_display:
            break

        home_p = matrix["home_ge_x"][i]
        away_p = matrix["away_ge_x"][i]
        total_p = matrix["total_ge_x"][i]

        print(f"{x:>4} | {home_p:>11.1%} | {away_p:>11.1%} | {total_p:>11.1%}")

    print("-" * 70)


def predict_match(
    home_player_id: int,
    away_player_id: int,
    league_id: int,
    best_of_sets: int = 0,
    best_of_legs: int = 5,
    is_set_format: bool = False,
    max_180s_display: int = 15,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
):
    """
    Predict match outcome between two players.
    """
    store = ParquetStore(data_dir)

    # Get player names
    home_name = get_player_name(store, home_player_id)
    away_name = get_player_name(store, away_player_id)

    print("\n" + "=" * 70)
    print("MATCH PREDICTION")
    print("=" * 70)
    print(f"\n{home_name} vs {away_name}")
    print(f"League ID: {league_id}")

    if is_set_format:
        print(f"Format: Best of {best_of_sets} sets (best of {best_of_legs} legs per set)")
    else:
        print(f"Format: Best of {best_of_legs} legs")

    # Load models
    glicko = load_glicko2_system(f"{models_dir}/glicko2_system.json")
    model_180s = load_180s_model(f"{models_dir}/model_180s_visit_level.json")

    # Use MODUS-specific model for league_id=38 if available
    model_180s_modus = load_180s_model(f"{models_dir}/model_180s_modus.json")
    if league_id == 38 and model_180s_modus is not None:
        model_180s = model_180s_modus

    # Glicko-2 ratings
    print("\n" + "-" * 70)
    print("RATINGS (Glicko-2)")
    print("-" * 70)

    home_glicko = None
    away_glicko = None
    home_rd = None
    away_rd = None
    home_win_prob = None
    glicko_ci = None

    if glicko:
        home_player = glicko.ratings.get(home_player_id)
        away_player = glicko.ratings.get(away_player_id)

        if home_player is not None and away_player is not None:
            home_glicko = home_player.rating
            away_glicko = away_player.rating
            home_rd = home_player.rd
            away_rd = away_player.rd

            # Win probability from Glicko-2
            home_win_prob, _, meta = glicko.win_probability(
                home_player_id, away_player_id
            )
            glicko_ci = (meta.get("p_a_lower"), meta.get("p_a_upper"))
            if glicko_ci[0] is None:
                glicko_ci = None

            print(f"\n  {home_name}: {home_glicko:.1f} (RD {home_rd:.0f})")
            print(f"  {away_name}: {away_glicko:.1f} (RD {away_rd:.0f})")
            print(f"  Difference: {home_glicko - away_glicko:+.1f}")

            # Show TV bonus if available
            if hasattr(home_player, 'tv_bonus') and home_player.tv_match_count >= 5:
                print(f"  {home_name} TV bonus: {home_player.tv_bonus:+.1f}")
            if hasattr(away_player, 'tv_bonus') and away_player.tv_match_count >= 5:
                print(f"  {away_name} TV bonus: {away_player.tv_bonus:+.1f}")
        else:
            print(f"\n  One or both players not found in rating system")
    else:
        print("\n  Glicko-2 system not loaded")

    # Win probabilities
    print("\n" + "-" * 70)
    print("WIN PROBABILITIES")
    print("-" * 70)

    if home_win_prob is not None:
        print(f"\n  {home_name}: {home_win_prob:.1%}")
        print(f"  {away_name}: {1 - home_win_prob:.1%}")
        if glicko_ci:
            print(f"  95% CI: [{glicko_ci[0]:.1%}, {glicko_ci[1]:.1%}]")

    # 180s prediction
    lambda_home = None
    lambda_away = None
    overdispersion = 1.0

    if model_180s:
        format_params = {
            "best_of_sets": best_of_sets,
            "best_of_legs": best_of_legs,
            "is_set_format": is_set_format,
        }

        result = model_180s.predict(
            home_player_id=home_player_id,
            away_player_id=away_player_id,
            league_id=league_id,
            format_params=format_params,
        )

        lambda_home = result.lambda_home
        lambda_away = result.lambda_away
        overdispersion = model_180s.get_overdispersion(format_params)

    # 180s probability matrix
    if lambda_home is not None and lambda_away is not None:
        matrix = compute_180s_probability_matrix(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            overdispersion=overdispersion,
            max_180s=max_180s_display,
        )

        print_180s_matrix(matrix, home_name, away_name, max_180s_display)
    else:
        print("\n180s model not loaded or players not found")

    # Most 180s head-to-head
    most_180s = None
    if lambda_home is not None and lambda_away is not None:
        from src.models.predictions import compute_most_180s_probabilities
        most_180s = compute_most_180s_probabilities(
            lambda_home, lambda_away, overdispersion=overdispersion
        )

        print("\n" + "-" * 70)
        print("MOST 180s (Head-to-Head)")
        print("-" * 70)
        print(f"\n  {home_name}: {most_180s['p_home_more']:.1%}")
        print(f"  Draw: {most_180s['p_draw']:.1%}")
        print(f"  {away_name}: {most_180s['p_away_more']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if home_win_prob is not None:
        print(f"\nWin Probability:")
        print(f"  {home_name}: {home_win_prob:.1%}")
        print(f"  {away_name}: {1 - home_win_prob:.1%}")

    if lambda_home is not None:
        print(f"\nExpected 180s: {lambda_home + lambda_away:.1f} total")

    print()

    return {
        "home_player_id": home_player_id,
        "away_player_id": away_player_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_glicko": home_glicko,
        "away_glicko": away_glicko,
        "home_rd": home_rd,
        "away_rd": away_rd,
        "home_win_prob": home_win_prob,
        "home_win_prob_ci": glicko_ci,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "most_180s": most_180s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict match outcome between two darts players"
    )
    parser.add_argument(
        "home_player_id",
        type=int,
        help="Home player ID",
    )
    parser.add_argument(
        "away_player_id",
        type=int,
        help="Away player ID",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=2,
        help="League ID (default: 2 = World Championship)",
    )
    parser.add_argument(
        "--best-of-sets",
        type=int,
        default=0,
        help="Best of N sets (0 for leg format, default: 0)",
    )
    parser.add_argument(
        "--best-of-legs",
        type=int,
        default=5,
        help="Best of N legs (default: 5)",
    )
    parser.add_argument(
        "--set-format",
        action="store_true",
        help="Use set format (default: leg format)",
    )
    parser.add_argument(
        "--max-180s",
        type=int,
        default=15,
        help="Maximum 180s to show in probability matrix (default: 15)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Data directory (default: data/processed)",
    )
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Models directory (default: data/models)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args()

    result = predict_match(
        home_player_id=args.home_player_id,
        away_player_id=args.away_player_id,
        league_id=args.league_id,
        best_of_sets=args.best_of_sets,
        best_of_legs=args.best_of_legs,
        is_set_format=args.set_format,
        max_180s_display=args.max_180s,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
