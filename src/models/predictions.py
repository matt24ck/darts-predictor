"""
Prediction API Functions

High-level functions for predicting 180s distributions and win probabilities.
Uses optimal models: Glicko2System for ratings, VisitLevel180sModel for 180s.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..storage.parquet_store import ParquetStore
from .visit_level_180s import VisitLevel180sModel
from .glicko2_system import Glicko2System

logger = logging.getLogger(__name__)


def _nb_cdf(x, mu, phi):
    """CDF of Negative Binomial with mean mu and overdispersion phi (Var = mu * phi).

    Falls back to Poisson when phi <= 1.
    """
    from scipy import stats

    if phi <= 1.0 or mu <= 0:
        return stats.poisson.cdf(x, mu)
    n = mu / (phi - 1.0)
    p = 1.0 / phi
    return stats.nbinom.cdf(x, n, p)


def _nb_pmf(x, mu, phi):
    """PMF of Negative Binomial with mean mu and overdispersion phi (Var = mu * phi).

    Falls back to Poisson when phi <= 1.
    """
    from scipy import stats

    if phi <= 1.0 or mu <= 0:
        return stats.poisson.pmf(x, mu)
    n = mu / (phi - 1.0)
    p = 1.0 / phi
    return stats.nbinom.pmf(x, n, p)


# Global model instances (lazy loaded)
_model_180s: Optional[VisitLevel180sModel] = None
_glicko_system: Optional[Glicko2System] = None
_store: Optional[ParquetStore] = None


def _get_store(data_dir: str = "data/processed") -> ParquetStore:
    """Get or create the data store."""
    global _store
    if _store is None:
        _store = ParquetStore(data_dir)
    return _store


def _get_model_180s(
    data_dir: str = "data/processed",
    model_path: Optional[str] = None,
) -> VisitLevel180sModel:
    """Get or create the visit-level 180s model."""
    global _model_180s

    if _model_180s is not None:
        return _model_180s

    _model_180s = VisitLevel180sModel()

    # Try to load from file
    if model_path is None:
        model_path = "data/models/model_180s_visit_level.json"

    if Path(model_path).exists():
        _model_180s.load(model_path)
        return _model_180s

    # Fit from data
    store = _get_store(data_dir)
    matches_df = store.get_matches()
    visits_df = store.get_visits()
    stats_df = store.get_match_stats()

    if not matches_df.empty:
        _model_180s.fit(visits_df, matches_df, stats_df)

    return _model_180s


def _get_glicko_system(
    data_dir: str = "data/processed",
    model_path: Optional[str] = None,
) -> Glicko2System:
    """Get or create the Glicko-2 system."""
    global _glicko_system

    if _glicko_system is not None and _glicko_system.ratings:
        return _glicko_system

    _glicko_system = Glicko2System()

    # Try to load from file
    if model_path is None:
        model_path = "data/models/glicko2_system.json"

    if Path(model_path).exists():
        _glicko_system.load(model_path)
        return _glicko_system

    # Train from data
    store = _get_store(data_dir)
    matches_df = store.get_matches()
    stats_df = store.get_match_stats()

    if not matches_df.empty:
        _glicko_system.train(matches_df, stats_df)

    return _glicko_system


def predict_180_distribution(
    home_player_id: int,
    away_player_id: int,
    league_id: int,
    format_params: Optional[Dict[str, Any]] = None,
    line_x: Optional[int] = None,
    data_dir: str = "data/processed",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict the 180s distribution for a match.

    Args:
        home_player_id: ID of the home player
        away_player_id: ID of the away player
        league_id: ID of the league
        format_params: Dict with best_of_sets, best_of_legs, is_set_format
        line_x: Optional specific line to compute probabilities for
        data_dir: Data directory path
        model_path: Optional path to saved model

    Returns:
        Dict containing lambda_total, lambda_home, lambda_away, etc.
    """
    model = _get_model_180s(data_dir, model_path)

    if format_params is None:
        format_params = {
            "best_of_sets": 0,
            "best_of_legs": 5,
            "is_set_format": False,
        }

    result = model.predict(
        home_player_id=home_player_id,
        away_player_id=away_player_id,
        league_id=league_id,
        format_params=format_params,
    )

    phi = model.get_overdispersion(format_params)

    response = {
        "lambda_total": result.lambda_total,
        "lambda_home": result.lambda_home,
        "lambda_away": result.lambda_away,
        "expected_value": result.lambda_total,
        "overdispersion": phi,
        "home_player_id": home_player_id,
        "away_player_id": away_player_id,
        "league_id": league_id,
        "format_params": format_params,
    }

    if line_x is not None:
        response["line_x"] = line_x
        response["prob_ge_x"] = 1.0 - _nb_cdf(line_x - 1, result.lambda_total, phi) if line_x > 0 else 1.0
        response["prob_lt_x"] = _nb_cdf(line_x - 1, result.lambda_total, phi) if line_x > 0 else 0.0

    return response


def predict_win_probability(
    home_player_id: int,
    away_player_id: int,
    league_id: Optional[int] = None,
    format_params: Optional[Dict[str, Any]] = None,
    data_dir: str = "data/processed",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict win probabilities for a match using Glicko-2.

    Args:
        home_player_id: ID of the home player
        away_player_id: ID of the away player
        league_id: Optional league ID
        format_params: Optional format parameters
        data_dir: Data directory path
        model_path: Optional path to saved model

    Returns:
        Dict with p_home_win, p_away_win, ratings, confidence_interval
    """
    glicko = _get_glicko_system(data_dir, model_path)

    p_home, ci, _ = glicko.win_probability(home_player_id, away_player_id)

    home_player = glicko.ratings.get(home_player_id)
    away_player = glicko.ratings.get(away_player_id)

    home_rating = home_player.rating if home_player else 1500
    away_rating = away_player.rating if away_player else 1500
    home_rd = home_player.rd if home_player else 350
    away_rd = away_player.rd if away_player else 350

    return {
        "p_home_win": p_home,
        "p_away_win": 1 - p_home,
        "home_player_id": home_player_id,
        "away_player_id": away_player_id,
        "home_rating": home_rating,
        "away_rating": away_rating,
        "home_rd": home_rd,
        "away_rd": away_rd,
        "rating_difference": home_rating - away_rating,
        "confidence_interval": ci,
        "model": "glicko2",
    }


def compute_most_180s_probabilities(
    lambda_home: float,
    lambda_away: float,
    overdispersion: float = 1.0,
    max_count: int = 40,
) -> Dict[str, float]:
    """
    Compute P(home has most 180s), P(draw), P(away has most 180s).

    Uses direct PMF enumeration over independent Negative Binomial RVs
    (falls back to Poisson when overdispersion <= 1).

    Args:
        lambda_home: Expected 180s for home player
        lambda_away: Expected 180s for away player
        overdispersion: Var/Mean ratio from the 180s model
        max_count: Truncation point for enumeration

    Returns:
        Dict with p_home_more, p_draw, p_away_more
    """
    p_home_more = 0.0
    p_draw = 0.0

    for k in range(max_count + 1):
        pmf_home_k = _nb_pmf(k, lambda_home, overdispersion)
        pmf_away_k = _nb_pmf(k, lambda_away, overdispersion)

        # P(draw at k)
        p_draw += pmf_home_k * pmf_away_k

        # P(home > away): home scores k, away scores < k
        if k >= 1:
            p_home_more += pmf_home_k * _nb_cdf(k - 1, lambda_away, overdispersion)

    p_away_more = 1.0 - p_home_more - p_draw

    return {
        "p_home_more": round(p_home_more, 4),
        "p_draw": round(p_draw, 4),
        "p_away_more": round(p_away_more, 4),
    }


def predict_most_180s(
    home_player_id: int,
    away_player_id: int,
    league_id: int,
    format_params: Optional[Dict[str, Any]] = None,
    data_dir: str = "data/processed",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict which player will have the most 180s in a match.

    Args:
        home_player_id: ID of the home player
        away_player_id: ID of the away player
        league_id: ID of the league
        format_params: Dict with best_of_sets, best_of_legs, is_set_format
        data_dir: Data directory path
        model_path: Optional path to saved model

    Returns:
        Dict with lambda values and head-to-head probabilities
    """
    model = _get_model_180s(data_dir, model_path)

    if format_params is None:
        format_params = {
            "best_of_sets": 0,
            "best_of_legs": 5,
            "is_set_format": False,
        }

    result = model.predict(
        home_player_id=home_player_id,
        away_player_id=away_player_id,
        league_id=league_id,
        format_params=format_params,
    )

    phi = model.get_overdispersion(format_params)
    probs = compute_most_180s_probabilities(
        result.lambda_home, result.lambda_away, overdispersion=phi
    )

    return {
        "lambda_home": result.lambda_home,
        "lambda_away": result.lambda_away,
        "lambda_total": result.lambda_total,
        "overdispersion": phi,
        **probs,
        "home_player_id": home_player_id,
        "away_player_id": away_player_id,
        "league_id": league_id,
        "format_params": format_params,
    }


def get_player_180_stats(
    player_id: int,
    data_dir: str = "data/processed",
) -> Dict[str, Any]:
    """Get 180s statistics for a player."""
    store = _get_store(data_dir)
    summary = store.get_player_stats_summary(player_id)
    return summary


def get_leaderboard(
    n: int = 50,
    data_dir: str = "data/processed",
) -> Dict[str, Any]:
    """Get Glicko-2 rating leaderboard."""
    glicko = _get_glicko_system(data_dir)
    top_df = glicko.get_top_players(n)

    store = _get_store(data_dir)
    players_df = store.get_players()

    if not players_df.empty and not top_df.empty:
        top_df = top_df.merge(
            players_df[["player_id", "name"]],
            on="player_id",
            how="left",
        )

    leaderboard = []
    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        entry = {
            "rank": rank,
            "player_id": row["player_id"],
            "name": row.get("name", "Unknown"),
            "rating": row["rating"],
            "matches": row.get("matches", row.get("match_count", 0)),
        }
        if "rd" in row:
            entry["rd"] = row["rd"]
        leaderboard.append(entry)

    return {
        "leaderboard": leaderboard,
        "total_players": len(glicko.ratings),
        "model": "glicko2",
    }


def reload_models(
    data_dir: str = "data/processed",
) -> Dict[str, bool]:
    """Force reload all models from data."""
    global _model_180s, _glicko_system, _store

    _model_180s = None
    _glicko_system = None
    _store = None

    results = {}

    try:
        model = _get_model_180s(data_dir)
        results["model_180s"] = True
    except Exception as e:
        logger.error(f"Error loading 180s model: {e}")
        results["model_180s"] = False

    try:
        glicko = _get_glicko_system(data_dir)
        results["glicko_system"] = len(glicko.ratings) > 0
    except Exception as e:
        logger.error(f"Error loading Glicko-2 system: {e}")
        results["glicko_system"] = False

    return results
