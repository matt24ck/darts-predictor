"""
Kelly Criterion staking calculations.

Supports full Kelly and fractional Kelly with configurable
minimum edge thresholds to avoid noise bets.
"""

from typing import Dict, Optional


def kelly_stake(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    min_edge: float = 0.03,
) -> float:
    """
    Calculate Kelly criterion stake as a fraction of bankroll.

    Args:
        model_prob: Model's estimated probability of the outcome.
        decimal_odds: Bookmaker decimal odds (e.g. 1.55).
        fraction: Kelly fraction (0.25 = quarter-Kelly, 1.0 = full Kelly).
        min_edge: Minimum edge required to recommend a bet.

    Returns:
        Recommended stake as proportion of bankroll (0.0 if no bet).
    """
    if decimal_odds <= 1.0 or model_prob <= 0 or model_prob >= 1:
        return 0.0

    implied_prob = 1.0 / decimal_odds
    edge = model_prob - implied_prob

    if edge < min_edge:
        return 0.0

    b = decimal_odds - 1  # net odds (profit per unit staked)
    full_kelly = (model_prob * b - (1 - model_prob)) / b
    return max(0.0, full_kelly * fraction)


def kelly_analysis(
    model_prob: float,
    decimal_odds: float,
    bankroll: float = 100.0,
    fraction: float = 0.25,
    min_edge: float = 0.03,
) -> Dict:
    """
    Full Kelly analysis including edge, stake, and expected value.

    Returns dict with all betting metrics.
    """
    implied_prob = 1.0 / decimal_odds if decimal_odds > 1 else 1.0
    edge = model_prob - implied_prob
    stake_pct = kelly_stake(model_prob, decimal_odds, fraction, min_edge)
    stake_units = stake_pct * bankroll
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)  # EV per unit

    return {
        "model_probability": round(model_prob, 4),
        "decimal_odds": round(decimal_odds, 2),
        "implied_probability": round(implied_prob, 4),
        "edge": round(edge, 4),
        "edge_pct": round(edge * 100, 2),
        "kelly_fraction": fraction,
        "stake_pct": round(stake_pct, 4),
        "stake_pct_display": round(stake_pct * 100, 2),
        "stake_units": round(stake_units, 2),
        "expected_value": round(ev, 4),
        "bet_recommended": stake_pct > 0,
        "bankroll": bankroll,
    }
