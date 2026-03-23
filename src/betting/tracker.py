"""
Bet tracker — creates predictions, records bets, settles results, computes P&L.

Orchestrates the flow: model prediction -> odds comparison -> Kelly stake -> bet record -> settlement.
"""

from typing import Dict, List, Optional

from ..storage.sqlite_store import SqliteStore
from .kelly import kelly_stake, kelly_analysis


class BetTracker:
    """High-level bet tracking operations built on SqliteStore."""

    def __init__(self, db: SqliteStore):
        self.db = db

    # =========================================================================
    # Match + Prediction Creation
    # =========================================================================

    def ensure_match(
        self,
        home_player_id: int,
        away_player_id: int,
        home_player_name: str,
        away_player_name: str,
        match_date: str,
        league_id: int = None,
        league_name: str = None,
        best_of_sets: int = 0,
        best_of_legs: int = 5,
        is_set_format: bool = False,
        external_match_id: str = None,
    ) -> int:
        """Create or find an upcoming match. Returns match ID."""
        return self.db.upsert_upcoming_match(
            external_match_id=external_match_id,
            home_player_id=home_player_id,
            away_player_id=away_player_id,
            home_player_name=home_player_name,
            away_player_name=away_player_name,
            league_id=league_id,
            league_name=league_name,
            match_date=match_date,
            best_of_sets=best_of_sets,
            best_of_legs=best_of_legs,
            is_set_format=int(is_set_format),
        )

    def create_match_predictions(
        self,
        upcoming_match_id: int,
        win_probability: Dict,
        home_name: str,
        away_name: str,
        expected_180s: Dict = None,
    ) -> List[int]:
        """
        Create prediction records for a match from model output.

        Args:
            upcoming_match_id: ID of the upcoming match.
            win_probability: Dict with 'home_win', 'away_win', 'confidence_interval'.
            home_name, away_name: Player names for descriptions.
            expected_180s: Optional dict with '180s' prediction data.

        Returns:
            List of prediction IDs created.
        """
        prediction_ids = []

        if win_probability:
            ci = win_probability.get("confidence_interval", {})

            # Home win prediction
            pid = self.db.create_prediction(
                upcoming_match_id=upcoming_match_id,
                prediction_type="match_winner",
                selection="home",
                model_probability=win_probability["home_win"],
                market_description=f"{home_name} to win",
                confidence_lower=ci.get("home_lower"),
                confidence_upper=ci.get("home_upper"),
            )
            prediction_ids.append(pid)

            # Away win prediction
            pid = self.db.create_prediction(
                upcoming_match_id=upcoming_match_id,
                prediction_type="match_winner",
                selection="away",
                model_probability=win_probability["away_win"],
                market_description=f"{away_name} to win",
                confidence_lower=1 - ci.get("home_upper", 1) if ci.get("home_upper") else None,
                confidence_upper=1 - ci.get("home_lower", 0) if ci.get("home_lower") else None,
            )
            prediction_ids.append(pid)

        if expected_180s:
            # Total 180s over/under predictions at various lines
            total = expected_180s.get("expected_total", 0)
            ci_180s = expected_180s.get("confidence_interval", {})

            pid = self.db.create_prediction(
                upcoming_match_id=upcoming_match_id,
                prediction_type="total_180s_expected",
                selection=f"total_{total}",
                model_probability=0.5,  # median expectation
                market_description=f"Expected total 180s: {total}",
                model_expected_value=total,
                confidence_lower=ci_180s.get("lower"),
                confidence_upper=ci_180s.get("upper"),
            )
            prediction_ids.append(pid)

        return prediction_ids

    # =========================================================================
    # Bet Placement
    # =========================================================================

    def place_bet(
        self,
        user_id: int,
        prediction_id: int,
        decimal_odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        bookmaker: str = None,
        odds_snapshot_id: int = None,
        actual_stake: float = None,
    ) -> Optional[Dict]:
        """
        Place a bet on a prediction at given odds.

        Calculates Kelly stake and records the bet.
        Returns bet details dict or None if no value (edge < min_edge).
        """
        # Get prediction to get model probability
        conn = self.db._get_conn()
        try:
            pred = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        finally:
            conn.close()

        if not pred:
            return None

        model_prob = pred["model_probability"]
        analysis = kelly_analysis(model_prob, decimal_odds, bankroll, kelly_fraction, min_edge)

        # Record the bet even if not recommended (user might override)
        bet_id = self.db.create_bet(
            user_id=user_id,
            prediction_id=prediction_id,
            decimal_odds=decimal_odds,
            model_probability=model_prob,
            edge=analysis["edge"],
            kelly_fraction=kelly_fraction,
            recommended_stake=analysis["stake_units"],
            bankroll_at_time=bankroll,
            bookmaker=bookmaker,
            odds_snapshot_id=odds_snapshot_id,
            actual_stake=actual_stake,
        )

        return {
            "bet_id": bet_id,
            "prediction_id": prediction_id,
            **analysis,
        }

    # =========================================================================
    # Settlement
    # =========================================================================

    def settle_match(self, upcoming_match_id: int, home_won: bool, home_score: str = None, away_score: str = None):
        """
        Settle all bets for a match based on result.

        Args:
            upcoming_match_id: The match ID.
            home_won: True if home player won.
            home_score, away_score: Score strings for record.
        """
        # Update match status
        self.db.update_match_status(
            upcoming_match_id, "completed", home_score, away_score
        )

        # Get all bets for this match
        conn = self.db._get_conn()
        try:
            rows = conn.execute(
                """SELECT b.id as bet_id, p.prediction_type, p.selection
                   FROM bets b
                   JOIN predictions p ON b.prediction_id = p.id
                   WHERE p.upcoming_match_id = ? AND b.result = 'pending'""",
                (upcoming_match_id,),
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            bet_id = row["bet_id"]
            ptype = row["prediction_type"]
            selection = row["selection"]

            if ptype == "match_winner":
                if selection == "home":
                    result = "win" if home_won else "loss"
                elif selection == "away":
                    result = "loss" if home_won else "win"
                else:
                    result = "void"
            else:
                # Non-match-winner bets need manual settlement for now
                continue

            self.db.settle_bet(bet_id, result)

    def settle_180s_bet(self, bet_id: int, actual_180s: int):
        """Settle a 180s over/under bet based on actual count."""
        conn = self.db._get_conn()
        try:
            row = conn.execute(
                """SELECT p.prediction_type, p.selection
                   FROM bets b JOIN predictions p ON b.prediction_id = p.id
                   WHERE b.id = ?""",
                (bet_id,),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return

        selection = row["selection"]
        ptype = row["prediction_type"]

        if ptype == "total_180s_over":
            line = float(selection.replace("over_", ""))
            result = "win" if actual_180s > line else "loss"
        elif ptype == "total_180s_under":
            line = float(selection.replace("under_", ""))
            result = "win" if actual_180s < line else "loss"
        else:
            return

        self.db.settle_bet(bet_id, result)

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_value_bets(self, upcoming_match_id: int, min_edge: float = 0.03) -> List[Dict]:
        """Get all predictions with positive edge for a match, based on latest odds."""
        predictions = self.db.get_predictions_for_match(upcoming_match_id)
        odds = self.db.get_odds_for_match(upcoming_match_id)

        value_bets = []
        for pred in predictions:
            if pred["prediction_type"] not in ("match_winner",):
                continue

            # Find best odds for this selection
            matching_odds = [
                o for o in odds if o["selection"] == pred["selection"]
            ]
            if not matching_odds:
                continue

            best = max(matching_odds, key=lambda o: o["decimal_odds"])
            edge = pred["model_probability"] - best["implied_probability"]

            if edge >= min_edge:
                value_bets.append({
                    "prediction_id": pred["id"],
                    "prediction_type": pred["prediction_type"],
                    "selection": pred["selection"],
                    "market_description": pred["market_description"],
                    "model_probability": pred["model_probability"],
                    "best_odds": best["decimal_odds"],
                    "best_bookmaker": best["bookmaker"],
                    "implied_probability": best["implied_probability"],
                    "edge": round(edge, 4),
                    "edge_pct": round(edge * 100, 2),
                    "odds_snapshot_id": best["id"],
                })

        return sorted(value_bets, key=lambda x: x["edge"], reverse=True)
