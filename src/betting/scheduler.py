"""
Daily match scheduler — discovers matches, runs predictions, fetches odds,
auto-places virtual bets on edges, and settles completed matches.

This is the core pipeline that builds the model's public track record.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..storage.sqlite_store import SqliteStore
from .kelly import kelly_analysis
from .odds import OddsClient, build_player_lookup, build_alias_lookup
from .tracker import BetTracker

logger = logging.getLogger(__name__)

SYSTEM_USER_ID = 1  # Reserved model user for auto-bets
DEFAULT_BANKROLL = 100.0
DEFAULT_KELLY_FRACTION = 0.25
MIN_EDGE = 0.03  # 3% minimum edge for auto-bet


class DailyScheduler:
    """Orchestrates the daily match tracking pipeline."""

    def __init__(self, db: SqliteStore, tracker: BetTracker, odds_client: OddsClient,
                 models: Dict, store, players_list: List[Dict]):
        """
        Args:
            db: SQLite store
            tracker: BetTracker instance
            odds_client: OddsClient for The Odds API
            models: Dict with 'glicko', '180s', '180s_modus' model instances
            store: ParquetStore for historical match data
            players_list: List of {player_id, name} dicts
        """
        self.db = db
        self.tracker = tracker
        self.odds_client = odds_client
        self.models = models
        self.store = store
        self.players_list = players_list

        # Build player lookup for name matching
        players_df = store.get_players()
        self.player_lookup = build_player_lookup(players_df)
        aliases = db.get_all_aliases()
        self.alias_lookup = build_alias_lookup(aliases)

    def run_daily_pipeline(self) -> Dict:
        """
        Run the full daily pipeline.

        Returns summary dict with counts of matches found, predictions made, bets placed.
        """
        summary = {
            "matches_discovered": 0,
            "predictions_made": 0,
            "odds_fetched": 0,
            "bets_placed": 0,
            "bets_settled": 0,
            "unmatched_players": [],
            "errors": [],
        }

        # 1. Discover matches from Odds API
        logger.info("Step 1: Fetching matches from The Odds API...")
        matches = self._discover_matches_from_odds_api(summary)

        # 2. Supplement from parquet (Statorium data)
        logger.info("Step 2: Checking parquet data for additional matches...")
        self._discover_matches_from_parquet(summary)

        # 3. Generate predictions for all un-predicted matches
        logger.info("Step 3: Generating predictions...")
        self._generate_predictions(summary)

        # 4. Auto-place bets on edges
        logger.info("Step 4: Auto-placing bets on value edges...")
        self._auto_place_bets(summary)

        # 5. Settle completed matches
        logger.info("Step 5: Settling completed matches...")
        self._settle_completed_matches(summary)

        logger.info(f"Pipeline complete: {summary}")
        return summary

    # =========================================================================
    # Step 1: Discover matches from The Odds API
    # =========================================================================

    def _discover_matches_from_odds_api(self, summary: Dict) -> List[Dict]:
        """Fetch events from Odds API, create upcoming_matches, save odds."""
        events = self.odds_client.fetch_darts_events()
        if not events:
            logger.info("No events from Odds API")
            return []

        parsed, unmatched = self.odds_client.parse_events(
            events, self.player_lookup, self.alias_lookup
        )
        summary["unmatched_players"] = list(unmatched)

        # Save unmatched as aliases for manual resolution
        for name in unmatched:
            logger.warning(f"Unmatched player: {name}")

        matches_created = []
        for event in parsed:
            if not event["home_player_id"] or not event["away_player_id"]:
                logger.warning(
                    f"Skipping {event['home_name']} vs {event['away_name']} — "
                    f"couldn't match player IDs"
                )
                continue

            if not event["match_date"]:
                continue

            # Create/update match
            match_id = self.tracker.ensure_match(
                home_player_id=event["home_player_id"],
                away_player_id=event["away_player_id"],
                home_player_name=event["home_name"],
                away_player_name=event["away_name"],
                match_date=event["match_date"],
                external_match_id=event["event_id"],
            )
            summary["matches_discovered"] += 1
            matches_created.append(match_id)

            # Save odds from each bookmaker
            for bookie in event["bookmakers"]:
                self.db.save_odds_snapshot(
                    upcoming_match_id=match_id,
                    bookmaker=bookie["name"],
                    market_type="h2h",
                    selection="home",
                    decimal_odds=bookie["home_odds"],
                )
                self.db.save_odds_snapshot(
                    upcoming_match_id=match_id,
                    bookmaker=bookie["name"],
                    market_type="h2h",
                    selection="away",
                    decimal_odds=bookie["away_odds"],
                )
            self.db.mark_match_odds_fetched(match_id)
            summary["odds_fetched"] += 1

        return matches_created

    # =========================================================================
    # Step 2: Supplement from parquet (historical Statorium data)
    # =========================================================================

    def _discover_matches_from_parquet(self, summary: Dict):
        """Find matches in parquet data that aren't yet tracked."""
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()

        matches_df = self.store.get_matches()
        if matches_df.empty:
            return

        # Filter to today/tomorrow matches
        if "match_date" in matches_df.columns:
            matches_df["match_date_str"] = matches_df["match_date"].astype(str)
            upcoming = matches_df[
                (matches_df["match_date_str"] >= today) &
                (matches_df["match_date_str"] <= tomorrow)
            ]

            for _, row in upcoming.iterrows():
                home_id = row.get("home_player_id")
                away_id = row.get("away_player_id")
                if not home_id or not away_id:
                    continue

                home_name = self._get_player_name(int(home_id))
                away_name = self._get_player_name(int(away_id))
                if not home_name or not away_name:
                    continue

                match_date = str(row.get("match_date", ""))[:10]

                self.tracker.ensure_match(
                    home_player_id=int(home_id),
                    away_player_id=int(away_id),
                    home_player_name=home_name,
                    away_player_name=away_name,
                    match_date=match_date,
                    league_id=int(row.get("league_id", 0)) if row.get("league_id") else None,
                    best_of_sets=int(row.get("best_of_sets", 0)),
                    best_of_legs=int(row.get("best_of_legs", 5)),
                    is_set_format=bool(row.get("is_set_format", False)),
                )

    # =========================================================================
    # Step 3: Generate predictions
    # =========================================================================

    def _generate_predictions(self, summary: Dict):
        """Run model predictions for all matches that haven't been predicted yet."""
        matches = self.db.get_upcoming_matches(status="scheduled")

        glicko = self.models.get("glicko")
        model_180s = self.models.get("180s")

        for match in matches:
            if match.get("predictions_generated"):
                continue

            match_id = match["id"]
            home_id = match["home_player_id"]
            away_id = match["away_player_id"]

            win_prob = None
            expected_180s = None

            # Glicko-2 win probability
            if glicko:
                home_rating = glicko.ratings.get(home_id)
                away_rating = glicko.ratings.get(away_id)

                if home_rating and away_rating:
                    p_home, p_away, meta = glicko.calibrated_win_probability(home_id, away_id)
                    ci = meta or {}
                    win_prob = {
                        "home_win": round(p_home, 4),
                        "away_win": round(p_away, 4),
                        "confidence_interval": {
                            "home_lower": round(ci.get("p_a_lower", 0), 4),
                            "home_upper": round(ci.get("p_a_upper", 1), 4),
                        },
                    }
                else:
                    logger.warning(
                        f"No ratings for {match['home_player_name']} vs {match['away_player_name']}"
                    )
                    continue  # Can't bet without win probability

            if not win_prob:
                continue

            # 180s prediction
            if model_180s:
                league_id = match.get("league_id") or 2
                active_180s = model_180s
                if league_id == 38 and self.models.get("180s_modus"):
                    active_180s = self.models["180s_modus"]

                try:
                    pred = active_180s.predict(
                        home_player_id=home_id,
                        away_player_id=away_id,
                        league_id=league_id,
                        format_params={
                            "best_of_sets": match.get("best_of_sets", 0),
                            "best_of_legs": match.get("best_of_legs", 5),
                            "is_set_format": bool(match.get("is_set_format", 0)),
                        },
                    )
                    expected_180s = {
                        "expected_total": round(pred.lambda_total, 2),
                        "expected_home": round(pred.lambda_home, 2),
                        "expected_away": round(pred.lambda_away, 2),
                        "confidence_interval": {
                            "lower": round(pred.lambda_total_lower, 2),
                            "upper": round(pred.lambda_total_upper, 2),
                        },
                    }
                except Exception as e:
                    logger.warning(f"180s prediction failed for match {match_id}: {e}")

            # Create prediction records
            self.tracker.create_match_predictions(
                upcoming_match_id=match_id,
                win_probability=win_prob,
                home_name=match["home_player_name"],
                away_name=match["away_player_name"],
                expected_180s=expected_180s,
            )
            self.db.mark_match_predictions_generated(match_id)
            summary["predictions_made"] += 1

    # =========================================================================
    # Step 4: Auto-place bets on value edges
    # =========================================================================

    def _auto_place_bets(self, summary: Dict):
        """Place virtual bets on all selections with edge >= MIN_EDGE."""
        matches = self.db.get_upcoming_matches(status="scheduled")

        for match in matches:
            if not match.get("predictions_generated") or not match.get("odds_fetched"):
                continue

            match_id = match["id"]

            # Check if bets already placed for this match
            conn = self.db._get_conn()
            try:
                existing = conn.execute(
                    """SELECT COUNT(*) as cnt FROM bets b
                       JOIN predictions p ON b.prediction_id = p.id
                       WHERE b.user_id = ? AND p.upcoming_match_id = ?""",
                    (SYSTEM_USER_ID, match_id),
                ).fetchone()
                if existing["cnt"] > 0:
                    continue
            finally:
                conn.close()

            # Get value bets for this match
            value_bets = self.tracker.get_value_bets(match_id, min_edge=MIN_EDGE)

            for vb in value_bets:
                result = self.tracker.place_bet(
                    user_id=SYSTEM_USER_ID,
                    prediction_id=vb["prediction_id"],
                    decimal_odds=vb["best_odds"],
                    bankroll=DEFAULT_BANKROLL,
                    kelly_fraction=DEFAULT_KELLY_FRACTION,
                    min_edge=MIN_EDGE,
                    bookmaker=vb["best_bookmaker"],
                    odds_snapshot_id=vb.get("odds_snapshot_id"),
                )
                if result and result.get("bet_id"):
                    summary["bets_placed"] += 1
                    logger.info(
                        f"Auto-bet: {vb['market_description']} @ {vb['best_odds']} "
                        f"(edge {vb['edge_pct']}%, stake {result['stake_units']})"
                    )

    # =========================================================================
    # Step 5: Settle completed matches
    # =========================================================================

    def _settle_completed_matches(self, summary: Dict):
        """Settle bets for matches that have completed."""
        today = date.today().isoformat()
        scheduled = self.db.get_upcoming_matches(status="scheduled")

        # Only try to settle matches from before today
        past_matches = [m for m in scheduled if m["match_date"] < today]

        matches_df = self.store.get_matches()
        if matches_df.empty:
            return

        for match in past_matches:
            home_id = match["home_player_id"]
            away_id = match["away_player_id"]
            match_date = match["match_date"]

            # Find result in parquet data
            result_df = matches_df[
                (matches_df["home_player_id"] == home_id) &
                (matches_df["away_player_id"] == away_id)
            ]

            if "match_date" in result_df.columns:
                result_df = result_df[result_df["match_date"].astype(str).str[:10] == match_date]

            if result_df.empty:
                # Try reversed (away/home)
                result_df = matches_df[
                    (matches_df["home_player_id"] == away_id) &
                    (matches_df["away_player_id"] == home_id)
                ]
                if "match_date" in result_df.columns:
                    result_df = result_df[result_df["match_date"].astype(str).str[:10] == match_date]

                if not result_df.empty:
                    # Reversed — away was listed as home
                    row = result_df.iloc[0]
                    is_set = bool(row.get("is_set_format", False))
                    if is_set:
                        home_won = row["away_sets"] > row["home_sets"]
                    else:
                        home_won = row["away_legs"] > row["home_legs"]

                    score = f"{row.get('away_sets', row.get('away_legs', '?'))}-{row.get('home_sets', row.get('home_legs', '?'))}"
                    self.tracker.settle_match(
                        match["id"], home_won,
                        home_score=str(score.split("-")[0]),
                        away_score=str(score.split("-")[1]),
                    )
                    summary["bets_settled"] += 1
                    continue

                logger.debug(
                    f"No result found for {match['home_player_name']} vs {match['away_player_name']} on {match_date}"
                )
                continue

            row = result_df.iloc[0]
            is_set = bool(row.get("is_set_format", False))
            if is_set:
                home_won = row["home_sets"] > row["away_sets"]
            else:
                home_won = row["home_legs"] > row["away_legs"]

            home_score = str(row.get("home_sets" if is_set else "home_legs", "?"))
            away_score = str(row.get("away_sets" if is_set else "away_legs", "?"))

            self.tracker.settle_match(
                match["id"], home_won,
                home_score=home_score,
                away_score=away_score,
            )
            summary["bets_settled"] += 1
            logger.info(
                f"Settled: {match['home_player_name']} vs {match['away_player_name']} "
                f"({home_score}-{away_score})"
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_player_name(self, player_id: int) -> Optional[str]:
        """Look up player name by ID."""
        for p in self.players_list:
            if p["player_id"] == player_id:
                return p["name"]
        return None
