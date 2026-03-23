"""
The Odds API client for fetching bookmaker odds on darts matches.

Provides match discovery (events with odds) and player name matching.
"""

import logging
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SPORT_KEY = "darts_pdc"
BASE_URL = "https://api.the-odds-api.com/v4"


class OddsClient:
    """Client for The Odds API — fetches darts match events with bookmaker odds."""

    def __init__(self, api_key: str, regions: str = "uk,eu", cache_ttl: int = 1800):
        self.api_key = api_key
        self.regions = regions
        self.cache_ttl = cache_ttl  # seconds
        self._cache = {}
        self._cache_time = 0
        self._remaining_requests = None

    def fetch_darts_events(self) -> List[Dict]:
        """
        Fetch all upcoming darts events with h2h odds.

        Returns list of events, each containing:
        - id, home_team, away_team, commence_time
        - bookmakers[] with h2h odds
        """
        now = time.time()
        if self._cache and (now - self._cache_time) < self.cache_ttl:
            logger.info("Using cached odds data")
            return self._cache.get("events", [])

        if not self.api_key:
            logger.warning("No ODDS_API_KEY configured — skipping odds fetch")
            return []

        try:
            resp = requests.get(
                f"{BASE_URL}/sports/{SPORT_KEY}/odds/",
                params={
                    "apiKey": self.api_key,
                    "regions": self.regions,
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            resp.raise_for_status()

            self._remaining_requests = resp.headers.get("x-requests-remaining")
            logger.info(
                f"Odds API: fetched {len(resp.json())} events "
                f"(remaining requests: {self._remaining_requests})"
            )

            events = resp.json()
            self._cache = {"events": events}
            self._cache_time = now
            return events

        except requests.RequestException as e:
            logger.error(f"Odds API request failed: {e}")
            return self._cache.get("events", [])

    def parse_events(self, events: List[Dict], player_lookup: Dict[str, int],
                     alias_lookup: Dict[str, int]) -> List[Dict]:
        """
        Parse raw Odds API events into structured match + odds data.

        Args:
            events: Raw API response list
            player_lookup: {player_name_lower: player_id} from parquet
            alias_lookup: {alias_lower: player_id} from player_aliases table

        Returns list of dicts with:
            - event_id, home_name, away_name, commence_time
            - home_player_id, away_player_id (or None if unmatched)
            - bookmakers: [{name, home_odds, away_odds}]
            - best_home_odds, best_away_odds, best_home_bookmaker, best_away_bookmaker
        """
        parsed = []
        unmatched_names = set()

        for event in events:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")
            commence = event.get("commence_time", "")

            home_id = self._match_player(home_name, player_lookup, alias_lookup)
            away_id = self._match_player(away_name, player_lookup, alias_lookup)

            if not home_id:
                unmatched_names.add(home_name)
            if not away_id:
                unmatched_names.add(away_name)

            # Extract odds from each bookmaker
            bookmakers = []
            best_home_odds = 0
            best_away_odds = 0
            best_home_bookie = ""
            best_away_bookie = ""

            for bookie in event.get("bookmakers", []):
                bookie_name = bookie.get("title", "Unknown")
                markets = bookie.get("markets", [])

                for market in markets:
                    if market.get("key") != "h2h":
                        continue

                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    home_odds = outcomes.get(home_name, 0)
                    away_odds = outcomes.get(away_name, 0)

                    if home_odds > 0 and away_odds > 0:
                        bookmakers.append({
                            "name": bookie_name,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                        })

                        if home_odds > best_home_odds:
                            best_home_odds = home_odds
                            best_home_bookie = bookie_name
                        if away_odds > best_away_odds:
                            best_away_odds = away_odds
                            best_away_bookie = bookie_name

            # Parse commence time
            match_date = ""
            if commence:
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    match_date = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass

            parsed.append({
                "event_id": event.get("id", ""),
                "home_name": home_name,
                "away_name": away_name,
                "home_player_id": home_id,
                "away_player_id": away_id,
                "match_date": match_date,
                "commence_time": commence,
                "bookmakers": bookmakers,
                "best_home_odds": best_home_odds,
                "best_away_odds": best_away_odds,
                "best_home_bookmaker": best_home_bookie,
                "best_away_bookmaker": best_away_bookie,
            })

        if unmatched_names:
            logger.warning(f"Unmatched player names from Odds API: {unmatched_names}")

        return parsed, unmatched_names

    def _match_player(self, name: str, player_lookup: Dict[str, int],
                      alias_lookup: Dict[str, int]) -> Optional[int]:
        """Match an Odds API player name to a player_id."""
        if not name:
            return None

        name_lower = name.strip().lower()

        # 1. Exact match in player table
        if name_lower in player_lookup:
            return player_lookup[name_lower]

        # 2. Alias table
        if name_lower in alias_lookup:
            return alias_lookup[name_lower]

        # 3. Fuzzy match
        best_ratio = 0
        best_id = None
        for known_name, pid in player_lookup.items():
            ratio = SequenceMatcher(None, name_lower, known_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = pid

        if best_ratio >= 0.85:
            return best_id

        return None


def build_player_lookup(players_df) -> Dict[str, int]:
    """Build {name_lower: player_id} lookup from players DataFrame."""
    lookup = {}
    for _, row in players_df.iterrows():
        name = str(row.get("name", "")).strip().lower()
        if name:
            lookup[name] = int(row["player_id"])
        # Also add short_name and full_name
        for field in ("short_name", "full_name"):
            alt = str(row.get(field, "")).strip().lower()
            if alt and alt != "nan" and alt != "none":
                lookup[alt] = int(row["player_id"])
    return lookup


def build_alias_lookup(aliases: List[Dict]) -> Dict[str, int]:
    """Build {alias_lower: player_id} from player_aliases records."""
    return {a["alias"].lower(): a["player_id"] for a in aliases if a.get("alias")}
