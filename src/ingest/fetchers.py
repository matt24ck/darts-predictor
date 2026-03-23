"""
Data Fetchers for Statorium Darts API

Modules for fetching leagues, seasons, match lists, and match details.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schema.models import League, Season, Match, Player
from .api_client import StatoriumClient

logger = logging.getLogger(__name__)


class LeagueSeasonFetcher:
    """
    Fetches league and season data from the API.

    Handles:
    - Fetching all leagues
    - Fetching seasons for specific leagues
    - Filtering to the most recent N seasons
    """

    def __init__(self, client: StatoriumClient):
        """
        Initialize the fetcher.

        Args:
            client: Statorium API client
        """
        self.client = client

    def fetch_all_leagues(self) -> List[League]:
        """
        Fetch all available leagues.

        Returns:
            List of League objects
        """
        response = self.client.get_leagues()
        leagues = []

        for league_data in response.get("leagues", []):
            league = League.from_api(league_data)
            leagues.append(league)
            logger.debug(f"Fetched league: {league.league_name} (ID: {league.league_id})")

        logger.info(f"Fetched {len(leagues)} leagues")
        return leagues

    def fetch_seasons_for_league(
        self,
        league_id: int,
        num_seasons: int = 5,
    ) -> Tuple[League, List[Season]]:
        """
        Fetch seasons for a specific league.

        Args:
            league_id: ID of the league
            num_seasons: Number of most recent seasons to return

        Returns:
            Tuple of (League, list of Season objects)
        """
        response = self.client.get_league(league_id)
        league_data = response.get("league", {})

        league = League(
            league_id=league_data.get("id", league_id),
            league_name=league_data.get("name", "Unknown"),
        )

        seasons = []
        for season_data in league_data.get("seasons", []):
            season = Season.from_api(season_data, league_id)
            seasons.append(season)

        # Sort by year (descending) to get most recent first
        seasons = self._sort_seasons_by_recency(seasons)

        # Take only the requested number of most recent seasons
        selected_seasons = seasons[:num_seasons]

        # Log the years that were selected
        selected_years = [self._extract_year(s) for s in selected_seasons]
        logger.info(
            f"Selected {len(selected_seasons)} most recent seasons for {league.league_name}: "
            f"years {selected_years} (from {len(seasons)} total with valid years)"
        )

        return league, selected_seasons

    def _sort_seasons_by_recency(self, seasons: List[Season]) -> List[Season]:
        """
        Sort seasons by year (most recent first).

        Only includes seasons where we can reliably extract a year.
        Seasons without a valid year are excluded to avoid incorrect ordering.
        """
        import re

        # Separate seasons with valid years from those without
        seasons_with_year = []
        seasons_without_year = []

        for s in seasons:
            year = self._extract_year(s)
            if year is not None:
                seasons_with_year.append((s, year))
            else:
                seasons_without_year.append(s)
                logger.warning(
                    f"Could not extract year from season '{s.season_name}' "
                    f"(ID: {s.season_id}) - excluding from selection"
                )

        # Sort by year descending (most recent first)
        seasons_with_year.sort(key=lambda x: x[1], reverse=True)

        # Return sorted seasons (only those with valid years)
        sorted_seasons = [s for s, year in seasons_with_year]

        if sorted_seasons:
            years = [self._extract_year(s) for s in sorted_seasons[:5]]
            logger.debug(f"Sorted seasons by year (most recent first): {years}...")

        return sorted_seasons

    def _extract_year(self, season: Season) -> Optional[int]:
        """
        Extract year from a season.

        Tries multiple approaches:
        1. Use pre-parsed year from Season object
        2. Parse season_name as integer (if it looks like a year)
        3. Extract 4-digit year from season_name string (e.g., "2024-25" -> 2024)
        4. Handle "Series X" format (MODUS) - map to pseudo-year for sorting

        Returns:
            Year as integer, or None if not extractable
        """
        import re

        # Try pre-parsed year first
        if season.year is not None and 1990 <= season.year <= 2100:
            return season.year

        # Try parsing season_name directly as a year
        try:
            year = int(season.season_name)
            if 1990 <= year <= 2100:  # Sanity check for valid year range
                return year
        except (ValueError, TypeError):
            pass

        # Try extracting 4-digit year from string (e.g., "2024-25", "Season 2024")
        if season.season_name:
            match = re.search(r'\b(19|20)\d{2}\b', season.season_name)
            if match:
                return int(match.group())

        # Try from full_name if available
        if season.full_name:
            match = re.search(r'\b(19|20)\d{2}\b', season.full_name)
            if match:
                return int(match.group())

        # Handle "Series X" format (e.g., MODUS Super Series)
        # Map series number to pseudo-year: Series 13 -> 3013 (ensures proper ordering)
        if season.season_name:
            series_match = re.search(r'[Ss]eries\s*(\d+)', season.season_name)
            if series_match:
                series_num = int(series_match.group(1))
                # Use 3000 + series_num as pseudo-year to sort above real years
                return 3000 + series_num

        return None

    def fetch_all_configured_seasons(
        self,
        league_ids: Dict[int, str],
        num_seasons: int = 5,
    ) -> Tuple[List[League], List[Season]]:
        """
        Fetch seasons for all configured leagues.

        Args:
            league_ids: Dict mapping league_id -> league_name
            num_seasons: Number of seasons per league

        Returns:
            Tuple of (list of Leagues, list of Seasons)
        """
        all_leagues = []
        all_seasons = []

        for league_id, expected_name in league_ids.items():
            try:
                league, seasons = self.fetch_seasons_for_league(league_id, num_seasons)
                all_leagues.append(league)
                all_seasons.extend(seasons)
                logger.info(f"Processed league: {league.league_name}")
            except Exception as e:
                logger.error(f"Failed to fetch seasons for league {league_id}: {e}")

        return all_leagues, all_seasons


class MatchListFetcher:
    """
    Fetches match lists from season calendars.

    Parses knockout brackets and matchday structures to extract match IDs.
    """

    def __init__(self, client: StatoriumClient):
        """
        Initialize the fetcher.

        Args:
            client: Statorium API client
        """
        self.client = client

    def fetch_matches_for_season(
        self,
        season: Season,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all match references for a season.

        Args:
            season: Season to fetch matches for

        Returns:
            List of match info dicts containing match_id and metadata
        """
        response = self.client.get_matches_by_season(season.season_id)
        calendar = response.get("calendar", {})

        match_infos = []
        matchdays = calendar.get("matchdays", [])

        for matchday in matchdays:
            matchday_id = matchday.get("matchdayID")
            matchday_name = matchday.get("matchdayName")
            matchday_type = matchday.get("matchdayType")

            # Parse knockout bracket
            knock_bracket = matchday.get("knockBracket", [])
            bracket_match_ids = self._extract_match_ids_from_bracket(knock_bracket)

            for match_id in bracket_match_ids:
                match_infos.append({
                    "match_id": match_id,
                    "league_id": season.league_id,
                    "season_id": season.season_id,
                    "matchday_id": matchday_id,
                    "matchday_name": matchday_name,
                    "matchday_type": matchday_type,
                })

            # Also parse inline matches list if present
            inline_matches = matchday.get("matches", [])
            for match in inline_matches:
                mid = match.get("matchID")
                if mid and mid not in [m["match_id"] for m in match_infos]:
                    match_infos.append({
                        "match_id": mid,
                        "league_id": season.league_id,
                        "season_id": season.season_id,
                        "matchday_id": matchday_id,
                        "matchday_name": matchday_name,
                        "matchday_type": matchday_type,
                    })

        logger.info(
            f"Found {len(match_infos)} matches for season {season.season_name} "
            f"(league {season.league_id})"
        )

        return match_infos

    def _extract_match_ids_from_bracket(
        self,
        bracket: List[Any],
    ) -> Set[int]:
        """
        Recursively extract match IDs from knockout bracket structure.

        The bracket can be:
        - A list of dicts with "match_id" keys
        - A dict where values are dicts with "match_id" keys
        - Nested combinations of the above

        Args:
            bracket: Knockout bracket data

        Returns:
            Set of match IDs
        """
        match_ids = set()

        if isinstance(bracket, list):
            for item in bracket:
                if isinstance(item, dict):
                    # Direct match entry
                    if "match_id" in item:
                        mid = item["match_id"]
                        if isinstance(mid, list):
                            for m in mid:
                                if isinstance(m, int) and m > 0:
                                    match_ids.add(m)
                        elif isinstance(mid, int) and mid > 0:
                            match_ids.add(mid)
                    # Nested structure
                    else:
                        for value in item.values():
                            if isinstance(value, dict) and "match_id" in value:
                                mid = value["match_id"]
                                if isinstance(mid, list):
                                    for m in mid:
                                        if isinstance(m, int) and m > 0:
                                            match_ids.add(m)
                                elif isinstance(mid, int) and mid > 0:
                                    match_ids.add(mid)
                            elif isinstance(value, (list, dict)):
                                match_ids.update(self._extract_match_ids_from_bracket(value))
                elif isinstance(item, list):
                    match_ids.update(self._extract_match_ids_from_bracket(item))

        elif isinstance(bracket, dict):
            for key, value in bracket.items():
                if key == "match_id":
                    if isinstance(value, list):
                        for m in value:
                            if isinstance(m, int) and m > 0:
                                match_ids.add(m)
                    elif isinstance(value, int) and value > 0:
                        match_ids.add(value)
                elif isinstance(value, dict):
                    match_ids.update(self._extract_match_ids_from_bracket(value))
                elif isinstance(value, list):
                    match_ids.update(self._extract_match_ids_from_bracket(value))

        return match_ids


class MatchDetailFetcher:
    """
    Fetches detailed match data by match ID.

    Returns raw match data for parsing.
    """

    def __init__(self, client: StatoriumClient):
        """
        Initialize the fetcher.

        Args:
            client: Statorium API client
        """
        self.client = client

    def fetch_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed match data.

        Args:
            match_id: ID of the match

        Returns:
            Raw match data dict, or None on failure
        """
        try:
            response = self.client.get_match(match_id)
            return response.get("match", {})
        except Exception as e:
            logger.error(f"Failed to fetch match {match_id}: {e}")
            return None

    def fetch_matches_batch(
        self,
        match_infos: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Fetch multiple matches.

        Args:
            match_infos: List of match info dicts from MatchListFetcher
            progress_callback: Optional callback(current, total)

        Returns:
            List of (match_info, match_data) tuples
        """
        results = []
        total = len(match_infos)

        for i, match_info in enumerate(match_infos):
            match_id = match_info["match_id"]
            match_data = self.fetch_match(match_id)

            if match_data:
                results.append((match_info, match_data))
            else:
                logger.warning(f"Skipping match {match_id} - no data returned")

            if progress_callback:
                progress_callback(i + 1, total)

        logger.info(f"Fetched {len(results)}/{total} matches successfully")
        return results
