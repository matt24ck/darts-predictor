"""
Match Data Parsers

Parses raw API match data into structured schema objects.
Handles both visit-level and aggregate-only match data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..schema.models import (
    Match,
    Leg,
    Visit,
    MatchStat,
    Player,
    StatField,
)

logger = logging.getLogger(__name__)


class MatchParser:
    """
    Parses raw match data from Statorium API.

    Supports:
    - Visit-level matches (with sets/legs/throws data)
    - Aggregate-only matches (statistics only)
    """

    def parse_match(
        self,
        match_data: Dict[str, Any],
        match_info: Dict[str, Any],
    ) -> Tuple[
        Match,
        List[Player],
        List[Leg],
        List[Visit],
        List[MatchStat],
    ]:
        """
        Parse raw match data into structured objects.

        Args:
            match_data: Raw match data from API
            match_info: Metadata from match list (league_id, season_id, etc.)

        Returns:
            Tuple of (Match, players, legs, visits, stats)
        """
        match_id = match_data.get("matchID", match_info.get("match_id"))
        league_id = match_info.get("league_id")
        season_id = match_info.get("season_id")
        matchday_id = match_data.get("matchdayID", match_info.get("matchday_id"))

        # Parse participants
        home_participant = match_data.get("homeParticipant", {})
        away_participant = match_data.get("awayParticipant", {})

        home_player = self._parse_player(home_participant)
        away_player = self._parse_player(away_participant)
        players = [p for p in [home_player, away_player] if p is not None]

        # Parse match metadata
        match_date = self._parse_date(match_data.get("matchDate"))
        venue = match_data.get("matchVenue", {})

        # Parse format
        start_score = self._parse_int(match_data.get("startScore"), 501)
        best_of_sets = self._parse_int(match_data.get("bestOfSets"), 0)
        best_of_legs = self._parse_int(match_data.get("bestOfLegs"), 0)
        is_set_format = best_of_sets > 0

        # Parse scores
        home_sets = 0
        away_sets = 0
        home_legs = 0
        away_legs = 0

        score_data = match_data.get("score", {})
        detailed = score_data.get("detailed", {})

        if is_set_format and detailed:
            # Count sets from detailed score
            for set_num, set_score in detailed.items():
                h_legs = set_score.get("homeLegs", 0)
                a_legs = set_score.get("awayLegs", 0)
                home_legs += h_legs
                away_legs += a_legs
                if h_legs > a_legs:
                    home_sets += 1
                elif a_legs > h_legs:
                    away_sets += 1
        else:
            # Use participant scores directly
            home_sets = home_participant.get("score", 0) if is_set_format else 0
            away_sets = away_participant.get("score", 0) if is_set_format else 0
            if not is_set_format:
                home_legs = home_participant.get("score", 0)
                away_legs = away_participant.get("score", 0)

        # Check for visit-level data
        sets_data = score_data.get("sets", [])
        has_visit_data = self._has_visit_data(sets_data)

        # Create Match object
        match = Match(
            match_id=match_id,
            league_id=league_id,
            season_id=season_id,
            matchday_id=matchday_id,
            match_date=match_date,
            venue_id=venue.get("venueID"),
            venue_name=venue.get("venueName"),
            home_player_id=home_player.player_id if home_player else None,
            away_player_id=away_player.player_id if away_player else None,
            home_sets=home_sets,
            away_sets=away_sets,
            home_legs=home_legs,
            away_legs=away_legs,
            start_score=start_score,
            best_of_sets=best_of_sets,
            best_of_legs=best_of_legs,
            has_visit_data=has_visit_data,
            is_set_format=is_set_format,
            matchday_name=match_info.get("matchday_name"),
            matchday_type=match_info.get("matchday_type"),
            status_id=match_data.get("matchStatus", {}).get("statusID"),
        )

        # Parse legs and visits (if visit data available)
        legs = []
        visits = []

        if has_visit_data and home_player and away_player:
            legs, visits = self._parse_visit_data(
                sets_data=sets_data,
                match_id=match_id,
                league_id=league_id,
                season_id=season_id,
                home_player_id=home_player.player_id,
                away_player_id=away_player.player_id,
            )

        # Parse statistics
        stats = self._parse_statistics(
            match_data.get("statistic", []),
            match_id=match_id,
            home_player_id=home_player.player_id if home_player else None,
            away_player_id=away_player.player_id if away_player else None,
        )

        # Validate 180s count if we have both visit data and stats
        if has_visit_data and stats:
            self._validate_180s_count(visits, stats)

        return match, players, legs, visits, stats

    def _parse_player(self, participant: Dict[str, Any]) -> Optional[Player]:
        """Parse player from participant data."""
        player_id = participant.get("participantID")
        if not player_id:
            return None

        name = participant.get("participantName", "")
        # Skip players with missing or empty names
        if not name or not name.strip():
            logger.warning(f"Skipping player {player_id} with missing name")
            return None

        return Player(
            player_id=player_id,
            name=name,
            short_name=participant.get("particShortName"),
            full_name=participant.get("particFullName"),
        )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Could not parse date: {date_str}")
            return None

    def _parse_int(self, value: Any, default: int = 0) -> int:
        """Parse value to int, returning default on failure."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _has_visit_data(self, sets_data: List[Dict[str, Any]]) -> bool:
        """Check if sets data contains visit-level throws."""
        if not sets_data:
            return False

        for set_info in sets_data:
            legs = set_info.get("legs", [])
            for leg in legs:
                throws = leg.get("throws", [])
                if throws:
                    return True
        return False

    def _parse_visit_data(
        self,
        sets_data: List[Dict[str, Any]],
        match_id: int,
        league_id: int,
        season_id: int,
        home_player_id: int,
        away_player_id: int,
    ) -> Tuple[List[Leg], List[Visit]]:
        """
        Parse visit-level data from sets/legs/throws structure.

        Args:
            sets_data: List of set data
            match_id: Match ID
            league_id: League ID
            season_id: Season ID
            home_player_id: Home player ID
            away_player_id: Away player ID

        Returns:
            Tuple of (legs, visits)
        """
        legs = []
        visits = []

        for set_info in sets_data:
            set_no = set_info.get("set", 1)

            for leg_info in set_info.get("legs", []):
                leg_no = leg_info.get("leg", 1)
                throws = leg_info.get("throws", [])

                # Track per-leg stats
                home_throws = 0
                away_throws = 0
                home_180s = 0
                away_180s = 0
                leg_winner = None

                for visit_index, throw in enumerate(throws):
                    is_home = throw.get("isHome", 0) == 1
                    player_id = home_player_id if is_home else away_player_id
                    score = throw.get("score", 0)
                    attempts = throw.get("attempts", 3)

                    # Create visit
                    visit = Visit(
                        match_id=match_id,
                        league_id=league_id,
                        season_id=season_id,
                        set_no=set_no,
                        leg_no=leg_no,
                        visit_index=visit_index,
                        player_id=player_id,
                        score=score,
                        attempts=attempts,
                    )

                    # Check if this is the checkout (last throw wins the leg)
                    # We'll determine leg winner from last throw
                    leg_winner = player_id

                    visits.append(visit)

                    # Update counts
                    if is_home:
                        home_throws += 1
                        if visit.is_180:
                            home_180s += 1
                    else:
                        away_throws += 1
                        if visit.is_180:
                            away_180s += 1

                # Mark last visit as checkout
                if visits and throws:
                    visits[-1].is_checkout = True

                # Create leg
                leg = Leg(
                    match_id=match_id,
                    league_id=league_id,
                    season_id=season_id,
                    set_no=set_no,
                    leg_no=leg_no,
                    leg_winner_player_id=leg_winner,
                    home_throws_count=home_throws,
                    away_throws_count=away_throws,
                    home_180s=home_180s,
                    away_180s=away_180s,
                )
                legs.append(leg)

        return legs, visits

    def _parse_statistics(
        self,
        stats_data: List[Dict[str, Any]],
        match_id: int,
        home_player_id: Optional[int],
        away_player_id: Optional[int],
    ) -> List[MatchStat]:
        """
        Parse match statistics.

        Args:
            stats_data: Raw statistics from API
            match_id: Match ID
            home_player_id: Home player ID
            away_player_id: Away player ID

        Returns:
            List of MatchStat objects
        """
        stats = []

        for stat in stats_data:
            field_id = stat.get("fieldID")
            field_name = stat.get("fieldName", "")
            home_value = stat.get("homeValue")
            away_value = stat.get("awayValue")

            # Create stat for home player
            if home_player_id is not None:
                stats.append(MatchStat(
                    match_id=match_id,
                    player_id=home_player_id,
                    stat_field_id=field_id,
                    stat_field_name=field_name,
                    value=home_value,
                ))

            # Create stat for away player
            if away_player_id is not None:
                stats.append(MatchStat(
                    match_id=match_id,
                    player_id=away_player_id,
                    stat_field_id=field_id,
                    stat_field_name=field_name,
                    value=away_value,
                ))

        return stats

    def _validate_180s_count(
        self,
        visits: List[Visit],
        stats: List[MatchStat],
    ) -> None:
        """
        Validate that 180s counted from visits matches the statistics.

        Logs a warning if there's a mismatch.
        """
        # Count 180s from visits by player
        visit_180s = {}
        for visit in visits:
            if visit.is_180:
                visit_180s[visit.player_id] = visit_180s.get(visit.player_id, 0) + 1

        # Get 180s from stats
        for stat in stats:
            if stat.stat_field_id == StatField.THROWN_180:
                player_id = stat.player_id
                stat_value = stat.numeric_value
                visit_count = visit_180s.get(player_id, 0)

                if stat_value is not None and stat_value != visit_count:
                    logger.warning(
                        f"180s mismatch for player {player_id}: "
                        f"stats={stat_value}, visits={visit_count}"
                    )


class ParsedMatchData:
    """Container for all parsed data from a match."""

    def __init__(
        self,
        match: Match,
        players: List[Player],
        legs: List[Leg],
        visits: List[Visit],
        stats: List[MatchStat],
    ):
        self.match = match
        self.players = players
        self.legs = legs
        self.visits = visits
        self.stats = stats

    @property
    def home_180s(self) -> int:
        """Get total 180s for home player from visits or stats."""
        if self.visits:
            return sum(
                1 for v in self.visits
                if v.is_180 and v.player_id == self.match.home_player_id
            )
        # Fall back to stats
        for stat in self.stats:
            if (stat.stat_field_id == StatField.THROWN_180 and
                    stat.player_id == self.match.home_player_id):
                val = stat.numeric_value
                return int(val) if val else 0
        return 0

    @property
    def away_180s(self) -> int:
        """Get total 180s for away player from visits or stats."""
        if self.visits:
            return sum(
                1 for v in self.visits
                if v.is_180 and v.player_id == self.match.away_player_id
            )
        # Fall back to stats
        for stat in self.stats:
            if (stat.stat_field_id == StatField.THROWN_180 and
                    stat.player_id == self.match.away_player_id):
                val = stat.numeric_value
                return int(val) if val else 0
        return 0

    @property
    def total_180s(self) -> int:
        """Get total 180s in match."""
        return self.home_180s + self.away_180s
