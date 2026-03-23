"""
Data Models for Darts Pipeline

Defines the schema for all entities in the darts analytics system.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Dict, List, Optional, Any


class StatField(IntEnum):
    """Standard statistic field IDs from Statorium API."""
    AVERAGE_3_DARTS = 1
    THROWN_180 = 2
    THROWN_OVER_140 = 3
    THROWN_OVER_100 = 4
    HIGHEST_CHECKOUT = 5
    CHECKOUTS_OVER_100 = 6
    CHECKOUTS_ACCURACY = 7


@dataclass
class Player:
    """
    Player entity.

    Attributes:
        player_id: Unique identifier (participantID from API)
        name: Display name
        short_name: Abbreviated name
        full_name: Complete name
    """
    player_id: int
    name: str
    short_name: Optional[str] = None
    full_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "short_name": self.short_name,
            "full_name": self.full_name,
        }

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Player":
        """Create Player from API participant data."""
        return cls(
            player_id=data["participantID"],
            name=data.get("participantName", ""),
            short_name=data.get("particShortName"),
            full_name=data.get("particFullName"),
        )


@dataclass
class League:
    """
    League/Tournament entity.

    Attributes:
        league_id: Unique identifier
        league_name: Name of the league/tournament
    """
    league_id: int
    league_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "league_id": self.league_id,
            "league_name": self.league_name,
        }

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "League":
        """Create League from API data."""
        return cls(
            league_id=data["id"],
            league_name=data["name"],
        )


@dataclass
class Season:
    """
    Season entity.

    Attributes:
        season_id: Unique identifier
        league_id: Parent league ID
        season_name: Season name (typically year)
        full_name: Full descriptive name
        year: Extracted year (if parseable)
    """
    season_id: int
    league_id: int
    season_name: str
    full_name: Optional[str] = None
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "season_id": self.season_id,
            "league_id": self.league_id,
            "season_name": self.season_name,
            "full_name": self.full_name,
            "year": self.year,
        }

    @classmethod
    def from_api(cls, data: Dict[str, Any], league_id: int) -> "Season":
        """Create Season from API data."""
        season_name = data.get("seasonName", "")
        year = None
        try:
            year = int(season_name)
        except (ValueError, TypeError):
            pass

        return cls(
            season_id=data["seasonID"],
            league_id=league_id,
            season_name=season_name,
            full_name=data.get("fullName"),
            year=year,
        )


@dataclass
class Match:
    """
    Match entity.

    Attributes:
        match_id: Unique identifier
        league_id: Parent league ID
        season_id: Parent season ID
        matchday_id: Matchday/Round ID
        match_date: Date of match
        venue_id: Venue identifier
        venue_name: Venue name
        home_player_id: Home player ID
        away_player_id: Away player ID
        home_sets: Sets won by home player
        away_sets: Sets won by away player
        home_legs: Total legs won by home player
        away_legs: Total legs won by away player
        start_score: Starting score (e.g., 501)
        best_of_sets: Best of X sets
        best_of_legs: Best of X legs (per set or total)
        has_visit_data: Whether visit-level data is available
        is_set_format: Whether match uses sets (vs legs only)
        matchday_name: Name of matchday/round
        matchday_type: Type of matchday
    """
    match_id: int
    league_id: int
    season_id: int
    matchday_id: int
    match_date: Optional[date] = None
    venue_id: Optional[int] = None
    venue_name: Optional[str] = None
    home_player_id: Optional[int] = None
    away_player_id: Optional[int] = None
    home_sets: int = 0
    away_sets: int = 0
    home_legs: int = 0
    away_legs: int = 0
    start_score: int = 501
    best_of_sets: int = 0
    best_of_legs: int = 0
    has_visit_data: bool = False
    is_set_format: bool = False
    matchday_name: Optional[str] = None
    matchday_type: Optional[int] = None
    status_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league_id": self.league_id,
            "season_id": self.season_id,
            "matchday_id": self.matchday_id,
            "match_date": (
                self.match_date.isoformat()
                if hasattr(self.match_date, 'isoformat')
                else self.match_date
            ) if self.match_date else None,
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "home_player_id": self.home_player_id,
            "away_player_id": self.away_player_id,
            "home_sets": self.home_sets,
            "away_sets": self.away_sets,
            "home_legs": self.home_legs,
            "away_legs": self.away_legs,
            "start_score": self.start_score,
            "best_of_sets": self.best_of_sets,
            "best_of_legs": self.best_of_legs,
            "has_visit_data": self.has_visit_data,
            "is_set_format": self.is_set_format,
            "matchday_name": self.matchday_name,
            "matchday_type": self.matchday_type,
            "status_id": self.status_id,
        }

    @property
    def home_won(self) -> Optional[bool]:
        """Determine if home player won."""
        if self.is_set_format:
            if self.home_sets > self.away_sets:
                return True
            elif self.away_sets > self.home_sets:
                return False
        else:
            if self.home_legs > self.away_legs:
                return True
            elif self.away_legs > self.home_legs:
                return False
        return None

    @property
    def winner_id(self) -> Optional[int]:
        """Get the winner's player ID."""
        won = self.home_won
        if won is True:
            return self.home_player_id
        elif won is False:
            return self.away_player_id
        return None


@dataclass
class Leg:
    """
    Leg entity (only for matches with visit data).

    Attributes:
        match_id: Parent match ID
        league_id: League ID
        season_id: Season ID
        set_no: Set number (1-indexed)
        leg_no: Leg number within set (1-indexed)
        leg_winner_player_id: Player who won the leg
        home_throws_count: Number of throws by home player
        away_throws_count: Number of throws by away player
        home_180s: 180s by home player in this leg
        away_180s: 180s by away player in this leg
    """
    match_id: int
    league_id: int
    season_id: int
    set_no: int
    leg_no: int
    leg_winner_player_id: Optional[int] = None
    home_throws_count: int = 0
    away_throws_count: int = 0
    home_180s: int = 0
    away_180s: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league_id": self.league_id,
            "season_id": self.season_id,
            "set_no": self.set_no,
            "leg_no": self.leg_no,
            "leg_winner_player_id": self.leg_winner_player_id,
            "home_throws_count": self.home_throws_count,
            "away_throws_count": self.away_throws_count,
            "home_180s": self.home_180s,
            "away_180s": self.away_180s,
        }


@dataclass
class Visit:
    """
    Visit (throw) entity - a single scoring visit in a leg.

    Attributes:
        match_id: Parent match ID
        league_id: League ID
        season_id: Season ID
        set_no: Set number
        leg_no: Leg number
        visit_index: Order of visit within the leg (0-indexed)
        player_id: Player making the visit
        score: Points scored
        attempts: Number of darts thrown
        is_180: Whether this was a 180
        is_140_plus: Whether score >= 140
        is_100_plus: Whether score >= 100
        is_checkout: Whether this completed the leg
    """
    match_id: int
    league_id: int
    season_id: int
    set_no: int
    leg_no: int
    visit_index: int
    player_id: int
    score: int
    attempts: int
    is_180: bool = False
    is_140_plus: bool = False
    is_100_plus: bool = False
    is_checkout: bool = False

    def __post_init__(self):
        self.is_180 = (self.score == 180)
        self.is_140_plus = (self.score >= 140)
        self.is_100_plus = (self.score >= 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league_id": self.league_id,
            "season_id": self.season_id,
            "set_no": self.set_no,
            "leg_no": self.leg_no,
            "visit_index": self.visit_index,
            "player_id": self.player_id,
            "score": self.score,
            "attempts": self.attempts,
            "is_180": self.is_180,
            "is_140_plus": self.is_140_plus,
            "is_100_plus": self.is_100_plus,
            "is_checkout": self.is_checkout,
        }


@dataclass
class MatchStat:
    """
    Match statistics for a player.

    Attributes:
        match_id: Parent match ID
        player_id: Player ID
        stat_field_id: Field identifier
        stat_field_name: Field name
        value: Statistic value (can be int or float)
    """
    match_id: int
    player_id: int
    stat_field_id: int
    stat_field_name: str
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "player_id": self.player_id,
            "stat_field_id": self.stat_field_id,
            "stat_field_name": self.stat_field_name,
            "value": self.value,
        }

    @property
    def numeric_value(self) -> Optional[float]:
        """Get value as float, handling string values."""
        if self.value is None or self.value == "":
            return None
        try:
            return float(self.value)
        except (ValueError, TypeError):
            return None


@dataclass
class FormatParams:
    """
    Match format parameters for predictions.

    Attributes:
        start_score: Starting score (e.g., 501, 301)
        best_of_sets: Best of X sets (0 for leg-only formats)
        best_of_legs: Best of X legs (per set for set formats, or total for leg-only)
        is_set_format: Whether using sets format

    Examples:
        # First to 6 legs (best of 11)
        FormatParams.first_to_legs(6)

        # Best of 7 sets, best of 5 legs per set (World Championship final)
        FormatParams.sets_format(best_of_sets=13, best_of_legs=5)

        # Premier League format: best of 11 legs
        FormatParams.first_to_legs(6)
    """
    start_score: int = 501
    best_of_sets: int = 0
    best_of_legs: int = 5
    is_set_format: bool = False

    @classmethod
    def first_to_legs(cls, first_to: int, start_score: int = 501) -> "FormatParams":
        """
        Create format for 'first to X legs' (leg-only format).

        Args:
            first_to: Number of legs to win (e.g., 6 for first to 6)
            start_score: Starting score (default 501)

        Returns:
            FormatParams configured for leg-only format

        Example:
            # Players Championship: first to 6 legs
            params = FormatParams.first_to_legs(6)
        """
        best_of = 2 * first_to - 1  # first to 6 = best of 11
        return cls(
            start_score=start_score,
            best_of_sets=0,
            best_of_legs=best_of,
            is_set_format=False,
        )

    @classmethod
    def sets_format(
        cls,
        best_of_sets: int,
        best_of_legs: int = 5,
        start_score: int = 501,
    ) -> "FormatParams":
        """
        Create format for set-based matches.

        Args:
            best_of_sets: Best of X sets (e.g., 13 for World Championship final)
            best_of_legs: Best of X legs per set (default 5)
            start_score: Starting score (default 501)

        Returns:
            FormatParams configured for set format

        Example:
            # World Championship final: best of 13 sets, best of 5 legs
            params = FormatParams.sets_format(13, 5)
        """
        return cls(
            start_score=start_score,
            best_of_sets=best_of_sets,
            best_of_legs=best_of_legs,
            is_set_format=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_score": self.start_score,
            "best_of_sets": self.best_of_sets,
            "best_of_legs": self.best_of_legs,
            "is_set_format": self.is_set_format,
        }

    @property
    def expected_max_legs(self) -> int:
        """Estimate maximum possible legs in match."""
        if self.is_set_format:
            max_sets = self.best_of_sets
            legs_per_set = self.best_of_legs
            return max_sets * legs_per_set
        else:
            return self.best_of_legs

    @property
    def first_to_legs_value(self) -> int:
        """For leg-only formats, return the 'first to X' value."""
        if self.is_set_format:
            raise ValueError("This is a set format, not first-to-legs")
        return (self.best_of_legs + 1) // 2

    @property
    def first_to_sets_value(self) -> int:
        """For set formats, return the 'first to X sets' value."""
        if not self.is_set_format:
            raise ValueError("This is a leg-only format, not sets")
        return (self.best_of_sets + 1) // 2
