"""
Base Data Store Interface

Abstract base class for data storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd

from ..schema.models import (
    Player,
    League,
    Season,
    Match,
    Leg,
    Visit,
    MatchStat,
)


class DataStore(ABC):
    """
    Abstract base class for data storage.

    Defines the interface for storing and retrieving darts data.
    """

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_players(self, players: List[Player]) -> int:
        """
        Save players to storage.

        Args:
            players: List of Player objects

        Returns:
            Number of players saved/updated
        """
        pass

    @abstractmethod
    def save_leagues(self, leagues: List[League]) -> int:
        """
        Save leagues to storage.

        Args:
            leagues: List of League objects

        Returns:
            Number of leagues saved/updated
        """
        pass

    @abstractmethod
    def save_seasons(self, seasons: List[Season]) -> int:
        """
        Save seasons to storage.

        Args:
            seasons: List of Season objects

        Returns:
            Number of seasons saved/updated
        """
        pass

    @abstractmethod
    def save_matches(self, matches: List[Match]) -> int:
        """
        Save matches to storage.

        Args:
            matches: List of Match objects

        Returns:
            Number of matches saved/updated
        """
        pass

    @abstractmethod
    def save_legs(self, legs: List[Leg]) -> int:
        """
        Save legs to storage.

        Args:
            legs: List of Leg objects

        Returns:
            Number of legs saved
        """
        pass

    @abstractmethod
    def save_visits(self, visits: List[Visit]) -> int:
        """
        Save visits to storage.

        Args:
            visits: List of Visit objects

        Returns:
            Number of visits saved
        """
        pass

    @abstractmethod
    def save_match_stats(self, stats: List[MatchStat]) -> int:
        """
        Save match statistics to storage.

        Args:
            stats: List of MatchStat objects

        Returns:
            Number of stats saved
        """
        pass

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_players(self, player_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get players from storage.

        Args:
            player_ids: Optional list of player IDs to filter

        Returns:
            DataFrame of players
        """
        pass

    @abstractmethod
    def get_leagues(self, league_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get leagues from storage.

        Args:
            league_ids: Optional list of league IDs to filter

        Returns:
            DataFrame of leagues
        """
        pass

    @abstractmethod
    def get_seasons(
        self,
        league_ids: Optional[List[int]] = None,
        season_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get seasons from storage.

        Args:
            league_ids: Optional list of league IDs to filter
            season_ids: Optional list of season IDs to filter

        Returns:
            DataFrame of seasons
        """
        pass

    @abstractmethod
    def get_matches(
        self,
        match_ids: Optional[List[int]] = None,
        league_ids: Optional[List[int]] = None,
        season_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
        has_visit_data: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Get matches from storage.

        Args:
            match_ids: Optional list of match IDs to filter
            league_ids: Optional list of league IDs to filter
            season_ids: Optional list of season IDs to filter
            player_id: Optional player ID (matches where player participated)
            has_visit_data: Optional filter for visit data availability

        Returns:
            DataFrame of matches
        """
        pass

    @abstractmethod
    def get_legs(
        self,
        match_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get legs from storage.

        Args:
            match_ids: Optional list of match IDs to filter

        Returns:
            DataFrame of legs
        """
        pass

    @abstractmethod
    def get_visits(
        self,
        match_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get visits from storage.

        Args:
            match_ids: Optional list of match IDs to filter
            player_id: Optional player ID to filter

        Returns:
            DataFrame of visits
        """
        pass

    @abstractmethod
    def get_match_stats(
        self,
        match_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
        stat_field_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get match statistics from storage.

        Args:
            match_ids: Optional list of match IDs to filter
            player_id: Optional player ID to filter
            stat_field_id: Optional stat field ID to filter

        Returns:
            DataFrame of match stats
        """
        pass

    # -------------------------------------------------------------------------
    # Aggregate Queries
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_player_stats_summary(
        self,
        player_id: int,
        league_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a player.

        Args:
            player_id: Player ID
            league_ids: Optional league IDs to filter

        Returns:
            Dict with summary statistics
        """
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all data from storage."""
        pass

    @abstractmethod
    def get_table_counts(self) -> Dict[str, int]:
        """
        Get record counts for all tables.

        Returns:
            Dict mapping table name to record count
        """
        pass
