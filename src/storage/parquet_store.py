"""
Parquet-based Data Store

Implements data storage using Parquet files for efficient analytics.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import DataStore
from ..schema.models import (
    Player,
    League,
    Season,
    Match,
    Leg,
    Visit,
    MatchStat,
)

logger = logging.getLogger(__name__)


class ParquetStore(DataStore):
    """
    Parquet-based data storage implementation.

    Stores each entity type in a separate Parquet file with
    efficient columnar storage for analytics queries.
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the Parquet store.

        Args:
            data_dir: Directory for storing Parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.players_path = self.data_dir / "players.parquet"
        self.leagues_path = self.data_dir / "leagues.parquet"
        self.seasons_path = self.data_dir / "seasons.parquet"
        self.matches_path = self.data_dir / "matches.parquet"
        self.legs_path = self.data_dir / "legs.parquet"
        self.visits_path = self.data_dir / "visits.parquet"
        self.match_stats_path = self.data_dir / "match_stats.parquet"

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    def save_players(self, players: List[Player]) -> int:
        """Save players to storage (upsert)."""
        if not players:
            return 0

        # Filter out players with missing/empty names
        valid_players = [
            p for p in players
            if p.name and p.name.strip()
        ]

        if not valid_players:
            return 0

        new_df = pd.DataFrame([p.to_dict() for p in valid_players])

        # Merge with existing
        if self.players_path.exists():
            existing_df = pd.read_parquet(self.players_path)
            # Update existing, add new
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["player_id"], keep="last"
            )
        else:
            combined = new_df

        # Filter out any rows with null/empty names
        combined = combined[
            combined["name"].notna() &
            (combined["name"].str.strip() != "")
        ]

        combined.to_parquet(self.players_path, index=False)
        return len(valid_players)

    def save_leagues(self, leagues: List[League]) -> int:
        """Save leagues to storage (upsert)."""
        if not leagues:
            return 0

        new_df = pd.DataFrame([l.to_dict() for l in leagues])

        if self.leagues_path.exists():
            existing_df = pd.read_parquet(self.leagues_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["league_id"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.leagues_path, index=False)
        return len(leagues)

    def save_seasons(self, seasons: List[Season]) -> int:
        """Save seasons to storage (upsert)."""
        if not seasons:
            return 0

        new_df = pd.DataFrame([s.to_dict() for s in seasons])

        if self.seasons_path.exists():
            existing_df = pd.read_parquet(self.seasons_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["season_id"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.seasons_path, index=False)
        return len(seasons)

    def save_matches(self, matches: List[Match]) -> int:
        """Save matches to storage (upsert)."""
        if not matches:
            return 0

        new_df = pd.DataFrame([m.to_dict() for m in matches])

        if self.matches_path.exists():
            existing_df = pd.read_parquet(self.matches_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["match_id"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.matches_path, index=False)
        return len(matches)

    def save_legs(self, legs: List[Leg]) -> int:
        """Save legs to storage (append, dedup on composite key)."""
        if not legs:
            return 0

        new_df = pd.DataFrame([l.to_dict() for l in legs])

        if self.legs_path.exists():
            existing_df = pd.read_parquet(self.legs_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["match_id", "set_no", "leg_no"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.legs_path, index=False)
        return len(legs)

    def save_visits(self, visits: List[Visit]) -> int:
        """Save visits to storage (append, dedup on composite key)."""
        if not visits:
            return 0

        new_df = pd.DataFrame([v.to_dict() for v in visits])

        if self.visits_path.exists():
            existing_df = pd.read_parquet(self.visits_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["match_id", "set_no", "leg_no", "visit_index"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.visits_path, index=False)
        return len(visits)

    def save_match_stats(self, stats: List[MatchStat]) -> int:
        """Save match statistics to storage."""
        if not stats:
            return 0

        new_df = pd.DataFrame([s.to_dict() for s in stats])

        # Convert 'value' column to string to handle mixed types (int, float, str)
        # Parquet requires consistent types within a column
        if "value" in new_df.columns:
            new_df["value"] = new_df["value"].apply(
                lambda x: str(x) if x is not None and x != "" else None
            )

        if self.match_stats_path.exists():
            existing_df = pd.read_parquet(self.match_stats_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=["match_id", "player_id", "stat_field_id"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(self.match_stats_path, index=False)
        return len(stats)

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    def _read_parquet_if_exists(self, path: Path) -> pd.DataFrame:
        """Read Parquet file if it exists, otherwise return empty DataFrame."""
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def get_players(self, player_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Get players from storage."""
        df = self._read_parquet_if_exists(self.players_path)
        if df.empty or player_ids is None:
            return df
        return df[df["player_id"].isin(player_ids)]

    def get_leagues(self, league_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Get leagues from storage."""
        df = self._read_parquet_if_exists(self.leagues_path)
        if df.empty or league_ids is None:
            return df
        return df[df["league_id"].isin(league_ids)]

    def get_seasons(
        self,
        league_ids: Optional[List[int]] = None,
        season_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Get seasons from storage."""
        df = self._read_parquet_if_exists(self.seasons_path)
        if df.empty:
            return df
        if league_ids is not None:
            df = df[df["league_id"].isin(league_ids)]
        if season_ids is not None:
            df = df[df["season_id"].isin(season_ids)]
        return df

    def get_matches(
        self,
        match_ids: Optional[List[int]] = None,
        league_ids: Optional[List[int]] = None,
        season_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
        has_visit_data: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Get matches from storage."""
        df = self._read_parquet_if_exists(self.matches_path)
        if df.empty:
            return df
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)]
        if league_ids is not None:
            df = df[df["league_id"].isin(league_ids)]
        if season_ids is not None:
            df = df[df["season_id"].isin(season_ids)]
        if player_id is not None:
            df = df[
                (df["home_player_id"] == player_id) |
                (df["away_player_id"] == player_id)
            ]
        if has_visit_data is not None:
            df = df[df["has_visit_data"] == has_visit_data]
        return df

    def get_legs(
        self,
        match_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Get legs from storage."""
        df = self._read_parquet_if_exists(self.legs_path)
        if df.empty or match_ids is None:
            return df
        return df[df["match_id"].isin(match_ids)]

    def get_visits(
        self,
        match_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get visits from storage."""
        df = self._read_parquet_if_exists(self.visits_path)
        if df.empty:
            return df
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)]
        if player_id is not None:
            df = df[df["player_id"] == player_id]
        return df

    def get_match_stats(
        self,
        match_ids: Optional[List[int]] = None,
        player_id: Optional[int] = None,
        stat_field_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get match statistics from storage."""
        df = self._read_parquet_if_exists(self.match_stats_path)
        if df.empty:
            return df
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)]
        if player_id is not None:
            df = df[df["player_id"] == player_id]
        if stat_field_id is not None:
            df = df[df["stat_field_id"] == stat_field_id]
        return df

    # -------------------------------------------------------------------------
    # Aggregate Queries
    # -------------------------------------------------------------------------

    def get_player_stats_summary(
        self,
        player_id: int,
        league_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Get summary statistics for a player."""
        matches_df = self.get_matches(player_id=player_id, league_ids=league_ids)
        visits_df = self.get_visits(player_id=player_id)
        stats_df = self.get_match_stats(player_id=player_id)

        if matches_df.empty:
            return {
                "player_id": player_id,
                "total_matches": 0,
            }

        # Filter visits to matches in scope
        if not visits_df.empty and league_ids:
            match_ids = matches_df["match_id"].tolist()
            visits_df = visits_df[visits_df["match_id"].isin(match_ids)]

        # Calculate summary
        total_matches = len(matches_df)

        # Win count
        wins = 0
        for _, match in matches_df.iterrows():
            is_home = match["home_player_id"] == player_id
            if match["is_set_format"]:
                home_won = match["home_sets"] > match["away_sets"]
            else:
                home_won = match["home_legs"] > match["away_legs"]
            if (is_home and home_won) or (not is_home and not home_won):
                wins += 1

        # Visit-level stats
        total_visits = len(visits_df) if not visits_df.empty else None
        total_180s = (
            visits_df["is_180"].sum() if not visits_df.empty and "is_180" in visits_df else None
        )
        p_180_per_visit = (
            total_180s / total_visits
            if total_visits and total_visits > 0 else None
        )

        return {
            "player_id": player_id,
            "total_matches": total_matches,
            "wins": wins,
            "win_rate": wins / total_matches if total_matches > 0 else 0,
            "total_visits": total_visits,
            "total_180s": total_180s,
            "p_180_per_visit": p_180_per_visit,
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all data from storage."""
        for path in [
            self.players_path,
            self.leagues_path,
            self.seasons_path,
            self.matches_path,
            self.legs_path,
            self.visits_path,
            self.match_stats_path,
        ]:
            if path.exists():
                path.unlink()
                logger.info(f"Deleted {path}")

    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables."""
        counts = {}
        for name, path in [
            ("players", self.players_path),
            ("leagues", self.leagues_path),
            ("seasons", self.seasons_path),
            ("matches", self.matches_path),
            ("legs", self.legs_path),
            ("visits", self.visits_path),
            ("match_stats", self.match_stats_path),
        ]:
            if path.exists():
                df = pd.read_parquet(path)
                counts[name] = len(df)
            else:
                counts[name] = 0
        return counts
