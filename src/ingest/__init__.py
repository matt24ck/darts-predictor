"""Data ingestion module for darts pipeline."""

from .api_client import StatoriumClient
from .fetchers import (
    LeagueSeasonFetcher,
    MatchListFetcher,
    MatchDetailFetcher,
)
from .parsers import MatchParser

__all__ = [
    "StatoriumClient",
    "LeagueSeasonFetcher",
    "MatchListFetcher",
    "MatchDetailFetcher",
    "MatchParser",
]
