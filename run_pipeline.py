#!/usr/bin/env python3
"""
Darts Data Pipeline Runner

Main script for running the full data ingestion pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PIPELINE_CONFIG
from src.ingest import (
    StatoriumClient,
    LeagueSeasonFetcher,
    MatchListFetcher,
    MatchDetailFetcher,
    MatchParser,
)
from src.storage import ParquetStore


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )


def run_pipeline(
    api_key: str,
    leagues: dict = None,
    num_seasons: int = 5,
    data_dir: str = "data/processed",
    verbose: bool = False,
    start_date: str = None,
):
    """
    Run the full data ingestion pipeline.

    Args:
        api_key: Statorium API key
        leagues: Dict of league_id -> name (uses config default if None)
        num_seasons: Number of seasons per league
        data_dir: Output data directory
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    leagues = leagues or PIPELINE_CONFIG.leagues

    logger.info("=" * 60)
    logger.info("Starting Darts Data Pipeline")
    logger.info(f"Leagues: {len(leagues)}")
    logger.info(f"Seasons per league: {num_seasons}")
    if start_date:
        logger.info(f"Date filter: {start_date} onwards")
    logger.info("=" * 60)

    # Initialize components
    client = StatoriumClient(api_key=api_key)
    store = ParquetStore(data_dir)
    parser = MatchParser()

    try:
        # Step 1: Fetch leagues and seasons
        logger.info("\n[1/4] Fetching leagues and seasons...")
        league_fetcher = LeagueSeasonFetcher(client)
        all_leagues, all_seasons = league_fetcher.fetch_all_configured_seasons(
            leagues, num_seasons
        )

        # Save leagues and seasons
        store.save_leagues(all_leagues)
        store.save_seasons(all_seasons)
        logger.info(f"Saved {len(all_leagues)} leagues, {len(all_seasons)} seasons")

        # Step 2: Fetch match lists
        logger.info("\n[2/4] Fetching match lists...")
        match_fetcher = MatchListFetcher(client)

        all_match_infos = []
        for season in all_seasons:
            try:
                match_infos = match_fetcher.fetch_matches_for_season(season)
                all_match_infos.extend(match_infos)
            except Exception as e:
                logger.error(f"Failed to fetch matches for season {season.season_id}: {e}")

        logger.info(f"Found {len(all_match_infos)} total matches")

        # Filter by start_date if provided
        if start_date:
            from datetime import datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            before_count = len(all_match_infos)
            all_match_infos = [
                m for m in all_match_infos
                if m.get("match_date") and datetime.strptime(m["match_date"][:10], "%Y-%m-%d") >= start_dt
            ]
            logger.info(f"Filtered to {len(all_match_infos)} matches from {start_date} onwards (skipped {before_count - len(all_match_infos)})")

        # Step 3: Fetch match details
        logger.info("\n[3/4] Fetching match details...")
        detail_fetcher = MatchDetailFetcher(client)

        all_matches = []
        all_players = []
        all_legs = []
        all_visits = []
        all_stats = []

        total = len(all_match_infos)
        for i, match_info in enumerate(all_match_infos):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{total}")

            try:
                match_data = detail_fetcher.fetch_match(match_info["match_id"])
                if match_data:
                    match, players, legs, visits, stats = parser.parse_match(
                        match_data, match_info
                    )
                    all_matches.append(match)
                    all_players.extend(players)
                    all_legs.extend(legs)
                    all_visits.extend(visits)
                    all_stats.extend(stats)
            except Exception as e:
                logger.warning(f"Error parsing match {match_info['match_id']}: {e}")

        # Step 4: Save to storage
        logger.info("\n[4/4] Saving to storage...")
        store.save_players(all_players)
        store.save_matches(all_matches)
        store.save_legs(all_legs)
        store.save_visits(all_visits)
        store.save_match_stats(all_stats)

        # Summary
        counts = store.get_table_counts()
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)

        # This run stats
        logger.info("\nThis run processed:")
        logger.info(f"  Leagues:  {len(all_leagues)}")
        logger.info(f"  Seasons:  {len(all_seasons)}")
        logger.info(f"  Matches:  {len(all_matches)}")
        logger.info(f"  Players:  {len(set(p.player_id for p in all_players))}")
        logger.info(f"  Legs:     {len(all_legs)}")
        logger.info(f"  Visits:   {len(all_visits)}")
        logger.info(f"  Stats:    {len(all_stats)}")

        # Date range of processed matches
        if all_matches:
            match_dates = [m.match_date for m in all_matches if m.match_date]
            if match_dates:
                logger.info(f"  Date range: {min(match_dates)[:10]} to {max(match_dates)[:10]}")

        # Total database stats
        logger.info("\nTotal in database:")
        for table, count in counts.items():
            logger.info(f"  {table}: {count:,}")

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run the darts data ingestion pipeline"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Statorium API key",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=5,
        help="Number of seasons per league (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Output data directory (default: data/processed)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--start-date",
        help="Only fetch matches from this date onwards (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    run_pipeline(
        api_key=args.api_key,
        num_seasons=args.seasons,
        data_dir=args.data_dir,
        verbose=args.verbose,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
