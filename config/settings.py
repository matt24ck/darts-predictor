"""
Darts Pipeline Configuration Settings

Central configuration for league/season selection, API settings,
and model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# =============================================================================
# API Configuration
# =============================================================================
API_BASE_URL = "https://darts.statorium.com/api/v1"
API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual API key

# Request settings
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 0.5   # delay between requests to avoid rate limiting
MAX_RETRIES = 3

# =============================================================================
# League Configuration
# =============================================================================
# Core high-standard PDC tournaments to include
CORE_LEAGUES: Dict[int, str] = {
    2: "PDC World Darts Championship",
    3: "PDC Premier League Darts",
    4: "PDC World Matchplay",
    5: "PDC World Grand Prix",
    6: "PDC Grand Slam of Darts",
    7: "PDC UK Open",
    8: "PDC Players Championship Finals",
    35: "Players Championships",
    37: "PDC World Masters",
    38: "MODUS Super Series"
}

# Number of most recent seasons to fetch per league
DEFAULT_SEASONS_TO_FETCH = 5

# =============================================================================
# Data Storage Configuration
# =============================================================================
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

# Storage format options: "parquet", "sqlite", "csv"
STORAGE_FORMAT = "parquet"
SQLITE_DB_PATH = f"{DATA_DIR}/darts.db"

# =============================================================================
# Pipeline Configuration
# =============================================================================
@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Leagues to include (league_id -> name)
    leagues: Dict[int, str] = field(default_factory=lambda: CORE_LEAGUES.copy())

    # Number of seasons to fetch per league
    seasons_to_fetch: int = DEFAULT_SEASONS_TO_FETCH

    # API settings
    api_base_url: str = API_BASE_URL
    api_key: str = API_KEY
    request_timeout: int = REQUEST_TIMEOUT
    request_delay: float = REQUEST_DELAY
    max_retries: int = MAX_RETRIES

    # Storage settings
    data_dir: str = DATA_DIR
    storage_format: str = STORAGE_FORMAT

    def add_league(self, league_id: int, league_name: str):
        """Add a league to the configuration."""
        self.leagues[league_id] = league_name

    def remove_league(self, league_id: int):
        """Remove a league from the configuration."""
        if league_id in self.leagues:
            del self.leagues[league_id]

    def set_seasons(self, num_seasons: int):
        """Set the number of seasons to fetch."""
        self.seasons_to_fetch = num_seasons


# Default pipeline configuration
PIPELINE_CONFIG = PipelineConfig()


# =============================================================================
# Venue Configuration (for imputation when API data is missing)
# =============================================================================
# Known venues with synthetic IDs (negative to avoid collision with API IDs)
KNOWN_VENUES: Dict[int, str] = {
    -1: "Alexandra Palace, London",           # World Championship
    -2: "Various (touring)",                  # Premier League
    -3: "Winter Gardens, Blackpool",          # World Matchplay
    -4: "Citywest Hotel, Dublin",             # World Grand Prix
    -5: "Aldersley Leisure Village, Wolverhampton",  # Grand Slam of Darts
    -6: "Butlin's Minehead Resort",           # UK Open
    -7: "Butlin's Minehead Resort",           # Players Championship Finals
    -8: "Various (floor events)",             # Players Championships
    -9: "Online/Studio",                      # MODUS Super Series
}

# Map league_id to default venue_id
LEAGUE_VENUE_MAPPING: Dict[int, int] = {
    2: -1,   # World Championship -> Alexandra Palace
    3: -2,   # Premier League -> Touring (multiple venues)
    4: -3,   # World Matchplay -> Winter Gardens Blackpool
    5: -4,   # World Grand Prix -> Citywest Dublin
    6: -5,   # Grand Slam -> Aldersley Leisure Village
    7: -6,   # UK Open -> Butlin's Minehead
    8: -7,   # Players Championship Finals -> Butlin's Minehead
    35: -8,  # Players Championships -> Various floor venues
    38: -9,  # MODUS Super Series -> Studio/Online
}

# Venue characteristics for modeling
@dataclass
class VenueCharacteristics:
    """Characteristics of a venue that may affect performance."""
    venue_id: int
    venue_name: str
    is_tv_event: bool = False           # Major televised event
    is_floor_event: bool = False        # Smaller floor/tour event
    has_crowd: bool = True              # Significant crowd presence
    is_touring: bool = False            # Multiple venues (Premier League)
    pressure_factor: float = 1.0        # Relative pressure (1.0 = baseline)

VENUE_CHARACTERISTICS: Dict[int, VenueCharacteristics] = {
    -1: VenueCharacteristics(-1, "Alexandra Palace", is_tv_event=True, has_crowd=True, pressure_factor=1.3),
    -2: VenueCharacteristics(-2, "Premier League (touring)", is_tv_event=True, is_touring=True, has_crowd=True, pressure_factor=1.2),
    -3: VenueCharacteristics(-3, "Winter Gardens Blackpool", is_tv_event=True, has_crowd=True, pressure_factor=1.25),
    -4: VenueCharacteristics(-4, "Citywest Dublin", is_tv_event=True, has_crowd=True, pressure_factor=1.15),
    -5: VenueCharacteristics(-5, "Aldersley Leisure Village", is_tv_event=True, has_crowd=True, pressure_factor=1.2),
    -6: VenueCharacteristics(-6, "Butlin's Minehead", is_tv_event=True, has_crowd=True, pressure_factor=1.1),
    -7: VenueCharacteristics(-7, "Butlin's Minehead (Finals)", is_tv_event=True, has_crowd=True, pressure_factor=1.15),
    -8: VenueCharacteristics(-8, "Floor Events (various)", is_floor_event=True, has_crowd=False, pressure_factor=0.9),
    -9: VenueCharacteristics(-9, "Studio/Online", is_floor_event=True, has_crowd=False, pressure_factor=0.85),
}


def impute_venue_id(league_id: int, venue_id: Optional[int] = None) -> int:
    """
    Impute venue_id based on league if not provided by API.

    Args:
        league_id: The league ID
        venue_id: The venue ID from the API (may be None)

    Returns:
        venue_id (original if valid, imputed if missing)
    """
    if venue_id is not None and venue_id > 0:
        return venue_id
    return LEAGUE_VENUE_MAPPING.get(league_id, -8)  # Default to floor event


def get_venue_features(venue_id: int) -> Dict[str, float]:
    """
    Get modeling features for a venue.

    Args:
        venue_id: The venue ID (can be real or imputed)

    Returns:
        Dict of feature name -> value
    """
    # Use imputed characteristics for known synthetic IDs
    if venue_id in VENUE_CHARACTERISTICS:
        vc = VENUE_CHARACTERISTICS[venue_id]
        return {
            "is_tv_event": 1.0 if vc.is_tv_event else 0.0,
            "is_floor_event": 1.0 if vc.is_floor_event else 0.0,
            "has_crowd": 1.0 if vc.has_crowd else 0.0,
            "is_touring": 1.0 if vc.is_touring else 0.0,
            "pressure_factor": vc.pressure_factor,
        }

    # Unknown venue - assume moderate TV event characteristics
    return {
        "is_tv_event": 0.5,
        "is_floor_event": 0.5,
        "has_crowd": 0.5,
        "is_touring": 0.0,
        "pressure_factor": 1.0,
    }
