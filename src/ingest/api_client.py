"""
Statorium Darts API Client

Provides HTTP client for interacting with the Statorium darts API.
"""

import logging
import time
from typing import Any, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class StatoriumClient:
    """
    HTTP client for Statorium darts API.

    Handles authentication, rate limiting, retries, and error handling.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://darts.statorium.com/api/v1",
        timeout: int = 30,
        request_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize the API client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            request_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_delay = request_delay
        self.max_retries = max_retries

        # Create session with retry strategy
        self.session = self._create_session()
        self._last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _wait_for_rate_limit(self):
        """Ensure minimum delay between requests."""
        if self.request_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)

    def _build_url(self, endpoint: str) -> str:
        """Build full URL for endpoint."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request to the API.

        Args:
            endpoint: API endpoint (e.g., "/leagues/2/")
            params: Additional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.RequestException: On request failure
        """
        self._wait_for_rate_limit()

        url = self._build_url(endpoint)

        # Add API key to params
        request_params = params.copy() if params else {}
        request_params["apikey"] = self.api_key

        logger.debug(f"Making request to: {url}")

        try:
            response = self.session.get(
                url,
                params=request_params,
                timeout=self.timeout,
            )
            self._last_request_time = time.time()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout for {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    def get_leagues(self) -> Dict[str, Any]:
        """
        Fetch all available leagues.

        Returns:
            API response containing leagues list
        """
        return self._make_request("/leagues/")

    def get_league(self, league_id: int) -> Dict[str, Any]:
        """
        Fetch league details including seasons.

        Args:
            league_id: ID of the league

        Returns:
            API response containing league with seasons
        """
        return self._make_request(f"/leagues/{league_id}/")

    def get_matches_by_season(self, season_id: int) -> Dict[str, Any]:
        """
        Fetch all matches for a season.

        Args:
            season_id: ID of the season

        Returns:
            API response containing calendar with matchdays and brackets
        """
        return self._make_request("/matches/", params={"season_id": season_id})

    def get_match(self, match_id: int) -> Dict[str, Any]:
        """
        Fetch detailed match data.

        Args:
            match_id: ID of the match

        Returns:
            API response containing full match details
        """
        return self._make_request(f"/matches/{match_id}/")

    def get_player(self, player_id: int) -> Dict[str, Any]:
        """
        Fetch player details.

        Args:
            player_id: ID of the player

        Returns:
            API response containing player data
        """
        return self._make_request(f"/participants/{player_id}/")

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
