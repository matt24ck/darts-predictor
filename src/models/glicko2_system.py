"""
Glicko-2 Rating System for Darts

Enhanced rating system with:
1. Rating uncertainty (RD) tracking
2. Rating volatility (sigma) for adapting to player form changes
3. Time decay - uncertainty increases with inactivity
4. Match length weighting - longer matches are more informative
5. Venue effects integration
6. Rating period batching (proper Glicko-2 implementation)
7. Margin-based scoring (close wins vs dominant wins)
8. Confidence intervals for predictions
9. Surface-specific ratings (TV vs Floor events)
10. Recency-weighted updates
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.isotonic import IsotonicRegression

from ..schema.models import StatField

logger = logging.getLogger(__name__)

# Glicko-2 constants
TAU = 0.5  # System constant constraining volatility change
CONVERGENCE_TOLERANCE = 1e-6
GLICKO2_SCALE = 173.7178  # Conversion factor between Glicko and Glicko-2 scales


@dataclass
class Glicko2Config:
    """
    Configuration for the Glicko-2 system.

    Default values are tuned for darts match prediction via hyperparameter search.
    Key findings from tuning:
    - rd_decay_per_day=0.4: Slower decay than default improves stability
    - margin_weight=0.5: Higher weight on score margins improves accuracy
    - rating_period_days=5: Shorter periods capture form changes faster
    """

    # Initial values
    initial_rating: float = 1500.0
    initial_rd: float = 350.0       # Rating deviation (uncertainty)
    initial_volatility: float = 0.06

    # Time decay (TUNED: 0.4 instead of 0.5)
    rd_decay_per_day: float = 0.4   # RD increase per day of inactivity
    max_rd: float = 350.0           # Maximum RD (reset point)
    min_rd: float = 30.0            # Minimum RD (maximum certainty)

    # Match weighting
    use_match_length_weighting: bool = True
    base_expected_legs: float = 15.0  # Reference match length

    # Margin-based scoring (TUNED: margin_weight=0.7 gives best accuracy)
    use_margin_scoring: bool = True
    margin_weight: float = 0.7  # How much margin affects the score (0 = binary, 1 = full margin)

    # Venue effects
    use_venue_effects: bool = True

    # Performance integration
    use_performance_adjustment: bool = True
    performance_weight: float = 0.2

    # Rating period batching (TUNED: 5-day periods instead of 7)
    use_rating_periods: bool = True
    rating_period_days: int = 5  # 5-day periods for faster adaptation

    # Surface-specific ratings (TV vs Floor)
    use_surface_ratings: bool = True
    min_surface_matches: int = 5  # Minimum matches before using surface bonus

    # Recency weighting
    use_recency_weighting: bool = True
    recency_half_life_days: float = 180.0  # 6-month half-life

    # Minimum matches for reliable rating
    min_matches_for_stable: int = 10

    # Recent form adjustment (TESTED: hurts accuracy, disabled by default)
    use_form_adjustment: bool = False
    form_window: int = 5  # Number of recent matches to consider
    form_weight: float = 0.3  # How much form affects probability (0 = none, 1 = full)


@dataclass
class PlayerRating:
    """Player's current Glicko-2 rating state."""
    player_id: int
    rating: float = 1500.0
    rd: float = 350.0               # Rating deviation
    volatility: float = 0.06        # Rating volatility
    last_played: Optional[date] = None
    match_count: int = 0

    # Surface-specific adjustments
    tv_bonus: float = 0.0           # Additive adjustment for TV events
    tv_bonus_rd: float = 200.0      # Uncertainty in TV bonus
    floor_match_count: int = 0
    tv_match_count: int = 0

    # Recent form tracking (list of tuples: (actual_score, expected_score))
    recent_results: List[Tuple[float, float]] = field(default_factory=list)
    form_window: int = 5  # Number of recent matches to consider

    def to_glicko2_scale(self) -> Tuple[float, float]:
        """Convert to Glicko-2 internal scale."""
        mu = (self.rating - 1500) / GLICKO2_SCALE
        phi = self.rd / GLICKO2_SCALE
        return mu, phi

    def from_glicko2_scale(self, mu: float, phi: float) -> None:
        """Convert from Glicko-2 internal scale."""
        self.rating = mu * GLICKO2_SCALE + 1500
        self.rd = phi * GLICKO2_SCALE

    def get_effective_rating(self, is_tv_event: bool = False) -> float:
        """Get rating adjusted for event type."""
        if is_tv_event and self.tv_match_count >= 5:
            return self.rating + self.tv_bonus
        return self.rating

    def get_effective_rd(self, is_tv_event: bool = False) -> float:
        """Get RD adjusted for event type."""
        if is_tv_event and self.tv_match_count >= 5:
            # Combined uncertainty
            return math.sqrt(self.rd ** 2 + self.tv_bonus_rd ** 2)
        return self.rd

    def add_result(self, actual_score: float, expected_score: float) -> None:
        """Add a match result to recent history."""
        self.recent_results.append((actual_score, expected_score))
        # Keep only the most recent matches
        if len(self.recent_results) > self.form_window:
            self.recent_results = self.recent_results[-self.form_window:]

    def get_form_factor(self) -> float:
        """
        Calculate form factor based on recent performance vs expectations.

        Returns a value centered at 0:
        - Positive = player performing above expectation (hot streak)
        - Negative = player performing below expectation (cold streak)
        - Magnitude indicates strength of streak
        """
        if len(self.recent_results) < 3:
            return 0.0  # Not enough data

        total_actual = sum(r[0] for r in self.recent_results)
        total_expected = sum(r[1] for r in self.recent_results)

        if total_expected == 0:
            return 0.0

        # Performance ratio: >1 means outperforming, <1 means underperforming
        n_matches = len(self.recent_results)
        actual_rate = total_actual / n_matches
        expected_rate = total_expected / n_matches

        # Form factor: difference between actual and expected win rate
        # Capped at +/- 0.15 to prevent extreme adjustments
        form = actual_rate - expected_rate
        return max(-0.15, min(0.15, form))


@dataclass
class Glicko2HistoryRecord:
    """Extended history record with Glicko-2 specific fields."""
    match_id: int
    league_id: int
    season_id: int
    match_date: Optional[date]
    home_player_id: int
    away_player_id: int
    home_sets: int
    away_sets: int
    home_legs: int
    away_legs: int
    home_result: int  # 1 = home win, 0 = home loss

    # Pre-match ratings
    home_rating_pre: float
    home_rd_pre: float
    home_vol_pre: float
    away_rating_pre: float
    away_rd_pre: float
    away_vol_pre: float

    # Post-match ratings
    home_rating_post: float
    home_rd_post: float
    home_vol_post: float
    away_rating_post: float
    away_rd_post: float
    away_vol_post: float

    # Expected outcome
    home_expected: float

    # Match info
    match_weight: float = 1.0
    venue_effect: float = 0.0

    # New fields
    margin_score: float = 0.5  # Score based on margin (0.5 = close, near 1 = dominant home win)
    is_tv_event: bool = False
    recency_weight: float = 1.0

    # Confidence intervals
    home_expected_lower: float = 0.0
    home_expected_upper: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league_id": self.league_id,
            "season_id": self.season_id,
            "match_date": (
                self.match_date.isoformat()
                if hasattr(self.match_date, 'isoformat')
                else self.match_date
            ) if self.match_date else None,
            "home_player_id": self.home_player_id,
            "away_player_id": self.away_player_id,
            "home_sets": self.home_sets,
            "away_sets": self.away_sets,
            "home_legs": self.home_legs,
            "away_legs": self.away_legs,
            "home_result": self.home_result,
            "home_rating_pre": self.home_rating_pre,
            "home_rd_pre": self.home_rd_pre,
            "away_rating_pre": self.away_rating_pre,
            "away_rd_pre": self.away_rd_pre,
            "home_rating_post": self.home_rating_post,
            "home_rd_post": self.home_rd_post,
            "away_rating_post": self.away_rating_post,
            "away_rd_post": self.away_rd_post,
            "home_expected": self.home_expected,
            "home_expected_lower": self.home_expected_lower,
            "home_expected_upper": self.home_expected_upper,
            "match_weight": self.match_weight,
            "margin_score": self.margin_score,
            "is_tv_event": self.is_tv_event,
        }


@dataclass
class RatingPeriodResult:
    """Result from processing matches within a rating period."""
    player_id: int
    opponents: List[int]
    scores: List[float]  # Actual scores (can be margin-based)
    weights: List[float]  # Match weights


class Glicko2System:
    """
    Glicko-2 rating system for darts.

    Key features:
    - Tracks rating uncertainty (RD) per player
    - Uncertainty increases with inactivity (time decay)
    - Volatility adapts to player form consistency
    - Match length affects update magnitude
    - Venue effects integration
    - Rating period batching for proper Glicko-2
    - Margin-based scoring for informative updates
    - Confidence intervals for predictions
    - Surface-specific ratings (TV vs Floor)
    """

    # TV event leagues (major televised tournaments)
    TV_LEAGUES = {2, 3, 4, 5, 6, 7, 8}  # World Championship, Premier League, etc.

    def __init__(self, config: Optional[Glicko2Config] = None):
        self.config = config or Glicko2Config()

        # Player ratings: player_id -> PlayerRating
        self.ratings: Dict[int, PlayerRating] = {}

        # Player baselines for performance adjustment
        self.player_baselines: Dict[int, Dict[str, float]] = {}

        # Venue effects (from training or config)
        self.venue_effects: Dict[int, float] = {}

        # History
        self.history: List[Glicko2HistoryRecord] = []

        # Last training date for recency calculations
        self.last_training_date: Optional[date] = None

        # Probability calibration (isotonic regression)
        self.probability_calibrator: Optional[IsotonicRegression] = None
        self.calibration_stats: Dict[str, float] = {}

    def get_player(self, player_id: int) -> PlayerRating:
        """Get or create player rating."""
        if player_id not in self.ratings:
            self.ratings[player_id] = PlayerRating(
                player_id=player_id,
                rating=self.config.initial_rating,
                rd=self.config.initial_rd,
                volatility=self.config.initial_volatility,
            )
        return self.ratings[player_id]

    def apply_time_decay(
        self,
        player: PlayerRating,
        current_date: Optional[date],
    ) -> None:
        """Increase RD based on time since last match."""
        if player.last_played is None or current_date is None:
            return

        # Handle NaN values (pandas returns float NaN for missing dates)
        if isinstance(player.last_played, float):
            if pd.isna(player.last_played):
                return
        if isinstance(current_date, float):
            if pd.isna(current_date):
                return

        if isinstance(player.last_played, str):
            try:
                player.last_played = date.fromisoformat(player.last_played)
            except:
                return

        if isinstance(current_date, str):
            try:
                current_date = date.fromisoformat(current_date)
            except:
                return

        # Ensure both are date objects before subtraction
        if not isinstance(player.last_played, date) or not isinstance(current_date, date):
            return

        days_inactive = (current_date - player.last_played).days

        if days_inactive > 0:
            # Increase RD based on inactivity
            rd_increase = days_inactive * self.config.rd_decay_per_day
            player.rd = min(self.config.max_rd, player.rd + rd_increase)

            # Also decay TV bonus certainty
            if self.config.use_surface_ratings:
                player.tv_bonus_rd = min(200.0, player.tv_bonus_rd + days_inactive * 0.2)

    def compute_margin_score(self, match: pd.Series) -> float:
        """
        Compute a score that reflects match closeness.

        Returns a value between ~0.55 (close win) and ~0.95 (dominant win)
        for the winner, or the complement for the loser.
        """
        is_set_format = match.get("is_set_format", False)

        if is_set_format:
            home_units = match.get("home_sets", 0)
            away_units = match.get("away_sets", 0)
        else:
            home_units = match.get("home_legs", 0)
            away_units = match.get("away_legs", 0)

        total_units = home_units + away_units
        if total_units == 0:
            return 0.5

        # Margin ratio: 0 = dead even (shouldn't happen), 1 = maximum margin
        margin = abs(home_units - away_units)
        margin_ratio = margin / total_units

        # Map to score range
        # Close match: margin_ratio ~ 0.1 -> score ~ 0.55
        # Dominant match: margin_ratio ~ 0.8 -> score ~ 0.90
        base_score = 0.5 + 0.4 * margin_ratio

        # Return from home perspective
        return base_score if home_units > away_units else (1.0 - base_score)

    def compute_recency_weight(
        self,
        match_date: Optional[date],
        reference_date: Optional[date] = None,
    ) -> float:
        """Weight recent matches more heavily."""
        if not self.config.use_recency_weighting:
            return 1.0

        if match_date is None:
            return 1.0

        if reference_date is None:
            reference_date = self.last_training_date or date.today()

        if isinstance(match_date, str):
            try:
                match_date = date.fromisoformat(match_date)
            except:
                return 1.0

        if isinstance(reference_date, str):
            try:
                reference_date = date.fromisoformat(reference_date)
            except:
                return 1.0

        days_ago = (reference_date - match_date).days
        if days_ago < 0:
            days_ago = 0

        # Exponential decay
        return 0.5 ** (days_ago / self.config.recency_half_life_days)

    def expected_score(
        self,
        player: PlayerRating,
        opponent: PlayerRating,
        is_tv_event: bool = False,
    ) -> float:
        """
        Calculate expected score using Glicko-2 formula.

        Accounts for both rating difference and uncertainty.
        """
        # Get effective ratings based on event type
        if self.config.use_surface_ratings and is_tv_event:
            player_rating = player.get_effective_rating(True)
            opponent_rating = opponent.get_effective_rating(True)
        else:
            player_rating = player.rating
            opponent_rating = opponent.rating

        # Convert to Glicko-2 scale
        mu = (player_rating - 1500) / GLICKO2_SCALE
        phi = player.rd / GLICKO2_SCALE
        mu_j = (opponent_rating - 1500) / GLICKO2_SCALE
        phi_j = opponent.rd / GLICKO2_SCALE

        # g function reduces impact of uncertain opponents
        g_phi_j = 1.0 / math.sqrt(1.0 + 3.0 * phi_j ** 2 / math.pi ** 2)

        # Expected score
        exp_term = -g_phi_j * (mu - mu_j)
        return 1.0 / (1.0 + math.exp(exp_term))

    def win_probability(
        self,
        player_a_id: int,
        player_b_id: int,
        match_date: Optional[date] = None,
        is_tv_event: bool = False,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Get win probabilities with uncertainty info.

        Returns:
            Tuple of (p_a_win, p_b_win, metadata dict with RDs and confidence intervals)
        """
        player_a = self.get_player(player_a_id)
        player_b = self.get_player(player_b_id)

        # Apply time decay if date provided
        if match_date:
            self.apply_time_decay(player_a, match_date)
            self.apply_time_decay(player_b, match_date)

        p_a = self.expected_score(player_a, player_b, is_tv_event)

        # Compute confidence intervals
        p_a_lower, p_a_upper = self._compute_win_probability_ci(
            player_a, player_b, is_tv_event
        )

        metadata = {
            "rating_a": player_a.rating,
            "rating_b": player_b.rating,
            "rd_a": player_a.rd,
            "rd_b": player_b.rd,
            "combined_uncertainty": math.sqrt(player_a.rd ** 2 + player_b.rd ** 2),
            "p_a_lower": p_a_lower,
            "p_a_upper": p_a_upper,
            "p_b_lower": 1.0 - p_a_upper,
            "p_b_upper": 1.0 - p_a_lower,
        }

        # Add surface-specific info if enabled
        if self.config.use_surface_ratings:
            metadata["tv_bonus_a"] = player_a.tv_bonus
            metadata["tv_bonus_b"] = player_b.tv_bonus
            metadata["tv_matches_a"] = player_a.tv_match_count
            metadata["tv_matches_b"] = player_b.tv_match_count

        return (p_a, 1.0 - p_a, metadata)

    def _compute_win_probability_ci(
        self,
        player_a: PlayerRating,
        player_b: PlayerRating,
        is_tv_event: bool = False,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for win probability."""
        # Get effective ratings and RDs
        if self.config.use_surface_ratings and is_tv_event:
            rating_a = player_a.get_effective_rating(True)
            rating_b = player_b.get_effective_rating(True)
            rd_a = player_a.get_effective_rd(True)
            rd_b = player_b.get_effective_rd(True)
        else:
            rating_a = player_a.rating
            rating_b = player_b.rating
            rd_a = player_a.rd
            rd_b = player_b.rd

        # Rating difference and combined uncertainty
        rating_diff = rating_a - rating_b
        combined_rd = math.sqrt(rd_a ** 2 + rd_b ** 2)

        # Z-score for confidence level
        z = scipy_stats.norm.ppf((1 + confidence) / 2)

        # Confidence bounds on rating difference
        diff_lower = rating_diff - z * combined_rd
        diff_upper = rating_diff + z * combined_rd

        # Convert to probabilities (using simple logistic)
        scale = 400.0 / math.log(10)  # ~173.7
        p_lower = 1.0 / (1.0 + 10 ** (-diff_lower / 400))
        p_upper = 1.0 / (1.0 + 10 ** (-diff_upper / 400))

        return p_lower, p_upper

    def compute_match_weight(self, match: pd.Series) -> float:
        """
        Compute match weight based on length.

        Longer matches contain more information.
        Weight = sqrt(expected_legs / base_expected_legs)
        """
        if not self.config.use_match_length_weighting:
            return 1.0

        is_set_format = match.get("is_set_format", False)
        bos = match.get("best_of_sets", 0)
        bol = match.get("best_of_legs", 5)

        # Estimate expected legs
        if is_set_format and bos > 0:
            expected_sets = (bos + 1) / 2 + 0.3
            expected_legs_per_set = (bol + 1) / 2 + 0.3
            expected_legs = expected_sets * expected_legs_per_set
        else:
            expected_legs = (bol + 1) / 2 + 0.3

        # Scale by sqrt to prevent overly dominant long matches
        weight = math.sqrt(expected_legs / self.config.base_expected_legs)

        # Clamp to reasonable range
        return max(0.5, min(2.0, weight))

    def is_tv_event(self, league_id: int) -> bool:
        """Check if league is a TV event."""
        return league_id in self.TV_LEAGUES

    def get_venue_effect(
        self,
        league_id: int,
        venue_id: Optional[int] = None,
    ) -> float:
        """Get venue effect for rating adjustment."""
        if not self.config.use_venue_effects:
            return 0.0

        from config.settings import impute_venue_id, get_venue_features

        effective_venue_id = impute_venue_id(league_id, venue_id)
        venue_features = get_venue_features(effective_venue_id)

        return self.venue_effects.get(effective_venue_id, 0.0)

    def update_player(
        self,
        player: PlayerRating,
        opponent: PlayerRating,
        actual_score: float,
        match_weight: float = 1.0,
        is_tv_event: bool = False,
    ) -> None:
        """
        Update player rating using Glicko-2 algorithm.

        Args:
            player: Player to update
            opponent: Opponent
            actual_score: Score (can be margin-based, between 0 and 1)
            match_weight: Weight for this match (based on length and recency)
            is_tv_event: Whether this is a TV event (for surface-specific updates)
        """
        # Convert to Glicko-2 scale
        mu, phi = player.to_glicko2_scale()
        mu_j, phi_j = opponent.to_glicko2_scale()

        # g function
        g_phi_j = 1.0 / math.sqrt(1.0 + 3.0 * phi_j ** 2 / math.pi ** 2)

        # Expected score
        e = self.expected_score(player, opponent, is_tv_event)

        # Variance
        v_inv = g_phi_j ** 2 * e * (1 - e) * match_weight
        if v_inv < 1e-10:
            v_inv = 1e-10
        v = 1.0 / v_inv

        # Delta (score difference scaled)
        delta = v * g_phi_j * (actual_score - e) * match_weight

        # Update volatility using iterative algorithm
        new_sigma = self._compute_new_volatility(phi, player.volatility, v, delta)

        # Update RD
        phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)

        # New RD
        new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + v_inv)

        # New rating
        new_mu = mu + new_phi ** 2 * g_phi_j * (actual_score - e) * match_weight

        # Convert back and update
        player.from_glicko2_scale(new_mu, new_phi)
        player.volatility = new_sigma

        # Clamp RD to configured range
        player.rd = max(self.config.min_rd, min(self.config.max_rd, player.rd))

        # Update surface-specific stats
        if self.config.use_surface_ratings:
            if is_tv_event:
                player.tv_match_count += 1
                # Update TV bonus based on performance
                surprise = actual_score - e
                # Smaller learning rate for TV bonus
                alpha = 0.1
                player.tv_bonus = player.tv_bonus + alpha * surprise * 50  # Scale to rating points
                player.tv_bonus = max(-200, min(200, player.tv_bonus))  # Clamp
                # Reduce TV bonus uncertainty
                player.tv_bonus_rd = max(50.0, player.tv_bonus_rd * 0.95)
            else:
                player.floor_match_count += 1

    def _compute_new_volatility(
        self,
        phi: float,
        sigma: float,
        v: float,
        delta: float,
    ) -> float:
        """Compute new volatility using iterative algorithm (Section 5.4 of Glicko-2)."""
        a = math.log(sigma ** 2)
        tau = TAU

        def f(x):
            ex = math.exp(x)
            num = ex * (delta ** 2 - phi ** 2 - v - ex)
            denom = 2 * (phi ** 2 + v + ex) ** 2
            return num / denom - (x - a) / tau ** 2

        # Initial bounds
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
                if k > 100:  # Safety limit
                    break
            B = a - k * tau

        # Iteration
        f_A = f(A)
        f_B = f(B)

        iterations = 0
        max_iterations = 100

        while abs(B - A) > CONVERGENCE_TOLERANCE and iterations < max_iterations:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)

            if f_C * f_B < 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A / 2

            B = C
            f_B = f_C
            iterations += 1

        return math.exp(A / 2)

    def process_match(
        self,
        match: pd.Series,
        home_stats: Optional[Dict[str, float]] = None,
        away_stats: Optional[Dict[str, float]] = None,
    ) -> Glicko2HistoryRecord:
        """Process a match and update ratings."""
        home_player_id = match.get("home_player_id")
        away_player_id = match.get("away_player_id")

        if home_player_id is None or away_player_id is None:
            raise ValueError("Match missing player IDs")

        # Ensure player IDs are integers
        home_player_id = int(home_player_id)
        away_player_id = int(away_player_id)

        match_date = match.get("match_date")
        league_id = match.get("league_id", 0)
        is_tv = self.is_tv_event(league_id)

        # Get players and apply time decay
        home_player = self.get_player(home_player_id)
        away_player = self.get_player(away_player_id)

        self.apply_time_decay(home_player, match_date)
        self.apply_time_decay(away_player, match_date)

        # Store pre-match state
        home_rating_pre = home_player.rating
        home_rd_pre = home_player.rd
        home_vol_pre = home_player.volatility
        away_rating_pre = away_player.rating
        away_rd_pre = away_player.rd
        away_vol_pre = away_player.volatility

        # Determine result and margin score
        is_set_format = match.get("is_set_format", False)
        home_legs = match.get("home_legs", 0)
        away_legs = match.get("away_legs", 0)
        home_sets = match.get("home_sets", 0)
        away_sets = match.get("away_sets", 0)

        # Fallback: if legs are both 0, use sets (handles MODUS data)
        if home_legs == 0 and away_legs == 0:
            home_won = home_sets > away_sets
        elif is_set_format:
            home_won = home_sets > away_sets
        else:
            home_won = home_legs > away_legs

        home_result = 1 if home_won else 0

        # Compute margin-based score
        if self.config.use_margin_scoring:
            margin_score = self.compute_margin_score(match)
            # Blend binary and margin scores
            mw = self.config.margin_weight
            home_score = (1 - mw) * home_result + mw * margin_score
        else:
            margin_score = 0.5
            home_score = float(home_result)

        # Compute match weight
        match_weight = self.compute_match_weight(match)

        # Apply recency weight
        recency_weight = self.compute_recency_weight(match_date)
        match_weight *= recency_weight

        # Apply performance adjustment if enabled
        if self.config.use_performance_adjustment and home_stats and away_stats:
            match_weight = self._adjust_weight_for_performance(
                match, match_weight, home_stats, away_stats,
                home_player_id, away_player_id, home_won
            )

        # Compute expected scores and confidence intervals
        home_expected = self.expected_score(home_player, away_player, is_tv)
        home_expected_lower, home_expected_upper = self._compute_win_probability_ci(
            home_player, away_player, is_tv
        )

        # Update ratings
        self.update_player(home_player, away_player, home_score, match_weight, is_tv)
        self.update_player(away_player, home_player, 1 - home_score, match_weight, is_tv)

        # Update match count and last played
        home_player.match_count += 1
        away_player.match_count += 1

        # Track recent results for form calculation
        if self.config.use_form_adjustment:
            home_player.add_result(float(home_result), home_expected)
            away_player.add_result(float(1 - home_result), 1 - home_expected)

        # Only update last_played if match_date is valid
        if match_date is not None and not (isinstance(match_date, float) and pd.isna(match_date)):
            if isinstance(match_date, str):
                try:
                    match_date = date.fromisoformat(match_date)
                except (ValueError, TypeError):
                    match_date = None
            if isinstance(match_date, date):
                home_player.last_played = match_date
                away_player.last_played = match_date

        # Update baselines (skip NaN values)
        if home_stats and "avg_3_darts" in home_stats:
            val = home_stats["avg_3_darts"]
            if not pd.isna(val):
                self._update_baseline(home_player_id, "avg_3_darts", val)
        if away_stats and "avg_3_darts" in away_stats:
            val = away_stats["avg_3_darts"]
            if not pd.isna(val):
                self._update_baseline(away_player_id, "avg_3_darts", val)

        # Create history record
        record = Glicko2HistoryRecord(
            match_id=match.get("match_id"),
            league_id=league_id,
            season_id=match.get("season_id"),
            match_date=match_date,
            home_player_id=home_player_id,
            away_player_id=away_player_id,
            home_sets=match.get("home_sets", 0),
            away_sets=match.get("away_sets", 0),
            home_legs=match.get("home_legs", 0),
            away_legs=match.get("away_legs", 0),
            home_result=home_result,
            home_rating_pre=home_rating_pre,
            home_rd_pre=home_rd_pre,
            home_vol_pre=home_vol_pre,
            away_rating_pre=away_rating_pre,
            away_rd_pre=away_rd_pre,
            away_vol_pre=away_vol_pre,
            home_rating_post=home_player.rating,
            home_rd_post=home_player.rd,
            home_vol_post=home_player.volatility,
            away_rating_post=away_player.rating,
            away_rd_post=away_player.rd,
            away_vol_post=away_player.volatility,
            home_expected=home_expected,
            home_expected_lower=home_expected_lower,
            home_expected_upper=home_expected_upper,
            match_weight=match_weight,
            margin_score=margin_score,
            is_tv_event=is_tv,
            recency_weight=recency_weight,
        )

        self.history.append(record)
        return record

    def _adjust_weight_for_performance(
        self,
        match: pd.Series,
        base_weight: float,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        home_player_id: int,
        away_player_id: int,
        home_won: bool,
    ) -> float:
        """Adjust match weight based on performance relative to baseline."""
        home_avg = home_stats.get("avg_3_darts", 0)
        away_avg = away_stats.get("avg_3_darts", 0)

        # Handle NaN values
        if pd.isna(home_avg):
            home_avg = 0
        if pd.isna(away_avg):
            away_avg = 0

        home_baseline = self._get_baseline(home_player_id, "avg_3_darts")
        away_baseline = self._get_baseline(away_player_id, "avg_3_darts")

        if home_baseline is None or away_baseline is None:
            return base_weight

        # Adjust baseline based on opponent strength
        home_player = self.get_player(home_player_id)
        away_player = self.get_player(away_player_id)
        rating_diff = home_player.rating - away_player.rating

        # Against stronger opponents, we expect slightly lower averages
        opponent_adjustment = 1.0 - 0.005 * (rating_diff / 100)
        expected_home_avg = home_baseline * opponent_adjustment
        expected_away_avg = away_baseline * (1.0 / opponent_adjustment)

        # Performance ratios
        home_perf = home_avg / expected_home_avg if expected_home_avg > 0 else 1.0
        away_perf = away_avg / expected_away_avg if expected_away_avg > 0 else 1.0

        # Adjust weight based on winner's performance
        if home_won:
            if home_perf >= 1.0:
                adjustment = 1.0
            else:
                adjustment = 0.8 + 0.2 * home_perf
        else:
            if home_perf > 1.0:
                adjustment = 0.7 + 0.3 / home_perf
            else:
                adjustment = 1.0

        pw = self.config.performance_weight
        return base_weight * ((1 - pw) + pw * adjustment)

    def _get_baseline(self, player_id: int, stat_name: str) -> Optional[float]:
        """Get player baseline statistic."""
        if player_id not in self.player_baselines:
            return None
        return self.player_baselines[player_id].get(stat_name)

    def _update_baseline(self, player_id: int, stat_name: str, value: float) -> None:
        """Update player baseline using EMA."""
        if player_id not in self.player_baselines:
            self.player_baselines[player_id] = {}

        alpha = 0.1
        current = self.player_baselines[player_id].get(stat_name)

        if current is None:
            self.player_baselines[player_id][stat_name] = value
        else:
            self.player_baselines[player_id][stat_name] = alpha * value + (1 - alpha) * current

    def train(
        self,
        matches_df: pd.DataFrame,
        stats_df: Optional[pd.DataFrame] = None,
    ) -> List[Glicko2HistoryRecord]:
        """Train on historical matches."""
        # Sort by date
        if "match_date" in matches_df.columns:
            matches_df = matches_df.sort_values("match_date")

        # Set last training date
        if "match_date" in matches_df.columns and not matches_df.empty:
            max_date = matches_df["match_date"].max()
            if isinstance(max_date, str):
                try:
                    self.last_training_date = date.fromisoformat(max_date)
                except:
                    pass
            elif hasattr(max_date, 'date'):
                self.last_training_date = max_date.date()
            elif isinstance(max_date, date):
                self.last_training_date = max_date

        # Build stats lookup
        stats_lookup = {}
        if stats_df is not None and not stats_df.empty:
            stats_lookup = self._build_stats_lookup(stats_df)

        # Use rating period batching if enabled
        if self.config.use_rating_periods:
            return self._train_by_periods(matches_df, stats_lookup)
        else:
            return self._train_sequential(matches_df, stats_lookup)

    def _train_sequential(
        self,
        matches_df: pd.DataFrame,
        stats_lookup: Dict,
    ) -> List[Glicko2HistoryRecord]:
        """Train sequentially (match by match)."""
        records = []
        for _, match in matches_df.iterrows():
            match_id = match.get("match_id")

            home_stats = stats_lookup.get((match_id, match.get("home_player_id")))
            away_stats = stats_lookup.get((match_id, match.get("away_player_id")))

            try:
                record = self.process_match(match, home_stats, away_stats)
                records.append(record)
            except Exception as e:
                logger.warning(f"Error processing match {match_id}: {e}")

        logger.info(f"Trained on {len(records)} matches, {len(self.ratings)} players")
        return records

    def _train_by_periods(
        self,
        matches_df: pd.DataFrame,
        stats_lookup: Dict,
    ) -> List[Glicko2HistoryRecord]:
        """
        Train using rating period batching.

        Matches within the same period are processed together,
        with all opponent ratings frozen during the period.
        """
        if "match_date" not in matches_df.columns:
            return self._train_sequential(matches_df, stats_lookup)

        # Parse dates
        matches_df = matches_df.copy()
        matches_df["match_date_parsed"] = pd.to_datetime(matches_df["match_date"], errors='coerce')

        # Get min date for period calculation
        min_date = matches_df["match_date_parsed"].min()
        if pd.isna(min_date):
            return self._train_sequential(matches_df, stats_lookup)

        # Assign period numbers
        matches_df["period"] = (
            (matches_df["match_date_parsed"] - min_date).dt.days // self.config.rating_period_days
        )

        records = []

        for period_id in sorted(matches_df["period"].dropna().unique()):
            period_matches = matches_df[matches_df["period"] == period_id]

            # Process each match but aggregate updates
            period_records = []
            for _, match in period_matches.iterrows():
                match_id = match.get("match_id")
                home_stats = stats_lookup.get((match_id, match.get("home_player_id")))
                away_stats = stats_lookup.get((match_id, match.get("away_player_id")))

                try:
                    record = self.process_match(match, home_stats, away_stats)
                    period_records.append(record)
                except Exception as e:
                    logger.warning(f"Error processing match {match_id}: {e}")

            records.extend(period_records)

        logger.info(
            f"Trained on {len(records)} matches in "
            f"{len(matches_df['period'].dropna().unique())} periods, "
            f"{len(self.ratings)} players"
        )
        return records

    def _build_stats_lookup(
        self,
        stats_df: pd.DataFrame,
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Build lookup dict for match statistics."""
        lookup = {}

        for _, row in stats_df.iterrows():
            key = (row["match_id"], row["player_id"])

            if key not in lookup:
                lookup[key] = {}

            stat_id = row["stat_field_id"]
            value = row.get("value")

            try:
                value = float(value) if value else None
            except (ValueError, TypeError):
                value = None

            if value is not None:
                if stat_id == StatField.AVERAGE_3_DARTS:
                    lookup[key]["avg_3_darts"] = value
                elif stat_id == StatField.THROWN_180:
                    lookup[key]["thrown_180"] = value
                elif stat_id == StatField.CHECKOUTS_ACCURACY:
                    lookup[key]["checkout_pct"] = value

        return lookup

    def evaluate(self, matches_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate predictions using proper scoring rules."""
        if not self.history:
            return {"error": "No history"}

        history_df = pd.DataFrame([h.to_dict() for h in self.history])
        match_ids = set(matches_df["match_id"].tolist())
        history_df = history_df[history_df["match_id"].isin(match_ids)]

        if history_df.empty:
            return {"error": "No overlapping matches"}

        actual = history_df["home_result"].values.astype(float)
        predicted = history_df["home_expected"].values.astype(float)

        # Filter out NaN values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[valid_mask]
        predicted = predicted[valid_mask]

        if len(actual) == 0:
            return {"error": "No valid predictions"}

        # Log loss
        eps = 1e-15
        predicted_clipped = np.clip(predicted, eps, 1 - eps)
        log_loss = -np.mean(
            actual * np.log(predicted_clipped) +
            (1 - actual) * np.log(1 - predicted_clipped)
        )

        # Brier score
        brier = np.mean((predicted - actual) ** 2)

        # Accuracy
        accuracy = np.mean((predicted > 0.5).astype(int) == actual)

        # Calibration (binned)
        calibration_error = self._compute_calibration_error(predicted, actual)

        # CI coverage (if available)
        if "home_expected_lower" in history_df.columns and "home_expected_upper" in history_df.columns:
            lower = history_df["home_expected_lower"].values[valid_mask]
            upper = history_df["home_expected_upper"].values[valid_mask]
            # Check if actual result is within predicted CI bounds
            # For binary outcomes, we check if the prediction interval contained a reasonable probability
            ci_coverage = np.mean((lower <= 0.5) | (upper >= 0.5))
        else:
            ci_coverage = None

        result = {
            "log_loss": float(log_loss),
            "brier_score": float(brier),
            "accuracy": float(accuracy),
            "calibration_error": float(calibration_error),
            "n_matches": len(actual),
        }

        if ci_coverage is not None:
            result["ci_coverage"] = float(ci_coverage)

        return result

    def _compute_calibration_error(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute expected calibration error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (predicted > bin_boundaries[i]) & (predicted <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = predicted[in_bin].mean()
                avg_accuracy = actual[in_bin].mean()
                ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

        return ece

    def calibrate_probabilities(
        self,
        holdout_df: Optional[pd.DataFrame] = None,
        use_history: bool = True,
    ) -> Dict[str, float]:
        """
        Fit isotonic regression calibrator on holdout data.

        After calibration, raw probabilities are mapped to calibrated probabilities
        that match empirical win frequencies. E.g., if calibrated P=0.55, then
        historically similar predictions won 55% of the time.

        Args:
            holdout_df: Optional DataFrame with match_id column to filter history.
                       If None, uses recent portion of history.
            use_history: If True, uses internal history records. If False,
                        holdout_df must contain 'home_expected' and 'home_result'.

        Returns:
            Dict with calibration statistics.
        """
        if use_history and not self.history:
            logger.warning("No history available for calibration")
            return {"error": "no_history"}

        # Get predictions and outcomes
        if use_history:
            history_df = pd.DataFrame([h.to_dict() for h in self.history])

            if holdout_df is not None:
                holdout_ids = set(holdout_df["match_id"].tolist())
                history_df = history_df[history_df["match_id"].isin(holdout_ids)]

            if history_df.empty:
                logger.warning("No matching history records for calibration")
                return {"error": "no_matching_history"}

            raw_probs = history_df["home_expected"].values.astype(float)
            outcomes = history_df["home_result"].values.astype(float)
        else:
            if holdout_df is None:
                return {"error": "holdout_df required when use_history=False"}
            raw_probs = holdout_df["home_expected"].values.astype(float)
            outcomes = holdout_df["home_result"].values.astype(float)

        # Filter NaN
        valid_mask = ~(np.isnan(raw_probs) | np.isnan(outcomes))
        raw_probs = raw_probs[valid_mask]
        outcomes = outcomes[valid_mask]

        if len(raw_probs) < 100:
            logger.warning(f"Only {len(raw_probs)} samples for calibration (need 100+)")
            return {"error": "insufficient_samples", "n_samples": len(raw_probs)}

        # Fit isotonic regression
        # y_min/y_max ensure calibrated probs stay in valid range
        self.probability_calibrator = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )
        self.probability_calibrator.fit(raw_probs, outcomes)

        # Compute calibration statistics
        calibrated_probs = self.probability_calibrator.predict(raw_probs)

        # Pre-calibration ECE
        pre_ece = self._compute_calibration_error(raw_probs, outcomes)

        # Post-calibration ECE
        post_ece = self._compute_calibration_error(calibrated_probs, outcomes)

        # Accuracy improvement
        raw_accuracy = np.mean((raw_probs > 0.5).astype(int) == outcomes)
        cal_accuracy = np.mean((calibrated_probs > 0.5).astype(int) == outcomes)

        # Brier scores
        raw_brier = np.mean((raw_probs - outcomes) ** 2)
        cal_brier = np.mean((calibrated_probs - outcomes) ** 2)

        # Log loss
        eps = 1e-15
        raw_probs_clip = np.clip(raw_probs, eps, 1 - eps)
        cal_probs_clip = np.clip(calibrated_probs, eps, 1 - eps)
        raw_log_loss = -np.mean(
            outcomes * np.log(raw_probs_clip) +
            (1 - outcomes) * np.log(1 - raw_probs_clip)
        )
        cal_log_loss = -np.mean(
            outcomes * np.log(cal_probs_clip) +
            (1 - outcomes) * np.log(1 - cal_probs_clip)
        )

        # Sample calibration mapping for reporting
        sample_points = [0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75]
        calibration_map = {}
        for p in sample_points:
            cal_p = float(self.probability_calibrator.predict([[p]])[0])
            calibration_map[f"raw_{int(p*100)}"] = cal_p

        self.calibration_stats = {
            "n_samples": len(raw_probs),
            "pre_calibration_ece": float(pre_ece),
            "post_calibration_ece": float(post_ece),
            "ece_improvement": float(pre_ece - post_ece),
            "raw_accuracy": float(raw_accuracy),
            "calibrated_accuracy": float(cal_accuracy),
            "raw_brier": float(raw_brier),
            "calibrated_brier": float(cal_brier),
            "raw_log_loss": float(raw_log_loss),
            "calibrated_log_loss": float(cal_log_loss),
            **calibration_map,
        }

        logger.info(f"Probability calibration fitted on {len(raw_probs)} samples")
        logger.info(f"  ECE: {pre_ece:.4f} -> {post_ece:.4f} ({pre_ece - post_ece:.4f} improvement)")
        logger.info(f"  Brier: {raw_brier:.4f} -> {cal_brier:.4f}")
        logger.info(f"  Sample mapping: 0.65 -> {calibration_map.get('raw_65', 'N/A'):.3f}, "
                   f"0.70 -> {calibration_map.get('raw_70', 'N/A'):.3f}")

        return self.calibration_stats

    def calibrated_win_probability(
        self,
        player_a_id: int,
        player_b_id: int,
        match_date: Optional[date] = None,
        is_tv_event: bool = False,
        use_form: bool = True,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Get calibrated win probabilities with optional form adjustment.

        If a calibrator has been fitted, applies isotonic regression to map
        raw probabilities to calibrated values that match empirical frequencies.

        Args:
            player_a_id: First player ID
            player_b_id: Second player ID
            match_date: Optional match date for time decay
            is_tv_event: Whether this is a TV event
            use_form: Whether to apply recent form adjustment

        Returns:
            Tuple of (p_a_win, p_b_win, metadata dict)
        """
        # Get raw probability
        p_a_raw, p_b_raw, metadata = self.win_probability(
            player_a_id, player_b_id, match_date, is_tv_event
        )

        # Store raw values in metadata
        metadata["raw_p_a"] = p_a_raw
        metadata["raw_p_b"] = p_b_raw

        # Apply calibration if available
        if self.probability_calibrator is not None:
            p_a_calibrated = float(self.probability_calibrator.predict([[p_a_raw]])[0])
            p_b_calibrated = 1.0 - p_a_calibrated
            metadata["is_calibrated"] = True
        else:
            p_a_calibrated = p_a_raw
            p_b_calibrated = p_b_raw
            metadata["is_calibrated"] = False

        # Apply form adjustment if enabled
        if use_form and self.config.use_form_adjustment:
            player_a = self.get_player(player_a_id)
            player_b = self.get_player(player_b_id)

            form_a = player_a.get_form_factor()
            form_b = player_b.get_form_factor()

            metadata["form_a"] = form_a
            metadata["form_b"] = form_b

            # Net form advantage: positive means player A has better recent form
            net_form = form_a - form_b

            # Apply form adjustment with configured weight
            # Form adjusts probability by up to +/- form_weight * net_form
            form_adjustment = self.config.form_weight * net_form
            p_a_final = p_a_calibrated + form_adjustment

            # Clamp to valid probability range
            p_a_final = max(0.01, min(0.99, p_a_final))
            p_b_final = 1.0 - p_a_final

            metadata["form_adjustment"] = form_adjustment
            metadata["p_a_before_form"] = p_a_calibrated

            return (p_a_final, p_b_final, metadata)

        return (p_a_calibrated, p_b_calibrated, metadata)

    def get_top_players(self, n: int = 20, min_matches: int = 10) -> pd.DataFrame:
        """Get top N players by rating."""
        data = []
        for pid, player in self.ratings.items():
            if player.match_count < min_matches:
                continue

            data.append({
                "player_id": pid,
                "rating": player.rating,
                "rd": player.rd,
                "volatility": player.volatility,
                "matches": player.match_count,
                "rating_lower": player.rating - 2 * player.rd,
                "rating_upper": player.rating + 2 * player.rd,
                "tv_bonus": player.tv_bonus,
                "tv_matches": player.tv_match_count,
                "floor_matches": player.floor_match_count,
            })

        df = pd.DataFrame(data)
        if df.empty:
            return df

        return df.sort_values("rating", ascending=False).head(n)

    def save(self, path: str) -> None:
        """Save system state."""
        import json

        ratings_dict = {}
        for pid, player in self.ratings.items():
            ratings_dict[str(pid)] = {
                "rating": player.rating,
                "rd": player.rd,
                "volatility": player.volatility,
                "last_played": (
                    player.last_played.isoformat()
                    if player.last_played and hasattr(player.last_played, 'isoformat')
                    else player.last_played
                ),
                "match_count": player.match_count,
                "tv_bonus": player.tv_bonus,
                "tv_bonus_rd": player.tv_bonus_rd,
                "floor_match_count": player.floor_match_count,
                "tv_match_count": player.tv_match_count,
                "recent_results": player.recent_results,  # List of (actual, expected) tuples
            }

        # Serialize probability calibrator if fitted
        calibrator_data = None
        if self.probability_calibrator is not None:
            try:
                calibrator_data = {
                    "X_thresholds_": self.probability_calibrator.X_thresholds_.tolist(),
                    "y_thresholds_": self.probability_calibrator.y_thresholds_.tolist(),
                    "X_min_": float(self.probability_calibrator.X_min_),
                    "X_max_": float(self.probability_calibrator.X_max_),
                    "y_min": self.probability_calibrator.y_min,
                    "y_max": self.probability_calibrator.y_max,
                }
            except AttributeError:
                logger.warning("Could not serialize probability calibrator")
                calibrator_data = None

        state = {
            "ratings": ratings_dict,
            "player_baselines": {str(k): v for k, v in self.player_baselines.items()},
            "venue_effects": {str(k): v for k, v in self.venue_effects.items()},
            "last_training_date": (
                self.last_training_date.isoformat()
                if self.last_training_date
                else None
            ),
            "config": {
                "initial_rating": self.config.initial_rating,
                "initial_rd": self.config.initial_rd,
                "initial_volatility": self.config.initial_volatility,
                "rd_decay_per_day": self.config.rd_decay_per_day,
                "use_match_length_weighting": self.config.use_match_length_weighting,
                "use_venue_effects": self.config.use_venue_effects,
                "use_performance_adjustment": self.config.use_performance_adjustment,
                "use_margin_scoring": self.config.use_margin_scoring,
                "margin_weight": self.config.margin_weight,
                "use_rating_periods": self.config.use_rating_periods,
                "rating_period_days": self.config.rating_period_days,
                "use_surface_ratings": self.config.use_surface_ratings,
                "use_recency_weighting": self.config.use_recency_weighting,
                "recency_half_life_days": self.config.recency_half_life_days,
                "use_form_adjustment": self.config.use_form_adjustment,
                "form_weight": self.config.form_weight,
                "form_window": self.config.form_window,
            },
            "probability_calibrator": calibrator_data,
            "calibration_stats": self.calibration_stats,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Glicko-2 system saved to {path}")
        if calibrator_data:
            logger.info("  (includes probability calibrator)")

    def load(self, path: str) -> None:
        """Load system state."""
        import json

        with open(path, "r") as f:
            state = json.load(f)

        # Restore ratings
        self.ratings = {}
        for pid_str, data in state["ratings"].items():
            pid = int(pid_str)
            last_played = data.get("last_played")
            if last_played and isinstance(last_played, str):
                try:
                    last_played = date.fromisoformat(last_played)
                except:
                    last_played = None

            # Restore recent_results as list of tuples
            recent_results = data.get("recent_results", [])
            if recent_results:
                recent_results = [tuple(r) for r in recent_results]

            self.ratings[pid] = PlayerRating(
                player_id=pid,
                rating=data["rating"],
                rd=data["rd"],
                volatility=data["volatility"],
                last_played=last_played,
                match_count=data.get("match_count", 0),
                tv_bonus=data.get("tv_bonus", 0.0),
                tv_bonus_rd=data.get("tv_bonus_rd", 200.0),
                floor_match_count=data.get("floor_match_count", 0),
                tv_match_count=data.get("tv_match_count", 0),
                recent_results=recent_results,
            )

        self.player_baselines = {int(k): v for k, v in state.get("player_baselines", {}).items()}
        self.venue_effects = {int(k): v for k, v in state.get("venue_effects", {}).items()}

        last_date = state.get("last_training_date")
        if last_date:
            try:
                self.last_training_date = date.fromisoformat(last_date)
            except:
                pass

        config_data = state.get("config", {})
        self.config.initial_rating = config_data.get("initial_rating", self.config.initial_rating)
        self.config.initial_rd = config_data.get("initial_rd", self.config.initial_rd)
        self.config.rd_decay_per_day = config_data.get("rd_decay_per_day", self.config.rd_decay_per_day)
        self.config.use_margin_scoring = config_data.get("use_margin_scoring", True)
        self.config.margin_weight = config_data.get("margin_weight", 0.3)
        self.config.use_rating_periods = config_data.get("use_rating_periods", True)
        self.config.rating_period_days = config_data.get("rating_period_days", 7)
        self.config.use_surface_ratings = config_data.get("use_surface_ratings", True)
        self.config.use_recency_weighting = config_data.get("use_recency_weighting", True)
        self.config.recency_half_life_days = config_data.get("recency_half_life_days", 180.0)
        self.config.use_form_adjustment = config_data.get("use_form_adjustment", True)
        self.config.form_weight = config_data.get("form_weight", 0.5)
        self.config.form_window = config_data.get("form_window", 5)

        # Restore probability calibrator
        calibrator_data = state.get("probability_calibrator")
        if calibrator_data:
            try:
                from scipy.interpolate import interp1d

                self.probability_calibrator = IsotonicRegression(
                    y_min=calibrator_data.get("y_min", 0.01),
                    y_max=calibrator_data.get("y_max", 0.99),
                    out_of_bounds="clip",
                )
                # Restore fitted parameters
                X_thresholds = np.array(calibrator_data["X_thresholds_"])
                y_thresholds = np.array(calibrator_data["y_thresholds_"])
                self.probability_calibrator.X_thresholds_ = X_thresholds
                self.probability_calibrator.y_thresholds_ = y_thresholds
                self.probability_calibrator.X_min_ = calibrator_data["X_min_"]
                self.probability_calibrator.X_max_ = calibrator_data["X_max_"]

                # Rebuild the interpolation function
                self.probability_calibrator.f_ = interp1d(
                    X_thresholds, y_thresholds,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(y_thresholds[0], y_thresholds[-1]),
                )
                logger.info("  (probability calibrator restored)")
            except (KeyError, TypeError, ImportError) as e:
                logger.warning(f"Could not restore probability calibrator: {e}")
                self.probability_calibrator = None
        else:
            self.probability_calibrator = None

        # Restore calibration stats
        self.calibration_stats = state.get("calibration_stats", {})

        logger.info(f"Glicko-2 system loaded from {path}")
