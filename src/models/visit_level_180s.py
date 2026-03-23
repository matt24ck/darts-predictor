"""
Visit-Level 180s Model

A statistically principled model that treats each 3-dart visit as a Bernoulli trial.
With 60k+ visit records, this provides more accurate player-level 180 rate estimates
than match-level aggregation.

Key features:
1. Visit-level logistic model: P(180|visit, player, league)
2. Empirical Bayes shrinkage: Players shrunk toward population mean based on sample size
3. League effects: Different leagues have different 180 rates
4. Fallback: Players without visit data use match-level stats with stronger shrinkage
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class PlayerVisitStats:
    """Visit-level statistics for a player."""
    player_id: int
    total_visits: int
    total_180s: int
    raw_rate: float  # Observed P(180|visit)
    shrunk_rate: float  # After shrinkage toward population
    rate_std: float  # Standard error of shrunk rate
    has_visit_data: bool  # True if from visit data, False if from match stats


@dataclass
class VisitLevel180sResult:
    """Prediction result from the visit-level model."""
    lambda_total: float  # Expected total 180s
    lambda_home: float  # Expected 180s for home player
    lambda_away: float  # Expected 180s for away player
    lambda_total_lower: float  # 95% CI lower bound
    lambda_total_upper: float  # 95% CI upper bound
    home_rate: float  # P(180|visit) for home player
    away_rate: float  # P(180|visit) for away player
    expected_visits: float  # Expected visits in the match
    league_multiplier: float  # League effect applied


class VisitLevel180sModel:
    """
    Visit-level 180s prediction model.

    Uses individual visit data (is_180 binary outcome) to estimate player-level
    180 rates with proper uncertainty quantification and shrinkage.
    """

    def __init__(
        self,
        prior_strength: float = 50.0,  # Effective prior sample size for shrinkage
        min_visits_for_direct: int = 30,  # Min visits to use direct estimate
        fallback_weight: float = 0.1,  # Weight for match-level fallback
    ):
        self.prior_strength = prior_strength
        self.min_visits_for_direct = min_visits_for_direct
        self.fallback_weight = fallback_weight

        # Population parameters
        self.population_rate: float = 0.05  # Prior P(180|visit)
        self.population_rate_std: float = 0.02  # Std of player rates

        # Player-level parameters
        self.player_stats: Dict[int, PlayerVisitStats] = {}

        # League effects (multiplicative)
        self.league_multipliers: Dict[int, float] = {}
        self.global_rate: float = 0.05

        # Format -> expected visits mapping
        self.format_visits: Dict[Tuple[int, int, bool], float] = {}
        self.base_visits_per_leg: float = 9.0

        # Overdispersion for match-level variance
        self.overdispersion: float = 1.2  # Global fallback
        self.overdispersion_by_format: Dict[str, float] = {}  # "leg" / "set" -> phi

        # Bias correction (calibrated after training)
        self.bias_correction: float = 0.0

        # Regression-based bias correction: adjustment = slope * expected_visits + intercept
        # Separate regressions for LEG vs SET formats (they have different patterns)
        self.bias_slope_leg: float = 0.0  # Per-visit adjustment for LEG format
        self.bias_intercept_leg: float = 0.0  # Base adjustment for LEG format
        self.bias_slope_set: float = 0.0  # Per-visit adjustment for SET format
        self.bias_intercept_set: float = 0.0  # Base adjustment for SET format

        # Legacy: format-specific bias corrections (kept for backwards compatibility)
        self.format_bias_corrections: Dict[Tuple[int, int, bool], float] = {}

        # Quality-tier bias corrections (keyed by tier: "low", "mid_low", "mid_high", "high")
        self.quality_tier_corrections: Dict[str, float] = {}
        self.quality_tier_thresholds: List[float] = []  # Rate thresholds for tiers

        self.is_fitted = False

    def fit(
        self,
        visits_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        stats_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fit the visit-level model.

        Args:
            visits_df: Visit-level data with is_180 column
            matches_df: Match data for format information
            stats_df: Optional match-level stats for fallback
        """
        logger.info("Fitting visit-level 180s model...")

        # Step 1: Filter to 3-dart visits
        full_visits = visits_df[visits_df["attempts"] == 3].copy()
        if full_visits.empty:
            full_visits = visits_df.copy()
            logger.warning("No 3-dart visits found, using all visits")

        logger.info(f"Training on {len(full_visits):,} 3-dart visits")

        # Step 2: Compute global rate
        self.global_rate = full_visits["is_180"].mean()
        self.population_rate = self.global_rate
        logger.info(f"Global 180 rate: {self.global_rate:.4f} ({100*self.global_rate:.2f}%)")

        # Step 3: Compute league effects
        self._compute_league_effects(full_visits)

        # Step 4: Compute player-level rates with shrinkage
        self._compute_player_rates(full_visits)

        # Step 5: Add fallback players from match stats
        if stats_df is not None and not stats_df.empty:
            self._add_fallback_players(stats_df, matches_df)

        # Step 6: Compute format -> visits mapping
        self._compute_format_visits(full_visits, matches_df)

        # Step 7: Estimate overdispersion
        self._estimate_overdispersion(full_visits, matches_df)

        self.is_fitted = True
        logger.info(f"Model fitted with {len(self.player_stats)} players "
                   f"({sum(1 for p in self.player_stats.values() if p.has_visit_data)} with visit data)")

    def calibrate(
        self,
        matches_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        visits_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Calibrate bias correction using actual match outcomes.

        Computes mean bias and optionally format-specific biases.

        Args:
            matches_df: Match data with format info
            stats_df: Match stats with actual 180s counts
            visits_df: Optional visit data for actual 180s counts

        Returns:
            Dict with calibration metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calibration")

        logger.info("Calibrating model bias...")

        # Get actual 180s per match
        stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
        if stat_180s.empty:
            stat_180s = stats_df[stats_df["stat_field_name"] == "Thrown 180"].copy()

        if not stat_180s.empty:
            stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)
            match_180s = stat_180s.groupby("match_id")["value_num"].sum().to_dict()
        else:
            match_180s = {}

        # Fallback to visit counts
        if visits_df is not None and not visits_df.empty:
            visit_180s = visits_df.groupby("match_id")["is_180"].sum().to_dict()
        else:
            visit_180s = {}

        predictions = []
        actuals = []
        format_data = []  # ((best_of_sets, best_of_legs, is_set_format), prediction, actual)

        for _, match in matches_df.iterrows():
            match_id = match["match_id"]
            actual = match_180s.get(match_id, visit_180s.get(match_id))

            if actual is None:
                continue

            home_id = match["home_player_id"]
            away_id = match["away_player_id"]
            league_id = match.get("league_id", 2)

            format_params = {
                "best_of_sets": match.get("best_of_sets", 0),
                "best_of_legs": match.get("best_of_legs", 5),
                "is_set_format": match.get("is_set_format", False),
            }

            # Predict without bias correction
            old_bias = self.bias_correction
            self.bias_correction = 0.0
            pred = self.predict(home_id, away_id, league_id, format_params)
            self.bias_correction = old_bias

            # Store full format key
            format_key = (
                int(format_params["best_of_sets"]),
                int(format_params["best_of_legs"]),
                bool(format_params["is_set_format"]),
            )

            predictions.append(pred.lambda_total)
            actuals.append(actual)
            format_data.append((format_key, pred.lambda_total, actual))

        if len(predictions) < 10:
            logger.warning("Not enough matches for calibration")
            return {"error": "insufficient_data"}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Compute overall bias
        mean_pred = np.mean(predictions)
        mean_actual = np.mean(actuals)
        bias = mean_actual - mean_pred  # Positive = underpredicting

        # Set bias correction
        self.bias_correction = bias
        logger.info(f"Bias correction set to {bias:.3f}")
        logger.info(f"  Mean predicted: {mean_pred:.2f}")
        logger.info(f"  Mean actual: {mean_actual:.2f}")

        # Analyze and store format-specific bias corrections
        format_df = pd.DataFrame(format_data, columns=["format_key", "pred", "actual"])
        format_df["residual"] = format_df["actual"] - format_df["pred"]

        # Group by full format key (best_of_sets, best_of_legs, is_set_format)
        format_bias = format_df.groupby("format_key").agg({
            "residual": ["mean", "count"],
            "pred": "mean",
            "actual": "mean",
        })
        format_bias.columns = ["bias", "count", "mean_pred", "mean_actual"]

        # Store format-specific corrections for formats with sufficient data
        self.format_bias_corrections = {}
        logger.info("\nFormat-specific bias corrections:")
        for format_key, row in format_bias.iterrows():
            bos, bol, is_set = format_key
            format_type = "SET" if is_set else "LEG"
            if row["count"] >= 15:  # Need enough samples for reliable estimate
                # Store the delta from global bias (format-specific adjustment)
                format_adjustment = row["bias"] - bias
                self.format_bias_corrections[format_key] = format_adjustment
                logger.info(f"  BO{bos}s/BO{bol}l ({format_type}): adjustment={format_adjustment:+.2f} "
                           f"(total bias={row['bias']:+.2f}, n={row['count']:.0f})")
            elif row["count"] >= 5:
                logger.info(f"  BO{bos}s/BO{bol}l ({format_type}): bias={row['bias']:+.2f} "
                           f"(n={row['count']:.0f}, not enough for correction)")

        # Apply format-specific corrections to compute corrected predictions
        corrected_preds = []
        for i, (format_key, pred, actual) in enumerate(format_data):
            format_adj = self.format_bias_corrections.get(format_key, 0.0)
            corrected_preds.append(pred + bias + format_adj)

        corrected_preds = np.array(corrected_preds)
        rmse = np.sqrt(np.mean((corrected_preds - actuals) ** 2))
        mae = np.mean(np.abs(corrected_preds - actuals))
        ss_tot = np.sum((actuals - mean_actual) ** 2)
        ss_res = np.sum((actuals - corrected_preds) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            "bias_correction": bias,
            "n_format_corrections": len(self.format_bias_corrections),
            "rmse_corrected": rmse,
            "mae_corrected": mae,
            "r2_corrected": r2,
            "n_matches": len(predictions),
        }

        logger.info(f"\nCorrected metrics (with format adjustments):")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  MAE: {mae:.3f}")
        logger.info(f"  R²: {r2:.3f}")

        return metrics

    def _compute_league_effects(self, visits_df: pd.DataFrame) -> None:
        """Compute league-specific 180 rate multipliers."""
        if "league_id" not in visits_df.columns:
            logger.warning("No league_id in visits data")
            return

        league_stats = visits_df.groupby("league_id").agg({
            "is_180": ["sum", "count", "mean"]
        })
        league_stats.columns = ["total_180s", "total_visits", "rate"]

        for league_id, row in league_stats.iterrows():
            if row["total_visits"] >= 100 and self.global_rate > 0:
                multiplier = row["rate"] / self.global_rate
                self.league_multipliers[int(league_id)] = multiplier
                logger.info(f"League {league_id}: rate={row['rate']:.4f}, "
                           f"multiplier={multiplier:.3f} ({row['total_visits']:.0f} visits)")

    def _compute_player_rates(self, visits_df: pd.DataFrame) -> None:
        """Compute player-level 180 rates with empirical Bayes shrinkage."""

        # Aggregate by player
        player_agg = visits_df.groupby("player_id").agg({
            "is_180": ["sum", "count"]
        })
        player_agg.columns = ["total_180s", "total_visits"]

        # Compute variance of player rates (for shrinkage)
        raw_rates = player_agg["total_180s"] / player_agg["total_visits"]

        # Only use players with sufficient data for variance estimate
        sufficient_data = player_agg["total_visits"] >= self.min_visits_for_direct
        if sufficient_data.sum() > 10:
            self.population_rate_std = raw_rates[sufficient_data].std()
        else:
            self.population_rate_std = 0.02  # Default

        logger.info(f"Population rate std: {self.population_rate_std:.4f}")

        # Compute shrunk rates for each player
        for player_id, row in player_agg.iterrows():
            n = row["total_visits"]
            k = row["total_180s"]
            raw_rate = k / n if n > 0 else self.population_rate

            # Empirical Bayes shrinkage
            # shrunk_rate = (n * raw_rate + prior_strength * prior_rate) / (n + prior_strength)
            shrunk_rate = (k + self.prior_strength * self.population_rate) / (n + self.prior_strength)

            # Standard error of shrunk rate (using posterior variance)
            # For Beta-Binomial: Var = (n * p * (1-p)) / (n + prior_strength)^2
            # Simplified approximation:
            rate_var = (shrunk_rate * (1 - shrunk_rate)) / (n + self.prior_strength)
            rate_std = np.sqrt(rate_var)

            self.player_stats[int(player_id)] = PlayerVisitStats(
                player_id=int(player_id),
                total_visits=int(n),
                total_180s=int(k),
                raw_rate=raw_rate,
                shrunk_rate=shrunk_rate,
                rate_std=rate_std,
                has_visit_data=True,
            )

        logger.info(f"Computed rates for {len(self.player_stats)} players from visit data")

    def _add_fallback_players(
        self,
        stats_df: pd.DataFrame,
        matches_df: pd.DataFrame,
    ) -> None:
        """Add players from match-level stats who don't have visit data."""

        # Get 180s stats
        stat_180s = stats_df[stats_df["stat_field_id"] == 2].copy()
        if stat_180s.empty:
            stat_180s = stats_df[stats_df["stat_field_name"] == "Thrown 180"].copy()

        if stat_180s.empty:
            return

        stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)

        # Aggregate by player
        player_180s = stat_180s.groupby("player_id")["value_num"].sum()

        # Count matches per player
        home_counts = matches_df.groupby("home_player_id").size()
        away_counts = matches_df.groupby("away_player_id").size()
        match_counts = home_counts.add(away_counts, fill_value=0)

        # Estimate visits from matches (heuristic)
        avg_visits_per_match = 50  # Rough estimate

        added = 0
        for player_id in player_180s.index:
            if player_id in self.player_stats:
                continue  # Already have visit data

            total_180s = player_180s[player_id]
            n_matches = match_counts.get(player_id, 0)

            if n_matches < 3:
                continue  # Too few matches

            if total_180s == 0:
                continue  # Can't distinguish "zero 180s" from "stats not populated"

            # Estimate visits
            est_visits = n_matches * avg_visits_per_match / 2  # Per player

            if est_visits < 10:
                continue

            # Estimate raw rate
            raw_rate = total_180s / est_visits
            raw_rate = min(raw_rate, 0.15)  # Cap at reasonable max

            # Apply stronger shrinkage for estimated data
            effective_prior = self.prior_strength * 2  # Double shrinkage
            shrunk_rate = (total_180s + effective_prior * self.population_rate) / (est_visits + effective_prior)

            # Higher uncertainty for estimated data
            rate_std = np.sqrt(shrunk_rate * (1 - shrunk_rate) / (est_visits / 2 + effective_prior))

            self.player_stats[int(player_id)] = PlayerVisitStats(
                player_id=int(player_id),
                total_visits=int(est_visits),
                total_180s=int(total_180s),
                raw_rate=raw_rate,
                shrunk_rate=shrunk_rate,
                rate_std=rate_std,
                has_visit_data=False,  # From match stats
            )
            added += 1

        logger.info(f"Added {added} players from match-level stats (fallback)")

    def _compute_format_visits(
        self,
        visits_df: pd.DataFrame,
        matches_df: pd.DataFrame,
    ) -> None:
        """Compute expected visits per format from actual data."""

        # Count visits per match
        visits_per_match = visits_df.groupby("match_id").size()

        # Merge with match format info
        match_formats = matches_df[["match_id", "best_of_sets", "best_of_legs", "is_set_format"]].copy()
        match_formats["total_visits"] = match_formats["match_id"].map(visits_per_match)
        match_formats = match_formats.dropna(subset=["total_visits"])

        if match_formats.empty:
            return

        # Group by format
        format_groups = match_formats.groupby(["best_of_sets", "best_of_legs", "is_set_format"])

        for (bos, bol, is_set), group in format_groups:
            if len(group) >= 5:  # Min matches for reliable estimate
                avg_visits = group["total_visits"].mean()
                self.format_visits[(int(bos), int(bol), bool(is_set))] = avg_visits

        # Compute base visits per leg
        if match_formats["total_visits"].notna().sum() > 0:
            # Estimate from data
            match_formats["total_legs"] = matches_df.loc[match_formats.index, "home_legs"] + \
                                         matches_df.loc[match_formats.index, "away_legs"]
            valid = match_formats["total_legs"] > 0
            if valid.sum() > 0:
                self.base_visits_per_leg = (match_formats.loc[valid, "total_visits"] /
                                           match_formats.loc[valid, "total_legs"]).mean()

        logger.info(f"Computed visits for {len(self.format_visits)} formats, "
                   f"base visits/leg: {self.base_visits_per_leg:.1f}")

    def _estimate_overdispersion(
        self,
        visits_df: pd.DataFrame,
        matches_df: pd.DataFrame,
    ) -> None:
        """Estimate overdispersion using Pearson residual dispersion.

        Computes sum((actual - predicted)^2 / predicted) / (n - 1),
        separately for leg and set formats, plus a global fallback.
        """
        # Count actual 180s per match
        match_180s = visits_df.groupby("match_id")["is_180"].sum()

        if len(match_180s) < 20:
            return

        # Accumulate Pearson residuals per format category
        pearson_by_format: Dict[str, list] = {"leg": [], "set": [], "all": []}

        for _, row in matches_df.iterrows():
            mid = row["match_id"]
            if mid not in match_180s.index:
                continue

            actual = float(match_180s[mid])

            try:
                format_params = {
                    "best_of_sets": int(row.get("best_of_sets", 0)),
                    "best_of_legs": int(row.get("best_of_legs", 5)),
                    "is_set_format": bool(row.get("is_set_format", False)),
                }
                pred = self.predict(
                    home_player_id=int(row["home_player_id"]),
                    away_player_id=int(row["away_player_id"]),
                    league_id=int(row.get("league_id", 2)),
                    format_params=format_params,
                )
                predicted = pred.lambda_total
                if predicted > 0.01:
                    residual = (actual - predicted) ** 2 / predicted
                    key = "set" if format_params["is_set_format"] else "leg"
                    pearson_by_format[key].append(residual)
                    pearson_by_format["all"].append(residual)
            except Exception:
                continue

        # Global overdispersion
        all_resids = pearson_by_format["all"]
        if len(all_resids) >= 20:
            self.overdispersion = max(1.0, sum(all_resids) / (len(all_resids) - 1))
        logger.info(f"Overdispersion global (Pearson, n={len(all_resids)}): {self.overdispersion:.2f}")

        # Per-format overdispersion
        self.overdispersion_by_format = {}
        for key in ("leg", "set"):
            resids = pearson_by_format[key]
            if len(resids) >= 20:
                phi = max(1.0, sum(resids) / (len(resids) - 1))
                self.overdispersion_by_format[key] = phi
                logger.info(f"Overdispersion {key} (Pearson, n={len(resids)}): {phi:.2f}")
            else:
                logger.info(f"Overdispersion {key}: insufficient data (n={len(resids)}), using global")

    def predict(
        self,
        home_player_id: int,
        away_player_id: int,
        league_id: int,
        format_params: Dict[str, Any],
    ) -> VisitLevel180sResult:
        """
        Predict expected 180s for a match.

        Args:
            home_player_id: ID of home player
            away_player_id: ID of away player
            league_id: League/tournament ID
            format_params: Dict with best_of_sets, best_of_legs, is_set_format
        """
        if not self.is_fitted:
            return VisitLevel180sResult(
                lambda_total=5.0,
                lambda_home=2.5,
                lambda_away=2.5,
                lambda_total_lower=1.0,
                lambda_total_upper=10.0,
                home_rate=self.population_rate,
                away_rate=self.population_rate,
                expected_visits=100.0,
                league_multiplier=1.0,
            )

        # Get player rates
        home_stats = self.player_stats.get(home_player_id)
        away_stats = self.player_stats.get(away_player_id)

        home_rate = home_stats.shrunk_rate if home_stats else self.population_rate
        away_rate = away_stats.shrunk_rate if away_stats else self.population_rate

        # Get rate uncertainties
        home_std = home_stats.rate_std if home_stats else self.population_rate_std
        away_std = away_stats.rate_std if away_stats else self.population_rate_std

        # Estimate expected visits
        expected_visits = self._estimate_visits(format_params)

        # Apply league effect
        league_mult = self.league_multipliers.get(league_id, 1.0)

        # Calculate expected 180s
        # Each player gets half the visits (approximately)
        home_visits = expected_visits / 2
        away_visits = expected_visits / 2

        home_lambda_raw = home_rate * league_mult * home_visits
        away_lambda_raw = away_rate * league_mult * away_visits
        lambda_total_raw = home_lambda_raw + away_lambda_raw

        # Calculate bias correction using regression on expected visits
        # Separate regressions for LEG vs SET formats
        is_set = format_params.get("is_set_format", False)
        if is_set:
            regression_adjustment = self.bias_intercept_set + self.bias_slope_set * expected_visits
        else:
            regression_adjustment = self.bias_intercept_leg + self.bias_slope_leg * expected_visits
        total_bias = self.bias_correction + regression_adjustment

        # Apply bias correction (distribute proportionally)
        if total_bias != 0.0 and lambda_total_raw > 0:
            home_share = home_lambda_raw / lambda_total_raw
            away_share = away_lambda_raw / lambda_total_raw
            home_lambda = home_lambda_raw + total_bias * home_share
            away_lambda = away_lambda_raw + total_bias * away_share
            lambda_total = lambda_total_raw + total_bias
        else:
            home_lambda = home_lambda_raw
            away_lambda = away_lambda_raw
            lambda_total = lambda_total_raw

        # Ensure non-negative
        home_lambda = max(0.0, home_lambda)
        away_lambda = max(0.0, away_lambda)
        lambda_total = max(0.0, lambda_total)

        # Compute uncertainty (combining rate uncertainty and Poisson variance)
        # Var(λ) ≈ visits² * var(rate) + rate² * var(visits) + visits * rate * overdispersion
        home_var = (home_visits ** 2) * (home_std ** 2) * (league_mult ** 2) + \
                   home_lambda * self.overdispersion
        away_var = (away_visits ** 2) * (away_std ** 2) * (league_mult ** 2) + \
                   away_lambda * self.overdispersion

        total_var = home_var + away_var
        total_std = np.sqrt(total_var)

        # 95% CI
        lambda_lower = max(0.1, lambda_total - 1.96 * total_std)
        lambda_upper = lambda_total + 1.96 * total_std

        return VisitLevel180sResult(
            lambda_total=lambda_total,
            lambda_home=home_lambda,
            lambda_away=away_lambda,
            lambda_total_lower=lambda_lower,
            lambda_total_upper=lambda_upper,
            home_rate=home_rate * league_mult,
            away_rate=away_rate * league_mult,
            expected_visits=expected_visits,
            league_multiplier=league_mult,
        )

    def _estimate_visits(self, format_params: Dict[str, Any]) -> float:
        """Estimate expected visits for a match format."""
        bos = format_params.get("best_of_sets", 0)
        bol = format_params.get("best_of_legs", 5)
        is_set = format_params.get("is_set_format", False)

        # Check if we have data for this exact format
        key = (int(bos), int(bol), bool(is_set))
        if key in self.format_visits:
            return self.format_visits[key]

        # Estimate from format parameters
        if is_set and bos > 0:
            # Set format: estimate legs per set, then total
            expected_sets = bos * 0.6  # ~60% of max sets played
            expected_legs_per_set = bol * 0.6
            total_legs = expected_sets * expected_legs_per_set
        else:
            # Leg format
            total_legs = bol * 0.6  # ~60% of max legs played

        return total_legs * self.base_visits_per_leg

    def get_player_rate(self, player_id: int) -> Tuple[float, float]:
        """Get a player's 180 rate and uncertainty."""
        stats = self.player_stats.get(player_id)
        if stats:
            return stats.shrunk_rate, stats.rate_std
        return self.population_rate, self.population_rate_std

    def get_overdispersion(self, format_params: Optional[Dict[str, Any]] = None) -> float:
        """Get overdispersion for a specific match format.

        Uses format-specific values when available, falls back to global.
        """
        if format_params and self.overdispersion_by_format:
            key = "set" if format_params.get("is_set_format", False) else "leg"
            if key in self.overdispersion_by_format:
                return self.overdispersion_by_format[key]
        return self.overdispersion

    def save(self, path: str) -> None:
        """Save model to JSON file."""
        import json

        player_stats_dict = {}
        for pid, ps in self.player_stats.items():
            player_stats_dict[str(pid)] = {
                "total_visits": ps.total_visits,
                "total_180s": ps.total_180s,
                "raw_rate": ps.raw_rate,
                "shrunk_rate": ps.shrunk_rate,
                "rate_std": ps.rate_std,
                "has_visit_data": ps.has_visit_data,
            }

        model_data = {
            "prior_strength": self.prior_strength,
            "min_visits_for_direct": self.min_visits_for_direct,
            "fallback_weight": self.fallback_weight,
            "population_rate": self.population_rate,
            "population_rate_std": self.population_rate_std,
            "global_rate": self.global_rate,
            "player_stats": player_stats_dict,
            "league_multipliers": {str(k): v for k, v in self.league_multipliers.items()},
            "format_visits": {str(k): v for k, v in self.format_visits.items()},
            "base_visits_per_leg": self.base_visits_per_leg,
            "overdispersion": self.overdispersion,
            "overdispersion_by_format": self.overdispersion_by_format,
            "bias_correction": self.bias_correction,
            "bias_slope_leg": self.bias_slope_leg,
            "bias_intercept_leg": self.bias_intercept_leg,
            "bias_slope_set": self.bias_slope_set,
            "bias_intercept_set": self.bias_intercept_set,
            # Legacy: format corrections (kept for backwards compatibility)
            "format_bias_corrections": {str(k): v for k, v in self.format_bias_corrections.items()},
            "is_fitted": self.is_fitted,
        }

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from JSON file."""
        import json

        with open(path, "r") as f:
            model_data = json.load(f)

        self.prior_strength = model_data.get("prior_strength", 50.0)
        self.min_visits_for_direct = model_data.get("min_visits_for_direct", 30)
        self.fallback_weight = model_data.get("fallback_weight", 0.1)
        self.population_rate = model_data["population_rate"]
        self.population_rate_std = model_data.get("population_rate_std", 0.02)
        self.global_rate = model_data.get("global_rate", self.population_rate)
        self.base_visits_per_leg = model_data.get("base_visits_per_leg", 9.0)
        self.overdispersion = model_data.get("overdispersion", 1.2)
        self.overdispersion_by_format = model_data.get("overdispersion_by_format", {})
        self.bias_correction = model_data.get("bias_correction", 0.0)
        self.bias_slope_leg = model_data.get("bias_slope_leg", model_data.get("bias_slope", 0.0))
        self.bias_intercept_leg = model_data.get("bias_intercept_leg", model_data.get("bias_intercept", 0.0))
        self.bias_slope_set = model_data.get("bias_slope_set", 0.0)
        self.bias_intercept_set = model_data.get("bias_intercept_set", 0.0)
        self.is_fitted = model_data["is_fitted"]

        # Reconstruct player stats
        self.player_stats = {}
        for pid_str, ps_dict in model_data["player_stats"].items():
            pid = int(pid_str)
            self.player_stats[pid] = PlayerVisitStats(
                player_id=pid,
                total_visits=ps_dict["total_visits"],
                total_180s=ps_dict["total_180s"],
                raw_rate=ps_dict["raw_rate"],
                shrunk_rate=ps_dict["shrunk_rate"],
                rate_std=ps_dict.get("rate_std", 0.01),
                has_visit_data=ps_dict.get("has_visit_data", True),
            )

        self.league_multipliers = {int(k): v for k, v in model_data["league_multipliers"].items()}
        self.format_visits = {eval(k): v for k, v in model_data.get("format_visits", {}).items()}

        # Load format bias corrections - handle both old (int key) and new (tuple key) formats
        raw_corrections = model_data.get("format_bias_corrections", {})
        self.format_bias_corrections = {}
        for k, v in raw_corrections.items():
            try:
                # Try to parse as tuple (new format)
                key = eval(k)
                if isinstance(key, tuple) and len(key) == 3:
                    self.format_bias_corrections[key] = v
                else:
                    # Old format: just best_of_legs as int - skip, will recalibrate
                    pass
            except:
                # Couldn't parse, skip
                pass

        logger.info(f"Model loaded from {path}")
        if self.format_bias_corrections:
            logger.info(f"  Format corrections for {len(self.format_bias_corrections)} format combinations")
