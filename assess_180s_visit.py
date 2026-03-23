#!/usr/bin/env python3
"""
Visit-Level 180s Model Diagnostics

Analyzes:
1. Residuals vs predicted (heteroscedasticity check)
2. Bias by format segment (Bo11, Bo13, Bo21, etc.)
3. Bias by player quality tier
4. Distributional fit (empirical vs predicted variance)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.storage import ParquetStore
from src.models import VisitLevel180sModel


def load_data_and_model():
    """Load data and trained model."""
    store = ParquetStore("data/processed")
    matches = store.get_matches()
    visits = store.get_visits()
    stats = store.get_match_stats()
    players = store.get_players()

    # Load model
    model = VisitLevel180sModel()
    model.load("data/models/model_180s_visit_level.json")

    return matches, visits, stats, players, model


def get_match_predictions_and_actuals(matches, visits, stats, model):
    """Generate predictions and get actuals for all matches."""
    # Get actual 180s from stats
    stat_180s = stats[stats["stat_field_id"] == 2].copy()
    if stat_180s.empty:
        stat_180s = stats[stats["stat_field_name"] == "Thrown 180"].copy()

    if not stat_180s.empty:
        stat_180s["value_num"] = pd.to_numeric(stat_180s["value"], errors="coerce").fillna(0)
        match_180s = stat_180s.groupby("match_id")["value_num"].sum().to_dict()
    else:
        match_180s = {}

    # Fallback to visit counts
    if not visits.empty:
        visit_180s = visits.groupby("match_id")["is_180"].sum().to_dict()
    else:
        visit_180s = {}

    results = []
    for _, match in matches.iterrows():
        match_id = match["match_id"]
        actual = match_180s.get(match_id, visit_180s.get(match_id))

        if actual is None:
            continue

        home_id = match["home_player_id"]
        away_id = match["away_player_id"]
        league_id = match.get("league_id", 2)
        best_of_legs = match.get("best_of_legs", 5)
        best_of_sets = match.get("best_of_sets", 0)
        is_set_format = match.get("is_set_format", False)

        format_params = {
            "best_of_sets": best_of_sets,
            "best_of_legs": best_of_legs,
            "is_set_format": is_set_format,
        }

        pred = model.predict(home_id, away_id, league_id, format_params)

        results.append({
            "match_id": match_id,
            "home_player_id": home_id,
            "away_player_id": away_id,
            "league_id": league_id,
            "best_of_legs": best_of_legs,
            "best_of_sets": best_of_sets,
            "is_set_format": is_set_format,
            "predicted": pred.lambda_total,
            "predicted_lower": pred.lambda_total_lower,
            "predicted_upper": pred.lambda_total_upper,
            "actual": actual,
            "home_rate": pred.home_rate,
            "away_rate": pred.away_rate,
            "expected_visits": pred.expected_visits,
        })

    return pd.DataFrame(results)


def analyze_residuals_vs_predicted(df):
    """Analyze residuals vs predicted values (heteroscedasticity check)."""
    print("\n" + "=" * 70)
    print("1. RESIDUALS VS PREDICTED (Heteroscedasticity Check)")
    print("=" * 70)

    df["residual"] = df["actual"] - df["predicted"]
    df["abs_residual"] = df["residual"].abs()

    # Bin by predicted value
    df["pred_bucket"] = pd.cut(df["predicted"], bins=[0, 2, 4, 6, 8, 10, 15, 100],
                                labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15+"])

    bucket_stats = df.groupby("pred_bucket", observed=True).agg({
        "residual": ["mean", "std", "count"],
        "abs_residual": "mean",
        "actual": "mean",
        "predicted": "mean",
    })
    bucket_stats.columns = ["bias", "std", "count", "mae", "mean_actual", "mean_pred"]

    print("\nResiduals by Predicted 180s Bucket:")
    print("-" * 70)
    print(f"{'Bucket':<10} {'Bias':>8} {'Std':>8} {'MAE':>8} {'m_act':>8} {'m_pred':>8} {'N':>8}")
    print("-" * 70)

    for bucket, row in bucket_stats.iterrows():
        if row["count"] >= 5:
            print(f"{bucket:<10} {row['bias']:>+8.2f} {row['std']:>8.2f} {row['mae']:>8.2f} "
                  f"{row['mean_actual']:>8.2f} {row['mean_pred']:>8.2f} {row['count']:>8.0f}")

    # Check if errors are worse for high-180 matches
    low_bucket = df[df["predicted"] <= 4]
    high_bucket = df[df["predicted"] > 8]

    if len(low_bucket) > 10 and len(high_bucket) > 10:
        print(f"\nComparison: Low (<=4) vs High (>8) predicted:")
        print(f"  Low:  MAE={low_bucket['abs_residual'].mean():.2f}, Bias={low_bucket['residual'].mean():+.2f}")
        print(f"  High: MAE={high_bucket['abs_residual'].mean():.2f}, Bias={high_bucket['residual'].mean():+.2f}")

        if high_bucket['abs_residual'].mean() > low_bucket['abs_residual'].mean() * 1.3:
            print("  [!]  High-180 matches have significantly worse errors (>30% higher MAE)")
        else:
            print("  [OK]  Errors scale reasonably with predicted values")


def analyze_bias_by_format(df):
    """Analyze bias by match format."""
    print("\n" + "=" * 70)
    print("2. BIAS BY FORMAT (Best-of-Legs)")
    print("=" * 70)

    df["residual"] = df["actual"] - df["predicted"]

    # Group by best_of_legs
    format_stats = df.groupby("best_of_legs").agg({
        "residual": ["mean", "std", "count"],
        "actual": "mean",
        "predicted": "mean",
    })
    format_stats.columns = ["bias", "std", "count", "mean_actual", "mean_pred"]
    format_stats = format_stats[format_stats["count"] >= 10].sort_index()

    print("\nBias by Best-of-Legs Format:")
    print("-" * 70)
    print(f"{'BO-Legs':<10} {'Bias':>8} {'Std':>8} {'u_act':>8} {'u_pred':>8} {'N':>8} {'Status'}")
    print("-" * 70)

    format_issues = []
    for bol, row in format_stats.iterrows():
        bias_pct = abs(row['bias'] / row['mean_actual']) * 100 if row['mean_actual'] > 0 else 0
        status = "[!] >15%" if bias_pct > 15 else "[OK]"
        print(f"BO{bol:<8} {row['bias']:>+8.2f} {row['std']:>8.2f} "
              f"{row['mean_actual']:>8.2f} {row['mean_pred']:>8.2f} {row['count']:>8.0f} {status}")

        if bias_pct > 15:
            format_issues.append((bol, row['bias'], row['count']))

    if format_issues:
        print("\nFormats with >15% bias (may need format-specific adjustments):")
        for bol, bias, n in format_issues:
            print(f"  BO{bol}: bias={bias:+.2f} (n={n})")

    # Check set format vs leg format
    set_matches = df[df["is_set_format"] == True]
    leg_matches = df[df["is_set_format"] == False]

    if len(set_matches) > 20 and len(leg_matches) > 20:
        print(f"\nSet Format vs Leg Format:")
        print(f"  Set format: Bias={set_matches['residual'].mean():+.2f}, MAE={set_matches['residual'].abs().mean():.2f} (n={len(set_matches)})")
        print(f"  Leg format: Bias={leg_matches['residual'].mean():+.2f}, MAE={leg_matches['residual'].abs().mean():.2f} (n={len(leg_matches)})")


def analyze_bias_by_player_quality(df, model):
    """Analyze bias by player quality tier."""
    print("\n" + "=" * 70)
    print("3. BIAS BY PLAYER QUALITY TIER")
    print("=" * 70)

    df["residual"] = df["actual"] - df["predicted"]

    # Get player rates from model
    player_rates = {}
    for pid, stats in model.player_stats.items():
        player_rates[pid] = stats.shrunk_rate

    # Calculate average player rate for each match
    def get_avg_rate(row):
        home_rate = player_rates.get(row["home_player_id"], model.population_rate)
        away_rate = player_rates.get(row["away_player_id"], model.population_rate)
        return (home_rate + away_rate) / 2

    df["avg_player_rate"] = df.apply(get_avg_rate, axis=1)

    # Bin by player quality (180 rate)
    rate_percentiles = df["avg_player_rate"].quantile([0.25, 0.5, 0.75])
    df["quality_tier"] = pd.cut(
        df["avg_player_rate"],
        bins=[0, rate_percentiles[0.25], rate_percentiles[0.5], rate_percentiles[0.75], 1.0],
        labels=["Low", "Mid-Low", "Mid-High", "High"]
    )

    tier_stats = df.groupby("quality_tier", observed=True).agg({
        "residual": ["mean", "std", "count"],
        "actual": "mean",
        "predicted": "mean",
        "avg_player_rate": "mean",
    })
    tier_stats.columns = ["bias", "std", "count", "mean_actual", "mean_pred", "avg_rate"]

    print("\nBias by Player Quality Tier (based on 180 rate):")
    print("-" * 80)
    print(f"{'Tier':<10} {'Avg Rate':>10} {'Bias':>8} {'Std':>8} {'u_act':>8} {'u_pred':>8} {'N':>8}")
    print("-" * 80)

    for tier, row in tier_stats.iterrows():
        print(f"{tier:<10} {row['avg_rate']*100:>9.2f}% {row['bias']:>+8.2f} {row['std']:>8.2f} "
              f"{row['mean_actual']:>8.2f} {row['mean_pred']:>8.2f} {row['count']:>8.0f}")

    # Check if bias varies significantly by quality
    if len(tier_stats) >= 2:
        max_bias = tier_stats["bias"].max()
        min_bias = tier_stats["bias"].min()
        if abs(max_bias - min_bias) > 1.0:
            print(f"\n[!]  Bias varies by {max_bias - min_bias:.2f} across quality tiers")
            print("   Consider quality-specific adjustments")
        else:
            print(f"\n[OK]  Bias is relatively stable across quality tiers (range: {max_bias - min_bias:.2f})")


def analyze_variance_fit(df, model):
    """Compare empirical variance vs predicted variance."""
    print("\n" + "=" * 70)
    print("4. DISTRIBUTIONAL FIT (Variance Check)")
    print("=" * 70)

    df["residual"] = df["actual"] - df["predicted"]
    df["squared_residual"] = df["residual"] ** 2

    # For Poisson, Var = Mean. For NB, Var = Mean + Mean²/theta
    # We use overdispersion parameter

    # Bin by predicted value
    df["pred_bucket"] = pd.cut(df["predicted"], bins=[0, 3, 5, 7, 10, 15, 100],
                                labels=["0-3", "3-5", "5-7", "7-10", "10-15", "15+"])

    bucket_stats = df.groupby("pred_bucket", observed=True).agg({
        "predicted": "mean",
        "actual": ["mean", "var"],
        "squared_residual": "mean",
        "residual": "count",
    })
    bucket_stats.columns = ["mean_pred", "mean_actual", "empirical_var", "mse", "count"]

    # Expected Poisson variance = mean
    bucket_stats["poisson_var"] = bucket_stats["mean_actual"]
    # Expected variance with overdispersion
    bucket_stats["expected_var"] = bucket_stats["mean_actual"] * model.overdispersion
    # Ratio
    bucket_stats["var_ratio"] = bucket_stats["empirical_var"] / bucket_stats["poisson_var"]

    print(f"\nModel overdispersion parameter: {model.overdispersion:.2f}")
    print(f"(Poisson = 1.0, higher = more variance than Poisson)")
    print()
    print("Variance by Predicted 180s Bucket:")
    print("-" * 80)
    print(f"{'Bucket':<10} {'u_pred':>8} {'Emp.Var':>10} {'Pois.Var':>10} {'Exp.Var':>10} {'Ratio':>8} {'N':>6}")
    print("-" * 80)

    underdispersed = []
    overdispersed = []

    for bucket, row in bucket_stats.iterrows():
        if row["count"] >= 10:
            ratio = row["var_ratio"]
            status = ""
            if ratio < 0.8:
                status = " (under)"
                underdispersed.append((bucket, ratio, row["count"]))
            elif ratio > model.overdispersion * 1.2:
                status = " (OVER)"
                overdispersed.append((bucket, ratio, row["count"]))

            print(f"{bucket:<10} {row['mean_pred']:>8.2f} {row['empirical_var']:>10.2f} "
                  f"{row['poisson_var']:>10.2f} {row['expected_var']:>10.2f} "
                  f"{ratio:>8.2f}{status} {row['count']:>6.0f}")

    if overdispersed:
        print("\n[!]  Buckets with higher-than-expected variance (model undercalling variance):")
        for bucket, ratio, n in overdispersed:
            print(f"   {bucket}: ratio={ratio:.2f} (expected ~{model.overdispersion:.2f}), n={n}")
        print("   Consider increasing overdispersion parameter")

    # Overall variance ratio
    overall_var = df["actual"].var()
    overall_mean = df["actual"].mean()
    overall_ratio = overall_var / overall_mean

    print(f"\nOverall: Empirical Var/Mean = {overall_ratio:.2f} (model uses {model.overdispersion:.2f})")
    if abs(overall_ratio - model.overdispersion) > 0.3:
        print(f"[!]  Consider adjusting overdispersion to ~{overall_ratio:.2f}")


def analyze_by_league(df):
    """Analyze bias by league."""
    print("\n" + "=" * 70)
    print("5. BIAS BY LEAGUE")
    print("=" * 70)

    df["residual"] = df["actual"] - df["predicted"]

    league_stats = df.groupby("league_id").agg({
        "residual": ["mean", "std", "count"],
        "actual": "mean",
        "predicted": "mean",
    })
    league_stats.columns = ["bias", "std", "count", "mean_actual", "mean_pred"]
    league_stats = league_stats[league_stats["count"] >= 10].sort_values("count", ascending=False)

    # League names
    league_names = {
        2: "PDC World Champ",
        3: "Premier League",
        4: "World Matchplay",
        5: "World Grand Prix",
        6: "Grand Slam",
        7: "UK Open",
        8: "PC Finals",
        35: "Players Champs",
        38: "MODUS Super Series",
    }

    print("\nBias by League:")
    print("-" * 80)
    print(f"{'League':<20} {'Bias':>8} {'Std':>8} {'u_act':>8} {'u_pred':>8} {'N':>8}")
    print("-" * 80)

    for league_id, row in league_stats.iterrows():
        name = league_names.get(league_id, f"League {league_id}")[:20]
        print(f"{name:<20} {row['bias']:>+8.2f} {row['std']:>8.2f} "
              f"{row['mean_actual']:>8.2f} {row['mean_pred']:>8.2f} {row['count']:>8.0f}")


def main():
    print("=" * 70)
    print("VISIT-LEVEL 180s MODEL DIAGNOSTICS")
    print("=" * 70)

    # Load data
    print("\nLoading data and model...")
    matches, visits, stats, players, model = load_data_and_model()
    print(f"Loaded {len(matches)} matches")
    print(f"  Global bias correction: {model.bias_correction:.3f}")
    if model.format_bias_corrections:
        print(f"  Format-specific corrections: {len(model.format_bias_corrections)} formats")
        for bol, adj in sorted(model.format_bias_corrections.items()):
            print(f"    BO{bol}: {adj:+.2f} (total: {model.bias_correction + adj:+.2f})")

    # Generate predictions
    print("Generating predictions for all matches...")
    df = get_match_predictions_and_actuals(matches, visits, stats, model)
    print(f"Got predictions for {len(df)} matches with actual 180s counts")

    # Run diagnostics
    analyze_residuals_vs_predicted(df)
    analyze_bias_by_format(df)
    analyze_bias_by_player_quality(df, model)
    analyze_variance_fit(df, model)
    analyze_by_league(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    residuals = df["actual"] - df["predicted"]
    print(f"\nOverall Metrics:")
    print(f"  N matches: {len(df)}")
    print(f"  Mean actual: {df['actual'].mean():.2f}")
    print(f"  Mean predicted: {df['predicted'].mean():.2f}")
    print(f"  Bias: {residuals.mean():+.2f}")
    print(f"  RMSE: {np.sqrt((residuals**2).mean()):.2f}")
    print(f"  MAE: {residuals.abs().mean():.2f}")

    ss_tot = ((df["actual"] - df["actual"].mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R²: {r2:.3f}")


if __name__ == "__main__":
    main()
