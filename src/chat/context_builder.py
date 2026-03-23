"""
Builds structured context for the Claude Haiku system prompt.

Injects current ratings, 180s stats, track record, and upcoming match data
so the model can answer questions about players, value, and performance.
"""

from datetime import date
from typing import Dict, List, Optional


def build_chat_context(
    glicko_model,
    players_list: List[Dict],
    db,
    store=None,
    model_180s=None,
) -> str:
    """
    Build system prompt with live data context.

    Args:
        glicko_model: Loaded Glicko2System (or None)
        players_list: List of {player_id, name} dicts
        db: SqliteStore instance
        store: ParquetStore instance (optional, for match history)
        model_180s: Loaded VisitLevel180sModel (or None)

    Returns:
        System prompt string
    """
    sections = [SYSTEM_PREAMBLE]

    name_lookup = {p["player_id"]: p["name"] for p in players_list}

    # Top player ratings
    if glicko_model and hasattr(glicko_model, "ratings"):
        sections.append(_build_ratings_section(glicko_model, name_lookup))

    # 180s stats
    if model_180s and hasattr(model_180s, "player_stats"):
        sections.append(_build_180s_section(model_180s, name_lookup))

    # Track record summary
    if db:
        sections.append(_build_track_record_section(db))

    # Today's matches / upcoming
    if db:
        sections.append(_build_upcoming_section(db))

    return "\n\n".join(sections)


SYSTEM_PREAMBLE = """You are a helpful assistant for a darts betting platform. You have access to \
live data from a Glicko-2 rating system, a visit-level 180s prediction model, and a betting track record.

Your role:
- Answer questions about player ratings, 180s rates, match predictions, and the model's track record.
- Present data clearly using tables or bullet points when appropriate.
- You may explain what the data means (e.g. "a higher rating means a stronger player") but do not \
give personal betting advice or tell users what to bet on.
- If asked about a player or stat not in the data below, say you don't have that information.
- Be concise.

Key concepts:
- Glicko-2 ratings: higher = stronger. RD (rating deviation) = uncertainty. TV bonus = adjustment for televised events.
- 180s rate: probability of throwing a 180 per 3-dart visit. Population average is ~4.7%.
- Edge: model probability minus bookmaker implied probability. Positive edge = potential value.
- Kelly criterion: the model uses quarter-Kelly (25%) with a 3% minimum edge threshold for auto-bets."""


def _build_ratings_section(glicko_model, name_lookup: Dict[int, str]) -> str:
    """Build top 30 player ratings section."""
    rated = []
    for pid, rating in glicko_model.ratings.items():
        name = name_lookup.get(pid, f"Player {pid}")
        rated.append((name, pid, rating.rating, rating.rd, getattr(rating, "tv_bonus", 0)))

    rated.sort(key=lambda x: x[2], reverse=True)
    top = rated[:30]

    lines = ["CURRENT TOP 30 PLAYER RATINGS (Glicko-2):"]
    lines.append("Rank | Player | Rating | RD | TV Bonus")
    for i, (name, pid, rating, rd, tv) in enumerate(top, 1):
        tv_str = f"+{tv:.0f}" if tv > 5 else "-"
        lines.append(f"{i:2d}. {name}: {rating:.0f} (RD {rd:.0f}, TV {tv_str})")

    return "\n".join(lines)


def _build_180s_section(model_180s, name_lookup: Dict[int, str]) -> str:
    """Build top 30 players by 180s rate."""
    stats = model_180s.player_stats
    if not stats:
        return ""

    rated = []
    for pid, pstats in stats.items():
        pid = int(pid) if isinstance(pid, str) else pid
        name = name_lookup.get(pid, f"Player {pid}")
        shrunk_rate = getattr(pstats, "shrunk_rate", 0) if not isinstance(pstats, dict) else pstats.get("shrunk_rate", 0)
        total_visits = getattr(pstats, "total_visits", 0) if not isinstance(pstats, dict) else pstats.get("total_visits", 0)
        total_180s = getattr(pstats, "total_180s", 0) if not isinstance(pstats, dict) else pstats.get("total_180s", 0)
        rated.append((name, shrunk_rate, total_visits, total_180s))

    rated.sort(key=lambda x: x[1], reverse=True)
    top = rated[:30]

    pop_rate = getattr(model_180s, "population_rate", 0.047)

    lines = [
        f"180s MODEL — TOP 30 PLAYERS BY 180 RATE (population average: {pop_rate*100:.1f}%):",
        "Player | 180 Rate (per visit) | Total Visits | Total 180s",
    ]
    for name, rate, visits, count in top:
        lines.append(f"  {name}: {rate*100:.2f}% ({visits} visits, {count} 180s)")

    # Overdispersion info
    od = getattr(model_180s, "overdispersion", None)
    od_by_fmt = getattr(model_180s, "overdispersion_by_format", {})
    if od or od_by_fmt:
        lines.append(f"\nOverdispersion: leg format={od_by_fmt.get('leg', od)}, set format={od_by_fmt.get('set', od)}")
        lines.append("(Overdispersion > 1 means more variance than Poisson — the model uses Negative Binomial)")

    return "\n".join(lines)


def _build_track_record_section(db) -> str:
    """Build model track record summary."""
    stats = db.get_track_record_stats(user_id=1)  # System model user

    if stats["total_bets"] == 0:
        return "MODEL TRACK RECORD:\nNo bets placed yet — the model is building its track record."

    lines = [
        "MODEL TRACK RECORD (all auto-bets):",
        f"Total bets: {stats['total_bets']}",
        f"Record: {stats['wins']}W - {stats['losses']}L ({stats['win_rate']*100:.1f}% win rate)",
        f"ROI: {stats['roi']:+.1f}%",
        f"Cumulative P&L: {stats['total_pnl']:+.2f} units",
        f"Average edge at bet time: {stats['avg_edge']*100:.1f}%",
    ]

    # Recent P&L history
    history = db.get_pnl_history(user_id=1)
    if history:
        recent = history[-10:]
        lines.append("\nLast 10 settled bets:")
        for h in recent:
            result_icon = "W" if h["result"] == "win" else "L"
            lines.append(
                f"  {result_icon} {h['match']}: {h['pnl']:+.2f} (cumulative: {h['cumulative_pnl']:+.2f})"
            )

    return "\n".join(lines)


def _build_upcoming_section(db) -> str:
    """Build upcoming matches with predictions and odds."""
    today = date.today().isoformat()
    matches = db.get_matches_for_date(today)

    if not matches:
        return "TODAY'S MATCHES:\nNo matches scheduled for today."

    lines = [f"TODAY'S MATCHES ({today}):"]

    for m in matches:
        line = f"- {m['home_player_name']} vs {m['away_player_name']}"
        if m.get("league_name"):
            line += f" ({m['league_name']})"

        # Add predictions
        for p in m.get("predictions", []):
            if p["prediction_type"] == "match_winner":
                line += f"\n  {p['selection'].title()}: {p['model_probability']*100:.1f}%"

                # Add best odds and edge
                odds_data = m.get("best_odds", {}).get(p["selection"], {})
                if odds_data.get("best_odds"):
                    implied = 1.0 / odds_data["best_odds"]
                    edge = p["model_probability"] - implied
                    line += (
                        f" | Best odds: {odds_data['best_odds']:.2f} ({odds_data.get('bookmaker', '')})"
                        f" | Edge: {edge*100:+.1f}%"
                    )
            elif p["prediction_type"] == "total_180s_expected":
                ev = p.get("model_expected_value")
                if ev:
                    ci = ""
                    if p.get("confidence_lower") and p.get("confidence_upper"):
                        ci = f" (95% CI: {p['confidence_lower']:.1f} - {p['confidence_upper']:.1f})"
                    line += f"\n  Expected 180s: {ev:.1f}{ci}"

        # Add auto-bet info
        for b in m.get("auto_bets", []):
            stake = b["actual_stake"] or b["recommended_stake"]
            if b["result"] == "pending":
                line += f"\n  BET: {stake:.2f}u @ {b['decimal_odds']:.2f} (pending)"
            elif b["result"] == "win":
                line += f"\n  BET: WON +{b['pnl']:.2f}u @ {b['decimal_odds']:.2f}"
            elif b["result"] == "loss":
                line += f"\n  BET: LOST {b['pnl']:.2f}u @ {b['decimal_odds']:.2f}"

        lines.append(line)

    return "\n".join(lines)
