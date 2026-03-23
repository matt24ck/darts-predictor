"""
Builds structured context for the Claude Haiku system prompt.

Injects current ratings, track record, and upcoming match data
so the model can answer questions about players, value, and performance.
"""

from datetime import date
from typing import Dict, List, Optional


def build_chat_context(
    glicko_model,
    players_list: List[Dict],
    db,
    store=None,
) -> str:
    """
    Build a ~2000-token system prompt with live data context.

    Args:
        glicko_model: Loaded Glicko2System (or None)
        players_list: List of {player_id, name} dicts
        db: SqliteStore instance
        store: ParquetStore instance (optional, for match history)

    Returns:
        System prompt string
    """
    sections = [SYSTEM_PREAMBLE]

    # Top player ratings
    if glicko_model and hasattr(glicko_model, "ratings"):
        sections.append(_build_ratings_section(glicko_model, players_list))

    # Track record summary
    if db:
        sections.append(_build_track_record_section(db))

    # Today's matches / upcoming
    if db:
        sections.append(_build_upcoming_section(db))

    return "\n\n".join(sections)


SYSTEM_PREAMBLE = """You are a data lookup tool for a darts betting platform. You are NOT an analyst \
and must NOT provide your own opinions, insights, predictions, or advice.

STRICT RULES:
- ONLY report data that is explicitly provided below. Never infer, speculate, or extrapolate.
- If the user asks for something not in the data, say "I don't have that data."
- Do NOT say "I think", "I recommend", "you should", or offer any subjective assessment.
- Do NOT interpret what the numbers mean or whether a bet is good or bad.
- Do NOT provide betting advice, strategy tips, or risk commentary.
- Present numbers exactly as given. Do not round, adjust, or editorialize.
- Keep responses short and factual. Use tables or bullet points.

You have access to the following data sections below:
- Player Glicko-2 ratings (higher = stronger, RD = uncertainty, TV bonus = televised event adjustment)
- Model betting track record (wins, losses, P&L from automated bets)
- Today's match predictions with bookmaker odds and calculated edges"""


def _build_ratings_section(glicko_model, players_list: List[Dict]) -> str:
    """Build top 25 player ratings section."""
    name_lookup = {p["player_id"]: p["name"] for p in players_list}

    rated = []
    for pid, rating in glicko_model.ratings.items():
        name = name_lookup.get(pid, f"Player {pid}")
        rated.append((name, rating.rating, rating.rd, getattr(rating, "tv_bonus", 0)))

    # Sort by rating descending
    rated.sort(key=lambda x: x[1], reverse=True)
    top = rated[:25]

    lines = ["CURRENT TOP 25 PLAYER RATINGS (Glicko-2):"]
    for i, (name, rating, rd, tv) in enumerate(top, 1):
        tv_str = f", TV bonus +{tv:.0f}" if tv > 5 else ""
        lines.append(f"{i:2d}. {name}: {rating:.0f} (RD {rd:.0f}{tv_str})")

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
        return "UPCOMING MATCHES:\nNo matches scheduled for today."

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
