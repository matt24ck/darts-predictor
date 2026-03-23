"""
SQLite Store for transactional data (users, bets, odds, chat).

Coexists alongside ParquetStore which handles analytics data.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SqliteStore:
    """SQLite-backed store for betting, auth, and chat data."""

    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
        finally:
            conn.close()

    # =========================================================================
    # Users
    # =========================================================================

    def create_user(self, email: str, password_hash: str, display_name: Optional[str] = None) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)",
                (email, password_hash, display_name),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_user_settings(self, user_id: int, bankroll: float, kelly_fraction: float):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET bankroll = ?, kelly_fraction = ? WHERE id = ?",
                (bankroll, kelly_fraction, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Upcoming Matches
    # =========================================================================

    def upsert_upcoming_match(self, **kwargs) -> int:
        conn = self._get_conn()
        try:
            # Check if match exists by external_match_id or player pair + date
            existing = None
            if kwargs.get("external_match_id"):
                existing = conn.execute(
                    "SELECT id FROM upcoming_matches WHERE external_match_id = ?",
                    (kwargs["external_match_id"],),
                ).fetchone()

            if not existing:
                existing = conn.execute(
                    """SELECT id FROM upcoming_matches
                       WHERE home_player_id = ? AND away_player_id = ? AND match_date = ?""",
                    (kwargs["home_player_id"], kwargs["away_player_id"], kwargs["match_date"]),
                ).fetchone()

            if existing:
                match_id = existing["id"]
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                conn.execute(
                    f"UPDATE upcoming_matches SET {sets}, updated_at = datetime('now') WHERE id = ?",
                    (*kwargs.values(), match_id),
                )
                conn.commit()
                return match_id
            else:
                cols = ", ".join(kwargs.keys())
                placeholders = ", ".join("?" for _ in kwargs)
                cursor = conn.execute(
                    f"INSERT INTO upcoming_matches ({cols}) VALUES ({placeholders})",
                    tuple(kwargs.values()),
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_upcoming_matches(self, status: str = "scheduled") -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM upcoming_matches WHERE status = ? ORDER BY match_date ASC",
                (status,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_upcoming_match(self, match_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM upcoming_matches WHERE id = ?", (match_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_match_status(self, match_id: int, status: str, home_score: str = None, away_score: str = None):
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE upcoming_matches
                   SET status = ?, home_score = ?, away_score = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (status, home_score, away_score, match_id),
            )
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Predictions
    # =========================================================================

    def create_prediction(self, upcoming_match_id: int, prediction_type: str,
                          selection: str, model_probability: float,
                          market_description: str = None,
                          model_expected_value: float = None,
                          confidence_lower: float = None,
                          confidence_upper: float = None) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO predictions
                   (upcoming_match_id, prediction_type, selection, model_probability,
                    market_description, model_expected_value, confidence_lower, confidence_upper)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (upcoming_match_id, prediction_type, selection, model_probability,
                 market_description, model_expected_value, confidence_lower, confidence_upper),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_predictions_for_match(self, upcoming_match_id: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE upcoming_match_id = ? ORDER BY created_at",
                (upcoming_match_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # =========================================================================
    # Odds Snapshots
    # =========================================================================

    def save_odds_snapshot(self, upcoming_match_id: int, bookmaker: str,
                           market_type: str, selection: str,
                           decimal_odds: float, prediction_id: int = None,
                           edge: float = None) -> int:
        implied = 1.0 / decimal_odds
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO odds_snapshots
                   (upcoming_match_id, prediction_id, bookmaker, market_type,
                    selection, decimal_odds, implied_probability, edge)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (upcoming_match_id, prediction_id, bookmaker, market_type,
                 selection, decimal_odds, implied, edge),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_odds_for_match(self, upcoming_match_id: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM odds_snapshots
                   WHERE upcoming_match_id = ?
                   ORDER BY fetched_at DESC""",
                (upcoming_match_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_best_odds(self, upcoming_match_id: int, selection: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM odds_snapshots
                   WHERE upcoming_match_id = ? AND selection = ?
                   ORDER BY decimal_odds DESC LIMIT 1""",
                (upcoming_match_id, selection),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # =========================================================================
    # Bets
    # =========================================================================

    def create_bet(self, user_id: int, prediction_id: int, decimal_odds: float,
                   model_probability: float, edge: float, kelly_fraction: float,
                   recommended_stake: float, bankroll_at_time: float,
                   bookmaker: str = None, odds_snapshot_id: int = None,
                   actual_stake: float = None) -> int:
        implied = 1.0 / decimal_odds
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO bets
                   (user_id, prediction_id, odds_snapshot_id, bookmaker, decimal_odds,
                    implied_probability, model_probability, edge, kelly_fraction,
                    recommended_stake, actual_stake, bankroll_at_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, prediction_id, odds_snapshot_id, bookmaker, decimal_odds,
                 implied, model_probability, edge, kelly_fraction,
                 recommended_stake, actual_stake, bankroll_at_time),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def settle_bet(self, bet_id: int, result: str):
        """Settle a bet and calculate P&L. result: 'win', 'loss', or 'void'."""
        conn = self._get_conn()
        try:
            bet = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
            if not bet:
                return

            stake = bet["actual_stake"] or bet["recommended_stake"]
            if result == "win":
                pnl = stake * (bet["decimal_odds"] - 1)
            elif result == "loss":
                pnl = -stake
            else:  # void
                pnl = 0.0

            conn.execute(
                """UPDATE bets SET result = ?, pnl = ?, settled_at = datetime('now')
                   WHERE id = ?""",
                (result, pnl, bet_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user_bets(self, user_id: int, status: str = None, limit: int = 100) -> List[Dict]:
        conn = self._get_conn()
        try:
            query = """
                SELECT b.*, p.prediction_type, p.selection, p.market_description,
                       m.home_player_name, m.away_player_name, m.league_name, m.match_date
                FROM bets b
                JOIN predictions p ON b.prediction_id = p.id
                JOIN upcoming_matches m ON p.upcoming_match_id = m.id
                WHERE b.user_id = ?
            """
            params = [user_id]
            if status:
                query += " AND b.result = ?"
                params.append(status)
            query += " ORDER BY b.created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_track_record_stats(self, user_id: int = None) -> Dict:
        """Get aggregate betting statistics. If user_id is None, returns global stats."""
        conn = self._get_conn()
        try:
            where = "WHERE user_id = ?" if user_id else "WHERE 1=1"
            params = [user_id] if user_id else []

            settled = conn.execute(
                f"SELECT * FROM bets {where} AND result IN ('win', 'loss')", params
            ).fetchall()

            if not settled:
                return {
                    "total_bets": 0, "wins": 0, "losses": 0,
                    "win_rate": 0, "total_pnl": 0, "roi": 0, "avg_edge": 0,
                }

            wins = sum(1 for b in settled if b["result"] == "win")
            losses = sum(1 for b in settled if b["result"] == "loss")
            total_pnl = sum(b["pnl"] for b in settled)
            total_staked = sum(b["actual_stake"] or b["recommended_stake"] for b in settled)
            avg_edge = sum(b["edge"] for b in settled) / len(settled)

            return {
                "total_bets": len(settled),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(settled) if settled else 0,
                "total_pnl": round(total_pnl, 2),
                "roi": round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0,
                "avg_edge": round(avg_edge, 4),
            }
        finally:
            conn.close()

    def get_pnl_history(self, user_id: int = None) -> List[Dict]:
        """Get cumulative P&L over time for charting."""
        conn = self._get_conn()
        try:
            where = "WHERE b.user_id = ?" if user_id else "WHERE 1=1"
            params = [user_id] if user_id else []

            rows = conn.execute(
                f"""SELECT b.settled_at, b.pnl, b.result, p.market_description,
                           m.home_player_name, m.away_player_name
                    FROM bets b
                    JOIN predictions p ON b.prediction_id = p.id
                    JOIN upcoming_matches m ON p.upcoming_match_id = m.id
                    {where} AND b.result IN ('win', 'loss')
                    ORDER BY b.settled_at ASC""",
                params,
            ).fetchall()

            cumulative = 0
            history = []
            for r in rows:
                cumulative += r["pnl"]
                history.append({
                    "date": r["settled_at"],
                    "pnl": r["pnl"],
                    "cumulative_pnl": round(cumulative, 2),
                    "result": r["result"],
                    "description": r["market_description"],
                    "match": f"{r['home_player_name']} vs {r['away_player_name']}",
                })
            return history
        finally:
            conn.close()

    # =========================================================================
    # Chat Messages
    # =========================================================================

    def save_chat_message(self, user_id: int, session_id: str, role: str, content: str) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO chat_messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, session_id, role, content),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_chat_history(self, user_id: int, session_id: str, limit: int = 50) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM chat_messages
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY created_at ASC LIMIT ?""",
                (user_id, session_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # =========================================================================
    # Player Aliases
    # =========================================================================

    def save_player_alias(self, player_id: int, alias: str, source: str = "manual"):
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO player_aliases (player_id, alias, source) VALUES (?, ?, ?)",
                (player_id, alias, source),
            )
            conn.commit()
        finally:
            conn.close()

    def find_player_by_alias(self, name: str) -> Optional[int]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT player_id FROM player_aliases WHERE alias = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            return row["player_id"] if row else None
        finally:
            conn.close()

    def get_matches_for_date(self, match_date: str) -> List[Dict]:
        """Get all matches for a date with predictions, best odds, and bets."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM upcoming_matches WHERE match_date = ? ORDER BY league_name, id",
                (match_date,),
            ).fetchall()
            matches = []
            for m in rows:
                match = dict(m)
                match_id = m["id"]

                # Get predictions
                preds = conn.execute(
                    "SELECT * FROM predictions WHERE upcoming_match_id = ?", (match_id,)
                ).fetchall()
                match["predictions"] = [dict(p) for p in preds]

                # Get best odds per selection
                odds = conn.execute(
                    """SELECT selection, MAX(decimal_odds) as best_odds,
                              bookmaker, implied_probability, edge
                       FROM odds_snapshots
                       WHERE upcoming_match_id = ?
                       GROUP BY selection""",
                    (match_id,),
                ).fetchall()
                match["best_odds"] = {o["selection"]: dict(o) for o in odds}

                # Get all odds for comparison
                all_odds = conn.execute(
                    "SELECT * FROM odds_snapshots WHERE upcoming_match_id = ? ORDER BY selection, decimal_odds DESC",
                    (match_id,),
                ).fetchall()
                match["all_odds"] = [dict(o) for o in all_odds]

                # Get model auto-bets (user_id=1)
                bets = conn.execute(
                    """SELECT * FROM bets WHERE user_id = 1
                       AND prediction_id IN (SELECT id FROM predictions WHERE upcoming_match_id = ?)""",
                    (match_id,),
                ).fetchall()
                match["auto_bets"] = [dict(b) for b in bets]

                matches.append(match)
            return matches
        finally:
            conn.close()

    def get_daily_summary(self, match_date: str) -> Dict:
        """Get summary stats for a single day's bets (model user only)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT b.* FROM bets b
                   JOIN predictions p ON b.prediction_id = p.id
                   JOIN upcoming_matches m ON p.upcoming_match_id = m.id
                   WHERE b.user_id = 1 AND m.match_date = ?""",
                (match_date,),
            ).fetchall()

            if not rows:
                return {"total_bets": 0, "value_bets": 0, "total_staked": 0,
                        "total_pnl": 0, "settled": 0, "pending": 0,
                        "wins": 0, "losses": 0}

            settled = [r for r in rows if r["result"] in ("win", "loss")]
            pending = [r for r in rows if r["result"] == "pending"]

            return {
                "total_bets": len(rows),
                "value_bets": len(rows),
                "total_staked": round(sum(r["actual_stake"] or r["recommended_stake"] for r in rows), 2),
                "total_pnl": round(sum(r["pnl"] or 0 for r in settled), 2),
                "settled": len(settled),
                "pending": len(pending),
                "wins": sum(1 for r in settled if r["result"] == "win"),
                "losses": sum(1 for r in settled if r["result"] == "loss"),
            }
        finally:
            conn.close()

    def mark_match_predictions_generated(self, match_id: int):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE upcoming_matches SET predictions_generated = 1, updated_at = datetime('now') WHERE id = ?",
                (match_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_match_odds_fetched(self, match_id: int):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE upcoming_matches SET odds_fetched = 1, updated_at = datetime('now') WHERE id = ?",
                (match_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_aliases(self) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM player_aliases").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# =============================================================================
# Schema
# =============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    bankroll REAL DEFAULT 100.0,
    kelly_fraction REAL DEFAULT 0.25,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS upcoming_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_match_id TEXT,
    home_player_id INTEGER NOT NULL,
    away_player_id INTEGER NOT NULL,
    home_player_name TEXT NOT NULL,
    away_player_name TEXT NOT NULL,
    league_id INTEGER,
    league_name TEXT,
    match_date TEXT NOT NULL,
    best_of_sets INTEGER DEFAULT 0,
    best_of_legs INTEGER DEFAULT 5,
    is_set_format INTEGER DEFAULT 0,
    status TEXT DEFAULT 'scheduled',
    home_score TEXT,
    away_score TEXT,
    predictions_generated INTEGER DEFAULT 0,
    odds_fetched INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upcoming_match_id INTEGER NOT NULL REFERENCES upcoming_matches(id),
    prediction_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    market_description TEXT,
    model_probability REAL NOT NULL,
    model_expected_value REAL,
    confidence_lower REAL,
    confidence_upper REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upcoming_match_id INTEGER NOT NULL REFERENCES upcoming_matches(id),
    prediction_id INTEGER REFERENCES predictions(id),
    bookmaker TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    implied_probability REAL NOT NULL,
    edge REAL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    odds_snapshot_id INTEGER REFERENCES odds_snapshots(id),
    bookmaker TEXT,
    decimal_odds REAL NOT NULL,
    implied_probability REAL NOT NULL,
    model_probability REAL NOT NULL,
    edge REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    recommended_stake REAL NOT NULL,
    actual_stake REAL,
    bankroll_at_time REAL,
    result TEXT DEFAULT 'pending',
    pnl REAL,
    settled_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS player_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    source TEXT,
    UNIQUE(player_id, alias)
);

CREATE INDEX IF NOT EXISTS idx_bets_user ON bets(user_id);
CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(upcoming_match_id);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(upcoming_match_id);
CREATE INDEX IF NOT EXISTS idx_chat_user_session ON chat_messages(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_status ON upcoming_matches(status);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON player_aliases(alias COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_upcoming_date ON upcoming_matches(match_date);

-- System user for auto-bets (id=1)
INSERT OR IGNORE INTO users (id, email, password_hash, display_name, bankroll, kelly_fraction)
VALUES (1, 'model@system', 'not-a-real-password', 'Model', 100.0, 0.25);
"""
