"""
Darts Value Betting Web Application

Flask app for predicting match outcomes, 180s distributions,
and finding value bets with Kelly criterion staking.

Optimal Models:
- Win Probability: Glicko-2 (tuned hyperparameters, calibrated)
- 180s Prediction: VisitLevel180sModel (Bernoulli per visit, empirical Bayes shrinkage)
"""

import os
import uuid
from datetime import date
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_login import LoginManager, current_user, login_required
from dotenv import load_dotenv
import json
from pathlib import Path
import numpy as np

from src.models import Glicko2System, VisitLevel180sModel
from src.storage import ParquetStore
from src.storage.sqlite_store import SqliteStore
from src.auth.models import User
from src.auth.routes import auth_bp, init_auth
from src.betting.kelly import kelly_analysis
from src.betting.tracker import BetTracker
from src.betting.odds import OddsClient
from src.betting.scheduler import DailyScheduler
from src.chat.context_builder import build_chat_context
from src.chat.haiku_client import stream_chat_response

load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Production settings
if os.environ.get("RAILWAY_ENVIRONMENT") or not app.debug:
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"

# Global storage for models and data
models = {}
store = None
db = None
tracker = None
scheduler = None
players_list = []
leagues_list = []


@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(int(user_id))
    return User.from_dict(user_data)


def load_models():
    """Load optimal models for prediction."""
    global models, store, db, tracker, scheduler, players_list, leagues_list

    # Load data store (parquet for analytics)
    store = ParquetStore("data/processed")

    # Load SQLite store (transactional data)
    db = SqliteStore("data/app.db")
    tracker = BetTracker(db)
    init_auth(db)

    # Load players
    players_df = store.get_players()
    players_list = players_df.sort_values('name')[['player_id', 'name']].to_dict('records')

    # Load leagues
    leagues_df = store.get_leagues()
    leagues_df = leagues_df.rename(columns={'league_name': 'name'})
    leagues_list = leagues_df.sort_values('name')[['league_id', 'name']].to_dict('records')

    # Load Glicko-2 (PRIMARY win probability model)
    try:
        glicko = Glicko2System()
        glicko.load("data/models/glicko2_system.json")
        models['glicko'] = glicko
        print("  Loaded Glicko-2 (primary win probability model)")
    except Exception as e:
        print(f"  Warning: Could not load Glicko-2: {e}")
        models['glicko'] = None

    # Load VisitLevel180sModel (PRIMARY 180s model - visit-level Bernoulli)
    try:
        model_180s = VisitLevel180sModel()
        model_180s.load("data/models/model_180s_visit_level.json")
        models['180s'] = model_180s
        print("  Loaded Visit-Level 180s model (primary)")
    except Exception as e:
        print(f"  Warning: Could not load 180s model: {e}")
        models['180s'] = None

    # Load MODUS-specific 180s model (used for league_id=38)
    try:
        model_modus = VisitLevel180sModel()
        model_modus.load("data/models/model_180s_modus.json")
        models['180s_modus'] = model_modus
        print("  Loaded MODUS-specific 180s model")
    except Exception:
        models['180s_modus'] = None

    # Initialize scheduler
    odds_api_key = os.environ.get("ODDS_API_KEY", "")
    odds_client = OddsClient(api_key=odds_api_key)
    scheduler = DailyScheduler(
        db=db, tracker=tracker, odds_client=odds_client,
        models=models, store=store, players_list=players_list,
    )


# Register blueprints
app.register_blueprint(auth_bp)


# =========================================================================
# Error Handlers
# =========================================================================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("base.html", content="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("base.html", content="Something went wrong."), 500


@app.route('/robots.txt')
def robots():
    return Response("User-agent: *\nAllow: /\nSitemap: /sitemap.xml\n", mimetype="text/plain")


@app.route('/sitemap.xml')
def sitemap():
    pages = ["/", "/tips", "/track-record", "/predict", "/chat"]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    base = request.host_url.rstrip("/")
    for page in pages:
        xml += f"  <url><loc>{base}{page}</loc></url>\n"
    xml += "</urlset>"
    return Response(xml, mimetype="application/xml")


# =========================================================================
# Page Routes
# =========================================================================

@app.route('/')
def dashboard():
    """Dashboard — upcoming matches and track record summary."""
    stats = db.get_track_record_stats() if db else {
        "total_bets": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_pnl": 0, "roi": 0, "avg_edge": 0,
    }
    upcoming = db.get_upcoming_matches() if db else []
    return render_template('dashboard.html', stats=stats, upcoming=upcoming)


@app.route('/predict')
def predict_page():
    """Match prediction tool."""
    return render_template('predict.html', players=players_list, leagues=leagues_list)


@app.route('/track-record')
def track_record():
    """Public track record page."""
    stats = db.get_track_record_stats() if db else {
        "total_bets": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_pnl": 0, "roi": 0, "avg_edge": 0,
    }
    pnl_history = db.get_pnl_history() if db else []

    # Get settled bets (public view)
    bets = []
    pending_bets = []
    if db:
        conn = db._get_conn()
        try:
            rows = conn.execute("""
                SELECT b.*, p.prediction_type, p.selection, p.market_description,
                       m.home_player_name, m.away_player_name, m.league_name, m.match_date
                FROM bets b
                JOIN predictions p ON b.prediction_id = p.id
                JOIN upcoming_matches m ON p.upcoming_match_id = m.id
                WHERE b.result IN ('win', 'loss')
                ORDER BY b.settled_at DESC LIMIT 100
            """).fetchall()
            bets = [dict(r) for r in rows]

            # Pending bets for the current logged-in user
            if current_user.is_authenticated:
                prows = conn.execute("""
                    SELECT b.*, p.prediction_type, p.selection, p.market_description,
                           m.home_player_name, m.away_player_name, m.league_name, m.match_date
                    FROM bets b
                    JOIN predictions p ON b.prediction_id = p.id
                    JOIN upcoming_matches m ON p.upcoming_match_id = m.id
                    WHERE b.user_id = ? AND b.result = 'pending'
                    ORDER BY b.created_at DESC
                """, (current_user.id,)).fetchall()
                pending_bets = [dict(r) for r in prows]
        finally:
            conn.close()

    return render_template('track_record.html', stats=stats, pnl_history=pnl_history,
                           bets=bets, pending_bets=pending_bets)


@app.route('/chat')
def chat_page():
    """AI chat assistant."""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    """Stream a chat response from Claude Haiku with darts context."""
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Build system context with live data
    system_prompt = build_chat_context(
        glicko_model=models.get("glicko"),
        players_list=players_list,
        db=db,
        store=store,
        model_180s=models.get("180s"),
    )

    # Save user message
    db.save_chat_message(current_user.id, session_id, "user", user_message)

    # Build message history from this session (last 20 messages for context)
    history = db.get_chat_history(current_user.id, session_id, limit=20)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    # Stream response via SSE
    def generate():
        full_response = []
        for chunk in stream_chat_response(api_key, system_prompt, messages):
            full_response.append(chunk)
            # SSE format
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Save assistant response
        assistant_text = "".join(full_response)
        db.save_chat_message(current_user.id, session_id, "assistant", assistant_text)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route('/tips')
@app.route('/tips/<match_date>')
def tips_page(match_date=None):
    """Daily tips page — every match for a given day with predictions, odds, edges, P&L."""
    if not match_date:
        match_date = date.today().isoformat()

    matches = db.get_matches_for_date(match_date) if db else []
    daily_summary = db.get_daily_summary(match_date) if db else {}
    global_stats = db.get_track_record_stats(user_id=1) if db else {}

    return render_template('tips.html',
                           matches=matches,
                           match_date=match_date,
                           daily_summary=daily_summary,
                           global_stats=global_stats)


# =========================================================================
# Prediction API (existing, unchanged)
# =========================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for match predictions."""
    data = request.json

    home_player_id = int(data['home_player_id'])
    away_player_id = int(data['away_player_id'])
    best_of_sets = int(data.get('best_of_sets', 0))
    best_of_legs = int(data.get('best_of_legs', 5))
    is_set_format = data.get('is_set_format', False)
    league_id = int(data.get('league_id', 2))

    # Get player names
    home_name = next((p['name'] for p in players_list if p['player_id'] == home_player_id), f"Player {home_player_id}")
    away_name = next((p['name'] for p in players_list if p['player_id'] == away_player_id), f"Player {away_player_id}")

    result = {
        'home_player': home_name,
        'away_player': away_name,
        'win_probability': None,
        'ratings': {},
        '180s': None,
        'model_info': {
            'win_model': None,
            '180s_model': None,
        }
    }

    # WIN PROBABILITY: Glicko-2
    if models.get('glicko'):
        home_player = models['glicko'].ratings.get(home_player_id)
        away_player = models['glicko'].ratings.get(away_player_id)

        if home_player and away_player:
            p_home, p_away, metadata = models['glicko'].calibrated_win_probability(home_player_id, away_player_id)

            result['win_probability'] = {
                'home_win': round(p_home, 4),
                'away_win': round(p_away, 4),
                'is_calibrated': metadata.get('is_calibrated', False),
                'raw_home_win': round(metadata.get('raw_p_a', p_home), 4),
                'confidence_interval': {
                    'home_lower': round(metadata.get('p_a_lower', 0), 4),
                    'home_upper': round(metadata.get('p_a_upper', 1), 4),
                },
                'combined_uncertainty': round(metadata.get('combined_uncertainty', 0), 1),
            }
            result['ratings']['glicko'] = {
                'home': round(home_player.rating, 1),
                'away': round(away_player.rating, 1),
                'home_rd': round(home_player.rd, 1),
                'away_rd': round(away_player.rd, 1),
                'home_tv_bonus': round(getattr(home_player, 'tv_bonus', 0.0), 1),
                'away_tv_bonus': round(getattr(away_player, 'tv_bonus', 0.0), 1),
            }
            result['model_info']['win_model'] = 'glicko2'

    # 180s PREDICTION: Visit-Level Model (MODUS-specific if league_id=38)
    model_180s = models.get('180s')
    if league_id == 38 and models.get('180s_modus'):
        model_180s = models['180s_modus']

    if model_180s:
        format_params = {
            'best_of_sets': best_of_sets,
            'best_of_legs': best_of_legs,
            'is_set_format': is_set_format,
        }

        pred_180s = model_180s.predict(
            home_player_id=home_player_id,
            away_player_id=away_player_id,
            league_id=league_id,
            format_params=format_params,
        )

        result['180s'] = {
            'expected_total': round(pred_180s.lambda_total, 2),
            'expected_home': round(pred_180s.lambda_home, 2),
            'expected_away': round(pred_180s.lambda_away, 2),
            'raw_prediction': round(pred_180s.lambda_total, 2),
            'confidence_interval': {
                'lower': round(pred_180s.lambda_total_lower, 2),
                'upper': round(pred_180s.lambda_total_upper, 2),
            },
            'home_rate': round(pred_180s.home_rate, 4),
            'away_rate': round(pred_180s.away_rate, 4),
            'expected_visits': round(pred_180s.expected_visits, 1),
            'league_multiplier': round(pred_180s.league_multiplier, 3),
        }
        result['model_info']['180s_model'] = 'visit_level'

    return jsonify(result)


@app.route('/api/180s_probability', methods=['POST'])
def compute_180s_probability():
    """Compute P(180s >= target) using Negative Binomial."""
    from src.models.predictions import _nb_cdf

    data = request.json

    lambda_home = float(data['lambda_home'])
    lambda_away = float(data['lambda_away'])
    target = int(data['target'])
    phi = float(data.get('overdispersion', 0))
    if phi <= 0 and models.get('180s'):
        fmt = {"is_set_format": data.get("is_set_format", False)}
        phi = models['180s'].get_overdispersion(fmt)

    lambda_total = lambda_home + lambda_away

    if target <= 0:
        prob = 1.0
        prob_home = 1.0
        prob_away = 1.0
    else:
        prob = 1.0 - _nb_cdf(target - 1, lambda_total, phi)
        prob_home = 1.0 - _nb_cdf(target - 1, lambda_home, phi)
        prob_away = 1.0 - _nb_cdf(target - 1, lambda_away, phi)

    return jsonify({
        'target': target,
        'lambda_total': round(lambda_total, 2),
        'overdispersion': round(phi, 2),
        'prob_total_ge': round(prob, 4),
        'prob_home_ge': round(prob_home, 4),
        'prob_away_ge': round(prob_away, 4),
    })


@app.route('/api/180s_matrix', methods=['POST'])
def compute_180s_matrix():
    """Compute full 180s probability matrix using Negative Binomial."""
    from src.models.predictions import _nb_cdf

    data = request.json

    lambda_home = float(data['lambda_home'])
    lambda_away = float(data['lambda_away'])
    max_180s = int(data.get('max_180s', 20))
    phi = float(data.get('overdispersion', 0))
    if phi <= 0 and models.get('180s'):
        fmt = {"is_set_format": data.get("is_set_format", False)}
        phi = models['180s'].get_overdispersion(fmt)

    lambda_total = lambda_home + lambda_away

    matrix = []
    for x in range(max_180s + 1):
        if x == 0:
            prob_home = 1.0
            prob_away = 1.0
            prob_total = 1.0
        else:
            prob_home = 1.0 - _nb_cdf(x - 1, lambda_home, phi)
            prob_away = 1.0 - _nb_cdf(x - 1, lambda_away, phi)
            prob_total = 1.0 - _nb_cdf(x - 1, lambda_total, phi)

        matrix.append({
            'x': x,
            'prob_home_ge': round(prob_home, 4),
            'prob_away_ge': round(prob_away, 4),
            'prob_total_ge': round(prob_total, 4),
        })

    return jsonify({
        'lambda_home': round(lambda_home, 2),
        'lambda_away': round(lambda_away, 2),
        'lambda_total': round(lambda_total, 2),
        'overdispersion': round(phi, 2),
        'matrix': matrix
    })


@app.route('/api/most_180s', methods=['POST'])
def compute_most_180s():
    """Compute P(home has most 180s), P(draw), P(away has most 180s) using NB."""
    from src.models.predictions import compute_most_180s_probabilities

    data = request.json
    lambda_home = float(data['lambda_home'])
    lambda_away = float(data['lambda_away'])
    phi = float(data.get('overdispersion', 0))
    if phi <= 0 and models.get('180s'):
        fmt = {"is_set_format": data.get("is_set_format", False)}
        phi = models['180s'].get_overdispersion(fmt)

    result = compute_most_180s_probabilities(lambda_home, lambda_away, overdispersion=phi)
    result['lambda_home'] = round(lambda_home, 2)
    result['lambda_away'] = round(lambda_away, 2)
    result['overdispersion'] = round(phi, 2)

    return jsonify(result)


# =========================================================================
# Betting API
# =========================================================================

@app.route('/api/kelly', methods=['POST'])
def calc_kelly():
    """Calculate Kelly criterion for given model probability and odds."""
    data = request.json
    model_prob = float(data['model_probability'])
    decimal_odds = float(data['decimal_odds'])
    bankroll = float(data.get('bankroll', 100.0))
    fraction = float(data.get('kelly_fraction', 0.25))
    min_edge = float(data.get('min_edge', 0.03))

    analysis = kelly_analysis(model_prob, decimal_odds, bankroll, fraction, min_edge)
    return jsonify(analysis)


@app.route('/api/bets/track', methods=['POST'])
@login_required
def track_bet():
    """
    Full flow: create match + prediction + bet from the predict page.

    Expects model prediction data + bookmaker odds.
    """
    data = request.json

    # 1. Ensure the match exists in upcoming_matches
    match_id = tracker.ensure_match(
        home_player_id=int(data['home_player_id']),
        away_player_id=int(data['away_player_id']),
        home_player_name=data['home_player_name'],
        away_player_name=data['away_player_name'],
        match_date=data.get('match_date', ''),
        league_id=data.get('league_id'),
        league_name=data.get('league_name'),
        best_of_sets=int(data.get('best_of_sets', 0)),
        best_of_legs=int(data.get('best_of_legs', 5)),
        is_set_format=data.get('is_set_format', False),
    )

    # 2. Create the prediction record
    selection = data['selection']  # 'home' or 'away'
    model_prob = float(data['model_probability'])
    ci = data.get('confidence_interval', {})

    if selection == 'home':
        desc = f"{data['home_player_name']} to win"
    else:
        desc = f"{data['away_player_name']} to win"

    prediction_id = db.create_prediction(
        upcoming_match_id=match_id,
        prediction_type="match_winner",
        selection=selection,
        model_probability=model_prob,
        market_description=desc,
        confidence_lower=ci.get('lower'),
        confidence_upper=ci.get('upper'),
    )

    # 3. Place the bet via tracker
    result = tracker.place_bet(
        user_id=current_user.id,
        prediction_id=prediction_id,
        decimal_odds=float(data['decimal_odds']),
        bankroll=current_user.bankroll,
        kelly_fraction=current_user.kelly_fraction,
        bookmaker=data.get('bookmaker'),
        actual_stake=float(data['actual_stake']) if data.get('actual_stake') else None,
    )

    if result:
        return jsonify({"status": "tracked", **result})
    else:
        return jsonify({"status": "error", "message": "Could not place bet"}), 400


@app.route('/api/bets/settle', methods=['POST'])
@login_required
def settle_bet():
    """Settle a bet with result."""
    data = request.json
    db.settle_bet(int(data['bet_id']), data['result'])
    return jsonify({"status": "settled"})


@app.route('/api/bets/settle-match', methods=['POST'])
@login_required
def settle_match():
    """Settle all bets for a match."""
    data = request.json
    tracker.settle_match(
        upcoming_match_id=int(data['match_id']),
        home_won=data['home_won'],
        home_score=data.get('home_score'),
        away_score=data.get('away_score'),
    )
    return jsonify({"status": "settled"})


@app.route('/api/bets/history')
@login_required
def bet_history():
    """Get current user's bet history."""
    bets = db.get_user_bets(current_user.id)
    return jsonify(bets)


@app.route('/api/bets/pending')
@login_required
def pending_bets():
    """Get current user's pending (unsettled) bets."""
    bets = db.get_user_bets(current_user.id, status='pending')
    return jsonify(bets)


@app.route('/api/track-record/stats')
def track_record_stats():
    """Get global track record statistics (public)."""
    stats = db.get_track_record_stats()
    return jsonify(stats)


@app.route('/api/track-record/pnl')
def track_record_pnl():
    """Get P&L history for charting (public)."""
    history = db.get_pnl_history()
    return jsonify(history)


# =========================================================================
# Admin / Pipeline API
# =========================================================================

@app.route('/api/admin/run-pipeline', methods=['POST'])
@login_required
def run_pipeline():
    """Trigger the daily match discovery + prediction + auto-bet pipeline."""
    if not scheduler:
        return jsonify({"error": "Scheduler not initialized"}), 500

    summary = scheduler.run_daily_pipeline()
    return jsonify(summary)


@app.route('/api/admin/add-match', methods=['POST'])
@login_required
def admin_add_match():
    """Manually add an upcoming match."""
    data = request.json
    match_id = tracker.ensure_match(
        home_player_id=int(data['home_player_id']),
        away_player_id=int(data['away_player_id']),
        home_player_name=data['home_player_name'],
        away_player_name=data['away_player_name'],
        match_date=data['match_date'],
        league_id=data.get('league_id'),
        league_name=data.get('league_name'),
        best_of_sets=int(data.get('best_of_sets', 0)),
        best_of_legs=int(data.get('best_of_legs', 5)),
        is_set_format=data.get('is_set_format', False),
    )
    return jsonify({"match_id": match_id, "status": "created"})


@app.route('/api/admin/settle-match', methods=['POST'])
@login_required
def admin_settle_match():
    """Manually settle a match with result."""
    data = request.json
    tracker.settle_match(
        upcoming_match_id=int(data['match_id']),
        home_won=data['home_won'],
        home_score=data.get('home_score'),
        away_score=data.get('away_score'),
    )
    return jsonify({"status": "settled"})


@app.route('/api/admin/predict-match', methods=['POST'])
@login_required
def admin_predict_match():
    """Manually trigger prediction + auto-bet for a specific match."""
    data = request.json
    match_id = int(data['match_id'])
    match = db.get_upcoming_match(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    # Re-run just the prediction and auto-bet steps for this match
    summary = {"predictions_made": 0, "bets_placed": 0}
    scheduler._generate_predictions(summary)
    scheduler._auto_place_bets(summary)
    return jsonify(summary)


@app.route('/api/admin/ingest-results', methods=['POST'])
@login_required
def admin_ingest_results():
    """
    Fetch results for pending matches from Statorium API and settle bets.

    Lightweight ingest — only fetches individual match details for matches
    we're already tracking, not the full pipeline.
    """
    from src.ingest.api_client import StatoriumClient
    from src.ingest.parsers import MatchParser

    statorium_key = os.environ.get("STATORIUM_API_KEY", "")
    if not statorium_key:
        return jsonify({"error": "STATORIUM_API_KEY not configured"}), 400

    pending = db.get_upcoming_matches(status="scheduled")
    today = date.today().isoformat()
    # Only try to settle matches from before today
    past = [m for m in pending if m["match_date"] < today]

    if not past:
        return jsonify({"status": "ok", "message": "No past matches to settle", "settled": 0})

    client = StatoriumClient(api_key=statorium_key)
    parser = MatchParser()
    settled_count = 0
    errors = []

    try:
        for match in past:
            # Try to find the match in Statorium by searching parquet data first
            # (cheaper than API calls)
            home_id = match["home_player_id"]
            away_id = match["away_player_id"]
            match_date = match["match_date"]

            # Check parquet store for result
            matches_df = store.get_matches()
            result_found = False

            if not matches_df.empty:
                # Try home/away as listed
                result_df = matches_df[
                    (matches_df["home_player_id"] == home_id) &
                    (matches_df["away_player_id"] == away_id)
                ]
                if "match_date" in result_df.columns:
                    result_df = result_df[
                        result_df["match_date"].astype(str).str[:10] == match_date
                    ]

                if result_df.empty:
                    # Try reversed
                    result_df = matches_df[
                        (matches_df["home_player_id"] == away_id) &
                        (matches_df["away_player_id"] == home_id)
                    ]
                    if "match_date" in result_df.columns:
                        result_df = result_df[
                            result_df["match_date"].astype(str).str[:10] == match_date
                        ]
                    if not result_df.empty:
                        row = result_df.iloc[0]
                        is_set = bool(row.get("is_set_format", False))
                        home_won = (row["away_sets"] > row["home_sets"]) if is_set else (row["away_legs"] > row["home_legs"])
                        tracker.settle_match(match["id"], home_won)
                        settled_count += 1
                        result_found = True
                else:
                    row = result_df.iloc[0]
                    is_set = bool(row.get("is_set_format", False))
                    home_won = (row["home_sets"] > row["away_sets"]) if is_set else (row["home_legs"] > row["away_legs"])
                    tracker.settle_match(match["id"], home_won)
                    settled_count += 1
                    result_found = True

            if result_found:
                continue

            # Not in parquet — try Statorium API directly
            # We need the Statorium match_id. Check external_match_id or search.
            ext_id = match.get("external_match_id")
            if not ext_id:
                continue

            try:
                match_data = client.get_match(int(ext_id))
                raw = match_data.get("match", match_data)

                # Check if match is finished (has scores)
                home_p = raw.get("homeParticipant", {})
                away_p = raw.get("awayParticipant", {})
                home_sets = int(home_p.get("sets", 0) or 0)
                away_sets = int(away_p.get("sets", 0) or 0)
                home_legs = int(home_p.get("legs", 0) or 0)
                away_legs = int(away_p.get("legs", 0) or 0)

                if home_sets == 0 and away_sets == 0 and home_legs == 0 and away_legs == 0:
                    continue  # Match not yet played

                is_set = home_sets > 0 or away_sets > 0
                home_won = (home_sets > away_sets) if is_set else (home_legs > away_legs)

                h_score = str(home_sets if is_set else home_legs)
                a_score = str(away_sets if is_set else away_legs)

                tracker.settle_match(match["id"], home_won, h_score, a_score)
                settled_count += 1

            except Exception as e:
                errors.append(f"Match {match['id']}: {str(e)}")
                logger.warning(f"Failed to fetch result for match {match['id']}: {e}")

    finally:
        client.close()

    return jsonify({
        "status": "ok",
        "checked": len(past),
        "settled": settled_count,
        "errors": errors,
    })


# =========================================================================
# Startup — always load models on import so gunicorn workers have data
# =========================================================================

print("Loading models...")
load_models()
print(f"Loaded {len(players_list)} players, models: {[k for k, v in models.items() if v]}")


if __name__ == '__main__':
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
