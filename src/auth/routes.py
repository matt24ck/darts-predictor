"""Authentication routes (login, register, logout)."""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from .models import User

auth_bp = Blueprint("auth", __name__)

# SqliteStore instance is set by the app on startup
_db = None


def init_auth(db):
    """Initialize auth module with the SqliteStore instance."""
    global _db
    _db = db


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        display_name = request.form.get("display_name", "").strip()

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        existing = _db.get_user_by_email(email)
        if existing:
            flash("An account with this email already exists.", "error")
            return render_template("register.html")

        password_hash = User.hash_password(password)
        user_id = _db.create_user(email, password_hash, display_name or None)

        user_data = _db.get_user_by_id(user_id)
        user = User.from_dict(user_data)
        login_user(user)
        flash("Account created successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user_data = _db.get_user_by_email(email)
        if not user_data:
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        user = User.from_dict(user_data)
        if not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        login_user(user)
        next_page = request.args.get("next")
        return redirect(next_page or url_for("dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("dashboard"))


@auth_bp.route("/settings", methods=["GET", "POST"])
@login_required
def user_settings():
    if request.method == "POST":
        bankroll = float(request.form.get("bankroll", 100.0))
        kelly_fraction = float(request.form.get("kelly_fraction", 0.25))

        kelly_fraction = max(0.0, min(1.0, kelly_fraction))
        bankroll = max(0.0, bankroll)

        _db.update_user_settings(current_user.id, bankroll, kelly_fraction)
        current_user.bankroll = bankroll
        current_user.kelly_fraction = kelly_fraction
        flash("Settings updated.", "success")
        return redirect(url_for("auth.user_settings"))

    return render_template("settings.html")
