"""User model for Flask-Login integration."""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin):
    """User model backed by SQLite via SqliteStore."""

    def __init__(self, id, email, password_hash, display_name=None,
                 bankroll=100.0, kelly_fraction=0.25, created_at=None):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.display_name = display_name or email.split("@")[0]
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.created_at = created_at

    @staticmethod
    def hash_password(password: str) -> str:
        return generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        if data is None:
            return None
        return cls(
            id=data["id"],
            email=data["email"],
            password_hash=data["password_hash"],
            display_name=data.get("display_name"),
            bankroll=data.get("bankroll", 100.0),
            kelly_fraction=data.get("kelly_fraction", 0.25),
            created_at=data.get("created_at"),
        )
