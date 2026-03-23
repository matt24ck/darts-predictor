"""Models module for darts pipeline."""

# Rating systems
from .glicko2_system import Glicko2System, Glicko2Config, PlayerRating, Glicko2HistoryRecord

# Visit-level 180s model (uses individual visit data)
from .visit_level_180s import VisitLevel180sModel, VisitLevel180sResult, PlayerVisitStats

# Prediction API
from .predictions import (
    predict_180_distribution,
    predict_win_probability,
    compute_most_180s_probabilities,
    predict_most_180s,
)

__all__ = [
    # Rating systems
    "Glicko2System",
    "Glicko2Config",
    "PlayerRating",
    "Glicko2HistoryRecord",

    # Visit-level 180s model
    "VisitLevel180sModel",
    "VisitLevel180sResult",
    "PlayerVisitStats",

    # Prediction API
    "predict_180_distribution",
    "predict_win_probability",
    "compute_most_180s_probabilities",
    "predict_most_180s",
]
