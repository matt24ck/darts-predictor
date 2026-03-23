# Darts Modelling Pipeline

End-to-end darts analytics pipeline built on the Statorium Darts API. Ingests historical match data, normalizes it into a clean analytics schema, and trains models for:

1. **180s Prediction**: Probability distribution for total 180s in a match
2. **Elo Rating System**: Win probabilities based on player ratings

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Ingestion

Run the pipeline to fetch and store data:

```bash
python run_pipeline.py --api-key YOUR_API_KEY
```

Options:
- `--seasons N`: Number of seasons per league (default: 5)
- `--data-dir PATH`: Output directory (default: data/processed)
- `--verbose`: Enable debug logging

### 2. Model Training

Train the prediction models:

```bash
python train_models.py
```

Options:
- `--model {180s,elo,all}`: Which model to train (default: all)
- `--base-k FLOAT`: Base K factor for Elo (default: 32.0)
- `--k-strategy {fixed,margin,performance,combined}`: K-scaling strategy

### 3. Making Predictions

```python
from src.models import predict_180_distribution, predict_win_probability

# Predict 180s distribution
result = predict_180_distribution(
    home_player_id=16839,  # Luke Littler
    away_player_id=127,    # Michael van Gerwen
    league_id=2,           # PDC World Championship
    format_params={
        "best_of_sets": 13,
        "best_of_legs": 5,
        "is_set_format": True,
    },
    line_x=10,  # Get P(>=10), P(=10), P(<10)
)

print(f"Expected 180s: {result['lambda_total']:.1f}")
print(f"P(>=10 180s): {result['prob_ge_x']:.3f}")

# Predict win probability
result = predict_win_probability(
    home_player_id=16839,
    away_player_id=127,
)

print(f"P(Littler wins): {result['p_home_win']:.3f}")
print(f"Elo difference: {result['elo_difference']:.0f}")
```

## Project Structure

```
darts_pipeline/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration (leagues, model params)
├── src/
│   ├── schema/
│   │   └── models.py         # Data models (Player, Match, Visit, etc.)
│   ├── ingest/
│   │   ├── api_client.py     # Statorium API client
│   │   ├── fetchers.py       # League, season, match fetchers
│   │   └── parsers.py        # Match data parsers
│   ├── storage/
│   │   ├── base.py           # Storage interface
│   │   └── parquet_store.py  # Parquet implementation
│   └── models/
│       ├── model_180s.py     # 180s prediction model
│       ├── elo_system.py     # Elo rating system
│       └── predictions.py    # High-level prediction API
├── data/
│   ├── processed/            # Parquet files
│   └── models/               # Saved model files
├── run_pipeline.py           # Data ingestion script
├── train_models.py           # Model training script
└── requirements.txt
```

## Data Schema

### Players
| Column | Type | Description |
|--------|------|-------------|
| player_id | int | Unique identifier |
| name | str | Display name |
| short_name | str | Abbreviated name |
| full_name | str | Complete name |

### Matches
| Column | Type | Description |
|--------|------|-------------|
| match_id | int | Unique identifier |
| league_id | int | Parent league |
| season_id | int | Parent season |
| match_date | date | Date of match |
| home_player_id | int | Home player |
| away_player_id | int | Away player |
| home_sets | int | Sets won by home |
| away_sets | int | Sets won by away |
| home_legs | int | Total legs won by home |
| away_legs | int | Total legs won by away |
| start_score | int | Starting score (e.g., 501) |
| best_of_sets | int | Format: best of X sets |
| best_of_legs | int | Format: best of X legs |
| has_visit_data | bool | Visit-level data available |
| is_set_format | bool | Uses sets (vs legs only) |

### Visits (visit-level matches only)
| Column | Type | Description |
|--------|------|-------------|
| match_id | int | Parent match |
| set_no | int | Set number |
| leg_no | int | Leg number |
| visit_index | int | Order in leg |
| player_id | int | Player ID |
| score | int | Points scored |
| attempts | int | Darts thrown |
| is_180 | bool | Score == 180 |
| is_140_plus | bool | Score >= 140 |

### Match Stats
| Column | Type | Description |
|--------|------|-------------|
| match_id | int | Parent match |
| player_id | int | Player ID |
| stat_field_id | int | Stat type |
| value | any | Stat value |

Standard stat fields:
- 1: Average 3 darts
- 2: Thrown 180
- 3: Thrown over 140
- 4: Thrown over 100
- 5: Highest checkout
- 6: Checkouts over 100
- 7: Checkout accuracy

### Elo History
| Column | Type | Description |
|--------|------|-------------|
| match_id | int | Match ID |
| home_elo_pre | float | Home pre-match Elo |
| away_elo_pre | float | Away pre-match Elo |
| home_elo_post | float | Home post-match Elo |
| away_elo_post | float | Away post-match Elo |
| home_expected | float | Expected score for home |
| k_factor_used | float | K factor applied |

## Configuration

Edit `config/settings.py` to customize:

### Leagues

```python
CORE_LEAGUES = {
    2: "PDC World Darts Championship",
    3: "PDC Premier League Darts",
    4: "PDC World Matchplay",
    5: "PDC World Grand Prix",
    6: "PDC Grand Slam of Darts",
    7: "PDC UK Open",
    8: "PDC Players Championship Finals",
    35: "Players Championships",
    38: "MODUS Super Series",
}
```

### Elo Configuration

```python
ELO_CONFIG = EloConfig(
    base_k=32.0,              # Base K factor
    initial_rating=1500.0,    # Starting rating
    scale=400.0,              # Rating scale
    k_scaling_strategy="combined",  # fixed, margin, performance, combined
    margin_k_min=0.5,         # Min K multiplier
    margin_k_max=1.5,         # Max K multiplier
    performance_weight=0.3,   # Weight of performance in combined strategy
)
```

### 180s Model Configuration

```python
MODEL_180S_CONFIG = Model180sConfig(
    model_type="poisson",
    min_player_matches=3,
    regularization_strength=0.01,
)
```

## Models

### 180s Model

Poisson regression for total 180s:

```
log(lambda) = intercept + player_effects + format_effects + league_effects
```

- Player effects based on historical 180 rates
- Format effects scale by expected legs per match
- Outputs full probability distribution P(X = x)

### Elo System

Standard Elo with K-scaling:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R'_A = R_A + K * (S_A - E_A)
```

K-scaling strategies:
- **fixed**: Constant K
- **margin**: Scale by match closeness
- **performance**: Scale by performance vs baseline
- **combined**: Weighted combination

"Punish-less" behavior:
- Close loss with good stats -> smaller rating decrease
- Big upset loss with bad stats -> larger rating decrease

## API Reference

### predict_180_distribution()

```python
result = predict_180_distribution(
    home_player_id: int,
    away_player_id: int,
    league_id: int,
    format_params: dict = None,
    line_x: int = None,
)
```

Returns:
- `lambda_total`: Expected total 180s
- `lambda_home`: Expected home player 180s
- `lambda_away`: Expected away player 180s
- `prob_equal_x`: P(total == x) if line_x provided
- `prob_ge_x`: P(total >= x) if line_x provided
- `prob_lt_x`: P(total < x) if line_x provided

### predict_win_probability()

```python
result = predict_win_probability(
    home_player_id: int,
    away_player_id: int,
)
```

Returns:
- `p_home_win`: Probability home player wins
- `p_away_win`: Probability away player wins
- `home_elo`: Home player's current rating
- `away_elo`: Away player's current rating
- `elo_difference`: Rating difference

### get_leaderboard()

```python
result = get_leaderboard(n=50)
```

Returns top N players by Elo rating.

## License

MIT
