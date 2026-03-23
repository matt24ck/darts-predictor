# Darts Match Predictor Web App

A web-based UI for predicting darts match outcomes and 180s distributions.

## Features

- **Player Selection**: Choose from all players in your database
- **Match Format**: Configure best-of-sets, best-of-legs, league
- **Win Probability Predictions**:
  - Glicko-2 (76.9% accuracy) 
  - Elo (72.8% accuracy)
  - Unified (60.4% accuracy)
- **Player Ratings**: View current ratings across all models
- **180s Prediction**:
  - Expected 180 counts per player
  - Target probability calculator (e.g., P(Total 180s ≥ 10))
  - Full probability matrix

## Quick Start

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open in Browser

Navigate to: `http://localhost:5000`

### 3. Make a Prediction

1. Select home and away players
2. Choose match format (set format or legs only)
3. Configure best-of-sets and best-of-legs
4. Click "Predict Match"

### 4. Explore 180s

Once prediction is shown:
- View expected 180 counts for each player
- Enter a target count and calculate probability
- Show full probability matrix for all counts

## API Endpoints

### POST /api/predict
Predict match outcome.

**Request:**
```json
{
  "home_player_id": 16839,
  "away_player_id": 5730,
  "best_of_sets": 7,
  "best_of_legs": 5,
  "is_set_format": true,
  "league_id": 2
}
```

**Response:**
```json
{
  "home_player": "Luke Littler",
  "away_player": "Luke Humphries",
  "predictions": {
    "elo": {"home_win": 0.765, "away_win": 0.235},
    "glicko": {"home_win": 0.865, "away_win": 0.135},
    "unified": {"home_win": 0.439, "away_win": 0.561, "confidence": 0.82}
  },
  "ratings": {...},
  "180s": {
    "lambda_home": 12.74,
    "lambda_away": 5.70,
    "lambda_total": 18.44
  }
}
```

### POST /api/180s_probability
Calculate P(180s ≥ target).

**Request:**
```json
{
  "lambda_home": 12.74,
  "lambda_away": 5.70,
  "target": 10
}
```

### POST /api/180s_matrix
Get full probability matrix.

**Request:**
```json
{
  "lambda_home": 12.74,
  "lambda_away": 5.70,
  "max_180s": 20
}
```

## Example Usage

### Predicting Luke Littler vs Luke Humphries

1. Select "Luke Littler" as home player
2. Select "Luke Humphries" as away player
3. Check "Set Format"
4. Set "Best of Sets" to 7
5. Set "Best of Legs" to 5
6. Click "Predict Match"

**Results:**
- Glicko-2: Littler 86.5%
- Elo: Littler 76.5%
- Unified: Humphries 56.1%
- Expected 180s: 18.4 total (Littler 12.7, Humphries 5.7)

### Finding Over/Under Value

If bookmaker line is "Total 180s Over/Under 15.5":

1. Run prediction to get expected total (e.g., 18.4)
2. Enter 16 in "Target 180 Count"
3. Click "Calculate Probability"
4. See P(Total ≥ 16) = 63.2%
5. If bookmaker offers Over at 50% odds → VALUE BET

## Customization

### Adding More Models

Edit `app.py` and add your model to the `load_models()` function:

```python
try:
    my_model = MyCustomModel()
    my_model.load("data/models/my_model.json")
    models['my_model'] = my_model
except:
    models['my_model'] = None
```

### Changing Styling

Edit `static/css/style.css` to customize colors, fonts, layout.

### Adding Features

- Add new API endpoints in `app.py`
- Update UI in `templates/index.html`
- Add JavaScript handlers in `static/js/app.js`

## Deployment

### Local Network Access

To access from other devices on your network:

```bash
python app.py
# Server runs on 0.0.0.0:5000
# Access from other devices at http://YOUR_IP:5000
```

### Production Deployment

For production use with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## Troubleshooting

**Models not loading:**
- Ensure model files exist in `data/models/`
- Check file paths in `app.py`

**Players not showing:**
- Verify `data/processed/players.parquet` exists
- Check database connection

**Port already in use:**
- Change port in `app.py`: `app.run(port=5001)`

## License

Same as main project.
