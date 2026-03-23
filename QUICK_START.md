#  Darts Predictor - Quick Start Guide

## Start the Web App (Easiest Way)

### Windows:
Double-click `start_web_app.bat`

### Mac/Linux:
```bash
python app.py
```

### Then:
Open your browser and go to: **http://localhost:5000**

---

## Using the Web Interface

### 1. Select Players
- **Home Player**: Choose from dropdown (e.g., Luke Littler)
- **Away Player**: Choose from dropdown (e.g., Luke Humphries)

### 2. Configure Match Format
- **Set Format**: Check if it's a set format match (like World Championship)
- **Best of Sets**: Enter number (e.g., 7 for World Championship final)
- **Best of Legs**: Enter legs per set (e.g., 5)
- **League**: Select tournament type

### 3. Click "Predict Match"

### 4. View Results

**Win Probabilities:**
- **Glicko-2** (76.9% accuracy) - MOST RELIABLE
- **Elo** (72.8% accuracy)
- **Unified** (60.4% accuracy)

**180s Prediction:**
- Expected 180s for each player
- Total expected 180s
- Probability calculator for any target count
- Full probability matrix

---

## Example Predictions

### World Championship Final Style
```
Home: Luke Littler
Away: Luke Humphries
Format: Set Format 
Best of Sets: 7
Best of Legs: 5
```

**Results:**
- Glicko-2: Littler 86.5% 
- Expected 180s: 18.4 total

### Premier League Style
```
Home: Michael van Gerwen
Away: Gerwyn Price
Format: Leg Format
Best of Legs: 11
```

---

## Understanding the Results

### Win Probabilities
- **70-80%**: Strong favorite
- **55-70%**: Moderate favorite
- **45-55%**: Toss-up
- **Use Glicko-2**: It's the most accurate model (76.9%)

### 180s Predictions
- **Expected count**: Average number of 180s
- **Target probability**: P(180s ≥ X)
- **MAE ±3.1**: Predictions typically within 3 180s

### Ratings
- **Number**: Skill rating (higher = better)
- **(±XX)**: Uncertainty (lower = more confident)

---

## Finding Betting Value

### Example: Over/Under 180s

**Your prediction:**
- Expected total: 18.4 180s
- P(Total ≥ 16) = 63.2%

**Bookmaker line:**
- Over 15.5 at 1.90 (52.6% implied)

**Analysis:**
- Your probability: 63.2%
- Bookmaker implied: 52.6%
- **Edge: 10.6%** → Potential value bet

### Example: Match Winner

**Your prediction (Glicko-2):**
- Littler: 86.5%

**Bookmaker odds:**
- Littler: 1.40 (71.4% implied)

**Analysis:**
- Your probability: 86.5%
- Bookmaker implied: 71.4%
- **Edge: 15.1%** → Strong value bet

** Remember:** Models aren't perfect. Only bet what you can afford to lose.

---

## Tips for Best Results

###  DO:
- Use Glicko-2 predictions (most accurate)
- Only bet when your probability differs from bookmaker by >10%
- Focus on emerging players (model catches form changes)
- Track your bets to verify edge

###  DON'T:
- Bet on every match
- Use the "average" probability (it's less accurate)
- Over-rely on 180s predictions (MAE is high)
- Bet large amounts on single matches

---

## Command Line Predictions (Alternative)

If you prefer command line:

```bash
python predict.py 16839 5730 --league-id 2 --best-of-sets 7 --best-of-legs 5 --set-format
```

Replace player IDs:
- 16839 = Luke Littler
- 5730 = Luke Humphries

Find player IDs in `data/processed/players.parquet` or the web interface.

---

## Troubleshooting

**Web app won't start:**
```bash
# Check if port 5000 is in use
# Try different port:
# Edit app.py, change last line to:
# app.run(debug=True, host='0.0.0.0', port=5001)
```

**No players showing:**
```bash
# Verify data files exist
ls data/processed/
# Should see: players.parquet, matches.parquet, etc.
```

**Predictions seem wrong:**
```bash
# Retrain models with latest data
python train_models.py --model all
```

---

## Next Steps

1. **Explore the interface** - Try different player matchups
2. **Compare predictions** - See how models agree/disagree
3. **Track real matches** - Verify prediction accuracy
4. **Paper trade** - Test betting strategy without real money
5. **Read WEB_APP_README.md** - Learn about API endpoints

---

## Support

For issues or questions:
1. Check WEB_APP_README.md for detailed docs
2. Review training logs in console output
3. Verify model files exist in `data/models/`

**Have fun predicting! **
