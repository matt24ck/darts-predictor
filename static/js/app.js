// Darts Predictor App

let currentPrediction = null;
window._lastPrediction = null;
let activeTab = 'standard';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Select2 for leagues
    $('#league-ids').select2({
        placeholder: 'Select leagues...',
        allowClear: true,
        width: '100%'
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Standard tab event listeners
    document.getElementById('predict-btn').addEventListener('click', predictMatch);
    document.getElementById('is-set-format').addEventListener('change', (e) => {
        const setsInput = document.getElementById('best-of-sets');
        setsInput.disabled = !e.target.checked;
        if (!e.target.checked) {
            setsInput.value = 0;
        } else {
            setsInput.value = 7;
        }
    });

    // MODUS tab event listeners
    document.getElementById('modus-predict-btn').addEventListener('click', predictMatch);

    // Shared result event listeners
    document.getElementById('calculate-prob-btn').addEventListener('click', calculateTargetProbability);
    document.getElementById('show-matrix-btn').addEventListener('click', showMatrix);
});

// Tab switching
function switchTab(tab) {
    activeTab = tab;

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Update tab panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === 'tab-' + tab);
    });

    // Hide results when switching tabs
    document.getElementById('results').style.display = 'none';
}

// Predict Match
async function predictMatch() {
    let homePlayerId, awayPlayerId, bestOfSets, bestOfLegs, isSetFormat, leagueId;

    if (activeTab === 'modus') {
        homePlayerId = document.getElementById('modus-home-player').value;
        awayPlayerId = document.getElementById('modus-away-player').value;
        bestOfSets = 0;
        bestOfLegs = parseInt(document.getElementById('modus-best-of-legs').value);
        isSetFormat = false;
        leagueId = 38;
    } else {
        homePlayerId = document.getElementById('home-player').value;
        awayPlayerId = document.getElementById('away-player').value;
        bestOfSets = parseInt(document.getElementById('best-of-sets').value);
        bestOfLegs = parseInt(document.getElementById('best-of-legs').value);
        isSetFormat = document.getElementById('is-set-format').checked;
        const leagueIds = $('#league-ids').val();
        leagueId = leagueIds && leagueIds.length > 0 ? parseInt(leagueIds[0]) : 2;
    }

    // Validation
    if (!homePlayerId || !awayPlayerId) {
        alert('Please select both players');
        return;
    }

    if (homePlayerId === awayPlayerId) {
        alert('Please select different players');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                home_player_id: homePlayerId,
                away_player_id: awayPlayerId,
                best_of_sets: bestOfSets,
                best_of_legs: bestOfLegs,
                is_set_format: isSetFormat,
                league_id: leagueId
            })
        });

        const data = await response.json();
        currentPrediction = data;
        window._lastPrediction = data;

        displayResults(data, isSetFormat, bestOfSets, bestOfLegs);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(data, isSetFormat, bestOfSets, bestOfLegs) {
    // Match header
    document.getElementById('match-title').textContent =
        `${data.home_player} vs ${data.away_player}`;

    let formatText = isSetFormat
        ? `Best of ${bestOfSets} sets (best of ${bestOfLegs} legs per set)`
        : `Best of ${bestOfLegs} legs`;
    document.getElementById('match-format').textContent = formatText;

    // Win probabilities
    displayWinProbabilities(data);

    // Ratings
    displayRatings(data);

    // 180s
    if (data['180s']) {
        display180s(data);
    } else {
        document.getElementById('180s-section').style.display = 'none';
    }

    // Show betting panel if win probabilities available
    const bettingPanel = document.getElementById('betting-panel');
    if (bettingPanel && data.win_probability) {
        bettingPanel.style.display = 'block';
    }

    // Show results
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Display Win Probabilities
function displayWinProbabilities(data) {
    const container = document.getElementById('win-probabilities');
    container.innerHTML = '';

    if (data.win_probability) {
        const pred = data.win_probability;
        const modelName = data.model_info.win_model === 'glicko2'
            ? 'Glicko-2 (63% Test Accuracy, Well-Calibrated)'
            : 'Unified Model (Backup)';
        const div = createPredictionCard(
            modelName,
            data.home_player,
            data.away_player,
            pred.home_win,
            pred.away_win,
            pred.confidence_interval
        );
        container.appendChild(div);
    } else {
        container.innerHTML = '<p style="color: var(--danger-color);">Predictions not available for these players.</p>';
    }
}

// Format decimal odds from probability
function formatOdds(prob) {
    if (prob <= 0) return '-';
    return (1 / prob).toFixed(2);
}

// Create Prediction Card
function createPredictionCard(modelName, homeName, awayName, homeProb, awayProb, confidenceInterval) {
    const div = document.createElement('div');
    div.className = 'model-prediction';

    let confidenceText = '';
    if (confidenceInterval && confidenceInterval.home_lower !== undefined) {
        confidenceText = `<div style="margin-top: 10px; color: #607d8b; font-size: 0.85rem;">
            95% CI: ${(confidenceInterval.home_lower * 100).toFixed(1)}% - ${(confidenceInterval.home_upper * 100).toFixed(1)}%
        </div>`;
    }

    const homeOdds = formatOdds(homeProb);
    const awayOdds = formatOdds(awayProb);

    div.innerHTML = `
        <div class="model-name">${modelName}</div>
        <div class="probability-bars">
            <div class="prob-bar">
                <div class="prob-label">
                    <span>${homeName}</span>
                    <span>${(homeProb * 100).toFixed(1)}%<span class="prob-odds">${homeOdds}</span></span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${homeProb * 100}%">
                        ${(homeProb * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
            <div class="prob-bar">
                <div class="prob-label">
                    <span>${awayName}</span>
                    <span>${(awayProb * 100).toFixed(1)}%<span class="prob-odds">${awayOdds}</span></span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${awayProb * 100}%; background: linear-gradient(90deg, #e74c3c, #c0392b)">
                        ${(awayProb * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>
        ${confidenceText}
    `;

    return div;
}

// Display Ratings
function displayRatings(data) {
    const container = document.getElementById('ratings');
    container.innerHTML = '';

    const grid = document.createElement('div');
    grid.className = 'ratings-grid';

    if (data.ratings.glicko) {
        const card = createRatingCard('Glicko-2', data.ratings.glicko, data.home_player, data.away_player);
        grid.appendChild(card);
    }

    container.appendChild(grid);
}

// Create Rating Card
function createRatingCard(modelName, ratings, homeName, awayName) {
    const div = document.createElement('div');
    div.className = 'rating-card';

    const homeRD = ratings.home_rd ? ` (RD ${ratings.home_rd.toFixed(0)})` : '';
    const awayRD = ratings.away_rd ? ` (RD ${ratings.away_rd.toFixed(0)})` : '';

    div.innerHTML = `
        <div class="rating-model">${modelName}</div>
        <div class="rating-row">
            <span>${homeName}:</span>
            <span class="rating-value">${ratings.home.toFixed(1)}${homeRD}</span>
        </div>
        <div class="rating-row">
            <span>${awayName}:</span>
            <span class="rating-value">${ratings.away.toFixed(1)}${awayRD}</span>
        </div>
        <div class="rating-row">
            <span>Difference:</span>
            <span class="rating-value">${(ratings.home - ratings.away).toFixed(1)}</span>
        </div>
    `;

    return div;
}

// Display 180s
function display180s(data) {
    const section = document.getElementById('180s-section');
    section.style.display = 'block';

    // Update model badge
    const badge = document.getElementById('180s-model-badge');
    if (activeTab === 'modus') {
        badge.textContent = 'MODUS Model';
        badge.className = 'model-badge modus';
    } else {
        badge.textContent = 'Global Model';
        badge.className = 'model-badge';
    }

    document.getElementById('home-player-label').textContent = data.home_player;
    document.getElementById('away-player-label').textContent = data.away_player;

    document.getElementById('expected-home-180s').textContent =
        data['180s'].expected_home.toFixed(2);
    document.getElementById('expected-away-180s').textContent =
        data['180s'].expected_away.toFixed(2);
    document.getElementById('expected-total-180s').textContent =
        data['180s'].expected_total.toFixed(2);

    // Set default target to expected total
    document.getElementById('target-180s').value =
        Math.round(data['180s'].expected_total);
}

// Calculate Target Probability
async function calculateTargetProbability() {
    if (!currentPrediction || !currentPrediction['180s']) {
        alert('Please run a prediction first');
        return;
    }

    const target = parseInt(document.getElementById('target-180s').value);

    try {
        const response = await fetch('/api/180s_probability', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                lambda_home: currentPrediction['180s'].expected_home,
                lambda_away: currentPrediction['180s'].expected_away,
                target: target
            })
        });

        const data = await response.json();
        displayTargetProbability(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to calculate probability');
    }
}

// Display Target Probability
function displayTargetProbability(data) {
    const container = document.getElementById('target-probability');

    container.innerHTML = `
        <div style="margin-bottom: 15px;">
            <strong>P(Total 180s >= ${data.target}):</strong>
            <span class="prob-highlight">${(data.prob_total_ge * 100).toFixed(1)}%</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <strong>${currentPrediction.home_player}:</strong>
                ${(data.prob_home_ge * 100).toFixed(1)}%
            </div>
            <div>
                <strong>${currentPrediction.away_player}:</strong>
                ${(data.prob_away_ge * 100).toFixed(1)}%
            </div>
        </div>
    `;
}

// Show Matrix
async function showMatrix() {
    if (!currentPrediction || !currentPrediction['180s']) {
        alert('Please run a prediction first');
        return;
    }

    const maxCount = parseInt(document.getElementById('max-180s').value);

    showLoading();

    try {
        const response = await fetch('/api/180s_matrix', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                lambda_home: currentPrediction['180s'].expected_home,
                lambda_away: currentPrediction['180s'].expected_away,
                max_180s: maxCount
            })
        });

        const data = await response.json();
        displayMatrix(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate matrix');
    } finally {
        hideLoading();
    }
}

// Display Matrix
function displayMatrix(data) {
    const container = document.getElementById('180s-matrix');

    let html = `
        <table class="matrix-table">
            <thead>
                <tr>
                    <th>180s Count</th>
                    <th>${currentPrediction.home_player}</th>
                    <th>${currentPrediction.away_player}</th>
                    <th>Total</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.matrix.forEach(row => {
        html += `
            <tr>
                <td>${row.x}</td>
                <td>${(row.prob_home_ge * 100).toFixed(1)}%</td>
                <td>${(row.prob_away_ge * 100).toFixed(1)}%</td>
                <td>${(row.prob_total_ge * 100).toFixed(1)}%</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// Loading
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}
