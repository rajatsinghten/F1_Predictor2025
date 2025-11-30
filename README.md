# F1 2025 AI Predictor

A machine learning model that predicts Formula 1 race winners using historical data and advanced algorithms.

## Quick Start

### Prerequisites
- Python 3.10+
- macOS/Linux/Windows

### 1. Clone & Setup

```bash
# Navigate to project
cd FormulaOne_Predictor2025

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

**Option A: Web Interface (Recommended)**
```bash
streamlit run f1_predictor.py
```
Then open `http://localhost:8501` in your browser.

**Option B: Test the Improved Model**
```bash
python3 test_improvements.py
```

**Option C: Train a New Model**
```bash
python3 << 'EOF'
from f1_predictor import F1Predictor

predictor = F1Predictor(model_type='gradient_boosting')
metrics = predictor.train_model(use_smote=True, optimize_threshold=True)
predictor.save_model('f1_model.joblib')

print(f" Model trained! F1 Score: {metrics['Test F1 Score']:.4f}")
EOF
```

## Project Structure

```
FormulaOne_Predictor2025/
 f1_predictor.py          # Main predictor & Streamlit app
 data_loader.py           # Data loading & feature engineering
 f1_model.joblib          # Trained ML model
 requirements.txt         # Python dependencies
 README.md               # This file
 IMPROVEMENTS.md         # Detailed improvement docs
 MODEL_SUMMARY.md        # Model performance summary
 test_improvements.py    # Verification script
 f1data/                 # Historical F1 data (CSV files)
     races.csv
     results.csv
     drivers.csv
     constructors.csv
     qualifying.csv
     circuits.csv
     ... (other CSVs)
```

## Features

### Prediction Capabilities
-  Predict race winners for upcoming 2025 F1 races
-  Win probability for each driver
-  Championship simulation
-  Feature importance analysis
-  Interactive web dashboard

### Model Performance
- **F1 Score**: 0.80 (80% accuracy on minority class)
- **Recall**: 75% (finds 3 out of 4 winners)
- **Precision**: 86% (reliable predictions)
- **ROC AUC**: 0.99 (excellent discrimination)

## How It Works

The model uses **Gradient Boosting** with:
- SMOTE oversampling for class balance
- Optimized decision threshold (0.70)
- 17 engineered features including:
  - Grid position
  - Recent win/podium rates
  - Average finishing position
  - Championship standing
  - Team performance metrics

## Using the Web App

1. **Race Predictor**: Select a 2025 race and get win probabilities
2. **Championship Simulation**: See predicted final standings
3. **Model Management**: Retrain or reload the model
4. **Calendar View**: Track race schedule and results

### Navigation
- Use sidebar to switch between features
- Click race name to make predictions
- View feature importance charts

## Command Line Usage

### Make Predictions
```python
from f1_predictor import F1Predictor

predictor = F1Predictor()
predictor.load_model()

# Predict a specific race
results = predictor.predict_2025_race('Brazilian Grand Prix')
print(results)
```

### Get Feature Importance
```python
print(predictor.feature_importance.head(10))
```

### Simulate Championship
```python
standings, constructor_standings, races = predictor.simulate_championship()
print(standings)
```

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `streamlit` - Web interface
- `plotly` - Interactive charts
- `imbalanced-learn` - SMOTE oversampling
- `xgboost` - Gradient boosting (optional)
- `joblib` - Model serialization

## Configuration

### Model Types Available
```python
# Gradient Boosting (Recommended)
predictor = F1Predictor(model_type='gradient_boosting')

# Random Forest
predictor = F1Predictor(model_type='random_forest')

# XGBoost (requires: brew install libomp)
predictor = F1Predictor(model_type='xgboost')
```

### Training Options
```python
predictor.train_model(
    use_smote=True,           # Apply SMOTE oversampling
    optimize_threshold=True   # Optimize decision boundary
)
```

## Data

The model is trained on historical F1 data (2022 and earlier):
- **Validation**: 2023 season
- **Test**: 2024 season
- **Prediction**: 2025 season

Data includes:
- Race results and qualifying
- Driver & constructor standings
- Circuit information
- Lap times and pit stops

## Important Notes

- Model predictions are probability estimates, not certainties
- Real F1 outcomes depend on many unpredictable factors
- Use predictions as one input among many analytical tools
- Historical performance doesn't guarantee future results
