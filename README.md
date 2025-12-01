# F1 Predictor 2025

Predict Formula 1 race winners using machine learning trained on 75 years of F1 data (1950-2024).

## Overview

This project uses historical F1 data to predict race winners with high accuracy:
- **Training**: 70 years of data (1950-2019)
- **Validation**: 1 year (2020)
- **Test**: 4 years (2021-2024) - for robust evaluation
- **Test winners**: ~68 (statistically reliable)

## Models

Three pre-trained models are available:

| Model | F1 Score | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| **Gradient Boosting** ⭐ | 0.9892 | 0.9787 | 1.0000 | 1.0000 |
| Random Forest | 0.7308 | 0.6552 | 0.8261 | 0.9899 |
| XGBoost | 0.7308 | 0.6552 | 0.8261 | 0.9899 |

**Recommended**: Use Gradient Boosting for best performance.

## Installation

```bash
pip install -r requirements.txt

streamlit run f1_predictor.py  
```

## Usage

### Load Pre-trained Model

```python
from f1_predictor import F1Predictor

# Load Gradient Boosting model (best performance)
predictor = F1Predictor(model_type='gradient_boosting')
predictor.load_model()  # Loads f1_model_gradient_boosting_5yr_test.joblib

# Make predictions for 2025
predictions = predictor.predict_2025_race('Australian Grand Prix')
```

### Train New Models

```bash
python3 train_models.py
```

This trains all 3 models on:
- Training: 1950-2019 (70 years)
- Validation: 2020
- Test: 2021-2024 (4 years, ~68 winners)



## Files

```
data_loader.py                             # Feature engineering & data loading
f1_predictor.py                            # Model training & prediction
train_models.py                            # Train all 3 models
requirements.txt                           # Dependencies
f1_model_gradient_boosting_5yr_test.joblib    # Trained GB model ⭐
f1_model_random_forest_5yr_test.joblib       # Trained RF model
f1_model_xgboost_5yr_test.joblib            # Trained XGBoost model
f1data/                                    # Historical F1 data (CSV files)
```

## Features Used

17 engineered features:
- `grid` - Starting grid position
- `qual_position_avg` - Recent qualifying performance
- `points_moving_avg` - Recent points trend
- `circuit_wins` - Historical wins at circuit
- `win_rate_5` - Winning rate (last 5 races)
- `podium_rate_5` - Podium rate (last 5 races)
- `avg_position_5` - Average finish position (last 5)
- `grid_to_position_gap` - Grid vs finish improvement
- `points_championship` - Championship points
- `position_championship` - Championship position
- `constructor_points_mean` - Team avg points
- `constructor_points_std` - Team consistency
- `constructor_position_mean` - Team avg position
- `constructor_wins` - Team total wins
- `nationality` - Driver nationality
- `nationality_constructor` - Team nationality
- `country` - Race country

## Training Details

### Class Imbalance Handling
- SMOTE oversampling: 5% → 50% balance
- Threshold optimization: Dynamic decision boundary
- Class weighting: Penalize winner misclassification

### Data Split
- **Temporal**: Train ≤2019, Val 2020, Test 2021-2024
- **Class balance**: ~5% winners (before SMOTE)
- **Total samples**: 26,759 historical results

### Model Hyperparameters

**Gradient Boosting**:
- n_estimators: 150
- max_depth: 4
- learning_rate: 0.1
- threshold: 0.15 (optimized on validation)

**Random Forest**:
- n_estimators: 200
- max_depth: 6
- threshold: 0.70 (optimized on validation)

**XGBoost**:
- n_estimators: 200
- max_depth: 6
- threshold: 0.70 (optimized on validation)

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- imbalanced-learn >= 0.10.1
- xgboost >= 1.7.0
- joblib >= 1.0.0
- streamlit >= 1.0.0 (optional, for web UI)
- plotly >= 5.0.0 (optional, for visualizations)

## License

MIT License

- Model predictions are probability estimates, not certainties
- Real F1 outcomes depend on many unpredictable factors
- Use predictions as one input among many analytical tools
- Historical performance doesn't guarantee future results
