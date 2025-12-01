import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from datetime import datetime, timedelta
import pytz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure this matches the filename of your data loader
from data_loader import F1DataLoaderFixed as F1DataLoader

# Try to import optional imbalance-handling libraries
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False

F1_CALENDAR_2025 = [
    "Australian Grand Prix - Melbourne (16 Mar)",
    "Chinese Grand Prix - Shanghai (23 Mar) ",
    "Japanese Grand Prix - Suzuka (6 Apr)",
    "Bahrain Grand Prix - Sakhir (13 Apr)",
    "Saudi Arabian Grand Prix - Jeddah (20 Apr)",
    "Miami Grand Prix - Miami (4 May)",
    "Emilia Romagna Grand Prix - Imola (18 May)",
    "Monaco Grand Prix - Monte Carlo (25 May)",
    "Spanish Grand Prix - Barcelona (1 Jun)",
    "Canadian Grand Prix - Montreal (15 Jun)",
    "Austrian Grand Prix - Spielberg (29 Jun)",
    "British Grand Prix - Silverstone (6 Jul)",
    "Belgian Grand Prix - Spa-Francorchamps (27 Jul)",
    "Hungarian Grand Prix - Budapest (3 Aug)",
    "Dutch Grand Prix - Zandvoort (31 Aug)",
    "Italian Grand Prix - Monza (7 Sep)",
    "Azerbaijan Grand Prix - Baku (21 Sep)",
    "Singapore Grand Prix - Singapore (5 Oct)",
    "United States Grand Prix - Austin (19 Oct)",
    "Mexico City Grand Prix - Mexico City (26 Oct)",
    "São Paulo Grand Prix - São Paulo (9 Nov)",
    "Las Vegas Grand Prix - Las Vegas (22 Nov)",
    "Qatar Grand Prix - Lusail (30 Nov)",
    "Abu Dhabi Grand Prix - Yas Marina (7 Dec)"
]

_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def parse_calendar_entry(entry: str, year=2025):
    """
    Parse a calendar entry like:
      "Australian Grand Prix - Melbourne (16 Mar) - COMPLETED"
    Returns:
      race_title (str), date_obj (datetime.date or None), explicitly_completed (bool)
    """
    explicitly_completed = 'COMPLETED' in entry.upper()
    date_match = re.search(r'\((\d{1,2})\s*([A-Za-z]{3})\)', entry)
    date_obj = None
    if date_match:
        day = int(date_match.group(1))
        mon = date_match.group(2).lower()
        month = _MONTH_MAP.get(mon[:3].lower())
        if month:
            try:
                date_obj = datetime(year, month, day).date()
            except Exception:
                date_obj = None

    name = entry
    name = re.sub(r'\s*-\s*COMPLETED\s*$', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'\s*\(\d{1,2}\s*[A-Za-z]{3}\)\s*', '', name).strip()
    return name, date_obj, explicitly_completed

class F1Predictor:
    def __init__(self, data_path='f1data', model_type='random_forest'):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.threshold = 0.5
        self.data_loader = F1DataLoader(data_path)
        self.feature_importance = None
        self.grid_2025 = None
        self.results_2025 = None
        self.feature_columns = None
        
        # Initialize the specific model architecture
        self._init_model_architecture()
        
        self.load_2025_data()

    def _init_model_architecture(self):
        """Helper to initialize the internal model based on model_type."""
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=8,
                random_state=42
            )
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_child_weight=10,
                scale_pos_weight=19,
                random_state=42,
                tree_method='hist',
                eval_metric='logloss'
            )
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=5,
                class_weight='balanced',
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )

    def load_2025_data(self):
        try:
            self.grid_2025 = pd.read_csv(f'{self.data_path}/f1_2025_grid.csv')
        except Exception as e:
            print(f"Could not load 2025 grid: {e}")
            self.grid_2025 = None

        try:
            self.results_2025 = pd.read_csv(f'{self.data_path}/f1_2025_results.csv')
            possible_cols = ['race_name', 'race', 'raceName', 'raceName', 'round']
            self.results_2025.columns = [c.strip() for c in self.results_2025.columns]
            if 'date' in self.results_2025.columns:
                try:
                    self.results_2025['date'] = pd.to_datetime(self.results_2025['date'], errors='coerce')
                except Exception:
                    pass
        except Exception as e:
            print(f"Could not load 2025 results: {e}")
            self.results_2025 = None

        self.completed_races_set = set()
        if self.results_2025 is not None:
            for name_col in ['race_name', 'race', 'raceName', 'raceName', 'round']:
                if name_col in self.results_2025.columns:
                    vals = self.results_2025[name_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
                    self.completed_races_set.update(vals)
        self.completed_races_set = set([s.lower() for s in self.completed_races_set])

    def get_driver_recent_results(self, driver_name, n=5):
        if self.results_2025 is None:
            return None
        driver_cols = [c for c in ['driver_name', 'driver', 'Driver'] if c in self.results_2025.columns]
        if not driver_cols:
            return None
        driver_col = driver_cols[0]
        driver_results = self.results_2025[self.results_2025[driver_col] == driver_name].copy()

        if 'date' in driver_results.columns and pd.api.types.is_datetime64_any_dtype(driver_results['date']):
            driver_results = driver_results.sort_values('date', ascending=False)
        elif 'raceId' in driver_results.columns:
            driver_results = driver_results.sort_values('raceId', ascending=False)
        else:
            driver_results = driver_results.sort_index(ascending=False)
        return driver_results.head(n)

    def _calendar_parsed(self):
        parsed = []
        for entry in F1_CALENDAR_2025:
            parsed.append(parse_calendar_entry(entry, year=2025))
        return parsed

    def _remaining_races_from_calendar_and_results(self):
        tz = pytz.timezone('Asia/Kolkata')
        today = datetime.now(tz).date()
        parsed = self._calendar_parsed()
        remaining = []
        for race_name, race_date, explicitly_completed in parsed:
            rn_lower = race_name.strip().lower()
            if explicitly_completed:
                continue
            if rn_lower in self.completed_races_set:
                continue
            if race_date is not None and race_date < today:
                continue
            remaining.append(race_name)
        return remaining

    def train_model(self, model_type=None, use_smote=False, optimize_threshold=True,
                    train_years=(1950, 2022), val_years=(2023, 2023), test_years=(2024, 2024)):
        """
        Train a specific model type. Can override the instance's current model_type.
        
        Args:
            model_type: Override instance model type if provided
            use_smote: Apply SMOTE oversampling for class imbalance
            optimize_threshold: Find optimal decision threshold on validation set
            train_years: tuple (min_year, max_year) for training set
            val_years: tuple (min_year, max_year) for validation set
            test_years: tuple (min_year, max_year) for test set
        """
        if model_type:
            self.model_type = model_type
            self._init_model_architecture()

        results = self.data_loader.prepare_features(
            train_years=train_years,
            val_years=val_years,
            test_years=test_years
        )
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns = results
        self.feature_columns = feature_columns
        
        # SMOTE
        if use_smote and HAS_IMBLEARN and y_train.sum() > 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                X_train, y_train = X_train_balanced, y_train_balanced
            except Exception as e:
                print(f"SMOTE failed: {e}")
        
        # Fit
        self.model.fit(X_train, y_train)
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        def safe_predict(model, X, threshold=None):
            if X is None or X.shape[0] == 0:
                return np.array([]), np.array([]), np.array([])
            probs = model.predict_proba(X)[:, 1]
            if threshold is None:
                threshold = 0.5
            preds = (probs >= threshold).astype(int)
            return preds, probs, np.array(probs)

        _, val_proba, _ = safe_predict(self.model, X_val)
        _, test_proba, _ = safe_predict(self.model, X_test)
        
        # Threshold Optimization
        optimal_threshold = 0.5
        if optimize_threshold and y_val.sum() > 0:
            optimal_threshold = self._find_optimal_threshold(y_val, val_proba)
        
        self.threshold = optimal_threshold
        
        # Final predictions with optimal threshold
        train_pred, train_proba, _ = safe_predict(self.model, X_train, optimal_threshold)
        val_pred, val_proba, _ = safe_predict(self.model, X_val, optimal_threshold)
        test_pred, test_proba, _ = safe_predict(self.model, X_test, optimal_threshold)

        # Calculate metrics for all splits
        train_metrics = {}
        val_metrics = {}
        test_metrics = {}
        
        if train_pred.size and len(y_train) > 0:
            train_metrics['accuracy'] = accuracy_score(y_train, train_pred)
            train_metrics['roc_auc'] = roc_auc_score(y_train, train_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_train, train_pred, average='binary')
            train_metrics['precision'] = precision
            train_metrics['recall'] = recall
            train_metrics['f1'] = f1
        
        if val_pred.size and len(y_val) > 0:
            val_metrics['accuracy'] = accuracy_score(y_val, val_pred)
            val_metrics['roc_auc'] = roc_auc_score(y_val, val_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='binary')
            val_metrics['precision'] = precision
            val_metrics['recall'] = recall
            val_metrics['f1'] = f1
            
        if test_pred.size and len(y_test) > 0:
            test_metrics['accuracy'] = accuracy_score(y_test, test_pred)
            test_metrics['roc_auc'] = roc_auc_score(y_test, test_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
            test_metrics['precision'] = precision
            test_metrics['recall'] = recall
            test_metrics['f1'] = f1
            
        return train_metrics, val_metrics, test_metrics, optimal_threshold
    
    def _find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        best_threshold = 0.5
        best_score = -1
        for threshold in np.arange(0.1, 0.91, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            if metric == 'f1':
                try:
                    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                    score = f1
                except: continue
            elif metric == 'recall':
                try:
                    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                    score = recall
                except: continue
            else:
                score = accuracy_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold

    def save_model(self, filename=None):
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        
        # Save with model_type in filename if not provided
        if filename is None:
            filename = f"f1_model_{self.model_type}_5yr_test.joblib"

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'preprocessor': self.data_loader.preprocessor,
            'feature_columns': self.data_loader.feature_columns
        }
        joblib.dump(model_data, filename)
        return f"Model saved to {filename}"

    def load_model(self, filename=None):
        # Load specific model type if filename not provided
        if filename is None:
            filename = f"f1_model_{self.model_type}_5yr_test.joblib"
            
        try:
            model_data = joblib.load(filename)
            self.model = model_data.get('model')
            self.model_type = model_data.get('model_type', 'random_forest')
            self.threshold = model_data.get('threshold', 0.5)
            self.feature_importance = model_data.get('feature_importance')
            preprocessor = model_data.get('preprocessor')
            feature_columns = model_data.get('feature_columns')
            if preprocessor is not None:
                self.data_loader.preprocessor = preprocessor
            if feature_columns is not None:
                self.data_loader.feature_columns = feature_columns
                self.feature_columns = feature_columns
            return True
        except (FileNotFoundError, OSError):
            return False

    def predict_2025_race(self, circuit_name, qualifying_results=None):
        if self.model is None:
            return None
        if self.grid_2025 is None:
            return None
        if self.data_loader.preprocessor is None:
            return None

        pred_df = self.grid_2025.copy()
        if 'grid' not in pred_df.columns:
            pred_df['grid'] = range(1, len(pred_df) + 1)
        
        if qualifying_results:
            for driver_id, position in qualifying_results.items():
                if 'driverId' in pred_df.columns:
                    pred_df.loc[pred_df['driverId'] == driver_id, 'grid'] = position

        if 'qual_position_avg' not in pred_df.columns:
            pred_df['qual_position_avg'] = pred_df['grid']

        pred_df['points_moving_avg'] = 0.0
        if self.results_2025 is not None and ('position' in self.results_2025.columns):
            driver_col = next((c for c in ['driver_name', 'driver', 'Driver'] if c in self.results_2025.columns), None)
            if driver_col:
                for idx, row in pred_df.iterrows():
                    driver_name = row.get('driver_name') or row.get('driver') or row.get('Driver')
                    if driver_name is None: continue
                    recent = self.get_driver_recent_results(driver_name)
                    if recent is None or recent.empty: continue
                    try:
                        pts = recent['position'].map(lambda x: max(26 - int(x), 0) if pd.notna(x) else 0).mean()
                        pred_df.at[idx, 'points_moving_avg'] = pts
                    except Exception:
                        pred_df.at[idx, 'points_moving_avg'] = 0.0

        if 'circuit_wins' not in pred_df.columns:
            pred_df['circuit_wins'] = 0

        pred_df['points_championship'] = pred_df.get('points_moving_avg', 0)
        pred_df['position_championship'] = pred_df['points_championship'].rank(ascending=False, method='min')

        if 'team_name' in pred_df.columns:
            constructor_stats = pred_df.groupby('team_name').agg({
                'points_moving_avg': ['mean', 'std']
            }).reset_index()
            constructor_stats.columns = ['team_name', 'constructor_points_mean', 'constructor_points_std']
            pred_df = pd.merge(pred_df, constructor_stats, on='team_name', how='left')
            pred_df['constructor_points_std'] = pred_df['constructor_points_std'].fillna(0)
            pred_df['constructor_points_mean'] = pred_df['constructor_points_mean'].fillna(0)
            pred_df['constructor_position_mean'] = pred_df['constructor_points_mean'].rank(ascending=False, method='min')
        else:
            pred_df['constructor_points_mean'] = 0
            pred_df['constructor_points_std'] = 0
            pred_df['constructor_position_mean'] = 999

        feature_columns = self.data_loader.feature_columns
        for col in feature_columns:
            if col not in pred_df.columns:
                pred_df[col] = 0 if col not in ['nationality', 'nationality_constructor', 'country'] else 'Unknown'

        pred_input = pred_df[feature_columns].copy()
        try:
            X_pred = self.data_loader.transform_with_preprocessor(pred_input)
            win_probs = self.model.predict_proba(X_pred)[:, 1]
        except Exception as e:
            print("Prediction Error:", e)
            return None

        results = pd.DataFrame({
            'Driver': pred_df.get('driver_name', pred_df.get('driver', pd.Series(range(len(pred_df))))),
            'Team': pred_df.get('team_name', pd.Series(['Unknown'] * len(pred_df))),
            'Grid': pred_df['grid'],
            'Win Probability': win_probs,
            'Championship Points': pred_df['points_championship']
        })

        return results.sort_values('Win Probability', ascending=False).reset_index(drop=True)

    def simulate_championship(self):
        if self.model is None or self.grid_2025 is None:
            return None, None, None

        championship_points = {driver: 0 for driver in self.grid_2025['driver_name']}

        if self.results_2025 is not None:
            for _, race in self.results_2025.iterrows():
                points_table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                pos = race.get('position')
                driver_name = None
                for c in ['driver_name', 'driver', 'Driver']:
                    if c in self.results_2025.columns and c in race:
                        driver_name = race[c]
                        break
                if driver_name is None: continue
                if pd.notna(pos) and int(pos) in points_table:
                    championship_points[driver_name] += points_table[int(pos)]
                if 'fastest_lap' in race and pd.notna(race['fastest_lap']):
                    championship_points[driver_name] += 1

        remaining_races = self._remaining_races_from_calendar_and_results()
        
        # Reliability factors
        team_reliability = {
            'Red Bull Racing': 1.05, 'Ferrari': 1.2, 'Mercedes': 1.15,
            'McLaren': 1.2, 'Aston Martin': 1.25, 'Alpine': 1.3,
            'Williams': 1.35, 'RB': 1.3, 'Kick Sauber': 1.35, 'Haas F1 Team': 1.4
        }
        driver_error_factor = {
            'Max Verstappen': 0.04, 'Yuki Tsunoda': 0.08, 'Charles Leclerc': 0.06,
            'Lewis Hamilton': 0.05, 'George Russell': 0.07, 'Andrea Kimi Antonelli': 0.12,
            'Lando Norris': 0.06, 'Oscar Piastri': 0.07, 'Fernando Alonso': 0.05,
            'Lance Stroll': 0.09, 'Pierre Gasly': 0.08, 'Jack Doohan': 0.11,
            'Alexander Albon': 0.08, 'Carlos Sainz': 0.07, 'Esteban Ocon': 0.08,
            'Oliver Bearman': 0.1, 'Nico Hulkenberg': 0.08, 'Gabriel Bortoleto': 0.12,
            'Liam Lawson': 0.09, 'Isack Hadjar': 0.11
        }

        race_results = []
        for race in remaining_races:
            results = self.predict_2025_race(race)
            if results is None: continue

            race_order = results.to_dict('records')

            for i in range(len(race_order)):
                if i >= len(race_order): break
                driver = race_order[i]
                team = driver.get('Team', 'Unknown')
                driver_name = driver.get('Driver', f"Driver_{i}")

                base_dnf_prob = 0.02
                position_factor = 1 + (i * 0.01)
                team_factor = team_reliability.get(team, 1.2)
                dnf_probability = base_dnf_prob * position_factor * team_factor

                if np.random.random() < dnf_probability:
                    race_order[i]['DNF'] = True
                    continue

                error_prob = driver_error_factor.get(driver_name, 0.1)
                if np.random.random() < error_prob:
                    positions_lost = np.random.randint(2, 6)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))

            # Random chaos
            if np.random.random() < 0.3:
                for _ in range(np.random.randint(1, 4)):
                    pos1 = np.random.randint(0, min(10, len(race_order)))
                    pos2 = np.random.randint(0, min(10, len(race_order)))
                    race_order[pos1], race_order[pos2] = race_order[pos2], race_order[pos1]

            for pos, driver in enumerate(race_order):
                if driver.get('DNF', False): continue
                points = 0
                if pos < 10:
                    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                    points = points_system[pos]
                    fastest_lap_prob = 0.3 if pos < 5 else 0.1
                    if np.random.random() < fastest_lap_prob:
                        points += 1
                championship_points[driver['Driver']] += points

            valid_results = [d for d in race_order if not d.get('DNF', False)]
            if len(valid_results) >= 3:
                race_results.append({
                    'Race': race,
                    'Winner': valid_results[0]['Driver'],
                    'Second': valid_results[1]['Driver'],
                    'Third': valid_results[2]['Driver']
                })

        final_standings = pd.DataFrame({
            'Driver': list(championship_points.keys()),
            'Points': list(championship_points.values())
        })

        if 'driver_name' in self.grid_2025.columns and 'team_name' in self.grid_2025.columns:
            final_standings = pd.merge(
                final_standings,
                self.grid_2025[['driver_name', 'team_name']],
                left_on='Driver',
                right_on='driver_name',
                how='left'
            ).drop('driver_name', axis=1)

        final_standings = final_standings.sort_values('Points', ascending=False).reset_index(drop=True)
        constructor_standings = final_standings.groupby('team_name', dropna=False)['Points'].sum().reset_index()
        constructor_standings = constructor_standings.sort_values('Points', ascending=False).reset_index(drop=True)

        return final_standings, constructor_standings, race_results

def main():
    st.set_page_config(page_title="F1 2025 AI Predictor", layout="wide", page_icon="F1")
    
    st.title("F1 2025 Season AI Predictor")

    # --- Sidebar for Model Selection ---
    st.sidebar.header("Model Settings")
    
    available_models = ["Random Forest", "Gradient Boosting"]
    if HAS_XGBOOST:
        available_models.append("XGBoost")
        
    model_choice = st.sidebar.selectbox("Active Model", available_models)
    
    model_map = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost"
    }
    selected_model_type = model_map[model_choice]

    # Instantiate predictor with the selected model type
    predictor = F1Predictor(model_type=selected_model_type)
    loaded = predictor.load_model() # loads file specific to this type (e.g. f1_model_xgboost.joblib)
    
    if loaded:
        st.sidebar.success(f"Model Loaded: {model_choice}")
    else:
        st.sidebar.warning(f"Not trained: {model_choice}")

    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", ["Race Predictor", "Model Comparison", "Championship Simulation", "Model Management", "Calendar"])

    # -------------------------------------------------------------------------
    # PAGE: MODEL COMPARISON (Different Approach)
    # -------------------------------------------------------------------------
    if page == "Model Comparison":
        st.header("Algorithm Comparison")
        st.markdown("Compare different machine learning approaches to justify the best model.")
        
        if st.button("Run Benchmark (Train All Models)"):
            metrics_df = []
            progress = st.progress(0)
            status = st.empty()
            
            # 1. Random Forest
            status.write("Training Random Forest...")
            p_rf = F1Predictor(model_type='random_forest')
            train_metrics, val_metrics, test_metrics, threshold = p_rf.train_model(
                model_type='random_forest',
                use_smote=True,
                optimize_threshold=True,
                train_years=(1950, 2019),
                val_years=(2020, 2020),
                test_years=(2021, 2024)
            )
            p_rf.save_model()
            m_rf = test_metrics.copy()
            m_rf['Model'] = 'Random Forest'
            m_rf['Threshold'] = threshold
            metrics_df.append(m_rf)
            progress.progress(33)
            
            # 2. Gradient Boosting
            status.write("Training Gradient Boosting...")
            p_gb = F1Predictor(model_type='gradient_boosting')
            train_metrics, val_metrics, test_metrics, threshold = p_gb.train_model(
                model_type='gradient_boosting',
                use_smote=True,
                optimize_threshold=True,
                train_years=(1950, 2019),
                val_years=(2020, 2020),
                test_years=(2021, 2024)
            )
            p_gb.save_model()
            m_gb = test_metrics.copy()
            m_gb['Model'] = 'Gradient Boosting'
            m_gb['Threshold'] = threshold
            metrics_df.append(m_gb)
            progress.progress(66)
            
            # 3. XGBoost (if available)
            if HAS_XGBOOST:
                status.write("Training XGBoost...")
                p_xgb = F1Predictor(model_type='xgboost')
                train_metrics, val_metrics, test_metrics, threshold = p_xgb.train_model(
                    model_type='xgboost',
                    use_smote=True,
                    optimize_threshold=True,
                    train_years=(1950, 2019),
                    val_years=(2020, 2020),
                    test_years=(2021, 2024)
                )
                p_xgb.save_model()
                m_xgb = test_metrics.copy()
                m_xgb['Model'] = 'XGBoost'
                m_xgb['Threshold'] = threshold
                metrics_df.append(m_xgb)
            
            progress.progress(100)
            status.success("Benchmark Complete")
            
            # Display Table
            df_metrics = pd.DataFrame(metrics_df).set_index('Model')
            st.subheader("Performance Metrics")
            st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Plot
            st.subheader("Metric Visualization")
            df_melted = df_metrics.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
            fig = px.bar(df_melted, x='Metric', y='Score', color='Model', barmode='group', title="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("✅ All models trained and saved! Select a model from the sidebar to use it.")

    # -------------------------------------------------------------------------
    # PAGE: RACE PREDICTOR
    # -------------------------------------------------------------------------
    elif page == "Race Predictor":
        st.header("Race Prediction")
        
        calendar_parsed = predictor._calendar_parsed()
        race_options = [item[0] for item in calendar_parsed]
        
        default_index = 0
        remaining_races = predictor._remaining_races_from_calendar_and_results()
        if remaining_races:
            try:
                default_index = race_options.index(remaining_races[0])
            except ValueError:
                default_index = 0

        selected_race = st.selectbox("Select Grand Prix", race_options, index=default_index)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"Predicting results for: **{selected_race}**")
            if st.button("Generate Prediction", type="primary"):
                if not loaded:
                    st.error(f"Please train the {model_choice} model first.")
                else:
                    with st.spinner("Analyzing..."):
                        results = predictor.predict_2025_race(selected_race)
                    
                    if results is not None:
                        display_df = results[['Driver', 'Team', 'Win Probability']].copy()
                        display_df['Win Probability'] = display_df['Win Probability'].map('{:.1%}'.format)
                        st.dataframe(display_df, hide_index=True)
                    else:
                        st.error("Prediction failed.")
            
            # Option to compare prediction consensus
            if st.checkbox("Compare Consensus (All Models)"):
                st.write("---")
                st.write("**Cross-Model Consensus**")
                
                for m_name, m_type in model_map.items():
                    if m_name == "XGBoost" and not HAS_XGBOOST:
                        continue
                        
                    temp_p = F1Predictor(model_type=m_type)
                    if temp_p.load_model():
                        res = temp_p.predict_2025_race(selected_race)
                        if res is not None:
                            winner = res.iloc[0]['Driver']
                            prob = res.iloc[0]['Win Probability']
                            st.metric(label=m_name, value=winner, delta=f"{prob:.1%} prob")
                    else:
                        st.caption(f"{m_name}: Not trained")

        with col2:
            if 'results' in locals() and results is not None:
                fig = px.bar(
                    results.head(10), 
                    x='Win Probability', 
                    y='Driver', 
                    orientation='h',
                    color='Win Probability',
                    title=f"Win Probability - Top 10: {selected_race}",
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE: CHAMPIONSHIP SIMULATION
    # -------------------------------------------------------------------------
    elif page == "Championship Simulation":
        st.header("Season Simulation")
        st.markdown(f"Simulating season using **{model_choice}** logic.")
        
        if st.button("Simulate Remaining Season"):
            if not loaded:
                st.error("Model not loaded.")
            else:
                with st.spinner("Simulating races..."):
                    final_drivers, final_constructors, race_logs = predictor.simulate_championship()
                
                if final_drivers is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Driver Standings")
                        st.dataframe(final_drivers, hide_index=True, use_container_width=True)
                        winner = final_drivers.iloc[0]
                        st.success(f"Predicted Champion: **{winner['Driver']}** ({winner['Points']} pts)")

                    with c2:
                        st.subheader("Constructor Standings")
                        st.dataframe(final_constructors, hide_index=True, use_container_width=True)
                    
                    fig = px.bar(final_drivers.head(10), x='Driver', y='Points', color='Team', title="Projected Final Points")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("View Race-by-Race Simulation Logs"):
                        st.write(race_logs)
                else:
                    st.error("Simulation failed.")

    # -------------------------------------------------------------------------
    # PAGE: MODEL MANAGEMENT
    # -------------------------------------------------------------------------
    elif page == "Model Management":
        st.header("Model Training & Status")
        st.write(f"Active Model: **{model_choice}**")
        st.write(f"Status: **{'Loaded' if loaded else 'Not Trained'}**")
        
        if st.button(f"Train {model_choice}"):
            with st.spinner(f"Training {model_choice}..."):
                try:
                    train_metrics, val_metrics, test_metrics, threshold = predictor.train_model(
                        model_type=selected_model_type,
                        use_smote=True,
                        optimize_threshold=True,
                        train_years=(1950, 2019),
                        val_years=(2020, 2020),
                        test_years=(2021, 2024)
                    )
                    st.success("Training Complete!")
                    st.write("### Test Metrics")
                    st.json(test_metrics)
                    st.write(f"**Optimal Threshold**: {threshold:.2f}")
                    predictor.save_model()
                    st.write("Model saved to disk. Go to Race Predictor to generate predictions.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        if predictor.feature_importance is not None:
            st.subheader("Feature Importance")
            fig = px.bar(
                predictor.feature_importance.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title="Top Predictive Features"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig)

    # -------------------------------------------------------------------------
    # PAGE: CALENDAR
    # -------------------------------------------------------------------------
    elif page == "Calendar":
        st.header("2025 Race Calendar")
        calendar_data = []
        for entry in F1_CALENDAR_2025:
            name, date_obj, completed = parse_calendar_entry(entry)
            status = "Completed" if completed else ("Upcoming" if date_obj and date_obj >= datetime.now().date() else "Next")
            calendar_data.append({"Race": name, "Date": date_obj, "Status": status})
        
        st.dataframe(pd.DataFrame(calendar_data), use_container_width=True)

if __name__ == "__main__":
    main()