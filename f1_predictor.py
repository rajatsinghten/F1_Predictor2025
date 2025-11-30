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

# helper: parse calendar string -> (race_name, date_obj or None, explicitly_completed_flag)
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
    # extract the "(DD Mon)" part
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

    # race name: take everything up to the first "(" or before " - COMPLETED"
    # prefer the left-most " - (" or just strip trailing tags
    name = entry
    # remove trailing " - COMPLETED" if present
    name = re.sub(r'\s*-\s*COMPLETED\s*$', '', name, flags=re.IGNORECASE).strip()
    # remove date parentheses portion for a cleaner name
    name = re.sub(r'\s*\(\d{1,2}\s*[A-Za-z]{3}\)\s*', '', name).strip()
    return name, date_obj, explicitly_completed

class F1Predictor:
    def __init__(self, data_path='f1data', model_type='gradient_boosting'):
        self.data_path = data_path
        self.model_type = model_type  # 'random_forest', 'gradient_boosting', or 'xgboost'
        self.model = None
        self.threshold = 0.5  # Will be optimized during training
        
        # Initialize model based on type
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=20,
                min_samples_leaf=8,
                random_state=42
            )
        elif model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_child_weight=10,
                scale_pos_weight=19,  # Approximate class weight (non-winners : winners)
                random_state=42,
                tree_method='hist',
                eval_metric='logloss'
            )
        else:
            # Default to Random Forest with class weights
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
        
        self.data_loader = F1DataLoader(data_path)
        self.feature_importance = None
        self.grid_2025 = None
        self.results_2025 = None
        self.feature_columns = None
        self.scaler_stats = None  # Store for threshold optimization
        self.load_2025_data()
        try:
            self.load_model()
        except Exception:
            pass

    def load_2025_data(self):
        """Load 2025 CSVs and build a set of completed races from actual results."""
        try:
            self.grid_2025 = pd.read_csv(f'{self.data_path}/f1_2025_grid.csv')
        except Exception as e:
            print(f"Could not load 2025 grid: {e}")
            self.grid_2025 = None

        try:
            self.results_2025 = pd.read_csv(f'{self.data_path}/f1_2025_results.csv')
            # normalize race name column detection
            # common possibilities: 'race_name', 'race', 'raceName'
            possible_cols = ['race_name', 'race', 'raceName', 'raceName', 'round']
            self.results_2025.columns = [c.strip() for c in self.results_2025.columns]
            # try to parse dates if there is a date column
            if 'date' in self.results_2025.columns:
                try:
                    self.results_2025['date'] = pd.to_datetime(self.results_2025['date'], errors='coerce')
                except Exception:
                    pass
        except Exception as e:
            print(f"Could not load 2025 results: {e}")
            self.results_2025 = None

        # Build a set of race names that have results (be forgiving about column names)
        self.completed_races_set = set()
        if self.results_2025 is not None:
            # check for candidate race name columns
            for name_col in ['race_name', 'race', 'raceName', 'raceName', 'round']:
                if name_col in self.results_2025.columns:
                    # add normalized lowercase names
                    vals = self.results_2025[name_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
                    self.completed_races_set.update(vals)
                    # don't break — collect from all possible columns (some datasets have different naming)
        # final: lower-case set for matching
        self.completed_races_set = set([s.lower() for s in self.completed_races_set])

    def get_driver_recent_results(self, driver_name, n=5):
        """Get recent results for a driver in 2025 (most recent first)."""
        if self.results_2025 is None:
            return None

        # best-effort column names for driver
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
        """Return list of parsed calendar entries: (race_name, date_obj, explicitly_completed)"""
        parsed = []
        for entry in F1_CALENDAR_2025:
            parsed.append(parse_calendar_entry(entry, year=2025))
        return parsed

    def _remaining_races_from_calendar_and_results(self):
        """
        Compute remaining races using:
          - explicit 'COMPLETED' label in calendar entries
          - actual results present in results_2025
          - calendar date compared to "today" (Asia/Kolkata timezone)
        Returns a list of race names (strings).
        """
        # use Asia/Kolkata timezone and today's date
        tz = pytz.timezone('Asia/Kolkata')
        today = datetime.now(tz).date()

        parsed = self._calendar_parsed()
        remaining = []
        for race_name, race_date, explicitly_completed in parsed:
            rn_lower = race_name.strip().lower()
            # if calendar explicitly says completed, treat as done
            if explicitly_completed:
                continue
            # if results file contains this race, it's completed
            if rn_lower in self.completed_races_set:
                continue
            # if race date exists and is in the past (before today), treat as completed
            if race_date is not None and race_date < today:
                # If race date has passed but we have no results, still treat as completed to avoid simulating past events.
                # (You may want to change behavior if you want to simulate historical races.)
                continue
            # else consider it remaining/upcoming
            remaining.append(race_name)
        return remaining

    def predict_2025_race(self, circuit_name, qualifying_results=None):
        """
        Predict the outcome of a 2025 race.

        Uses the data_loader.preprocessor and feature_columns to produce
        the correct model input. Returns a DataFrame of drivers with win
        probabilities sorted descending.
        """
        if self.model is None:
            print("Model not loaded or trained.")
            return None
        if self.grid_2025 is None:
            print("2025 grid not available.")
            return None
        if self.data_loader.preprocessor is None or not self.data_loader.feature_columns:
            print("Preprocessor or feature columns not available. Train the model first.")
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
            # simplified points calc — consistent with training approximation (if any)
            driver_col = next((c for c in ['driver_name', 'driver', 'Driver'] if c in self.results_2025.columns), None)
            if driver_col:
                for idx, row in pred_df.iterrows():
                    driver_name = row.get('driver_name') or row.get('driver') or row.get('Driver')
                    if driver_name is None:
                        continue
                    recent = self.get_driver_recent_results(driver_name)
                    if recent is None or recent.empty:
                        continue
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
        if not feature_columns:
            print("No feature columns available from data loader.")
            return None

        for col in feature_columns:
            if col not in pred_df.columns:
                pred_df[col] = 0 if col not in ['nationality', 'nationality_constructor', 'country'] else 'Unknown'

        pred_input = pred_df[feature_columns].copy()

        try:
            X_pred = self.data_loader.transform_with_preprocessor(pred_input)
        except Exception as e:
            print("Error transforming prediction input:", e)
            return None

        try:
            win_probs = self.model.predict_proba(X_pred)[:, 1]
        except Exception as e:
            print("Error during model prediction:", e)
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
        """Simulate the remaining races of the 2025 championship."""
        if self.model is None or self.grid_2025 is None:
            print("Model or 2025 grid not available.")
            return None, None, None

        championship_points = {driver: 0 for driver in self.grid_2025['driver_name']}

        if self.results_2025 is not None:
            for _, race in self.results_2025.iterrows():
                points_table = {
                    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
                }
                pos = race.get('position')
                driver_name = None
                for c in ['driver_name', 'driver', 'Driver']:
                    if c in self.results_2025.columns and c in race:
                        driver_name = race[c]
                        break
                if driver_name is None:
                    continue
                if pd.notna(pos) and int(pos) in points_table:
                    championship_points[driver_name] += points_table[int(pos)]
                if 'fastest_lap' in race and pd.notna(race['fastest_lap']):
                    championship_points[driver_name] += 1

        # Get remaining races by combining calendar + actual results + dates
        remaining_races = self._remaining_races_from_calendar_and_results()

        # Same reliability and error factors (unchanged)
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
            if results is None:
                continue

            race_order = results.to_dict('records')

            # simulate incidents (same logic as before)
            for i in range(len(race_order)):
                if i >= len(race_order):
                    break
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

                if np.random.random() < 0.05:
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))

                if np.random.random() < 0.08:
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))

            if np.random.random() < 0.3:
                for _ in range(np.random.randint(1, 4)):
                    pos1 = np.random.randint(0, min(10, len(race_order)))
                    pos2 = np.random.randint(0, min(10, len(race_order)))
                    race_order[pos1], race_order[pos2] = race_order[pos2], race_order[pos1]

            for pos, driver in enumerate(race_order):
                if driver.get('DNF', False):
                    continue
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

    def train_model(self, use_smote=False, optimize_threshold=True):
        """
        Improved training with class imbalance handling and threshold optimization.
        
        Args:
            use_smote: Whether to apply SMOTE oversampling (requires imbalanced-learn)
            optimize_threshold: Whether to optimize decision threshold on validation set
        """
        results = self.data_loader.prepare_features()
        if len(results) < 8:
            raise RuntimeError("Unexpected return from data_loader.prepare_features()")
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns = results
        self.feature_columns = feature_columns
        
        # Apply SMOTE if requested and available
        if use_smote and HAS_IMBLEARN and y_train.sum() > 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"SMOTE applied: {len(y_train)} -> {len(y_train_balanced)} samples")
                X_train, y_train = X_train_balanced, y_train_balanced
            except Exception as e:
                print(f"SMOTE failed: {e}. Continuing without SMOTE.")
        
        # Train the model
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

        # Get predictions with default threshold
        train_pred, train_proba, _ = safe_predict(self.model, X_train)
        val_pred, val_proba, _ = safe_predict(self.model, X_val)
        test_pred, test_proba, _ = safe_predict(self.model, X_test)
        
        # Optimize threshold on validation set if requested
        optimal_threshold = 0.5
        if optimize_threshold and y_val.sum() > 0:
            optimal_threshold = self._find_optimal_threshold(y_val, val_proba)
            print(f"Optimized threshold: {optimal_threshold:.3f}")
            # Re-predict with optimal threshold
            train_pred, _, _ = safe_predict(self.model, X_train, optimal_threshold)
            val_pred, _, _ = safe_predict(self.model, X_val, optimal_threshold)
            test_pred, _, _ = safe_predict(self.model, X_test, optimal_threshold)
        
        self.threshold = optimal_threshold

        metrics = {}
        if train_pred.size:
            metrics['Train Accuracy'] = accuracy_score(y_train, train_pred)
            metrics['Train ROC AUC'] = roc_auc_score(y_train, train_proba)
        else:
            metrics['Train Accuracy'] = np.nan
            metrics['Train ROC AUC'] = np.nan

        if val_pred.size:
            metrics['Validation Accuracy'] = accuracy_score(y_val, val_pred)
            metrics['Validation ROC AUC'] = roc_auc_score(y_val, val_proba)
        else:
            metrics['Validation Accuracy'] = np.nan
            metrics['Validation ROC AUC'] = np.nan

        if test_pred.size:
            metrics['Test Accuracy'] = accuracy_score(y_test, test_pred)
            metrics['Test ROC AUC'] = roc_auc_score(y_test, test_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
            metrics['Test Precision'] = precision
            metrics['Test Recall'] = recall
            metrics['Test F1 Score'] = f1
        else:
            metrics['Test Accuracy'] = np.nan
            metrics['Test ROC AUC'] = np.nan
            metrics['Test Precision'] = np.nan
            metrics['Test Recall'] = np.nan
            metrics['Test F1 Score'] = np.nan

        return metrics
    
    def _find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        """
        Find optimal classification threshold by maximizing specified metric on validation data.
        Metrics: 'f1', 'precision', 'recall', 'balanced_accuracy'
        """
        best_threshold = 0.5
        best_score = -1
        
        for threshold in np.arange(0.1, 0.91, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                    score = f1
                except:
                    continue
            elif metric == 'recall':
                try:
                    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                    score = recall
                except:
                    continue
            else:
                score = accuracy_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

    def save_model(self, filename='f1_model.joblib'):
        if self.model is None:
            raise RuntimeError("No trained model to save.")
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

    def load_model(self, filename='f1_model.joblib'):
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
        return f"Model loaded from {filename}"

def main():
    st.set_page_config(page_title="F1 2025 AI Predictor", layout="wide", page_icon="🏎️")
    
    st.title("🏎️ F1 2025 Season AI Predictor")
    
    # Initialize the predictor (Cached to prevent reloading on every interaction)
    @st.cache_resource
    def get_predictor():
        return F1Predictor()
    
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Failed to initialize predictor. Please check data files. Error: {e}")
        return

    # Sidebar for Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", ["Race Predictor", "Championship Simulation", "Model Management", "Calendar"])

    # -------------------------------------------------------------------------
    # PAGE 1: RACE PREDICTOR
    # -------------------------------------------------------------------------
    if page == "Race Predictor":
        st.header("🔮 Race Prediction")
        
        # Parse calendar to get race names for the dropdown
        calendar_parsed = predictor._calendar_parsed()
        race_options = [item[0] for item in calendar_parsed]
        
        # Try to find the next upcoming race to set as default
        default_index = 0
        remaining_races = predictor._remaining_races_from_calendar_and_results()
        if remaining_races:
            try:
                # Find index of first remaining race in the full list
                default_index = race_options.index(remaining_races[0])
            except ValueError:
                default_index = 0

        selected_race = st.selectbox("Select Grand Prix", race_options, index=default_index)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"Predicting results for: **{selected_race}**")
            if st.button("Generate Prediction", type="primary"):
                with st.spinner("Analyzing driver stats and grid data..."):
                    results = predictor.predict_2025_race(selected_race)
                
                if results is not None:
                    # Formatting for display
                    display_df = results[['Driver', 'Team', 'Win Probability']].copy()
                    display_df['Win Probability'] = display_df['Win Probability'].map('{:.1%}'.format)
                    st.dataframe(display_df, hide_index=True)
                else:
                    st.error("Prediction failed. Ensure model is trained and grid data is available.")

        with col2:
            if 'results' in locals() and results is not None:
                # Plotly Chart
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
    # PAGE 2: CHAMPIONSHIP SIMULATION
    # -------------------------------------------------------------------------
    elif page == "Championship Simulation":
        st.header("🏆 Season Simulation")
        st.markdown("Simulate the remainder of the season based on current standings and AI predictions.")
        
        if st.button("Simulate Remaining Season"):
            with st.spinner("Simulating every lap of the remaining races..."):
                final_drivers, final_constructors, race_logs = predictor.simulate_championship()
            
            if final_drivers is not None:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Driver Standings")
                    st.dataframe(final_drivers, hide_index=True, use_container_width=True)
                    
                    # Winner Highlight
                    winner = final_drivers.iloc[0]
                    st.success(f"🏆 Predicted Champion: **{winner['Driver']}** ({winner['Points']} pts)")

                with c2:
                    st.subheader("Constructor Standings")
                    st.dataframe(final_constructors, hide_index=True, use_container_width=True)
                
                # Visualization of points
                fig = px.bar(final_drivers.head(10), x='Driver', y='Points', color='Team', title="Projected Final Points")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Race-by-Race Simulation Logs"):
                    st.write(race_logs)
            else:
                st.error("Simulation failed. Check data availability.")

    # -------------------------------------------------------------------------
    # PAGE 3: MODEL MANAGEMENT
    # -------------------------------------------------------------------------
    elif page == "Model Management":
        st.header("⚙️ Model Training & Status")
        
        st.write(f"**Model Loaded:** {'Yes' if predictor.model else 'No'}")
        
        if st.button("Train New Model"):
            with st.spinner("Training Random Forest Classifier..."):
                try:
                    metrics = predictor.train_model()
                    st.success("Training Complete!")
                    st.json(metrics)
                    predictor.save_model()
                    st.write("Model saved to disk.")
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
    # PAGE 4: CALENDAR
    # -------------------------------------------------------------------------
    elif page == "Calendar":
        st.header("📅 2025 Race Calendar")
        calendar_data = []
        for entry in F1_CALENDAR_2025:
            name, date_obj, completed = parse_calendar_entry(entry)
            status = "✅ Completed" if completed else ("📅 Upcoming" if date_obj and date_obj >= datetime.now().date() else "⏭️ Next")
            calendar_data.append({"Race": name, "Date": date_obj, "Status": status})
        
        st.dataframe(pd.DataFrame(calendar_data), use_container_width=True)

if __name__ == "__main__":
    main()