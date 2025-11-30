import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class F1DataLoaderFixed:
    """
    Loads F1 CSVs from a folder, builds time-safe features (no leakage),
    and returns train/val/test (X, y) plus a fitted preprocessing pipeline
    that can be applied at inference time.
    """

    def __init__(self, data_path='f1data'):
        self.data_path = data_path
        self.data = {}
        self.preprocessor = None
        self.feature_columns = None

    def load_all_data(self):
        files = [
            'races', 'results', 'drivers', 'constructors',
            'qualifying', 'circuits', 'driver_standings',
            'constructor_standings'
        ]

        for f in files:
            path = f'{self.data_path}/{f}.csv'
            self.data[f] = pd.read_csv(path)

        # Coerce important numeric types
        if 'points' in self.data['results'].columns:
            self.data['results']['points'] = pd.to_numeric(self.data['results']['points'], errors='coerce')
        if 'position' in self.data['results'].columns:
            self.data['results']['position'] = pd.to_numeric(self.data['results']['position'], errors='coerce')

        return self.data

    def prepare_race_data(self):
        # merge results + basic race info
        df = pd.merge(
            self.data['results'],
            self.data['races'][['raceId', 'year', 'round', 'circuitId']],
            on='raceId',
            how='left'
        )

        # driver nationality
        df = pd.merge(
            df,
            self.data['drivers'][['driverId', 'nationality']],
            on='driverId',
            how='left',
            suffixes=(None, '_driver')
        )

        # constructor nationality
        df = pd.merge(
            df,
            self.data['constructors'][['constructorId', 'nationality']],
            on='constructorId',
            how='left',
            suffixes=(None, '_constructor')
        )
        # rename to avoid duplicate column names confusion
        if 'nationality_constructor' not in df.columns and 'nationality_constructor' not in df.columns:
            # after the merge the constructor nationality will be in 'nationality' (conflict resolved by suffixes)
            if 'nationality_constructor' not in df.columns and 'nationality_constructor' not in df.columns:
                # attempt to rename the constructor nationality if it exists with suffix
                for c in df.columns:
                    if c.endswith('_constructor'):
                        df = df.rename(columns={c: 'nationality_constructor'})

        # circuits
        circuit_columns = ['circuitId', 'name', 'location', 'country']
        df = pd.merge(
            df,
            self.data['circuits'][circuit_columns],
            on='circuitId',
            how='left',
            suffixes=('', '_circuit')
        )

        # Ensure numeric grid, points and position exist as numeric
        for col in ['grid', 'points', 'position']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by raceId so temporal transforms are correct
        df = df.sort_values(['driverId', 'raceId']).reset_index(drop=True)

        return df

    def add_features(self, df):
        # 1) Recent performance: rolling mean of points over last 3 *past* races (no leakage)
        df['points_moving_avg'] = df.groupby('driverId')['points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )

        # 2) Qualifying performance: compute rolling mean of qualifying *past* positions
        qual = pd.merge(
            self.data['qualifying'],
            self.data['races'][['raceId', 'year']],
            on='raceId',
            how='left'
        )
        if 'position' in qual.columns:
            qual['position'] = pd.to_numeric(qual['position'], errors='coerce')
            qual = qual.sort_values(['driverId', 'raceId'])
            qual['qual_position_avg'] = qual.groupby('driverId')['position'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
            )
            # merge the qual_position_avg (may be NaN for first races)
            df = pd.merge(
                df,
                qual[['raceId', 'driverId', 'qual_position_avg']],
                on=['raceId', 'driverId'],
                how='left'
            )
        else:
            df['qual_position_avg'] = np.nan

        # 3) Circuit-specific wins up to (but not including) current race
        df['circuit_wins'] = df.sort_values('raceId').groupby(['driverId', 'circuitId'])['position'].transform(
            lambda x: (x == 1).cumsum().shift(1).fillna(0)
        )

        # 4) Championship points/position from driver_standings:
        #    shift by 1 so standings reflect state BEFORE the current race (no leakage)
        if 'driver_standings' in self.data:
            ds = self.data['driver_standings'].copy()
            # ensure numeric
            for c in ['points', 'position']:
                if c in ds.columns:
                    ds[c] = pd.to_numeric(ds[c], errors='coerce')
            ds = ds.sort_values(['driverId', 'raceId'])
            ds[['points_championship', 'position_championship']] = ds.groupby('driverId')[['points', 'position']].shift(1)
            ds = ds[['raceId', 'driverId', 'points_championship', 'position_championship']]
            df = pd.merge(df, ds, on=['raceId', 'driverId'], how='left')
        else:
            df['points_championship'] = np.nan
            df['position_championship'] = np.nan

        # 5) Constructor historical stats (expanding mean/std/mean position) computed up to previous race
        # Compute expanding stats grouped by constructorId on the driver-level df
        df = df.sort_values(['constructorId', 'raceId'])
        df['constructor_points_mean'] = df.groupby('constructorId')['points'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['constructor_points_std'] = df.groupby('constructorId')['points'].transform(
            lambda x: x.expanding().std().shift(1)
        )
        df['constructor_position_mean'] = df.groupby('constructorId')['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )

        # Replace NaNs in constructor std with 0 (std undefined for single obs)
        df['constructor_points_std'] = df['constructor_points_std'].fillna(0)

        # If still NaN for means (e.g., first-ever race for constructor), fill with sensible defaults:
        # - mean points: 0
        # - position mean: max position seen so far (or large number)
        df['constructor_points_mean'] = df['constructor_points_mean'].fillna(0)
        df['constructor_position_mean'] = df['constructor_position_mean'].fillna(df['position'].max() if 'position' in df.columns else 999)

        # final safety numeric coercions
        for col in ['points_moving_avg', 'qual_position_avg', 'circuit_wins',
                    'points_championship', 'position_championship',
                    'constructor_points_mean', 'constructor_points_std', 'constructor_position_mean']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def build_and_fit_preprocessor(self, X_train, numeric_cols, categorical_cols):
        """
        Build a ColumnTransformer pipeline:
        - numeric: SimpleImputer(mean) + StandardScaler
        - categorical: SimpleImputer(constant 'Unknown') + OrdinalEncoder(handle_unknown -> -1)
        """
        # numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # categorical pipeline
        # OrdinalEncoder with unknown_value requires (scikit-learn >= 0.24). unknown_value=-1 treats unseen categories as -1.
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ], remainder='drop', sparse_threshold=0)

        preprocessor.fit(X_train)
        self.preprocessor = preprocessor
        return preprocessor

    def transform_with_preprocessor(self, X):
        """
        Apply fitted preprocessor and return a DataFrame with same feature column order.
        """
        arr = self.preprocessor.transform(X)
        # Build column names: numeric_cols scaled + categorical_cols encoded
        # extract names from transformer spec
        num_cols = [name for name in self.preprocessor.transformers_[0][2]]
        cat_cols = [name for name in self.preprocessor.transformers_[1][2]]
        out_cols = num_cols + cat_cols
        return pd.DataFrame(arr, columns=out_cols, index=X.index)

    def prepare_features(self):
        """
        Main entry point: loads data, creates features, splits by year, fits preprocessing
        and returns (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns)
        """
        self.load_all_data()
        df = self.prepare_race_data()
        df = self.add_features(df)

        # target: winner (position == 1) -- position is from results table and is numeric
        df['winner'] = (df['position'] == 1).astype(int)

        # Desired feature list (keep same order for preprocessor)
        desired_features = [
            'grid',
            'qual_position_avg',
            'points_moving_avg',
            'circuit_wins',
            'points_championship',
            'position_championship',
            'constructor_points_mean',
            'constructor_points_std',
            'constructor_position_mean',
            'nationality',
            'nationality_constructor',
            'country'
        ]

        # Warn about missing columns and build final list containing only available columns
        available = []
        for c in desired_features:
            if c not in df.columns:
                # don't raise — just warn via print (keeps class non-fatal for different CSV schemas)
                print(f"Warning: Column '{c}' not found; it will be excluded from features.")
            else:
                available.append(c)

        self.feature_columns = available.copy()

        # Split by year — keep this logic but check existence of 'year'
        if 'year' not in df.columns:
            raise ValueError("Column 'year' required for temporal split but not present in data.")
        train_df = df[df['year'] <= 2022].copy()
        val_df = df[df['year'] == 2023].copy()
        test_df = df[df['year'] == 2024].copy()

        # If any of the splits are empty, still proceed but warn
        if train_df.empty:
            print("Warning: Training split is empty (no rows with year <= 2022).")
        if val_df.empty:
            print("Warning: Validation split is empty (no rows with year == 2023).")
        if test_df.empty:
            print("Warning: Test split is empty (no rows with year == 2024).")

        # Prepare X and y for each split (select only available features)
        X_train = train_df[self.feature_columns].copy()
        y_train = train_df['winner'].copy()
        X_val = val_df[self.feature_columns].copy()
        y_val = val_df['winner'].copy()
        X_test = test_df[self.feature_columns].copy()
        y_test = test_df['winner'].copy()

        # Determine numeric vs categorical columns inside available features
        # Treat these as categorical: nationality, nationality_constructor, country (if present)
        categorical_candidates = ['nationality', 'nationality_constructor', 'country']
        categorical_cols = [c for c in categorical_candidates if c in self.feature_columns]
        numeric_cols = [c for c in self.feature_columns if c not in categorical_cols]

        # Fit preprocessor on training features only (prevents leakage)
        if not X_train.empty:
            self.build_and_fit_preprocessor(X_train, numeric_cols, categorical_cols)

            # transform datasets and keep DataFrame form
            X_train_t = self.transform_with_preprocessor(X_train)
            X_val_t = self.transform_with_preprocessor(X_val) if not X_val.empty else pd.DataFrame(columns=X_train_t.columns)
            X_test_t = self.transform_with_preprocessor(X_test) if not X_test.empty else pd.DataFrame(columns=X_train_t.columns)
        else:
            # If no training data, fit preprocessor on concatenation to avoid crashes (but this is not ideal)
            combined = pd.concat([X_train, X_val, X_test], axis=0)
            self.build_and_fit_preprocessor(combined, numeric_cols, categorical_cols)
            X_train_t = self.transform_with_preprocessor(X_train)
            X_val_t = self.transform_with_preprocessor(X_val)
            X_test_t = self.transform_with_preprocessor(X_test)

        # Return transformed features (as DataFrames), labels and the fitted preprocessor
        return (
            X_train_t, y_train,
            X_val_t, y_val,
            X_test_t, y_test,
            self.preprocessor,
            self.feature_columns
        )
