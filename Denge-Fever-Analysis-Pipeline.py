import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

savePlotfile = True
AllCities = True


class DataManager:
    def __init__(self, train_features_path, train_labels_path, test_features_path):
        self.train_features_path = train_features_path
        self.train_labels_path = train_labels_path
        self.test_features_path = test_features_path
        self.train_df = None
        self.test_df = None

    def load(self):
        # Using placeholder paths matching standard DengAI datasets
        train_features = pd.read_csv(self.train_features_path)
        train_labels = pd.read_csv(self.train_labels_path)
        test_features = pd.read_csv(self.test_features_path)

        self.train_df = train_features.merge(
            train_labels, on=["city", "year", "weekofyear"]
        )
        self.test_df = test_features
        return self

    def get_city_data(self, city):
        city = city.lower()
        if city not in ["sj", "iq"]:
            raise ValueError("Unknown city")
        df = self.train_df[self.train_df["city"] == city].copy()
        return df, city
        
    def split_by_city(self):
        return self.train_df.groupby("city")


@dataclass
class CleanDataPipeline:
    def __init__(self, df, city=None):
        self.city = city
        self.df = df.copy()
        self.results = {}
        self.quality_report = {}
        self.scalers = {}
        self.raw_df = None

    def create_datetime(self):
    
        self.df["date"] = pd.to_datetime(
            self.df["year"].astype(str) + "-01-01"
        ) + pd.to_timedelta(
            (self.df["weekofyear"] - 1) * 7,
            unit="D"
        )
    
        # cyclical encoding
        self.df["week_sin"] = np.sin(
            2 * np.pi * self.df["weekofyear"] / 52
        )
    
        self.df["week_cos"] = np.cos(
            2 * np.pi * self.df["weekofyear"] / 52
        )
    
        return self

    def validate_data(self):
        print("\nMissing values:")
        print(self.df.isnull().sum().sort_values(ascending=False).head(20))
        print("\nDuplicate rows:")
        print(self.df.duplicated().sum())
        return self

    def data_quality_check(self):
        df = self.df
        report = {}
        report["missing"] = df.isna().sum().sort_values(ascending=False)

        numeric_cols = df.select_dtypes(include=np.number).columns
        bad_cols = []
        for col in numeric_cols:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isna().sum() > df[col].isna().sum():
                bad_cols.append(col)

        report["potential_string_issues"] = bad_cols
        report["object_columns"] = df.select_dtypes(include="object").columns.tolist()
        self.quality_report = report
        return self

    def preprocess_data(self, drop_columns=None, drop_na=False):
        df = self.df.copy()
        if drop_columns is None:
            # Keep identifiers like year/weekofyear for grouping and merge alignment
            drop_columns = ["week_start_date", "date"]

        print("\n=== PREPROCESSING DATA ===")
        df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        existing_cols = [c for c in drop_columns if c in df.columns]
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)

        before = len(df)
        if drop_na:
            df.dropna(inplace=True)
            print(f"Dropped {before - len(df)} rows with NaN")
        else:
            # Impute using median values safely
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            print("Filled missing values with median column metrics.")

        self.df = df
        print(f"Final shape: {df.shape}")
        return self

    def save_raw_data(self):
        self.raw_df = self.df.copy()
        return self

    def preview_data(self, rows=5):
        print("\n=== DATA PREVIEW ===")
        print(self.df.head(rows))
        return self

    def normalize_features(self, scaling_config, exclude_columns=None):
        df = self.df.copy()
        if exclude_columns is None:
            exclude_columns = ["total_cases", "year", "weekofyear", "city"]

        for col, method in scaling_config.items():
            if col not in df.columns or col in exclude_columns:
                continue

            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler

        self.df = df
        return self


@dataclass
class MaschineLearningPipeline:
    def __init__(self, df_train, df_test, city=None):
        self.city = city
        self.train_df = df_train.copy()
        self.test_df = df_test.copy()
        
        # FIX: Keep raw test IDs safe for submission generating step
        self.test_df_raw = df_test.copy() 
        self.results = {}
        self.scaler = None
        self.model = None
        
        # Data partitions
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test = None
        self.predictions = None

    def encode_features(self):
        # Explicitly encode structural columns keeping structural columns safe
        if "city" in self.train_df.columns:
            self.train_df = pd.get_dummies(self.train_df, columns=["city"], drop_first=False)
            self.test_df = pd.get_dummies(self.test_df, columns=["city"], drop_first=False)
            
            # Align features ensuring missing columns are cast as 0 (not NaN)
            self.train_df, self.test_df = self.train_df.align(
                self.test_df, join="left", axis=1, fill_value=0
            )
        return self

    def scale_features(self):
        self.scaler = StandardScaler()
        exclude = ["total_cases", "year", "weekofyear"]
        cols = [c for c in self.train_df.columns if c not in exclude and not c.startswith("city_")]

        self.train_df[cols] = self.scaler.fit_transform(self.train_df[cols])
        self.test_df[cols] = self.scaler.transform(self.test_df[cols])
        return self

    def create_validation_split(self, test_size=0.2):
        # Time-series splits sorted chronologically
        df = self.train_df.copy().sort_values(["year", "weekofyear"])
        split_idx = int(len(df) * (1 - test_size))

        train_split_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        self.X_train = train_split_df.drop(columns=["total_cases"])
        self.y_train = train_split_df["total_cases"]
        self.X_val = val_df.drop(columns=["total_cases"])
        self.y_val = val_df["total_cases"]

        print(f"Train split structure: {self.X_train.shape}, Validation split structure: {self.X_val.shape}")
        return self

    def prepare_xy(self, target="total_cases"):
        # If create_validation_split was not explicitly called, default to complete train set
        if self.X_train is None:
            self.X_train = self.train_df.drop(columns=[target])
            self.y_train = self.train_df[target]

        self.X_test = self.test_df.drop(columns=[target], errors="ignore").copy()
        return self

    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")
        return self

    def evaluate(self):
        # Validate metrics locally using the valid partition if available
        if self.X_val is not None and self.y_val is not None:
            val_preds = self.model.predict(self.X_val)
            mae = mean_absolute_error(self.y_val, val_preds)
            self.results["val_mae"] = mae
            print(f"Validation Set MAE Score: {mae:.3f}")
        else:
            print("Skipping evaluation: Call create_validation_split() prior to running evaluations.")
        return self

    def predict(self):
        self.predictions = self.model.predict(self.X_test)
        return self

    def create_submission(self, output_path="submission.csv"):
        # Safely align with raw test data frames
        submission = self.test_df_raw[["city", "year", "weekofyear"]].copy()
        submission["total_cases"] = np.round(self.predictions).astype(int)
        submission["total_cases"] = submission["total_cases"].clip(lower=0)

        # Fix structural empty path issues 
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        submission.to_csv(output_path, index=False)
        print(f"Submission successfully saved to {output_path}")
        return self


# --- Execution Execution Script ---

scaling_config = {
    "reanalysis_air_temp_k": "standard",
    "reanalysis_relative_humidity_percent": "standard",
    "precipitation_amt_mm": "minmax",
    "ndvi_se": "standard",
}

# Ensure your workspace paths point precisely to your datasets location
dm = DataManager(
    train_features_path="data/dengue_features_train.csv",
    train_labels_path="data/dengue_labels_train.csv",
    test_features_path="data/dengue_features_test.csv",
)

df = dm.load().train_df
dt = dm.test_df
city = None

analysis_pipeline = (
    CleanDataPipeline(df, city)
    .create_datetime()
    .validate_data()
    .data_quality_check()
    .save_raw_data()
    .preprocess_data(drop_na=False)
    .preview_data()
)

ml_pipeline = (
    MaschineLearningPipeline(analysis_pipeline.df, dt, city)
    .encode_features()
    .scale_features()
    .create_validation_split(test_size=0.2)  # Enables local validation
    .prepare_xy()
    .train_model()
    .evaluate()
    .predict()
    .create_submission()
)