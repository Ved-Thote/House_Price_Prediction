"""
=============================================================================
train_model.py — House Price Prediction System
=============================================================================
Purpose : Load and preprocess the housing dataset, train a Linear Regression
          model, evaluate it, generate visualisations, and persist the trained
          artefacts (model + encoders + scaler) to disk with Pickle.

Author  : College ML Project
Python  : 3.8+
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import pickle
import warnings

# ── Third-Party Libraries ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION — centralise all file paths in one place
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "housing.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "trained_model.pkl")

# Ensure the models/ directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# 2. DATA LOADING
# =============================================================================

class DataLoader:
    """Responsible for reading the raw CSV dataset."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """Read CSV and return a DataFrame; raise if file is missing."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found at: {self.filepath}\n"
                "Please place housing.csv inside the dataset/ folder."
            )
        df = pd.read_csv(self.filepath)
        print(f"[INFO] Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")
        return df


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """
    Handles all cleaning and feature-engineering steps:
      • Remove duplicates
      • Fill missing values
      • Encode categorical columns
      • Scale numeric features
    """

    def __init__(self):
        # Encoders are fitted here so the GUI can reuse the exact same mapping
        self.location_encoder   = LabelEncoder()
        self.furnishing_encoder = LabelEncoder()
        self.scaler             = StandardScaler()

        # Categorical columns this preprocessor handles
        self.cat_cols  = ["Location", "Furnishing"]
        # Numeric feature columns (excluding the target)
        self.num_cols  = ["Area", "Bedrooms", "Bathrooms", "Floors",
                          "Parking", "HouseAge"]

    # ── public method ────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """
        Clean, encode, and scale the full DataFrame.
        Returns (X_scaled, y, feature_names).
        """
        df = df.copy()

        # 3a. Remove duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        print(f"[INFO] Duplicates removed : {before - len(df)}")

        # 3b. Fill missing numeric values with column median
        for col in self.num_cols + ["Price"]:
            if col in df.columns:
                median_val = df[col].median()
                missing    = df[col].isna().sum()
                if missing:
                    df[col].fillna(median_val, inplace=True)
                    print(f"[INFO] Filled {missing} NaN(s) in '{col}' with median {median_val:.0f}")

        # Fill missing categorical values with mode
        for col in self.cat_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                missing  = df[col].isna().sum()
                if missing:
                    df[col].fillna(mode_val, inplace=True)
                    print(f"[INFO] Filled {missing} NaN(s) in '{col}' with mode '{mode_val}'")

        # 3c. Label-encode categorical columns
        df["Location_enc"]   = self.location_encoder.fit_transform(df["Location"])
        df["Furnishing_enc"] = self.furnishing_encoder.fit_transform(df["Furnishing"])

        # 3d. Assemble feature matrix
        feature_cols = self.num_cols + ["Location_enc", "Furnishing_enc"]
        X = df[feature_cols].values
        y = df["Price"].values

        # 3e. Standard-scale features (mean=0, std=1)
        X_scaled = self.scaler.fit_transform(X)

        print(f"[INFO] Features after preprocessing : {feature_cols}")
        print(f"[INFO] Final dataset shape          : X={X_scaled.shape}, y={y.shape}")
        return X_scaled, y, feature_cols, df

    def transform_single(self, input_dict: dict) -> np.ndarray:
        """
        Transform a single user-supplied record (from the GUI) using the
        already-fitted encoders and scaler.
        """
        location_enc   = self.location_encoder.transform([input_dict["Location"]])[0]
        furnishing_enc = self.furnishing_encoder.transform([input_dict["Furnishing"]])[0]

        row = np.array([[
            input_dict["Area"],
            input_dict["Bedrooms"],
            input_dict["Bathrooms"],
            input_dict["Floors"],
            input_dict["Parking"],
            input_dict["HouseAge"],
            location_enc,
            furnishing_enc,
        ]])
        return self.scaler.transform(row)


# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

class ModelTrainer:
    """Trains a Linear Regression model and computes evaluation metrics."""

    def __init__(self):
        self.model    = LinearRegression()
        self.metrics  = {}

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Split into train/test sets, fit the model, and compute metrics.
        Returns (y_test, y_pred) for plotting.
        """
        # 80-20 split with a fixed random seed for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"[INFO] Train size : {len(X_train)} | Test size : {len(X_test)}")

        # Fit the model
        self.model.fit(X_train, y_train)

        # Generate predictions
        y_pred = self.model.predict(X_test)

        # Compute metrics
        self.metrics = {
            "R² Score" : r2_score(y_test, y_pred),
            "MAE"      : mean_absolute_error(y_test, y_pred),
            "MSE"      : mean_squared_error(y_test, y_pred),
            "RMSE"     : np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        print("\n" + "=" * 50)
        print("         MODEL EVALUATION METRICS")
        print("=" * 50)
        for name, value in self.metrics.items():
            if name == "R² Score":
                print(f"  {name:<12} : {value:.4f}")
            else:
                print(f"  {name:<12} : ₹ {value:,.0f}")
        print("=" * 50 + "\n")

        return y_test, y_pred


# =============================================================================
# 5. VISUALISATION
# =============================================================================

class Visualizer:
    """Generates and saves all project charts."""

    CHART_DIR = os.path.join(BASE_DIR, "charts")

    def __init__(self):
        os.makedirs(self.CHART_DIR, exist_ok=True)
        # Consistent colour palette
        sns.set_theme(style="whitegrid", palette="muted")

    def _save(self, filename: str):
        path = os.path.join(self.CHART_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[INFO] Chart saved → {path}")

    # ── individual charts ────────────────────────────────────────────────────

    def correlation_heatmap(self, df: pd.DataFrame):
        """Heatmap of feature correlations with the target Price."""
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            numeric_df.corr(),
            annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5,
            annot_kws={"size": 9},
        )
        plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
        self._save("correlation_heatmap.png")

    def price_distribution(self, df: pd.DataFrame):
        """Histogram + KDE of house prices."""
        plt.figure(figsize=(9, 5))
        sns.histplot(df["Price"] / 1_000_000, kde=True, bins=20, color="#4C72B0")
        plt.xlabel("Price (₹ Millions)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.title("House Price Distribution", fontsize=14, fontweight="bold")
        self._save("price_distribution.png")

    def area_vs_price(self, df: pd.DataFrame):
        """Scatter plot of Area vs Price coloured by Location."""
        plt.figure(figsize=(9, 5))
        locations = df["Location"].unique()
        colours   = sns.color_palette("Set2", len(locations))
        for loc, col in zip(locations, colours):
            subset = df[df["Location"] == loc]
            plt.scatter(
                subset["Area"], subset["Price"] / 1_000_000,
                label=loc, color=col, alpha=0.7, edgecolors="white", s=60,
            )
        plt.xlabel("Area (sq ft)", fontsize=11)
        plt.ylabel("Price (₹ Millions)", fontsize=11)
        plt.title("Area vs House Price", fontsize=14, fontweight="bold")
        plt.legend(title="Location")
        self._save("area_vs_price.png")

    def actual_vs_predicted(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Scatter of actual vs predicted prices with a perfect-fit line."""
        plt.figure(figsize=(8, 6))
        plt.scatter(
            y_test / 1_000_000, y_pred / 1_000_000,
            alpha=0.65, color="#DD8452", edgecolors="white", s=70, label="Predictions",
        )
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min()) / 1_000_000
        max_val = max(y_test.max(), y_pred.max()) / 1_000_000
        plt.plot([min_val, max_val], [min_val, max_val],
                 "b--", linewidth=1.5, label="Perfect Fit")
        plt.xlabel("Actual Price (₹ Millions)", fontsize=11)
        plt.ylabel("Predicted Price (₹ Millions)", fontsize=11)
        plt.title("Actual vs Predicted House Prices", fontsize=14, fontweight="bold")
        plt.legend()
        self._save("actual_vs_predicted.png")

    def generate_all(self, df: pd.DataFrame, y_test, y_pred):
        """Convenience method — generate every chart in one call."""
        self.correlation_heatmap(df)
        self.price_distribution(df)
        self.area_vs_price(df)
        self.actual_vs_predicted(y_test, y_pred)


# =============================================================================
# 6. MODEL PERSISTENCE
# =============================================================================

class _TrainModelUnpickler(pickle.Unpickler):
    """Custom unpickler that maps script-defined __main__ classes back to train_model."""

    def find_class(self, module, name):
        if module == "__main__" and name in {
            "DataLoader", "DataPreprocessor", "ModelTrainer",
            "Visualizer", "ModelSerializer",
        }:
            module = "train_model"
        return super().find_class(module, name)


class ModelSerializer:
    """Save and load trained model artefacts using Pickle."""

    @staticmethod
    def save(path: str, payload: dict):
        """Persist model + preprocessor to a .pkl file."""
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Model saved → {path}")

    @staticmethod
    def load(path: str) -> dict:
        """Load and return the artefact dictionary from a .pkl file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model found at: {path}\nRun train_model.py first.")
        with open(path, "rb") as f:
            payload = _TrainModelUnpickler(f).load()
        print(f"[INFO] Model loaded from → {path}")
        return payload


# =============================================================================
# 7. PIPELINE ORCHESTRATION
# =============================================================================

def print_dataset_statistics(df: pd.DataFrame):
    """Print a tidy summary of the raw dataset."""
    print("\n" + "=" * 60)
    print("              DATASET STATISTICS")
    print("=" * 60)
    print(df.describe().to_string())
    print("=" * 60 + "\n")


def run_training_pipeline():
    """
    Full end-to-end pipeline:
      load → statistics → preprocess → train → evaluate → visualise → save
    """
    print("\n" + "=" * 60)
    print("    HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1 — Load data
    loader = DataLoader(DATASET_PATH)
    df_raw = loader.load()

    # Step 2 — Display statistics
    print_dataset_statistics(df_raw)

    # Step 3 — Preprocess
    preprocessor = DataPreprocessor()
    X, y, feature_names, df_clean = preprocessor.fit_transform(df_raw)

    # Step 4 — Train & evaluate
    trainer = ModelTrainer()
    y_test, y_pred = trainer.train(X, y)

    # Step 5 — Visualise
    viz = Visualizer()
    viz.generate_all(df_clean, y_test, y_pred)

    # Step 6 — Persist artefacts
    payload = {
        "model"              : trainer.model,
        "preprocessor"       : preprocessor,
        "feature_names"      : feature_names,
        "metrics"            : trainer.metrics,
        "location_classes"   : list(preprocessor.location_encoder.classes_),
        "furnishing_classes" : list(preprocessor.furnishing_encoder.classes_),
    }
    ModelSerializer.save(MODEL_PATH, payload)

    print("[SUCCESS] Training pipeline complete!")
    return payload


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_training_pipeline()
