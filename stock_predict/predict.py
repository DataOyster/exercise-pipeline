# Logistic Regression model inference with stock_data.csv and save results locally
# Versione per test locale con modello .pkl (scikit-learn)

import os
import joblib
import pandas as pd
from datetime import datetime

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "models/models_logreg_model.pkl"   # local pickle model
DATA_PATH  = "data/stock_data.csv"              # local CSV data
LOCAL_CSV_PATH = "outputs/predictions.csv"

# ---------------- FUNCTIONS ----------------
def load_model(model_path: str):
    """Load scikit-learn model from pickle file."""
    model = joblib.load(model_path)
    return model

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV from local path into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola la feature 'change' come differenza close - open."""
    df["change"] = df["close"] - df["open"]
    return df

def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predice usando la feature 'change'."""
    df = prepare_features(df)

    X = df[["change"]]   # usa SOLO la colonna 'change'

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["prediction"] = preds
    out["probability"] = probs
    out["load_date"] = datetime.utcnow().date()
    return out

def save_to_csv(df: pd.DataFrame, path: str):
    """Save predictions as local CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Saved local CSV -> {path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print(f"Loading Pickle model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"Loading CSV data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Rows fetched: {len(df)}")

    print("Running predictions with .pkl model...")
    result_df = predict_df(model, df)

    # Save results
    save_to_csv(result_df, LOCAL_CSV_PATH)

    print("Done ✅")
