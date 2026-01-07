import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tpot import TPOTRegressor
import joblib

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
SEED = 42
GENERATIONS = 4          # increase for a more thorough search
POP_SIZE = 30            # larger population → more diversity
DATA_FILE = "nigeria_economic_data.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "tpot_tax_model.pkl")

# --------------------------------------------------------------
# 1️⃣  LOAD DATA
# --------------------------------------------------------------
def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found – run process_dataset.py first.")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # Feature engineering – create policy‑rate proxies if raw revenue columns exist
    if "CIT_Revenue" in df.columns and "VAT_Revenue" in df.columns:
        raw_cit = df["CIT_Revenue"] / df["Oil_Price"]
        raw_vat = df["VAT_Revenue"] / df["Oil_Price"]
        cit_scaler = 25.0 / raw_cit.mean()
        vat_scaler = 55.0 / raw_vat.mean()
        df["SME_Tax"] = raw_cit * cit_scaler
        df["VAT_Recovery"] = raw_vat * vat_scaler
    return df

# --------------------------------------------------------------
# 2️⃣  TRAIN TPOT AUTO‑ML MODEL (TIME-SERIES AWARE)
# --------------------------------------------------------------
def train_automl(df: pd.DataFrame):
    from sklearn.model_selection import TimeSeriesSplit
    
    feature_cols = [
        "Oil_Price", "SME_Tax", "VAT_Recovery", 
        "Digital_Penetration", "Remittances_USD", "Inflation_Rate"
    ]
    
    # Ensure columns exist and handle missing values
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.ffill().fillna(0)  # Forward fill for economic gaps

    X = df[feature_cols]
    y = df["GDP_Growth"]

    # CHRONOLOGICAL SPLIT: Take the most recent 20% as the test set
    # This simulates predicting future from past (no time-travel cheating)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n[AutoML] Training on data up to: {X_train.index.max()}")
    print(f"[AutoML] Testing on data from: {X_test.index.min()} to {X_test.index.max()}")

    # Use TimeSeriesSplit for internal cross-validation
    # This tells TPOT to respect chronological order during model search
    cv_strategy = TimeSeriesSplit(n_splits=5)

    print("\n[AutoML] Starting TPOT search (time-series aware)…")
    tpot = TPOTRegressor(
        generations=GENERATIONS,
        population_size=POP_SIZE,
        cv=cv_strategy,
        scoring='r2',
        verbose=2,
        random_state=SEED,
    )
    
    tpot.fit(X_train, y_train)

    # Evaluate on holdout (future) data
    y_pred = tpot.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n--› Model R² on future holdout set: {r2:.4f}")

    # Persist the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tpot.fitted_pipeline_, MODEL_PATH)
    print(f"Model pipeline saved to: {MODEL_PATH}")
    
    return tpot, r2

# --------------------------------------------------------------
# 3️⃣  MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    model, r2 = train_automl(df)
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model accuracy (R²): {r2:.4f}")
    print("\n📄 To generate the full HTML report with charts and AI summary:")
    print("   python tax_model.py --simulations 1000")
    print("="*60)
