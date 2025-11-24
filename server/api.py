from pathlib import Path
from datetime import timedelta
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Reuso dos caminhos do script de treino
DATA_PATH = Path("data/4174560.csv")
ARTIFACT_DIR = Path("server/artifacts")
TEMP_MODEL_FILE = ARTIFACT_DIR / "temperature_model.pkl"
PRECIP_MODEL_FILE = ARTIFACT_DIR / "precip_model.pkl"
META_FILE = ARTIFACT_DIR / "meta.pkl"
FORECAST_HORIZON_DEFAULT = 14

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_raw():
    return pd.read_csv(DATA_PATH)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    for col in ["PRCP", "TAVG", "TMAX", "TMIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TMAX"] = df["TMAX"].fillna(df["TAVG"])
    df["TMIN"] = df["TMIN"].fillna(df["TAVG"])
    df["PRCP_MM"] = df["PRCP"].fillna(0.0)
    df["RAIN"] = (df["PRCP_MM"] > 0).astype(int)
    agg = df.groupby("DATE").agg({
        "TAVG": "mean",
        "TMAX": "mean",
        "TMIN": "mean",
        "PRCP_MM": "mean",
        "RAIN": "max"
    }).reset_index()
    return agg.sort_values("DATE").reset_index(drop=True)

def build_features(df: pd.DataFrame):
    df = df.copy()
    df["DAYOFYEAR"] = df["DATE"].dt.dayofyear
    df["MONTH"] = df["DATE"].dt.month
    df["SIN_DAY"] = np.sin(2 * np.pi * df["DAYOFYEAR"] / 365.25)
    df["COS_DAY"] = np.cos(2 * np.pi * df["DAYOFYEAR"] / 365.25)
    df["SIN_MONTH"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["COS_MONTH"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    for lag in [1, 2, 3, 7]:
        df[f"TAVG_LAG_{lag}"] = df["TAVG"].shift(lag)
        df[f"RAIN_LAG_{lag}"] = df["RAIN"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith(("SIN_", "COS_", "TAVG_LAG_", "RAIN_LAG_"))] + ["RAIN", "PRCP_MM"]
    X = df[feature_cols]
    return X, feature_cols, df

def ensure_artifacts():
    if not TEMP_MODEL_FILE.exists() or not PRECIP_MODEL_FILE.exists() or not META_FILE.exists():
        raise RuntimeError("Modelos não encontrados. Rode: python server/train_and_eval.py")

def forecast(days: int):
    ensure_artifacts()
    temp_model = load(TEMP_MODEL_FILE)
    rain_model = load(PRECIP_MODEL_FILE)
    clean_df = clean(load_raw())
    history = clean_df.copy()
    last_date = clean_df["DATE"].max()
    future = []
    for step in range(1, days + 1):
        target = last_date + timedelta(days=step)
        X_hist, feat_cols, feat_df = build_features(history)
        if len(X_hist) == 0:
            break
        x_input = X_hist.iloc[-1:]
        tavg_pred = float(temp_model.predict(x_input)[0])
        if hasattr(rain_model, "predict_proba"):
            rain_prob = float(rain_model.predict_proba(x_input)[0][1])
        else:
            rain_prob = 0.0
        future.append({
            "DATE": target,
            "TAVG_PRED": tavg_pred,
            "RAIN_PROB": rain_prob
        })
        # adicionar linha sintética
        history = pd.concat([history, pd.DataFrame([{
            "DATE": target,
            "TAVG": tavg_pred,
            "TMAX": tavg_pred,
            "TMIN": tavg_pred,
            "PRCP_MM": 0.0,
            "RAIN": int(rain_prob >= 0.5)
        }])], ignore_index=True)
    return future

app = FastAPI(title="API Clima SP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(days: int = Query(FORECAST_HORIZON_DEFAULT, ge=1, le=60)):
    data = forecast(days)
    return {"days": days, "predictions": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)