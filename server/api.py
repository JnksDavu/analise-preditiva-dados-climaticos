from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/4174560.csv")
ARTIFACT_DIR = Path("server/artifacts")
HW_MODEL_FILE = ARTIFACT_DIR / "holtwinters_model.pkl"
SARIMA_MODEL_FILE = ARTIFACT_DIR / "sarima_model.pkl"
META_FILE = ARTIFACT_DIR / "meta.pkl"
FORECAST_HORIZON_DEFAULT = 14

# Import treino para fallback automático
try:
    from server.train_and_eval import main as auto_train
except ImportError:
    from train_and_eval import main as auto_train

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    for col in ["PRCP", "TAVG", "TMAX", "TMIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TAVG"])
    agg = (df.groupby("DATE")
           .agg({"TAVG": "mean"})
           .reset_index()
           .sort_values("DATE"))
    return agg.reset_index(drop=True)

def ensure_artifacts():
    needed = [HW_MODEL_FILE, SARIMA_MODEL_FILE, META_FILE]
    if not all(p.exists() for p in needed):
        auto_train()
    if not all(p.exists() for p in needed):
        raise RuntimeError("Falha ao gerar modelos Holt-Winters/SARIMA.")

def forecast_both(days: int):
    ensure_artifacts()
    hw_model = load_pickle(HW_MODEL_FILE)
    sarima_model = load_pickle(SARIMA_MODEL_FILE)
    meta = load_pickle(META_FILE)
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)
    y = pd.Series(df["TAVG"].values, index=df["DATE"]).asfreq("D").interpolate(limit_direction="both")
    last_date = y.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")

    hw_forecast = hw_model.forecast(days)
    sarima_forecast = sarima_model.get_forecast(days).predicted_mean
    hw_forecast.index = future_dates
    sarima_forecast.index = future_dates

    registros = []
    for dt in future_dates:
        registros.append({
            "data": str(dt.date()),
            "temperatura_prevista_holt_winters": float(hw_forecast.loc[dt]),
            "temperatura_prevista_sarima": float(sarima_forecast.loc[dt])
        })
    return registros, meta

app = FastAPI(title="API Previsão Climática - SARIMA vs Holt-Winters")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(days: int = Query(FORECAST_HORIZON_DEFAULT, ge=1, le=60)):
    previsoes, meta = forecast_both(days)
    return {
        "dias": days,
        "previsoes": previsoes,
        "metricas": meta.get("metricas", {}),
        "info": {
            "tamanho_dataset": meta.get("dataset_tamanho"),
            "tamanho_treino": meta.get("tamanho_treino"),
            "tamanho_teste": meta.get("tamanho_teste"),
            "ultima_data": meta.get("ultima_data"),
            "periodicidade": meta.get("periodicidade"),
            "sazonalidade": meta.get("sazonalidade")
        }
    }

if __name__ == "__main__":
    import uvicorn
    ensure_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=8000)