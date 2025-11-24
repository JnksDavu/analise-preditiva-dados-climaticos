from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_PATH = Path("data/4174560.csv")
ARTIFACT_DIR = Path("server/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SARIMA_MODEL_FILE = ARTIFACT_DIR / "sarima_model.pkl"
HW_MODEL_FILE = ARTIFACT_DIR / "holtwinters_model.pkl"
META_FILE = ARTIFACT_DIR / "meta.pkl"

TEST_RATIO = 0.2
SEASONAL_PERIOD = 7  # semanal (dados diários)

# ---------- UTIL ----------
def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    for col in ["PRCP", "TAVG", "TMAX", "TMIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TAVG" in df.columns:
        df["TMAX"] = df.get("TMAX", df["TAVG"]).fillna(df["TAVG"])
        df["TMIN"] = df.get("TMIN", df["TAVG"]).fillna(df["TAVG"])
    df["PRCP_MM"] = df.get("PRCP", 0.0).fillna(0.0)
    agg = (df.groupby("DATE")
           .agg({"TAVG": "mean",
                 "TMAX": "mean",
                 "TMIN": "mean",
                 "PRCP_MM": "mean"})
           .reset_index()
           .sort_values("DATE"))
    agg = agg.dropna(subset=["TAVG"]).reset_index(drop=True)
    return agg

def load_series() -> pd.Series:
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)
    y = pd.Series(df["TAVG"].values, index=df["DATE"]).asfreq("D")
    y = y.interpolate(limit_direction="both")
    return y

def split_train_test(y: pd.Series, test_ratio: float) -> Tuple[pd.Series, pd.Series]:
    n = len(y)
    if n < 10:
        return y, y.iloc[0:0]
    split = int(n * (1 - test_ratio))
    split = min(max(split, 1), n - 1)
    return y.iloc[:split], y.iloc[split:]

# ---------- MODELOS ----------
def fit_holt_winters(y_train: pd.Series, seasonal_periods: int = SEASONAL_PERIOD):
    use_season = len(y_train) >= 2 * seasonal_periods
    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=("add" if use_season else None),
            seasonal_periods=(seasonal_periods if use_season else None),
            initialization_method="estimated",
        ).fit()
    except Exception:
        # Fallback simples sem sazonalidade
        model = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit()
    return model

def fit_sarima(y_train: pd.Series, seasonal_periods: int = SEASONAL_PERIOD):
    if len(y_train) < seasonal_periods + 5:
        order = (1, 1, 0)
        seasonal_order = (0, 0, 0, 0)
    else:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, seasonal_periods)
    try:
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception:
        # Fallback ARIMA simples
        model = SARIMAX(
            y_train,
            order=(1, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return model

# ---------- MÉTRICAS ----------
def calc_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

# ---------- EXECUÇÃO TREINO ----------
def main():
    y = load_series()
    y_train, y_test = split_train_test(y, TEST_RATIO)

    hw_model = fit_holt_winters(y_train)
    sarima_model = fit_sarima(y_train)

    steps = len(y_test)
    if steps > 0:
        hw_pred = hw_model.forecast(steps)
        sarima_pred = sarima_model.get_forecast(steps).predicted_mean
        hw_pred.index = y_test.index
        sarima_pred.index = y_test.index
        metrics_hw = calc_metrics(y_test, hw_pred.values)
        metrics_sarima = calc_metrics(y_test, sarima_pred.values)
    else:
        metrics_hw = {"rmse": None, "mae": None, "mape": None, "r2": None}
        metrics_sarima = {"rmse": None, "mae": None, "mape": None, "r2": None}

    save_pickle(hw_model, HW_MODEL_FILE)
    save_pickle(sarima_model, SARIMA_MODEL_FILE)
    save_pickle({
        "dataset_tamanho": int(len(y)),
        "tamanho_treino": int(len(y_train)),
        "tamanho_teste": int(len(y_test)),
        "ultima_data": str(y.index.max().date()) if len(y) else None,
        "periodicidade": "D",
        "sazonalidade": SEASONAL_PERIOD,
        "metricas": {
            "holt_winters": metrics_hw,
            "sarima": metrics_sarima
        }
    }, META_FILE)

    print("Treino concluído.")
    print("Métricas Holt-Winters:", metrics_hw)
    print("Métricas SARIMA:", metrics_sarima)
    print("Arquivos gerados:")
    for p in [HW_MODEL_FILE, SARIMA_MODEL_FILE, META_FILE]:
        print(" -", p, "OK" if p.exists() else "FALHOU")

if __name__ == "__main__":
    main()