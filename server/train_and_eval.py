from pathlib import Path
from datetime import timedelta
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# ---------- CONFIG ----------
DATA_PATH = Path("data/4174560.csv")
ARTIFACT_DIR = Path("server/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_MODEL_FILE = ARTIFACT_DIR / "temperature_model.pkl"
PRECIP_MODEL_FILE = ARTIFACT_DIR / "precip_model.pkl"
META_FILE = ARTIFACT_DIR / "meta.pkl"

TEST_RATIO = 0.2

# ---------- UTIL ----------
def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- PREPROCESS ----------
def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

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
    y_temp = df["TAVG"]
    y_rain = df["RAIN"]
    return X, y_temp, y_rain, feature_cols, df

def split_time(X, y1, y2, test_ratio):
    n = len(X)
    split = int(n * (1 - test_ratio))
    return (X.iloc[:split], X.iloc[split:],
            y1.iloc[:split], y1.iloc[split:],
            y2.iloc[:split], y2.iloc[split:])

# ---------- MODELOS ----------
def train_temp(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_rain(X_train, y_train):
    if len(set(y_train)) < 2:
        prob = float(np.mean(y_train))
        class Dummy:
            def predict(self, X): return np.zeros(len(X), dtype=int)
            def predict_proba(self, X): return np.column_stack([1 - prob, np.full(len(X), prob)])
        return Dummy()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
    ])
    pipe.fit(X_train, y_train)
    return pipe

# ---------- MÉTRICAS ----------
def metrics_reg(model, X_test, y_test):
    if len(X_test) == 0:
        return {"rmse": None, "r2": None}
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)  # remover parâmetro squared para compatibilidade
    rmse = mse ** 0.5
    return {
        "rmse": rmse,
        "r2": r2_score(y_test, pred)
    }

def metrics_clf(model, X_test, y_test):
    if len(X_test) == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    }

# ---------- EXECUÇÃO TREINO ----------
def main():
    raw = load_raw(DATA_PATH)
    clean_df = clean(raw)
    X, y_temp, y_rain, feature_cols, full_df = build_features(clean_df)
    (X_train, X_test,
     y_temp_train, y_temp_test,
     y_rain_train, y_rain_test) = split_time(X, y_temp, y_rain, TEST_RATIO)

    temp_model = train_temp(X_train, y_temp_train)
    rain_model = train_rain(X_train, y_rain_train)

    temp_metrics = metrics_reg(temp_model, X_test, y_temp_test)
    rain_metrics = metrics_clf(rain_model, X_test, y_rain_test)

    save(temp_model, TEMP_MODEL_FILE)
    save(rain_model, PRECIP_MODEL_FILE)
    save({
        "feature_cols": feature_cols,
        "last_date": str(clean_df["DATE"].max()),
        "rows": len(clean_df),
        "temp_metrics": temp_metrics,
        "rain_metrics": rain_metrics
    }, META_FILE)

    print("Treino concluído.")
    print("Métricas temperatura:", temp_metrics)
    print("Métricas chuva:", rain_metrics)
    print("Artefatos salvos em:", ARTIFACT_DIR)

if __name__ == "__main__":
    main()