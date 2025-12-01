from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.seasonal import seasonal_decompose

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
    """
    Limpeza/aggregação dos dados brutos para uso na API.
    Mantém TAVG, TMAX, TMIN e PRCP_MM por data.
    """
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])

    for col in ["PRCP", "TAVG", "TMAX", "TMIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preenchimento básico semelhante ao usado no treino
    if "TAVG" in df.columns:
        df["TMAX"] = df.get("TMAX", df["TAVG"]).fillna(df["TAVG"])
        df["TMIN"] = df.get("TMIN", df["TAVG"]).fillna(df["TAVG"])

    df["PRCP_MM"] = df.get("PRCP", 0.0).fillna(0.0)

    agg = (
        df.groupby("DATE")
        .agg(
            {
                "TAVG": "mean",
                "TMAX": "mean",
                "TMIN": "mean",
                "PRCP_MM": "mean",
            }
        )
        .reset_index()
        .sort_values("DATE")
    )

    agg = agg.dropna(subset=["TAVG"]).reset_index(drop=True)
    return agg


def build_eda_payload() -> dict:
    """
    Calcula estatísticas descritivas e outras infos de EDA
    a partir do dataset bruto.
    """
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)


    df["DATE"] = pd.to_datetime(df["DATE"])

    # Série diária TAVG para o gráfico histórico
    hist_df = df[["DATE", "TAVG"]].dropna().sort_values("DATE")

    # Estatísticas descritivas só para colunas numéricas
    stats_df = df.select_dtypes(include="number").describe().reset_index()
    stats_df = stats_df.rename(columns={"index": "estatistica"})

    estatisticas_map = {
    "count": "Total (count)",
    "mean": "Média (mean)",
    "std": "Desvio padrão (std)",
    "min": "Mínimo (min)",
    "25%": "1º quartil (25%)",
    "50%": "Mediana (50%)",
    "75%": "3º quartil (75%)",
    "max": "Máximo (max)"
    }

    stats_df["estatistica"] = stats_df["estatistica"].map(estatisticas_map)

    # Valores ausentes por coluna
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["coluna", "faltantes"]

    # Amostra dos dados brutos
    sample_df = df.head(50)  # pode ajustar o tamanho

    # ------------- Tendência (média móvel 30 dias) -------------
    y = hist_df.set_index("DATE")["TAVG"].asfreq("D")
    y = y.interpolate(limit_direction="both")

    trend_30 = y.rolling(window=30, center=True).mean().dropna()
    trend_df = trend_30.reset_index()
    trend_df.columns = ["DATE", "trend_30"]

    # ------------- Decomposição sazonal (periodo semanal=7) -------------
    try:
        decomp = seasonal_decompose(y, model="additive", period=7)
        decomp_df = pd.DataFrame(
            {
                "DATE": y.index,
                "observed": decomp.observed,
                "trend": decomp.trend,
                "seasonal": decomp.seasonal,
                "resid": decomp.resid,
            }
        ).dropna()
    except Exception:
        decomp_df = pd.DataFrame()

    # ------------- Distribuição (histograma) -------------
    tavg_values = y.dropna().values
    if len(tavg_values) > 0:
        counts, bin_edges = np.histogram(tavg_values, bins=20)
        hist_bins = []
        for i in range(len(counts)):
            hist_bins.append(
                {
                    "bin_left": float(bin_edges[i]),
                    "bin_right": float(bin_edges[i + 1]),
                    "count": int(counts[i]),
                }
            )

        # Boxplot / outliers (regra 1.5 IQR)
        q1 = float(np.percentile(tavg_values, 25))
        q2 = float(np.percentile(tavg_values, 50))
        q3 = float(np.percentile(tavg_values, 75))
        iqr = q3 - q1
        lower_whisker = float(q1 - 1.5 * iqr)
        upper_whisker = float(q3 + 1.5 * iqr)
        outliers = [
            float(v)
            for v in tavg_values
            if v < lower_whisker or v > upper_whisker
        ]
    else:
        hist_bins = []
        q1 = q2 = q3 = iqr = lower_whisker = upper_whisker = None
        outliers = []

    payload = {
        "n_linhas": int(len(df)),
        "colunas": list(df.columns),
        "data_min": df["DATE"].min().date().isoformat() if len(df) else None,
        "data_max": df["DATE"].max().date().isoformat() if len(df) else None,
        "amostra": sample_df.to_dict(orient="records"),
        "stats": stats_df.to_dict(orient="records"),
        "missing": missing_df.to_dict(orient="records"),
        "hist_tavg": hist_df.to_dict(orient="records"),
        # novos campos
        "trend_30": trend_df.to_dict(orient="records"),
        "decomp": decomp_df.to_dict(orient="records"),
        "hist_bins": hist_bins,
        "boxplot": {
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "lower_whisker": lower_whisker,
            "upper_whisker": upper_whisker,
            "outliers": outliers,
        },
    }
    return payload


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
    y = (
        pd.Series(df["TAVG"].values, index=df["DATE"])
        .asfreq("D")
        .interpolate(limit_direction="both")
    )
    last_date = y.index.max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=days, freq="D"
    )

    hw_forecast = hw_model.forecast(days)
    sarima_forecast = sarima_model.get_forecast(days).predicted_mean
    hw_forecast.index = future_dates
    sarima_forecast.index = future_dates

    registros = []
    for dt in future_dates:
        registros.append(
            {
                "data": str(dt.date()),
                "temperatura_prevista_holt_winters": float(hw_forecast.loc[dt]),
                "temperatura_prevista_sarima": float(sarima_forecast.loc[dt]),
            }
        )
    return registros, meta


app = FastAPI(title="API Previsão Climática - SARIMA vs Holt-Winters")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/eda")
def eda():
    """
    Endpoint para análise exploratória dos dados brutos.
    """
    return build_eda_payload()


@app.get("/predict")
def predict(days: int = Query(FORECAST_HORIZON_DEFAULT, ge=1, le=60)):
    previsoes, meta = forecast_both(days)
    return {
        "dias": days,
        "previsoes": previsoes,
        "metricas": meta.get("metricas", {}),
        "metricas_cv": meta.get("metricas_cv", {}),
        "analise_residuos_sarima": meta.get("analise_residuos_sarima", {}),
        "info": {
            "tamanho_dataset": meta.get("dataset_tamanho"),
            "tamanho_treino": meta.get("tamanho_treino"),
            "tamanho_teste": meta.get("tamanho_teste"),
            "ultima_data": meta.get("ultima_data"),
            "periodicidade": meta.get("periodicidade"),
            "sazonalidade": meta.get("sazonalidade"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    ensure_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=8000)
