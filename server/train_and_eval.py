from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_PATH = Path("data/4174560.csv")
ARTIFACT_DIR = Path("server/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SARIMA_MODEL_FILE = ARTIFACT_DIR / "sarima_model.pkl"
HW_MODEL_FILE = ARTIFACT_DIR / "holtwinters_model.pkl"
META_FILE = ARTIFACT_DIR / "meta.pkl"

TEST_RATIO = 0.2
SEASONAL_PERIOD = 7 

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

def fit_naive(y_train: pd.Series, seasonal_periods: int = SEASONAL_PERIOD):
    class NaiveModel:
        def __init__(self, last_n_values, seasonal_periods):
            # Armazena os últimos N valores de treino
            self.last_n_values = last_n_values
            self.seasonal_periods = seasonal_periods

        def forecast(self, steps):
            # Previsão: repete o último valor de 'seasonal_periods' dias atrás
            # Se for Naive simples (period=1), repete o último valor
            
            # Pegamos o valor da última data que está N passos atrás, onde N é a sazonalidade.
            if len(self.last_n_values) < self.seasonal_periods:
                # Fallback para Naive simples (usa o último valor)
                forecast_value = self.last_n_values.iloc[-1] if len(self.last_n_values) > 0 else 0
            else:
                # Naive Sazonal: valor de N dias atrás
                forecast_value = self.last_n_values.iloc[-self.seasonal_periods]
            
            return pd.Series([forecast_value] * steps, name="naive_pred")

    return NaiveModel(y_train, seasonal_periods)

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

def analyze_residuals(y_true: pd.Series, y_pred: np.ndarray, lags: int = 20) -> Dict[str, Any]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"ljung_box_p_value": None, "acf_lags": []}

    # 1. Calcular Resíduos
    residuals = y_true - y_pred
    
    # 2. Teste de Ljung-Box para Ruído Branco
    # H0: Os resíduos são ruído branco (não há autocorrelação)
    try:
        # Pega o p-valor do primeiro lag (ou o último, dependendo da biblioteca/versão)
        # Usamos .iloc[-1] para ser mais robusto
        ljung_box_results = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        ljung_box_p_value = float(ljung_box_results["lb_pvalue"].iloc[-1])
    except Exception:
        ljung_box_p_value = None

    # 3. Autocorrelação (ACF)
    try:
        # Calcula a ACF (até o lag definido)
        acf_values = acf(residuals, nlags=lags, adjusted=False)
        acf_list = [float(v) for v in acf_values]
    except Exception:
        acf_list = []
        
    return {
        "ljung_box_p_value": ljung_box_p_value,
        "acf_lags": acf_list,
        # A média dos resíduos deve ser próxima de zero
        "mean_residuals": float(np.mean(residuals))
    }

# ---------- VALIDAÇÃO CRUZADA ----------

def run_cross_validation(y: pd.Series, model_type: str, n_splits: int = 5) -> Dict[str, float]:
    n_splits = max(min(n_splits, len(y) // 100), 2) # Garante que haja splits suficientes
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(y) // (n_splits + 1))
    
    all_metrics = []

    # Itera sobre os splits de treino/teste gerados sequencialmente
    for train_index, test_index in tscv.split(y):
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        # Seleciona a função de ajuste (fit) com base no tipo de modelo
        if model_type == "holt_winters":
            model = fit_holt_winters(y_train, SEASONAL_PERIOD)
            y_pred = model.forecast(len(y_test))
            y_pred.index = y_test.index
        elif model_type == "sarima":
            model = fit_sarima(y_train, SEASONAL_PERIOD)
            y_pred = model.get_forecast(len(y_test)).predicted_mean
            y_pred.index = y_test.index 
        elif model_type == "naive":
            model = fit_naive(y_train, SEASONAL_PERIOD)
            y_pred = model.forecast(len(y_test))
            y_pred.index = y_test.index 
        else:
            continue
        
        # Calcula as métricas para este split
        metrics = calc_metrics(y_test, y_pred.values)
        all_metrics.append(metrics)
        
    if not all_metrics:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}
    
    # Calcula a média das métricas em todos os splits
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics if m[metric] is not None]))
        for metric in all_metrics[0].keys()
    }
    
    return avg_metrics

# ---------- EXECUÇÃO TREINO ----------
def main():
    y = load_series()
    print("Iniciando Validação Cruzada (Walk-Forward)...")
    cv_metrics_hw = run_cross_validation(y, "holt_winters")
    cv_metrics_sarima = run_cross_validation(y, "sarima")
    cv_metrics_naive = run_cross_validation(y, "naive")
    print("Validação Cruzada concluída.")

    y_train, y_test = split_train_test(y, TEST_RATIO)

    # 1. Modelos Avançados
    hw_model = fit_holt_winters(y_train)
    sarima_model = fit_sarima(y_train)

    # 2. Modelo Baseline
    naive_model = fit_naive(y_train)

    steps = len(y_test)
    if steps > 0:
        # Previsões
        hw_pred = hw_model.forecast(steps)
        sarima_pred = sarima_model.get_forecast(steps).predicted_mean
        naive_pred = naive_model.forecast(steps)

        # Alinhar índices para cálculo de métricas
        hw_pred.index = y_test.index
        sarima_pred.index = y_test.index
        naive_pred.index = y_test.index

        # Cálculo de Métricas
        metrics_hw = calc_metrics(y_test, hw_pred.values)
        metrics_sarima = calc_metrics(y_test, sarima_pred.values)
        metrics_naive = calc_metrics(y_test, naive_pred.values)

        residual_analysis_sarima = analyze_residuals(y_test, sarima_pred.values)
    else:
        metrics_hw = {"rmse": None, "mae": None, "mape": None, "r2": None}
        metrics_sarima = {"rmse": None, "mae": None, "mape": None, "r2": None}
        metrics_naive = {"rmse": None, "mae": None, "mape": None, "r2": None}

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
            "sarima": metrics_sarima,
            "naive": metrics_naive
        },
        "metricas_cv": { 
            "holt_winters": cv_metrics_hw,
            "sarima": cv_metrics_sarima,
            "naive": cv_metrics_naive,
        },
        "analise_residuos_sarima": residual_analysis_sarima
    }, META_FILE)

    print("Treino concluído.")
    print("Métricas Holt-Winters:", metrics_hw)
    print("Métricas SARIMA:", metrics_sarima)
    print("Métricas Naive:", metrics_naive)
    print(f"Teste Ljung-Box (SARIMA): p-value = {residual_analysis_sarima.get('ljung_box_p_value'):.4f}")
    print("Arquivos gerados:")
    for p in [HW_MODEL_FILE, SARIMA_MODEL_FILE, META_FILE]:
        print(" -", p, "OK" if p.exists() else "FALHOU")

if __name__ == "__main__":
    main()