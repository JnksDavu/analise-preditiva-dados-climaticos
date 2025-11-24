import streamlit as st
import pandas as pd
import altair as alt
import requests
from pathlib import Path

API_URL = "http://localhost:8000"
DATA_FILE = Path("data/4174560.csv")

st.title("Dashboard - Previsão Climática São Paulo (NOAA)")
st.caption("Comparação entre modelos SARIMA e Holt‑Winters para previsão de temperatura média diária (TAVG).")

COLS_PT = {
    "DATE": "Data",
    "TAVG": "Temperatura média (TAVG)",
    "TMAX": "Temperatura máxima (TMAX)",
    "TMIN": "Temperatura mínima (TMIN)",
    "PRCP": "Precipitação (PRCP)",
    "PRCP_MM": "Precipitação (mm)"
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

df_raw = load_data()
df_hist = (df_raw.groupby("DATE").agg({"TAVG": "mean"}).reset_index().sort_values("DATE"))

st.subheader("Histórico de Temperatura Média (TAVG)")
st.line_chart(df_hist.set_index("DATE")["TAVG"])

st.subheader("Gerar Previsão")
dias = st.slider("Dias futuros", 1, 30, 14)
if st.button("Prever com SARIMA e Holt‑Winters"):
    try:
        resp = requests.get(f"{API_URL}/predict", params={"days": dias}, timeout=60)
        conteudo = resp.text
        resp.raise_for_status()
        data = resp.json()

        if "previsoes" not in data:
            st.error("Resposta da API não contém chave 'previsoes'. Conteúdo bruto:")
            st.code(conteudo)
        else:
            df_future = pd.DataFrame(data["previsoes"])
            df_future["data"] = pd.to_datetime(df_future["data"])

            base_hist = alt.Chart(df_hist.tail(90)).mark_line(color="steelblue").encode(
                x=alt.X("DATE:T", title="Data"),
                y=alt.Y("TAVG:Q", title="Temperatura média (TAVG)"),
                tooltip=[alt.Tooltip("DATE:T", title="Data"),
                         alt.Tooltip("TAVG:Q", title="TAVG")]
            ).properties(title="Histórico (90 dias) e Previsões")

            hw_line = alt.Chart(df_future).mark_line(color="crimson").encode(
                x=alt.X("data:T", title="Data"),
                y=alt.Y("temperatura_prevista_holt_winters:Q", title="Temp. prevista Holt‑Winters"),
                tooltip=[alt.Tooltip("data:T", title="Data"),
                         alt.Tooltip("temperatura_prevista_holt_winters:Q", title="Holt‑Winters")]
            )

            sarima_line = alt.Chart(df_future).mark_line(color="seagreen").encode(
                x=alt.X("data:T", title="Data"),
                y=alt.Y("temperatura_prevista_sarima:Q", title="Temp. prevista SARIMA"),
                tooltip=[alt.Tooltip("data:T", title="Data"),
                         alt.Tooltip("temperatura_prevista_sarima:Q", title="SARIMA")]
            )

            st.altair_chart((base_hist + hw_line + sarima_line).interactive(), use_container_width=True)

            st.subheader("Métricas (Teste)")
            metricas = data.get("metricas", {})
            col1, col2 = st.columns(2)
            hw = metricas.get("holt_winters", {})
            sa = metricas.get("sarima", {})
            with col1:
                st.markdown("Holt‑Winters")
                st.write({
                    "RMSE": hw.get("rmse"),
                    "MAE": hw.get("mae"),
                    "MAPE (%)": hw.get("mape"),
                    "R²": hw.get("r2")
                })
            with col2:
                st.markdown("SARIMA")
                st.write({
                    "RMSE": sa.get("rmse"),
                    "MAE": sa.get("mae"),
                    "MAPE (%)": sa.get("mape"),
                    "R²": sa.get("r2")
                })

            st.subheader("Previsões")
            st.dataframe(df_future.rename(columns={
                "data": "Data",
                "temperatura_prevista_holt_winters": "Temp. Holt‑Winters",
                "temperatura_prevista_sarima": "Temp. SARIMA"
            }))

    except Exception as e:
        st.error(f"Falha ao obter previsões: {e}")

st.subheader("Amostra dos Dados Brutos")
st.dataframe(df_raw.rename(columns=COLS_PT).head(20))