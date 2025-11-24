import streamlit as st
import pandas as pd
import altair as alt
import requests
from pathlib import Path

API_URL = "http://localhost:8000"
DATA_FILE = Path("data/4174560.csv")

st.title("Dashboard - Previsão Climática São Paulo (NOAA)")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

df_raw = load_data()
st.write("Amostra dos dados:", df_raw.head())

hist = (df_raw.groupby("DATE")
        .agg({"TAVG": "mean"})
        .reset_index()
        .sort_values("DATE"))

st.subheader("Histórico TAVG")
st.line_chart(hist.set_index("DATE")["TAVG"])

st.subheader("Gerar Previsão")
days = st.slider("Dias futuros", 1, 30, 14)
if st.button("Prever"):
    resp = requests.get(f"{API_URL}/predict", params={"days": days})
    if resp.ok:
        data = resp.json()
        df_future = pd.DataFrame(data["predictions"])
        df_future["DATE"] = pd.to_datetime(df_future["DATE"])
        st.success("Previsão gerada")
        # Temperatura
        chart_hist = alt.Chart(hist.tail(90)).mark_line(color="blue").encode(
            x="DATE:T", y="TAVG:Q", tooltip=["DATE", "TAVG"]
        )
        chart_future = alt.Chart(df_future).mark_line(color="red").encode(
            x="DATE:T", y="TAVG_PRED:Q", tooltip=["DATE", "TAVG_PRED", "RAIN_PROB"]
        )
        st.altair_chart((chart_hist + chart_future).interactive(), use_container_width=True)
        # Probabilidade chuva
        st.subheader("Probabilidade de Chuva")
        rain_chart = alt.Chart(df_future).mark_bar().encode(
            x="DATE:T",
            y=alt.Y("RAIN_PROB:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["DATE", "RAIN_PROB"]
        )
        st.altair_chart(rain_chart.interactive(), use_container_width=True)
        st.write(df_future)
    else:
        st.error("Falha na chamada à API")