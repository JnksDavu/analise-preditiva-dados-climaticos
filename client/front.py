import streamlit as st
import pandas as pd
import altair as alt
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.title("Dashboard - Previsão Climática São Paulo (NOAA)")
st.caption(
    "Comparação entre modelos SARIMA e Holt-Winters para previsão de temperatura média diária (TAVG)."
)

COLS_PT = {
    "DATE": "Data",
    "TAVG": "Temperatura média (TAVG)",
    "TMAX": "Temperatura máxima (TMAX)",
    "TMIN": "Temperatura mínima (TMIN)",
    "PRCP": "Precipitação (PRCP)",
    "PRCP_MM": "Precipitação (mm)",
}


# -----------------------------
# 1) BUSCAR EDA NO BACKEND
# -----------------------------
@st.cache_data
def get_eda():
    resp = requests.get(f"{API_URL}/eda", timeout=60)
    conteudo = resp.text
    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        st.error("Falha ao obter EDA da API. Resposta bruta:")
        st.code(conteudo)
        raise
    return data


eda = get_eda()

# Converter estruturas em DataFrames
df_hist = pd.DataFrame(eda.get("hist_tavg", []))
if not df_hist.empty:
    df_hist["DATE"] = pd.to_datetime(df_hist["DATE"])

df_stats = pd.DataFrame(eda.get("stats", []))
df_missing = pd.DataFrame(eda.get("missing", []))
df_sample = pd.DataFrame(eda.get("amostra", []))

# novos dfs para análises extras
df_trend = pd.DataFrame(eda.get("trend_30", []))
if not df_trend.empty:
    df_trend["DATE"] = pd.to_datetime(df_trend["DATE"])

df_decomp = pd.DataFrame(eda.get("decomp", []))
if not df_decomp.empty:
    df_decomp["DATE"] = pd.to_datetime(df_decomp["DATE"])

df_hist_bins = pd.DataFrame(eda.get("hist_bins", []))
boxplot = eda.get("boxplot", {})

# -----------------------------
# 2) NOVO DASH: EDA / DADOS BRUTOS
# -----------------------------
st.header("Análise Exploratória de Dados (EDA)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Qtd. registros", eda.get("n_linhas", "-"))
with col2:
    st.metric("Data inicial", eda.get("data_min", "-"))
with col3:
    st.metric("Data final", eda.get("data_max", "-"))

st.subheader("Amostra dos Dados Brutos")
if not df_sample.empty:
    st.dataframe(df_sample.rename(columns=COLS_PT))
else:
    st.info("Amostra não disponível.")

st.subheader("Estatísticas Descritivas")
if not df_stats.empty:
    st.dataframe(df_stats.rename(columns=lambda c: COLS_PT.get(c, c)))
else:
    st.info("Estatísticas não disponíveis.")

st.subheader("Valores Ausentes")
if not df_missing.empty:
    st.dataframe(df_missing)
else:
    st.info("Informação de valores ausentes não disponível.")

# -----------------------------
# 3) DISTRIBUIÇÃO DOS DADOS
# -----------------------------
st.subheader("Distribuição da Temperatura (Histograma)")
if not df_hist_bins.empty:
    df_hist_bins["bin_center"] = (
        df_hist_bins["bin_left"] + df_hist_bins["bin_right"]
    ) / 2.0

    chart_dist = (
        alt.Chart(df_hist_bins)
        .mark_bar()
        .encode(
            x=alt.X("bin_center:Q", title="Temperatura média (TAVG)"),
            y=alt.Y("count:Q", title="Frequência"),
            tooltip=[
                alt.Tooltip("bin_left:Q", title="De"),
                alt.Tooltip("bin_right:Q", title="Até"),
                alt.Tooltip("count:Q", title="Contagem"),
            ],
        )
    )
    st.altair_chart(chart_dist, use_container_width=True)
else:
    st.info("Distribuição não disponível.")

st.subheader("Resumo da Distribuição (Boxplot)")
if boxplot and boxplot.get("q1") is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Q1 (25%)", boxplot["q1"])
        st.write("Q3 (75%)", boxplot["q3"])
    with col2:
        st.write("Mediana", boxplot["median"])
        st.write("IQR", boxplot["iqr"])
    with col3:
        st.write("Limite inferior", boxplot["lower_whisker"])
        st.write("Limite superior", boxplot["upper_whisker"])
        st.write("Qtd. outliers", len(boxplot.get("outliers", [])))
else:
    st.info("Boxplot não disponível.")

# -----------------------------
# 4) TENDÊNCIA E SAZONALIDADE
# -----------------------------
st.subheader("Tendência (Média Móvel de 30 dias)")
if not df_trend.empty:
    chart_trend = (
        alt.Chart(df_trend)
        .mark_line()
        .encode(
            x=alt.X("DATE:T", title="Data"),
            y=alt.Y("trend_30:Q", title="Temperatura média (TAVG)"),
            tooltip=["DATE:T", "trend_30:Q"],
        )
    )
    st.altair_chart(chart_trend, use_container_width=True)
else:
    st.info("Não foi possível calcular a tendência.")

st.subheader("Decomposição da Série Temporal (Tendência, Sazonalidade, Resíduos)")
if not df_decomp.empty:
    base = alt.Chart(df_decomp).encode(x=alt.X("DATE:T", title="Data"))

    chart_obs = (
        base.mark_line()
        .encode(y=alt.Y("observed:Q", title="Observado"))
        .properties(title="Componente Observado")
    )
    chart_trend_dec = (
        base.mark_line()
        .encode(y=alt.Y("trend:Q", title="Tendência"))
        .properties(title="Componente de Tendência")
    )
    chart_seasonal = (
        base.mark_line()
        .encode(y=alt.Y("seasonal:Q", title="Sazonalidade"))
        .properties(title="Componente Sazonal (ciclo semanal)")
    )
    chart_resid = (
        base.mark_line()
        .encode(y=alt.Y("resid:Q", title="Resíduos"))
        .properties(title="Resíduos")
    )

    st.altair_chart(chart_obs, use_container_width=True)
    st.altair_chart(chart_trend_dec, use_container_width=True)
    st.altair_chart(chart_seasonal, use_container_width=True)
    st.altair_chart(chart_resid, use_container_width=True)
else:
    st.info("Não foi possível decompor a série temporal.")

# -----------------------------
# 5) HISTÓRICO TAVG (usa EDA)
# -----------------------------
st.subheader("Histórico de Temperatura Média (TAVG)")
if not df_hist.empty:
    st.line_chart(df_hist.set_index("DATE")["TAVG"])
else:
    st.info("Série histórica de TAVG não disponível para plotagem.")

# -----------------------------
# 6) PREVISÃO – continua igual
# -----------------------------
st.subheader("Gerar Previsão")
dias = st.slider("Dias futuros", 1, 30, 14)

if st.button("Prever com SARIMA e Holt-Winters"):
    try:
        resp = requests.get(
            f"{API_URL}/predict", params={"days": dias}, timeout=60
        )
        conteudo = resp.text
        resp.raise_for_status()
        data = resp.json()

        if "previsoes" not in data:
            st.error(
                "Resposta da API não contém chave 'previsoes'. Conteúdo bruto:"
            )
            st.code(conteudo)
        else:
            df_future = pd.DataFrame(data["previsoes"])
            df_future["data"] = pd.to_datetime(df_future["data"])

            base_hist = (
                alt.Chart(df_hist.tail(90))
                .mark_line(color="steelblue")
                .encode(
                    x=alt.X("DATE:T", title="Data"),
                    y=alt.Y("TAVG:Q", title="Temperatura média (TAVG)"),
                    tooltip=[
                        alt.Tooltip("DATE:T", title="Data"),
                        alt.Tooltip("TAVG:Q", title="TAVG"),
                    ],
                )
                .properties(title="Histórico (90 dias) e Previsões")
            )

            hw_line = (
                alt.Chart(df_future)
                .mark_line(color="crimson")
                .encode(
                    x=alt.X("data:T", title="Data"),
                    y=alt.Y(
                        "temperatura_prevista_holt_winters:Q",
                        title="Temp. prevista Holt-Winters",
                    ),
                    tooltip=[
                        alt.Tooltip("data:T", title="Data"),
                        alt.Tooltip(
                            "temperatura_prevista_holt_winters:Q",
                            title="Holt-Winters",
                        ),
                    ],
                )
            )

            sarima_line = (
                alt.Chart(df_future)
                .mark_line(color="seagreen")
                .encode(
                    x=alt.X("data:T", title="Data"),
                    y=alt.Y(
                        "temperatura_prevista_sarima:Q",
                        title="Temp. prevista SARIMA",
                    ),
                    tooltip=[
                        alt.Tooltip("data:T", title="Data"),
                        alt.Tooltip(
                            "temperatura_prevista_sarima:Q", title="SARIMA"
                        ),
                    ],
                )
            )

            st.altair_chart(
                (base_hist + hw_line + sarima_line).interactive(),
                use_container_width=True,
            )

            st.subheader("Métricas (Teste)")
            metricas = data.get("metricas", {})
            col1, col2 = st.columns(2)
            hw = metricas.get("holt_winters", {})
            sa = metricas.get("sarima", {})
            with col1:
                st.markdown("Holt-Winters")
                st.write(
                    {
                        "RMSE": hw.get("rmse"),
                        "MAE": hw.get("mae"),
                        "MAPE (%)": hw.get("mape"),
                        "R²": hw.get("r2"),
                    }
                )
            with col2:
                st.markdown("SARIMA")
                st.write(
                    {
                        "RMSE": sa.get("rmse"),
                        "MAE": sa.get("mae"),
                        "MAPE (%)": sa.get("mape"),
                        "R²": sa.get("r2"),
                    }
                )

            st.subheader("Previsões")
            st.dataframe(
                df_future.rename(
                    columns={
                        "data": "Data",
                        "temperatura_prevista_holt_winters": "Temp. Holt-Winters",
                        "temperatura_prevista_sarima": "Temp. SARIMA",
                    }
                )
            )

    except Exception as e:
        st.error(f"Falha ao obter previsões: {e}")
