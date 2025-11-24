"""
Dashboard Streamlit para Análise de Séries Temporais - Dados Climáticos NOAA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import subprocess
import sys

# Desabilitar downloads - usar apenas arquivo local
USE_LOCAL_ONLY = True

# Configuração da página
st.set_page_config(
    page_title="Dashboard NOAA - Análise de Séries Temporais",
    page_icon="🌡️",
    layout="wide"
)

# Título
st.title("🌡️ Dashboard - Análise de Séries Temporais NOAA")
st.markdown("---")

# Função para carregar dados do PSV
def load_ghcnh_psv(file_path):
    """Carrega e processa arquivo PSV do GHCNh"""
    try:
        # Ler arquivo PSV (pipe-separated values)
        df = pd.read_csv(file_path, sep='|', low_memory=False)
        
        # Converter coluna DATE para datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Extrair temperatura (já está em °C no GHCNh)
            if 'temperature' in df.columns:
                # Filtrar valores válidos
                df = df[df['temperature'].notna()].copy()
                
                # Agregar dados horários para diários
                df['date'] = df['DATE'].dt.date
                
                # Agregar por dia
                daily_df = df.groupby('date').agg({
                    'temperature': ['mean', 'min', 'max'],
                    'precipitation': lambda x: x.sum() if x.notna().any() else np.nan,
                    'wind_speed': 'mean',
                    'relative_humidity': 'mean'
                }).reset_index()
                
                # Flatten column names
                daily_df.columns = ['date', 'temperature', 'temp_min', 'temp_max', 
                                   'precipitation', 'wind_speed', 'relative_humidity']
                
                # Converter date de date para datetime
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                
                # Selecionar colunas principais
                result_df = daily_df[['date', 'temperature']].copy()
                result_df = result_df[result_df['temperature'].notna()].copy()
                result_df = result_df.sort_values('date').reset_index(drop=True)
                
                return result_df
        return None
    except Exception as e:
        st.error(f"Erro ao processar PSV: {e}")
        return None

# Função para carregar dados
@st.cache_data(ttl=60)  # Cache por 60 segundos
def load_data():
    """Carrega dados do arquivo PSV, CSV ou cria dados sintéticos"""
    # Prioridade 1: Arquivo PSV local (dados de São Paulo)
    psv_file = 'data/GHCNh_AAI0000TNCA_2025.psv'
    
    if os.path.exists(psv_file):
        try:
            df = load_ghcnh_psv(psv_file)
            if df is not None and len(df) > 0:
                return df, f"Dados de São Paulo carregados do PSV ({len(df)} dias)"
        except Exception as e:
            st.warning(f"Erro ao carregar PSV: {e}")
    
    # Prioridade 2: Arquivo CSV processado
    data_file = 'noaa_data.csv'
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            return df, "Dados reais carregados (CSV)"
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return create_synthetic_data(), "Erro ao carregar - usando dados sintéticos"
    
    # Fallback: dados sintéticos
    return create_synthetic_data(), "Dados sintéticos (arquivo não encontrado)"

def create_synthetic_data():
    """Cria dados sintéticos"""
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    trend = np.linspace(15, 18, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 2, n)
    temperature = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'temperature': temperature
    })

# Carregar dados
df, data_status = load_data()
df_ts = df.set_index('date').copy()

# Sidebar
st.sidebar.header("📊 Controles")
st.sidebar.info(f"Status: {data_status}")
st.sidebar.info(f"Registros: {len(df):,}")
st.sidebar.info(f"Período: {df['date'].min().date()} a {df['date'].max().date()}")

# Processar dados
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Processar Dados")
st.sidebar.markdown("**Arquivo local:**")
st.sidebar.info("Usando apenas dados do arquivo PSV local")

if st.sidebar.button("🔄 Reprocessar Dados do PSV"):
    with st.sidebar:
        with st.spinner("Processando dados..."):
            # Executar script de processamento
            result = subprocess.run(
                [sys.executable, "download_data.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                st.success("✓ Dados processados!")
                # Limpar cache para recarregar dados
                load_data.clear()
                st.rerun()
            else:
                st.error("Erro ao processar dados")
                st.code(result.stderr)
                if result.stdout:
                    st.code(result.stdout)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Dados:**")
st.sidebar.markdown("Arquivo PSV local: `data/GHCNh_AAI0000TNCA_2025.psv`")

# Filtros
st.sidebar.subheader("Filtros")
date_range = st.sidebar.date_input(
    "Período",
    value=(df['date'].min().date(), df['date'].max().date()),
    min_value=df['date'].min().date(),
    max_value=df['date'].max().date()
)

if len(date_range) == 2:
    df_filtered = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
else:
    df_filtered = df

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Temperatura Média", f"{df_filtered['temperature'].mean():.2f} °C")
with col2:
    st.metric("Temperatura Máxima", f"{df_filtered['temperature'].max():.2f} °C")
with col3:
    st.metric("Temperatura Mínima", f"{df_filtered['temperature'].min():.2f} °C")
with col4:
    st.metric("Desvio Padrão", f"{df_filtered['temperature'].std():.2f} °C")

st.markdown("---")

# Gráficos principais
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Série Temporal", "📊 Estatísticas", "🗓️ Sazonalidade", "📉 Análise Temporal", "🏆 Benchmark de Modelos"])

with tab1:
    st.subheader("Série Temporal de Temperatura")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['temperature'],
        mode='lines',
        name='Temperatura',
        line=dict(color='red', width=1),
        hovertemplate='<b>Data:</b> %{x}<br><b>Temperatura:</b> %{y:.2f} °C<extra></extra>'
    ))
    
    # Linha de tendência
    z = np.polyfit(range(len(df_filtered)), df_filtered['temperature'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=p(range(len(df_filtered))),
        mode='lines',
        name='Tendência',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Data",
        yaxis_title="Temperatura (°C)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuição de Temperatura**")
        fig_hist = px.histogram(
            df_filtered,
            x='temperature',
            nbins=50,
            title='Histograma de Temperatura',
            labels={'temperature': 'Temperatura (°C)', 'count': 'Frequência'}
        )
        fig_hist.update_traces(marker_color='red', opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.write("**Boxplot**")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df_filtered['temperature'],
            name='Temperatura',
            marker_color='lightblue'
        ))
        fig_box.update_layout(
            title='Boxplot de Temperatura',
            yaxis_title='Temperatura (°C)',
            template='plotly_white'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Tabela de estatísticas
    st.write("**Estatísticas Detalhadas**")
    stats = df_filtered['temperature'].describe()
    stats_df = pd.DataFrame({
        'Estatística': stats.index,
        'Valor': stats.values
    })
    stats_df['Valor'] = stats_df['Valor'].round(2)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Análise de Sazonalidade")
    
    df_filtered['month'] = df_filtered['date'].dt.month
    df_filtered['year'] = df_filtered['date'].dt.year
    df_filtered['day_of_year'] = df_filtered['date'].dt.dayofyear
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperatura média por mês
        monthly = df_filtered.groupby('month')['temperature'].mean()
        fig_month = go.Figure()
        fig_month.add_trace(go.Scatter(
            x=monthly.index,
            y=monthly.values,
            mode='lines+markers',
            name='Temp. Média Mensal',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='red')
        ))
        fig_month.update_layout(
            title='Temperatura Média Mensal',
            xaxis_title='Mês',
            yaxis_title='Temperatura Média (°C)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            template='plotly_white'
        )
        st.plotly_chart(fig_month, use_container_width=True)
    
    with col2:
        # Temperatura média por ano
        yearly = df_filtered.groupby('year')['temperature'].mean()
        fig_year = go.Figure()
        fig_year.add_trace(go.Scatter(
            x=yearly.index,
            y=yearly.values,
            mode='lines+markers',
            name='Temp. Média Anual',
            line=dict(color='darkred', width=2),
            marker=dict(size=8, color='red')
        ))
        fig_year.update_layout(
            title='Temperatura Média Anual (Tendência)',
            xaxis_title='Ano',
            yaxis_title='Temperatura Média (°C)',
            template='plotly_white'
        )
        st.plotly_chart(fig_year, use_container_width=True)
    
    # Heatmap mensal
    st.write("**Heatmap Mensal-Anual**")
    pivot_data = df_filtered.groupby(['year', 'month'])['temperature'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='year', columns='month', values='temperature')
    
    fig_heat = px.imshow(
        pivot_table,
        labels=dict(x="Mês", y="Ano", color="Temperatura (°C)"),
        title="Temperatura Média por Mês e Ano",
        color_continuous_scale='RdYlBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("Análise Temporal Avançada")
    
    # Médias móveis
    window = st.slider("Janela de Média Móvel (dias)", 7, 365, 30)
    
    df_filtered_sorted = df_filtered.sort_values('date').copy()
    df_filtered_sorted['ma'] = df_filtered_sorted['temperature'].rolling(window=window).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=df_filtered_sorted['date'],
        y=df_filtered_sorted['temperature'],
        mode='lines',
        name='Temperatura',
        line=dict(color='lightgray', width=0.5),
        opacity=0.5
    ))
    fig_ma.add_trace(go.Scatter(
        x=df_filtered_sorted['date'],
        y=df_filtered_sorted['ma'],
        mode='lines',
        name=f'Média Móvel ({window} dias)',
        line=dict(color='red', width=2)
    ))
    fig_ma.update_layout(
        title=f'Série Temporal com Média Móvel ({window} dias)',
        xaxis_title="Data",
        yaxis_title="Temperatura (°C)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Decomposição (simplificada)
    st.write("**Decomposição da Série Temporal**")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df_ts_filtered = df_filtered_sorted.set_index('date')['temperature']
        
        if len(df_ts_filtered) >= 730:  # Precisa de pelo menos 2 anos
            decomposition = seasonal_decompose(df_ts_filtered, model='additive', period=365)
            
            fig_decomp = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Tendência', 'Sazonalidade', 'Resíduos'),
                vertical_spacing=0.05
            )
            
            fig_decomp.add_trace(go.Scatter(x=df_ts_filtered.index, y=decomposition.observed, name='Original'), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=df_ts_filtered.index, y=decomposition.trend, name='Tendência'), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=df_ts_filtered.index, y=decomposition.seasonal, name='Sazonalidade'), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=df_ts_filtered.index, y=decomposition.resid, name='Resíduos'), row=4, col=1)
            
            fig_decomp.update_layout(height=800, showlegend=False, template='plotly_white')
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.warning("Dados insuficientes para decomposição (mínimo 2 anos)")
    except Exception as e:
        st.warning(f"Erro na decomposição: {e}")

with tab5:
    st.subheader("🏆 Benchmark de Modelos - Comparação de Desempenho")
    
    # Verificar se existe arquivo de comparação de modelos
    comparison_file = 'comparacao_modelos.csv'
    
    if os.path.exists(comparison_file):
        try:
            comparison_df = pd.read_csv(comparison_file)
            
            st.info("📊 **Metodologia:** Comparação de 8 modelos diferentes usando métricas padronizadas (MAE, RMSE, MAPE, R²). O modelo com melhor desempenho foi selecionado para previsões finais.")
            
            # Filtrar modelos válidos
            valid_models = comparison_df.dropna(subset=['MAE (Test)']).copy()
            
            if len(valid_models) > 0:
                # Identificar melhor modelo (por R²)
                best_model = valid_models.loc[valid_models['R² (Test)'].idxmax(), 'Modelo']
                best_r2 = valid_models.loc[valid_models['R² (Test)'].idxmax(), 'R² (Test)']
                
                # Highlight do melhor modelo
                st.success(f"✅ **Melhor Modelo Selecionado:** {best_model} (R² = {best_r2:.4f})")
                st.markdown("---")
                
                # Tabela comparativa completa
                st.write("**📋 Tabela Comparativa Completa de Modelos**")
                
                # Formatar tabela para exibição
                display_df = valid_models.copy()
                display_df = display_df.round(4)
                
                # Ordenar por R² (melhor primeiro)
                display_df = display_df.sort_values('R² (Test)', ascending=False)
                
                st.dataframe(
                    display_df.style.background_gradient(subset=['R² (Test)'], cmap='Greens')
                    .background_gradient(subset=['MAE (Test)', 'RMSE (Test)', 'MAPE (Test)'], cmap='Reds_r'),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Gráficos comparativos
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Comparação de Métricas - MAE (Mean Absolute Error)**")
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Bar(
                        x=valid_models['Modelo'],
                        y=valid_models['MAE (Test)'],
                        marker=dict(
                            color=valid_models['MAE (Test)'],
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title="MAE")
                        ),
                        text=[f"{val:.3f}" for val in valid_models['MAE (Test)']],
                        textposition='outside'
                    ))
                    fig_mae.update_layout(
                        title='MAE por Modelo (Menor é Melhor)',
                        xaxis_title='Modelo',
                        yaxis_title='MAE (°C)',
                        template='plotly_white',
                        height=400,
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
                
                with col2:
                    st.write("**📊 Comparação de Métricas - R² (Coeficiente de Determinação)**")
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Bar(
                        x=valid_models['Modelo'],
                        y=valid_models['R² (Test)'],
                        marker=dict(
                            color=valid_models['R² (Test)'],
                            colorscale='Greens',
                            showscale=True,
                            colorbar=dict(title="R²")
                        ),
                        text=[f"{val:.3f}" for val in valid_models['R² (Test)']],
                        textposition='outside'
                    ))
                    fig_r2.update_layout(
                        title='R² por Modelo (Maior é Melhor)',
                        xaxis_title='Modelo',
                        yaxis_title='R²',
                        template='plotly_white',
                        height=400,
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**📊 Comparação de Métricas - RMSE (Root Mean Squared Error)**")
                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(go.Bar(
                        x=valid_models['Modelo'],
                        y=valid_models['RMSE (Test)'],
                        marker=dict(
                            color=valid_models['RMSE (Test)'],
                            colorscale='Oranges',
                            showscale=True,
                            colorbar=dict(title="RMSE")
                        ),
                        text=[f"{val:.3f}" for val in valid_models['RMSE (Test)']],
                        textposition='outside'
                    ))
                    fig_rmse.update_layout(
                        title='RMSE por Modelo (Menor é Melhor)',
                        xaxis_title='Modelo',
                        yaxis_title='RMSE (°C)',
                        template='plotly_white',
                        height=400,
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col4:
                    st.write("**📊 Comparação de Métricas - MAPE (Mean Absolute Percentage Error)**")
                    fig_mape = go.Figure()
                    fig_mape.add_trace(go.Bar(
                        x=valid_models['Modelo'],
                        y=valid_models['MAPE (Test)'],
                        marker=dict(
                            color=valid_models['MAPE (Test)'],
                            colorscale='Blues',
                            showscale=True,
                            colorbar=dict(title="MAPE (%)")
                        ),
                        text=[f"{val:.2f}%" for val in valid_models['MAPE (Test)']],
                        textposition='outside'
                    ))
                    fig_mape.update_layout(
                        title='MAPE por Modelo (Menor é Melhor)',
                        xaxis_title='Modelo',
                        yaxis_title='MAPE (%)',
                        template='plotly_white',
                        height=400,
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig_mape, use_container_width=True)
                
                st.markdown("---")
                
                # Ranking visual
                st.write("**🏅 Ranking dos Modelos (Ordenado por R²)**")
                
                # Criar ranking
                ranking_df = valid_models[['Modelo', 'R² (Test)', 'MAE (Test)', 'RMSE (Test)', 'MAPE (Test)']].copy()
                ranking_df = ranking_df.sort_values('R² (Test)', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                ranking_df = ranking_df[['Rank', 'Modelo', 'R² (Test)', 'MAE (Test)', 'RMSE (Test)', 'MAPE (Test)']]
                ranking_df = ranking_df.round(4)
                
                # Destacar melhor modelo
                def highlight_best(row):
                    if row['Rank'] == 1:
                        return ['background-color: #90EE90'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    ranking_df.style.apply(highlight_best, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Resumo do benchmark
                st.write("**📝 Resumo do Benchmark**")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Total de Modelos Testados", len(valid_models))
                
                with summary_col2:
                    st.metric("Melhor R²", f"{best_r2:.4f}")
                
                with summary_col3:
                    best_mae = valid_models.loc[valid_models['MAE (Test)'].idxmin(), 'MAE (Test)']
                    st.metric("Melhor MAE", f"{best_mae:.4f} °C")
                
                # Descrição dos modelos testados
                st.markdown("""
                **Modelos Testados:**
                - **Baseline:** Naive, Média Móvel, Suavização Exponencial
                - **Estatísticos:** ARIMA, SARIMA, Holt-Winters
                - **Machine Learning:** Prophet, LSTM
                
                **Critério de Seleção:** Modelo com maior R² (coeficiente de determinação) no conjunto de teste.
                """)
                
            else:
                st.warning("⚠️ Arquivo de comparação encontrado, mas nenhum modelo válido foi processado.")
                
        except Exception as e:
            st.error(f"Erro ao carregar comparação de modelos: {e}")
            st.info("💡 Execute o notebook completo para gerar o arquivo 'comparacao_modelos.csv' com os resultados do benchmark.")
    else:
        st.warning("⚠️ Arquivo de comparação de modelos não encontrado.")
        st.info("""
        **Para ver as comparações de modelos (benchmark):**
        
        1. Execute o notebook Jupyter completo (`analise_series_temporais_noaa.ipynb`)
        2. Execute todas as células até a seção de "Comparação de Modelos"
        3. O arquivo `comparacao_modelos.csv` será gerado automaticamente
        4. Recarregue esta página para ver as comparações
        
        **O que o benchmark compara:**
        - 8 modelos diferentes de séries temporais
        - Métricas: MAE, RMSE, MAPE, R²
        - Seleção do melhor modelo baseado em desempenho
        """)
        
        # Mostrar estrutura esperada
        with st.expander("📋 Ver estrutura esperada do arquivo de comparação"):
            example_df = pd.DataFrame({
                'Modelo': ['Naive', 'Média Móvel', 'ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
                'MAE (Test)': [2.5, 2.1, 1.8, 1.6, 1.5, 1.4],
                'RMSE (Test)': [3.2, 2.8, 2.3, 2.1, 2.0, 1.9],
                'MAPE (Test)': [8.5, 7.2, 6.1, 5.5, 5.2, 4.9],
                'R² (Test)': [0.65, 0.72, 0.81, 0.85, 0.88, 0.90]
            })
            st.dataframe(example_df, use_container_width=True, hide_index=True)

# Rodapé
st.markdown("---")
st.markdown("**Dashboard de Análise de Séries Temporais - Dados Climáticos NOAA**")

