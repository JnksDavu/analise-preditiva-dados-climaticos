#!/usr/bin/env python3
"""
Script para processar dados climáticos da NOAA a partir de arquivo PSV local
Usa apenas arquivo local, sem conexão com internet
"""

import pandas as pd
import numpy as np
import os

def load_ghcnh_psv(file_path):
    """
    Carrega e processa arquivo PSV do GHCNh (Global Historical Climatology Network - Hourly).
    
    Parâmetros:
    -----------
    file_path : str
        Caminho para o arquivo .psv
        
    Retorna:
    --------
    pd.DataFrame com dados processados (diário) ou None se falhar
    """
    try:
        print(f"Carregando arquivo PSV: {file_path}...")
        
        # Ler arquivo PSV (pipe-separated values)
        df = pd.read_csv(file_path, sep='|', low_memory=False)
        print(f"  Total de registros: {len(df)}")
        
        # Converter coluna DATE para datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Extrair temperatura (já está em °C no GHCNh)
            if 'temperature' in df.columns:
                # Filtrar valores válidos (remover NaN)
                df = df[df['temperature'].notna()].copy()
                
                # Agregar dados horários para diários
                # Criar coluna de data (sem hora)
                df['date'] = df['DATE'].dt.date
                
                # Agregar por dia
                daily_df = df.groupby('date').agg({
                    'temperature': ['mean', 'min', 'max'],
                    'precipitation': lambda x: x.sum() if x.notna().any() else np.nan,
                    'wind_speed': 'mean',
                    'relative_humidity': 'mean',
                    'dew_point_temperature': 'mean'
                }).reset_index()
                
                # Flatten column names
                daily_df.columns = ['date', 'temperature', 'temp_min', 'temp_max', 
                                   'precipitation', 'wind_speed', 'relative_humidity', 'dew_point']
                
                # Converter date de date para datetime
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                
                # Selecionar colunas principais
                result_df = daily_df[['date', 'temperature', 'temp_min', 'temp_max']].copy()
                
                # Remover linhas onde temperatura média é NaN
                result_df = result_df[result_df['temperature'].notna()].copy()
                result_df = result_df.sort_values('date').reset_index(drop=True)
                
                print(f"✓ Dados processados: {len(result_df)} dias")
                print(f"  Período: {result_df['date'].min()} a {result_df['date'].max()}")
                print(f"  Temperatura média: {result_df['temperature'].mean():.2f} °C")
                
                return result_df
            else:
                print("❌ Coluna 'temperature' não encontrada no arquivo")
                return None
        else:
            print("❌ Coluna 'DATE' não encontrada no arquivo")
            return None
            
    except Exception as e:
        print(f"❌ Erro ao processar arquivo PSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_synthetic_data():
    """Cria dados sintéticos como fallback"""
    print("Criando dados sintéticos...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    trend = np.linspace(15, 18, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 2, n)
    temperature = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature
    })
    
    print(f"✓ Dados sintéticos criados: {len(df)} registros")
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("PROCESSAMENTO DE DADOS CLIMÁTICOS - NOAA")
    print("=" * 60)
    
    # Usar apenas arquivo PSV local
    psv_file = os.getenv('PSV_FILE', 'data/GHCNh_AAI0000TNCA_2025.psv')
    
    if os.path.exists(psv_file):
        print(f"\n📁 Arquivo PSV local encontrado: {psv_file}")
        df = load_ghcnh_psv(psv_file)
        
        if df is not None and len(df) > 0:
            # Salvar em formato CSV para uso no dashboard
            output_file = 'noaa_data.csv'
            # Manter apenas colunas necessárias para compatibilidade
            df_output = df[['date', 'temperature']].copy()
            df_output.to_csv(output_file, index=False)
            print(f"\n✓ Dados salvos em: {output_file}")
            print(f"  Total: {len(df_output)} registros")
            print(f"  Período: {df_output['date'].min()} a {df_output['date'].max()}")
            exit(0)
        else:
            print("\n❌ Erro: Não foi possível processar o arquivo PSV")
            exit(1)
    else:
        print(f"\n❌ Erro: Arquivo PSV não encontrado: {psv_file}")
        print("\nPor favor, coloque o arquivo PSV na pasta 'data/'")
        print("Exemplo: data/GHCNh_AAI0000TNCA_2025.psv")
        exit(1)

