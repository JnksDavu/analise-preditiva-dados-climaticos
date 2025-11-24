#!/usr/bin/env python3
"""
Script para processar arquivo PSV do GHCNh e converter para CSV diário
"""

import pandas as pd
import numpy as np
import os
import sys

def process_ghcnh_psv(input_file, output_file=None):
    """
    Processa arquivo PSV do GHCNh e converte para CSV diário.
    
    Parâmetros:
    -----------
    input_file : str
        Caminho para o arquivo .psv
    output_file : str, opcional
        Caminho para o arquivo de saída .csv (padrão: base do input + .csv)
        
    Retorna:
    --------
    pd.DataFrame com dados processados
    """
    try:
        print("=" * 60)
        print("PROCESSAMENTO DE ARQUIVO PSV - GHCNh")
        print("=" * 60)
        print(f"\nArquivo de entrada: {input_file}")
        
        # Verificar se arquivo existe
        if not os.path.exists(input_file):
            print(f"❌ Erro: Arquivo não encontrado: {input_file}")
            return None
        
        # Ler arquivo PSV
        print(f"\n📖 Lendo arquivo PSV...")
        df = pd.read_csv(input_file, sep='|', low_memory=False)
        print(f"  Total de registros horários: {len(df):,}")
        
        # Informações sobre a estação
        if 'Station_name' in df.columns:
            station_name = df['Station_name'].iloc[0] if len(df) > 0 else "N/A"
            print(f"  Estação: {station_name}")
        
        # Converter coluna DATE para datetime
        if 'DATE' not in df.columns:
            print("❌ Erro: Coluna 'DATE' não encontrada")
            return None
        
        print(f"  Convertendo datas...")
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Informações sobre o período
        print(f"  Período: {df['DATE'].min()} a {df['DATE'].max()}")
        
        # Verificar coluna de temperatura
        if 'temperature' not in df.columns:
            print("❌ Erro: Coluna 'temperature' não encontrada")
            return None
        
        # Filtrar valores válidos de temperatura
        df_valid = df[df['temperature'].notna()].copy()
        print(f"  Registros com temperatura válida: {len(df_valid):,} ({100*len(df_valid)/len(df):.1f}%)")
        
        if len(df_valid) == 0:
            print("❌ Erro: Nenhum registro válido de temperatura encontrado")
            return None
        
        # Agregar dados horários para diários
        print(f"\n📊 Agregando dados horários para diários...")
        df_valid['date'] = df_valid['DATE'].dt.date
        
        # Agregar por dia
        daily_agg = {
            'temperature': ['mean', 'min', 'max'],
        }
        
        # Adicionar outras colunas se disponíveis
        if 'precipitation' in df_valid.columns:
            daily_agg['precipitation'] = lambda x: x.sum() if x.notna().any() else np.nan
        
        if 'wind_speed' in df_valid.columns:
            daily_agg['wind_speed'] = 'mean'
        
        if 'relative_humidity' in df_valid.columns:
            daily_agg['relative_humidity'] = 'mean'
        
        if 'dew_point_temperature' in df_valid.columns:
            daily_agg['dew_point_temperature'] = 'mean'
        
        daily_df = df_valid.groupby('date').agg(daily_agg).reset_index()
        
        # Flatten column names
        new_columns = ['date']
        for col in daily_agg.keys():
            if col == 'temperature':
                new_columns.extend(['temperature', 'temp_min', 'temp_max'])
            else:
                new_columns.append(col)
        
        daily_df.columns = new_columns
        
        # Converter date de date para datetime
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Ordenar por data
        daily_df = daily_df.sort_values('date').reset_index(drop=True)
        
        # Estatísticas
        print(f"\n✓ Dados processados: {len(daily_df)} dias")
        print(f"  Período: {daily_df['date'].min().date()} a {daily_df['date'].max().date()}")
        print(f"  Temperatura média: {daily_df['temperature'].mean():.2f} °C")
        print(f"  Temperatura mínima: {daily_df['temp_min'].min():.2f} °C")
        print(f"  Temperatura máxima: {daily_df['temp_max'].max():.2f} °C")
        
        # Salvar arquivo de saída
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + '_daily.csv'
        
        # Salvar apenas colunas principais para compatibilidade
        output_df = daily_df[['date', 'temperature']].copy()
        output_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Arquivo salvo: {output_file}")
        print(f"  Total: {len(output_df)} registros")
        
        return daily_df
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Arquivo padrão
    default_file = "data/GHCNh_AAI0000TNCA_2025.psv"
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        input_file = default_file
        output_file = None
    
    # Processar arquivo
    result = process_ghcnh_psv(input_file, output_file)
    
    if result is not None:
        print("\n✓ Processamento concluído com sucesso!")
        sys.exit(0)
    else:
        print("\n❌ Falha no processamento")
        sys.exit(1)

