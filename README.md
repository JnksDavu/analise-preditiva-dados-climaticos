# analise-preditiva-dados-climaticos

# Previsão de Dados Climáticos (São Paulo / NOAA)

Projeto acadêmico do Grupo 6 para análise e previsão de temperatura média (TAVG) e probabilidade de chuva (RAIN_PROB) usando dados históricos (arquivo CSV NOAA).

## Estrutura

```
analise-preditiva-dados-climaticos/
  data/
    4174560.csv
  server/
    train_and_eval.py   (treino e avaliação)
    api.py              (API FastAPI de previsão)
    artifacts/          (modelos e meta gerados após treino)
  client/
    front.py            (dashboard Streamlit)
  requierements.txt     (dependências)
```

## Requisitos

- Python 3.11 (recomendado)
- Pip atualizado

## 1. Criar e Ativar Ambiente Virtual

Dentro da pasta do projeto:

```bash
python3 -m venv venv
source venv/bin/activate
```

(Para sair depois: `deactivate`)

## 2. Instalar Dependências

Se já existir `requierements.txt`, use:

```bash
pip install -r requierements.txt
```

(Se renomear para `requirements.txt`, ajuste o comando.)

## 3. Treinar os Modelos

Gera os artefatos em `server/artifacts/`:

```bash
python server/train_and_eval.py
```

Saída esperada inclui métricas de temperatura e chuva.

## 4. Subir a API

```bash
python server/api.py
```

A API ficará em `http://localhost:8000`.

### Testar endpoint de previsão (exemplo 10 dias)

Em outro terminal (ainda com venv ativo):

```bash
curl "http://localhost:8000/predict?days=10"
```

Retorno JSON com lista de objetos contendo DATE, TAVG_PRED e RAIN_PROB.

## 5. Executar o Dashboard

Em novo terminal (ou mesmo) com venv ativo:

```bash
streamlit run client/front.py
```

Acesse o endereço indicado (geralmente `http://localhost:8501`).

## 6. Fluxo Resumido

1. Criar venv
2. Ativar venv
3. Instalar dependências
4. Treinar: `python server/train_and_eval.py`
5. Subir API: `python server/api.py`
6. Abrir dashboard: `streamlit run client/front.py`

## 7. Endpoints

- `GET /predict?days=N`  
  Parâmetro `days`: inteiro (1 a 60). Retorna previsão diária futura.

## 8. Lógica de Modelagem

- Regressão Linear para previsão de TAVG.
- Regressão Logística (ou modelo dummy) para probabilidade de chuva.
- Features:
  - Codificação cíclica de dia do ano e mês (`SIN_DAY`, `COS_DAY`, `SIN_MONTH`, `COS_MONTH`)
  - Lags de temperatura e chuva (1, 2, 3, 7 dias)
  - Flag de chuva e precipitação agregada.
- Divisão temporal (treino 80%, teste 20%).

## 9. Artefatos Gerados

Após treino:
- `temperature_model.pkl`
- `precip_model.pkl`
- `meta.pkl` (informações de métricas e features)

Local: `server/artifacts/`

## 10. Atualização / Novo Treino

Se adicionar novos dados ao CSV:
```bash
python server/train_and_eval.py
```