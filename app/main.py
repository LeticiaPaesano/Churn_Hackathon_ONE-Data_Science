from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Literal, Optional, List
from pathlib import Path

import os
import logging
import joblib
import pandas as pd
import numpy as np
import tempfile


# =========================================================
# LOGGING
# =========================================================
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================
# PATHS / ARTIFACTS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
artifacts: dict = {}

# =========================================================
# LIFESPAN 
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega artefatos do modelo ao iniciar a aplicação."""
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    # Define chaves obrigatórias no artefato
    required_keys = {
        "model",
        "scaler",
        "columns",
        "threshold_cost",
        "balance_median",
        "salary_median",
        "raw_columns",
        "numeric_stats"
    }

    missing = required_keys - set(loaded.keys())
    if missing:
        raise RuntimeError(f"Artefato inválido. Chaves ausentes: {missing}")

    artifacts.update(loaded)
    yield
    artifacts.clear()


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="ChurnInsight API",
    version="3.2.0",
    lifespan=lifespan
)

# =========================================================
# UTILIDADES
# =========================================================
def gerar_recomendacao(nivel_risco: str) -> str:
    """Gera recomendação baseada no nível de risco."""
    mapa = {
        "ALTO": "Ação imediata recomendada: contato ativo e oferta personalizada",
        "MÉDIO": "Monitoramento recomendado e campanhas de retenção",
        "BAIXO": "Cliente estável - manutenção padrão"
    }
    return mapa.get(nivel_risco, "Manutenção padrão")


def calcular_explicabilidade_local(
    model,
    X: np.ndarray,
    feature_names: List[str],
    baseline_proba: float,
    input_data: dict
) -> List[str]:
    """
    Calcula as 3 features mais impactantes para a previsão (explicabilidade).
    Usa perturbação: remove uma feature por vez e mede mudança na probabilidade.
    """
    
    mapeamento = {
        "CreditScore": "CreditScore",
        "Age": "Age",
        "Tenure": "Tenure",
        "Balance": "Balance",
        "EstimatedSalary": "EstimatedSalary",
        "Geography_Germany": "Geography",
        "Geography_Spain": "Geography",
        "Gender_Male": "Gender",
        "Balance_Salary_Ratio": "Balance",
        "Age_Tenure": "Age",
        "High_Value_Customer": "Balance"
    }

    impactos = []
    for i, feature in enumerate(feature_names):
        X_mod = X.copy()
        X_mod[0, i] = 0
        proba_mod = model.predict_proba(X_mod)[0, 1]
        impactos.append((feature, abs(baseline_proba - proba_mod)))

    # Ordena por impacto decrescente
    impactos = sorted(impactos, key=lambda x: x[1], reverse=True)

    features_saida = []
    for feat, _ in impactos:
        contrato = mapeamento.get(feat)
        if not contrato:
            continue

        if contrato in ("Geography", "Gender"):
            valor = input_data.get(contrato)
            if valor and valor not in features_saida:
                features_saida.append(valor)
        else:
            if contrato not in features_saida:
                features_saida.append(contrato)

        if len(features_saida) >= 3:
            break

    return features_saida[:3]


# =========================================================
# SCHEMAS PYDANTIC
# =========================================================
class CustomerInput(BaseModel):
    """Schema de entrada para previsão individual."""
    CreditScore: int = Field(..., ge=350, le=900, description="Score de crédito entre 350-900")
    Geography: Literal["France", "Germany", "Spain"] = Field(..., description="País: France, Germany ou Spain")
    Gender: Literal["Male", "Female"] = Field(..., description="Gênero: Male ou Female")
    Age: int = Field(..., ge=18, le=92, description="Idade entre 18-92")
    Tenure: int = Field(..., ge=0, le=10, description="Anos como cliente: 0-10")
    Balance: float = Field(..., ge=0, description="Saldo em conta")
    EstimatedSalary: float = Field(..., ge=0, description="Salário estimado")


class PredictionOutput(BaseModel):
    """Schema de saída da previsão."""
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str
    explicabilidade: Optional[List[str]] = None


class HealthOutput(BaseModel):
    """Schema de saída do health check."""
    status: str
    model_version: Optional[str]
    threshold_cost: Optional[float]
    features_count: Optional[int]


# =========================================================
# ENDPOINT: PREVISÃO INDIVIDUAL
# =========================================================
@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput) -> PredictionOutput:
    """
    Realiza previsão de churn para um único cliente.
    
    Entrada: CustomerInput com dados do cliente
    Saída: PredictionOutput com previsão, probabilidade, risco e recomendação
    """
    
    if not artifacts:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se o arquivo model.joblib existe."
        )

    # Converte entrada para DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # One-Hot Encoding
    df["Geography_Germany"] = int(data.Geography == "Germany")
    df["Geography_Spain"] = int(data.Geography == "Spain")
    df["Gender_Male"] = int(data.Gender == "Male")

    # Feature Engineering (usa medianas do treino)
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    
    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts["balance_median"]) &
        (df["EstimatedSalary"] > artifacts["salary_median"])
    ).astype(int)

    # Garante que todas as colunas existem (ordem final)
    for col in artifacts["columns"]:
        if col not in df:
            df[col] = 0

    df = df[artifacts["columns"]]

    # Normalização
    X_scaled = artifacts["scaler"].transform(df)

    # Predição
    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts["threshold_cost"]

    # Classifica risco (APENAS 3 níveis)
    if proba >= threshold:
        risco = "ALTO"
        previsao = "Vai cancelar"
    elif proba >= 0.6 * threshold:
        risco = "MÉDIO"
        previsao = "Vai continuar"
    else:
        risco = "BAIXO"
        previsao = "Vai continuar"

    # Explicabilidade apenas para risco ALTO
    explicabilidade = None
    if risco == "ALTO":
        explicabilidade = calcular_explicabilidade_local(
            artifacts["model"],
            X_scaled,
            artifacts["columns"],
            proba,
            input_dict
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=round(proba, 4),
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade
    )


# =========================================================
# ENDPOINT: PREVISÃO EM LOTE
# =========================================================
@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...)):
    """
    Realiza previsão em lote a partir de arquivo CSV.
    
    Esperado: CSV com colunas: CreditScore, Geography, Gender, Age, Tenure, Balance, EstimatedSalary
    Retorna: CSV com previsões + flagging de erros/outliers
    """
    
    if not artifacts:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado"
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Arquivo deve ser CSV"
        )

    df = pd.read_csv(file.file)
    colunas_necessarias = artifacts["raw_columns"]
    colunas_faltantes = set(colunas_necessarias) - set(df.columns)

    if colunas_faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colunas ausentes: {sorted(list(colunas_faltantes))}"
        )

    threshold = artifacts["threshold_cost"]
    resultados = []

    for idx, row in df.iterrows():
        problemas = []

        # Valida valores nulos
        for col in colunas_necessarias:
            if pd.isna(row[col]):
                problemas.append(f"Valor nulo em {col}")

        # Detecta outliers (z-score > 3)
        for col, stats in artifacts["numeric_stats"].items():
            if col in row and not pd.isna(row[col]):
                z = abs((row[col] - stats["mean"]) / (stats["std"] + 1e-6))
                if z > 3:
                    problemas.append(f"Outlier em {col}")

        erro_linha = len(problemas) > 0
        previsao = "Erro"
        risco = "Erro"
        proba = None
        explicabilidade = None

        if not erro_linha:
            try:
                input_dict = row.to_dict()
                df_linha = pd.DataFrame([input_dict])

                # One-Hot Encoding
                df_linha["Geography_Germany"] = int(row["Geography"] == "Germany")
                df_linha["Geography_Spain"] = int(row["Geography"] == "Spain")
                df_linha["Gender_Male"] = int(row["Gender"] == "Male")

                # Feature Engineering
                df_linha["Balance_Salary_Ratio"] = row["Balance"] / (row["EstimatedSalary"] + 1)
                df_linha["Age_Tenure"] = row["Age"] * row["Tenure"]
                
                df_linha["High_Value_Customer"] = int(
                    row["Balance"] > artifacts["balance_median"] and
                    row["EstimatedSalary"] > artifacts["salary_median"]
                )

                # Garante colunas
                for col in artifacts["columns"]:
                    if col not in df_linha:
                        df_linha[col] = 0

                df_linha = df_linha[artifacts["columns"]]
                X_scaled = artifacts["scaler"].transform(df_linha)

                proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])

                # Classifica
                if proba >= threshold:
                    risco = "ALTO"
                    previsao = "Vai cancelar"
                elif proba >= 0.6 * threshold:
                    risco = "MÉDIO"
                    previsao = "Vai continuar"
                else:
                    risco = "BAIXO"
                    previsao = "Vai continuar"

                # Explicabilidade para ALTO
                if risco == "ALTO":
                    explicabilidade = calcular_explicabilidade_local(
                        artifacts["model"],
                        X_scaled,
                        artifacts["columns"],
                        proba,
                        input_dict
                    )

            except Exception as e:
                previsao = "Erro"
                risco = "Erro"
                proba = None
                problemas.append(str(e))

        resultados.append({
            **row.to_dict(),
            "previsao": previsao,
            "probabilidade": round(proba, 4) if proba is not None else None,
            "nivel_risco": risco,
            "explicabilidade": "|".join(explicabilidade) if explicabilidade else None,
            "erro_linha": erro_linha,
            "detalhes_erro": "||".join(problemas) if problemas else None
        })

    df_resultado = pd.DataFrame(resultados)

    # Salva em temp
    nome_saida = file.filename.replace(".csv", "_previsionado.csv")
    output_path = Path(tempfile.gettempdir()) / nome_saida
    df_resultado.to_csv(output_path, index=False)

    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=nome_saida
    )


# =========================================================
# ENDPOINT: HEALTH CHECK
# =========================================================
@app.get("/health", response_model=HealthOutput)
def health() -> HealthOutput:
    """Verifica status da API e do modelo carregado."""
    return HealthOutput(
        status="UP",
        model_version=artifacts.get("model_version"),
        threshold_cost=artifacts.get("threshold_cost"),
        features_count=len(artifacts.get("columns", []))
    )


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    """Rota raiz com informações da API."""
    return {
        "nome": "ChurnInsight API",
        "versao": "3.2.0",
        "endpoints": {
            "GET /": "Info da API",
            "GET /health": "Status do modelo",
            "POST /previsao": "Previsão individual",
            "POST /previsao-lote": "Previsão em lote (CSV)"
        }
    }
