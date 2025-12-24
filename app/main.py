import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "..", "models", "model.joblib")
)

app = FastAPI(
    title="ChurnInsight API",
    description="API de predição de churn com recomendação de negócio",
    version="1.0.0"
)

artifacts = {}

# =========================================================
# STARTUP — CARREGAMENTO DO MODELO
# =========================================================

@app.on_event("startup")
def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Modelo não encontrado no caminho esperado.")

    try:
        loaded = joblib.load(MODEL_PATH)

        artifacts["model"] = loaded["model"]
        artifacts["scaler"] = loaded["scaler"]
        artifacts["columns"] = loaded["columns"]
        artifacts["threshold"] = loaded.get("threshold", 0.35)
        artifacts["balance_median"] = loaded.get("balance_median", 0.0)
        artifacts["salary_median"] = loaded.get("salary_median", 0.0)

        print("✅ Modelo e artefatos carregados com sucesso.")

    except Exception as e:
        raise RuntimeError(f"Falha crítica no carregamento do modelo: {str(e)}")

# =========================================================
# CONTRATOS DE COMUNICAÇÃO (API CONTRACT)
# =========================================================

class CustomerInput(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    EstimatedSalary: float


class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str

# =========================================================
# FUNÇÕES DE APOIO (NEGÓCIO)
# =========================================================

def gerar_recomendacao(nivel_risco: str) -> str:
    if nivel_risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    elif nivel_risco == "MÉDIO":
        return "Monitoramento recomendado e campanhas de retenção"
    else:
        return "Cliente estável - manutenção padrão"

# =========================================================
# ENDPOINT DE PREVISÃO
# =========================================================

@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput) -> PredictionOutput:
    if "model" not in artifacts:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado."
        )

    try:
        # -------------------------------
        # 1. Entrada → DataFrame
        # -------------------------------
        df = pd.DataFrame([data.dict()])

        # -------------------------------
        # 2. Feature Engineering
        # -------------------------------
        df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
        df["Age_Tenure"] = df["Age"] * df["Tenure"]

        df["High_Value_Customer"] = (
            (df["Balance"] > artifacts["balance_median"]) &
            (df["EstimatedSalary"] > artifacts["salary_median"])
        ).astype(int)

        # -------------------------------
        # 3. One-Hot Encoding Manual
        # -------------------------------
        for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
            df[col] = 0

        if data.Geography == "Germany":
            df["Geography_Germany"] = 1
        elif data.Geography == "Spain":
            df["Geography_Spain"] = 1

        if data.Gender == "Male":
            df["Gender_Male"] = 1

        # -------------------------------
        # 4. Ordenação das features
        # -------------------------------
        df_final = df[artifacts["columns"]]

        # -------------------------------
        # 5. Escalonamento
        # -------------------------------
        X_input = artifacts["scaler"].transform(df_final)

        # -------------------------------
        # 6. Predição
        # -------------------------------
        proba = float(
            artifacts["model"].predict_proba(X_input)[0, 1]
        )

        threshold = artifacts["threshold"]

        if proba >= threshold:
            nivel_risco = "ALTO"
        elif proba >= threshold * 0.7:
            nivel_risco = "MÉDIO"
        else:
            nivel_risco = "BAIXO"

        previsao = (
            "Vai cancelar"
            if proba >= threshold
            else "Vai continuar"
        )

        return {
            "previsao": previsao,
            "probabilidade": round(proba, 4),
            "nivel_risco": nivel_risco,
            "recomendacao": gerar_recomendacao(nivel_risco)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {str(e)}"
        )
