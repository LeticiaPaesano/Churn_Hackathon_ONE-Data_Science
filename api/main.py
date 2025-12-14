import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# --- CONFIGURAÇÃO ---
MODEL_PATH = "api/model.joblib"

app = FastAPI(
    title="ChurnInsight API",
    description="API de Previsão de Churn usando Random Forest",
    version="1.0.0"
)

# Variáveis globais para armazenar os artefatos carregados
artifacts = {}

@app.on_event("startup")
def load_model():
    """Carrega o modelo .joblib ao iniciar a API"""
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
        return

    try:
        loaded = joblib.load(MODEL_PATH)
        artifacts["model"] = loaded["model"]
        artifacts["scaler"] = loaded["scaler"]
        artifacts["columns"] = loaded["columns"]
        artifacts["threshold"] = loaded.get("threshold", 0.35)
        artifacts["balance_median"] = loaded.get("balance_median", 0.0)
        artifacts["salary_median"] = loaded.get("salary_median", 0.0)
        print("✅ Modelo e artefatos carregados com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")

# --- CONTRATO DE DADOS ---
class CustomerInput(BaseModel):
    CreditScore: int
    Geography: str  # 'France', 'Germany', 'Spain'
    Gender: str     # 'Male', 'Female'
    Age: int
    Tenure: int
    Balance: float
    EstimatedSalary: float

class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str

# --- ENDPOINT DE PREVISÃO ---
@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        # 1. Converter entrada para DataFrame
        df = pd.DataFrame([data.dict()])

        # 2. Feature Engineering (Replicando lógica de treino)
        # Razão Saldo/Salário
        df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
        # Interação Idade/Fidelidade
        df['Age_Tenure'] = df['Age'] * df['Tenure']
        # Cliente de Alto Valor
        df['High_Value_Customer'] = (
            (df['Balance'] > artifacts["balance_median"]) & 
            (df['EstimatedSalary'] > artifacts["salary_median"])
        ).astype(int)

        # 3. One-Hot Encoding Manual
        # Cria colunas zeradas
        for col in ['Geography_Germany', 'Geography_Spain', 'Gender_Male']:
            df[col] = 0
            
        # Preenche conforme regra
        if data.Geography == 'Germany': df['Geography_Germany'] = 1
        if data.Geography == 'Spain': df['Geography_Spain'] = 1
        if data.Gender == 'Male': df['Gender_Male'] = 1

        # 4. Selecionar colunas na ordem correta
        df_final = df[artifacts["columns"]]

        # 5. Escalar dados
        X_input = artifacts["scaler"].transform(df_final)

        # 6. Previsão
        proba = float(artifacts["model"].predict_proba(X_input)[0, 1])
        is_churn = proba >= artifacts["threshold"]

        return {
            "previsao": "Vai cancelar" if is_churn else "Vai continuar",
            "probabilidade": round(proba, 4),
            "nivel_risco": "ALTO" if proba > 0.5 else "BAIXO"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
