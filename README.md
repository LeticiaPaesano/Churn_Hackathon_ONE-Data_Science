<h1 id="inicio" align="center">
  ChurnInsight — Data Science<br>
  <img src="https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow" alt="Status" width="180" height="30" />
  <img src="https://img.shields.io/badge/Versão-1.2.1-blue" alt="Versão" width="100" height="30" />
</h1>

<h2 align="center">🔗 Repositórios Relacionados</h2>

O **ChurnInsight** é uma solução ecossistêmica. Este repositório foca exclusivamente na inteligência de dados e modelagem preditiva.

🌐 **Ecossistema do Projeto:**
*   👉 [**ChurnInsight — Backend**](https://github.com/renancvitor/churninsight-backend-h12-25b) (Node.js / Integração)
*   👉 [**ChurnInsight — Frontend**](https://github.com/lucasns06/churninsight-frontend) (Interface do Usuário)

---

### 🚀 API em Produção (Swagger UI)
🔗 **[https://churn-hackathon.onrender.com/docs](https://churn-hackathon.onrender.com/docs)**

⚠️ **Nota para o Squad:** A documentação interativa em `/docs` é a **Single Source of Truth** para o contrato da API. Verifique sempre os schemas antes de integrar.

---

<h2 align="center">📑 Sumário</h2>

*   [Visão Geral do Projeto](#visao-geral)
*   [Propósito do Repositório](#proposito)
*   [Abordagem de Data Science](#abordagem)
*   [Tecnologias e Ferramentas](#tecnologias)
*   [Estrutura do Repositório](#estrutura)
*   [Dicionário de Dados](#dicionario)
*   [Fonte dos Dados](#fonte-dados)
*   [Integração com o Backend](#integracao)
*   [Métricas e Resultados](#metricas)
*   [Primeiros Entregáveis](#entregaveis)
*   [Decisões Técnicas](#decisoes)
*   [Como Executar a API](#como-executar)
*   [Deploy com Docker](#deploy)
*   [Contribuições](#contribuicoes)

---

<h2 id="visao-geral" align="center">Visão Geral do Projeto</h2>

Desenvolvido para o **Hackathon da Alura**, o ChurnInsight utiliza Machine Learning para antecipar o cancelamento de clientes. O diferencial desta camada é não apenas dizer *quem* vai sair, mas oferecer o **porquê** (explicabilidade) e **o que fazer** (recomendação estratégica).

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="proposito" align="center">Propósito do Repositório</h2>

Este repositório centraliza:
*   A exploração estatística e tratamento de dados.
*   O treinamento de modelos robustos e exportação de pipelines.
*   A **API de Inferência** que serve o modelo para o mundo real.
*   Garantia de qualidade via **Testes Automatizados**.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="abordagem" align="center">Abordagem de Data Science</h2>       

### 🔹 1. Pré-processamento
*   Limpeza de metadados (`RowNumber`, `CustomerId`, `Surname`).
*   **One-Hot Encoding** para variáveis geográficas e de gênero.
*   Normalização rigorosa com `StandardScaler` (protegido contra *data leakage*).

### 🔹 2. Engenharia de Features
Criação de indicadores de comportamento:
*   `Age_Tenure`: Interação entre maturidade e fidelidade.
*   `Balance_Salary_Ratio`: Proporção de acúmulo financeiro vs ganho estimado.
*   `High_Value_Customer`: Flag para clientes acima da mediana financeira.

### 🔹 3. Modelagem e Explicabilidade
*   **Modelo:** `RandomForestClassifier` (200 árvores).
*   **Estratégia:** Pesos balanceados (`1:3`) para focar no Churn.
*   **Inovação:** Implementação de **Explicabilidade Local**. Se um cliente tem alto risco, a API identifica quais variáveis (ex: Idade, Saldo) foram determinantes para essa pontuação.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="tecnologias" align="center">Tecnologias e Ferramentas</h2>

As tecnologias previstas incluem:

- **🐍 Python** 3 — linguagem base da solução
- **📊 pandas** 2.3.3 e **numpy** 2.4.0 — manipulação e análise de dados
- **🤖 scikit-learn** 1.8.0 — modelagem, pré-processamento e métricas
- **💾 joblib** 1.5.3 — serialização do pipeline de Machine Learning
- **🌐 FastAPI** 0.127.0 — API REST para inferência do modelo
- **🔧 Uvicorn** 0.40.0 — servidor ASGI para execução da API
- **📦 pyarrow** 22.0.0 — leitura e escrita de dados em formato Parquet

### Ferramentas de Apoio
- **🧪 Jupyter Notebook / Google Colab** — EDA, experimentação e prototipação
- **🔗 Git & GitHub** — versionamento de código e colaboração
- **🐳 Docker & Docker Compose** — padronização de ambiente e deploy
- **☁️ Render** — hospedagem e execução da API em produção
  
---

<h2 id="estrutura" align="center">Estrutura do Repositório</h2>

```plaintext
app/
└── models/
| └── model.joblib                # Pipeline de ML (Modelo + Scaler)
├── __init__.py
└── main.py                       # API FastAPI

data/
├── Churn.csv                     # Dados brutos (origem)
└── dataset.parquet               # Dados tratados (pós-EDA e features)

notebooks/
└── Churn_Hackathon.ipynb         # EDA + Modelagem

tests/
├── __init__.py
├── conftest.py
├── stress_test.py
├── test_api.py
├── test_health.py
├── test_integration_previsao.py
├── test_unit_utils.py
└── teste_unit_explicabilidade.py

.gitignore
Dockerfile
README.md
docker-compose.yml
requirements.txt
```

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---
<h2 id="dicionario" align="center">Dicionário de Dados</h2>

| Coluna        | Descrição                         | Faixa Esperada                           |
|---------------|-----------------------------------|------------------------------------------|
| CreditScore   | Score financeiro do cliente       | 0 – 1000                                 |
| Geography     | País de origem do cliente         | France, Germany, Spain                   |
| Age           | Idade do cliente                  | 18 – 92 anos                             |
| Tenure        | Anos de relacionamento            | 0 – 10 anos                              |
| Balance       | Saldo em conta                    | R$ 0 – 500.000                           |
| Exited        | Target (indicador de churn)       | 1 = Sim (churn) / 0 = Não (permanece)    |

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="fonte-dados" align="center">Fonte dos Dados</h2>

Dataset público via Kaggle: **[Willian Oliveira](https://www.kaggle.com/datasets/willianoliveiragibin/customer-churn/data/code)** 

Base utilizada: Customer Churn new.csv.

---

<h2 id="integracao" align="center">Integração com o Backend</h2>

A API valida os dados antes de processar. Entradas fora do limite retornam ``HTTP 422``.

📥 Entrada

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 5,
  "Balance": 60000,
  "EstimatedSalary": 80000
}

```

📤 Saída

```json
{
  "previsao": "Vai continuar",
  "probabilidade": 0.24,
  "nivel_risco": "BAIXO",
  "recomendacao": "Cliente estável - manutenção padrão"
}

```
⚠️ O contrato final será validado em conjunto com o squad Back-end.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="metricas" align="center">Métricas e Resultados do Modelo (Teste)</h2>

O modelo final foi avaliado em uma base de teste (dados nunca vistos pelo modelo) para garantir sua capacidade de generalização. Abaixo, os indicadores de performance utilizando o **Threshold estratégico de 0.35**:

| Métrica              | Valor      |
| :--------------------| :--------- |
| **ROC-AUC**          | **0.7669** |
| **Acurácia**         | **79.00%** |
| **Recall (Churn)**   | **47.91%** |
| **Precisão (Churn)** | **48.39%** |

* 👉[**visualização técnica dos gráficos**](https://github.com/LeticiaPaesano/Churn_Hackathon/blob/main/docs/Documenta%C3%A7%C3%A3o%20T%C3%A9cnica%20de%20Visualiza%C3%A7%C3%B5es.md)

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="entregaveis" align="center">Primeiros Entregáveis do Squad</h2>

Rascunho dos principais entregáveis iniciais:

✅ **Concluídos:**

✅ Notebook EDA + Modelagem Final.
✅ API FastAPI v1.2.1 com Explicabilidade.
✅ Pipeline Serializado.
✅ Suite de Testes Automatizados.
✅ Dockerização Concluída.
⏳ Apresentação Final do Squad.

**Esses itens serão refinados com o decorrer do hackathon.**

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="decisoes" align="center">Decisões Técnicas</h2>

| Decisão            | Motivo                                      | Impacto                                         |
|--------------------|---------------------------------------------|-------------------------------------------------|
| Random Forest      | Melhor tratamento de relações não lineares  | Maior robustez e estabilidade do modelo         |
| Threshold 0.35     | Priorização da captura de clientes em risco | Aumento do Recall e redução de falsos negativos |
| Explicabilidade    | Necessidade de transparência no CRM         | Adoção de princípios de IA responsável          |


<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="como-executar" align="center">Como Executar a API de Modelo</h2>

1️⃣ Via Docker (Recomendado):

```docker-compose up --build```

Acesse:

API: 

```http://localhost:8000```

Swagger UI: 

```http://localhost:8000/docs```

2️⃣ Via Python Local (Desenvolvimento)

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
O parâmetro --reload deve ser utilizado apenas em ambiente de desenvolvimento.

Rodar Testes Automatizados
```pytest -v```

---

<h2 id="deploy" align="center">Deploy com Docker e Render</h2>

A API é empacotada via Docker e publicada automaticamente no Render Cloud.

**Endpoints Importantes**

Health Check: 

```GET /health```

Documentação (Swagger): 

```/docs```

**Produção**

```https://churn-hackathon.onrender.com/docs```

⚠️ A documentação em /docs é a fonte oficial e viva do contrato da API.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="contribuicoes" align="center">Contribuições</h2>

Contribuições do squad - Para colaborar:
1. Crie uma branch (git checkout -b feature/nome-da-feature)
2. Faça suas alterações
3. Envie um Pull Request descrevendo o que foi modificado

Durante o hackathon, manteremos comunicação constante para evitar conflitos ou trabalho duplicado.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>
