<h1 id="inicio" align="center">ChurnInsight — Data Science <br>
<img src="https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow" width="180" height="30" />
</h1>

<h2 id="inicio" align="center">🔗 ChurnInsight — Data Science</h2>

Este repositório contém o desenvolvimento do modelo preditivo utilizado pela
plataforma **ChurnInsight**, criada durante o **Hackathon da Alura**. Aqui estão concentrados os trabalhos de **análise de dados, engenharia de features,
treinamento do modelo** e a **API** que serve as previsões para o sistema.

**🚀 [Acesse a Documentação da API (Swagger UI)](https://churn-hackathon.onrender.com/docs)**

**⚠️Atenção Backend:** Para testes e integração, utilize sempre o final `/docs` na URL de produção para visualizar os schemas atualizados.

**👉 [Backend — ChurnInsight](https://github.com/renancvitor/churninsight-backend-h12-25b)**


---

<h2 align="center">📑 Sumário</h2>

- [Visão Geral do Projeto](#visao-geral)
- [Propósito do Repositório](#proposito)
- [Abordagem Geral de Data Science](#abordagem)
- [Tecnologias e Ferramentas](#tecnologias)
- [Estrutura do Repositório](#estrutura)
- [Integração com o Backend](#integracao)
- [Primeiros Entregáveis do Squad](#entregaveis)
- [Pontos em Aberto / Decisões do Time](#decisoes)
- [Como Executar a API de Modelo](#como-executar)
- [Contribuições](#contribuicoes)

---

<h2 id="visao-geral" align="center">Visão Geral do Projeto</h2>

O **ChurnInsight** é uma solução desenvolvida para prever a probabilidade de cancelamento de clientes (churn) em serviços de assinatura.  
Este repositório abriga **toda a parte de Data Science**, incluindo análise exploratória, preparação de dados, treinamento do modelo e exposição de previsões via API Python.

A proposta para o hackathon é entregar um **MVP funcional**, permitindo que o backend consulte a probabilidade de churn a partir de um JSON enviado pelo cliente.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="proposito" align="center">Propósito do Repositório</h2>

Este repositório existe para consolidar:

- A análise dos dados utilizada pelo squad DS.
- O desenvolvimento do modelo preditivo.
- O armazenamento do modelo final exportado.
- A API Python responsável por expor previsões ao backend.
- A documentação mínima necessária para execução e integração.

Tudo aqui está em fase de definição conjunta do time. 

O objetivo inicial é estabelecer uma base clara e organizada para o desenvolvimento.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="abordagem" align="center">Abordagem Geral de Data Science</h2>

A abordagem adotada pelo time de Data Science para o MVP foi a seguinte:

- **Pré-Processamento:** Remoção de colunas de identificação (`RowNumber`, `CustomerId`, `Surname`) e codificação de variáveis categóricas (`Geography` e `Gender`) utilizando **One-Hot Encoding**.

- **Engenharia de Features**: Criação das variáveis compostas `Age_Tenure`, `Balance_Salary_Ratio` e a flag `High_Value_Customer`, calculada exclusivamente a partir das medianas do conjunto de treino.

- **Modelagem:** Utilização do algoritmo **Random Forest Classifier** (`n_estimators=200`, `class_weight={0:1, 1:3}`), priorizando robustez, estabilidade e interpretabilidade.

- **Desbalanceamento:** Ajuste de `class_weight={0: 1, 1: 3}` e definição de **threshold = 0.35**, priorizando Recall da classe churn.

- **Pipeline Completo:** Treinamento, validação e teste encapsulados em um único pipeline, evitando data leakage.

- **Serialização:** Pipeline final exportado com `joblib` em `models/model.joblib`.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="tecnologias" align="center">Tecnologias e Ferramentas</h2>

As tecnologias previstas incluem:

- **🐍 Python** 3

- **📊 pandas** 2.3.3 e **numpy** 2.4.0

- **🤖 scikit-learn** 1.8.0 — modelagem, pré-processamento e métricas

- **💾 joblib** 1.5.3 — serialização do pipeline de ML

- **🌐 FastAPI** 0.127.0 — API de inferência

- **🔧 Uvicorn** 0.40.0 — servidor ASGI

- **📦 pyarrow** 22.0.0 — leitura e escrita de dados em formato Parquet

Ferramentas de apoio:

- **🧪 Jupyter Notebook / Google Colab** — desenvolvimento, EDA e experimentação

- **🔗 Git & GitHub** — versionamento de código e colaboração em equipe

- **☁️ Render** — deploy e hospedagem da API em produção

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="estrutura" align="center">Estrutura do Repositório</h2>

A estrutura abaixo é um **ponto de partida** e deve evoluir conforme decisões do squad:

```plaintext
app/
 ├── __init__.py
 └── main.py              # API FastAPI 
 └── models/
     └── model.joblib     # Pipeline serializado
data/
 ├── Churn.csv            # Dados brutos (origem)
 └── dataset.parquet

notebooks/
 └── Churn_Hackathon.ipynb
.gitignore
README.md
requirements.txt
```
Links adicionais podem ser adicionados conforme a documentação evoluir.

*A estrutura final do repositório reflete a implantação da API*

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---
<h2 id="dicionario" align="center">Dicionário de Dados</h2>

### 📊 Dicionário de Dados

| Coluna Original   |                 Significado                 |
|-------------------|---------------------------------------------|
| RowNumber         | Número da linha no conjunto de dados.       |
| Customer ID       | Identificador único de cada cliente.        |
| Surname           | Sobrenome do cliente.                       |
| CreditScore       | Indicador financeiro |
| Geography         | Localização geográfica do cliente.          |
| Gender            | Gênero (Male/Female)           |
| Age               | Idade do cliente.                           |
| Tenure            | Tempo de permanência (0-10 anos).  |
| Balance           | Saldo em conta.                  |
| EstimatedSalary   | Estimativa de salário anual.           |
| Exited            | **Target:** 1=Churn, 0=Permanece (20.37%)|


<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="fonte-dados" align="center">Fonte dos Dados</h2>

Os dados utilizados neste projeto foram obtidos no Kaggle, no seguinte dataset público:

**🔗 Customer Churn [Willian Oliveira](https://www.kaggle.com/datasets/willianoliveiragibin/customer-churn/data/code)** 

O arquivo utilizado pelo squad DS é:

`Churn new.csv` 

---

<h2 id="integracao" align="center">Integração com o Backend</h2>

A comunicação entre DS e Backend ocorrerá via API Python, que deve receber um JSON contendo informações do cliente e retornar:

- previsão textual (“Vai cancelar” ou “Vai continuar”)
- probabilidade numérica associada ao churn

## ⚠️ Regras de Validação (Limites da API)
Para garantir a estabilidade, a API possui validações rigorosas. Dados fora destas faixas retornarão ```Erro 422```:

**Campo	Regra** / **Limite**
**CreditScore**	Inteiro entre 0 e 1000
**Age**	Entre 18 e 92 anos
**Tenure**	Entre 0 e 10 anos
**Balance**	Máximo de 500.000,00
**EstimatedSalary**	Entre 523.00 e 500.000,00

📥 Exemplo de Chamada Payload (sujeito a alterações)

📥 Entrada
```json
{
  "Surname": "Campbell",
  "CreditScore": 350,
  "Geography": "France",
  "Gender": "Male",
  "Age": 39,
  "Tenure": 0,
  "Balance": 109733.2,
  "EstimatedSalary": 123602.11
}

```

📤 Saída

```json
{
  "surname": "Campbell",
  "classificacao_score": "Regular",
  "previsao": "Vai cancelar",
  "probabilidade": 0.395,
  "nivel_risco": "ALTO",
  "recomendacao": "Ação imediata recomendada: contato ativo e oferta personalizada"
}

```

⚠️ O contrato final será validado em conjunto com o squad Back-end.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="Métricas" align="center">Métricas e Resultados do Modelo (Teste)</h2>

**ROC-AUC:** 0.7669

**Acurácia:** 79.00%

**Recall (Churn):** 47.91%

**Precisão (Churn):** 48.39%

**Threshold:** 0.35

**🎯 Critério de sucesso:** priorização do Recall para reduzir falsos negativos (clientes que cancelariam sem intervenção).

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---



<h2 id="entregaveis" align="center">Primeiros Entregáveis do Squad</h2>

Rascunho dos principais entregáveis iniciais:

✅ **Concluídos:**

- [x] Notebook completo com EDA + modelagem

- [x] Pipeline com features derivadas (sem leakage)  

- [x] **Modelo final serializado** (`model/model.joblib`)

- [x] API FastAPI funcional (Colab + ngrok)

- [x] Documentação com métricas e contrato JSON


⏳ **Em progresso:** Integração backend + apresentação

**Esses itens serão refinados com o decorrer do hackathon.**

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="decisoes" align="center">Pontos em Aberto / Decisões Pendentes</h2>

| Tema                  | Decisão Final                          | Impacto                          |
|-----------------------|----------------------------------------|----------------------------------|
| **Encoding**    | **One-Hot Encoding** (3 colunas dummy) | Melhor performance que LabelEnc |
| **Threshold**   | **0.35** (otimizado para Recall) | Recall 47.91% |Precisão 48.39%   |
| **Features leakage** | Medianas calculadas **apenas no treino** |Boas práticas ML garantidas |
| **Top Features** | Age (24.6%) > Salary (14.5%) > CreditScore | Foco estratégico correto |


 **🏆 Métricas Finais (Teste):** ROC-AUC 0.7669 | Acurácia 79.00%


**Estas decisões serão registradas neste README conforme forem tomadas.**

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="como-executar" align="center">Como Executar a API de Modelo</h2>

Estes são passos gerais necessários para rodar a API de previsões; poderão ser ajustados conforme a implementação:

1.  **Instalar dependências**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Rodar o serviço Uvicorn (na raiz do projeto)**
    ```bash
    uvicorn app.main:app --reload
    ```
    O parâmetro --reload é recomendado apenas para ambiente de desenvolvimento.
    
4.  **Acessar a Documentação**
    A API ficará disponível na porta 8000. Acesse a documentação interativa (Swagger UI) em:
    ```
    http://localhost:8000/docs
    ```

***Em ambientes como GitHub Codespaces, utilize o endereço público associado à porta 8000 e acrescente /docs ao final da URL.***

---

## 🌐 Conexão com o deploy em produção (Render)

É altamente recomendável **complementar essa seção** com um apontamento direto para produção, por exemplo:

```markdown
### 🚀 API em Produção

A aplicação também está disponível em ambiente de produção, hospedada na plataforma **Render**:

https://churn-hackathon.onrender.com/docs

```
<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="contribuicoes" align="center">Contribuições</h2>

Contribuições do squad - Para colaborar:
1. Crie uma branch (git checkout -b feature/nome-da-feature)
2. Faça suas alterações
3. Envie um Pull Request descrevendo o que foi modificado

Durante o hackathon, manteremos comunicação constante para evitar conflitos ou trabalho duplicado.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>
