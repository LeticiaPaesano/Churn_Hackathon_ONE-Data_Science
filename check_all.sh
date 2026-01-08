#!/bin/bash
echo "--------------------------------------------------"
echo "í´ 1. VALIDANDO SINTAXE"
python -m compileall . -q || exit 1

echo "--------------------------------------------------"
echo "í·ª 2. RODANDO TESTES (PYTEST)"
# O 'python -m pytest' resolve o erro de 'No module named app' no Windows
export PYTHONPATH=$PYTHONPATH:.
python -m pytest -v -W ignore || exit 1

echo "--------------------------------------------------"
echo "íº€ 3. TESTANDO API E ESTRESSE LOCAL"
# Inicia a API em background
python -m uvicorn app.main:app --port 8000 > api_log_temp.txt 2>&1 &
API_PID=$!

echo "Aguardando 10 segundos para carga do modelo..."
sleep 10 

# Executa o teste de estresse
python stress_test.py

echo "--------------------------------------------------"
echo "í»‘ FINALIZANDO"
# Tenta fechar a API de forma limpa no Windows
taskkill //F //PID $API_PID 2>/dev/null || kill -9 $API_PID 2>/dev/null
rm api_log_temp.txt
echo "âœ… REPOSITÃ“RIO TOTALMENTE VALIDADO NO WINDOWS!"
