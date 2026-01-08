#!/bin/bash
echo "--------------------------------------------------"
echo "🔍 1. VERIFICANDO SINTAXE"
python3 -m compileall . -q || exit 1

echo "--------------------------------------------------"
echo "🧪 2. RODANDO PYTEST (INTEGRAÇÃO E UNIDADE)"
export PYTHONPATH=$PYTHONPATH:.
pytest -W ignore || exit 1

echo "--------------------------------------------------"
echo "🚀 3. TESTANDO ESTRESSE LOCAL"
# Sobe a API temporariamente
python3 -m uvicorn app.main:app --port 8000 > api_log_temp.txt 2>&1 &
API_PID=$!
sleep 5

# Executa o teste de estresse que agora está na RAIZ
python3 stress_test.py

echo "--------------------------------------------------"
echo "�� FINALIZANDO"
kill $API_PID
rm api_log_temp.txt
echo "✅ REPOSITÓRIO TOTALMENTE VALIDADO E ORGANIZADO!"
