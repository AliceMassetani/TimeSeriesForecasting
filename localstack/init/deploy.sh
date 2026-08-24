#!/bin/bash
# =============================================================================
# LocalStack Init Script — Provisioning automatico
# =============================================================================
# Questo script viene eseguito automaticamente all'avvio di LocalStack.
# Crea: Lambda + EventBridge (trigger orario) + Metriche fake in CloudWatch.
# =============================================================================

set -euo pipefail

echo "========================================="
echo " Provisioning Predictive Autoscaler"
echo "========================================="

# --- Variabili ---
FUNCTION_NAME="predictive-autoscaler"
RULE_NAME="autoscaler-hourly"
REGION="eu-central-1"
LAMBDA_CODE_DIR="/etc/localstack/lambda-code"
FLEET_NAME="SAAS_FLEET_PROD"
HISTORY_HOURS=740

# --- 1. Impacchetta la Lambda ---
echo "[1/5] Impacchettamento Lambda..."
cd "$LAMBDA_CODE_DIR"
pip install -r requirements.txt -t . --quiet
zip -r /tmp/lambda.zip . -x "__pycache__/*" "*.pyc"
echo "       ZIP creato: $(du -h /tmp/lambda.zip | cut -f1)"

# --- 2. Crea la funzione Lambda ---
echo "[2/5] Creazione funzione Lambda..."
awslocal lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime python3.12 \
    --handler lambda_function.handler \
    --zip-file fileb:///tmp/lambda.zip \
    --role arn:aws:iam::000000000000:role/lambda-role \
    --timeout 180 \
    --memory-size 256 \
    --environment "Variables={
        ENVIRONMENT=local,
        BACKEND_URL=http://backend:8000,
        FLEET_NAME=$FLEET_NAME,
        HISTORY_HOURS=$HISTORY_HOURS,
        LOCALSTACK_URL=http://localstack:4566,
        AWS_DEFAULT_REGION=$REGION
    }" \
    --region "$REGION" || awslocal lambda update-function-code --function-name "$FUNCTION_NAME" --zip-file fileb:///tmp/lambda.zip --region "$REGION" || true
echo "       Funzione '$FUNCTION_NAME' pronta."

# --- 3. Crea regola EventBridge (trigger orario) ---
echo "[3/5] Creazione regola EventBridge..."
awslocal events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "rate(1 hour)" \
    --state ENABLED \
    --region "$REGION"
echo "       Regola '$RULE_NAME' creata (rate: 1 hour)."

# --- 4. Collega EventBridge → Lambda ---
echo "[4/5] Collegamento EventBridge → Lambda..."
LAMBDA_ARN=$(awslocal lambda get-function \
    --function-name "$FUNCTION_NAME" \
    --query 'Configuration.FunctionArn' \
    --output text \
    --region "$REGION")

awslocal events put-targets \
    --rule "$RULE_NAME" \
    --targets "Id=1,Arn=$LAMBDA_ARN" \
    --region "$REGION"
echo "       EventBridge → Lambda collegato."

# --- 5. Inserisci metriche fake in CloudWatch ---
echo "[5/5] Inserimento $HISTORY_HOURS ore di metriche fake in CloudWatch..."

# Genera metriche orarie per HISTORY_HOURS ore
# Valori realistici: basso di notte (0-5), alto di giorno (30-75)
METRIC_DATA=""
CURRENT_TIME=$(date -u +%s)

for i in $(seq 0 $((HISTORY_HOURS - 1))); do
    TIMESTAMP=$((CURRENT_TIME - (HISTORY_HOURS - i) * 3600))
    HOUR=$(date -u -d "@$TIMESTAMP" +%H 2>/dev/null || date -u -r "$TIMESTAMP" +%H)
    
    # Simula pattern realistico: basso di notte, alto di giorno
    if [ "$HOUR" -ge 8 ] && [ "$HOUR" -le 18 ]; then
        VALUE=$((RANDOM % 46 + 30))  # 30-75 (orario lavorativo)
    elif [ "$HOUR" -ge 6 ] && [ "$HOUR" -le 20 ]; then
        VALUE=$((RANDOM % 21 + 10))  # 10-30 (transizione)
    else
        VALUE=$((RANDOM % 6))         # 0-5 (notte)
    fi
    
    ISO_TIME=$(date -u -d "@$TIMESTAMP" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$TIMESTAMP" +%Y-%m-%dT%H:%M:%SZ)
    
    METRIC_DATA="$METRIC_DATA MetricName=InUseCapacity,Dimensions=[{Name=Fleet,Value=$FLEET_NAME}],Timestamp=$ISO_TIME,Value=$VALUE,Unit=Count"
    
    # CloudWatch accetta max 1000 datapoint per chiamata — flush ogni 500
    if [ $(((i + 1) % 500)) -eq 0 ]; then
        awslocal cloudwatch put-metric-data \
            --namespace "AWS/AppStream" \
            --metric-data $METRIC_DATA \
            --region "$REGION" 2>/dev/null
        METRIC_DATA=""
        echo "       ...inseriti $((i + 1))/$HISTORY_HOURS datapoint"
    fi
done

# Flush rimanenti
if [ -n "$METRIC_DATA" ]; then
    awslocal cloudwatch put-metric-data \
        --namespace "AWS/AppStream" \
        --metric-data $METRIC_DATA \
        --region "$REGION" 2>/dev/null
fi

echo "       $HISTORY_HOURS metriche fake inserite."

echo "========================================="
echo " Provisioning completato!"
echo " Per invocare: awslocal lambda invoke"
echo "   --function-name $FUNCTION_NAME /tmp/out.json"
echo "========================================="
