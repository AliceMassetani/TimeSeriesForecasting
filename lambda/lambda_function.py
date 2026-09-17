# =============================================================================
# Lambda Function — Predictive Autoscaler
# =============================================================================
# Questa Lambda viene triggerata da EventBridge (schedule) e:
#   1. Raccoglie le metriche InUseCapacity da CloudWatch
#   2. Le invia al Backend come CSV per generare il forecast
#   3. Legge la previsione più imminente dal Backend
#   4. Imposta la DesiredCapacity su AppStream (o logga in dry-run)
#
# Se la generazione del forecast fallisce, usa la previsione della
# run precedente come fallback (le vecchie previsioni restano nel DB).
# =============================================================================

import os
import json
import logging
import boto3
import urllib.request
import urllib.error
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configurazione via variabili d'ambiente ---
ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")            # "local" | "production"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
FLEET_NAME = os.environ.get("FLEET_NAME", "SAAS_FLEET_PROD")
HISTORY_HOURS = int(os.environ.get("HISTORY_HOURS", "740"))
LOCALSTACK_URL = os.environ.get("LOCALSTACK_URL", "http://localstack:4566")


# =============================================================================
# Step 1 — Raccolta metriche da CloudWatch
# =============================================================================
def get_cloudwatch_metrics():
    """
    Chiama CloudWatch per ottenere le ultime HISTORY_HOURS ore di InUseCapacity.
    In locale usa LocalStack, in produzione usa gli endpoint AWS standard.
    """
    endpoint_url = LOCALSTACK_URL if ENVIRONMENT == "local" else None
    cw = boto3.client("cloudwatch", endpoint_url=endpoint_url)

    response = cw.get_metric_data(
        MetricDataQueries=[{
            "Id": "inuse",
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/AppStream",
                    "MetricName": "InUseCapacity",
                    "Dimensions": [{"Name": "Fleet", "Value": FLEET_NAME}]
                },
                "Period": 3600,
                "Stat": "Average"
            },
            "ReturnData": True
        }],
        StartTime=datetime.utcnow() - timedelta(hours=HISTORY_HOURS),
        EndTime=datetime.utcnow()
    )

    timestamps = response["MetricDataResults"][0]["Timestamps"]
    values = response["MetricDataResults"][0]["Values"]

    logger.info(f"CloudWatch: ricevuti {len(timestamps)} data-point")
    return list(zip(timestamps, values))


# =============================================================================
# Step 2 — Formattazione metriche come CSV
# =============================================================================
def format_as_csv(metrics):
    """
    Converte la lista di (timestamp, value) in bytes CSV.
    Il formato è identico ai CSV caricati manualmente dal frontend.
    """
    csv = "TimeStamp,InUseCapacity\n"
    for ts, val in sorted(metrics, key=lambda x: x[0]):
        csv += f"{ts.strftime('%Y/%m/%d %H:%M:%S')},{val}\n"
    return csv.encode("utf-8")


# =============================================================================
# Step 3 — Upload CSV al backend (POST /forecast/run)
# =============================================================================
def trigger_forecast(csv_bytes):
    """
    Invia il CSV al backend come multipart file upload.
    Usa l'endpoint interno senza autenticazione (POST /internal/forecast/run).
    """
    url = f"{BACKEND_URL}/internal/forecast/run?history_hours=-1&quantile=0.9"

    boundary = "----LambdaBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="cloudwatch.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode())

    logger.info(f"Forecast generato: {result.get('message', result)}")
    return result


# =============================================================================
# Step 4 — Lettura previsione (GET /forecast/latest-prediction)
# =============================================================================
def get_prediction():
    """
    Chiama il Backend FastAPI per ottenere la prossima previsione.
    Ritorna il dizionario con desired_capacity e i dettagli.
    """
    url = f"{BACKEND_URL}/internal/forecast/latest-prediction"
    logger.info(f"Chiamata al backend: {url}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())

    logger.info(f"Risposta dal backend: {data}")
    return data


# =============================================================================
# Step 5 — Scaling AppStream
# =============================================================================
def update_fleet_capacity(desired_capacity: int):
    """
    Imposta la DesiredCapacity sul fleet AppStream.
    In ambiente 'local' esegue un dry-run (solo log).
    """
    if ENVIRONMENT == "production":
        client = boto3.client("appstream")
        response = client.update_fleet(
            Name=FLEET_NAME,
            ComputeCapacity={
                "DesiredInstances": desired_capacity
            }
        )
        logger.info(f"AppStream fleet aggiornato: {response}")
        return {"action": "updated", "fleet": FLEET_NAME, "desired": desired_capacity}
    else:
        # --- DRY-RUN: logga senza toccare AppStream ---
        logger.info(
            f"[DRY-RUN] Avrei impostato {desired_capacity} istanze "
            f"sul fleet '{FLEET_NAME}'"
        )
        return {"action": "dry-run", "fleet": FLEET_NAME, "desired": desired_capacity}


# =============================================================================
# Handler — Entry point con fallback
# =============================================================================
def handler(event, context):
    """
    Entry point della Lambda. Triggerata da EventBridge ogni ora.
    Se la generazione del forecast fallisce, usa la previsione precedente.
    """
    logger.info(f"Lambda invocata. Ambiente: {ENVIRONMENT}")
    logger.info(f"Evento ricevuto: {json.dumps(event, default=str)}")

    try:
        # 1. Raccolta metriche da CloudWatch
        metrics = get_cloudwatch_metrics()
        csv_bytes = format_as_csv(metrics)

        # 2. Genera forecast (può fallire — il fallback usa le previsioni precedenti)
        try:
            trigger_forecast(csv_bytes)
        except Exception as e:
            logger.warning(f"Forecast fallito, uso previsione precedente: {e}")

        # 3. Legge la previsione (nuova o fallback dalla run precedente)
        prediction = get_prediction()
        desired_capacity = prediction.get("desired_capacity")

        if desired_capacity is None:
            raise ValueError("Il backend non ha restituito 'desired_capacity'")

        # 4. Applica la capacità (o dry-run)
        result = update_fleet_capacity(int(desired_capacity))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Autoscaling eseguito con successo",
                "prediction": prediction,
                "result": result
            })
        }

    except urllib.error.URLError as e:
        logger.error(f"Errore di connessione al backend: {e}")
        return {
            "statusCode": 502,
            "body": json.dumps({"error": f"Backend non raggiungibile: {str(e)}"})
        }
    except Exception as e:
        logger.error(f"Errore nella Lambda: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
