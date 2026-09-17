# =============================================================================
# Router INTERNO — Comunicazione Service-to-Service (Lambda → Backend)
# =============================================================================
# Questi endpoint NON richiedono autenticazione JWT.
# Sono protetti a livello di rete: Nginx non inoltra il prefisso /internal/,
# quindi sono raggiungibili SOLO dai container sulla stessa rete Docker.
# =============================================================================

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services.forecast import forecast_service
from ..services.data_processing import data_processing_service
from ..repositories.forecast_repository import ForecastRepository

router = APIRouter()


@router.get("/forecast/latest-prediction")
async def internal_get_latest_prediction(db: Session = Depends(get_db)):
    """
    Restituisce la previsione futura più imminente.
    Chiamato dalla Lambda per impostare la DesiredCapacity su AppStream.
    """
    repo = ForecastRepository(db)
    row = repo.get_latest_prediction()

    if row is None:
        raise HTTPException(status_code=404, detail="Nessuna previsione futura disponibile")

    desired = (
        row.prediction_kalman_rounded
        or row.prediction_pid_rounded
        or row.prediction_p90_rounded
        or row.prediction_rounded
    )

    return {
        "desired_capacity": desired,
        "timestamp": row.timestamp.isoformat(),
        "prediction_p50": row.prediction,
        "prediction_p90": row.prediction_p90,
        "prediction_kalman": row.prediction_kalman,
        "prediction_pid": row.prediction_pid,
    }


@router.post("/forecast/run")
async def internal_run_forecast(
    file: UploadFile = File(...),
    history_hours: int = -1,
    quantile: float = 0.9,
    db: Session = Depends(get_db)
):
    """
    Endpoint interno per la Lambda: carica CSV e genera previsioni.
    """
    TARGET = "InUseCapacity"
    repo = ForecastRepository(db)

    try:
        contents = await file.read()
        df = data_processing_service.process_aws_csv(contents, TARGET)
        res = forecast_service.run_forecast_pipeline(df, TARGET, repo, history_hours=history_hours, quantile=quantile)

        return {
            "message": f"Forecast completato! Storico: {res['history']} ore, Previsioni: {res['predictions']} ore.",
            "status": "success",
            "target": TARGET,
            "details": res
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la generazione del forecast: {str(e)}")
