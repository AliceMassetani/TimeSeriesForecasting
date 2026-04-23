from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime
from typing import Optional

from ..db.session import get_db
from ..db.base import PredictionResult
from ..services.forecasting import forecasting_service
from ..crud.prediction import delete_all_predictions, get_predictions
from ..schemas.chart import PredictionChart

router = APIRouter()

@router.get("/history", response_model=PredictionChart)
async def get_prediction_history(
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Recupera lo storico delle previsioni formattato per grafici (Chart.js / Angular).
    Filtra per numero di righe (limit) o intervallo di date.
    """
    # 1. Recupero dati dal DB via CRUD (SQLAlchemy)
    db_results = get_predictions(db, limit, start_date, end_date)
    
    if not db_results:
        raise HTTPException(status_code=404, detail="Nessun dato trovato per i filtri selezionati")

    # 2. Trasformazione veloce in DataFrame Pandas
    # Estraiamo i dati dagli oggetti ORM in un formato tabellare
    data = []
    for r in db_results:
        data.append({
            "timestamp": r.timestamp,
            "actual": r.actual_value,
            "prediction": r.prediction,
            "diff": r.diff_instances,
            "pred_round": r.prediction_rounded,
            "act_round": r.actual_rounded,
            "diff_round": r.diff_rounded_instances
        })
    
    df = pd.DataFrame(data)

    # 3. Restituzione del formato PredictionChart (liste separate)
    return PredictionChart(
        labels=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        actual=df['actual'].tolist(),
        prediction=df['prediction'].tolist(),
        diff_instances=df['diff'].tolist(),
        prediction_rounded=df['pred_round'].tolist(),
        actual_rounded=df['act_round'].tolist(),
        diff_rounded_instances=df['diff_round'].tolist()
    )

@router.delete("/")
async def clear_predictions(db: Session = Depends(get_db)):
    """
    Elimina tutta la cronologia delle previsioni.
    """
    try:
        count = delete_all_predictions(db)
        return {"message": f"Successo: eliminate {count} righe", "status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la pulizia: {str(e)}")

@router.post("/predict")
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint per caricare un CSV, eseguire il forecasting e salvare i risultati.
    """
    TARGET = "InUseCapacity"
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), parse_dates=['TimeStamp'])
        count = forecasting_service.run_prediction_pipeline(df, TARGET, db)
        return {
            "message": "Operazione completata con successo",
            "predictions_saved": count,
            "status": "success"
        }
    except Exception as e:
        # Gestione errori centralizzata
        raise HTTPException(status_code=400, detail=f"Errore durante la pipeline: {str(e)}")
