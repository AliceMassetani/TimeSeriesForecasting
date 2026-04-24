from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime
from typing import Optional

from ..db.session import get_db
from ..services.forecast import forecast_service
from ..repositories.forecast_repository import ForecastRepository
from ..schemas.forecast_chart import ForecastChart

router = APIRouter()

@router.get("/history", response_model=ForecastChart)
async def get_forecast_history(db: Session = Depends(get_db)):
    """
    Recupera la cronologia delle previsioni future generate.
    """
    repo = ForecastRepository(db)
    chart_data = forecast_service.get_forecast_chart_data(repo)
    
    if not chart_data.labels:
        raise HTTPException(status_code=404, detail="Nessun forecast trovato")

    return chart_data

@router.delete("/")
async def clear_forecasts(db: Session = Depends(get_db)):
    """
    Elimina tutta la cronologia delle previsioni (Forecast).
    """
    repo = ForecastRepository(db)
    try:
        count = repo.delete_all()
        return {"message": f"Successo: eliminate {count} righe", "status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la pulizia: {str(e)}")

@router.post("/run")
async def run_forecast_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Carica un CSV (dati storici recenti) per generare previsioni FUTURE per le prossime 24h.
    """
    TARGET = "InUseCapacity"
    repo = ForecastRepository(db)
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verifica colonne minime
        if "TimeStamp" not in df.columns or TARGET not in df.columns:
            raise HTTPException(status_code=400, detail=f"Il CSV deve contenere TimeStamp e {TARGET}")

        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Generiamo il forecast (default 2 ore)
        res = forecast_service.run_forecast_pipeline(df, TARGET, repo)
        
        return {
            "message": f"Forecast completato! Storico: {res['history']} ore, Previsioni: {res['predictions']} ore.",
            "status": "success",
            "target": TARGET,
            "details": res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la generazione del forecast: {str(e)}")
