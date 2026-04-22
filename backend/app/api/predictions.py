from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import io

from ..db.session import get_db
from ..services.forecasting import forecasting_service

router = APIRouter()

@router.post("/predict")
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint per caricare un CSV e generare previsioni.
    L'intera pipeline (calcolo + salvataggio) è gestita dal Service Layer.
    """
    TARGET = "InUseCapacity"
    
    try:
        # 1. API Layer: Ricezione dati
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), parse_dates=['TimeStamp'])
        
        # 2. Service Layer: Esecuzione dell'intera pipeline (unica chiamata)
        count = forecasting_service.run_prediction_pipeline(df, TARGET, db)
            
        return {
            "message": "Operazione completata con successo",
            "predictions_saved": count,
            "status": "success"
        }

    except Exception as e:
        # Gestione errori centralizzata
        raise HTTPException(status_code=400, detail=f"Errore durante la pipeline: {str(e)}")
