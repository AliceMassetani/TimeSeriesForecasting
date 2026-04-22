from app.entities.prediction_result import PredictionResult
from app.schemas.chart import PredictionChart
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import io

from ..db.session import get_db
from ..services.forecasting import forecasting_service
from ..crud.prediction import create_predictions, get_predictions, delete_all_predictions
from ..schemas.chart import PredictionChart
from datetime import datetime
from typing import Optional

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
    
    # 2. Trasformazione in formato "colonnare" tramite il modello Pydantic
    chart_data = PredictionChart(
        labels=[],
        actual=[],
        prediction=[],
        diff_percentage=[],
        prediction_rounded=[],
        actual_rounded=[],
        diff_rounded_percentage=[]
    )
    
    for row in db_results:
        # Formattazione data "human readable"
        chart_data.labels.append(row.timestamp.strftime("%Y-%m-%d %H:%M"))
        chart_data.actual.append(row.actual_value)
        chart_data.prediction.append(row.prediction)
        chart_data.diff_percentage.append(row.diff_percentage)
        chart_data.prediction_rounded.append(row.prediction_rounded)
        chart_data.actual_rounded.append(row.actual_rounded)
        chart_data.diff_rounded_percentage.append(row.diff_rounded_percentage)
        
    return chart_data

@router.delete("/")
async def clear_predictions(db: Session = Depends(get_db)):
    """
    Elimina manualmente tutta la cronologia delle previsioni dal database.
    """
    try:
        count = delete_all_predictions(db)
        return {"message": f"Successo: eliminate {count} righe", "status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la pulizia: {str(e)}")

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

@router.get("/chart", response_model=PredictionChart)
async def get_chart(db: Session = Depends(get_db)):
    
    try:
        query = db.query(PredictionResult).order_by(PredictionResult.timestamp.asc())

        # Esegui la query e converti in DataFrame Pandas
        # Pandas "mappa" le colonne del DB automaticamente
        df = pd.read_sql(query.statement, query.session.bind)

        if df.empty:
            raise HTTPException(status_code=404, detail="Nessun risultato trovato")

        return PredictionChart(
            labels=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            actual=df['actual_value'].tolist(),
            prediction=df['prediction'].tolist(),
            diff_percentage=df['diff_percentage'].tolist(),
            prediction_rounded=df['prediction_rounded'].tolist(),
            actual_rounded=df['actual_rounded'].tolist(),
            diff_rounded_percentage=df['diff_rounded_percentage'].tolist()
        )
    except Exception as e:
        # Gestione errori centralizzata
        raise HTTPException(status_code=500, detail=f"Errore durante il recupero dei dati: {str(e)}")
