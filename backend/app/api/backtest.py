from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime
from typing import Optional

from ..db.session import get_db
from ..services.backtest import backtest_service
from ..repositories.backtest_repository import BacktestRepository
from ..schemas.backtest_chart import BacktestChart
from ..schemas.backtest_metrics import BacktestMetrics
from ..services.data_processing import data_processing_service

router = APIRouter()

@router.get("/history", response_model=BacktestChart)
async def get_backtest_history(
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Recupera lo storico dei backtest eseguiti.
    """
    repo = BacktestRepository(db)
    chart_data = backtest_service.get_backtest_chart_data(repo, limit, start_date, end_date)
    
    if not chart_data.labels:
        raise HTTPException(status_code=404, detail="Nessun dato trovato")

    return chart_data

@router.delete("/")
async def clear_backtests(db: Session = Depends(get_db)):
    """
    Elimina tutta la cronologia dei backtest.
    """
    repo = BacktestRepository(db)
    try:
        count = repo.delete_all()
        return {"message": f"Successo: eliminate {count} righe", "status": "cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la pulizia: {str(e)}")

@router.post("/run")
async def run_backtest_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint per caricare un CSV, eseguire un backtest e salvare i risultati.
    """
    TARGET = "InUseCapacity"
    repo = BacktestRepository(db)
    
    try:
        contents = await file.read()
        
        # 1. Pulizia e Validazione Dati (Riutilizzo lo stesso service del Forecast)
        df = data_processing_service.process_aws_csv(contents, TARGET)
        
        # 2. Esecuzione Backtest (con metriche in-memory)
        res = backtest_service.run_backtest_pipeline(df, TARGET, repo)
        
        return {
            "message": "Backtest completato con successo",
            "results_saved": res["count"],
            "metrics": res["metrics"],
            "status": "success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore durante il backtest: {str(e)}")
