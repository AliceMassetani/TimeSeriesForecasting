from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
import csv
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
async def run_backtest_upload(
    file: UploadFile = File(...),
    pid_kp: Optional[float] = Form(None),
    pid_ki: Optional[float] = Form(None),
    pid_kd: Optional[float] = Form(None),
    pid_exp: Optional[float] = Form(None),
    pid_scale_down: Optional[float] = Form(None),
    pid_quantile: Optional[float] = Form(None),
    pid_max_derivative: Optional[float] = Form(None),
    pid_acceleration_factor: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Endpoint per caricare un CSV, eseguire un backtest e salvare i risultati.
    Accetta parametri PID e quantile opzionali (default da config).
    """
    TARGET = "InUseCapacity"
    repo = BacktestRepository(db)
    
    try:
        contents = await file.read()
        
        # 1. Pulizia e Validazione Dati (Riutilizzo lo stesso service del Forecast)
        df = data_processing_service.process_aws_csv(contents, TARGET)
        
        # 2. Esecuzione Pipeline
        res = backtest_service.run_backtest_pipeline(
            df, TARGET, repo,
            pid_kp=pid_kp, pid_ki=pid_ki, pid_kd=pid_kd,
            pid_exp=pid_exp, pid_scale_down=pid_scale_down,
            pid_quantile=pid_quantile,
            pid_max_derivative=pid_max_derivative,
            pid_acceleration_factor=pid_acceleration_factor
        )
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
    # Validazione Parametri
    if limit is not None and limit <= 0:
        raise HTTPException(status_code=400, detail="Limit deve essere maggiore di 0")
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="La data di inizio deve essere precedente a quella di fine")

    repo = BacktestRepository(db)
    chart_data = backtest_service.get_backtest_chart_data(repo, limit, start_date, end_date)
    
    if not chart_data.labels:
        raise HTTPException(status_code=404, detail="Nessun dato trovato")

    return chart_data

@router.get("/export-csv")
async def export_backtest_csv(mode: str = "all", db: Session = Depends(get_db)):
    """
    Esporta i risultati del backtest in formato CSV.
    Mode: 'original', 'pid', 'all'
    """
    repo = BacktestRepository(db)
    results = repo.get_all()
    
    if not results:
        raise HTTPException(status_code=404, detail="Nessun dato di backtest trovato.")

    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header dinamico
    header = ["Timestamp", "Actual_Value"]
    if mode == "original" or mode == "all":
        header.extend(["Original_Prediction", "Diff_Original"])
    if mode == "pid" or mode == "all":
        header.extend(["PID_Prediction", "Diff_PID"])
    
    writer.writerow(header)
    
    # Dati
    for r in results:
        row = [r.timestamp, r.actual_value]
        if mode == "original" or mode == "all":
            row.extend([r.prediction, (r.prediction - r.actual_value) if r.actual_value is not None else 0])
        if mode == "pid" or mode == "all":
            row.extend([r.prediction_pid, (r.prediction_pid - r.actual_value) if r.actual_value is not None and r.prediction_pid is not None else 0])
        writer.writerow(row)
    
    output.seek(0)
    
    filename = f"backtest_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )