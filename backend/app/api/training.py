from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from ..services.training import training_service
from ..services.forecast import forecast_service
from ..services.backtest import backtest_service

router = APIRouter()

@router.post("/")
async def train_model(
    file: UploadFile = File(...),
    input_chunk_len: Optional[int] = Form(None),
    output_chunk_len: Optional[int] = Form(None),
    n_epochs: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    hidden_size: Optional[int] = Form(None),
    ff_size: Optional[int] = Form(None),
    num_blocks: Optional[int] = Form(None),
    dropout: Optional[float] = Form(None),
    learning_rate: Optional[float] = Form(None)
):
    """
    Endpoint per addestrare un nuovo modello CANDIDATO (sfidante).
    """
    TARGET = "InUseCapacity"
    params = {
        "input_chunk_len": input_chunk_len,
        "output_chunk_len": output_chunk_len,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "ff_size": ff_size,
        "num_blocks": num_blocks,
        "dropout": dropout,
        "learning_rate": learning_rate
    }
    
    print(f"DEBUG: Parametri ricevuti dall'API: {params}")
    
    try:
        contents = await file.read()
        result = training_service.run_training_pipeline(contents, TARGET, params=params)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'addestramento: {str(e)}")

@router.post("/load-candidate")
async def load_candidate():
    """
    Carica temporaneamente il modello CANDIDATO nei service per testarlo.
    """
    forecast_service.load_model(model_type="candidate")
    backtest_service.load_model(model_type="candidate")
    
    return {
        "message": "Modello CANDIDATO caricato con successo per test.",
        "active_model": "candidate"
    }

@router.post("/load-champion")
async def load_champion():
    """
    Ripristina il modello CHAMPION (produzione) nei service.
    """
    forecast_service.load_model(model_type="champion")
    backtest_service.load_model(model_type="champion")
    
    return {
        "message": "Modello CHAMPION ripristinato con successo.",
        "active_model": "champion"
    }

@router.post("/promote")
async def promote_model():
    """
    Promuove il modello candidato a CHAMPION sovrascrivendo quello attuale.
    """
    # 1. Promozione fisica dei file
    result = training_service.promote_candidate()
    
    # 2. Caricamento forzato del nuovo champion nei service
    forecast_service.load_model(model_type="champion")
    backtest_service.load_model(model_type="champion")
    
    return result

@router.post("/rollback")
async def rollback_model():
    """
    Ripristina il Champion precedente dal backup e lo ricarica.
    """
    result = training_service.rollback_champion()
    
    # Ricarica il modello ripristinato nei service
    forecast_service.load_model(model_type="champion")
    backtest_service.load_model(model_type="champion")
    
    return result
