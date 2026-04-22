from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..entities.prediction_result import PredictionResult
from typing import List, Optional
from datetime import datetime

def create_predictions(db: Session, predictions: List[PredictionResult]):
    """
    Esegue l'inserimento bulk dei risultati della previsione nel database.
    """
    try:
        db.add_all(predictions)
        db.commit()
        return len(predictions)
    except Exception as e:
        db.rollback()
        raise e

def get_predictions(
    db: Session, 
    limit: Optional[int] = None, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> List[PredictionResult]:
    """
    Recupera i risultati delle previsioni con filtri opzionali.
    """
    query = db.query(PredictionResult)
    
    if start_date:
        query = query.filter(PredictionResult.timestamp >= start_date)
    if end_date:
        query = query.filter(PredictionResult.timestamp <= end_date)
    
    if limit:
        # Recupera gli ultimi N risultati (ordinando per data decrescente)
        results = query.order_by(desc(PredictionResult.timestamp)).limit(limit).all()
        # Inverte la lista per restituirli in ordine cronologico (ideale per i grafici)
        results.reverse()
        return results
    
    # Se non c'è limit, ritorna tutto in ordine cronologico
    return query.order_by(PredictionResult.timestamp).all()

def delete_all_predictions(db: Session) -> int:
    """
    Elimina tutti i record dalla tabella delle previsioni.
    Ritorna il numero di righe eliminate.
    """
    try:
        num_deleted = db.query(PredictionResult).delete()
        db.commit()
        return num_deleted
    except Exception as e:
        db.rollback()
        raise e
