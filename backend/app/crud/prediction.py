from sqlalchemy.orm import Session
from ..db.base import PredictionResult
from typing import List

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
