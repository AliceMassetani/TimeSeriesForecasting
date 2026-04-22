from pydantic import BaseModel, ConfigDict
from datetime import datetime

class PredictionResultBase(BaseModel):
    """
    Schema base per i risultati delle previsioni.
    Definisce i campi comuni tra creazione e lettura.
    """
    timestamp: datetime
    prediction: float
    actual_value: float
    diff_percentage: float
    prediction_rounded: int
    actual_rounded: int
    diff_rounded_percentage: float

class PredictionResultCreate(PredictionResultBase):
    """
    Schema per la creazione di un nuovo risultato.
    Al momento identico alla base, ma utile per espansioni future.
    """
    pass

class PredictionResult(PredictionResultBase):
    """
    Schema per la lettura dei risultati (include l'ID del database).
    """
    id: int

    # Configurazione per Pydantic v2 per permettere la lettura da oggetti ORM (SQLAlchemy)
    model_config = ConfigDict(from_attributes=True)
