from pydantic import BaseModel
from typing import List, Optional

class PredictionChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Angular/Chart.js).
    Ritorna i dati in formato "colonnare" ovvero liste di valori.
    """
    labels: List[Optional[str]]
    actual: List[Optional[float]]
    prediction: List[Optional[float]]
    diff_instances: List[Optional[float]]
    prediction_rounded: List[Optional[int]]
    actual_rounded: List[Optional[int]]
    diff_rounded_instances: List[Optional[float]]
