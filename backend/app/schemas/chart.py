from pydantic import BaseModel
from typing import List

class PredictionChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Angular/Chart.js).
    Ritorna i dati in formato "colonnare" ovvero liste di valori.
    """
    labels: List[str]                  # Timestamp come stringhe leggibili
    actual: List[float]                # Valori reali
    prediction: List[float]            # Valori previsti
    diff_percentage: List[float]       # Scostamento %
    prediction_rounded: List[int]      # Valori previsti arrotondati
    actual_rounded: List[int]          # Valori reali arrotondati
    diff_rounded_percentage: List[float] # Scostamento % (arrotondato)
