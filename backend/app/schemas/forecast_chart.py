from pydantic import BaseModel
from typing import List, Optional

class ForecastChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Grafico di Forecast futuro).
    """
    labels: List[Optional[str]]
    prediction: List[Optional[float]]
    prediction_p70: List[Optional[float]]
    prediction_rounded: List[Optional[int]]
    prediction_p70_rounded: List[Optional[int]]
    actual: List[Optional[int]]
