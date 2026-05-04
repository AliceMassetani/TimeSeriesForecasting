from pydantic import BaseModel
from typing import List, Optional

class ForecastChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Grafico di Forecast futuro).
    """
    labels: List[Optional[str]]
    prediction: List[Optional[float]]
    prediction_p10: List[Optional[float]]
    prediction_p90: List[Optional[float]]
    prediction_rounded: List[Optional[int]]
    prediction_p10_rounded: List[Optional[int]]
    prediction_p90_rounded: List[Optional[int]]
    actual: List[Optional[int]]
