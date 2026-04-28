from pydantic import BaseModel
from typing import List, Optional

class BacktestChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Grafico di Backtest).
    """
    labels: List[Optional[str]]
    actual: List[Optional[float]]
    prediction: List[Optional[float]]
    prediction_p70: List[Optional[float]]
    diff_instances: List[Optional[float]]
    prediction_rounded: List[Optional[int]]
    prediction_p70_rounded: List[Optional[int]]
    actual_rounded: List[Optional[int]]
    diff_rounded_instances: List[Optional[float]]
