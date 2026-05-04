from pydantic import BaseModel
from typing import List, Optional
from .backtest_metrics import BacktestMetrics

class BacktestChart(BaseModel):
    """
    Schema ottimizzato per il frontend (Grafico di Backtest).
    """
    labels: List[Optional[str]]
    actual: List[Optional[float]]
    prediction: List[Optional[float]]
    prediction_p10: List[Optional[float]]
    prediction_p90: List[Optional[float]]
    diff_instances: List[Optional[float]]
    prediction_rounded: List[Optional[int]]
    prediction_p10_rounded: List[Optional[int]]
    prediction_p90_rounded: List[Optional[int]]
    actual_rounded: List[Optional[int]]
    diff_rounded_instances: List[Optional[float]]
    metrics: Optional[BacktestMetrics] = None
