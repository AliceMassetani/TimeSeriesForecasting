from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class BacktestResultBase(BaseModel):
    """
    Schema base per i risultati del backtest.
    """
    timestamp: datetime
    prediction: float
    actual_value: Optional[float] = None
    diff_instances: Optional[float] = None
    prediction_rounded: int
    actual_rounded: Optional[int] = None
    diff_rounded_instances: Optional[float] = None

class BacktestResultCreate(BacktestResultBase):
    pass

class BacktestResult(BacktestResultBase):
    id: int
    model_config = ConfigDict(from_attributes=True)
