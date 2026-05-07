from pydantic import BaseModel
from typing import Optional

class BacktestMetrics(BaseModel):
    rmse: float
    mse: float
    mae: float
    r2: float
    mse_pid: Optional[float] = None
    rmse_pid: Optional[float] = None
    mae_pid: Optional[float] = None
    r2_pid: Optional[float] = None
