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
    
    # Nuove metriche di business (Originale)
    under_count: Optional[int] = 0
    over_count: Optional[int] = 0
    under_sum: Optional[float] = 0.0
    over_sum: Optional[float] = 0.0

    # Nuove metriche di business (PID)
    under_count_pid: Optional[int] = 0
    over_count_pid: Optional[int] = 0
    under_sum_pid: Optional[float] = 0.0
    over_sum_pid: Optional[float] = 0.0

    # Metriche Kalman
    mse_kalman: Optional[float] = None
    rmse_kalman: Optional[float] = None
    mae_kalman: Optional[float] = None
    r2_kalman: Optional[float] = None
    under_count_kalman: Optional[int] = 0
    over_count_kalman: Optional[int] = 0
    under_sum_kalman: Optional[float] = 0.0
    over_sum_kalman: Optional[float] = 0.0
