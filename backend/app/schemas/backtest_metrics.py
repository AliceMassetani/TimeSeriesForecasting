from pydantic import BaseModel

class BacktestMetrics(BaseModel):
    rmse: float
    mse: float
    mae: float
    r2: float
