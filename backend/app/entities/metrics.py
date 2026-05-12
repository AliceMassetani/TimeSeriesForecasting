from sqlalchemy import Column, Integer, Float
from ..db.base_class import Base

class Metrics(Base):
    """
    Tabella 'metrics' per memorizzare l'ultimo risultato del backtest.
    """
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, default=1)
    mae = Column(Float)
    rmse = Column(Float)
    mse = Column(Float)
    r2 = Column(Float)
    mse_pid = Column(Float, nullable=True)
    rmse_pid = Column(Float, nullable=True)
    mae_pid = Column(Float, nullable=True)
    r2_pid = Column(Float, nullable=True)

    # Nuove metriche di business (Originale)
    under_count = Column(Integer, nullable=True, default=0)
    over_count = Column(Integer, nullable=True, default=0)
    under_sum = Column(Float, nullable=True, default=0.0)
    over_sum = Column(Float, nullable=True, default=0.0)

    # Nuove metriche di business (PID)
    under_count_pid = Column(Integer, nullable=True, default=0)
    over_count_pid = Column(Integer, nullable=True, default=0)
    under_sum_pid = Column(Float, nullable=True, default=0.0)
    over_sum_pid = Column(Float, nullable=True, default=0.0)

    # Metriche Kalman
    mse_kalman = Column(Float, nullable=True)
    rmse_kalman = Column(Float, nullable=True)
    mae_kalman = Column(Float, nullable=True)
    r2_kalman = Column(Float, nullable=True)
    under_count_kalman = Column(Integer, nullable=True, default=0)
    over_count_kalman = Column(Integer, nullable=True, default=0)
    under_sum_kalman = Column(Float, nullable=True, default=0.0)
    over_sum_kalman = Column(Float, nullable=True, default=0.0)
