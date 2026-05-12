from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base_class import Base

class BacktestResult(Base):
    """
    Rappresenta i risultati di un Backtest (confronto storico).
    """
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[float] = mapped_column(Float) # Questo rimane il P50
    prediction_p10: Mapped[float] = mapped_column(Float, nullable=True)
    prediction_p90: Mapped[float] = mapped_column(Float, nullable=True)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    diff_instances: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_rounded: Mapped[int] = mapped_column(Integer) # Arrotondamento P50
    prediction_p10_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_p90_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actual_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    diff_rounded_instances: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_pid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_pid_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_kalman: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_kalman_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
