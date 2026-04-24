from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base_class import Base

class ForecastResult(Base):
    """
    Rappresenta i risultati di un Forecast futuro (previsione pura).
    """
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Nel forecast puro, l'actual_value potrebbe non esserci finché il tempo non passa
    actual_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
