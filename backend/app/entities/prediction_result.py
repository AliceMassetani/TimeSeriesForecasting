from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base_class import Base

class PredictionResult(Base):
    """
    Rappresenta la tabella 'predictions' nel database MariaDB.
    Usa lo stile SQLAlchemy 2.0 con Mapped e mapped_column.
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[float] = mapped_column(Float)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    diff_instances: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_rounded: Mapped[int] = mapped_column(Integer)
    actual_rounded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    diff_rounded_instances: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
