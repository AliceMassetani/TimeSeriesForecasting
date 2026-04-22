from sqlalchemy import Column, Integer, Float, DateTime
from ..db.base_class import Base

class PredictionResult(Base):
    """
    Rappresenta la tabella 'predictions' nel database MariaDB.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    prediction = Column(Float)
    actual_value = Column(Float)
    diff_percentage = Column(Float)
    prediction_rounded = Column(Integer)
    actual_rounded = Column(Integer)
    diff_rounded_percentage = Column(Float)
