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
