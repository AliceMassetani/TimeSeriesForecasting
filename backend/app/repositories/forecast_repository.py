from sqlalchemy.orm import Session
from sqlalchemy import desc
import pandas as pd
from typing import List, Optional
from datetime import datetime
from ..entities.forecast_result import ForecastResult
from ..schemas.forecast_chart import ForecastChart
from .base import IForecastRepository
from ..schemas.forecast_chart import ForecastChart

class ForecastRepository(IForecastRepository):
    """
    Repository per la gestione della persistenza dei Forecast.
    """
    def __init__(self, db: Session):
        self.db = db

    def _get_base_query(self):
        query = self.db.query(ForecastResult)
        query = query.order_by(ForecastResult.timestamp)
        return query

    def get_chart_data(self) -> ForecastChart:
        query = self._get_base_query()
        
        # Soluzione ad alte prestazioni che bypassa il bug del container
        with self.db.get_bind().connect() as conn:
            result = conn.execute(query.statement)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        
        if df.empty:
            return ForecastChart(
                labels=[], 
                prediction=[], 
                prediction_p10=[],
                prediction_p90=[],
                prediction_rounded=[], 
                prediction_p10_rounded=[],
                prediction_p90_rounded=[],
                actual=[]
            )

        # Convertiamo tutto in 'object' per permettere a None di coesistere con i numeri
        # Senza astype(object), Pandas forzerebbe i None a tornare NaN per mantenere il tipo float
        df = df.astype(object).where(pd.notnull(df), None)
        
        return ForecastChart(
            labels=df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M')).tolist(),
            prediction=df['prediction'].tolist(),
            prediction_p10=df['prediction_p10'].tolist(),
            prediction_p90=df['prediction_p90'].tolist(),
            prediction_rounded=df['prediction_rounded'].tolist(),
            prediction_p10_rounded=df['prediction_p10_rounded'].tolist(),
            prediction_p90_rounded=df['prediction_p90_rounded'].tolist(),
            prediction_pid=df['prediction_pid'].tolist(),
            prediction_pid_rounded=df['prediction_pid_rounded'].tolist(),
            actual=df['actual_value'].tolist()
        )

    def create_bulk(self, results: List[ForecastResult]) -> int:
        try:
            self.db.add_all(results)
            self.db.commit()
            return len(results)
        except Exception as e:
            self.db.rollback()
            raise e

    def delete_all(self) -> int:
        try:
            num_deleted = self.db.query(ForecastResult).delete()
            self.db.commit()
            return num_deleted
        except Exception as e:
            self.db.rollback()
            raise e
