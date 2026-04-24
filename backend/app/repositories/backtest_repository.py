from sqlalchemy.orm import Session
from sqlalchemy import desc
import pandas as pd
from typing import List, Optional
from datetime import datetime
from ..entities.backtest_result import BacktestResult
from ..schemas.backtest_chart import BacktestChart
from .base import IBacktestRepository

class BacktestRepository(IBacktestRepository):
    """
    Implementazione SQLAlchemy del Repository del Backtest.
    """
    def __init__(self, db: Session):
        self.db = db

    def _get_base_query(self, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        query = self.db.query(BacktestResult)
        
        if start_date:
            query = query.filter(BacktestResult.timestamp >= start_date)
        if end_date:
            query = query.filter(BacktestResult.timestamp <= end_date)
        
        if limit:
            query = query.order_by(desc(BacktestResult.timestamp)).limit(limit)
        else:
            query = query.order_by(BacktestResult.timestamp)
            
        return query

    def get_chart_data(self, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        query = self._get_base_query(limit, start_date, end_date)
        # Soluzione ad alte prestazioni che bypassa il bug del container
        with self.db.get_bind().connect() as conn:
            result = conn.execute(query.statement)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))


        if limit:
            df = df.iloc[::-1].reset_index(drop=True)

        return BacktestChart(
            labels=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            actual=df['actual_value'].tolist(),
            prediction=df['prediction'].tolist(),
            diff_instances=df['diff_instances'].tolist(),
            prediction_rounded=df['prediction_rounded'].tolist(),
            actual_rounded=df['actual_rounded'].tolist(),
            diff_rounded_instances=df['diff_rounded_instances'].tolist()
        )

    def create_bulk(self, results: List[BacktestResult]) -> int:
        try:
            self.db.add_all(results)
            self.db.commit()
            return len(results)
        except Exception as e:
            self.db.rollback()
            raise e

    def delete_all(self) -> int:
        try:
            num_deleted = self.db.query(BacktestResult).delete()
            self.db.commit()
            return num_deleted
        except Exception as e:
            self.db.rollback()
            raise e
