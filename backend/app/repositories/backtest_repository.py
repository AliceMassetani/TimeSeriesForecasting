from sqlalchemy.orm import Session
from sqlalchemy import desc
import pandas as pd
from typing import List, Optional
from datetime import datetime
from ..entities.backtest_result import BacktestResult
from ..entities.metrics import Metrics
from ..schemas.backtest_chart import BacktestChart
from ..schemas.backtest_metrics import BacktestMetrics
from .base import IBacktestRepository

class BacktestRepository(IBacktestRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_chart_data(self, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        query = self.db.query(BacktestResult)
        if start_date: query = query.filter(BacktestResult.timestamp >= start_date)
        if end_date: query = query.filter(BacktestResult.timestamp <= end_date)
        if limit:
            query = query.order_by(desc(BacktestResult.timestamp)).limit(limit)
        else:
            query = query.order_by(BacktestResult.timestamp)

        with self.db.get_bind().connect() as conn:
            result = conn.execute(query.statement)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

        if df.empty:
            return BacktestChart(
                labels=[],
                actual=[],
                prediction=[],
                prediction_p70=[],
                diff_instances=[],
                prediction_rounded=[],
                prediction_p70_rounded=[],
                actual_rounded=[],
                diff_rounded_instances=[],
                metrics=None
            )

        if limit: df = df.iloc[::-1].reset_index(drop=True)

        # RECUPERO LE METRICHE DALLA TABELLA 'metrics'
        m = self.db.query(Metrics).first()
        metrics = BacktestMetrics(mse=m.mse, rmse=m.rmse, mae=m.mae, r2=m.r2) if m else None

        return BacktestChart(
            labels=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            actual=df['actual_value'].tolist(),
            prediction=df['prediction'].tolist(),
            prediction_p70=df['prediction_p70'].tolist(),
            diff_instances=df['diff_instances'].tolist(),
            prediction_rounded=df['prediction_rounded'].tolist(),
            prediction_p70_rounded=df['prediction_p70_rounded'].tolist(),
            actual_rounded=df['actual_rounded'].tolist(),
            diff_rounded_instances=df['diff_rounded_instances'].tolist(),
            metrics=metrics
        )

    def save_metrics(self, metrics: BacktestMetrics):
        self.db.query(Metrics).delete()
        new_metrics = Metrics(id=1, mse=metrics.mse, rmse=metrics.rmse, mae=metrics.mae, r2=metrics.r2)
        self.db.add(new_metrics)
        self.db.commit()

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
            self.db.query(Metrics).delete()
            num_deleted = self.db.query(BacktestResult).delete()
            self.db.commit()
            return num_deleted
        except Exception as e:
            self.db.rollback()
            raise e
