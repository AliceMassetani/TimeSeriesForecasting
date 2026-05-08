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
                prediction_p10=[],
                prediction_p90=[],
                diff_instances=[],
                prediction_rounded=[],
                prediction_p10_rounded=[],
                prediction_p90_rounded=[],
                actual_rounded=[],
                diff_rounded_instances=[],
                metrics=None
            )

        if limit: df = df.iloc[::-1].reset_index(drop=True)

        # RECUPERO LE METRICHE DALLA TABELLA 'metrics'
        m = self.db.query(Metrics).first()
        metrics = BacktestMetrics(
            mse=m.mse, rmse=m.rmse, mae=m.mae, r2=m.r2,
            mse_pid=getattr(m, 'mse_pid', None), 
            rmse_pid=getattr(m, 'rmse_pid', None), 
            mae_pid=getattr(m, 'mae_pid', None), 
            r2_pid=getattr(m, 'r2_pid', None),
            # Nuove metriche recuperate dal DB
            under_count=getattr(m, 'under_count', 0),
            over_count=getattr(m, 'over_count', 0),
            under_sum=getattr(m, 'under_sum', 0.0),
            over_sum=getattr(m, 'over_sum', 0.0),
            under_count_pid=getattr(m, 'under_count_pid', 0),
            over_count_pid=getattr(m, 'over_count_pid', 0),
            under_sum_pid=getattr(m, 'under_sum_pid', 0.0),
            over_sum_pid=getattr(m, 'over_sum_pid', 0.0)
        ) if m else None

        return BacktestChart(
            labels=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            actual=df['actual_value'].tolist(),
            prediction=df['prediction'].tolist(),
            prediction_p10=df['prediction_p10'].tolist(),
            prediction_p90=df['prediction_p90'].tolist(),
            diff_instances=df['diff_instances'].tolist(),
            prediction_rounded=df['prediction_rounded'].tolist(),
            prediction_p10_rounded=df['prediction_p10_rounded'].tolist(),
            prediction_p90_rounded=df['prediction_p90_rounded'].tolist(),
            actual_rounded=df['actual_rounded'].tolist(),
            diff_rounded_instances=df['diff_rounded_instances'].tolist(),
            prediction_pid=df['prediction_pid'].tolist() if 'prediction_pid' in df.columns else [],
            prediction_pid_rounded=df['prediction_pid_rounded'].tolist() if 'prediction_pid_rounded' in df.columns else [],
            metrics=metrics
        )

    def save_metrics(self, metrics: BacktestMetrics):
        self.db.query(Metrics).delete()
        new_metrics = Metrics(
            id=1, 
            mse=metrics.mse, rmse=metrics.rmse, mae=metrics.mae, r2=metrics.r2,
            mse_pid=metrics.mse_pid, rmse_pid=metrics.rmse_pid, mae_pid=metrics.mae_pid, r2_pid=metrics.r2_pid,
            # Salvataggio nuove metriche
            under_count=metrics.under_count,
            over_count=metrics.over_count,
            under_sum=metrics.under_sum,
            over_sum=metrics.over_sum,
            under_count_pid=metrics.under_count_pid,
            over_count_pid=metrics.over_count_pid,
            under_sum_pid=metrics.under_sum_pid,
            over_sum_pid=metrics.over_sum_pid
        )
        self.db.add(new_metrics)
        self.db.commit()

    def create_bulk(self, results: List[BacktestResult]) -> int:
        try:
            self.db.bulk_save_objects(results)
            self.db.commit()
            return len(results)
        except Exception as e:
            self.db.rollback()
            raise e

    def get_all(self) -> List[BacktestResult]:
        return self.db.query(BacktestResult).order_by(BacktestResult.timestamp).all()

    def delete_all(self) -> int:
        try:
            self.db.query(Metrics).delete()
            num_deleted = self.db.query(BacktestResult).delete()
            self.db.commit()
            return num_deleted
        except Exception as e:
            self.db.rollback()
            raise e
