import os
import pandas as pd
from darts import TimeSeries

# --- FIX DEFINITIVO PER PYTORCH 2.6+ SECURITY ---
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# -----------------------------------------------

from darts.models import TSMixerModel
from darts.utils.missing_values import fill_missing_values
from darts.metrics import rmse, mse, mae, r2_score
from ..schemas.backtest_metrics import BacktestMetrics
from ..entities.backtest_result import BacktestResult
from ..schemas.backtest_chart import BacktestChart
from ..repositories.base import IBacktestRepository
from typing import List, Optional
from datetime import datetime

CHAMPION_PATH = os.path.join("app", "ml_models", "tsmixer_champion.pt")
CANDIDATE_PATH = os.path.join("app", "ml_models", "tsmixer_candidate.pt")

class BacktestService:
    def __init__(self):
        self.model = None
        self.active_model_type = "champion"
        self.load_model()

    def load_model(self, model_type: str = "champion"):
        path = CHAMPION_PATH if model_type == "champion" else CANDIDATE_PATH
        if os.path.exists(path):
            try:
                self.model = TSMixerModel.load(path)
                self.active_model_type = model_type
            except Exception as e:
                print(f"ERRORE: {str(e)}")
        else:
            print(f"Modello non trovato")

    def run_backtest_pipeline(self, df: pd.DataFrame, target: str, repository: IBacktestRepository) -> dict:
        repository.delete_all()
        # 1. Calcolo
        data = self._execute_backtest(df, target)
        # 2. Salvataggio Punti
        count = repository.create_bulk(data["results"])
        # 3. Salvataggio Metriche "Ufficiali"
        repository.save_metrics(data["metrics"])
        return {"count": count, "metrics": data["metrics"]}

    def _execute_backtest(self, df: pd.DataFrame, target: str) -> dict:
        if self.model is None: raise RuntimeError("Modello non caricato.")
        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        series = fill_missing_values(TimeSeries.from_dataframe(df_clean, time_col="TimeStamp", value_cols=target, freq="h"))
        forecast = self.model.historical_forecasts(series, start=self.model.input_chunk_length, forecast_horizon=2, stride=1, num_samples=200, retrain=False, last_points_only=True)
        
        forecast_p50 = forecast.quantile(0.5)
        forecast_p90 = forecast.quantile(0.9)
        forecast_p10 = forecast.quantile(0.1)
        metrics = BacktestMetrics(rmse=rmse(series, forecast_p50), mse=mse(series, forecast_p50), mae=mae(series, forecast_p50), r2=r2_score(series, forecast_p50))
        
        results = []
        df_p50 = forecast_p50.to_dataframe()
        df_p90 = forecast_p90.to_dataframe()
        df_p10 = forecast_p10.to_dataframe()

        for ts, row_p50 in df_p50.iterrows():
            real_val = df_clean.loc[df_clean['TimeStamp'] == ts, target]
            if real_val.empty: continue
            real = real_val.values[0]
            pred_p50 = row_p50.iloc[0]
            pred_p10 = df_p10.loc[ts].iloc[0]
            pred_p90 = df_p90.loc[ts].iloc[0]
            results.append(BacktestResult(
                timestamp=ts, 
                prediction=pred_p50, 
                prediction_p10=pred_p10,
                prediction_p90=pred_p90, 
                actual_value=real,
                diff_instances=pred_p50 - real, 
                prediction_rounded=round(pred_p50),
                prediction_p10_rounded=round(pred_p10),
                prediction_p90_rounded=round(pred_p90), 
                actual_rounded=round(real),
                diff_rounded_instances=round(pred_p50) - round(real)
            ))
        return {"results": results, "metrics": metrics}

    def get_backtest_chart_data(self, repository: IBacktestRepository, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        return repository.get_chart_data(limit, start_date, end_date)

backtest_service = BacktestService()
