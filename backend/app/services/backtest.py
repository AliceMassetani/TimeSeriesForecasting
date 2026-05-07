import os
import pandas as pd
import numpy as np
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
from .pid_controller import NonLinearPID
from ..core.config import settings

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

    def run_backtest_pipeline(self, df: pd.DataFrame, target: str, repository: IBacktestRepository,
                               pid_kp=None, pid_ki=None, pid_kd=None,
                               pid_exp=None, pid_scale_down=None, pid_quantile=None) -> dict:
        repository.delete_all()
        data = self._execute_backtest(df, target, pid_kp=pid_kp, pid_ki=pid_ki, pid_kd=pid_kd,
                                      pid_exp=pid_exp, pid_scale_down=pid_scale_down, pid_quantile=pid_quantile)
        # 2. Salvataggio Punti
        count = repository.create_bulk(data["results"])
        # 3. Salvataggio Metriche "Ufficiali"
        repository.save_metrics(data["metrics"])
        return {"count": count, "metrics": data["metrics"]}

    def _execute_backtest(self, df: pd.DataFrame, target: str,
                          pid_kp=None, pid_ki=None, pid_kd=None,
                          pid_exp=None, pid_scale_down=None, pid_quantile=None) -> dict:
        if self.model is None: raise RuntimeError("Modello non caricato.")

        # Parametri PID con fallback ai valori di config
        kp = pid_kp if pid_kp is not None else settings.PID_KP
        ki = pid_ki if pid_ki is not None else settings.PID_KI
        kd = pid_kd if pid_kd is not None else settings.PID_KD
        exp = pid_exp if pid_exp is not None else settings.PID_DERIVATIVE_EXP
        scale_down = pid_scale_down if pid_scale_down is not None else settings.PID_SCALE_DOWN_PENALTY
        # Quantile: default 0.9 (P90) per over-provisioning sistematico
        quantile = pid_quantile if pid_quantile is not None else 0.9
        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        series = fill_missing_values(TimeSeries.from_dataframe(df_clean, time_col="TimeStamp", value_cols=target, freq="h"))
        forecast = self.model.historical_forecasts(
            series, 
            start=self.model.input_chunk_length, 
            forecast_horizon=2, 
            stride=1, 
            num_samples=100, # Ridotto da 200 a 100 per velocizzare (sufficiente per stabilità quantili)
            retrain=False, 
            last_points_only=True,
            verbose=False
        )
        
        forecast_p50 = forecast.quantile(0.5)
        forecast_p90 = forecast.quantile(0.9)
        forecast_p10 = forecast.quantile(0.1)

        # Inizializzazione PID
        pid = NonLinearPID(
            kp=kp,
            ki=ki,
            kd=kd,
            derivative_exp=exp,
            scale_down_penalty=scale_down
        )
        
        pid_predictions = []
        df_p50 = forecast_p50.to_dataframe()
        df_p90 = forecast_p90.to_dataframe()
        df_p10 = forecast_p10.to_dataframe()

        if df_p50.empty:
            return {"results": [], "metrics": BacktestMetrics(rmse=0, mse=0, mae=0, r2=0)}

        # Inizializziamo con il primo valore previsto per evitare il "salto" iniziale che sballa la derivata
        current_pid_val = float(df_p50.iloc[0].iloc[0])
        
        # Definiamo dei limiti ragionevoli basati sul dataset (es. 0 e il triplo del valore massimo previsto)
        max_limit = float(df_p50.max().iloc[0] * 3)
        
        # Loop PID: usa il quantile scelto dall'utente come setpoint
        forecast_setpoint = forecast.quantile(quantile)
        df_setpoint = forecast_setpoint.to_dataframe()
        for (ts, row_p50), (_, row_sp) in zip(df_p50.iterrows(), df_setpoint.iterrows()):
            pred_sp = float(row_sp.iloc[0])
            current_pid_val = pid.update(pred_sp, current_pid_val, min_val=0.0, max_val=max_limit)
            pid_predictions.append(current_pid_val)

        # Creazione di una serie temporale per il PID per calcolare le metriche
        pid_series = TimeSeries.from_times_and_values(forecast_p50.time_index, pid_predictions)

        metrics = BacktestMetrics(
            rmse=rmse(series, forecast_p50), 
            mse=mse(series, forecast_p50), 
            mae=mae(series, forecast_p50), 
            r2=r2_score(series, forecast_p50),
            rmse_pid=rmse(series, pid_series),
            mse_pid=mse(series, pid_series),
            mae_pid=mae(series, pid_series),
            r2_pid=r2_score(series, pid_series)
        )

        results = []
        for i, (ts, row_p50) in enumerate(df_p50.iterrows()):
            real_val = df_clean.loc[df_clean['TimeStamp'] == ts, target]
            if real_val.empty: continue
            real = real_val.values[0]
            pred_p50 = row_p50.iloc[0]
            pred_p10 = df_p10.loc[ts].iloc[0]
            pred_p90 = df_p90.loc[ts].iloc[0]
            pred_pid = pid_predictions[i]

            # Protezione finale contro valori non finiti (NaN/Inf) prima del cast a int
            pred_pid_rounded = None
            if pred_pid is not None and np.isfinite(pred_pid):
                pred_pid_rounded = int(round(pred_pid))

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
                diff_rounded_instances=round(pred_p50) - round(real),
                prediction_pid=pred_pid,
                prediction_pid_rounded=pred_pid_rounded
            ))
        return {"results": results, "metrics": metrics}

    def get_backtest_chart_data(self, repository: IBacktestRepository, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        return repository.get_chart_data(limit, start_date, end_date)

backtest_service = BacktestService()
