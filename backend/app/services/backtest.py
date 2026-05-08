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
                               pid_exp=None, pid_scale_down=None, pid_quantile=None,
                               pid_max_derivative=None, pid_acceleration_factor=None) -> dict:
        repository.delete_all()
        data = self._execute_backtest(df, target, pid_kp=pid_kp, pid_ki=pid_ki, pid_kd=pid_kd,
                                      pid_exp=pid_exp, pid_scale_down=pid_scale_down, pid_quantile=pid_quantile,
                                      pid_max_derivative=pid_max_derivative, pid_acceleration_factor=pid_acceleration_factor)
        # 2. Salvataggio Punti
        count = repository.create_bulk(data["results"])
        # 3. Salvataggio Metriche "Ufficiali"
        repository.save_metrics(data["metrics"])
        return {"count": count, "metrics": data["metrics"]}

    def _execute_backtest(self, df: pd.DataFrame, target: str,
                          pid_kp=None, pid_ki=None, pid_kd=None,
                          pid_exp=None, pid_scale_down=None, pid_quantile=None,
                          pid_max_derivative=None, pid_acceleration_factor=None) -> dict:
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

        # Analisi dinamica del dataset per impostare limiti coerenti
        target_series = df[target]
        max_observed = float(target_series.max())
        volatility = float(target_series.diff().abs().quantile(settings.PID_VOLATILITY_QUANTILE)) # Variazione massima "normale"
        
        # Heuristic: max_derivative dovrebbe permettere reazioni ai picchi ma non rumore infinito
        if pid_max_derivative is None:
            # Usiamo il moltiplicatore da config, o almeno il 10% del massimo per servizi molto piatti
            max_d = max(volatility * settings.PID_VOLATILITY_MULT, max_observed * 0.1)
        else:
            max_d = pid_max_derivative

        # Limite massimo di sicurezza per il clipping: moltiplicatore da config sul massimo storico
        safety_limit = max_observed * settings.PID_SAFETY_MULT
        acc_f = pid_acceleration_factor if pid_acceleration_factor is not None else settings.PID_ACCELERATION_FACTOR

        pid = NonLinearPID(
            kp=kp, 
            ki=ki,
            kd=kd,
            derivative_exp=exp,
            scale_down_penalty=scale_down,
            max_derivative=max_d,
            acceleration_factor=acc_f
        )
        
        pid_predictions = []
        df_p50 = forecast_p50.to_dataframe()
        df_p90 = forecast_p90.to_dataframe()
        df_p10 = forecast_p10.to_dataframe()

        if df_p50.empty:
            return {"results": [], "metrics": BacktestMetrics(rmse=0, mse=0, mae=0, r2=0)}

        # Loop PID: Feedback Reale
        # Il setpoint è il forecast futuro, il feedback (PV) è il valore reale passato 
        forecast_setpoint = forecast.quantile(quantile)
        df_setpoint = forecast_setpoint.to_dataframe()
        
        # Inizializziamo il feedback con il primo valore reale disponibile
        first_ts = df_p50.index[0]
        first_real_val = df_clean.loc[df_clean['TimeStamp'] == first_ts, target]
        
        # Se non c'è il dato reale, usiamo il primo valore del setpoint scelto (es. P90) 
        # per evitare che la differenza P90-P50 venga vista come un errore reale al primo step
        current_feedback_val = float(first_real_val.values[0]) if not first_real_val.empty else float(df_setpoint.iloc[0].iloc[0])
        
        for (ts, row_p50), (_, row_sp) in zip(df_p50.iterrows(), df_setpoint.iterrows()):
            pred_sp = float(row_sp.iloc[0])
            
            # 1. Update PID: 
            # Setpoint = Cosa prevediamo di servire
            # Current_value = Cosa abbiamo servito realmente un attimo prima (Feedback)
            current_pid_val = pid.update(pred_sp, current_feedback_val, min_val=0.0, max_val=safety_limit)
            pid_predictions.append(current_pid_val)
            
            # 2. Prepariamo il feedback per il prossimo passo (il valore reale a questo timestamp)
            real_val_now = df_clean.loc[df_clean['TimeStamp'] == ts, target]
            if not real_val_now.empty:    
                current_feedback_val = float(real_val_now.values[0])
            else:
                # Fallback se manca il dato reale (usiamo la nostra previsione precedente)
                current_feedback_val = current_pid_val

        # Creazione di una serie temporale per il PID per calcolare le metriche
        pid_series = TimeSeries.from_times_and_values(forecast_p50.time_index, pid_predictions)

        # Inizializzazione contatori per le nuove metriche
        under_count, over_count = 0, 0
        under_sum, over_sum = 0.0, 0.0
        under_count_pid, over_count_pid = 0, 0
        under_sum_pid, over_sum_pid = 0.0, 0.0

        results = []
        for i, (ts, row_p50) in enumerate(df_p50.iterrows()):
            real_val = df_clean.loc[df_clean['TimeStamp'] == ts, target]
            if real_val.empty: continue
            real = float(real_val.values[0])
            
            # Recuperiamo il valore del quantile selezionato (setpoint base)
            pred_base = float(df_setpoint.loc[ts].iloc[0])
            
            pred_p10 = float(df_p10.loc[ts].iloc[0])
            pred_p90 = float(df_p90.loc[ts].iloc[0])
            pred_pid = float(pid_predictions[i])

            # Calcolo metriche per modello originale (basato sul quantile scelto)
            real_rounded = int(round(real))
            pred_base_rounded = int(round(pred_base))
            
            # CRIT-8: Calcolo scostamenti sui valori FLOAT per non mascherare errori piccoli
            diff_raw = pred_base - real
            if diff_raw < -0.01: # Soglia minima per considerare under-provisioning
                under_count += 1
                under_sum += abs(diff_raw)
            elif diff_raw > 0.01:
                over_count += 1
                over_sum += diff_raw

            # Calcolo metriche per PID
            pred_pid_val = pred_pid if pred_pid is not None and np.isfinite(pred_pid) else 0.0
            diff_pid = pred_pid_val - real
            
            if diff_pid < -0.01:
                under_count_pid += 1
                under_sum_pid += abs(diff_pid)
            elif diff_pid > 0.01:
                over_count_pid += 1
                over_sum_pid += diff_pid

            # Protezione finale contro valori non finiti (NaN/Inf) prima del cast a int
            pred_pid_rounded = None
            if pred_pid is not None and np.isfinite(pred_pid):
                pred_pid_rounded = int(round(pred_pid))

            results.append(BacktestResult(
                timestamp=ts, 
                prediction=pred_base, 
                prediction_p10=pred_p10,
                prediction_p90=pred_p90, 
                actual_value=real,
                diff_instances=pred_base - real, 
                prediction_rounded=round(pred_base),
                prediction_p10_rounded=round(pred_p10),
                prediction_p90_rounded=round(pred_p90), 
                actual_rounded=round(real),
                diff_rounded_instances=round(pred_base) - round(real),
                prediction_pid=pred_pid,
                prediction_pid_rounded=pred_pid_rounded
            ))
        metrics = BacktestMetrics(
            rmse=rmse(series, forecast_setpoint), 
            mse=mse(series, forecast_setpoint), 
            mae=mae(series, forecast_setpoint), 
            r2=r2_score(series, forecast_setpoint),
            rmse_pid=rmse(series, pid_series),
            mse_pid=mse(series, pid_series),
            mae_pid=mae(series, pid_series),
            r2_pid=r2_score(series, pid_series),
            under_count=under_count,
            over_count=over_count,
            under_sum=round(under_sum),
            over_sum=round(over_sum),
            under_count_pid=under_count_pid,
            over_count_pid=over_count_pid,
            under_sum_pid=round(under_sum_pid),
            over_sum_pid=round(over_sum_pid)
        )

        return {"results": results, "metrics": metrics}

    def get_backtest_chart_data(self, repository: IBacktestRepository, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        return repository.get_chart_data(limit, start_date, end_date)

backtest_service = BacktestService()
