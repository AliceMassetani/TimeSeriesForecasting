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

from ..entities.forecast_result import ForecastResult
from ..schemas.forecast_chart import ForecastChart
from .kalman_filter import KalmanFilter
from ..core.config import settings
from typing import List, Optional
import numpy as np
from datetime import datetime

CHAMPION_PATH = os.path.join("app", "ml_models", "tsmixer_champion.pt")
CANDIDATE_PATH = os.path.join("app", "ml_models", "tsmixer_candidate.pt")

class ForecastService:
    """
    Service responsabile per le previsioni FUTURE (Forecasting).
    """
    def __init__(self):
        self.model = None
        self.active_model_type = "champion"
        self.load_model()

    def load_model(self, model_type: str = "champion"):
        """
        Carica il modello specificato (champion o candidate).
        """
        path = CHAMPION_PATH if model_type == "champion" else CANDIDATE_PATH
        
        if os.path.exists(path):
            print(f"Caricamento modello ({model_type}) da: {path}")
            try:
                self.model = TSMixerModel.load(path)
                self.active_model_type = model_type
                print(f"Modello {model_type} caricato correttamente!")
            except Exception as e:
                print(f"ERRORE critico nel caricamento modello {model_type}: {str(e)}")
        else:
            print(f"ERRORE: Modello {model_type} non trovato in {path}")

    def run_forecast(self, df: pd.DataFrame, target: str, n: int = 2, history_hours: int = -1, quantile: float = 0.9) -> dict:
        """
        Salva una porzione di dati reali e genera n ore di previsioni.
        history_hours: quante ore di storico includere (se -1, prende tutto).
        quantile: il quantile di riferimento per la correzione adattiva.
        """
        if self.model is None:
            raise RuntimeError("Il modello non è stato caricato correttamente.")

        # 1. Pulizia e ordinamento
        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        
        # 2. Estraiamo lo storico richiesto (tail prende gli ultimi N elementi)
        if history_hours == -1 or history_hours >= len(df_clean):
            history_df = df_clean
        else:
            history_df = df_clean.tail(history_hours)
            
        results = []
        for _, row in history_df.iterrows():
            results.append(ForecastResult(
                timestamp=row['TimeStamp'],
                actual_value=int(row[target]),
                prediction=None,
                prediction_rounded=None
            ))
        h_count = len(results)

        # 3. Preparazione serie per Darts e Previsione
        series = fill_missing_values(
            TimeSeries.from_dataframe(history_df, time_col="TimeStamp", value_cols=target, freq="h")
        )

        # Prevediamo le prossime n ore
        prediction_series = self.model.predict(n=n, series=series, num_samples=200)
        forecast_p50_df = prediction_series.quantile(0.5).to_dataframe()
        forecast_p10_df = prediction_series.quantile(0.1).to_dataframe()
        forecast_p90_df = prediction_series.quantile(0.9).to_dataframe()
        
        for ts, pred_row in forecast_p50_df.iterrows():
            pred_p50 = pred_row.iloc[0]
            pred_p10 = forecast_p10_df.loc[ts].iloc[0]
            pred_p90 = forecast_p90_df.loc[ts].iloc[0]
            
            results.append(ForecastResult(
                timestamp=ts,
                prediction=pred_p50,
                prediction_p10=pred_p10,
                prediction_p90=pred_p90,
                prediction_rounded=int(round(pred_p50)),
                prediction_p10_rounded=int(round(pred_p10)),
                prediction_p90_rounded=int(round(pred_p90)),
                actual_value=None
            ))

        # --- CORREZIONE (PID/Kalman) ---
        # 1. Analizziamo la storia recente per stimare il bias attuale
        kf = KalmanFilter()
        current_bias = 0.0
        
        # Calcoliamo i limiti di sicurezza basati sulla storia caricata
        max_observed = float(history_df[target].max())
        safety_limit = max_observed * settings.PID_SAFETY_MULT
        
        # Se abbiamo abbastanza storia, facciamo un warm-up del filtro
        if len(history_df) > self.model.input_chunk_length + 2:
            try:
                hist_series = fill_missing_values(TimeSeries.from_dataframe(history_df, time_col="TimeStamp", value_cols=target, freq="h"))
                hist_forecasts = self.model.historical_forecasts(
                    hist_series,
                    start=self.model.input_chunk_length,
                    forecast_horizon=2,
                    stride=1,
                    num_samples=50, # Pochi campioni bastano per il bias
                    retrain=False,
                    last_points_only=True,
                    verbose=False
                ).quantile(quantile) # Utilizziamo il quantile scelto dall'utente
                
                # Update Kalman Filter loop
                last_real_val = float(history_df.iloc[0][target])
                prev_forecast = float(hist_forecasts.values()[0][0])
                
                for ts, val in hist_forecasts.items():
                    ts_dt = pd.Timestamp(ts)
                    real_row = history_df[history_df['TimeStamp'] == ts_dt]
                    if not real_row.empty:
                        real_val = float(real_row.iloc[0][target])
                        error = real_val - prev_forecast
                        current_bias = kf.update(error)
                        prev_forecast = float(val)
            except Exception as e:
                print(f"Avviso: impossibile eseguire warm-up Kalman nel forecast: {e}")

        # 2. Applichiamo la correzione ai risultati (storia e futuro)
        # Per la storia, usiamo il bias che si è evoluto
        # Per il futuro, manteniamo l'ultimo bias stimato (o lo facciamo evolvere?)
        # Nel backtest lo facevamo evolvere perché avevamo i dati. Nel forecast futuro lo teniamo fisso.
        
        for res in results:
            # Se è storia e abbiamo il valore reale, potremmo ricalcolare il bias punto per punto
            # ma per semplicità e coerenza col "futuro", usiamo l'ultimo bias per i punti di previsione.
            if res.prediction is not None:
                # Applichiamo il bias al P90 (o P50? Backtest usa P90 come setpoint)
                # In Backtest: current_pid_val = pred_sp + estimated_bias
                # dove pred_sp è il quantile scelto (default P90)
                
                # Usiamo il quantile scelto dall'utente come base per la correzione
                if quantile == 0.5:
                    base_val = res.prediction
                elif quantile == 0.1:
                    base_val = res.prediction_p10
                elif quantile == 0.9:
                    base_val = res.prediction_p90
                else:
                    # Fallback dinamico se il quantile non è uno dei pre-calcolati (es. 0.75, 0.95)
                    # Nota: per ora gestiamo i principali, altrimenti usiamo il P90 come riferimento prudente
                    base_val = res.prediction_p90 if res.prediction_p90 is not None else res.prediction
                
                corrected_val = (base_val if base_val is not None else res.prediction) + current_bias
                res.prediction_pid = float(np.clip(corrected_val, 0.0, safety_limit))
                res.prediction_pid_rounded = int(round(res.prediction_pid))

        # --- BRIDGE THE GAP ---
        # Colleghiamo l'ultimo punto storico alla prima previsione per continuità visiva
        if h_count > 0 and len(results) > h_count:
            last_history = results[h_count - 1]
            if last_history.actual_value is not None:
                val = float(last_history.actual_value)
                last_history.prediction = val
                last_history.prediction_p10 = val
                last_history.prediction_p90 = val
                last_history.prediction_pid = val
                last_history.prediction_rounded = int(val)
                last_history.prediction_p10_rounded = int(val)
                last_history.prediction_p90_rounded = int(val)
                last_history.prediction_pid_rounded = int(val)

        p_count = len([r for r in results if r.prediction is not None])
            
        return {
            "results": results,
            "history_count": h_count,
            "prediction_count": p_count
        }

    def run_forecast_pipeline(self, df: pd.DataFrame, target: str, repository, n: int = 2, history_hours: int = -1, quantile: float = 0.9) -> dict:
        """
        Pipeline: Cancella vecchi forecast e salva i nuovi.
        """
        repository.delete_all()
        data = self.run_forecast(df, target, n, history_hours, quantile)
        total = repository.create_bulk(data["results"])
        
        return {
            "total": total,
            "history": data["history_count"],
            "predictions": data["prediction_count"]
        }

    def get_forecast_chart_data(self, repository) -> ForecastChart:
        return repository.get_chart_data()

forecast_service = ForecastService()
