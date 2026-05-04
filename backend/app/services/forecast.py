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
from typing import List, Optional
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

    def run_forecast(self, df: pd.DataFrame, target: str, n: int = 2, history_hours: int = -1) -> dict:
        """
        Salva una porzione di dati reali e genera n ore di previsioni.
        history_hours: quante ore di storico includere (se -1, prende tutto).
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

        # --- BRIDGE THE GAP ---
        # Colleghiamo l'ultimo punto storico alla prima previsione
        if h_count > 0 and len(results) > h_count:
            last_history = results[h_count - 1]
            if last_history.actual_value is not None:
                val = float(last_history.actual_value)
                last_history.prediction = val
                last_history.prediction_p10 = val
                last_history.prediction_p90 = val
                last_history.prediction_rounded = int(val)
                last_history.prediction_p10_rounded = int(val)
                last_history.prediction_p90_rounded = int(val)
        p_count = len(results) - h_count
            
        return {
            "results": results,
            "history_count": h_count,
            "prediction_count": p_count
        }

    def run_forecast_pipeline(self, df: pd.DataFrame, target: str, repository, n: int = 2, history_hours: int = -1) -> dict:
        """
        Pipeline: Cancella vecchi forecast e salva i nuovi.
        """
        repository.delete_all()
        data = self.run_forecast(df, target, n, history_hours)
        total = repository.create_bulk(data["results"])
        
        return {
            "total": total,
            "history": data["history_count"],
            "predictions": data["prediction_count"]
        }

    def get_forecast_chart_data(self, repository) -> ForecastChart:
        return repository.get_chart_data()

forecast_service = ForecastService()
