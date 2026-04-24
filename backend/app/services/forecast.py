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

    def run_forecast(self, df: pd.DataFrame, target: str, n: int = 2) -> dict:
        """
        Salva l'ultima settimana di dati reali e genera n ore di previsioni.
        Restituisce un dizionario con i risultati e i conteggi.
        """
        if self.model is None:
            raise RuntimeError("Il modello non è stato caricato correttamente.")

        # 1. Pulizia e ordinamento
        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        
        # 2. Estraiamo l'ultima settimana di dati reali (max 168 ore)
        last_week_df = df_clean.tail(168)
        results = []
        
        for _, row in last_week_df.iterrows():
            results.append(ForecastResult(
                timestamp=row['TimeStamp'],
                actual_value=int(row[target]),
                prediction=None,
                prediction_rounded=None
            ))
        h_count = len(results)

        # 3. Preparazione serie per Darts e Previsione
        series = fill_missing_values(
            TimeSeries.from_dataframe(df_clean, time_col="TimeStamp", value_cols=target, freq="h")
        )

        # Prevediamo le prossime n ore
        prediction_series = self.model.predict(n=n, series=series)
        forecast_df = prediction_series.to_dataframe()
        
        for ts, pred_val in forecast_df.iterrows():
            pred = pred_val.iloc[0]
            results.append(ForecastResult(
                timestamp=ts,
                prediction=pred,
                prediction_rounded=int(round(pred)),
                actual_value=None
            ))
        p_count = len(results) - h_count
            
        return {
            "results": results,
            "history_count": h_count,
            "prediction_count": p_count
        }

    def run_forecast_pipeline(self, df: pd.DataFrame, target: str, repository, n: int = 2) -> dict:
        """
        Pipeline: Cancella vecchi forecast e salva i nuovi.
        Restituisce i dettagli dell'operazione.
        """
        repository.delete_all()
        data = self.run_forecast(df, target, n)
        total = repository.create_bulk(data["results"])
        
        return {
            "total": total,
            "history": data["history_count"],
            "predictions": data["prediction_count"]
        }

    def get_forecast_chart_data(self, repository) -> ForecastChart:
        return repository.get_chart_data()

forecast_service = ForecastService()
