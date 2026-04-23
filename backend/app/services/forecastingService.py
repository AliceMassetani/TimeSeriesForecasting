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

from ..entities.prediction_result import ForecastingResult
from ..schemas.chart import PredictionChart
from ..repositories.base import IPredictionRepository
from typing import List, Optional
from datetime import datetime

MODEL_PATH = os.path.join("app", "ml_models", "tsmixer_champion_target.pt")

class ForecastingService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print(f"Caricamento modello da: {MODEL_PATH}")
            try:
                self.model = TSMixerModel.load(MODEL_PATH)
                print("Modello caricato correttamente!")
            except Exception as e:
                print(f"ERRORE critico nel caricamento modello: {str(e)}")
        else:
            print(f"ERRORE: Modello non trovato in {MODEL_PATH}")

    def process_forecasting(self, df: pd.DataFrame, target: str) -> List[PredictionResult]:
        """
        Esegue la logica di forecasting pura (ML).
        """
        if self.model is None:
            raise RuntimeError("Il modello non è stato caricato correttamente.")

        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        series = fill_missing_values(
            TimeSeries.from_dataframe(df_clean, time_col="TimeStamp", value_cols=target, freq="h")
        )

        forecast = self.model.historical_forecasts(
            series,
            start=self.model.input_chunk_length,
            forecast_horizon=2,
            stride=1,
            retrain=False,
            last_points_only=True
        )
        
        results = []
        forecast_df = forecast.to_dataframe()
        
        for ts, pred_val in forecast_df.iterrows():
            real_value_series = df_clean.loc[df_clean['TimeStamp'] == ts, target]   
            if real_value_series.empty:
                continue
            
            real = real_value_series.values[0]
            pred = pred_val.iloc[0]
              
            pred_round = round(pred)
            real_round = round(real)
            
            # --- LOGICA DI CALCOLO (DIFFERENZA ISTANZE ASSOLUTE) ---
            pred_rect = max(0, pred)
            real_rect = max(0, real)
            diff_val = pred_rect - real_rect
            
            pred_round_rect = max(0, pred_round)
            real_round_rect = max(0, real_round)
            diff_round_val = pred_round_rect - real_round_rect
            
            res = PredictionResult(
                timestamp=ts,
                prediction=pred,
                actual_value=real,
                diff_instances=diff_val,
                prediction_rounded=pred_round,
                actual_rounded=real_round,
                diff_rounded_instances=diff_round_val
            )
            results.append(res)
            
        return results

    def run_prediction_pipeline(self, df: pd.DataFrame, target: str, repository: IPredictionRepository) -> int:
        """
        Coordina la pipeline: cancella vecchi dati, genera nuove previsioni e le salva.
        """
        repository.delete_all()
        prediction_results = self.process_forecasting(df, target)
        count = repository.create_bulk(prediction_results)
        return count

    def get_chart_data(
        self,
        repository: IPredictionRepository,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PredictionChart:
        """
        Recupera i dati formattati delegando al repository.
        Responsabilità: Business logic / Coordinamento.
        """
        return repository.get_chart_data(limit, start_date, end_date)

forecasting_service = ForecastingService()
