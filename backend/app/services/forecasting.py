import os
import pandas as pd
from darts import TimeSeries

# --- FIX DEFINITIVO PER PYTORCH 2.6+ SECURITY ---
# Forziamo torch.load a permettere il caricamento del modello locale.
# Questo risolve i blocchi su "QuantileRegression", "Adam", "MSELoss", ecc. 
# che Darts usa internamente.
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    # Forza la disattivazione del controllo "weights_only" per permettere il caricamento completo
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# -----------------------------------------------

# Gli import pesanti di Darts devono avvenire DOPO il patch di torch.load
from darts.models import TSMixerModel
from darts.utils.missing_values import fill_missing_values

from ..db.base import PredictionResult
from ..crud.prediction import create_predictions
from sqlalchemy.orm import Session

# Il percorso è relativo alla directory 'backend' da cui viene lanciato uvicorn
MODEL_PATH = os.path.join("app", "ml_models", "tsmixer_champion_target.pt")

class ForecastingService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Carica il modello TSMixer se esiste."""
        if os.path.exists(MODEL_PATH):
            print(f"Caricamento modello da: {MODEL_PATH}")
            try:
                self.model = TSMixerModel.load(MODEL_PATH)
                print("Modello caricato correttamente!")
            except Exception as e:
                print(f"ERRORE critico nel caricamento modello: {str(e)}")
        else:
            print(f"ERRORE: Modello non trovato in {MODEL_PATH}")

    def process_forecasting(self, df: pd.DataFrame, target: str):
        """
        Esegue la previsione e trasforma i dati in oggetti PredictionResult (DB Ready).
        """
        if self.model is None:
            raise RuntimeError("Il modello non è stato caricato correttamente.")

        # 1. Pre-elaborazione
        df_clean = df.sort_values("TimeStamp").reset_index(drop=True)
        series = fill_missing_values(
            TimeSeries.from_dataframe(df_clean, time_col="TimeStamp", value_cols=target, freq="h")
        )

        # 2. Esecuzione previsione
        forecast = self.model.historical_forecasts(
            series,
            start=self.model.input_chunk_length,
            forecast_horizon=2,
            stride=1,
            retrain=False,
            last_points_only=True
        )
        
        # 3. Trasformazione e Calcoli (Business Logic)
        results = []
        forecast_df = forecast.to_dataframe()
        
        for ts, pred_val in forecast_df.iterrows():
            real_value_series = df_clean.loc[df_clean['TimeStamp'] == ts, target]   
            if real_value_series.empty:
                continue
            
            real = real_value_series.values[0]
            pred = pred_val.iloc[0]
              
            # Logica di arrotondamento e calcolo differenze spostata dal Controller al Servizio
            pred_round = round(pred)
            real_round = round(real)
            
            res = PredictionResult(
                timestamp=ts,
                prediction=pred,
                actual_value=real,
                diff_percentage=((pred - real) / real) * 100 if real != 0 else 0,
                prediction_rounded=pred_round,
                actual_rounded=real_round,
                diff_rounded_percentage=((pred_round - real_round) / real_round) * 100 if real_round != 0 else 0
            )
            results.append(res)
            
        return results

    def run_prediction_pipeline(self, df: pd.DataFrame, target: str, db: Session):
        """
        Orchestratore unico: esegue la previsione e salva i risultati nel DB.
        """
        # 1. Calcoli e trasformazioni
        prediction_results = self.process_forecasting(df, target)
        
        # 2. Persistenza tramite CRUD
        count = create_predictions(db, prediction_results)
        
        return count

# Singleton per caricare il modello una sola volta all'avvio
forecasting_service = ForecastingService()
