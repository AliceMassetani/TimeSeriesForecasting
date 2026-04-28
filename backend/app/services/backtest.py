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
    """
    Service responsabile per le operazioni di Backtesting (simulazione su dati passati).
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

    def run_backtest(self, df: pd.DataFrame, target: str) -> dict:
        """
        Esegue un backtest storico. Restituisce un dizionario con risultati e metriche.
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
            num_samples=200,
            retrain=False,
            last_points_only=True
        )
        
        # --- ESTRAZIONE QUANTILI PER VISUALIZZAZIONE ---
        # Poiché forecast è probabilistico, prendiamo la mediana (P50) e il P70
        forecast_p50 = forecast.quantile(0.5)
        forecast_p70 = forecast.quantile(0.7)

        # --- CALCOLO METRICHE (Sempre sulla mediana) ---
        metrics = self._calculate_performance_metrics(series, forecast_p50)
        
        results = []
        df_p50 = forecast_p50.to_dataframe()
        df_p70 = forecast_p70.to_dataframe()
        
        for ts, row_p50 in df_p50.iterrows():
            real_value_series = df_clean.loc[df_clean['TimeStamp'] == ts, target]
            if real_value_series.empty:
                continue
            
            real = real_value_series.values[0]
            pred_p50 = row_p50.iloc[0]
            pred_p70 = df_p70.loc[ts].iloc[0] # Estraiamo il P70 corrispondente
            
            # Arrotondamenti
            p50_round = round(pred_p50)
            p70_round = round(pred_p70)
            real_round = round(real)
            
            # --- LOGICA DI CALCOLO (DIFFERENZA ISTANZE ASSOLUTE >= 0) ---
            pred_rect = max(0, pred_p50)
            real_rect = max(0, real)
            diff_val = pred_rect - real_rect
            
            p50_round_rect = max(0, p50_round)
            real_round_rect = max(0, real_round)
            diff_round_val = p50_round_rect - real_round_rect
            
            res = BacktestResult(
                timestamp=ts,
                prediction=pred_p50,
                prediction_p70=pred_p70,
                actual_value=real,
                diff_instances=diff_val,
                prediction_rounded=p50_round,
                prediction_p70_rounded=p70_round,
                actual_rounded=real_round,
                diff_rounded_instances=diff_round_val
            )
            results.append(res)
            
        return {
            "results": results,
            "metrics": metrics
        }

    def _calculate_performance_metrics(self, actual_series: TimeSeries, pred_series: TimeSeries) -> BacktestMetrics:
        """
        Calcola RMSE, MSE, MAE e R2 in modo efficiente su TimeSeries in memoria.
        """
        return BacktestMetrics(
            rmse=rmse(actual_series, pred_series),
            mse=mse(actual_series, pred_series),
            mae=mae(actual_series, pred_series),
            r2=r2_score(actual_series, pred_series)
        )

    def run_backtest_pipeline(self, df: pd.DataFrame, target: str, repository: IBacktestRepository) -> dict:
        """
        Pipeline: coordina simulazione e salvataggio.
        """
        repository.delete_all()
        
        # 1. Simulazione + Metriche
        data = self.run_backtest(df, target)
        
        # 2. Salvataggio
        count = repository.create_bulk(data["results"])
        
        return {
            "count": count,
            "metrics": data["metrics"]
        }

    def get_backtest_chart_data(
        self,
        repository: IBacktestRepository,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestChart:
        """
        Recupera i dati del backtest formattati per il grafico.
        """
        return repository.get_chart_data(limit, start_date, end_date)

backtest_service = BacktestService()
