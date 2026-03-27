import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import holidays
from darts import TimeSeries
from darts.models import Chronos2Model, Prophet, ARIMA
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_model_tournament():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # --- 1. Caricamento e Preparazione Dati ---
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df_resampled = df.set_index('TimeStamp').resample('3h').max().ffill().bfill().fillna(0).reset_index()

    # Aggiunta Festività Italiane
    it_holidays = holidays.Italy()
    df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: 1 if x in it_holidays else 0)

    # --- 3. Definizione Target e Tutte le Covariate ---
    target_col = 'InUseCapacity'
    
    # Prendiamo TUTTE le colonne tranne il target e il tempo
    # Questo include: DesiredCapacity, Errors, Utilization, ActualCapacity, is_holiday, hour, day_of_week
    cov_cols = [col for col in df_resampled.columns if col not in [target_col, 'TimeStamp', 'item_id']]
    
    print(f"Target: {target_col}")
    print(f"Covariate Totali ({len(cov_cols)}): {cov_cols}")

    # Creazione Serie Darts
    series = TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=target_col)
    covariates = TimeSeries.from_dataframe(df_resampled, time_col='TimeStamp', value_cols=cov_cols)

    # Split 90 giorni (720 step)
    prediction_length = 720
    train_series, test_series = series[:-prediction_length], series[-prediction_length:]
    y_true = test_series.values().flatten()

    # --- 4. Arena dei Modelli ---
    models = {
        "ARIMA": ARIMA(), 
        "Prophet": Prophet(country_holidays='IT'),
        "Chronos-2": Chronos2Model(
            input_chunk_length=600, 
            output_chunk_length=360 
        ) 
    }

    results = {}

    # --- 3. Ciclo di Training e Prediction ---
    for name, model in models.items():
        print(f"\nRunning: {name}...")
        
        # Passiamo le covariate a chi le supporta
        kwargs = {'future_covariates': covariates} if model.supports_future_covariates else {}

        model.fit(train_series, **kwargs)
        
        # Prediction
        if getattr(model, "is_probabilistic", False):
            forecast = model.predict(n=prediction_length, num_samples=100, **kwargs)
            p50_series = forecast.quantile(0.5)
        else:
            p50_series = model.predict(n=prediction_length, **kwargs)
        
        y_pred = p50_series.values().flatten()
        mae = mean_absolute_error(y_true, y_pred)
        res_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        results[name] = {"p50_series": p50_series, "mae": mae, "rmse": res_rmse}
        print(f"{name} MAE: {mae:.2f} | RMSE: {res_rmse:.2f}")

    # --- 4. Generazione Grafici ---
    # Grafico 1: Full Benchmark (90 giorni)
    plt.figure(figsize=(18, 7))
    test_series.plot(label='Dati Reali', color='green', linewidth=2)
    for name, data in results.items():
        data["p50_series"].plot(label=f'{name} (MAE: {data["mae"]:.2f})')
    plt.title('Benchmark 90 Giorni - Autoscaling Predittivo')
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'darts_benchmark_full.png'))

    # Grafico 2: ZOOM (Ultimi 7 giorni)
    plt.figure(figsize=(18, 7))
    # Zoomiamo sugli ultimi 56 step (7 giorni * 8 campioni/giorno)
    zoom_steps = 56
    test_series[-zoom_steps:].plot(label='Dati Reali', color='green', linewidth=3)
    for name, data in results.items():
        data["p50_series"][-zoom_steps:].plot(label=f'{name}', linewidth=2)
    plt.title('Dettaglio Settimanale (Zoom ultimi 7 giorni)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outputs_dir, 'darts_benchmark_zoom.png'))

    print("\nGrafici salvati in /outputs!")

if __name__ == "__main__":
    run_model_tournament()