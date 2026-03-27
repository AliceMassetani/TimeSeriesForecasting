import pandas as pd
import torch
import holidays
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import Chronos2Pipeline

def run_long_term_chronos():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading and Resampling Data (3H Max) ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Resample and fill gaps
    df_resampled = df.set_index('TimeStamp').resample('3h').max()
    df_resampled = df_resampled.ffill().reset_index()

    # Covariates
    df_resampled['item_id'] = 'appstream_fleet'
    df_resampled['hour'] = df_resampled['TimeStamp'].dt.hour
    df_resampled['day_of_week'] = df_resampled['TimeStamp'].dt.dayofweek
    it_holidays = holidays.Italy()
    df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: int(x in it_holidays))

    print("--- 2. Chronological Split (Last 90 Days) ---")
    prediction_length = 720 
    
    train_df = df_resampled.iloc[:-prediction_length].copy()
    test_df = df_resampled.iloc[-prediction_length:].copy()
    future_df = test_df[['TimeStamp', 'item_id', 'hour', 'day_of_week', 'is_holiday']].copy()

    print("--- 3. Initializing Chronos-2 Model ---")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("--- 4. Running Inference (Target: InUseCapacity) ---")
    forecast_df = pipeline.predict_df(
        df=train_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="TimeStamp",
        target="InUseCapacity", 
        prediction_length=prediction_length,
        quantile_levels=[0.5, 0.9] 
    )

    print("--- 5. Evaluating Performance (3 Months) ---")
    y_pred = forecast_df['0.5'].values
    y_true = test_df['InUseCapacity'].values 

    chronos_mae = mean_absolute_error(y_true, y_pred)
    chronos_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n✅ CHRONOS-2 RESULTS (Users Prediction):")
    print(f"MAE: {chronos_mae:.2f} Users")
    print(f"RMSE: {chronos_rmse:.2f} Users")

    # Outputs
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 6. Saving Visual Evaluation (Full 90 Days) ---")
    plt.figure(figsize=(16, 8))
    context_days = 30 * 8
    plt.plot(train_df['TimeStamp'].tail(context_days), train_df['InUseCapacity'].tail(context_days), label='History (Last 30 days)', color='black', linewidth=1)
    plt.plot(test_df['TimeStamp'], test_df['InUseCapacity'], label='Actual Users', color='green', linewidth=2)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.5'], label=f'Chronos P50 (MAE: {chronos_mae:.2f})', color='blue', linewidth=2)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.9'], label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(forecast_df['TimeStamp'], forecast_df['0.5'], forecast_df['0.9'], color='red', alpha=0.1)

    plt.title('Capacity Planning: Chronos-2 90-Day Forecast (Predicted Concurrent Users)')
    plt.xlabel('Time')
    plt.ylabel('Max Concurrent Users (InUseCapacity)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outputs_dir, 'long_term_chronos_90days.png'))

    print("--- 7. Saving Visual Evaluation (30-Day ZOOM) ---")
    plt.figure(figsize=(16, 8))
    zoom_steps = 30 * 8
    test_df_zoom = test_df.tail(zoom_steps)
    forecast_df_zoom = forecast_df.tail(zoom_steps)
    
    plt.plot(test_df_zoom['TimeStamp'], test_df_zoom['InUseCapacity'], label='Actual Users', color='green', linewidth=2, marker='o', markersize=3)
    plt.plot(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.5'], label='Chronos P50', color='blue', linewidth=2.5)
    plt.plot(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.9'], label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(forecast_df_zoom['TimeStamp'], forecast_df_zoom['0.5'], forecast_df_zoom['0.9'], color='red', alpha=0.1)

    plt.title('Capacity Planning: Chronos-2 Forecast - ZOOM on Last 30 Days')
    plt.xlabel('Time')
    plt.ylabel('Max Concurrent Users (InUseCapacity)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outputs_dir, 'long_term_chronos_90days_ZOOM.png'))

if __name__ == "__main__":
    run_long_term_chronos()