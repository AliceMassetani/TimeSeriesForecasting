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
    
    # Resample every 3 hours taking the maximum peak for Capacity Planning
    df_resampled = df.set_index('TimeStamp').resample('3h').max().reset_index()
    df_resampled = df_resampled.dropna()

    # Recreate the necessary covariates on the resampled dataset
    df_resampled['item_id'] = 'appstream_fleet'
    df_resampled['hour'] = df_resampled['TimeStamp'].dt.hour
    df_resampled['day_of_week'] = df_resampled['TimeStamp'].dt.dayofweek
    it_holidays = holidays.Italy()
    df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: int(x in it_holidays))

    print("--- 2. Chronological Split (Last 90 Days) ---")
    # 90 days * 8 records per day = 720 steps
    prediction_length = 720 
    
    train_df = df_resampled.iloc[:-prediction_length].copy()
    test_df = df_resampled.iloc[-prediction_length:].copy()
    
    # Create the future dataframe with covariates for inference
    future_df = test_df[['TimeStamp', 'item_id', 'hour', 'day_of_week', 'is_holiday']].copy()

    print("--- 3. Initializing Chronos-2 Model ---")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="auto",
        dtype=torch.bfloat16,
    )

    print("--- 4. Running Inference (Long Term) ---")
    forecast_df = pipeline.predict_df(
        df=train_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="TimeStamp",
        target="CapacityUtilization",
        prediction_length=prediction_length,
        quantile_levels=[0.5, 0.9] 
    )

    print("--- 5. Evaluating Performance (3 Months) ---")
    y_pred = forecast_df['0.5'].values
    y_true = test_df['CapacityUtilization'].values

    chronos_mae = mean_absolute_error(y_true, y_pred)
    chronos_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n✅ CHRONOS-2 RESULTS (90 Days / 3H Resampling):")
    print(f"MAE: {chronos_mae:.2f}")
    print(f"RMSE: {chronos_rmse:.2f}")

    print("--- 6. Saving Visual Evaluation ---")
    plt.figure(figsize=(16, 8))
    
    # Visual context: last 30 days of history
    context_days = 30 * 8
    plt.plot(train_df['TimeStamp'].tail(context_days), train_df['CapacityUtilization'].tail(context_days), label='History (Last 30 days)', color='black', linewidth=1)
    
    # Actual Data (Green)
    plt.plot(test_df['TimeStamp'], test_df['CapacityUtilization'], label='Actual Data (Reality)', color='green', linewidth=2)
    
    # Chronos P50 (Blue) and P90 (Red)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.5'], label=f'Chronos P50 (MAE: {chronos_mae:.2f})', color='blue', linewidth=2)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.9'], label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(forecast_df['TimeStamp'], forecast_df['0.5'], forecast_df['0.9'], color='red', alpha=0.1)

    plt.title('Capacity Planning: Chronos-2 90-Day Forecast (3H Peaks)')
    plt.xlabel('Time')
    plt.ylabel('Max Capacity Utilization (3H)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    plot_path = os.path.join(outputs_dir, 'long_term_chronos_90days.png')
    plt.savefig(plot_path)
    print(f"✅ Chronos-2 chart saved successfully at: {plot_path}")

if __name__ == "__main__":
    run_long_term_chronos()