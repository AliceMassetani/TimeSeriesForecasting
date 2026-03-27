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

    print("--- 1. Loading and Resampling Data ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Resample and fill gaps
    frequency='12h'
    df_resampled = df.set_index('TimeStamp').resample(frequency).max()
    df_resampled = df_resampled.ffill().reset_index()

    # Covariates
    df_resampled['item_id'] = 'appstream_fleet'
    df_resampled['hour'] = df_resampled['TimeStamp'].dt.hour
    df_resampled['day_of_week'] = df_resampled['TimeStamp'].dt.dayofweek
    it_holidays = holidays.Italy()
    df_resampled['is_holiday'] = df_resampled['TimeStamp'].apply(lambda x: int(x in it_holidays))

    print("--- 2. Chronological Split ---")
    prediction_length = 365*2 
    
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

# --- 6. Annual Visualization (12H Resolution) ---
    print("--- 6. Generating Annual Forecast Plot (12H Resolution) ---")
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(20, 10))
    
    # Plot Actual Data: lower alpha and thin line to avoid cluttering 1-year view
    plt.plot(test_df['TimeStamp'], test_df['InUseCapacity'], 
             label='Actual Data', color='green', alpha=0.5, linewidth=1)
    
    # Plot Chronos P50 (Median Forecast)
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.5'], 
             label='Chronos P50 (Median)', color='blue', linewidth=2)
    
    # Plot Safety Threshold P90
    plt.plot(forecast_df['TimeStamp'], forecast_df['0.9'], 
             label='Safety Threshold P90', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Fill Uncertainty Area between P50 and P90
    plt.fill_between(forecast_df['TimeStamp'], forecast_df['0.5'], forecast_df['0.9'], 
                     color='red', alpha=0.1, label='Safety Margin (P50-P90)')

    # --- X-Axis Configuration for 1-Year Scale ---
    ax = plt.gca()
    # Set a tick for every month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # Format label: Month Abbreviation + Year (e.g., Jan 2026)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.title('Chronos-2: Annual Capacity Planning (365-Day Horizon - 12H Resolution)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Concurrent Users (Peak)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the main annual plot
    plt.savefig(os.path.join(outputs_dir, 'annual_chronos_1year_12h.png'), dpi=300)

    # --- 7. Strategic Zoom (Last 90 Days) ---
    print("--- 7. Generating 90-Day Strategic Zoom ---")
    plt.figure(figsize=(16, 8))
    
    # Calculate steps for 90 days at 12h resolution (90 * 2 = 180 points)
    zoom_steps = 90 * 2 
    test_zoom = test_df.tail(zoom_steps)
    forecast_zoom = forecast_df.tail(zoom_steps)
    
    plt.plot(test_zoom['TimeStamp'], test_zoom['InUseCapacity'], 
             label='Actual Data', color='green', linewidth=2, marker='o', markersize=3)
    plt.plot(forecast_zoom['TimeStamp'], forecast_zoom['0.5'], 
             label='Chronos P50', color='blue', linewidth=2.5)
    plt.plot(forecast_zoom['TimeStamp'], forecast_zoom['0.9'], 
             label='P90 Threshold', color='red', linestyle=':', linewidth=2)
    
    plt.fill_between(forecast_zoom['TimeStamp'], forecast_zoom['0.5'], forecast_zoom['0.9'], 
                     color='red', alpha=0.15)
    
    plt.title('90-Day Zoom: Detailed Model Performance Analysis (Q4)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Users')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the zoomed plot
    plt.savefig(os.path.join(outputs_dir, 'annual_chronos_1year_zoom.png'), dpi=300)

if __name__ == "__main__":
    run_long_term_chronos()