import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_long_term_prophet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading and Resampling Data ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Resample every 3 hours taking the maximum peak and fill missing values
    df_resampled = df.set_index('TimeStamp').resample('3h').max()
    df_resampled = df_resampled.ffill().reset_index()
    
    # Using InUseCapacity (actual users) instead of CapacityUtilization
    df_resampled = df_resampled.rename(columns={'TimeStamp': 'ds', 'InUseCapacity': 'y'})

    print("--- 2. Chronological Split (Last 90 Days) ---")
    prediction_length = 720 
    train_df = df_resampled.iloc[:-prediction_length].copy()
    test_df = df_resampled.iloc[-prediction_length:].copy()

    # LOGARITHMIC TRANSFORMATION: strictly prevents negative user counts and smoothes variance
    train_df['y'] = np.log1p(train_df['y'])

    print("--- 3. Running Prophet Model ---")
    prophet_model = Prophet(
        interval_width=0.80, 
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=True
    )
    prophet_model.add_country_holidays(country_name='IT')
    prophet_model.fit(train_df)
    
    print("--- 4. Running Inference ---")
    future = test_df[['ds']].copy()
    prophet_forecast = prophet_model.predict(future)
    
    # LOGARITHMIC INVERSION: converting data back to real scale (actual number of users)
    prophet_p50 = np.expm1(prophet_forecast['yhat'].values)
    prophet_p90 = np.expm1(prophet_forecast['yhat_upper'].values)
    y_true = test_df['y'].values

    print("--- 5. Evaluating Performance ---")
    prophet_mae = mean_absolute_error(y_true, prophet_p50)
    prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_p50))
    
    print(f"\n✅ PROPHET RESULTS (Users Prediction):")
    print(f"MAE: {prophet_mae:.2f} Users")
    print(f"RMSE: {prophet_rmse:.2f} Users")

    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 6. Saving Visual Evaluation (Full 90 Days) ---")
    plt.figure(figsize=(16, 8))
    
    # Converting historical data back to normal scale for the plot
    history_y = np.expm1(train_df['y'].tail(30 * 8))
    plt.plot(train_df['ds'].tail(30 * 8), history_y, label='History (Last 30 days)', color='black', linewidth=1)
    
    plt.plot(test_df['ds'], y_true, label='Actual Users', color='green', linewidth=2)
    plt.plot(test_df['ds'], prophet_p50, label=f'Prophet P50 (MAE: {prophet_mae:.2f})', color='orange', linewidth=2)
    plt.plot(test_df['ds'], prophet_p90, label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(test_df['ds'], prophet_p50, prophet_p90, color='red', alpha=0.1)

    plt.title('Capacity Planning: Prophet 90-Day Forecast (Predicted Concurrent Users)')
    plt.xlabel('Time')
    plt.ylabel('Max Concurrent Users (InUseCapacity)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'long_term_prophet_90days.png'))

    print("--- 7. Saving Visual Evaluation (30-Day ZOOM) ---")
    plt.figure(figsize=(16, 8))
    zoom_steps = 30 * 8
    test_zoom = test_df.tail(zoom_steps)
    p50_zoom = prophet_p50[-zoom_steps:]
    p90_zoom = prophet_p90[-zoom_steps:]
    
    plt.plot(test_zoom['ds'], test_zoom['y'], label='Actual Users', color='green', linewidth=2, marker='o', markersize=3)
    plt.plot(test_zoom['ds'], p50_zoom, label='Prophet P50', color='orange', linewidth=2.5)
    plt.plot(test_zoom['ds'], p90_zoom, label='Safety Threshold P90', color='red', linestyle=':', linewidth=1.5)
    plt.fill_between(test_zoom['ds'], p50_zoom, p90_zoom, color='red', alpha=0.1)

    plt.title('Capacity Planning: Prophet Forecast - ZOOM on Last 30 Days')
    plt.xlabel('Time')
    plt.ylabel('Max Concurrent Users (InUseCapacity)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'long_term_prophet_90days_ZOOM.png'))

if __name__ == "__main__":
    run_long_term_prophet()