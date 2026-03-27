import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import holidays
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_long_term_prophet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    print("--- 1. Loading and Resampling Data ---")
    df = pd.read_csv(data_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # Add holiday and time features
    it_holidays = holidays.Italy()
    df['hour'] = df['TimeStamp'].dt.hour
    df['day_of_week'] = df['TimeStamp'].dt.dayofweek
    df['is_holiday'] = df['TimeStamp'].apply(lambda x: int(x in it_holidays))

    # Resample every 3 hours taking the maximum peak and fill missing values
    frequency='12h'
    df_resampled = df.set_index('TimeStamp').resample(frequency).agg({
        'InUseCapacity': 'max',
        'hour': 'first',
        'day_of_week': 'first',
        'is_holiday': 'first'
    })
    df_resampled = df_resampled.ffill().reset_index()
    
    # Using InUseCapacity (actual users) instead of CapacityUtilization
    df_resampled = df_resampled.rename(columns={'TimeStamp': 'ds', 'InUseCapacity': 'y'})

    print("--- 2. Chronological Split ---")
    prediction_length = 365*2 
    train_df = df_resampled.iloc[:-prediction_length].copy()
    test_df = df_resampled.iloc[-prediction_length:].copy()


    print("--- 3. Running Prophet Model ---")
    prophet_model = Prophet(
        interval_width=0.80, 
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    # Add covariates as regressors
    prophet_model.add_regressor('hour')
    prophet_model.add_regressor('day_of_week')
    prophet_model.add_regressor('is_holiday')
    
    # Add country holidays
    prophet_model.add_country_holidays(country_name='IT')

    prophet_model.fit(train_df)
    
    print("--- 4. Running Inference ---")
    future = test_df[['ds', 'hour', 'day_of_week', 'is_holiday']].copy()
    prophet_forecast = prophet_model.predict(future)
    

    prophet_p50 = np.maximum(0, prophet_forecast['yhat'].values)
    prophet_p90 = np.maximum(0, prophet_forecast['yhat_upper'].values)
    
    y_true = test_df['y'].values

    print("--- 5. Evaluating Performance ---")
    prophet_mae = mean_absolute_error(y_true, prophet_p50)
    prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_p50))
    
    print(f"\n✅ PROPHET RESULTS (Users Prediction):")
    print(f"MAE: {prophet_mae:.2f} Users")
    print(f"RMSE: {prophet_rmse:.2f} Users")


# --- 6. Annual Visualization (12H Resolution) ---
    print("--- 6. Generating Annual Forecast Plot (12H Resolution) ---")
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(20, 10))
    
    # Plot Actual Data: using lower alpha to manage the high density of 1-year data
    plt.plot(test_df['ds'], test_df['y'], 
             label='Actual Data', color='black', alpha=0.4, linewidth=1)
    
    # Plot Prophet P50 (Median Prediction)
    # Note: prophet_p50 should be the exp-transformed yhat
    plt.plot(test_df['ds'], prophet_p50, 
             label='Prophet P50 (Median)', color='orange', linewidth=2)
    
    # Plot Safety Threshold P90
    # Note: prophet_p90 should be the exp-transformed yhat_upper
    plt.plot(test_df['ds'], prophet_p90, 
             label='Safety Threshold P90', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Fill Uncertainty Area (Safety Margin)
    plt.fill_between(test_df['ds'], prophet_p50, prophet_p90, 
                     color='red', alpha=0.1, label='Safety Margin (P50-P90)')

    # --- X-Axis Configuration for 1-Year Scale ---
    ax = plt.gca()
    # Ensure one tick per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # Format: Month Abbreviation + Year (e.g., Mar 2026)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.title('Prophet: Annual Capacity Planning (365-Day Horizon - 12H Resolution)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Concurrent Users (Peak)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='major', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the main annual plot
    plt.savefig(os.path.join(outputs_dir, 'annual_prophet_1year_12h.png'), dpi=300)

    # --- 7. Strategic Zoom (Last 90 Days) ---
    print("--- 7. Generating 90-Day Strategic Zoom ---")
    plt.figure(figsize=(16, 8))
    
    # Calculate steps for 90 days at 12h resolution (90 * 2 = 180 points)
    zoom_steps = 90 * 2
    test_zoom = test_df.tail(zoom_steps)
    p50_zoom = prophet_p50[-zoom_steps:]
    p90_zoom = prophet_p90[-zoom_steps:]
    
    plt.plot(test_zoom['ds'], test_zoom['y'], 
             label='Actual Data', color='black', alpha=0.6, linewidth=2, marker='o', markersize=3)
    plt.plot(test_zoom['ds'], p50_zoom, 
             label='Prophet P50', color='orange', linewidth=2.5)
    plt.plot(test_zoom['ds'], p90_zoom, 
             label='P90 Threshold', color='red', linestyle=':', linewidth=2)
    
    plt.fill_between(test_zoom['ds'], p50_zoom, p90_zoom, 
                     color='red', alpha=0.15)
    
    plt.title('90-Day Zoom: Prophet Detailed Performance Analysis (Q4)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Users')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the zoomed plot
    plt.savefig(os.path.join(outputs_dir, 'annual_prophet_1year_zoom.png'), dpi=300)


if __name__ == "__main__":
    run_long_term_prophet()