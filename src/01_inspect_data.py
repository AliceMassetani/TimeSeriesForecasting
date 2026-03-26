import pandas as pd
import os

# Define path to reach the datasets folder
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Dati.csv'))

try:
    # Load dataset with semicolon separator
    df = pd.read_csv(path, skiprows=4)

    # Renaming the column Label to TimeStamp
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'TimeStamp'})  

    print("--- DATA INSPECTION ---")

    # 1. Missing Values (NaN) Check
    # Chronos-2 works best with continuous sequences; gaps should be identified
    print(f"Missing values (NaN) in CapacityUtilization: {df['CapacityUtilization'].isnull().sum()}")
    
    

    # 2. Temporal Analysis
    # Ensure the TimeStamp column is correctly formatted for sequence alignment
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        start = df['TimeStamp'].min()
        end = df['TimeStamp'].max()
        
        # Calculate the median distance between data points to check frequency
        delta = df['TimeStamp'].diff().median()
        
        print(f"Time Range: from {start} to {end}")
        print(f"Data Frequency: approximately every {delta}")
        
    print(f"Total rows available: {len(df)}")

except Exception as e:
    print(f"Error during data inspection: {e}")