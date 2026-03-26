import pandas as pd
import holidays
import os

def prepare_data():
    # 1. Paths configuration (going up from src/ to datasets/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'datasets', 'Dati.csv')
    output_path = os.path.join(base_dir, '..', 'datasets', 'Dati_processed.csv')

    print("--- 1. Loading Raw Dataset ---")
    # Clean reading of the original AWS CSV file
    df = pd.read_csv(data_path, skiprows=4)

    # Renaming the column Label to TimeStamp
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'TimeStamp'})  

    # Timestamp conversion (dayfirst=True prevents pandas warnings)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)

    print("--- 2. Feature Engineering (Covariates for Chronos-2) ---")
    # Extracting temporal features directly from the TimeStamp column
    df['hour'] = df['TimeStamp'].dt.hour
    df['day_of_week'] = df['TimeStamp'].dt.dayofweek
    
    # Adding Italian holidays flag (cloud load changes on holidays)
    it_holidays = holidays.Italy()
    df['is_holiday'] = df['TimeStamp'].apply(lambda x: int(x in it_holidays))

    # Adding the mandatory identification column required by Chronos-2 architecture
    df['item_id'] = 'appstream_fleet'

    print("--- 3. Saving Processed Dataset ---")
    # Saving the CSV without the index to keep TimeStamp as a standard column
    df.to_csv(output_path, index=False)
    print(f"Dataset ready for inference saved at: {output_path}")

if __name__ == "__main__":
    prepare_data()