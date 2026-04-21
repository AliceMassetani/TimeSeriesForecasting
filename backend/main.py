from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from darts.models import TSMixerModel
import pandas as pd
import io
import os

# Configurazione Database (legge dal tuo .env tramite Docker)
DB_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definizione della tabella come l'hai chiesta
class PredictionResult(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    prediction = Column(Float)
    actual_value = Column(Float)
    diff_percentage = Column(Float)
    prediction_rounded = Column(Integer)
    actual_rounded = Column(Integer)
    diff_rounded_percentage = Column(Float)

# Crea la tabella se non esiste
Base.metadata.create_all(bind=engine)

# Carica il modello addestrato
model_path = "models/tsmixer_champion_target.pt"
model = TSMixerModel.load(model_path)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Benvenuto all'API di TimeSeriesForcasting!"}

@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):

    TARGET = "InUseCapacity"
    START = model.input_chunk_length #168
    FORCAST_HORIZON = 2
    STRIDE = 1

    # 1. Leggi il CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')), parse_dates=['TimeStamp'])
    
    # 2. Esegui la previsione
    # Assicurati che il DataFrame abbia colonne 'timestamp' e 'InUseCapacity'
    df = df.sort_values("TimeStamp").reset_index(drop=True)
    series = TimeSeries.from_dataframe(df, time_col='TimeStamp', value_cols=TARGET)
    series = fill_missing_values(TimeSeries.from_dataframe(df, time_col="TimeStamp", value_cols=TARGET, freq="h"))


    forecast = model.historical_forecasts(
        series,
        #prende START come contesto iniziale,
        start=START,
        #poi fa una previsione ti FORECAST_HORIZON ore,
        forecast_horizon=FORCAST_HORIZON,
        #poi fa un passo avanti di STRIDE ore e ripete il processo,
        stride=STRIDE,
        #retrain=False significa che non riaddestra il modello,
        retrain=False,
        #verbose=True significa che stampa il progresso,
        verbose=True,
        #last_point_only=True significa che prende solo l'ultimo punto della previsione,
        #considerando che uno stesso punto viene previsto 2 volte e la seconda
        #è la piu accurata perchè svolta sui dati più recenti
        last_points_only=True
    )

    # Per ora simuliamo un risultato
    results = []
    for ts, pred_val in forecast.to_dataframe().iterrows():
        # ts è il timestamp, pred_val è il valore predetto
    
        # Cerchiamo il valore reale nel DF originale usando il timestamp come filtro
        real_value_series = df.loc[df['TimeStamp'] == ts, TARGET]   

        if real_value_series.empty:
            continue
        
        real = real_value_series.values[0]
        pred = pred_val.iloc[0] # Estrae il numero dal mini-array di Darts
          
        # Calcoli arrotondati che hai richiesto
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
    
    # 3. Salva nel DB
    db = SessionLocal()
    try:
        db.add_all(results)
        db.commit()
    finally:
        db.close()
    
    return {"message": f"Salvate {len(results)} previsioni nel database"}