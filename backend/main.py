from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import engine
from app.db.base import Base
from app.api.backtest import router as backtest_router
from app.api.forecast import router as forecast_router
from app.api.training import router as training_router

# Creazione delle tabelle nel database all'avvio
Base.metadata.create_all(bind=engine)

# Inizializzazione dell'app FastAPI
app = FastAPI(
    title="Predictive Autoscaler ML API",
    description="API per Backtesting, Forecasting e Training della capacità InUse",
    version="1.0.0"
)

# Configurazione CORS - ESSENZIALE per far funzionare il frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint di benvenuto
@app.get("/")
async def root():
    return {
        "message": "Benvenuto all'API di Predictive Autoscaler!",
        "status": "Online",
        "docs": "/docs"
    }

# Inclusione dei router
app.include_router(backtest_router, prefix="/backtest", tags=["Backtest"])
app.include_router(forecast_router, prefix="/forecast", tags=["Forecast"])
app.include_router(training_router, prefix="/train", tags=["Training"])