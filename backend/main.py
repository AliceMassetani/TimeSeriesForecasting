from fastapi import FastAPI
from app.db.session import engine
from app.db.base import Base
from app.api.backtest import router as backtest_router

# Creazione delle tabelle nel database all'avvio
Base.metadata.create_all(bind=engine)

# Inizializzazione dell'app FastAPI
app = FastAPI(
    title="Predictive Autoscaler ML API",
    description="API per Backtesting e Forecasting della capacità InUse",
    version="1.0.0"
)

# Endpoint di benvenuto
@app.get("/")
async def root():
    return {
        "message": "Benvenuto all'API di Predictive Autoscaler!",
        "status": "Online",
        "docs": "/docs"
    }

# Inclusione dei router (Controllers) per rendere l'app modulare
app.include_router(backtest_router, prefix="/backtest", tags=["Backtest"])