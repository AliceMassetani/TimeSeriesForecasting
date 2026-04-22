from fastapi import FastAPI
from app.db.session import engine
from app.db.base import Base
from app.api.predictions import router as predictions_router

# Creazione delle tabelle nel database all'avvio
Base.metadata.create_all(bind=engine)

# Inizializzazione dell'app FastAPI
app = FastAPI(
    title="Predictive Autoscaler ML API",
    description="API per la previsione della capacità InUse basata su modelli TSMixer",
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
# Nota: puoi aggiungere un prefix se vuoi, es: prefix="/api"
app.include_router(predictions_router, tags=["Forecasting"])