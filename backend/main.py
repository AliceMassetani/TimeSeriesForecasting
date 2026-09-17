from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Carica il file .env dalla root del progetto
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.db.session import engine
from app.db.base import Base
from app.api.backtest import router as backtest_router
from app.api.forecast import router as forecast_router
from app.api.training import router as training_router
from app.api.auth import router as auth_router
from app.api.internal import router as internal_router
from app.core.security import get_current_user

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

# Endpoint di benvenuto (pubblico)
@app.get("/")
async def root():
    return {
        "message": "Benvenuto all'API di Predictive Autoscaler!",
        "status": "Online",
        "docs": "/docs"
    }

# Router PUBBLICO — Autenticazione (register / login)
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])

# Router PROTETTI — Richiedono JWT valido nell'header Authorization: Bearer <token>
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtest"], dependencies=[Depends(get_current_user)])
app.include_router(forecast_router, prefix="/api/forecast", tags=["Forecast"], dependencies=[Depends(get_current_user)])
app.include_router(training_router, prefix="/api/train", tags=["Training"], dependencies=[Depends(get_current_user)])

# Router INTERNO — Comunicazione Service-to-Service (Lambda → Backend)
# Nessuna autenticazione JWT: protetto a livello di rete (Nginx non inoltra /internal/)
app.include_router(internal_router, prefix="/internal", tags=["Internal"])