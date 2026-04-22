from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..core.config import settings

# L'engine viene creato usando la URL validata da Pydantic Settings
engine = create_engine(settings.db_url)

# Configurazione della sessione
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency per iniettare la sessione del database nelle rotte FastAPI
def get_db():
    """
    Produce una sessione del database e si assicura che venga chiusa
    dopo l'uso, anche in caso di errori.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
