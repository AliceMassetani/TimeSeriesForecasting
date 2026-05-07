from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configurazione centralizzata dell'applicazione tramite Pydantic Settings.
    Legge automaticamente le variabili d'ambiente o dal file .env.
    """
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_NAME: str

    # PID Controller Settings — P+D stabile, Kd ridotto per evitare oscillazioni
    PID_KP: float = 0.3
    PID_KI: float = 0.0
    PID_KD: float = 0.05
    PID_DERIVATIVE_EXP: float = 1.0
    PID_SCALE_DOWN_PENALTY: float = 0.6

    # Configurazione Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignora variabili extra non definite qui
    )

    @property
    def db_url(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}/{self.DB_NAME}"

# Istanza globale delle impostazioni
settings = Settings()
