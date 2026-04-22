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
