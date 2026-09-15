from pydantic import BaseModel, Field
from datetime import datetime


class UserRegister(BaseModel):
    """Schema per la richiesta di registrazione."""
    username: str = Field(..., min_length=3, max_length=150, description="Username univoco")
    password: str = Field(..., min_length=6, max_length=128, description="Password in chiaro (verrà hashata con BCrypt)")


class UserLogin(BaseModel):
    """Schema per la richiesta di login."""
    username: str = Field(..., description="Username registrato")
    password: str = Field(..., description="Password in chiaro")


class TokenResponse(BaseModel):
    """Schema per la risposta contenente il token JWT."""
    access_token: str = Field(..., description="Token JWT firmato")
    token_type: str = Field(default="bearer", description="Tipo di token")


class UserOut(BaseModel):
    """Schema per i dati utente restituiti (senza password)."""
    id: int
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}
