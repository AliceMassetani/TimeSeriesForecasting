from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt

from ..db.session import get_db
from ..services.auth_service import auth_service
from ..repositories.user_repository import UserRepository
from ..entities.user import User

# Schema di sicurezza HTTP Bearer — FastAPI genera automaticamente
# il campo "Authorization" nella documentazione Swagger/OpenAPI
bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency FastAPI che protegge gli endpoint.
    1. Estrae il token dall'header Authorization: Bearer <token>
    2. Decodifica e valida il JWT
    3. Verifica che l'utente esista nel database
    4. Restituisce l'oggetto User o lancia 401

    Uso: aggiungere Depends(get_current_user) ai router protetti.
    """
    token = credentials.credentials

    try:
        payload = auth_service.decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token scaduto",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token non valido",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: int = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token non valido: campo 'sub' mancante",
            headers={"WWW-Authenticate": "Bearer"},
        )

    repo = UserRepository(db)
    user = repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utente non trovato",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
