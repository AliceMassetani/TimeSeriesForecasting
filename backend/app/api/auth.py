from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..schemas.auth import UserRegister, UserLogin, TokenResponse, UserOut
from ..services.auth_service import auth_service
from ..repositories.user_repository import UserRepository
from ..entities.user import User
from ..core.security import get_current_user

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister, db: Session = Depends(get_db)):
    """
    Registra un nuovo utente.
    - Verifica che lo username non sia già in uso.
    - Hash della password con BCrypt.
    - Salva l'utente nel database.
    """
    repo = UserRepository(db)

    # Controlla duplicati
    if repo.get_by_username(data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username già registrato"
        )

    # Crea l'utente con password hashata
    user = User(
        username=data.username,
        password_hash=auth_service.hash_password(data.password),
    )
    created_user = repo.create(user)

    return created_user


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin, db: Session = Depends(get_db)):
    """
    Autentica un utente e restituisce un token JWT.
    - Cerca l'utente per username.
    - Verifica la password con BCrypt.
    - Genera e restituisce un JWT firmato con JWT_SECRET.
    """
    repo = UserRepository(db)
    user = repo.get_by_username(data.username)

    if not user or not auth_service.verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.create_token(user.id, user.username)

    return TokenResponse(access_token=token)

@router.delete("/users/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Elimina l'account utente.
    Prevenzione IDOR: verifica che l'ID passato nell'URL sia uguale a quello decodificato dal JWT.
    """
    if id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operazione non consentita: non puoi eliminare l'account di un altro utente (IDOR Prevented)."
        )
    
    repo = UserRepository(db)
    success = repo.delete_user(id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato.")
    return
