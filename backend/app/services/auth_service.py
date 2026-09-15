from datetime import datetime, timedelta, timezone
import bcrypt
import jwt
from ..core.config import settings


class AuthService:
    """
    Servizio di autenticazione stateless.
    - BCrypt per l'hashing sicuro delle password
    - JWT per la generazione e verifica dei token
    """

    # ── Password Hashing (BCrypt) ──────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        """Genera un hash BCrypt dalla password in chiaro."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifica che la password in chiaro corrisponda all'hash BCrypt."""
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )

    # ── JWT Token ──────────────────────────────────────────────────────

    @staticmethod
    def create_token(user_id: int, username: str) -> str:
        """
        Genera un token JWT firmato con il secret dall'ambiente.
        Il payload contiene: sub (user_id), username, exp (scadenza).
        """
        payload = {
            "sub": user_id,
            "username": username,
            "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.JWT_EXPIRATION),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

    @staticmethod
    def decode_token(token: str) -> dict:
        """
        Decodifica e valida un token JWT.
        Solleva jwt.ExpiredSignatureError se scaduto,
        jwt.InvalidTokenError per token non validi.
        """
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])


# Istanza singleton del servizio
auth_service = AuthService()
