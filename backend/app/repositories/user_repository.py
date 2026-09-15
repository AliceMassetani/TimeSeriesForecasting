from typing import Optional
from sqlalchemy.orm import Session
from ..entities.user import User


class UserRepository:
    """
    Repository per le operazioni CRUD sulla tabella users.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_by_username(self, username: str) -> Optional[User]:
        """Cerca un utente per username."""
        return self.db.query(User).filter(User.username == username).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Cerca un utente per ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def create(self, user: User) -> User:
        """Inserisce un nuovo utente nel database."""
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
