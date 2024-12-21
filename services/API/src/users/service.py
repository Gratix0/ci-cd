import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import TGUser


class UserService:
    def __init__(self, engine):
        self.engine = engine

    def get_user(self, tg_id: str):
        with Session(self.engine) as session:
            state = select(TGUser).where(TGUser.tg_id == tg_id)
            found = session.scalars(state).first()
            return found

    def create_user(self, user_id: str):
        with Session(self.engine) as session:
            user = TGUser(tg_id=user_id)
            session.add(user)
            session.commit()
            session.refresh(user)

    def update_user(self, user: TGUser):
        with Session(self.engine) as session:
            statement = select(TGUser).where(TGUser.tg_id == user.tg_id)
            db_object = session.scalars(statement).one()
            db_object.last_order = datetime.datetime.utcnow()
            session.commit()
            session.refresh(db_object)
