import datetime

from sqlalchemy import TIMESTAMP, String, Integer, Column, func, ForeignKey, create_engine

from sqlalchemy.orm import relationship, declarative_base
from .config import DB_PASS, DB_USER, DB_HOST, DB_NAME, DB_PORT


engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
engine.connect()
Base = declarative_base()


class TGUser(Base):
    __tablename__ = "tg_user"

    user_id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    tg_id = Column(Integer, unique=True)
    first_order = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    last_order = Column(TIMESTAMP, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    questions = relationship("UserQuestions", back_populates="")



class UserQuestions(Base):
    __tablename__ = "user_questions"

    record_id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("tg_user.user_id"))
    question_theme = Column(String)

    user = relationship("TGUser",back_populates="questions")


class Responses(Base):
    __tablename__ = "responses"

    response_id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    response_text = Column(String)


Base.metadata.create_all(bind=engine)
