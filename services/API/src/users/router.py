import asyncio
import time

from fastapi import APIRouter, Response
from fastapi import Depends, HTTPException
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.orm import Session

from fastapi_cache.decorator import cache
# from sqlalchemy import create_engine
#
#
# from .service import UserService
# from .config import DB_PASS, DB_USER, DB_HOST, DB_NAME, DB_PORT

router = APIRouter()

# engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
# engine.connect()
# userService = UserService(engine=engine)


# @router.get("/add_new_user")
# @cache(expire=30)
# async def add_user(message: dict):
#     user = userService.get_user(message["user_id"])
#     if user is None:
#         userService.create_user(message["user_id"])
#
#     else:
#         userService.update_user(user)
#
#     user = userService.get_user(message["user_id"])
#     return {
#         "status": "200",
#         "result": message["message"]  # Возвращаем результат обработки задачи
#     }
