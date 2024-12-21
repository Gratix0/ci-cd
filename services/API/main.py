from src.users.router import router as users_router
from src.task.router import task_router

from fastapi import FastAPI

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import uvicorn


from redis import asyncio as aioredis

app = FastAPI(
    title="AI-MPT-API"
)

app.include_router(users_router)
app.include_router(task_router)

@app.on_event("startup")
async def startup():
    redis = await aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8085, reload=True)
