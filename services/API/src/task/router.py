from aiogram.client.session import aiohttp

from .task import add_message_to_queue
from fastapi import APIRouter, BackgroundTasks, HTTPException

task_router = APIRouter(prefix="/Celery")

url = "http://localhost:8000/answer_to_bot"

@task_router.post("/add_to_processing")
async def add_to_processing(message: dict):
    async with aiohttp.ClientSession() as session:
        try:
            data = {"question": message}
            async with session.post(url, json=data) as response:  # Используем async with для асинхронного контекста
                if response.status == 200:  # Проверяем успешный ответ
                    json_response = await response.json()  # Ожидаем получения JSON-ответа
                    return {
                        "status": "Success",
                        "result": json_response["result"]  # Возвращаем результат обработки задачи
                    }
                else:
                    print(f"Request failed with status code: {response.status}")
                    res = await response.json()
                    res = res["detail"]
                    raise HTTPException(status_code=response.status, detail=res)
        except HTTPException as e:
            raise
        except aiohttp.ClientError as e:
            raise

