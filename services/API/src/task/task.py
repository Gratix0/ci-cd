from aiogram.client.session import aiohttp
from celery import Celery

# Настраиваем Celery с брокером и результатным бэкендом
celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Адрес вашего брокера (Redis)
    backend='redis://localhost:6379/1'  # Адрес бэкенда для хранения результатов
)


url = "http://localhost:8000/answer_to_bot"

@celery.task
async def add_message_to_queue(message: str):
    async with aiohttp.ClientSession() as session:
        try:
            data = {"question": message}
            with session.post(url, json=data) as response:
                if response.status == 200:  # Check for successful response
                    return response.json()["result"]
                else:
                    print(f"Request failed with status code: {response.status}")
                    return None
        except aiohttp.ClientError as e:
            print(f"An error occurred: {e}")
            return None

