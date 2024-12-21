import asyncio
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.client.session import aiohttp
from aiogram.filters import CommandStart
from aiogram.types import Message

TOKEN = "7614079028:AAH29eU_J1Bot87IQUXZxu91-WQQaEF_mXw"

bot = Bot(token=TOKEN)
dp = Dispatcher()

API_URL = 'http://localhost:8085/Celery/add_to_processing'


async def send_to_api(user_message):
    async with aiohttp.ClientSession() as session:
        data = {"question": user_message}
        async with session.post(API_URL, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["result"]  # Возвращаем результат обработки
            else:
                result = await response.json()
                result = result["detail"]
                return result


@dp.message(CommandStart())
async def start(message: Message):
    await message.reply("Привет!")


@dp.message(F.text)
async def any_message_handler(message: Message):
    # Отправляем сообщение на API и ждем результат
    result = await send_to_api(message.text)
    await message.reply(result)



async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot is close")
