import redis

# Подключение к Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Проверка подключения
try:
    pong = r.ping()
    if pong:
        print("Успешное подключение к Redis!")
except redis.exceptions.ConnectionError as e:
    print(f"Ошибка подключения к Redis: {e}")