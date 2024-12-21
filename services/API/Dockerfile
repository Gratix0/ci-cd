FROM python:3.11.3-slim-buster
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r req.txt
CMD ["python", "-m", "main.py"]
