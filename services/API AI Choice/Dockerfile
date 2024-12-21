FROM python:3.11.3-slim-buster
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r req.txt && pip freeze > installed_packages.txt
CMD ["python", "main.py"]
