FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Cloud Run inyecta el puerto en la variable $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
