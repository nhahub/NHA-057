FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./

RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
