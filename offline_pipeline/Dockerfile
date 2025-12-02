# Lightweight CPU image for offline pipeline
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for torch/sentence-transformers and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Run the main entrypoint directly (files live in /app)
CMD ["python", "-u", "main.py"]
