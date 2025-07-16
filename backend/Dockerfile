FROM python:3.11-slim

WORKDIR /app

ENV PROJECT_ROOT_DIR=/app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*


ENV SENTENCE_TRANSFORMERS_HOME=/app/cache
ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY save_model.py .
RUN python save_model.py

COPY ./app /app/app

EXPOSE 8000
