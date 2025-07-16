FROM python:3.11-slim

WORKDIR /app

# Tạo cache cho model và dữ liệu
ENV TRANSFORMERS_CACHE=/app/cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache
ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải trước model embedding để tiết kiệm thời gian khi khởi động
COPY save_model.py .
RUN python save_model.py

COPY ./app /app/app

EXPOSE 8000
