FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code
COPY backend /app/backend

ENV PYTHONPATH=/app
ENV CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db
ENV BM25_PERSIST_PATH=/app/data/bm25_index.pkl

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
