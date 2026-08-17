FROM python:3.11-slim

WORKDIR /app

# Copy frontend requirements and install
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy frontend code
COPY frontend /app/frontend

ENV PYTHONPATH=/app
ENV BACKEND_URL=http://backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
