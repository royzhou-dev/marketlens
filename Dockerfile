FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss-cpu, lxml, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid the 2.5 GB CUDA download
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install gunicorn and remaining dependencies
COPY be/requirements.txt .
RUN pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY be/ ./be/
COPY fe/ ./fe/
COPY company_tickers.json .

# Create writable directories for runtime data
# (FAISS index and forecast models are regenerated each restart on free tier)
RUN mkdir -p /app/be/faiss_index /app/be/forecast_models /app/be/sentiment_cache

WORKDIR /app/be

EXPOSE 7860

# Use shell form so ${PORT:-7860} is expanded at runtime
# gthread workers allow concurrent SSE streaming without blocking
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --worker-class gthread --workers 1 --threads 4 --timeout 300 app:app"]
