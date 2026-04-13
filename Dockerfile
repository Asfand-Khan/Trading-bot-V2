FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p ml_models historical_data

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8080/health',timeout=5); assert r.status_code==200" || exit 1

# Expose ports
EXPOSE 8080 5000

# Default: run the bot
CMD ["python", "main.py"]
