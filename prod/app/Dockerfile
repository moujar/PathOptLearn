FROM python:3.11-slim

WORKDIR /app

# System deps: gcc + libpq for psycopg2-binary, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Wait for Postgres + Neo4j before starting (handled by depends_on + healthchecks in compose)
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
