# 1. Create your env file
cp .env.example .env

# 2. Build and start all services
docker compose up --build

# 3. First run only: wait ~30s for ollama-init to pull the model
#    Watch with: docker compose logs -f ollama-init
