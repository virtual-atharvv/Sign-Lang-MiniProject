FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (includes OpenHands transitive deps)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full codebase
COPY . .

# Install the OpenHands repo as a local package (already cloned in openhands_repo/)
# This makes `from openhands.apis.inference import ...` work without sys.path hacks.
RUN if [ -d openhands_repo ]; then \
        pip install --no-cache-dir -e openhands_repo/; \
    else \
        echo "WARNING: openhands_repo not found. Cloning from GitHub..."; \
        git clone https://github.com/AI4Bharat/OpenHands.git openhands_repo; \
        pip install --no-cache-dir -e openhands_repo/; \
    fi

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# BUILD_KNN_ON_START=true → build_knn_index.py runs at startup if index missing.
# Set to false to skip (requires knn_embeddings.npy to already exist in the image).
ENV BUILD_KNN_ON_START=false

EXPOSE 7860

# Startup script: optionally build KNN index, then launch FastAPI
COPY backend/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
