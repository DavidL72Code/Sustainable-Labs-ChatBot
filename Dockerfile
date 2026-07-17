# Hugging Face Space (Docker SDK) — Flask backend for the SSL Chatbot
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    CHATBOT_HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# System deps needed by chromadb (sqlite, build tooling for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app
COPY Chatbot.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY SEED_DOCUMENTS/ ./SEED_DOCUMENTS/

# Include the prebuilt vector store so startup can skip first-run indexing
COPY chroma_db/ ./chroma_db/

# HF Spaces sometimes runs as non-root; make the cache + chroma dirs writable
RUN mkdir -p /app/chroma_db /app/.cache && chmod -R 777 /app/chroma_db /app/.cache

EXPOSE 7860

CMD ["python", "Chatbot.py"]
