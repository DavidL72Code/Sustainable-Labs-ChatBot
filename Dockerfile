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
# Every root-level module, not a hand-maintained list. Naming them one by one
# already shipped an image without supabase_store.py once, and splitting
# Chatbot.py into modules makes that mistake easy to repeat.
COPY *.py ./
COPY verified_question_bank.json ./

# API only — no templates/ or static/. Vercel serves the UI from frontend/.

# Include the prebuilt vector store so startup can skip first-run indexing
COPY chroma_db/ ./chroma_db/

# HF Spaces sometimes runs as non-root; make the cache + chroma dirs writable
RUN mkdir -p /app/chroma_db /app/.cache && chmod -R 777 /app/chroma_db /app/.cache

EXPOSE 7860

CMD ["python", "Chatbot.py"]
