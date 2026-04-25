FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

# System deps needed by some Python wheels (torch, numpy, gradio, etc.)
# + Node.js 20 for building the React frontend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require a non-root user (UID 1000) with a writable
# HOME directory. Create the user FIRST so we can copy files with the
# right ownership.
RUN useradd --create-home --shell /bin/bash --uid 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/usr/local/bin:$PATH

WORKDIR /home/user/app
RUN chown -R user:user /home/user

USER user

# Install Python deps first for better layer caching.
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Build the React frontend (no env vars needed — Hume creds are fetched
# at runtime from /api/config which reads HF Space secrets).
COPY --chown=user:user frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --no-audit --no-fund

# Copy the rest of the app (including the pre-built frontend/dist that
# we now commit to git — bypasses HF Spaces' silently-failing npm build).
COPY --chown=user:user . .

# Build the React bundle ONLY IF a pre-built dist isn't already shipped.
# This makes deploys deterministic — the dist that gets served is the
# one committed to the repo, not a re-build that may fail on HF.
RUN if [ ! -f frontend/dist/index.html ]; then \
        echo "[docker] no pre-built dist; running npm run build" && \
        cd frontend && npm run build; \
    else \
        echo "[docker] pre-built dist found; skipping npm run build"; \
    fi

# HF Spaces injects $PORT (default 7860). server:api is the FastAPI app
# defined in server.py — exposes /reset, /step, /state and mounts the
# Vishwamitra Gradio UI at /ui.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:api --host 0.0.0.0 --port ${PORT:-7860}"]
