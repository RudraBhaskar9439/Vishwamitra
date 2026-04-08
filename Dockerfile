FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

# System deps needed by some Python wheels (torch, numpy, gradio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
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

# Copy the rest of the app.
COPY --chown=user:user . .

# HF Spaces injects $PORT (default 7860). server:api is the FastAPI app
# defined in server.py — exposes /reset, /step, /state and mounts the
# Vishwamitra Gradio UI at /ui.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:api --host 0.0.0.0 --port ${PORT:-7860}"]
