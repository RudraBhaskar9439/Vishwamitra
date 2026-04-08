FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 7860

# Run the OpenEnv FastAPI server (mounts Gradio UI at /ui).
# To run the headless inference suite instead:
#   docker run ... python inference.py
CMD ["uvicorn", "server:api", "--host", "0.0.0.0", "--port", "7860"]
