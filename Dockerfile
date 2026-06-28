FROM python:3.11-slim

# Build-time override: pass your HF repo id when building
# e.g.  docker build --build-arg HF_MODEL_NAME=Sonu-Kumar-IIT/mlops-group38-emotion .
ARG HF_MODEL_NAME=Sonu-kumar-IIT/distilbert-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_MODEL_NAME=${HF_MODEL_NAME} \
    INPUT_TEXT="I feel really happy and excited today!" \
    HF_TOKEN=""

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Pre-create directories that config.py creates at import time,
# then create a non-root user and hand over ownership for security
RUN mkdir -p /app/artifacts/eval_results /app/checkpoints && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.inference.infrence_from_hub"]
