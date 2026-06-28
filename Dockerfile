FROM python:3.11-slim

ARG HF_MODEL_NAME=your-username/mlops-group38-emotion

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python", "-m", "src.inference.infrence_from_hub"]
