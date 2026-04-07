ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY conversation_dataset_generator/ conversation_dataset_generator/
COPY generate.py .
COPY character-config/ character-config/

ENTRYPOINT ["python3", "generate.py"]
