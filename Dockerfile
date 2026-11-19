FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir ".[server]"

EXPOSE 8000
CMD ["blinkthink-serve"]
