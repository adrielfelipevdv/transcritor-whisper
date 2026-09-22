# Serviço próprio de transcrição — Whisper local (faster-whisper), sem API paga.
# Imagem CPU por padrão. Para GPU, trocar a base por uma imagem com CUDA
# (ex.: nvidia/cuda:12.1.0-runtime-ubuntu22.04) mantendo o resto igual.
FROM python:3.11-slim

# FFmpeg é obrigatório (extração/normalização de áudio).
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Cache do modelo Whisper — monte um volume aqui em produção para não baixar
# o modelo de novo a cada deploy (ver seção 32/README).
ENV WHISPER_MODEL_CACHE_DIR=/models
ENV TEMP_DIR=/tmp/whisper-jobs
RUN mkdir -p /models /tmp/whisper-jobs

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
