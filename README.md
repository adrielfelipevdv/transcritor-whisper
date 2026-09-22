---
title: Transcritor Whisper
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8100
pinned: false
---

# MARVENDAS — Serviço de Transcrição (Whisper local)

Serviço próprio de speech-to-text para a tela **Análise de Criativos → Diagnóstico Criativo**
do MARVENDAS. Substitui a chamada à API paga da OpenAI (`whisper-1`) por um
motor Whisper **executado localmente** (`faster-whisper`), sem depender de
tokens/API de IA paga.

Stack: **Python + FastAPI + faster-whisper + FFmpeg**. Roda fora da Vercel
(processo/container próprio), porque transcrição pode consumir bastante
RAM/CPU e demorar minutos — incompatível com o limite de execução de uma
função serverless.

## Rodando localmente (sem Docker)

Requisitos: Python 3.11+, FFmpeg instalado no PATH (`ffmpeg -version` deve funcionar).

```bash
cd services/whisper-transcription
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # ajuste se quiser
export $(cat .env | xargs)  # ou defina as envs manualmente no Windows
uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
```

Teste:

```bash
curl http://localhost:8100/health
# {"status":"ok","ffmpeg":true,"modelReady":true,"model":"small"}
```

O primeiro request de transcrição baixa o modelo `small` (ou o que estiver em
`WHISPER_MODEL`) automaticamente do Hugging Face Hub para `WHISPER_MODEL_CACHE_DIR`
— só acontece uma vez (fica em cache).

## Rodando via Docker

```bash
cd services/whisper-transcription
cp .env.example .env
docker compose up --build
```

Isso sobe o serviço na porta `8100`, com volumes persistentes para o cache do
modelo (`whisper-models`) e o diretório temporário de jobs (`whisper-tmp`).

Para rodar com GPU (NVIDIA + CUDA), troque a imagem base do `Dockerfile` para
uma imagem com CUDA (ex.: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` + instalar
Python) e rode o container com `--gpus all` / `runtime: nvidia` no compose.
`WHISPER_DEVICE=auto` detecta CUDA automaticamente quando disponível.

## Endpoints

Todos exigem o header `x-service-token: <WHISPER_SERVICE_TOKEN>` (autenticação
interna serviço-a-serviço — **não é token de IA paga**; sem essa env definida
o serviço roda sem autenticação, útil só em dev local).

- `GET /health` — status do serviço, se o FFmpeg está disponível e se o modelo já carregou.
- `POST /transcriptions` — body `{"url": "https://..."}`. Retorna `{"jobId": "...", "status": "queued"}`.
- `POST /transcriptions/upload` — multipart, campo `file`. Retorna `{"jobId": "...", "status": "queued"}`.
- `GET /transcriptions/{jobId}` — status do job: `{"status", "phase", "progress", "error"?, "result"?}`.
- `DELETE /transcriptions/{jobId}` — cancela um job ainda não concluído (best-effort).

Formato de `result` (quando `status === "completed"`):

```json
{
  "language": "pt",
  "duration": 71.4,
  "model": "small",
  "device": "cpu",
  "text": "Quanto tempo leva para ter resultado no Mercado Livre? ...",
  "segments": [
    { "start": 0.0, "end": 4.8, "text": "Quanto tempo leva para ter resultado no Mercado Livre?" },
    { "start": 4.8, "end": 10.2, "text": "Isso depende muito do estágio em que você está." }
  ]
}
```

Com `WHISPER_WORD_TIMESTAMPS=true`, cada segmento também traz `"words": [{"word","start","end"}, ...]`.

## Configuração (tudo centralizado em `app/config.py`)

| Variável | Default | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `small` | `small` \| `medium` \| `large-v3` \| `distil-large-v3` \| ... |
| `WHISPER_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `auto` \| `int8` \| `float16` \| `float32` |
| `WHISPER_LANGUAGE` | `pt` | Vazio/`null` = detecção automática |
| `WHISPER_WORD_TIMESTAMPS` | `false` | Liga timestamps por palavra (mais custo de CPU) |
| `WHISPER_VAD_FILTER` | `true` | VAD embutido do faster-whisper (Silero por baixo) |
| `TRANSCRIPTION_CONCURRENCY` | `1` | Jobs simultâneos |
| `MAX_FILE_SIZE_MB` | `300` | Teto de upload/download |
| `MAX_DURATION_SECONDS` | `1200` | Teto de duração do áudio/vídeo |
| `WHISPER_SERVICE_TOKEN` | vazio | Token interno — obrigatório em produção |
| `WHISPER_MODEL_CACHE_DIR` | `/models` | Onde o modelo baixado fica em cache |
| `TEMP_DIR` | `/tmp/whisper-jobs` | Diretório de trabalho por job (limpo ao final) |

## Testando com um modelo diferente

Trocar só a env `WHISPER_MODEL` (e reiniciar o serviço) — nenhum código muda:

```bash
WHISPER_MODEL=medium docker compose up --build
```

## Não usa API de IA paga

Este serviço nunca faz nenhuma chamada de rede para `api.openai.com`,
`generativelanguage.googleapis.com` ou qualquer outro provedor de IA. A única
chamada de rede que ele faz é para BAIXAR o arquivo de vídeo/áudio que o
usuário forneceu por URL (com proteção anti-SSRF — ver `app/security.py`).
