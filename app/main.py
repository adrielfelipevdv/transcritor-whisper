"""
Serviço próprio de transcrição — MARVENDAS.

FastAPI + faster-whisper + FFmpeg. Roda fora da Vercel (processo/container
próprio) porque transcrição pode consumir bastante RAM/CPU e demorar minutos
— incompatível com o limite de execução de uma função serverless.

Nenhuma chamada deste serviço depende de API de IA paga por token. O modelo
Whisper roda LOCALMENTE, uma vez carregado na memória do processo.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .ffmpeg_utils import assert_within_duration_limit, extract_audio_wav, ffmpeg_available
from .jobs import Job, job_manager
from .schemas import CreateJobFromUrl, CreateJobResponse, JobStatusResponse
from .security import safe_download
from .transcriber import TranscribeProgress, load_model, model_ready, transcribe_wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-transcription")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.temp_dir, exist_ok=True)
    if not ffmpeg_available():
        logger.warning("FFmpeg/ffprobe não encontrado no PATH — o serviço vai recusar todos os jobs até isso ser corrigido.")
    else:
        try:
            load_model()  # carrega uma vez no boot — nunca por requisição (seção 33)
        except Exception:
            logger.exception("Falha ao carregar o modelo Whisper no boot. Primeira transcrição tentará de novo.")
    yield


app = FastAPI(title="MARVENDAS Whisper Transcription Service", lifespan=lifespan)


def require_service_token(x_service_token: str | None = Header(default=None)) -> None:
    """Só o backend do MARVENDAS deve conseguir chamar este serviço — não é
    token pago de IA, é autenticação interna serviço-a-serviço (seção 34)."""
    if not settings.service_token:
        # Sem token configurado = ambiente de dev local; não bloqueia, mas loga.
        logger.warning("WHISPER_SERVICE_TOKEN não configurado — endpoint está SEM autenticação.")
        return
    if x_service_token != settings.service_token:
        raise HTTPException(status_code=401, detail="Token de serviço inválido")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_available(),
        "modelReady": model_ready(),
        "model": settings.whisper_model,
    }


async def _process_job(job: Job, media_path: str) -> None:
    """Corpo assíncrono de um job: valida duração, extrai áudio, transcreve,
    reporta progresso, grava resultado. `media_path` já existe em disco
    (upload salvo ou download concluído) dentro de job.temp_dir."""
    import asyncio

    job_manager.update(job.id, phase="Preparando arquivo…", progress=10)
    duration = await assert_within_duration_limit(media_path)

    job_manager.update(job.id, phase="Extraindo áudio…", progress=20)
    wav_path = os.path.join(job.temp_dir, "audio.wav")
    await extract_audio_wav(media_path, wav_path)

    def on_progress(p: TranscribeProgress) -> None:
        job_manager.update(job.id, phase=p.phase, progress=p.progress)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: transcribe_wav(wav_path, on_progress))

    job_manager.update(job.id, status="completed", phase="Concluído", progress=100, result=result)


@app.post("/transcriptions", response_model=CreateJobResponse, dependencies=[Depends(require_service_token)])
async def create_from_url(body: CreateJobFromUrl):
    job = await job_manager.create()
    job.temp_dir = tempfile.mkdtemp(prefix=f"job-{job.id}-", dir=settings.temp_dir)
    media_path = os.path.join(job.temp_dir, "source")

    async def runner(j: Job) -> None:
        job_manager.update(j.id, phase="Baixando arquivo…", progress=5)
        await safe_download(body.url, media_path)
        await _process_job(j, media_path)

    await job_manager.run(job.id, runner)
    return CreateJobResponse(jobId=job.id, status=job.status)


@app.post("/transcriptions/upload", response_model=CreateJobResponse, dependencies=[Depends(require_service_token)])
async def create_from_upload(file: UploadFile = File(...)):
    job = await job_manager.create()
    job.temp_dir = tempfile.mkdtemp(prefix=f"job-{job.id}-", dir=settings.temp_dir)
    media_path = os.path.join(job.temp_dir, "source")

    size = 0
    with open(media_path, "wb") as f:
        while chunk := await file.read(1024 * 256):
            size += len(chunk)
            if size > settings.max_file_size_bytes:
                f.close()
                import shutil
                shutil.rmtree(job.temp_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail="Arquivo acima do limite permitido")
            f.write(chunk)

    async def runner(j: Job) -> None:
        await _process_job(j, media_path)

    await job_manager.run(job.id, runner)
    return CreateJobResponse(jobId=job.id, status=job.status)


@app.get("/transcriptions/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_service_token)])
async def get_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado (ou expirou)")
    return JobStatusResponse(
        jobId=job.id, status=job.status, phase=job.phase, progress=job.progress,
        error=job.error, result=job.result,
    )


@app.delete("/transcriptions/{job_id}", dependencies=[Depends(require_service_token)])
async def cancel(job_id: str):
    ok = job_manager.request_cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job não encontrado ou já finalizado")
    return JSONResponse({"cancelled": True})
