"""
Fila de jobs em memória, com concorrência limitada (semáforo) e diretório
temporário isolado por job (limpo em success/failure/cancel).

Não usa banco/broker externo — o serviço é um único worker (ver seção 29 do
pedido: "controle de concorrência", não "fila distribuída"). Se um dia
precisar escalar horizontalmente, trocar por Redis/RQ mantendo o mesmo
contrato de status.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import settings
from .schemas import TranscriptionResult

logger = logging.getLogger("whisper-transcription")


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | processing | completed | failed | cancelled
    phase: str = "Na fila…"
    progress: int = 0
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    temp_dir: Optional[str] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._semaphore = asyncio.Semaphore(settings.transcription_concurrency)
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id)
        async with self._lock:
            self._jobs[job_id] = job
        self._cleanup_expired()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)

    def request_cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status in ("completed", "failed", "cancelled"):
            return False
        job.cancel_requested = True
        if job.status == "queued" and job.task:
            job.task.cancel()
        return True

    async def run(self, job_id: str, coro_factory) -> None:
        """Roda `coro_factory()` respeitando o limite de concorrência.
        `coro_factory` recebe o `Job` e deve chamar `job_manager.update(...)`
        para reportar progresso/estado."""
        job = self._jobs[job_id]

        async def _runner():
            async with self._semaphore:
                if job.cancel_requested:
                    self.update(job_id, status="cancelled", phase="Cancelado")
                    return
                self.update(job_id, status="processing", phase="Preparando arquivo…", progress=5)
                try:
                    await coro_factory(job)
                except asyncio.CancelledError:
                    self.update(job_id, status="cancelled", phase="Cancelado")
                except Exception as e:  # noqa: BLE001 — nunca vaza traceback pro cliente
                    logger.exception("Job %s falhou", job_id)
                    self.update(job_id, status="failed", error=_friendly_error(e))
                finally:
                    job.finished_at = time.time()
                    self._cleanup_job_files(job)

        job.task = asyncio.create_task(_runner())

    def _cleanup_job_files(self, job: Job) -> None:
        if job.temp_dir and os.path.isdir(job.temp_dir):
            shutil.rmtree(job.temp_dir, ignore_errors=True)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if j.finished_at and (now - j.finished_at) > settings.job_ttl_seconds
        ]
        for jid in expired:
            self._jobs.pop(jid, None)


def _friendly_error(e: Exception) -> str:
    """Nunca mostra traceback Python — mensagens amigáveis por tipo conhecido,
    genérico honesto para o resto (seção 27)."""
    from .ffmpeg_utils import FfmpegNotFoundError, InvalidMediaError, MediaTooLongError
    from .security import DownloadTooLargeError, UnsafeUrlError

    mapping = {
        FfmpegNotFoundError: "Serviço de transcrição mal configurado (FFmpeg ausente). Avise o suporte.",
        InvalidMediaError: "Arquivo inválido, corrompido ou sem áudio legível.",
        MediaTooLongError: str(e),
        DownloadTooLargeError: "Arquivo muito grande para transcrever.",
        UnsafeUrlError: "URL não permitida ou inacessível.",
    }
    for exc_type, msg in mapping.items():
        if isinstance(e, exc_type):
            return msg
    return "Falha ao transcrever. Tente novamente ou use outro arquivo/URL."


job_manager = JobManager()
