"""Modelos Pydantic — contrato HTTP do serviço. Não conhece nada do frontend,
só devolve estrutura genérica de transcrição (start/end/text + metadados)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[list[WordTimestamp]] = None


class TranscriptionResult(BaseModel):
    language: str
    duration: float
    model: str
    device: str
    segments: list[TranscriptSegment]
    text: str  # concatenação dos segmentos — conveniência para quem só quer a string


class CreateJobFromUrl(BaseModel):
    url: str = Field(..., description="URL pública do vídeo/áudio a transcrever")


class JobStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    phase: str = ""
    progress: int = Field(0, ge=0, le=100)
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None


class CreateJobResponse(BaseModel):
    jobId: str
    status: JobStatus
