"""Extração/normalização de áudio via FFmpeg. Nunca altera o arquivo original
do usuário — sempre gera um novo arquivo mono/16kHz/wav ao lado."""
from __future__ import annotations

import asyncio
import json
import shutil

from .config import settings


class FfmpegNotFoundError(Exception):
    pass


class InvalidMediaError(Exception):
    pass


class MediaTooLongError(Exception):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


async def probe_duration_seconds(input_path: str) -> float:
    """Lê a duração via ffprobe — falha rápido em arquivo corrompido/inválido
    antes de gastar tempo extraindo áudio."""
    if not ffmpeg_available():
        raise FfmpegNotFoundError("ffmpeg/ffprobe não encontrado no PATH")
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", input_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise InvalidMediaError(f"Arquivo inválido ou corrompido: {stderr.decode(errors='ignore')[:300]}")
    try:
        data = json.loads(stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise InvalidMediaError("Não foi possível ler a duração do arquivo") from e


async def extract_audio_wav(input_path: str, output_path: str) -> None:
    """Converte para WAV mono 16kHz (formato ideal para o Whisper) — extrai só
    o áudio, sem carregar/processar o vídeo inteiro no modelo."""
    if not ffmpeg_available():
        raise FfmpegNotFoundError("ffmpeg não encontrado no PATH")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", input_path,
        "-vn",              # descarta vídeo
        "-ac", "1",         # mono
        "-ar", "16000",     # 16kHz
        "-f", "wav",
        output_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise InvalidMediaError(f"FFmpeg falhou ao extrair áudio: {stderr.decode(errors='ignore')[:400]}")


async def assert_within_duration_limit(input_path: str) -> float:
    duration = await probe_duration_seconds(input_path)
    if duration > settings.max_duration_seconds:
        raise MediaTooLongError(
            f"Duração {duration:.0f}s acima do limite de {settings.max_duration_seconds}s"
        )
    return duration
