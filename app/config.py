"""
Configuração central do serviço de transcrição — NENHUM valor deve ficar
hardcoded fora daqui (pedido explícito: "não espalhar configuração pelo código").

Tudo é lido de variáveis de ambiente, com defaults seguros para rodar sem GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ── Modelo Whisper ───────────────────────────────────────────────────────
    # Configurável sem tocar em código: small | medium | large-v3 | distil-large-v3 | ...
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    # auto | cpu | cuda — "auto" detecta CUDA em runtime e cai para CPU se ausente.
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    # int8 é o mais leve/rápido em CPU; float16 é o ideal em GPU. "auto" escolhe
    # com base no device resolvido (ver transcriber.py).
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
    # Custa mais processamento — desligado por padrão (pedido explícito: "se
    # houver custo relevante de performance, deixar configurável").
    whisper_word_timestamps: bool = _bool("WHISPER_WORD_TIMESTAMPS", False)
    # Idioma fixo em pt-BR por padrão; None (auto) fica disponível via env.
    whisper_language: str | None = os.getenv("WHISPER_LANGUAGE", "pt") or None
    whisper_beam_size: int = _int("WHISPER_BEAM_SIZE", 5)
    # VAD embutido do faster-whisper (Silero VAD por baixo dos panos) — evita
    # blocos gigantes/silêncio sendo transcrito como fala.
    whisper_vad_filter: bool = _bool("WHISPER_VAD_FILTER", True)
    whisper_vad_min_silence_ms: int = _int("WHISPER_VAD_MIN_SILENCE_MS", 500)

    # ── Onde o modelo fica em cache (não baixar de novo a cada boot/transcrição) ─
    model_cache_dir: str = os.getenv("WHISPER_MODEL_CACHE_DIR", "/models")

    # ── Limites ──────────────────────────────────────────────────────────────
    max_file_size_bytes: int = _int("MAX_FILE_SIZE_MB", 300) * 1024 * 1024
    max_duration_seconds: int = _int("MAX_DURATION_SECONDS", 60 * 20)  # 20 min
    download_timeout_seconds: int = _int("DOWNLOAD_TIMEOUT_SECONDS", 60)

    # ── Concorrência (fila) ──────────────────────────────────────────────────
    # Quantos jobs de transcrição podem rodar AO MESMO TEMPO. Em CPU, manter
    # baixo (o modelo já usa vários threads internamente); em GPU pode subir.
    transcription_concurrency: int = _int("TRANSCRIPTION_CONCURRENCY", 1)
    # Threads de CPU usadas pelo próprio faster-whisper por transcrição.
    whisper_cpu_threads: int = _int("WHISPER_CPU_THREADS", 4)

    # ── Diretório temporário (limpo por job em success/failure/cancel) ──────
    temp_dir: str = os.getenv("TEMP_DIR", "/tmp/whisper-jobs")

    # ── Segurança ────────────────────────────────────────────────────────────
    # Token compartilhado — só o backend do MARVENDAS deve chamar este serviço.
    # NÃO é um token pago de IA, é autenticação interna serviço-a-serviço.
    service_token: str = os.getenv("WHISPER_SERVICE_TOKEN", "")

    # ── Retenção de jobs em memória (TTL de limpeza) ─────────────────────────
    job_ttl_seconds: int = _int("JOB_TTL_SECONDS", 60 * 30)  # 30 min

    allowed_url_schemes: tuple[str, ...] = field(default_factory=lambda: ("http", "https"))


settings = Settings()
