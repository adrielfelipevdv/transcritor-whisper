"""
Wrapper do faster-whisper. O modelo é carregado UMA VEZ (singleton, no
lifespan do FastAPI) e reaproveitado entre requisições — nunca
load/transcribe/unload por chamada (pedido explícito, seção 33).

VAD (voz vs. silêncio) usa o filtro embutido do faster-whisper, que por baixo
dos panos usa o Silero VAD — não precisamos integrar uma segunda dependência
separada para isso.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import settings
from .schemas import TranscriptSegment, TranscriptionResult, WordTimestamp

logger = logging.getLogger("whisper-transcription")

_model = None  # singleton — ver load_model()
_resolved_device = "cpu"
_resolved_compute_type = "int8"


def _detect_device() -> str:
    if settings.whisper_device != "auto":
        return settings.whisper_device
    try:
        import ctranslate2  # faster-whisper depende de ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # noqa: BLE001 — nunca falhar só por não ter CUDA
        pass
    return "cpu"


def _detect_compute_type(device: str) -> str:
    if settings.whisper_compute_type != "auto":
        return settings.whisper_compute_type
    # int8 é o melhor custo/benefício em CPU; float16 aproveita tensor cores em GPU.
    return "float16" if device == "cuda" else "int8"


def load_model():
    """Carrega o modelo uma vez. Chamado no startup do FastAPI (lifespan)."""
    global _model, _resolved_device, _resolved_compute_type
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel  # import tardio: só quando for usar de fato

    _resolved_device = _detect_device()
    _resolved_compute_type = _detect_compute_type(_resolved_device)

    logger.info(
        "Carregando modelo Whisper %r (device=%s, compute_type=%s)",
        settings.whisper_model, _resolved_device, _resolved_compute_type,
    )
    kwargs = dict(
        device=_resolved_device,
        compute_type=_resolved_compute_type,
        download_root=settings.model_cache_dir,
    )
    if _resolved_device == "cpu":
        kwargs["cpu_threads"] = settings.whisper_cpu_threads
    try:
        _model = WhisperModel(settings.whisper_model, **kwargs)
    except Exception:
        # Nunca falhar o boot só porque CUDA não está disponível/configurado
        # corretamente — cai para CPU automaticamente (pedido explícito).
        if _resolved_device != "cpu":
            logger.warning("Falha ao inicializar em %s, caindo para CPU.", _resolved_device, exc_info=True)
            _resolved_device = "cpu"
            _resolved_compute_type = _detect_compute_type("cpu")
            _model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type=_resolved_compute_type,
                cpu_threads=settings.whisper_cpu_threads,
                download_root=settings.model_cache_dir,
            )
        else:
            raise
    logger.info("Modelo carregado (device=%s, compute_type=%s).", _resolved_device, _resolved_compute_type)
    return _model


def model_ready() -> bool:
    return _model is not None


@dataclass
class TranscribeProgress:
    phase: str
    progress: int


def transcribe_wav(
    wav_path: str,
    on_progress: "callable[[TranscribeProgress], None] | None" = None,
) -> TranscriptionResult:
    """Transcreve um WAV mono/16kHz já normalizado. Bloqueante — deve ser
    chamado via run_in_executor/thread pelo chamador assíncrono."""
    model = load_model()

    if on_progress:
        on_progress(TranscribeProgress(phase="Detectando falas…", progress=35))

    segments_gen, info = model.transcribe(
        wav_path,
        language=settings.whisper_language,
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        vad_parameters=dict(min_silence_duration_ms=settings.whisper_vad_min_silence_ms),
        word_timestamps=settings.whisper_word_timestamps,
    )

    if on_progress:
        on_progress(TranscribeProgress(phase="Transcrevendo…", progress=55))

    segments: list[TranscriptSegment] = []
    duration = info.duration or 0.0
    for seg in segments_gen:
        words = None
        if settings.whisper_word_timestamps and seg.words:
            words = [WordTimestamp(word=w.word.strip(), start=w.start, end=w.end) for w in seg.words]
        segments.append(
            TranscriptSegment(start=round(seg.start, 2), end=round(seg.end, 2), text=seg.text.strip(), words=words)
        )
        if on_progress and duration > 0:
            pct = 55 + min(40, int((seg.end / duration) * 40))
            on_progress(TranscribeProgress(phase="Transcrevendo…", progress=pct))

    if on_progress:
        on_progress(TranscribeProgress(phase="Processando blocos…", progress=97))

    full_text = " ".join(s.text for s in segments if s.text).strip()

    return TranscriptionResult(
        language=info.language or settings.whisper_language or "pt",
        duration=round(duration, 2),
        model=settings.whisper_model,
        device=_resolved_device,
        segments=segments,
        text=full_text,
    )
