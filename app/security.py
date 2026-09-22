"""
Download seguro de URL fornecida pelo usuário — proteção contra SSRF.

Regras:
- só http/https;
- resolve o host e recusa IPs privados/loopback/link-local/reservados
  (inclusive quando o host resolve para um desses via DNS — não confia só na
  string do hostname);
- bloqueia endpoints de metadata de cloud (169.254.169.254 já cai na faixa
  link-local, mas fica explícito);
- segue redirects manualmente, um a um, validando CADA hop (um 302 pode
  apontar para localhost mesmo que a URL original fosse "segura");
- limite de tamanho (stream, aborta assim que passar do teto — nunca baixa o
  arquivo inteiro antes de checar);
- timeout de conexão/leitura.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from .config import settings

MAX_REDIRECTS = 5


class UnsafeUrlError(Exception):
    pass


class DownloadTooLargeError(Exception):
    pass


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # não parseou -> trata como inseguro
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _assert_safe_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in settings.allowed_url_schemes:
        raise UnsafeUrlError(f"Protocolo não permitido: {parsed.scheme!r}")
    if not parsed.hostname:
        raise UnsafeUrlError("URL sem host")
    host = parsed.hostname.lower()
    if host in ("localhost",):
        raise UnsafeUrlError("Host bloqueado (localhost)")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"Não foi possível resolver o host: {host}") from e
    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            raise UnsafeUrlError(f"Host resolve para IP privado/reservado ({ip}) — bloqueado")


async def safe_download(url: str, dest_path: str) -> int:
    """Baixa `url` para `dest_path`, validando cada hop de redirect e
    abortando se ultrapassar `settings.max_file_size_bytes`. Retorna o
    tamanho em bytes baixado."""
    current_url = url
    total = 0
    async with httpx.AsyncClient(
        timeout=settings.download_timeout_seconds,
        follow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0 (MARVENDAS-transcription-service)"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _assert_safe_host(current_url)
            async with client.stream("GET", current_url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("Redirect sem Location")
                    current_url = httpx.URL(current_url).join(location).human_repr()
                    continue
                if resp.status_code >= 400:
                    raise UnsafeUrlError(f"Falha ao baixar (HTTP {resp.status_code})")

                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > settings.max_file_size_bytes:
                    raise DownloadTooLargeError(
                        f"Arquivo declara {content_length} bytes, acima do limite de "
                        f"{settings.max_file_size_bytes} bytes"
                    )

                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        total += len(chunk)
                        if total > settings.max_file_size_bytes:
                            raise DownloadTooLargeError(
                                f"Download ultrapassou o limite de {settings.max_file_size_bytes} bytes"
                            )
                        f.write(chunk)
                return total
    raise UnsafeUrlError("Excedeu o número máximo de redirects")
