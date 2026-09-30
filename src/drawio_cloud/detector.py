from __future__ import annotations

import gzip
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


DRAWIO_XML_RE = re.compile(rb"<mxfile\b|<mxGraphModel\b|<diagram\b")
DRAWIO_MARKERS = (b"mxfile", b"mxGraphModel", b"mxgraph", b"diagrams.net", b"/drawio")
EMBEDDED_EXTENSIONS = {".svg", ".png", ".pdf"}
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DetectionResult:
    matched: bool
    reasons: tuple[str, ...] = ()


def _stream_matches(stream: BinaryIO, patterns: Iterable[bytes | re.Pattern[bytes]]) -> bool:
    pats = tuple(patterns)
    overlap = 128
    tail = b""
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            return False
        data = tail + chunk
        for pattern in pats:
            if isinstance(pattern, bytes):
                if pattern in data:
                    return True
            elif pattern.search(data):
                return True
        tail = data[-overlap:]


def _inspect_tar(tf: tarfile.TarFile) -> DetectionResult:
    reasons: set[str] = set()
    for member in tf:
        if not member.isfile():
            continue
        suffix = Path(member.name).suffix.lower()
        if suffix not in EMBEDDED_EXTENSIONS and suffix != ".drawio":
            continue
        extracted = tf.extractfile(member)
        if extracted is None:
            continue
        with extracted:
            if suffix == ".drawio":
                if _stream_matches(extracted, (DRAWIO_XML_RE,)):
                    reasons.add("direct_drawio_xml")
            elif _stream_matches(extracted, DRAWIO_MARKERS):
                reasons.add(f"embedded_marker_{suffix[1:]}")
    return DetectionResult(bool(reasons), tuple(sorted(reasons)))


def inspect_source_package(path: str | Path) -> DetectionResult:
    """Inspect one original per-paper arXiv source package.

    Most inputs are gzip-compressed tar files. A minority are a single gzip
    payload; those are scanned conservatively for draw.io XML/metadata markers.
    The original package is never modified.
    """
    source = Path(path)
    try:
        with tarfile.open(source, mode="r:*") as tf:
            return _inspect_tar(tf)
    except (tarfile.ReadError, tarfile.CompressionError, EOFError):
        pass

    try:
        with gzip.open(source, "rb") as stream:
            if _stream_matches(stream, (DRAWIO_XML_RE, *DRAWIO_MARKERS)):
                return DetectionResult(True, ("single_payload_drawio_marker",))
    except (OSError, EOFError):
        return DetectionResult(False, ("invalid_source_package",))
    return DetectionResult(False, ())


def inspect_source_bytes(data: bytes) -> DetectionResult:
    """Test helper that inspects an in-memory source package."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            return _inspect_tar(tf)
    except (tarfile.ReadError, tarfile.CompressionError, EOFError):
        pass
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
            if _stream_matches(stream, (DRAWIO_XML_RE, *DRAWIO_MARKERS)):
                return DetectionResult(True, ("single_payload_drawio_marker",))
    except (OSError, EOFError):
        return DetectionResult(False, ("invalid_source_package",))
    return DetectionResult(False, ())
