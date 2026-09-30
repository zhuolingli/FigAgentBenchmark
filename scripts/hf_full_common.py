#!/usr/bin/env python3
"""Shared helpers for the full Draw.io Hugging Face dataset build."""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote


DRAWIO = os.environ.get("DRAWIO_BIN", "drawio")
MARKERS = (b"mxfile", b"mxGraphModel", b"mxgraph", b"diagrams.net", b"/drawio")
CANDIDATE_EXTENSIONS = {".svg", ".png", ".pdf"}
MX_RE = re.compile(rb"<mxfile\b|<mxGraphModel\b|<diagram\b")
VERTEX_RE = re.compile(r'vertex="1"')
EDGE_RE = re.compile(r'edge="1"')
IMAGE_RE = re.compile(r"shape=image|image=data:|image=http|data:image", re.I)
VALUE_RE = re.compile(r'value="([^"]+)"')
TAG_RE = re.compile(r"<[^>]+>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str, limit: int = 100) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (normalized or "diagram")[:limit]


def guess_single_file_extension(data: bytes) -> str:
    head = data[:4096].lstrip().lower()
    if head.startswith(b"%pdf"):
        return ".pdf"
    if b"\\documentclass" in head or b"\\begin{document}" in head:
        return ".tex"
    return ".src"


def unpack_arxiv_gz(source: Path, destination: Path) -> None:
    """Safely unpack one per-paper arXiv .gz package."""
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(source, "r:gz") as archive:
            archive.extractall(destination, filter="data")
            return
    except (tarfile.ReadError, tarfile.CompressionError):
        pass
    with gzip.open(source, "rb") as handle:
        data = handle.read()
    (destination / f"source{guess_single_file_extension(data)}").write_bytes(data)


def has_drawio_marker(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            overlap = b""
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    return False
                data = overlap + chunk
                if any(marker in data for marker in MARKERS):
                    return True
                overlap = data[-32:]
    except OSError:
        return False


def is_drawio_xml(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if not MX_RE.search(data):
        return False
    try:
        ET.fromstring(data.decode("utf-8", errors="replace"))
    except ET.ParseError:
        return False
    return True


def terminate_process_group(process_group: int) -> None:
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return


def run_process_group(
    command: list[str],
    timeout: float,
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    # Do not use PIPE + communicate() here. Electron/DBus may launch a helper
    # which inherits the pipe's write end. The draw.io CLI and wrapper can then
    # exit successfully while communicate() waits for an EOF that never comes,
    # turning a completed export into a full-length timeout. A regular temporary
    # file preserves diagnostics without tying process completion to pipe EOF;
    # this is equivalent to the historical renderer's proven DEVNULL behavior.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as output_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process.pid)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        finally:
            terminate_process_group(process.pid)
        output_file.seek(0)
        output = output_file.read()[-4000:]
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "output": output,
    }


def export_recursive(
    items: list[tuple[str, Path]],
    output_dir: Path,
    run_root: Path,
    output_format: str,
    *,
    scale: float | None = None,
    chunk_size: int = 60,
    retry_missing: bool = False,
    timeout_base: int = 180,
    timeout_per_file: int = 5,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Batch-export files through draw.io and return successes and failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    successes: dict[str, Path] = {}
    failures: dict[str, dict[str, Any]] = {}

    def output_path(name: str) -> Path:
        return output_dir / f"{name}.{output_format}"

    def run_chunk(chunk: list[tuple[str, Path]], depth: int = 0) -> None:
        pending = [(name, path) for name, path in chunk if not output_path(name).is_file()]
        if not pending:
            return
        batch_root = run_root / f"batch_{uuid.uuid4().hex[:12]}"
        input_dir = batch_root / "in"
        input_dir.mkdir(parents=True)
        try:
            for name, source in pending:
                suffix = source.suffix or ".drawio"
                (input_dir / f"{name}{suffix}").symlink_to(source)
            command = [
                DRAWIO,
                "--no-sandbox",
                "--export",
                "--recursive",
                "--format",
                output_format,
                "--border",
                "0",
                "--output",
                str(output_dir),
            ]
            if scale is not None:
                command += ["--scale", str(scale)]
            command.append(str(input_dir))
            result = run_process_group(
                command,
                timeout=timeout_base + timeout_per_file * len(pending),
                # Electron performs substantial startup-time work relative to
                # its current directory. The repository lives on NTFS/FUSE,
                # where that can turn a ~2 s cold start into several minutes.
                # Keep the child in this ext4 scratch directory instead.
                cwd=batch_root,
            )
        finally:
            shutil.rmtree(batch_root, ignore_errors=True)

        missing = []
        for name, source in pending:
            produced = output_path(name)
            if produced.is_file() and produced.stat().st_size > 100:
                successes[name] = produced
            else:
                missing.append((name, source))
        # Recovery follows the original process_tar.py behavior: one batch
        # attempt, then record failures and continue. Rendering callers can opt
        # into recursive isolation explicitly with retry_missing=True.
        should_retry = retry_missing and bool(missing) and len(pending) > 1
        if should_retry and depth < 8:
            midpoint = max(1, len(missing) // 2)
            run_chunk(missing[:midpoint], depth + 1)
            run_chunk(missing[midpoint:], depth + 1)
            missing = [(name, source) for name, source in missing if name not in successes]
        for name, _source in missing:
            failures[name] = {
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
                "output": result["output"],
            }

    for start in range(0, len(items), chunk_size):
        run_chunk(items[start : start + chunk_size])
    return successes, failures


def inflate_diagram(body: str) -> str:
    try:
        raw = base64.b64decode(body.strip())
        return unquote(zlib.decompress(raw, -15).decode("utf-8", "replace"))
    except Exception:
        return ""


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def diagram_children(root: ET.Element) -> list[ET.Element]:
    return [child for child in list(root) if local_name(child) == "diagram"]


def page_body(page: ET.Element) -> str:
    children = list(page)
    if children:
        return "".join(ET.tostring(child, encoding="unicode") for child in children)
    body = page.text or ""
    if body.strip() and "<mxGraphModel" not in body:
        body = inflate_diagram(body) or body
    return body


def page_metrics(body: str) -> dict[str, int]:
    vertices = len(VERTEX_RE.findall(body))
    edges = len(EDGE_RE.findall(body))
    images = len(IMAGE_RE.findall(body))
    texts = sum(bool(TAG_RE.sub("", value).strip()) for value in VALUE_RE.findall(body))
    return {
        "vertices": vertices,
        "edges": edges,
        "elements": vertices + edges,
        "images": images,
        "texts": texts,
    }


def estimate_page_size(body: str) -> tuple[float, float]:
    """Estimate cropped diagram dimensions from mxGeometry coordinates."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return 1000.0, 800.0
    coordinates: list[tuple[float, float, float, float]] = []
    for element in root.iter():
        if local_name(element) != "mxGeometry":
            continue
        try:
            x = float(element.attrib.get("x", 0) or 0)
            y = float(element.attrib.get("y", 0) or 0)
            width = float(element.attrib.get("width", 0) or 0)
            height = float(element.attrib.get("height", 0) or 0)
        except ValueError:
            continue
        if all(math.isfinite(value) for value in (x, y, width, height)):
            coordinates.append((x, y, max(width, 0), max(height, 0)))
    if not coordinates:
        try:
            return (
                float(root.attrib.get("pageWidth", 1000) or 1000),
                float(root.attrib.get("pageHeight", 800) or 800),
            )
        except ValueError:
            return 1000.0, 800.0
    minimum_x = min(row[0] for row in coordinates)
    minimum_y = min(row[1] for row in coordinates)
    maximum_x = max(row[0] + row[2] for row in coordinates)
    maximum_y = max(row[1] + row[3] for row in coordinates)
    return max(1.0, maximum_x - minimum_x + 20), max(1.0, maximum_y - minimum_y + 20)


def adaptive_scale(width: float, height: float) -> float:
    """Prefer 2x, while avoiding enlargement beyond 4096px or 20MP."""
    longest = max(width, height, 1.0)
    pixels = max(width * height, 1.0)
    desired = min(2.0, 4096.0 / longest, math.sqrt(20_000_000.0 / pixels))
    desired = max(1.0, desired)
    for bucket in (2.0, 1.5, 1.25, 1.0):
        if bucket <= desired + 1e-9:
            return bucket
    return 1.0


def page_specs(source: Path) -> tuple[ET.Element, list[dict[str, Any]]]:
    """Parse a Draw.io file into non-empty page specifications."""
    root = ET.parse(source).getroot()
    pages = diagram_children(root)
    if not pages and local_name(root) == "mxGraphModel":
        body = ET.tostring(root, encoding="unicode")
        metrics = page_metrics(body)
        width, height = estimate_page_size(body)
        return root, [{
            "page_number": 1,
            "page_name": "Page-1",
            "body": body,
            "metrics": metrics,
            "estimated_width": width,
            "estimated_height": height,
            "scale": adaptive_scale(width, height),
        }]
    specs = []
    for index, page in enumerate(pages, start=1):
        body = page_body(page)
        metrics = page_metrics(body)
        if metrics["elements"] == 0:
            continue
        width, height = estimate_page_size(body)
        specs.append({
            "page_number": index,
            "page_name": page.attrib.get("name") or f"Page-{index}",
            "body": body,
            "metrics": metrics,
            "estimated_width": width,
            "estimated_height": height,
            "scale": adaptive_scale(width, height),
        })
    return root, specs


def write_single_page(source: Path, destination: Path, page_number: int) -> None:
    root = ET.parse(source).getroot()
    pages = diagram_children(root)
    if not pages and local_name(root) == "mxGraphModel":
        shutil.copy2(source, destination)
        return
    if page_number < 1 or page_number > len(pages):
        raise IndexError(f"page {page_number} outside 1..{len(pages)}: {source}")
    one_page = ET.Element(root.tag, root.attrib)
    one_page.text = root.text
    one_page.append(copy.deepcopy(pages[page_number - 1]))
    ET.ElementTree(one_page).write(destination, encoding="utf-8", xml_declaration=True)


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
