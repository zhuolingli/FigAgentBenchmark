#!/usr/bin/env python3
"""Render recovered Draw.io files to high-resolution PNGs with checkpoints.

Each source file is parsed into non-empty pages, each page is written as a
single-page Draw.io document in ext4 scratch, then pages are recursively
exported in batches through the canonical Draw.io Desktop wrapper.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from hf_full_common import (
    atomic_write_json,
    atomic_write_jsonl,
    export_recursive,
    page_specs,
    png_dimensions,
    safe_name,
    write_single_page,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def accepted_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "accepted":
                rows.append(row)
    rows.sort(key=lambda row: row["arxiv_id"])
    return rows


def valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 100:
        return False
    width, height = png_dimensions(path)
    return width > 0 and height > 0


def valid_complete(root: Path, row: dict[str, Any]) -> bool:
    paper_dir = root / row["paper_path"]
    status_path = paper_dir / "render_status.json"
    if not status_path.is_file():
        return False
    try:
        status = read_json(status_path)
        if status.get("status") != "complete":
            return False
        figures = status["figures"]
        if status.get("drawio_files") != row["drawio_files"]:
            return False
        if status.get("png_images") != len(figures):
            return False
        return all(valid_png(paper_dir / figure["image_path"]) for figure in figures)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def make_batches(rows: list[dict[str, Any]], drawios_per_batch: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_drawios = 0
    for row in rows:
        count = max(1, int(row.get("drawio_files", 1)))
        if current and current_drawios + count > drawios_per_batch:
            batches.append(current)
            current = []
            current_drawios = 0
        current.append(row)
        current_drawios += count
    if current:
        batches.append(current)
    return batches


def task_key(arxiv_id: str, drawio_name: str, page_number: int) -> str:
    value = f"{arxiv_id}/{drawio_name}/{page_number}".encode()
    digest = hashlib.sha1(value).hexdigest()[:12]
    return f"{safe_name(arxiv_id, 20)}_{digest}"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def prepare_tasks(
    root: Path,
    rows: list[dict[str, Any]],
    batch_root: Path,
    force_scale: float | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    paper_tasks: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    single_root = batch_root / "single_pages"
    single_root.mkdir(parents=True)
    for row in rows:
        arxiv_id = row["arxiv_id"]
        paper_dir = root / row["paper_path"]
        drawio_files = sorted((paper_dir / "drawio_raw").glob("*.drawio"))
        minimum_reuse_scale = force_scale if force_scale is not None else 2.0
        existing_figures: dict[tuple[str, int], float] = {}
        status_path = paper_dir / "render_status.json"
        if status_path.is_file():
            try:
                previous = read_json(status_path)
                for figure in previous.get("figures", []):
                    image_path = paper_dir / figure["image_path"]
                    scale = float(figure.get("scale", 0))
                    if scale >= minimum_reuse_scale and valid_png(image_path):
                        key = (figure["drawio_path"], int(figure["page_number"]))
                        existing_figures[key] = scale
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                existing_figures.clear()
        tasks: list[dict[str, Any]] = []
        if len(drawio_files) != row["drawio_files"]:
            errors.append({
                "arxiv_id": arxiv_id,
                "error": f"drawio count mismatch: manifest={row['drawio_files']} actual={len(drawio_files)}",
            })
            paper_tasks[arxiv_id] = tasks
            continue
        for source in drawio_files:
            try:
                _root, specs = page_specs(source)
                if not specs:
                    raise RuntimeError("no non-empty pages")
                for spec in specs:
                    page_number = int(spec["page_number"])
                    key = task_key(arxiv_id, source.name, page_number)
                    image_name = f"{source.stem}_p{page_number:03d}.png"
                    drawio_path = f"drawio_raw/{source.name}"
                    figure_key = (drawio_path, page_number)
                    existing_scale = existing_figures.get(figure_key)
                    image_path = paper_dir / "images" / image_name
                    # Recover valid orphan PNGs left by a timed-out batch.
                    if existing_scale is None and minimum_reuse_scale <= 1.0 and valid_png(image_path):
                        existing_scale = 1.0
                    existing = existing_scale is not None
                    single = single_root / f"{key}.drawio"
                    if not existing:
                        write_single_page(source, single, page_number)
                    tasks.append({
                        "key": key,
                        "source": source,
                        "single": single,
                        "existing": existing,
                        "drawio_path": drawio_path,
                        "image_path": f"images/{image_name}",
                        "page_number": page_number,
                        "page_name": spec["page_name"],
                        "scale": existing_scale if existing else (force_scale if force_scale is not None else float(spec["scale"])),
                        "metrics": spec["metrics"],
                        "estimated_width": spec["estimated_width"],
                        "estimated_height": spec["estimated_height"],
                    })
            except Exception as error:
                errors.append({
                    "arxiv_id": arxiv_id,
                    "drawio": source.name,
                    "error": f"{type(error).__name__}: {error}",
                })
        paper_tasks[arxiv_id] = tasks
    return paper_tasks, errors


def process_batch(
    root: Path,
    rows: list[dict[str, Any]],
    scratch: Path,
    chunk_size: int,
    force_scale: float | None,
) -> dict[str, Any]:
    batch_id = uuid.uuid4().hex[:12]
    batch_root = scratch / f"batch_{batch_id}"
    batch_root.mkdir(parents=True)
    started = time.time()
    try:
        paper_tasks, preparation_errors = prepare_tasks(root, rows, batch_root, force_scale)
        errors_by_paper: dict[str, list[dict[str, Any]]] = {row["arxiv_id"]: [] for row in rows}
        for error in preparation_errors:
            errors_by_paper[error["arxiv_id"]].append(error)

        outputs: dict[str, Path] = {}
        failures: dict[str, dict[str, Any]] = {}
        by_scale: dict[float, list[dict[str, Any]]] = {}
        for tasks in paper_tasks.values():
            for task in tasks:
                if not task["existing"]:
                    by_scale.setdefault(task["scale"], []).append(task)
        for scale, tasks in sorted(by_scale.items(), reverse=True):
            scale_root = batch_root / f"scale_{str(scale).replace('.', '_')}"
            successes, scale_failures = export_recursive(
                [(task["key"], task["single"]) for task in tasks],
                scale_root / "out",
                scale_root / "runs",
                "png",
                scale=scale,
                chunk_size=chunk_size,
                retry_missing=False,
                timeout_base=180,
                timeout_per_file=6,
            )
            outputs.update(successes)
            failures.update(scale_failures)

        complete = failed = png_images = 0
        for row in rows:
            arxiv_id = row["arxiv_id"]
            paper_dir = root / row["paper_path"]
            tasks = paper_tasks[arxiv_id]
            paper_errors = errors_by_paper[arxiv_id]
            figures = []
            for task in tasks:
                destination = paper_dir / task["image_path"]
                produced = destination if task["existing"] else outputs.get(task["key"])
                if produced is None or not valid_png(produced):
                    detail = failures.get(task["key"], {"error": "missing or invalid PNG"})
                    paper_errors.append({
                        "drawio_path": task["drawio_path"],
                        "page_number": task["page_number"],
                        "error": detail,
                    })
                    continue
                if not task["existing"]:
                    atomic_copy(produced, destination)
                width, height = png_dimensions(destination)
                figures.append({
                    "drawio_path": task["drawio_path"],
                    "image_path": task["image_path"],
                    "page_number": task["page_number"],
                    "page_name": task["page_name"],
                    "scale": task["scale"],
                    "width": width,
                    "height": height,
                    **task["metrics"],
                })
            figures.sort(key=lambda value: (value["drawio_path"], value["page_number"]))
            is_complete = not paper_errors and len(figures) == len(tasks) and bool(figures)
            status = {
                "arxiv_id": arxiv_id,
                "status": "complete" if is_complete else "failed",
                "drawio_files": row["drawio_files"],
                "png_images": len(figures),
                "figures": figures,
                "errors": paper_errors,
                "render_pipeline": "draw.io Desktop recursive export of split single-page XML",
                "resolution_policy": "adaptive scale up to 2x, 4096px longest side and 20MP estimate guard",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write_json(paper_dir / "render_status.json", status)
            png_images += len(figures)
            if is_complete:
                complete += 1
            else:
                failed += 1
        return {
            "papers": len(rows),
            "complete": complete,
            "failed": failed,
            "png_images": png_images,
            "seconds": round(time.time() - started, 3),
        }
    except Exception as error:
        return {
            "papers": len(rows),
            "complete": 0,
            "failed": len(rows),
            "png_images": 0,
            "seconds": round(time.time() - started, 3),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        shutil.rmtree(batch_root, ignore_errors=True)


def consolidate(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = []
    drawio_files = png_images = failed = 0
    for row in rows:
        paper_dir = root / row["paper_path"]
        if not valid_complete(root, row):
            failed += 1
            continue
        status = read_json(paper_dir / "render_status.json")
        manifest.append({
            "arxiv_id": row["arxiv_id"],
            "folder": row["paper_path"],
            "render_status_path": f"{row['paper_path']}/render_status.json",
            "drawio_files": status["drawio_files"],
            "png_images": status["png_images"],
        })
        drawio_files += status["drawio_files"]
        png_images += status["png_images"]
    atomic_write_jsonl(root / "render_manifest.jsonl", manifest)
    summary = {
        "input_accepted_papers": len(rows),
        "complete_papers": len(manifest),
        "failed_papers": failed,
        "drawio_files": drawio_files,
        "png_images": png_images,
        "complete": len(manifest) == len(rows),
        "version": 1,
    }
    atomic_write_json(root / "render_summary.json", summary)
    return summary


def rows_for_consolidation(root: Path, selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never let a retry subset replace the canonical complete manifest."""
    canonical = root / "recovery_manifest.jsonl"
    return accepted_rows(canonical) if canonical.is_file() else selected_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-drawios", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--limit-papers", type=int)
    parser.add_argument("--force-scale", type=float)
    args = parser.parse_args()
    if min(args.workers, args.batch_drawios, args.chunk_size) < 1:
        parser.error("workers and batch sizes must be positive")
    args.scratch.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.root / "recovery_manifest.jsonl"
    rows = accepted_rows(manifest)
    pending = [row for row in rows if not valid_complete(args.root, row)]
    if args.limit_papers is not None:
        pending = pending[: args.limit_papers]
    batches = make_batches(pending, args.batch_drawios)
    log(
        f"accepted={len(rows)} pending={len(pending)} batches={len(batches)} "
        f"workers={args.workers} batch_drawios={args.batch_drawios}"
    )
    counts: Counter[str] = Counter()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(
                process_batch,
                args.root,
                batch,
                args.scratch,
                args.chunk_size,
                args.force_scale,
            )
            for batch in batches
        ]
        for completed, job in enumerate(futures.as_completed(jobs), start=1):
            result = job.result()
            counts["papers"] += result["papers"]
            counts["complete"] += result["complete"]
            counts["failed"] += result["failed"]
            counts["png_images"] += result["png_images"]
            if result.get("error"):
                log(f"batch_error={result['error']}")
            log(
                f"batches={completed}/{len(batches)} processed={counts['papers']}/{len(pending)} "
                f"complete={counts['complete']} failed={counts['failed']} png={counts['png_images']} "
                f"last_seconds={result['seconds']}"
            )
    summary = consolidate(args.root, rows_for_consolidation(args.root, rows))
    summary["this_run"] = dict(counts)
    atomic_write_json(args.root / "render_run_summary.json", summary)
    log(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed_papers"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
