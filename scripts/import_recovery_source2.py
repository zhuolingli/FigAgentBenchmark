#!/usr/bin/env python3
"""Incrementally import rendered source-2 papers into FigAgentDataset layout."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def accepted_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "accepted":
                rows.append(row)
    return sorted(rows, key=lambda row: row["arxiv_id"])


def valid_source(
    source_root: Path,
    row: dict[str, Any],
    *,
    include_partial: bool = False,
) -> bool:
    paper_dir = source_root / row["paper_path"]
    status_path = paper_dir / "render_status.json"
    if not status_path.is_file():
        return False
    try:
        status = read_json(status_path)
        if status.get("status") != "complete" and not (
            include_partial and status.get("status") == "failed" and status.get("figures")
        ):
            return False
        if status.get("drawio_files") != row["drawio_files"]:
            return False
        if len(list((paper_dir / "drawio_raw").glob("*.drawio"))) != row["drawio_files"]:
            return False
        if status.get("png_images") != len(status["figures"]):
            return False
        for figure in status["figures"]:
            if not (paper_dir / figure["drawio_path"]).is_file():
                return False
            image = paper_dir / figure["image_path"]
            if not image.is_file() or image.stat().st_size <= 100:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_destination(paper_dir: Path) -> bool:
    marker = paper_dir / "import_status.json"
    metadata = paper_dir / "metadata.json"
    if not marker.is_file() or not metadata.is_file():
        return False
    try:
        status = read_json(marker)
        return (
            status.get("status") == "complete"
            and status.get("source") == "cloud_prefiltered_source_packages"
            and len(list((paper_dir / "drawio").glob("*.drawio"))) == status["drawio_files"]
            and len(list((paper_dir / "images").glob("*.png"))) == status["png_images"]
        )
    except Exception:
        return False


def import_paper(
    source_root: Path,
    output_root: Path,
    row: dict[str, Any],
    *,
    include_partial: bool = False,
) -> dict[str, Any]:
    arxiv_id = row["arxiv_id"]
    source_dir = source_root / row["paper_path"]
    destination = output_root / "papers" / arxiv_id
    if destination.exists():
        if valid_destination(destination):
            return {"status": "skipped", "arxiv_id": arxiv_id}
        return {
            "status": "collision",
            "arxiv_id": arxiv_id,
            "error": f"destination exists and is not this completed source: {destination}",
        }
    if not valid_source(source_root, row, include_partial=include_partial):
        return {"status": "not_ready", "arxiv_id": arxiv_id}

    render = read_json(source_dir / "render_status.json")
    source_record = read_json(source_dir / "record.json")
    staging = output_root / ".building" / f"{arxiv_id}_{uuid.uuid4().hex[:8]}"
    drawio_dir = staging / "drawio"
    image_dir = staging / "images"
    drawio_dir.mkdir(parents=True)
    image_dir.mkdir()
    try:
        # Preserve all source diagrams for partial papers; figures only maps
        # pages which Draw.io Desktop successfully exported.
        drawio_names = sorted(path.name for path in (source_dir / "drawio_raw").glob("*.drawio"))
        for name in drawio_names:
            shutil.copy2(source_dir / "drawio_raw" / name, drawio_dir / name)

        figures = []
        for figure in render["figures"]:
            drawio_name = Path(figure["drawio_path"]).name
            image_name = Path(figure["image_path"]).name
            shutil.copy2(source_dir / figure["image_path"], image_dir / image_name)
            figures.append({
                "drawio_path": f"drawio/{drawio_name}",
                "image_path": f"images/{image_name}",
                "page_number": figure["page_number"],
            })

        write_json(staging / "metadata.json", {
            "paper": {
                "arxiv_id": arxiv_id,
                "title": source_record.get("title"),
            },
            "figures": figures,
        })
        write_json(staging / "text_status.json", {"arxiv_id": arxiv_id, "status": "pending"})
        shutil.copy2(source_dir / "record.json", staging / "source_record.json")
        shutil.copy2(source_dir / "render_status.json", staging / "source_render_status.json")
        write_json(staging / "import_status.json", {
            "status": "complete",
            "source": "cloud_prefiltered_source_packages",
            "render_completeness": render["status"],
            "drawio_files": len(drawio_names),
            "png_images": len(figures),
            "render_errors": len(render.get("errors", [])),
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, destination)
        return {
            "status": "imported",
            "arxiv_id": arxiv_id,
            "drawio_files": len(drawio_names),
            "png_images": len(figures),
        }
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "error": f"{type(error).__name__}: {error}",
        }


def consolidate(output: Path) -> dict[str, Any]:
    manifest = []
    sources: Counter[str] = Counter()
    drawio_files = png_images = incomplete = partial_render_papers = 0
    for paper_dir in sorted((output / "papers").iterdir()):
        marker = paper_dir / "import_status.json"
        metadata = paper_dir / "metadata.json"
        if not marker.is_file() or not metadata.is_file():
            incomplete += 1
            continue
        try:
            status = read_json(marker)
            if status.get("status") != "complete":
                incomplete += 1
                continue
            if status.get("render_completeness") == "failed":
                partial_render_papers += 1
            metadata_value = read_json(metadata)
            arxiv_id = metadata_value["paper"]["arxiv_id"]
            if arxiv_id != paper_dir.name:
                incomplete += 1
                continue
            source = status.get("source", "unknown")
            sources[source] += 1
            drawio_files += int(status["drawio_files"])
            png_images += int(status["png_images"])
            manifest.append({
                "arxiv_id": arxiv_id,
                "folder": f"papers/{arxiv_id}",
                "metadata_path": f"papers/{arxiv_id}/metadata.json",
                "paper_markdown_path": f"papers/{arxiv_id}/paper.md",
                "text_status_path": f"papers/{arxiv_id}/text_status.json",
            })
        except Exception:
            incomplete += 1
    manifest.sort(key=lambda row: row["arxiv_id"])
    write_jsonl(output / "manifest.jsonl", manifest)
    summary = {
        "papers": len(manifest),
        "drawio_files": drawio_files,
        "png_images": png_images,
        "sources": dict(sorted(sources.items())),
        "incomplete_paper_dirs": incomplete,
        "partially_rendered_papers": partial_render_papers,
        "source_data_modified": False,
    }
    write_json(output / "import_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-papers", type=int)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Import incomplete render statuses that still contain valid PNG figures.",
    )
    args = parser.parse_args()
    manifest = args.manifest or args.source / "recovery_manifest.jsonl"
    rows = accepted_rows(manifest)
    ready = [
        row for row in rows
        if valid_source(args.source, row, include_partial=args.include_partial)
    ]
    if args.limit_papers is not None:
        ready = ready[: args.limit_papers]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".building").mkdir(exist_ok=True)
    log(f"accepted={len(rows)} ready={len(ready)} workers={args.workers}")

    counts: Counter[str] = Counter()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(
                import_paper,
                args.source,
                args.output,
                row,
                include_partial=args.include_partial,
            )
            for row in ready
        ]
        for completed, job in enumerate(futures.as_completed(jobs), start=1):
            result = job.result()
            counts[result["status"]] += 1
            if result["status"] in {"failed", "collision"}:
                log(json.dumps(result, ensure_ascii=False, sort_keys=True))
            if completed % 500 == 0 or completed == len(jobs):
                log(f"completed={completed}/{len(jobs)} counts={dict(counts)}")

    summary = consolidate(args.output)
    summary["run_counts"] = dict(counts)
    write_json(args.output / "import_run_summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 2 if counts["failed"] or counts["collision"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
