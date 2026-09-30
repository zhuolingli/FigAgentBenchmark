#!/usr/bin/env python3
"""Build one scoring record per rendered FigAgentDataset image."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from quality_select_v2 import diagram_pages, image_features, page_metrics


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def public_relative(value: str, prefix: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
        raise ValueError(f"invalid {prefix} path: {value!r}")
    return path


def index_paper(item: tuple[Path, dict[str, Any]]) -> list[dict[str, Any]]:
    dataset, manifest_row = item
    arxiv_id = str(manifest_row["arxiv_id"])
    paper_dir = dataset / "papers" / arxiv_id
    metadata = read_json(paper_dir / "metadata.json")
    if metadata.get("paper", {}).get("arxiv_id") != arxiv_id:
        raise ValueError(f"metadata arXiv ID mismatch: {paper_dir}")
    figures = metadata.get("figures")
    if not isinstance(figures, list):
        raise ValueError(f"figures must be a list: {paper_dir}")

    page_cache: dict[str, tuple[list[tuple[str, str]], str]] = {}
    rows = []
    for figure_index, figure in enumerate(figures, 1):
        if not isinstance(figure, dict):
            raise ValueError(f"invalid figure #{figure_index}: {paper_dir}")
        drawio_relative = public_relative(str(figure["drawio_path"]), "drawio")
        image_relative = public_relative(str(figure["image_path"]), "images")
        page_number = int(figure["page_number"])
        drawio = paper_dir.joinpath(*drawio_relative.parts)
        image = paper_dir.joinpath(*image_relative.parts)
        if not drawio.is_file() or not image.is_file():
            raise FileNotFoundError(f"missing figure pair: {drawio} / {image}")

        cache_key = drawio_relative.as_posix()
        if cache_key not in page_cache:
            try:
                page_cache[cache_key] = (diagram_pages(drawio), "")
            except Exception as error:
                page_cache[cache_key] = ([], f"{type(error).__name__}: {error}"[:300])
        pages, source_error = page_cache[cache_key]
        if 1 <= page_number <= len(pages):
            name, body = pages[page_number - 1]
            metrics = page_metrics(name, body)
            mapping_error = ""
        else:
            metrics = page_metrics(f"Page-{page_number}", "")
            mapping_error = f"page index {page_number} outside source page count {len(pages)}"

        dataset_drawio = f"papers/{arxiv_id}/{drawio_relative.as_posix()}"
        dataset_image = f"papers/{arxiv_id}/{image_relative.as_posix()}"
        row = {
            "figure_id": f"{arxiv_id}:{image_relative.as_posix()}",
            "arxiv_id": arxiv_id,
            "figure_index": figure_index,
            "diagram": drawio.stem,
            "page_number": page_number,
            "drawio_path": dataset_drawio,
            "image_path": dataset_image,
            # Compatibility fields consumed by the established scoring scripts.
            "source_drawio": str(drawio),
            "png": str(image),
            "source_pages": len(pages),
            "source_error": source_error,
            "mapping_error": mapping_error,
            **metrics,
            **image_features(image),
        }
        row["valid"] = not (
            row["source_error"] or row["mapping_error"] or row["image_error"]
        )
        rows.append(row)
    return rows


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("manifest rows must be objects")
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-papers", type=int)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.rebuild:
        log(f"index exists, skipping: {args.output}")
        return 0

    dataset = args.dataset.resolve()
    manifest = read_manifest(dataset / "manifest.jsonl")
    if args.limit_papers is not None:
        manifest = manifest[: args.limit_papers]
    if not manifest:
        raise SystemExit("no manifest rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex[:8]}.tmp")
    images = 0
    items = [(dataset, row) for row in manifest]
    log(f"papers={len(items)} workers={args.workers}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                for done, rows in enumerate(executor.map(index_paper, items), 1):
                    handle.write("".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in rows
                    ))
                    images += len(rows)
                    if done % 250 == 0 or done == len(items):
                        handle.flush()
                        log(f"indexed={done}/{len(items)} images={images}")
        os.replace(temporary, args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    log(f"DONE papers={len(items)} images={images} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
