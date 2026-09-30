#!/usr/bin/env python3
"""Recover true Draw.io sources from the cloud-prefiltered per-paper packages.

The input .gz files and the legacy dataset are read-only. This stage does not
render PNGs; it only performs the expensive embedded-diagram recovery in large
batches and writes resumable per-paper state. Rendering is unified later so old
and new sources receive exactly the same high-resolution treatment.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from hf_full_common import (
    CANDIDATE_EXTENSIONS,
    atomic_write_json,
    atomic_write_jsonl,
    export_recursive,
    has_drawio_marker,
    is_drawio_xml,
    read_jsonl,
    safe_name,
    sha256,
    unpack_arxiv_gz,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def state_path(state_root: Path, row: dict[str, Any]) -> Path:
    return state_root / row["month"] / f"{row['arxiv_id']}.json"


def source_path(source_root: Path, row: dict[str, Any]) -> Path:
    return source_root / row["month"] / f"{row['arxiv_id']}.gz"


def usable_existing_state(
    output_root: Path,
    state_root: Path,
    row: dict[str, Any],
    retry_failed: bool,
) -> dict[str, Any] | None:
    path = state_path(state_root, row)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    status = state.get("status")
    if status == "accepted":
        paper = output_root / state.get("paper_path", "")
        if (paper / "record.json").is_file():
            return state
        return None
    if status == "rejected":
        return state
    if status == "failed" and not retry_failed:
        return state
    return None


def find_candidates(paper_root: Path, batch_prefix: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sequence = 0
    for path in paper_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        direct = suffix == ".drawio" and is_drawio_xml(path)
        embedded = suffix in CANDIDATE_EXTENSIONS and has_drawio_marker(path)
        if not (direct or embedded):
            continue
        sequence += 1
        candidates.append({
            "name": f"{batch_prefix}_{sequence:04d}",
            "source": path,
            "source_rel": path.relative_to(paper_root).as_posix(),
            "source_ext": suffix,
            "direct": direct,
        })
    return candidates


def write_state(state_root: Path, row: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "version": 1,
        "arxiv_id": row["arxiv_id"],
        "month": row["month"],
        "source_package": str(source_path(Path("."), row)),
        "cloud_manifest": row,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **value,
    }
    atomic_write_json(state_path(state_root, row), result)
    return result


def process_batch(
    batch_number: int,
    rows: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    state_root: Path,
    scratch_root: Path,
) -> list[dict[str, Any]]:
    batch_id = f"b{batch_number:06d}_{uuid.uuid4().hex[:8]}"
    work = scratch_root / batch_id
    unpacked_root = work / "unpacked"
    recovered_root = work / "recovered"
    run_root = work / "drawio_runs"
    for directory in (unpacked_root, recovered_root, run_root):
        directory.mkdir(parents=True, exist_ok=True)

    infos: dict[str, dict[str, Any]] = {}
    embedded_items: list[tuple[str, Path]] = []
    early_results: dict[str, dict[str, Any]] = {}
    try:
        for position, row in enumerate(rows, start=1):
            arxiv_id = row["arxiv_id"]
            package = source_path(source_root, row)
            paper_root = unpacked_root / safe_name(arxiv_id)
            if not package.is_file():
                early_results[arxiv_id] = write_state(state_root, row, {
                    "status": "failed",
                    "reason": "missing_source_package",
                    "error": str(package),
                })
                continue
            try:
                unpack_arxiv_gz(package, paper_root)
            except Exception as error:
                early_results[arxiv_id] = write_state(state_root, row, {
                    "status": "failed",
                    "reason": "unpack_failed",
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            prefix = f"p{batch_number:06d}_{position:04d}"
            candidates = find_candidates(paper_root, prefix)
            infos[arxiv_id] = {
                "row": row,
                "paper_root": paper_root,
                "candidates": candidates,
            }
            for candidate in candidates:
                if not candidate["direct"]:
                    embedded_items.append((candidate["name"], candidate["source"]))

        recovered_outputs: dict[str, Path] = {}
        recovery_failures: dict[str, dict[str, Any]] = {}
        if embedded_items:
            recovered_outputs, recovery_failures = export_recursive(
                embedded_items,
                recovered_root,
                run_root,
                "xml",
                chunk_size=60,
                retry_missing=False,
                timeout_base=300,
                timeout_per_file=0,
            )

        results = list(early_results.values())
        for arxiv_id, info in infos.items():
            row = info["row"]
            recovered: list[tuple[dict[str, Any], Path]] = []
            failed_candidates = []
            for candidate in info["candidates"]:
                if candidate["direct"]:
                    recovered.append((candidate, candidate["source"]))
                    continue
                exported = recovered_outputs.get(candidate["name"])
                if exported and is_drawio_xml(exported):
                    recovered.append((candidate, exported))
                elif candidate["name"] in recovery_failures:
                    failed_candidates.append({
                        "source_rel": candidate["source_rel"],
                        **recovery_failures[candidate["name"]],
                    })

            unique: list[tuple[dict[str, Any], Path, str]] = []
            hashes = set()
            for candidate, recovered_path in recovered:
                file_hash = sha256(recovered_path)
                if file_hash in hashes:
                    continue
                hashes.add(file_hash)
                unique.append((candidate, recovered_path, file_hash))

            if not unique:
                command_failed = any(
                    failure.get("timed_out") or failure.get("returncode") not in {0, None}
                    for failure in failed_candidates
                )
                status = "failed" if command_failed else "rejected"
                reason = "drawio_recovery_failed" if command_failed else "no_valid_drawio_recovered"
                results.append(write_state(state_root, row, {
                    "status": status,
                    "reason": reason,
                    "candidate_count": len(info["candidates"]),
                    "failed_candidates": failed_candidates,
                }))
                continue

            paper_dir = output_root / "papers" / arxiv_id
            if paper_dir.exists():
                if (paper_dir / "record.json").is_file():
                    results.append(write_state(state_root, row, {
                        "status": "accepted",
                        "reason": "existing_output",
                        "paper_path": f"papers/{arxiv_id}",
                        "drawio_files": len(list((paper_dir / "drawio_raw").glob("*.drawio"))),
                    }))
                    continue
                results.append(write_state(state_root, row, {
                    "status": "failed",
                    "reason": "unexpected_existing_output",
                    "error": str(paper_dir),
                }))
                continue

            staging = output_root / ".building" / f"{safe_name(arxiv_id)}_{uuid.uuid4().hex[:8]}"
            drawio_dir = staging / "drawio_raw"
            drawio_dir.mkdir(parents=True, exist_ok=False)
            diagrams = []
            try:
                for index, (candidate, recovered_path, file_hash) in enumerate(unique, start=1):
                    stem = safe_name(Path(candidate["source_rel"]).stem)
                    filename = f"{index:03d}_{stem}.drawio"
                    destination = drawio_dir / filename
                    shutil.copy2(recovered_path, destination)
                    diagrams.append({
                        "drawio_id": f"d{index:03d}",
                        "path": f"drawio_raw/{filename}",
                        "sha256": file_hash,
                        "source_rel": candidate["source_rel"],
                        "source_ext": candidate["source_ext"],
                        "recovery": "direct" if candidate["direct"] else "embedded_drawio_xml",
                    })
                atomic_write_json(staging / "record.json", {
                    "version": 1,
                    "arxiv_id": arxiv_id,
                    "month": row["month"],
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "source_dataset": "cloud_prefiltered_source_packages",
                    "source_package": str(source_path(source_root, row)),
                    "source_package_sha256": row.get("sha256"),
                    "source_tar": row.get("source_tar"),
                    "detection_reasons": row.get("reasons", []),
                    "diagrams": diagrams,
                })
                paper_dir.parent.mkdir(parents=True, exist_ok=True)
                os.rename(staging, paper_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
            results.append(write_state(state_root, row, {
                "status": "accepted",
                "reason": "valid_drawio_recovered",
                "paper_path": f"papers/{arxiv_id}",
                "candidate_count": len(info["candidates"]),
                "drawio_files": len(diagrams),
                "duplicate_drawio_removed": len(recovered) - len(unique),
                "failed_candidates": failed_candidates,
            }))
        return results
    except Exception:
        log(f"batch {batch_number} crashed:\n{traceback.format_exc()}")
        results = list(early_results.values())
        finished_ids = {result["arxiv_id"] for result in results}
        for row in rows:
            if row["arxiv_id"] in finished_ids:
                continue
            results.append(write_state(state_root, row, {
                "status": "failed",
                "reason": "batch_exception",
                "error": traceback.format_exc()[-4000:],
            }))
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


def consolidate_manifest(
    selected_rows: list[dict[str, Any]],
    state_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for source_row in selected_rows:
        path = state_path(state_root, source_row)
        if not path.is_file():
            rows.append({
                "arxiv_id": source_row["arxiv_id"],
                "month": source_row["month"],
                "status": "pending",
            })
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_write_jsonl(output_root / "recovery_manifest.jsonl", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("work/scratch/recovery"))
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.offset < 0:
        parser.error("batch-size/workers must be positive and offset non-negative")

    source_rows = read_jsonl(args.manifest)
    selected_rows = source_rows[args.offset :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    state_root = args.output / ".state"
    state_root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    pending = []
    existing = Counter()
    for row in selected_rows:
        state = usable_existing_state(args.output, state_root, row, args.retry_failed)
        if state is None:
            pending.append(row)
        else:
            existing[state.get("status", "unknown")] += 1
    batches = [pending[start : start + args.batch_size] for start in range(0, len(pending), args.batch_size)]
    log(
        f"selected={len(selected_rows)} pending={len(pending)} batches={len(batches)} "
        f"batch_size={args.batch_size} workers={args.workers} existing={dict(existing)}"
    )

    counts = Counter(existing)
    processed = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        submitted = {
            executor.submit(
                process_batch,
                batch_number,
                batch,
                args.source_root,
                args.output,
                state_root,
                args.scratch,
            ): batch_number
            for batch_number, batch in enumerate(batches, start=1)
        }
        for completed_batches, future in enumerate(futures.as_completed(submitted), start=1):
            results = future.result()
            processed += len(results)
            counts.update(result.get("status", "unknown") for result in results)
            atomic_write_json(args.output / "run_summary.json", {
                "version": 1,
                "selected": len(selected_rows),
                "pending_at_start": len(pending),
                "processed_this_run": processed,
                "completed_batches": completed_batches,
                "total_batches": len(batches),
                "counts": dict(counts),
            })
            log(
                f"batches={completed_batches}/{len(batches)} papers={processed}/{len(pending)} "
                f"counts={dict(counts)}"
            )

    manifest_rows = consolidate_manifest(selected_rows, state_root, args.output)
    final_counts = Counter(row.get("status", "unknown") for row in manifest_rows)
    summary = {
        "version": 1,
        "input_manifest": str(args.manifest.resolve()),
        "source_root": str(args.source_root.resolve()),
        "output": str(args.output.resolve()),
        "selected": len(selected_rows),
        "counts": dict(final_counts),
        "drawio_files": sum(int(row.get("drawio_files", 0)) for row in manifest_rows),
        "complete": not any(row.get("status") == "pending" for row in manifest_rows),
    }
    atomic_write_json(args.output / "recovery_summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not final_counts.get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
