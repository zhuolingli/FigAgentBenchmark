#!/usr/bin/env python3
"""Join model veto scores, rank candidates, and build a non-destructive gallery."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(os.environ.get(
    "DRAWIO_QUALITY_ROOT",
    "work",
))
OUT = ROOT / "QUALITY_SELECTION_V2"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def slug(value: str, limit: int = 100) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "diagram")[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=OUT / "coarse_candidates.jsonl")
    parser.add_argument("--scores", type=Path, default=OUT / "model_scores.jsonl")
    parser.add_argument("--eligible", type=Path, default=OUT / "model_eligible.jsonl")
    parser.add_argument("--model-rejects", type=Path, default=OUT / "model_rejects.jsonl")
    parser.add_argument("--selected", type=Path, default=OUT / "selected.jsonl")
    parser.add_argument("--gallery", type=Path, default=OUT / "gallery")
    parser.add_argument("--summary", type=Path, default=OUT / "selection_summary.json")
    parser.add_argument("--top-n", type=int, default=5000)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--ranking", choices=("structure", "siglip"), default="structure")
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    scores = {row["png"]: row for row in read_jsonl(args.scores)}
    missing = [row["png"] for row in candidates if row["png"] not in scores]
    if missing:
        raise SystemExit(f"missing model scores for {len(missing)} candidates; first={missing[0]}")

    eligible = []
    rejected = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(scores[row["png"]])
        keep = row["model_keep"]
        if args.margin_threshold is not None:
            keep = row["model_margin"] is not None and row["model_margin"] >= args.margin_threshold
        row["selection_keep"] = keep
        if not keep:
            row["selection_reject_reason"] = (
                "siglip_margin_below_threshold" if args.margin_threshold is not None else "model_veto"
            )
        (eligible if keep else rejected).append(row)
    if args.ranking == "siglip":
        eligible.sort(
            key=lambda row: (
                row["model_margin"], row["structure_score"], row["elements"], row["png"]
            ),
            reverse=True,
        )
    else:
        eligible.sort(
            key=lambda row: (row["structure_score"], row["elements"], row["texts"], row["png"]),
            reverse=True,
        )
    selected = eligible if args.top_n <= 0 else eligible[:args.top_n]
    for rank, row in enumerate(selected, 1):
        row["final_rank"] = rank

    gallery_tmp = args.gallery.with_name(args.gallery.name + ".tmp")
    if gallery_tmp.exists():
        shutil.rmtree(gallery_tmp)
    gallery_tmp.mkdir(parents=True)
    for row in selected:
        stem = (
            f"{row['final_rank']:05d}_{slug(row['arxiv_id'], 24)}_"
            f"{slug(row['diagram'])}_p{int(row['page_number']):02d}"
        )
        png_link = gallery_tmp / f"{stem}.png"
        drawio_link = gallery_tmp / f"{stem}.drawio"
        png_link.symlink_to(Path(row["png"]))
        drawio_link.symlink_to(Path(row["source_drawio"]))
        row["gallery_png"] = str(args.gallery / png_link.name)
        row["gallery_drawio"] = str(args.gallery / drawio_link.name)
    if args.gallery.exists():
        shutil.rmtree(args.gallery)
    gallery_tmp.replace(args.gallery)

    write_jsonl(args.eligible, eligible)
    write_jsonl(args.model_rejects, rejected)
    write_jsonl(args.selected, selected)
    summary = {
        "coarse_candidates": len(candidates),
        "model_kept": len(eligible),
        "model_vetoed": len(rejected),
        "selected": len(selected),
        "requested_top_n": args.top_n,
        "margin_threshold": args.margin_threshold,
        "ranking": args.ranking,
        "gallery": str(args.gallery),
        "source_data_modified": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
