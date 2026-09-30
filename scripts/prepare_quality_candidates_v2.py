#!/usr/bin/env python3
"""Create a robust, statistics-driven coarse candidate set after deduplication."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get(
    "DRAWIO_QUALITY_ROOT",
    "work",
))
OUT = ROOT / "QUALITY_SELECTION_V2"
DEFAULT_INPUT = OUT / "unique_pages.jsonl"
DEFAULT_CANDIDATES = OUT / "coarse_candidates.jsonl"
DEFAULT_REJECTS = OUT / "structural_rejects.jsonl"
DEFAULT_THRESHOLDS = OUT / "thresholds.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def percentile(rows: list[dict], field: str, point: float) -> float:
    return float(np.percentile([float(row[field]) for row in rows], point))


def level(value: float, cap: float) -> float:
    return math.log1p(min(max(value, 0.0), cap)) / max(math.log1p(cap), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rejects", type=Path, default=DEFAULT_REJECTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--retain-fraction", type=float, default=0.60)
    parser.add_argument("--threshold-config", type=Path, help="Use recorded hard limits and score caps")
    args = parser.parse_args()
    if not 0 < args.retain_fraction <= 1:
        parser.error("--retain-fraction must be in (0, 1]")

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")

    caps = {
        "elements": percentile(rows, "elements", 95),
        "texts": percentile(rows, "texts", 95),
        "edges": percentile(rows, "edges", 95),
        "pixels": percentile(rows, "pixels", 90),
    }
    soft_limits = {
        "elements_p99": percentile(rows, "elements", 99),
        "aspect_p95": percentile(rows, "aspect", 95),
    }
    limits = {
        "min_elements": max(6, int(percentile(rows, "elements", 5))),
        "min_vertices": 3,
        "min_side": 80,
        "min_pixels": 20_000,
        "max_pixels": min(100_000_000, max(20_000_000, int(percentile(rows, "pixels", 99.8)))),
        "max_aspect": min(12.0, max(8.0, percentile(rows, "aspect", 99.5))),
        "min_ink_ratio": 0.02,
        "max_ink_ratio": 0.995,
        "min_gray_std": 2.0,
        "extreme_edge_count": max(250, int(percentile(rows, "edges", 99.5))),
        "extreme_edge_vertex_ratio": 8.0,
    }

    if args.threshold_config is not None:
        recorded = json.loads(args.threshold_config.read_text())
        caps = recorded["score_caps_p95_or_p90"]
        soft_limits = recorded["soft_limits"]
        limits = recorded["hard_limits"]

    eligible = []
    rejected = []
    for original in rows:
        row = dict(original)
        reasons = []
        if row["elements"] < limits["min_elements"]:
            reasons.append("too_few_elements")
        if row["verts"] < limits["min_vertices"]:
            reasons.append("too_few_vertices")
        if min(row["png_w"], row["png_h"]) < limits["min_side"]:
            reasons.append("side_too_short")
        if row["pixels"] < limits["min_pixels"]:
            reasons.append("too_few_pixels")
        if row["pixels"] > limits["max_pixels"]:
            reasons.append("extreme_pixel_count")
        if row["aspect"] > limits["max_aspect"]:
            reasons.append("extreme_aspect_ratio")
        if not limits["min_ink_ratio"] <= row["ink_ratio"] <= limits["max_ink_ratio"]:
            reasons.append("extreme_ink_ratio")
        if row["gray_std"] < limits["min_gray_std"]:
            reasons.append("near_blank_image")
        edge_vertex_ratio = row["edges"] / max(row["verts"], 1)
        if (row["edges"] > limits["extreme_edge_count"]
                and edge_vertex_ratio > limits["extreme_edge_vertex_ratio"]):
            reasons.append("extreme_repeated_edges")

        score = (
            0.52 * level(row["elements"], caps["elements"])
            + 0.23 * level(row["texts"], caps["texts"])
            + 0.17 * level(row["edges"], caps["edges"])
            + 0.08 * level(row["pixels"], caps["pixels"])
        )
        if row["texts"] > 0 and row["edges"] > 0:
            score += 0.04
        if row["elements"] > soft_limits["elements_p99"]:
            score -= 0.10
        if row["aspect"] > soft_limits["aspect_p95"]:
            score -= 0.08
        if row["ink_ratio"] > 0.97 or row["ink_ratio"] < 0.05:
            score -= 0.12
        if edge_vertex_ratio > 5:
            score -= min(0.20, 0.02 * (edge_vertex_ratio - 5))
        row["structure_score"] = round(score, 8)
        row["edge_vertex_ratio"] = round(edge_vertex_ratio, 6)

        if reasons:
            row["structural_reject_reasons"] = reasons
            rejected.append(row)
        else:
            eligible.append(row)

    eligible.sort(key=lambda row: (row["structure_score"], row["elements"], row["png"]), reverse=True)
    target = min(len(eligible), round(len(rows) * args.retain_fraction))
    candidates = eligible[:target]
    cutoff = candidates[-1]["structure_score"] if candidates else None
    for rank, row in enumerate(candidates, 1):
        row["coarse_rank"] = rank
        row["structure_percentile"] = round(1.0 - (rank - 1) / max(len(eligible) - 1, 1), 8)
    for row in eligible[target:]:
        row["structural_reject_reasons"] = ["below_structure_cutoff"]
        rejected.append(row)

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.candidates, candidates)
    write_jsonl(args.rejects, rejected)
    report = {
        "input_unique_pages": len(rows),
        "hard_eligible": len(eligible),
        "hard_rejected": len(rows) - len(eligible),
        "retain_fraction_of_unique": args.retain_fraction,
        "coarse_candidates": len(candidates),
        "structure_cutoff": cutoff,
        "score_caps_p95_or_p90": caps,
        "soft_limits": soft_limits,
        "hard_limits": limits,
        "policy": "hard obvious-invalid rules, then robust capped structure score; model is a later veto",
    }
    args.thresholds.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
