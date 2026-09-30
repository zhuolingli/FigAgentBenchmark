#!/usr/bin/env python3
"""Build a page-level index, conservatively deduplicate PNGs, and report stats.

The source dataset is read-only. All generated manifests live in
QUALITY_SELECTION_V2 and galleries are created later as symlinks.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(os.environ.get(
    "DRAWIO_QUALITY_ROOT",
    "work",
))
DS = Path(os.environ.get("DRAWIO_QUALITY_DATASET", ROOT / "MY_DRAWIO_DATASET"))
OUT = Path(os.environ.get("DRAWIO_QUALITY_OUTPUT", ROOT / "QUALITY_SELECTION_V2"))
MANIFEST = DS / "quality_manifest.jsonl"
ALL_PAGES = OUT / "all_pages.jsonl"
UNIQUE_PAGES = OUT / "unique_pages.jsonl"
CLUSTERS = OUT / "dedup_clusters.jsonl"
STATS = OUT / "stats.json"

PNG_RE = re.compile(r"^page_(\d+)\.png$")
VERT = re.compile(r'vertex="1"')
EDGE = re.compile(r'edge="1"')
IMG = re.compile(r"shape=image|image=data:|image=http|data:image", re.I)
VALUE = re.compile(r'value="([^"]+)"')
TAG = re.compile(r"<[^>]+>")
GMODEL = re.compile(r"<mxGraphModel")


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def read_drawio(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def inflate_diagram(body: str) -> str:
    try:
        raw = base64.b64decode(body.strip())
        return unquote(zlib.decompress(raw, -15).decode("utf-8", "replace"))
    except Exception:
        return ""


def diagram_pages(source: Path) -> list[tuple[str, str]]:
    """Return every page slot as (name, XML), including self-closing blanks."""
    text = read_drawio(source)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [("Page-1", text)]
    pages = [child for child in list(root) if child.tag.rsplit("}", 1)[-1] == "diagram"]
    if not pages:
        return [("Page-1", text)]
    result = []
    for i, page in enumerate(pages, 1):
        children = list(page)
        if children:
            body = "".join(ET.tostring(child, encoding="unicode") for child in children)
        else:
            body = page.text or ""
            if body.strip() and not GMODEL.search(body):
                body = inflate_diagram(body) or body
        result.append((page.attrib.get("name") or f"Page-{i}", body))
    return result


def page_metrics(name: str, body: str) -> dict:
    verts = len(VERT.findall(body))
    edges = len(EDGE.findall(body))
    imgs = len(IMG.findall(body))
    texts = sum(bool(TAG.sub("", value).strip()) for value in VALUE.findall(body))
    return {
        "page_name": name,
        "verts": verts,
        "edges": edges,
        "elements": verts + edges,
        "boxes": max(verts - imgs, 0),
        "images": imgs,
        "texts": texts,
    }


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def image_features(path: Path) -> dict:
    data = path.read_bytes()
    width = height = 0
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
    result = {
        "png_bytes": len(data),
        "png_w": width,
        "png_h": height,
        "pixels": width * height,
        "aspect": round(max(width, height) / max(1, min(width, height)), 6),
        "sha256": hashlib.sha256(data).hexdigest(),
        "phash": "",
        "dhash": "",
        "ink_ratio": 0.0,
        "gray_mean": 0.0,
        "gray_std": 0.0,
        "image_error": "",
    }
    try:
        with Image.open(io.BytesIO(data)) as image:
            gray = image.convert("RGB")
            # White-composite transparent pixels before grayscale conversion.
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                gray = Image.alpha_composite(white, rgba).convert("RGB")
            sample = gray.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
            sample_arr = np.asarray(sample, dtype=np.float32)
            ph_arr = np.asarray(sample.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
            low = dctn(ph_arr, type=2, norm="ortho")[:8, :8]
            median = float(np.median(low.reshape(-1)[1:]))
            ph_bits = low > median
            ph_bits[0, 0] = False
            dh_arr = np.asarray(gray.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            result.update({
                "phash": bits_to_hex(ph_bits),
                "dhash": bits_to_hex(dh_arr[:, 1:] > dh_arr[:, :-1]),
                "ink_ratio": round(float(np.mean(sample_arr < 245)), 6),
                "gray_mean": round(float(np.mean(sample_arr)), 4),
                "gray_std": round(float(np.std(sample_arr)), 4),
            })
    except Exception as exc:
        result["image_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def is_cs_paper(arxiv_id: str) -> bool:
    try:
        record = json.loads((DS / arxiv_id / "record.json").read_text(encoding="utf-8"))
        categories = record.get("categories") or []
        if isinstance(categories, str):
            categories = categories.split()
        return any(str(category).startswith("cs.") for category in categories)
    except Exception:
        return False


def diagram_pngs(directory: Path) -> list[tuple[int, Path]]:
    rows = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                match = PNG_RE.match(entry.name)
                if match and entry.is_file(follow_symlinks=True):
                    rows.append((int(match.group(1)), Path(entry.path)))
    except OSError:
        pass
    return sorted(rows)


def index_diagram(item: tuple[dict, bool]) -> list[dict]:
    manifest_row, is_cs = item
    aid, diagram = manifest_row["arxiv_id"], manifest_row["diagram"]
    directory = DS / aid / "diagrams" / diagram
    try:
        pages = diagram_pages(directory / "source.drawio")
    except Exception as exc:
        pages = []
        source_error = f"{type(exc).__name__}: {exc}"[:300]
    else:
        source_error = ""
    metrics = [page_metrics(name, body) for name, body in pages]
    rows = []
    for page_number, png in diagram_pngs(directory):
        if 1 <= page_number <= len(metrics):
            page = metrics[page_number - 1]
            mapping_error = ""
        else:
            page = page_metrics(f"Page-{page_number}", "")
            mapping_error = f"page index {page_number} outside source page count {len(metrics)}"
        row = {
            "arxiv_id": aid,
            "diagram": diagram,
            "page_number": page_number,
            "png": str(png),
            "source_drawio": str(directory / "source.drawio"),
            "source_pages": len(metrics),
            "is_cs": is_cs,
            "is_main": bool(manifest_row.get("is_main")),
            "old_quality_pass": bool(manifest_row.get("pass")),
            "old_quality_score": manifest_row.get("score"),
            "source_error": source_error,
            "mapping_error": mapping_error,
            **page,
            **image_features(png),
        }
        row["valid"] = not (row["source_error"] or row["mapping_error"] or row["image_error"])
        rows.append(row)
    return rows


def unique_manifest_rows() -> list[dict]:
    seen = set()
    rows = []
    for row in read_jsonl(MANIFEST):
        key = (row["arxiv_id"], row["diagram"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def run_index(workers: int, include_non_cs: bool, rebuild: bool) -> None:
    if ALL_PAGES.exists() and not rebuild:
        log(f"index exists, skipping: {ALL_PAGES}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = unique_manifest_rows()
    paper_ids = list(dict.fromkeys(row["arxiv_id"] for row in manifest_rows))
    with cf.ThreadPoolExecutor(max_workers=min(workers, 24)) as ex:
        flags = list(ex.map(is_cs_paper, paper_ids))
    cs_by_paper = dict(zip(paper_ids, flags))
    items = [(row, cs_by_paper[row["arxiv_id"]]) for row in manifest_rows
             if include_non_cs or cs_by_paper[row["arxiv_id"]]]
    log(f"indexing {len(items)}/{len(manifest_rows)} diagrams; workers={workers}")
    tmp = ALL_PAGES.with_suffix(".jsonl.tmp")
    page_count = 0
    with tmp.open("w", encoding="utf-8") as fh, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(index_diagram, items), 1):
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                page_count += 1
            if i % 500 == 0:
                log(f"  indexed {i}/{len(items)} diagrams; pages={page_count}")
    tmp.replace(ALL_PAGES)
    log(f"wrote {ALL_PAGES}: {page_count} pages")


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value: str, index: int) -> None:
        if self.root is None:
            self.root = [value, index, {}]
            return
        node = self.root
        while True:
            distance = hamming(value, node[0])
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, index, {}]
                return
            node = child

    def query(self, value: str, radius: int) -> list[int]:
        if self.root is None:
            return []
        found = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node[0])
            if distance <= radius:
                found.append(node[1])
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def keep_key(row: dict) -> tuple:
    return (
        int(row["elements"]),
        int(row["texts"]),
        int(row["pixels"]),
        int(row["png_bytes"]),
        int(bool(row["is_main"])),
        row["png"],
    )


@lru_cache(maxsize=8192)
def content_thumbnail(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalize away empty canvas so sparse, unrelated drawings do not hash together."""
    with Image.open(path) as source:
        if source.mode in ("RGBA", "LA") or "transparency" in source.info:
            rgba = source.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(white, rgba).convert("L")
        else:
            image = source.convert("L")
        sample = image.copy()
    sample.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    array = np.asarray(sample)
    ink = array < 248
    if not np.any(ink):
        raise ValueError("blank image")
    ys, xs = np.where(ink)
    left, right = max(0, int(xs.min()) - 2), min(sample.width, int(xs.max()) + 3)
    top, bottom = max(0, int(ys.min()) - 2), min(sample.height, int(ys.max()) + 3)
    cropped = sample.crop((left, top, right, bottom))
    scale = min(64 / cropped.width, 64 / cropped.height)
    size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(resized, ((64 - size[0]) // 2, (64 - size[1]) // 2))
    normalized = np.asarray(canvas, dtype=np.float32)
    return normalized, normalized < 235


def near_duplicate(left: dict, right: dict) -> bool:
    if not left["phash"] or not right["phash"] or not left["dhash"] or not right["dhash"]:
        return False
    aspect_gap = abs(math.log(max(left["aspect"], 1e-6) / max(right["aspect"], 1e-6)))
    hashes_match = (
        hamming(left["phash"], right["phash"]) <= 4
        and hamming(left["dhash"], right["dhash"]) <= 6
        and aspect_gap <= 0.025
        and abs(left["ink_ratio"] - right["ink_ratio"]) <= 0.025
        and abs(left["gray_mean"] - right["gray_mean"]) <= 8.0
    )
    if not hashes_match or not left.get("png") or not right.get("png"):
        return False
    try:
        left_gray, left_ink = content_thumbnail(left["png"])
        right_gray, right_ink = content_thumbnail(right["png"])
    except Exception:
        return False
    mae = float(np.mean(np.abs(left_gray - right_gray)))
    union = int(np.count_nonzero(left_ink | right_ink))
    ink_iou = float(np.count_nonzero(left_ink & right_ink)) / max(union, 1)
    return mae <= 18.0 and ink_iou >= 0.65


def run_dedup(rebuild: bool) -> None:
    if UNIQUE_PAGES.exists() and CLUSTERS.exists() and not rebuild:
        log(f"dedup outputs exist, skipping: {UNIQUE_PAGES}")
        return
    rows = [row for row in read_jsonl(ALL_PAGES) if row["is_cs"] and row["valid"]]
    rows.sort(key=lambda row: row["png"])
    exact = defaultdict(list)
    for row in rows:
        exact[row["sha256"]].append(row)
    representatives = [max(group, key=keep_key) for group in exact.values()]
    representatives.sort(key=lambda row: row["png"])
    exact_members = {max(group, key=keep_key)["png"]: group for group in exact.values()}

    union = UnionFind(len(representatives))
    tree = BKTree()
    for i, row in enumerate(representatives):
        if row["phash"]:
            for other in tree.query(row["phash"], 4):
                if near_duplicate(row, representatives[other]):
                    union.union(i, other)
            tree.add(row["phash"], i)
    components = defaultdict(list)
    for i, row in enumerate(representatives):
        components[union.find(i)].append(row)

    # Split transitive chains using complete linkage: every pair in a final near
    # duplicate group must satisfy the conservative comparison directly.
    groups = []
    for component in components.values():
        buckets = []
        for row in sorted(component, key=keep_key, reverse=True):
            for bucket in buckets:
                if all(near_duplicate(row, member) for member in bucket):
                    bucket.append(row)
                    break
            else:
                buckets.append([row])
        groups.extend(buckets)

    winners = []
    cluster_rows = []
    ordered_groups = sorted(groups, key=lambda group: min(row["png"] for row in group))
    for number, near_group in enumerate(ordered_groups, 1):
        all_members = []
        for representative in near_group:
            all_members.extend(exact_members[representative["png"]])
        winner = max(all_members, key=keep_key)
        cluster_id = f"cluster_{number:06d}"
        winner = dict(winner)
        winner.update({
            "dedup_cluster": cluster_id,
            "dedup_cluster_size": len(all_members),
            "dedup_exact_variants": len(near_group),
        })
        winners.append(winner)
        if len(all_members) > 1:
            cluster_rows.append({
                "cluster_id": cluster_id,
                "size": len(all_members),
                "exact_variants": len(near_group),
                "keep": winner["png"],
                "keep_elements": winner["elements"],
                "members": [{
                    "png": row["png"],
                    "elements": row["elements"],
                    "sha256": row["sha256"],
                    "phash": row["phash"],
                } for row in sorted(all_members, key=lambda row: row["png"])],
            })
    winners.sort(key=lambda row: row["png"])
    write_jsonl(UNIQUE_PAGES, winners)
    write_jsonl(CLUSTERS, cluster_rows)
    log(f"dedup: valid_cs={len(rows)} exact_representatives={len(representatives)} "
        f"unique={len(winners)} removed={len(rows)-len(winners)} duplicate_clusters={len(cluster_rows)}")


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    points = [0, 1, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99, 100]
    return {f"p{point}": round(float(np.percentile(array, point)), 6) for point in points}


def run_stats() -> None:
    indexed = read_jsonl(ALL_PAGES)
    unique = read_jsonl(UNIQUE_PAGES)
    fields = ["elements", "verts", "edges", "boxes", "images", "texts", "png_bytes",
              "png_w", "png_h", "pixels", "aspect", "ink_ratio", "gray_mean", "gray_std"]
    stats = {
        "counts": {
            "indexed_pages": len(indexed),
            "indexed_cs_pages": sum(bool(row["is_cs"]) for row in indexed),
            "valid_cs_pages": sum(bool(row["is_cs"] and row["valid"]) for row in indexed),
            "mapping_errors": sum(bool(row["mapping_error"]) for row in indexed),
            "source_errors": sum(bool(row["source_error"]) for row in indexed),
            "image_errors": sum(bool(row["image_error"]) for row in indexed),
            "unique_pages": len(unique),
            "duplicates_removed": sum(bool(row["is_cs"] and row["valid"]) for row in indexed) - len(unique),
        },
        "quantiles": {field: quantiles([row[field] for row in unique]) for field in fields},
        "zero_counts": {
            "elements": sum(row["elements"] == 0 for row in unique),
            "edges": sum(row["edges"] == 0 for row in unique),
            "texts": sum(row["texts"] == 0 for row in unique),
        },
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(stats["counts"], sort_keys=True))
    log(f"wrote {STATS}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("index", "dedup", "stats", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-non-cs", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.stage in ("index", "all"):
        run_index(args.workers, args.include_non_cs, args.rebuild)
    if args.stage in ("dedup", "all"):
        run_dedup(args.rebuild)
    if args.stage in ("stats", "all"):
        run_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
