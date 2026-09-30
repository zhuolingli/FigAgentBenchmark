#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import boto3




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    return [
        row["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for row in page.get("Contents", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_rows: dict[str, dict[str, Any]] = {}
    for key in list_keys(s3, args.bucket, "manifests/"):
        data = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read().decode()
        for line in data.splitlines():
            if line.strip():
                row = json.loads(line)
                manifest_rows[row["output_key"]] = row

    downloaded = skipped = 0
    local_manifest = dest / "selected_sources.jsonl"
    for key, row in sorted(manifest_rows.items()):
        target = dest / row["month"] / f"{row['arxiv_id']}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == row["size"] and sha256(target) == row["sha256"]:
            skipped += 1
            continue
        temporary = target.with_suffix(".gz.partial")
        s3.download_file(args.bucket, key, str(temporary))
        if temporary.stat().st_size != row["size"] or sha256(temporary) != row["sha256"]:
            raise RuntimeError(f"checksum mismatch: {key}")
        os.replace(temporary, target)
        downloaded += 1

    with local_manifest.open("w", encoding="utf-8") as output:
        for row in sorted(manifest_rows.values(), key=lambda item: (item["month"], item["arxiv_id"])):
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"records={len(manifest_rows)} downloaded={downloaded} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
