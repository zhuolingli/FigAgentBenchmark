from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .detector import inspect_source_package
from .months import descending_months


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


@dataclass
class MatchRecord:
    arxiv_id: str
    month: str
    source_tar: str
    source_key: str
    source_member: str
    output_key: str
    size: int
    sha256: str
    reasons: list[str]


class CloudScanner:
    def __init__(
        self,
        s3: Any,
        source_bucket: str,
        output_bucket: str,
        work_dir: str | Path,
        request_payer: str = "requester",
    ) -> None:
        self.s3 = s3
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        self.work_dir = Path(work_dir)
        self.request_payer = request_payer
        self.owner = f"{socket.gethostname()}-{os.getpid()}"
        self.claim_ttl_seconds = 1800
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def list_source_objects(self, start: str, stop: str) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for month in descending_months(start, stop):
            prefix = f"src/arXiv_src_{month}_"
            token = None
            while True:
                kwargs: dict[str, Any] = {
                    "Bucket": self.source_bucket,
                    "Prefix": prefix,
                    "RequestPayer": self.request_payer,
                }
                if token:
                    kwargs["ContinuationToken"] = token
                response = self.s3.list_objects_v2(**kwargs)
                for item in response.get("Contents", []):
                    row = dict(item)
                    row["Month"] = month
                    objects.append(row)
                if not response.get("IsTruncated"):
                    break
                token = response["NextContinuationToken"]
        objects.sort(key=lambda row: row["Key"], reverse=True)
        return objects

    @staticmethod
    def completion_key(month: str, source_key: str) -> str:
        return f"state/completed/{month}/{PurePosixPath(source_key).name}.json"

    @staticmethod
    def manifest_key(month: str, source_key: str) -> str:
        return f"manifests/{month}/{PurePosixPath(source_key).name}.jsonl"

    @staticmethod
    def claim_key(month: str, source_key: str) -> str:
        return f"state/claims/{month}/{PurePosixPath(source_key).name}.json"

    @staticmethod
    def _is_precondition_error(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"} or status in {409, 412}

    def acquire_claim(self, month: str, source_key: str) -> bool:
        key = self.claim_key(month, source_key)
        payload = json_bytes({"version": 1, "owner": self.owner, "claimed_at": utc_now()})
        try:
            self.s3.put_object(
                Bucket=self.output_bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
            return True
        except self.s3.exceptions.ClientError as exc:
            if not self._is_precondition_error(exc):
                raise

        try:
            current = self.s3.get_object(Bucket=self.output_bucket, Key=key)
        except self.s3.exceptions.ClientError:
            return False
        last_modified = current["LastModified"]
        current["Body"].close()
        if datetime.now(timezone.utc) - last_modified <= timedelta(seconds=self.claim_ttl_seconds):
            return False
        try:
            self.s3.put_object(
                Bucket=self.output_bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                ServerSideEncryption="AES256",
                IfMatch=str(current["ETag"]).strip('"'),
            )
            print(f"reclaimed stale claim {key}", flush=True)
            return True
        except self.s3.exceptions.ClientError as exc:
            if self._is_precondition_error(exc):
                return False
            raise

    def release_claim(self, month: str, source_key: str) -> None:
        self.s3.delete_object(Bucket=self.output_bucket, Key=self.claim_key(month, source_key))

    def is_completed(self, month: str, source_key: str) -> bool:
        try:
            self.s3.head_object(
                Bucket=self.output_bucket,
                Key=self.completion_key(month, source_key),
            )
            return True
        except self.s3.exceptions.ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def _copy_member(self, stream: BinaryIO, destination: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        return size, digest.hexdigest()

    def _upload_match(
        self,
        path: Path,
        month: str,
        arxiv_id: str,
        sha256: str,
        reasons: tuple[str, ...],
    ) -> str:
        output_key = f"matches/{month}/{arxiv_id}.gz"
        self.s3.upload_file(
            str(path),
            self.output_bucket,
            output_key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "arxiv-id": arxiv_id,
                    "sha256": sha256,
                    "detection": ",".join(reasons),
                },
            },
        )
        return output_key

    def process_source_object(self, source_key: str, month: str) -> dict[str, Any]:
        started = utc_now()
        source_name = PurePosixPath(source_key).name
        matches: list[MatchRecord] = []
        papers_seen = 0
        invalid_packages = 0
        response = self.s3.get_object(
            Bucket=self.source_bucket,
            Key=source_key,
            RequestPayer=self.request_payer,
        )
        body = response["Body"]
        try:
            with tarfile.open(fileobj=body, mode="r|*") as outer:
                for member in outer:
                    if not member.isfile() or not member.name.lower().endswith(".gz"):
                        continue
                    extracted = outer.extractfile(member)
                    if extracted is None:
                        continue
                    papers_seen += 1
                    basename = PurePosixPath(member.name).name
                    arxiv_id = basename[:-3]
                    fd, temp_name = tempfile.mkstemp(prefix="paper-", suffix=".gz", dir=self.work_dir)
                    os.close(fd)
                    temp_path = Path(temp_name)
                    try:
                        size, sha256 = self._copy_member(extracted, temp_path)
                        detection = inspect_source_package(temp_path)
                        if "invalid_source_package" in detection.reasons:
                            invalid_packages += 1
                        if not detection.matched:
                            continue
                        output_key = self._upload_match(
                            temp_path, month, arxiv_id, sha256, detection.reasons
                        )
                        matches.append(
                            MatchRecord(
                                arxiv_id=arxiv_id,
                                month=month,
                                source_tar=source_name,
                                source_key=source_key,
                                source_member=member.name,
                                output_key=output_key,
                                size=size,
                                sha256=sha256,
                                reasons=list(detection.reasons),
                            )
                        )
                    finally:
                        temp_path.unlink(missing_ok=True)
        finally:
            body.close()

        manifest_key = self.manifest_key(month, source_key)
        manifest_data = b"".join(
            (json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n").encode()
            for row in matches
        )
        self.s3.put_object(
            Bucket=self.output_bucket,
            Key=manifest_key,
            Body=manifest_data,
            ContentType="application/x-ndjson",
            ServerSideEncryption="AES256",
        )
        summary = {
            "version": 1,
            "month": month,
            "source_bucket": self.source_bucket,
            "source_key": source_key,
            "source_etag": str(response.get("ETag", "")).strip('"'),
            "source_size": response.get("ContentLength"),
            "papers_seen": papers_seen,
            "invalid_packages": invalid_packages,
            "matches": len(matches),
            "manifest_key": manifest_key,
            "started_at": started,
            "finished_at": utc_now(),
        }
        self.s3.put_object(
            Bucket=self.output_bucket,
            Key=self.completion_key(month, source_key),
            Body=json_bytes(summary),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        return summary

    def run(
        self,
        start: str,
        stop: str,
        max_source_objects: int = 0,
        retries: int = 3,
    ) -> dict[str, Any]:
        objects = self.list_source_objects(start, stop)
        if max_source_objects > 0:
            objects = objects[:max_source_objects]
        processed = matches = rounds = 0
        completed_seen: set[str] = set()
        started = utc_now()
        while True:
            rounds += 1
            unfinished = claimed_elsewhere = 0
            for index, item in enumerate(objects, 1):
                month, key = item["Month"], item["Key"]
                if self.is_completed(month, key):
                    completed_seen.add(key)
                    continue
                unfinished += 1
                if not self.acquire_claim(month, key):
                    claimed_elsewhere += 1
                    continue
                try:
                    if self.is_completed(month, key):
                        completed_seen.add(key)
                        continue
                    last_error: Exception | None = None
                    for attempt in range(1, retries + 1):
                        try:
                            print(
                                f"[{index}/{len(objects)}] scan {key} attempt={attempt} owner={self.owner}",
                                flush=True,
                            )
                            summary = self.process_source_object(key, month)
                            processed += 1
                            matches += int(summary["matches"])
                            completed_seen.add(key)
                            print(
                                f"[{index}/{len(objects)}] done papers={summary['papers_seen']} "
                                f"matches={summary['matches']}",
                                flush=True,
                            )
                            last_error = None
                            break
                        except Exception as exc:
                            last_error = exc
                            print(f"[{index}/{len(objects)}] error attempt={attempt}: {exc!r}", flush=True)
                            time.sleep(min(30, 2**attempt))
                    if last_error is not None:
                        raise RuntimeError(f"failed after {retries} attempts: {key}") from last_error
                finally:
                    self.release_claim(month, key)
            if max_source_objects > 0 or unfinished == 0:
                break
            print(
                f"round={rounds} unfinished_at_start={unfinished} "
                f"claimed_elsewhere={claimed_elsewhere}; waiting for peers",
                flush=True,
            )
            time.sleep(10)
        return {
            "version": 2,
            "owner": self.owner,
            "from_month": start,
            "to_month": stop,
            "source_objects": len(objects),
            "processed_by_worker": processed,
            "completed_seen": len(completed_seen),
            "matches_by_worker": matches,
            "rounds": rounds,
            "started_at": started,
            "finished_at": utc_now(),
        }

    def clean_work_dir(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
