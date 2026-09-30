from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from .scanner import CloudScanner, json_bytes, utc_now


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_job_started_at(s3: Any, bucket: str) -> datetime:
    key = "state/job.json"
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        return datetime.fromisoformat(data["started_at"])
    except s3.exceptions.NoSuchKey:
        pass
    except s3.exceptions.ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
    started = datetime.now(timezone.utc)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json_bytes({"version": 1, "started_at": started.isoformat()}),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return started


def scale_to_zero(asg_name: str, region: str) -> None:
    if not asg_name:
        return
    boto3.client("autoscaling", region_name=region).update_auto_scaling_group(
        AutoScalingGroupName=asg_name,
        MinSize=0,
        MaxSize=16,
        DesiredCapacity=0,
    )


def run_scan(config_path: str, asg_name: str = "") -> int:
    config = load_config(config_path)
    region = config.get("region", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    output_bucket = config["output_bucket"]
    started = get_job_started_at(s3, output_bucket)
    max_hours = float(config.get("max_total_hours", 168))
    elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
    if elapsed_hours >= max_hours:
        print(f"global runtime guard reached: {elapsed_hours:.1f}h >= {max_hours:.1f}h", flush=True)
        scale_to_zero(asg_name, region)
        return 3

    scanner = CloudScanner(
        s3=s3,
        source_bucket=config.get("source_bucket", "arxiv"),
        output_bucket=output_bucket,
        work_dir=config.get("work_dir", "/var/tmp/drawio-cloud-scanner"),
    )
    try:
        summary = scanner.run(
            start=config["from_month"],
            stop=config["to_month"],
            max_source_objects=int(config.get("max_source_objects", 0)),
        )
        s3.put_object(
            Bucket=output_bucket,
            Key=f"state/runs/{int(time.time())}-{socket.gethostname()}-{os.getpid()}.json",
            Body=json_bytes(summary),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        if int(config.get("max_source_objects", 0)) == 0:
            s3.put_object(
                Bucket=output_bucket,
                Key="state/ALL_DONE.json",
                Body=json_bytes({"version": 1, "finished_at": utc_now(), "summary": summary}),
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        print(json.dumps(summary, indent=2), flush=True)
        scale_to_zero(asg_name, region)
        return 0
    except Exception as exc:
        failure = {"version": 1, "failed_at": utc_now(), "error": repr(exc)}
        s3.put_object(
            Bucket=output_bucket,
            Key=f"state/failures/{int(time.time())}-{socket.gethostname()}-{os.getpid()}.json",
            Body=json_bytes(failure),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        raise
    finally:
        scanner.clean_work_dir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drawio-cloud")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan arXiv source tars")
    scan.add_argument("--config", required=True)
    scan.add_argument("--asg-name", default=os.environ.get("DRAWIO_ASG_NAME", ""))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "scan":
        raise SystemExit(run_scan(args.config, args.asg_name))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
