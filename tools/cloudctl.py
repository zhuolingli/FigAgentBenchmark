#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


REGION = "us-east-1"
PROJECT = "drawio-arxiv-scanner"
ROLE_NAME = f"{PROJECT}-role"
PROFILE_NAME = f"{PROJECT}-profile"
SG_NAME = f"{PROJECT}-sg"
LT_NAME = f"{PROJECT}-lt"
ASG_NAME = f"{PROJECT}-asg"
ARTIFACT_KEY = "deployment/scanner.zip"
CONFIG_KEY = "deployment/config.json"




def clients() -> dict[str, Any]:
    session = boto3.Session(region_name=REGION)
    return {name: session.client(service) for name, service in {
        "sts": "sts", "s3": "s3", "iam": "iam", "ec2": "ec2",
        "ssm": "ssm", "asg": "autoscaling",
    }.items()}


def names(cs: dict[str, Any]) -> dict[str, str]:
    account = cs["sts"].get_caller_identity()["Account"]
    return {
        "account": account,
        "bucket": f"drawio-arxiv-matches-{account}-{REGION}",
    }


def ensure_bucket(s3: Any, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "abort-incomplete-multipart-uploads",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                }
            ]
        },
    )


def ensure_role(iam: Any, account: str, bucket: str) -> None:
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        iam.get_role(RoleName=ROLE_NAME)
        iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust))
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListArxivSources",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": "arn:aws:s3:::arxiv",
                "Condition": {"StringLike": {"s3:prefix": "src/arXiv_src_*"}},
            },
            {
                "Sid": "ReadArxivSources",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::arxiv/src/arXiv_src_*",
            },
            {
                "Sid": "ManageScannerOutput",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
            {
                "Sid": "StopOwnScannerGroup",
                "Effect": "Allow",
                "Action": "autoscaling:UpdateAutoScalingGroup",
                "Resource": f"arn:aws:autoscaling:{REGION}:{account}:autoScalingGroup:*:autoScalingGroupName/{ASG_NAME}",
            },
        ],
    }
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName=f"{PROJECT}-policy", PolicyDocument=json.dumps(policy))
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )
    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
    profile = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)["InstanceProfile"]
    if not any(role["RoleName"] == ROLE_NAME for role in profile.get("Roles", [])):
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)


def project_zip(project_root: Path) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ["pyproject.toml", "requirements.txt", "README.md"]:
            path = project_root / relative
            if path.exists():
                archive.write(path, relative)
        for path in sorted((project_root / "src").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(project_root))
    return output.getvalue()


def upload_deployment(s3: Any, bucket: str, project_root: Path, max_objects: int, workers: int, from_month: str = "2410", to_month: str = "2410", max_hours: int = 168) -> None:
    config = {
        "region": REGION,
        "source_bucket": "arxiv",
        "output_bucket": bucket,
        "from_month": from_month,
        "to_month": to_month,
        "max_source_objects": max_objects,
        "workers": workers,
        "max_total_hours": max_hours,
        "work_dir": "/var/tmp/drawio-cloud-scanner",
    }
    s3.put_object(
        Bucket=bucket,
        Key=ARTIFACT_KEY,
        Body=project_zip(project_root),
        ServerSideEncryption="AES256",
    )
    s3.put_object(
        Bucket=bucket,
        Key=CONFIG_KEY,
        Body=(json.dumps(config, indent=2) + "\n").encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def network(ec2: Any) -> tuple[str, str]:
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("no default VPC; create one or extend cloudctl with an explicit VPC")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "state", "Values": ["available"]}]
    )["Subnets"]
    if not subnets:
        raise RuntimeError("default VPC has no available subnet")
    subnet_id = sorted(subnets, key=lambda item: item["AvailableIpAddressCount"], reverse=True)[0]["SubnetId"]
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc_id]}]
    )["SecurityGroups"]
    if groups:
        group_id = groups[0]["GroupId"]
    else:
        group_id = ec2.create_security_group(
            GroupName=SG_NAME,
            Description="No-ingress security group for arXiv draw.io scanner",
            VpcId=vpc_id,
        )["GroupId"]
        ec2.create_tags(Resources=[group_id], Tags=[{"Key": "Project", "Value": PROJECT}])
    return subnet_id, group_id


def user_data(bucket: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/drawio-scanner-bootstrap.log) 2>&1
dnf install -y python3.11 python3.11-pip
python3.11 -m pip install --no-cache-dir boto3
mkdir -p /opt/drawio-scanner
python3.11 - <<'PY'
import boto3, zipfile
s3=boto3.client('s3',region_name='{REGION}')
s3.download_file('{bucket}','{ARTIFACT_KEY}','/opt/drawio-scanner/scanner.zip')
s3.download_file('{bucket}','{CONFIG_KEY}','/opt/drawio-scanner/config.json')
with zipfile.ZipFile('/opt/drawio-scanner/scanner.zip') as z: z.extractall('/opt/drawio-scanner/project')
PY
python3.11 -m pip install --no-cache-dir /opt/drawio-scanner/project
cat >/etc/systemd/system/drawio-scanner.service <<'UNIT'
[Unit]
Description=arXiv draw.io cloud scanner
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
Environment=DRAWIO_ASG_NAME={ASG_NAME}
ExecStart=/usr/local/bin/drawio-cloud scan --config /opt/drawio-scanner/config.json
StandardOutput=journal
StandardError=journal
TimeoutStartSec=infinity
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now drawio-scanner.service
"""


def ensure_launch_template(cs: dict[str, Any], bucket: str) -> str:
    ec2, ssm = cs["ec2"], cs["ssm"]
    subnet_id, group_id = network(ec2)
    ami = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )["Parameter"]["Value"]
    data = {
        "ImageId": ami,
        "InstanceType": "c7i-flex.large",
        "IamInstanceProfile": {"Name": PROFILE_NAME},
        "UserData": base64.b64encode(user_data(bucket).encode()).decode(),
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 20,
                    "VolumeType": "gp3",
                    "Encrypted": True,
                    "DeleteOnTermination": True,
                },
            }
        ],
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "SubnetId": subnet_id,
                "Groups": [group_id],
                "AssociatePublicIpAddress": True,
                "DeleteOnTermination": True,
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Project", "Value": PROJECT}]},
            {"ResourceType": "volume", "Tags": [{"Key": "Project", "Value": PROJECT}]},
        ],
    }
    try:
        existing = ec2.describe_launch_templates(LaunchTemplateNames=[LT_NAME])["LaunchTemplates"][0]
        version = ec2.create_launch_template_version(
            LaunchTemplateName=LT_NAME,
            SourceVersion=str(existing["LatestVersionNumber"]),
            LaunchTemplateData=data,
        )["LaunchTemplateVersion"]["VersionNumber"]
        ec2.modify_launch_template(LaunchTemplateName=LT_NAME, DefaultVersion=str(version))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "InvalidLaunchTemplateName.NotFoundException":
            raise
        response = ec2.create_launch_template(
            LaunchTemplateName=LT_NAME,
            VersionDescription="initial",
            LaunchTemplateData=data,
            TagSpecifications=[
                {"ResourceType": "launch-template", "Tags": [{"Key": "Project", "Value": PROJECT}]}
            ],
        )
        version = response["LaunchTemplate"]["LatestVersionNumber"]
    return str(version)


def ensure_asg(cs: dict[str, Any], version: str, start: bool, workers: int) -> None:
    asg = cs["asg"]
    desired = workers if start else 0
    try:
        current = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])["AutoScalingGroups"]
        if current:
            asg.update_auto_scaling_group(
                AutoScalingGroupName=ASG_NAME,
                LaunchTemplate={"LaunchTemplateName": LT_NAME, "Version": version},
                MinSize=0,
                MaxSize=workers,
                DesiredCapacity=desired,
            )
            return
        asg.create_auto_scaling_group(
            AutoScalingGroupName=ASG_NAME,
            LaunchTemplate={"LaunchTemplateName": LT_NAME, "Version": version},
            MinSize=0,
            MaxSize=workers,
            DesiredCapacity=desired,
            HealthCheckType="EC2",
            HealthCheckGracePeriod=300,
            Tags=[{"Key": "Project", "Value": PROJECT, "PropagateAtLaunch": True}],
        )
    except ClientError:
        raise


def status(cs: dict[str, Any], bucket: str) -> None:
    groups = cs["asg"].describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])["AutoScalingGroups"]
    if groups:
        group = groups[0]
        print(json.dumps({
            "asg": ASG_NAME,
            "desired": group["DesiredCapacity"],
            "instances": [
                {"id": row["InstanceId"], "state": row["LifecycleState"], "health": row["HealthStatus"]}
                for row in group["Instances"]
            ],
            "bucket": bucket,
        }, indent=2))
    else:
        print(json.dumps({"asg": None, "bucket": bucket}, indent=2))


def reset_job_clock(s3: Any, bucket: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key="state/job.json",
        Body=(json.dumps({"version": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}, indent=2) + "\n").encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["provision", "start", "stop", "status"])
    parser.add_argument("--max-source-objects", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 17))
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--from-month", default="2410")
    parser.add_argument("--to-month", default="2410")
    parser.add_argument("--max-hours", type=int, default=168)
    parser.add_argument("--full", action="store_true", help="Explicitly allow an unlimited archive scan")
    args = parser.parse_args()
    from drawio_cloud.months import descending_months
    list(descending_months(args.from_month, args.to_month))
    if args.max_source_objects < 0 or args.max_hours < 1:
        parser.error("object limit must be non-negative and max-hours positive")
    if args.max_source_objects == 0 and not args.full:
        parser.error("unlimited scanning requires --full")
    cs = clients()
    ns = names(cs)
    project_root = Path(__file__).resolve().parents[1]

    if args.command == "provision":
        ensure_bucket(cs["s3"], ns["bucket"])
        ensure_role(cs["iam"], ns["account"], ns["bucket"])
        upload_deployment(cs["s3"], ns["bucket"], project_root, args.max_source_objects, args.workers, args.from_month, args.to_month, args.max_hours)
        time.sleep(10)  # IAM instance-profile propagation
        version = ensure_launch_template(cs, ns["bucket"])
        ensure_asg(cs, version, start=not args.no_start, workers=args.workers)
        status(cs, ns["bucket"])
        return 0
    if args.command == "start":
        upload_deployment(cs["s3"], ns["bucket"], project_root, args.max_source_objects, args.workers, args.from_month, args.to_month, args.max_hours)
        reset_job_clock(cs["s3"], ns["bucket"])
        cs["asg"].update_auto_scaling_group(
            AutoScalingGroupName=ASG_NAME, MinSize=0, MaxSize=args.workers, DesiredCapacity=args.workers
        )
        status(cs, ns["bucket"])
        return 0
    if args.command == "stop":
        cs["asg"].update_auto_scaling_group(
            AutoScalingGroupName=ASG_NAME, MinSize=0, MaxSize=16, DesiredCapacity=0
        )
        status(cs, ns["bucket"])
        return 0
    status(cs, ns["bucket"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
