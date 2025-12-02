from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

import boto3
import orjson

logger = logging.getLogger(__name__)


def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def download_to_tempdir(client, bucket: str, key: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    dest = tmpdir / Path(key).name
    logger.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
    client.download_file(bucket, key, str(dest))
    return dest


def upload_jsonl(client, bucket: str, key: str, records: Iterable[dict]) -> None:
    tmpdir = Path(tempfile.mkdtemp())
    dest = tmpdir / "out.jsonl"
    with dest.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")
    logger.info("Uploading %s -> s3://%s/%s", dest, bucket, key)
    client.upload_file(str(dest), bucket, key)


def stream_jsonl_from_s3(client, bucket: str, key: str) -> Iterator[dict]:
    obj = client.get_object(Bucket=bucket, Key=key)
    for line in obj["Body"].iter_lines():
        if not line:
            continue
        yield json.loads(line)


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.append(item["Key"])
    return keys
