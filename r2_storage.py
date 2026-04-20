"""
r2_storage.py
Handles all Cloudflare R2 interactions for the movie app.
"""

import json
import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

ACCOUNT_ID   = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
ACCESS_KEY   = os.environ.get("R2_ACCESS_KEY_ID", "")
SECRET_KEY   = os.environ.get("R2_SECRET_ACCESS_KEY", "")
BUCKET_NAME  = os.environ.get("R2_BUCKET_NAME", "jz-movies")
ENDPOINT_URL = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

BOOKMARKS_KEY = "bookmarks.html"
CACHE_KEY     = "omdb_cache.json"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_bookmarks(file_bytes: bytes, original_filename: str = "") -> dict:
    try:
        _client().put_object(
            Bucket=BUCKET_NAME,
            Key=BOOKMARKS_KEY,
            Body=file_bytes,
            ContentType="text/html",
        )
        name_note = f" (saved from '{original_filename}')" if original_filename else ""
        return {
            "success": True,
            "message": (
                f"Bookmarks updated successfully{name_note}. "
                f"Stored as '{BOOKMARKS_KEY}' — filename doesn't matter."
            ),
            "original_name": original_filename,
        }
    except ClientError as e:
        return {
            "success": False,
            "message": f"Upload failed: {e.response['Error']['Message']}",
            "original_name": original_filename,
        }


def download_bookmarks():
    try:
        response = _client().get_object(Bucket=BUCKET_NAME, Key=BOOKMARKS_KEY)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def bookmarks_last_modified():
    try:
        resp = _client().head_object(Bucket=BUCKET_NAME, Key=BOOKMARKS_KEY)
        dt = resp.get("LastModified")
        return dt.strftime("%B %d, %Y at %I:%M %p UTC") if dt else None
    except ClientError:
        return None


def load_cache() -> dict:
    try:
        response = _client().get_object(Bucket=BUCKET_NAME, Key=CACHE_KEY)
        raw = response["Body"].read()
        return json.loads(raw)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}
        raise
    except json.JSONDecodeError:
        return {}


def save_cache(cache: dict) -> bool:
    try:
        _client().put_object(
            Bucket=BUCKET_NAME,
            Key=CACHE_KEY,
            Body=json.dumps(cache, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except ClientError:
        return False
