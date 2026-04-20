"""
r2_storage.py
Handles all Cloudflare R2 interactions for the movie app.
Stores two files in R2:
  - bookmarks.html  (your movie bookmark list, always saved under this fixed key)
  - omdb_cache.json (persistent OMDb ratings cache)
"""

import json
import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

# ── R2 credentials (set these as environment variables in Railway) ──────────
ACCOUNT_ID  = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
ACCESS_KEY  = os.environ.get("R2_ACCESS_KEY_ID", "")
SECRET_KEY  = os.environ.get("R2_SECRET_ACCESS_KEY", "")
BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "jz-movies")

ENDPOINT_URL = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

# Fixed keys — the uploaded filename is IGNORED; we always use these.
BOOKMARKS_KEY = "bookmarks.html"
CACHE_KEY     = "omdb_cache.json"


def _client():
    """Return a boto3 S3 client pointed at R2."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


# ── Bookmarks ────────────────────────────────────────────────────────────────

def upload_bookmarks(file_bytes: bytes, original_filename: str = "") -> dict:
    """
    Save bookmark HTML to R2 under the fixed key 'bookmarks.html'.
    The original_filename is only used for the confirmation message —
    it has no effect on where the file is stored.

    Returns: {"success": bool, "message": str, "original_name": str}
    """
    try:
        _client().put_object(
            Bucket=BUCKET_NAME,
            Key=BOOKMARKS_KEY,
            Body=file_bytes,
            ContentType="text/html",
        )
        name_note = (
            f" (saved from '{original_filename}')" if original_filename else ""
        )
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


def download_bookmarks() -> bytes | None:
    """
    Fetch the bookmarks HTML from R2.
    Returns raw bytes, or None if the file doesn't exist yet.
    """
    try:
        response = _client().get_object(Bucket=BUCKET_NAME, Key=BOOKMARKS_KEY)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def bookmarks_last_modified() -> str | None:
    """Return a human-readable last-modified timestamp, or None."""
    try:
        resp = _client().head_object(Bucket=BUCKET_NAME, Key=BOOKMARKS_KEY)
        dt = resp.get("LastModified")
        return dt.strftime("%B %d, %Y at %I:%M %p UTC") if dt else None
    except ClientError:
        return None


# ── OMDb Cache ───────────────────────────────────────────────────────────────

def load_cache() -> dict:
    """
    Download the OMDb cache from R2 and return it as a dict.
    Returns an empty dict if no cache exists yet.
    """
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
    """
    Upload the OMDb cache dict to R2.
    Returns True on success, False on failure.
    """
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
