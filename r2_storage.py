"""
r2_storage.py — Cloudflare R2 storage for JZ's Cinematch
All functions fail gracefully — credentials errors never crash the app.
"""

import json, os, datetime
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

ACCOUNT_ID   = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
ACCESS_KEY   = os.environ.get("R2_ACCESS_KEY_ID", "")
SECRET_KEY   = os.environ.get("R2_SECRET_ACCESS_KEY", "")
BUCKET_NAME  = os.environ.get("R2_BUCKET_NAME", "jz-movies")
ENDPOINT_URL = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

MOVIES_DB_KEY = "movies_db.json"
CACHE_KEY     = "omdb_cache.json"
BOOKMARKS_KEY = "bookmarks.html"   # kept for one-time import only


def _ok():
    return bool(ACCOUNT_ID and ACCESS_KEY and SECRET_KEY)

def _client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def _get(key):
    try:
        r = _client().get_object(Bucket=BUCKET_NAME, Key=key)
        return r["Body"].read()
    except Exception:
        return None

def _put(key, body, content_type="application/json"):
    try:
        _client().put_object(Bucket=BUCKET_NAME, Key=key,
                             Body=body, ContentType=content_type)
        return True
    except Exception:
        return False

def _delete(key):
    try:
        _client().delete_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False

def _exists(key):
    try:
        _client().head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False


# ── Movies Database ───────────────────────────────────────────────────────────

def movies_db_exists() -> bool:
    if not _ok(): return False
    return _exists(MOVIES_DB_KEY)

def load_movies_db() -> dict:
    """Returns {"movies": [...], "version": 1} or {} on failure."""
    if not _ok(): return {}
    raw = _get(MOVIES_DB_KEY)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}

def save_movies_db(data: dict) -> bool:
    if not _ok(): return False
    data["last_updated"] = datetime.datetime.utcnow().isoformat()
    return _put(MOVIES_DB_KEY, json.dumps(data, indent=2).encode("utf-8"))


# ── OMDb Cache (kept for import-time use) ────────────────────────────────────

def load_cache() -> dict:
    if not _ok(): return {}
    raw = _get(CACHE_KEY)
    if not raw: return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}

def save_cache(cache: dict) -> bool:
    if not _ok(): return False
    return _put(CACHE_KEY, json.dumps(cache, indent=2).encode("utf-8"))


# ── Bookmarks (one-time import only) ─────────────────────────────────────────

def bookmarks_exist() -> bool:
    if not _ok(): return False
    return _exists(BOOKMARKS_KEY)

def download_bookmarks():
    if not _ok(): return None
    return _get(BOOKMARKS_KEY)

def upload_bookmarks(file_bytes: bytes, original_filename: str = "") -> dict:
    if not _ok():
        return {"success": False, "message": "R2 credentials not configured.", "original_name": original_filename}
    ok = _put(BOOKMARKS_KEY, file_bytes, "text/html")
    if ok:
        return {"success": True,
                "message": f"Saved from '{original_filename}'.",
                "original_name": original_filename}
    return {"success": False, "message": "Upload failed.", "original_name": original_filename}


def load_json(key: str) -> dict:
    """Load any JSON file from R2 by key."""
    if not _ok(): return {}
    raw = _get(key)
    if not raw: return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def save_json(key: str, data: dict) -> bool:
    """Save any dict as JSON to R2 by key."""
    if not _ok(): return False
    body = json.dumps(data, ensure_ascii=False, default=str)
    return _put(key, body.encode("utf-8"))
