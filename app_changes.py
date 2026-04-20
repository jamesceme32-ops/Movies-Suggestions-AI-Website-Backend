# ─────────────────────────────────────────────────────────────────────────────
# CHANGES TO app.py
# Add these imports, replace cache functions, and add the upload route.
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. ADD to your imports at the top of app.py ──────────────────────────────

import r2_storage  # new file we created


# ── 2. REPLACE your existing load_cache / save_cache functions ───────────────
#    (delete the old ones that read/write a local JSON file)

def load_cache():
    """Load OMDb cache from R2 (persists across deploys)."""
    return r2_storage.load_cache()

def save_cache(cache):
    """Save OMDb cache to R2."""
    r2_storage.save_cache(cache)


# ── 3. REPLACE how you load the bookmarks HTML ───────────────────────────────
#    Find wherever your app reads the local bookmarks file (e.g. open("bookmarks.html"))
#    and replace with this:

def get_bookmarks_html():
    """
    Fetch bookmarks HTML from R2.
    Returns decoded HTML string, or None if not uploaded yet.
    """
    raw = r2_storage.download_bookmarks()
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")


# ── 4. ADD this upload route ─────────────────────────────────────────────────
#    Paste this into app.py alongside your other @app.route definitions.

@app.route("/upload", methods=["GET", "POST"])
def upload():
    message      = None
    success      = False
    last_modified = r2_storage.bookmarks_last_modified()

    if request.method == "POST":
        file = request.files.get("bookmarks_file")

        # Validate: file present
        if not file or file.filename == "":
            message = "No file selected. Please choose an HTML file."
            success = False

        # Validate: must be .html or .htm
        elif not file.filename.lower().endswith((".html", ".htm")):
            message = (
                f"'{file.filename}' doesn't look like an HTML file. "
                "Please export your bookmarks as an HTML file and try again."
            )
            success = False

        else:
            file_bytes = file.read()

            # Basic sanity check: should contain some HTML
            if b"<" not in file_bytes:
                message = "The file doesn't appear to be valid HTML. Please check and re-export."
                success = False
            else:
                result   = r2_storage.upload_bookmarks(file_bytes, file.filename)
                message  = result["message"]
                success  = result["success"]
                # Refresh timestamp after successful upload
                if success:
                    last_modified = r2_storage.bookmarks_last_modified()

    return render_template(
        "upload.html",
        message=message,
        success=success,
        last_modified=last_modified,
    )


# ── 5. UPDATE your index / main route ────────────────────────────────────────
#    Find your main route (probably @app.route("/")) and update it to handle
#    the case where no bookmarks file exists yet.
#
#    Example — adjust to match your actual route logic:

@app.route("/")
def index():
    html_content = get_bookmarks_html()

    if html_content is None:
        # No bookmarks uploaded yet — send user to upload page
        return render_template("index.html", no_bookmarks=True)

    # ... rest of your existing parsing / logic ...
    # pass html_content to your parser instead of opening a local file


# ── 6. ADD to requirements.txt ───────────────────────────────────────────────
#
#    boto3
#
#    (everything else you already have stays the same)


# ── 7. SET these environment variables in Railway ────────────────────────────
#
#    CLOUDFLARE_ACCOUNT_ID    your 32-char Cloudflare account ID
#    R2_ACCESS_KEY_ID         R2 API token access key
#    R2_SECRET_ACCESS_KEY     R2 API token secret key
#    R2_BUCKET_NAME           the bucket name you create (e.g. jz-movies)
