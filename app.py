# -*- coding: utf-8 -*-
"""
JZ's Movie Suggester — Flask Web App
"""

import re, json, threading, uuid, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify)
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "jzmovies-dev-key-change-in-prod")

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "YOUR_KEY_HERE")

# In-memory store keyed by session ID
# { session_id: { "df": DataFrame, "profile": dict, "predicted": Series, ... } }
STORE = {}
STORE_LOCK = threading.Lock()

# Cache lives in /tmp on the server (writable on Railway/Render)
CACHE_PATH = Path("/tmp/omdb_cache.json")

# Fetch progress tracker { session_id: { done, total, stats } }
PROGRESS = {}

# ── Constants ─────────────────────────────────────────────────────────────
YEAR_BINS   = [1920,1960,1970,1980,1990,1995,1999,2005,2010,2015,2020,2025,2030]
YEAR_LABELS = ["1920-1960","1961-1970","1971-1980","1981-1990","1991-1995",
               "1996-1999","2000-2005","2006-2010","2011-2015","2016-2020",
               "2021-2025","2026-Present"]

DUR_BINS   = [0, 60, 90, 120, 150, 180, float("inf")]
DUR_LABELS = ["<1h","1-1.5h","1.5-2h","2-2.5h","2.5-3h","3h+"]

STYLES = {
    "Trust IMDB": (0.70, 0.20, 0.10),
    "Balanced":   (0.40, 0.35, 0.25),
    "My Taste":   (0.15, 0.45, 0.40),
}


# ══════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════

def parse_bookmarks(html_bytes: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html_bytes.decode("utf-8", errors="replace"), "html.parser")

    def find_folder(parent, name):
        for h3 in parent.find_all("h3"):
            if h3.get_text(strip=True).lower() == name.lower():
                dl = h3.find_next_sibling("dl")
                if dl:
                    return dl
        return None

    by_years = None
    movies_dl = find_folder(soup, "Movies")
    if movies_dl:
        by_years = find_folder(movies_dl, "By Years")
    if not by_years:
        by_years = find_folder(soup, "By Years")
    if not by_years:
        by_years = soup

    links = by_years.find_all("a")
    rows  = []
    for a in links:
        raw = a.get_text(strip=False)
        url = a.get("href", "").strip()
        if not url.startswith("http"):
            continue

        score_m  = re.search(r"\((\d+)/10\)", raw)
        watched  = score_m is not None
        my_score = int(score_m.group(1)) if score_m else None
        clean    = re.split(r"=+|\(\d+/10\)", raw)[0].strip()

        year     = _first(re.findall(r"\b((?:19|20)\d{2})\b", clean))
        genre    = _first(re.findall(r"‧(.+?)‧", clean))
        duration = _first(re.findall(r"‧\s*(\d+h\s*\d*m|\d+\s*hours|\d+\s*mins)", clean))
        title    = re.sub(r"\(.*", "", clean).strip()

        if not title:
            continue

        # Build links
        imdb_url   = url if "imdb.com" in url else ""
        google_url = (
            f"https://www.google.com/search?q={quote_plus(title)}"
            + (f"+{year}" if year else "")
            + "+film"
        )

        rows.append({
            "Title":          title,
            "Release Year":   int(year) if year else None,
            "Genre":          genre or "",
            "Movie Duration": duration or "",
            "Watched":        watched,
            "My Score":       my_score,
            "IMDB Rating":    None,
            "Actors":         "",
            "IMDB URL":       imdb_url,
            "Google URL":     google_url,
        })

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["Title","Release Year"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _first(lst):
    return lst[0] if lst else None


# ══════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    genres       = df["Genre"].str.split("/", n=1, expand=True)
    df["genre1"] = genres[0].str.strip().str.lower().fillna("")
    df["genre2"] = (genres[1].str.strip().str.lower()
                    if 1 in genres.columns else df["genre1"])
    df["genre2"] = df["genre2"].fillna(df["genre1"])

    h  = df["Movie Duration"].str.extract(r"(\d+)\s*h",      expand=False).fillna(0).astype(int)
    m  = df["Movie Duration"].str.extract(r"(\d+)\s*m",      expand=False).fillna(0).astype(int)
    ho = df["Movie Duration"].str.extract(r"^(\d+)\s*hours", expand=False).fillna(0).astype(int)
    mo = df["Movie Duration"].str.extract(r"^(\d+)\s*mins",  expand=False).fillna(0).astype(int)
    df["duration_minutes"] = h*60 + m + ho*60 + mo

    df["Year Range"] = pd.cut(
        pd.to_numeric(df["Release Year"], errors="coerce"),
        bins=YEAR_BINS, labels=YEAR_LABELS, right=True
    ).astype(str)

    df["Duration Range"] = pd.cut(
        df["duration_minutes"], bins=DUR_BINS, labels=DUR_LABELS
    ).astype(str)

    return df


# ══════════════════════════════════════════════
#  OMDB FETCH
# ══════════════════════════════════════════════

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_cache(cache: dict):
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(CACHE_PATH)

def _extract_imdb_id(url: str):
    m = re.search(r"/(tt\d+)", url)
    return m.group(1) if m else None

def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _fetch_one_api(imdb_id: str) -> dict:
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"},
            timeout=8
        )
        data = r.json()
        if data.get("Response") == "True":
            return {
                "IMDB Rating": _safe_float(data.get("imdbRating")),
                "Actors":      data.get("Actors", ""),
            }
    except Exception:
        pass
    return {}

def fetch_omdb_background(session_id: str):
    """Runs in a background thread. Updates PROGRESS[session_id] as it goes."""
    with STORE_LOCK:
        df = STORE[session_id]["df"].copy()

    cache      = _load_cache()
    stats      = {"cached": 0, "fetched": 0, "failed": 0, "skipped": 0}
    results    = {}
    need_fetch = []

    for i, row in df.iterrows():
        imdb_id = _extract_imdb_id(row.get("IMDB URL", ""))
        if not imdb_id:
            stats["skipped"] += 1
            results[i] = {}
            continue
        if imdb_id in cache:
            results[i] = cache[imdb_id]
            stats["cached"] += 1
        else:
            need_fetch.append((i, imdb_id))

    total       = len(df)
    done_so_far = stats["cached"] + stats["skipped"]
    PROGRESS[session_id] = {"done": done_so_far, "total": total, "stats": stats.copy()}

    cache_lock = threading.Lock()

    def fetch_and_cache(task):
        idx, imdb_id = task
        data = _fetch_one_api(imdb_id)
        with cache_lock:
            cache[imdb_id] = data if data else {"IMDB Rating": None, "Actors": ""}
            _save_cache(cache)
        return idx, cache[imdb_id], bool(data)

    if need_fetch:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(fetch_and_cache, t): t for t in need_fetch}
            for future in as_completed(futures):
                idx, data, ok = future.result()
                results[idx] = data
                stats["fetched" if ok else "failed"] += 1
                done_so_far += 1
                PROGRESS[session_id] = {
                    "done": done_so_far, "total": total, "stats": stats.copy()
                }

    df["IMDB Rating"] = df.index.map(lambda i: results.get(i, {}).get("IMDB Rating"))
    df["Actors"]      = df.index.map(lambda i: results.get(i, {}).get("Actors", ""))

    with STORE_LOCK:
        STORE[session_id]["df"]     = df
        STORE[session_id]["fetched"] = True

    PROGRESS[session_id]["done"] = total  # signal complete


# ══════════════════════════════════════════════
#  TASTE PROFILE
# ══════════════════════════════════════════════

def build_taste_profile(df: pd.DataFrame) -> dict:
    rated = df[df["My Score"].notna() & df["Watched"]].copy()
    if rated.empty:
        return {}

    def norm_avg(scores):
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        rng = hi - lo or 1
        return {k: round((v - lo) / rng, 4) for k, v in scores.items()}

    genre_rows = pd.concat([
        rated[["My Score"]].assign(genre=rated["genre1"]),
        rated[["My Score"]].assign(genre=rated["genre2"]),
    ])
    raw_genre = {g: grp["My Score"].mean()
                 for g, grp in genre_rows.groupby("genre") if g}
    raw_era   = {str(e): grp["My Score"].mean()
                 for e, grp in rated.groupby("Year Range") if e and e != "nan"}
    raw_dur   = {str(d): grp["My Score"].mean()
                 for d, grp in rated.groupby("Duration Range") if d and d != "nan"}

    both = rated.dropna(subset=["IMDB Rating"])
    bias = float((both["My Score"] - both["IMDB Rating"]).mean()) if not both.empty else 0.0

    return {
        "genre":       norm_avg(raw_genre),
        "era":         norm_avg(raw_era),
        "duration":    norm_avg(raw_dur),
        "bias":        round(bias, 3),
        "rated_count": len(rated),
        "raw_genre":   {k: round(v,2) for k,v in raw_genre.items()},
        "raw_era":     {k: round(v,2) for k,v in raw_era.items()},
        "raw_dur":     {k: round(v,2) for k,v in raw_dur.items()},
    }


# ══════════════════════════════════════════════
#  ML PREDICTOR
# ══════════════════════════════════════════════

def train_predictor(df):
    if not ML_AVAILABLE:
        return None, None, None
    rated = df[df["My Score"].notna() & df["Watched"]].copy()
    if len(rated) < 30:
        return None, None, None

    feat = pd.get_dummies(rated[["genre1","genre2","Year Range","Duration Range"]])
    if "IMDB Rating" in rated.columns:
        feat["IMDB Rating"] = rated["IMDB Rating"].fillna(rated["IMDB Rating"].median())

    feat_cols = feat.columns.tolist()
    scaler    = StandardScaler()
    X         = scaler.fit_transform(feat.values)
    y         = rated["My Score"].values
    model     = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, scaler, feat_cols

def predict_scores(df, model, scaler, feat_cols):
    if model is None:
        return pd.Series([None] * len(df), index=df.index)
    feat = pd.get_dummies(df[["genre1","genre2","Year Range","Duration Range"]])
    if "IMDB Rating" in df.columns:
        med = df["IMDB Rating"].median()
        feat["IMDB Rating"] = df["IMDB Rating"].fillna(med if not pd.isna(med) else 7.0)
    feat  = feat.reindex(columns=feat_cols, fill_value=0)
    preds = np.clip(model.predict(scaler.transform(feat.values)), 1, 10)
    return pd.Series(preds.round(1), index=df.index)


# ══════════════════════════════════════════════
#  SCORING + FILTERING
# ══════════════════════════════════════════════

def score_and_filter(df, profile, predicted, style,
                     genre_override=None, era_override=None,
                     dur_override=None, actor_filter=None,
                     watched_filter="Both", top_n=100):
    result = df.copy()

    if watched_filter == "Unwatched only":
        result = result[~result["Watched"]]
    elif watched_filter == "Watched only":
        result = result[result["Watched"]]

    if genre_override:
        g = [x.lower().strip() for x in genre_override]
        result = result[result.apply(
            lambda r: any(gx in r["genre1"] or gx in r["genre2"] for gx in g), axis=1
        )]
    if era_override:
        result = result[result["Year Range"].isin(era_override)]
    if dur_override:
        result = result[result["Duration Range"].isin(dur_override)]
    if actor_filter and actor_filter.strip():
        result = result[
            result["Actors"].str.lower().str.contains(
                actor_filter.strip().lower(), na=False
            )
        ]

    if result.empty:
        return result

    w_imdb, w_taste, w_pred = STYLES.get(style, STYLES["Balanced"])

    imdb = result["IMDB Rating"].fillna(5.0)
    imdb_norm = (imdb - imdb.min()) / (imdb.max() - imdb.min() + 1e-9)

    def taste_match(row):
        if not profile:
            return 0.5
        g = max(
            profile.get("genre",{}).get(str(row.get("genre1","")), 0),
            profile.get("genre",{}).get(str(row.get("genre2","")), 0),
        )
        e = profile.get("era",     {}).get(str(row.get("Year Range","")),     0)
        d = profile.get("duration",{}).get(str(row.get("Duration Range","")), 0)
        return round((g + e + d) / 3, 4)

    taste = result.apply(taste_match, axis=1)

    pred_aligned = predicted.reindex(result.index).fillna(5.0)
    pred_norm    = (pred_aligned - 1) / 9

    result = result.copy()
    result["Composite Score"]    = (
        w_imdb * imdb_norm + w_taste * taste + w_pred * pred_norm
    ).round(4)
    result["Predicted My Score"] = predicted.reindex(result.index).round(1)
    result["Taste Match %"]      = (taste * 100).round(0).astype(int)

    return result.sort_values("Composite Score", ascending=False).head(top_n)


# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════

def get_sid():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    sid  = get_sid()
    f    = request.files.get("bookmarks")
    if not f:
        return redirect(url_for("index"))

    raw  = parse_bookmarks(f.read())
    df   = engineer_features(raw)

    with STORE_LOCK:
        STORE[sid] = {
            "df":      df,
            "fetched": False,
            "profile": {},
            "predicted": pd.Series(dtype=float),
            "model": None, "scaler": None, "feat_cols": None,
        }

    watched_n = int(df["Watched"].sum())
    rated_n   = int(df["My Score"].notna().sum())
    total_n   = len(df)

    return render_template("fetch.html",
                           total=total_n,
                           watched=watched_n,
                           rated=rated_n)


@app.route("/fetch/start", methods=["POST"])
def fetch_start():
    sid = get_sid()
    if sid not in STORE:
        return jsonify({"error": "session expired"}), 400
    PROGRESS[sid] = {"done": 0, "total": len(STORE[sid]["df"]), "stats": {}}
    t = threading.Thread(target=fetch_omdb_background, args=(sid,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/fetch/progress")
def fetch_progress():
    sid  = get_sid()
    prog = PROGRESS.get(sid, {"done": 0, "total": 1, "stats": {}})
    with STORE_LOCK:
        fetched = STORE.get(sid, {}).get("fetched", False)
    return jsonify({**prog, "complete": fetched})


@app.route("/suggest", methods=["GET", "POST"])
def suggest():
    sid = get_sid()
    if sid not in STORE:
        return redirect(url_for("index"))

    store = STORE[sid]
    df    = store["df"]

    # Build profile if not yet done
    if not store["profile"]:
        store["profile"]   = build_taste_profile(df)
        model, scaler, fc  = train_predictor(df)
        store["model"]     = model
        store["scaler"]    = scaler
        store["feat_cols"] = fc
        store["predicted"] = predict_scores(df, model, scaler, fc)

    profile   = store["profile"]
    predicted = store["predicted"]

    # Profile summary for display
    def top(d, n=3):
        return ", ".join(k for k,_ in sorted(d.items(), key=lambda x: -x[1])[:n])

    bias     = profile.get("bias", 0)
    bias_str = (f"+{bias:.1f}" if bias >= 0 else f"{bias:.1f}") + " vs IMDB"
    profile_summary = {
        "rated":  profile.get("rated_count", 0),
        "genres": top(profile.get("raw_genre", {})),
        "eras":   top(profile.get("raw_era",   {})),
        "durs":   top(profile.get("raw_dur",   {}), 2),
        "bias":   bias_str,
        "ml":     store["model"] is not None,
    }

    results = []
    if request.method == "POST":
        genre_raw = request.form.get("genre","").strip()
        genres    = [g.strip() for g in genre_raw.split(",") if g.strip()] or None
        eras      = request.form.getlist("era") or None
        durs      = request.form.getlist("dur") or None
        actor     = request.form.get("actor","").strip() or None
        style     = request.form.get("style","Balanced")
        watched   = request.form.get("watched","Both")

        res = score_and_filter(df, profile, predicted, style,
                               genre_override=genres,
                               era_override=eras,
                               dur_override=durs,
                               actor_filter=actor,
                               watched_filter=watched)

        for _, r in res.iterrows():
            title = r.get("Title","")
            year  = r.get("Release Year","")
            results.append({
                "title":      title,
                "year":       year,
                "genre":      r.get("Genre",""),
                "duration":   r.get("Movie Duration",""),
                "imdb":       r.get("IMDB Rating",""),
                "predicted":  r.get("Predicted My Score",""),
                "taste":      r.get("Taste Match %",""),
                "score":      r.get("Composite Score",""),
                "watched":    r.get("Watched", False),
                "my_score":   r.get("My Score",""),
                "actors":     r.get("Actors",""),
                "imdb_url":   r.get("IMDB URL",""),
                "google_url": r.get("Google URL",""),
            })

    watched_n = int(df["Watched"].sum())
    rated_n   = int(df["My Score"].notna().sum())

    return render_template("suggest.html",
                           profile=profile_summary,
                           year_labels=YEAR_LABELS,
                           dur_labels=DUR_LABELS,
                           styles=list(STYLES.keys()),
                           results=results,
                           total=len(df),
                           watched=watched_n,
                           rated=rated_n,
                           form=request.form)


@app.route("/reset")
def reset():
    sid = get_sid()
    with STORE_LOCK:
        STORE.pop(sid, None)
    PROGRESS.pop(sid, None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
