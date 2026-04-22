# -*- coding: utf-8 -*-
"""JZ's Cinematch — Flask Web App"""

import re, json, threading, uuid, os, time, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify)
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import r2_storage

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "jzmovies-dev-key-change-in-prod")

import traceback

@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    app.logger.error(f"500 error: {tb}")
    return f"<pre style='padding:20px;font-size:12px;'><b>500 Error — copy this and share it:</b>\n\n{tb}</pre>", 500


OMDB_API_KEY   = os.environ.get("OMDB_API_KEY", "")
TMDB_API_KEY   = os.environ.get("TMDB_API_KEY", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "cinematch")

STORE      = {}
STORE_LOCK = threading.Lock()
CACHE_PATH = Path("/tmp/omdb_cache.json")
PROGRESS   = {}

# ── App-level DB cache so we don't hit R2 on every page load ──────────────
_DB_CACHE      = None
_DB_CACHE_TIME = 0.0
_DB_CACHE_TTL  = 120  # seconds


def get_db() -> dict:
    global _DB_CACHE, _DB_CACHE_TIME
    if _DB_CACHE is None or time.time() - _DB_CACHE_TIME > _DB_CACHE_TTL:
        _DB_CACHE      = r2_storage.load_movies_db()
        _DB_CACHE_TIME = time.time()
    return _DB_CACHE

def invalidate_db():
    global _DB_CACHE
    _DB_CACHE = None


YEAR_BINS   = [1900, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
YEAR_LABELS = ["1900-1940", "1941-1950", "1951-1960", "1961-1970", "1971-1980",
               "1981-1990", "1991-2000", "2001-2010", "2011-2020", "2021-Present"]
DUR_BINS   = [0, 60, 90, 120, 150, 180, float("inf")]
DUR_LABELS = ["<1h", "1-1.5h", "1.5-2h", "2-2.5h", "2.5-3h", "3h+"]
STYLES = {
    "Trust IMDB": (0.70, 0.20, 0.10),
    "Balanced":   (0.40, 0.35, 0.25),
    "My Taste":   (0.15, 0.45, 0.40),
}

def movie_key(title, year):
    return f"{str(title).strip()}||{year or ''}"


# ══════════════════════════════════════════════
#  DATABASE ↔ DATAFRAME
# ══════════════════════════════════════════════

def movies_list_to_df(movies):
    if not movies: return pd.DataFrame()
    rows = []
    for m in movies:
        rows.append({
            "Title":          m.get("title", ""),
            "Release Year":   m.get("year"),
            "Genre":          m.get("genre", ""),
            "Movie Duration": m.get("duration", ""),
            "Watched":        m.get("watched", False),
            "My Score":       m.get("my_score"),
            "IMDB Rating":    m.get("imdb_rating"),
            "Actors":         m.get("actors", ""),
            "Cast":           m.get("cast", ""),
            "Director":       m.get("director", ""),
            "Plot":           m.get("plot", ""),
            "Language":       m.get("language", ""),
            "Poster URL":     m.get("poster_url", ""),
            "IMDB URL":       m.get("imdb_url", ""),
            "Google URL":     m.get("google_url", ""),
        })
    df = pd.DataFrame(rows)
    if "Release Year" in df.columns:
        df["Release Year"] = pd.to_numeric(df["Release Year"], errors="coerce")
    return df

def df_to_movies_list(df):
    movies = []
    for _, r in df.iterrows():
        year = r.get("Release Year")
        movies.append({
            "title":       r.get("Title", ""),
            "year":        int(year) if pd.notna(year) and year else None,
            "genre":       r.get("Genre", ""),
            "duration":    r.get("Movie Duration", ""),
            "watched":     bool(r.get("Watched", False)),
            "my_score":    r.get("My Score") if pd.notna(r.get("My Score", None)) else None,
            "imdb_rating": r.get("IMDB Rating") if pd.notna(r.get("IMDB Rating", None)) else None,
            "actors":      r.get("Actors", ""),
            "cast":        r.get("Cast", ""),
            "language":    r.get("Language", ""),
            "director":    r.get("Director", ""),
            "plot":        r.get("Plot", ""),
            "poster_url":  r.get("Poster URL", ""),
            "imdb_url":    r.get("IMDB URL", ""),
            "google_url":  r.get("Google URL", ""),
        })
    return movies


# ══════════════════════════════════════════════
#  PARSING (one-time import)
# ══════════════════════════════════════════════

def parse_bookmarks(html_bytes):
    soup = BeautifulSoup(html_bytes.decode("utf-8", errors="replace"), "html.parser")

    def find_folder(parent, name):
        for h3 in parent.find_all("h3"):
            if h3.get_text(strip=True).lower() == name.lower():
                dl = h3.find_next_sibling("dl")
                if dl: return dl
        return None

    by_years = None
    movies_dl = find_folder(soup, "Movies")
    if movies_dl: by_years = find_folder(movies_dl, "By Years")
    if not by_years: by_years = find_folder(soup, "By Years")
    if not by_years: by_years = soup

    rows = []
    for a in by_years.find_all("a"):
        raw = a.get_text(strip=False)
        url = a.get("href", "").strip()
        if not url.startswith("http"): continue
        score_m  = re.search(r"\((\d+)/10\)", raw)
        watched  = score_m is not None
        my_score = int(score_m.group(1)) if score_m else None
        clean    = re.split(r"=+|\(\d+/10\)", raw)[0].strip()
        year     = _first(re.findall(r"\b((?:19|20)\d{2})\b", clean))
        genre    = _first(re.findall(r"‧(.+?)‧", clean))
        duration = _first(re.findall(r"‧\s*(\d+h\s*\d*m|\d+\s*hours|\d+\s*mins)", clean))
        title    = re.sub(r"\(.*", "", clean).strip()
        if not title: continue
        imdb_url   = url if "imdb.com" in url else ""
        google_url = f"https://www.google.com/search?q={quote_plus(title + (' ' + year if year else '') + ' movie')}"
        rows.append({
            "Title": title, "Release Year": int(year) if year else None,
            "Genre": genre or "", "Movie Duration": duration or "",
            "Watched": watched, "My Score": my_score,
            "IMDB Rating": None, "Actors": "", "Director": "", "Plot": "",
            "IMDB URL": imdb_url, "Google URL": google_url,
        })

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["Title", "Release Year"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _first(lst): return lst[0] if lst else None


# ══════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════

def engineer_features(df):
    df = df.copy()
    genre_str    = df["Genre"].fillna("").str.replace(", ", "/", regex=False)
    genres       = genre_str.str.split("/", n=1, expand=True)
    df["genre1"] = genres[0].str.strip().str.lower().fillna("")
    df["genre2"] = (genres[1].str.strip().str.lower() if 1 in genres.columns else genres[0].str.strip().str.lower())
    df["genre2"] = df["genre2"].fillna(df["genre1"])
    dur = df["Movie Duration"].fillna("")
    h  = dur.str.extract(r"(\d+)\s*h",      expand=False).fillna(0).astype(int)
    m  = dur.str.extract(r"(\d+)\s*m",      expand=False).fillna(0).astype(int)
    ho = dur.str.extract(r"^(\d+)\s*hours", expand=False).fillna(0).astype(int)
    mo = dur.str.extract(r"^(\d+)\s*mins",  expand=False).fillna(0).astype(int)
    df["duration_minutes"] = h*60 + m + ho*60 + mo
    df["Year Range"] = pd.cut(pd.to_numeric(df["Release Year"], errors="coerce"),
                              bins=YEAR_BINS, labels=YEAR_LABELS, right=True).astype(str)
    df["Duration Range"] = pd.cut(df["duration_minutes"], bins=DUR_BINS, labels=DUR_LABELS).astype(str)
    return df

def extract_genres(df):
    genres = set()
    for col in ["genre1", "genre2"]:
        if col in df.columns:
            for g in df[col].dropna().unique():
                g = str(g).strip()
                if g and g != "nan": genres.add(g.title())
    return sorted(genres)

def fmt_genre(genre_str):
    """Add spaces around slashes for display: Crime/Drama → Crime / Drama"""
    if not genre_str: return ""
    return " / ".join(p.strip() for p in genre_str.split("/") if p.strip())


# ══════════════════════════════════════════════
#  OMDB
# ══════════════════════════════════════════════

def _warm_cache():
    if CACHE_PATH.exists(): return
    try:
        cloud = r2_storage.load_cache()
        if cloud:
            tmp = CACHE_PATH.with_suffix(".tmp")
            with open(tmp, "w") as f: json.dump(cloud, f)
            tmp.replace(CACHE_PATH)
    except Exception: pass

def _load_cache():
    _warm_cache()
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f: return json.load(f)
        except Exception: pass
    return {}

def _save_cache(cache):
    try:
        tmp = CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f: json.dump(cache, f)
        tmp.replace(CACHE_PATH)
    except Exception: pass

def _extract_imdb_id(url):
    m = re.search(r"/(tt\d+)", url)
    return m.group(1) if m else None

def _safe_float(val):
    try: return float(val)
    except: return None

def _fetch_one(imdb_id):
    if not OMDB_API_KEY: return {}
    try:
        r = requests.get("https://www.omdbapi.com/",
                         params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"}, timeout=8)
        d = r.json()
        if d.get("Response") == "True":
            actors = d.get("Actors", "")
            top6   = ", ".join(a.strip() for a in actors.split(",")[:6])
            # Pull first language only
            lang_raw = d.get("Language", "")
            language = lang_raw.split(",")[0].strip() if lang_raw else ""
            return {"IMDB Rating": _safe_float(d.get("imdbRating")),
                    "Actors": top6, "Director": d.get("Director",""),
                    "Plot": d.get("Plot",""), "Language": language}
    except Exception: pass
    return {}

def omdb_by_title(title: str, year=None) -> dict:
    """
    Fetch full movie details using title + year.
    Step 1: exact ?t= lookup.
    Step 2: if not found, fuzzy ?s= search — pick best year match.
    """
    if not OMDB_API_KEY: return {}

    def _parse_result(d):
        actors   = d.get("Actors", "")
        top6     = ", ".join(a.strip() for a in actors.split(",")[:6])
        lang_raw = d.get("Language", "")
        language = lang_raw.split(",")[0].strip() if lang_raw else ""
        imdb_id  = d.get("imdbID", "")
        return {
            "IMDB Rating": _safe_float(d.get("imdbRating")),
            "Actors":      top6,
            "Director":    d.get("Director", ""),
            "Plot":        d.get("Plot", ""),
            "Language":    language,
            "imdb_id":     imdb_id,
            "imdb_url":    f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "",
            "genre":       d.get("Genre", "").replace(", ", "/"),
            "duration":    d.get("Runtime", ""),
        }

    try:
        # Step 1: exact title lookup
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "short", "type": "movie"}
        if year: params["y"] = str(year)
        r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
        d = r.json()
        if d.get("Response") == "True":
            return _parse_result(d)

        # Step 2: fuzzy search fallback — filter to movies only, match year strictly
        params2 = {"apikey": OMDB_API_KEY, "s": title, "type": "movie"}
        r2 = requests.get("https://www.omdbapi.com/", params=params2, timeout=8)
        d2 = r2.json()
        if d2.get("Response") == "True":
            results = d2.get("Search", [])
            # Only keep entries that are actually movies (Type=movie) and year within 1 year
            movie_results = [x for x in results if x.get("Type","") == "movie"]
            if not movie_results:
                movie_results = results  # fallback if type not set

            best = None
            if year and movie_results:
                # Prefer exact year match
                exact = [x for x in movie_results
                         if str(x.get("Year",""))[:4] == str(year)]
                # Then within 1 year
                close = [x for x in movie_results
                         if abs(int((x.get("Year","0") or "0")[:4]) - int(year)) <= 1]
                if exact:
                    best = exact[0]
                elif close:
                    best = min(close, key=lambda x: abs(int((x.get("Year","0") or "0")[:4]) - int(year)))
                # Don't pick a result if year is off by more than 1
            elif movie_results:
                best = movie_results[0]

            if best:
                imdb_id = best.get("imdbID", "")
                if imdb_id:
                    r3 = requests.get("https://www.omdbapi.com/",
                        params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"},
                        timeout=8)
                    d3 = r3.json()
                    if d3.get("Response") == "True":
                        return _parse_result(d3)
    except Exception: pass
    return {}


def omdb_search(query, year=""):
    """Search OMDb by title — used by the Add Movie search tab."""
    if not OMDB_API_KEY: return []
    try:
        params = {"apikey": OMDB_API_KEY, "s": query, "type": "movie"}
        if year: params["y"] = year
        r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
        d = r.json()
        if d.get("Response") == "True": return d.get("Search", [])
    except Exception: pass
    return []


def omdb_details(imdb_id):
    if not OMDB_API_KEY: return {}
    try:
        r = requests.get("https://www.omdbapi.com/",
                         params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "full"}, timeout=8)
        d = r.json()
        if d.get("Response") == "True": return d
    except Exception: pass
    return {}


def tmdb_cast_and_poster(imdb_id: str) -> dict:
    """
    Fetch top-6 cast and poster URL from TMDb using an IMDb ID.
    Returns {"cast": "Actor1, Actor2, ...", "poster_url": "https://..."} or {}.
    """
    if not TMDB_API_KEY or not imdb_id:
        return {}
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/find/{imdb_id}",
            params={"api_key": TMDB_API_KEY, "external_source": "imdb_id"},
            timeout=8
        )
        results = r.json().get("movie_results", [])
        if not results:
            return {}
        tmdb_id      = results[0]["id"]
        poster_path  = results[0].get("poster_path", "")

        cr = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits",
            params={"api_key": TMDB_API_KEY},
            timeout=8
        )
        cast_list = cr.json().get("cast", [])
        top6 = ", ".join(c["name"] for c in cast_list[:8])

        # Language from TMDb
        lang_code = results[0].get("original_language", "")
        lang_map  = {
            "en": "English", "fr": "French", "de": "German", "es": "Spanish",
            "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
            "ru": "Russian", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
            "sv": "Swedish", "da": "Danish", "nl": "Dutch", "pl": "Polish",
            "fi": "Finnish", "no": "Norwegian", "tr": "Turkish", "he": "Hebrew",
            "hu": "Hungarian", "cs": "Czech", "ro": "Romanian", "uk": "Ukrainian",
        }
        language = lang_map.get(lang_code, lang_code.upper() if lang_code else "")

        return {
            "cast":       top6,
            "language":   language,
            "poster_url": f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else "",
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════

def fetch_omdb_background(session_id):
    with STORE_LOCK: df = STORE[session_id]["df"].copy()
    cache = _load_cache()
    stats = {"cached": 0, "fetched": 0, "failed": 0, "skipped": 0}
    results, need_fetch = {}, []

    for i, row in df.iterrows():
        imdb_id = _extract_imdb_id(row.get("IMDB URL", ""))
        if not imdb_id: stats["skipped"] += 1; results[i] = {}; continue
        if imdb_id in cache: results[i] = cache[imdb_id]; stats["cached"] += 1
        else: need_fetch.append((i, imdb_id))

    total = len(df); done_so_far = stats["cached"] + stats["skipped"]
    PROGRESS[session_id] = {"done": done_so_far, "total": total, "stats": stats.copy()}
    cache_lock = threading.Lock()

    def fetch_and_cache(task):
        idx, imdb_id = task
        data = _fetch_one(imdb_id)
        with cache_lock:
            cache[imdb_id] = data if data else {"IMDB Rating": None, "Actors": "", "Director": "", "Plot": ""}
            _save_cache(cache)
        return idx, cache[imdb_id], bool(data)

    if need_fetch:
        fetch_count = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_and_cache, t): t for t in need_fetch}
            for future in as_completed(futures):
                try:
                    idx, data, ok = future.result()
                    results[idx] = data
                    stats["fetched" if ok else "failed"] += 1
                    done_so_far += 1; fetch_count += 1
                    PROGRESS[session_id] = {"done": done_so_far, "total": total, "stats": stats.copy()}
                    if fetch_count % 50 == 0: r2_storage.save_cache(cache)
                except Exception: done_so_far += 1

    r2_storage.save_cache(cache)
    df["IMDB Rating"] = df.index.map(lambda i: results.get(i, {}).get("IMDB Rating"))
    df["Actors"]      = df.index.map(lambda i: results.get(i, {}).get("Actors", ""))
    df["Director"]    = df.index.map(lambda i: results.get(i, {}).get("Director", ""))
    df["Plot"]        = df.index.map(lambda i: results.get(i, {}).get("Plot", ""))

    with STORE_LOCK:
        STORE[session_id]["df"]      = df
        STORE[session_id]["fetched"] = True
    PROGRESS[session_id]["done"] = total


# ══════════════════════════════════════════════
#  TASTE PROFILE & ML
# ══════════════════════════════════════════════

def build_taste_profile(df):
    try:
        rated = df[df["My Score"].notna() & df["Watched"]].copy()
        if rated.empty: return {}

        # Global mean — the prior we shrink toward for small samples
        global_mean = rated["My Score"].mean()

        def bayesian_avg(groups, C=None):
            """
            Bayesian average: (C * global_mean + n * group_mean) / (C + n)
            C = confidence weight = avg group size (floor 3, ceil 15).
            Small groups get pulled toward global_mean; large groups keep their average.
            """
            if not groups: return {}
            counts = {k: len(v) for k, v in groups.items()}
            means  = {k: v["My Score"].mean() for k, v in groups.items()}
            if C is None:
                avg_count = sum(counts.values()) / len(counts)
                C = max(3, min(15, round(avg_count)))
            result = {}
            for k in groups:
                n   = counts[k]
                mu  = means[k]
                result[k] = (C * global_mean + n * mu) / (C + n)
            return result

        def norm_avg(scores):
            """Normalize a dict of scores to [0, 1]."""
            if not scores: return {}
            lo, hi = min(scores.values()), max(scores.values())
            rng = hi - lo or 1
            return {k: round((v - lo) / rng, 4) for k, v in scores.items()}

        # Build groups
        genre_rows = pd.concat([
            rated[["My Score"]].assign(genre=rated["genre1"]),
            rated[["My Score"]].assign(genre=rated["genre2"]),
        ])
        genre_groups = {g: grp for g, grp in genre_rows.groupby("genre") if g and g != "nan"}
        era_groups   = {str(e): grp for e, grp in rated.groupby("Year Range") if e and str(e) != "nan"}
        dur_groups   = {str(d): grp for d, grp in rated.groupby("Duration Range") if d and str(d) != "nan"}

        # Bayesian averages
        bay_genre = bayesian_avg(genre_groups)
        bay_era   = bayesian_avg(era_groups)
        bay_dur   = bayesian_avg(dur_groups)

        # Raw (simple) averages for display — so the UI shows honest numbers
        raw_genre = {k: round(grp["My Score"].mean(), 2) for k, grp in genre_groups.items()}
        raw_era   = {k: round(grp["My Score"].mean(), 2) for k, grp in era_groups.items()}
        raw_dur   = {k: round(grp["My Score"].mean(), 2) for k, grp in dur_groups.items()}

        # Count per category for display
        cnt_genre = {k: len(v) for k, v in genre_groups.items()}
        cnt_era   = {k: len(v) for k, v in era_groups.items()}

        both = rated.dropna(subset=["IMDB Rating"])
        bias = float((both["My Score"] - both["IMDB Rating"]).mean()) if not both.empty else 0.0

        return {
            # Normalized Bayesian scores — used for taste matching
            "genre":    norm_avg(bay_genre),
            "era":      norm_avg(bay_era),
            "duration": norm_avg(bay_dur),
            # Display values (simple averages)
            "raw_genre": raw_genre,
            "raw_era":   raw_era,
            "raw_dur":   raw_dur,
            # Counts
            "cnt_genre": cnt_genre,
            "cnt_era":   cnt_era,
            "bias":          round(bias, 3),
            "rated_count":   len(rated),
            "global_mean":   round(global_mean, 2),
        }
    except Exception: return {}

def train_predictor(df):
    try:
        if not ML_AVAILABLE: return None, None, None
        rated = df[df["My Score"].notna() & df["Watched"]].copy()
        if len(rated) < 30: return None, None, None
        feat = pd.get_dummies(rated[["genre1","genre2","Year Range","Duration Range"]])
        med = rated["IMDB Rating"].median()
        feat["IMDB Rating"] = rated["IMDB Rating"].fillna(med if pd.notna(med) else 7.0)
        feat_cols = feat.columns.tolist()
        scaler = StandardScaler(); model = Ridge(alpha=1.0)
        model.fit(scaler.fit_transform(feat.values), rated["My Score"].values)
        return model, scaler, feat_cols
    except Exception: return None, None, None

def predict_scores(df, model, scaler, feat_cols):
    try:
        if model is None: return pd.Series([None]*len(df), index=df.index)
        feat = pd.get_dummies(df[["genre1","genre2","Year Range","Duration Range"]])
        med = df["IMDB Rating"].median()
        feat["IMDB Rating"] = df["IMDB Rating"].fillna(med if pd.notna(med) else 7.0)
        feat = feat.reindex(columns=feat_cols, fill_value=0)
        preds = np.clip(model.predict(scaler.transform(feat.values)), 1, 10)
        return pd.Series(preds.round(1), index=df.index)
    except Exception: return pd.Series([None]*len(df), index=df.index)


# ══════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════

def score_and_filter(df, profile, predicted, style,
                     genre_override=None, era_override=None,
                     dur_override=None, person_filter=None,
                     watched_filter="Both", top_n=100):
    res = df.copy()
    if watched_filter == "Unwatched only": res = res[~res["Watched"]]
    elif watched_filter == "Watched only":  res = res[res["Watched"]]
    if genre_override:
        g = [x.lower().strip() for x in genre_override]
        res = res[res.apply(lambda r: any(gx in r["genre1"] or gx in r["genre2"] for gx in g), axis=1)]
    if era_override:   res = res[res["Year Range"].isin(era_override)]
    if dur_override:   res = res[res["Duration Range"].isin(dur_override)]
    if person_filter and person_filter.strip():
        q = person_filter.strip().lower()
        actors_col   = res["Actors"].fillna("").str.lower()
        director_col = res["Director"].fillna("").str.lower()
        res = res[actors_col.str.contains(q, na=False) | director_col.str.contains(q, na=False)]
    if res.empty: return res
    w_imdb, w_taste, w_pred = STYLES.get(style, STYLES["Balanced"])
    imdb = res["IMDB Rating"].fillna(5.0)
    imdb_norm = (imdb - imdb.min()) / (imdb.max() - imdb.min() + 1e-9)
    def taste_match(row):
        if not profile: return 0.5
        g = max(profile.get("genre",{}).get(str(row.get("genre1","")),0),
                profile.get("genre",{}).get(str(row.get("genre2","")),0))
        e = profile.get("era",    {}).get(str(row.get("Year Range","")),     0)
        d = profile.get("duration",{}).get(str(row.get("Duration Range","")),0)
        return round((g+e+d)/3, 4)
    taste     = res.apply(taste_match, axis=1)
    pred_norm = (predicted.reindex(res.index).fillna(5.0) - 1) / 9
    res = res.copy()
    res["Composite Score"]    = (w_imdb*imdb_norm + w_taste*taste + w_pred*pred_norm).round(4)
    res["Predicted My Score"] = predicted.reindex(res.index).round(1)
    res["Taste Match %"]      = (taste * 100).round(0).astype(int)
    return res.sort_values("Composite Score", ascending=False).head(top_n)


# ══════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════

def get_sid():
    if "sid" not in session: session["sid"] = str(uuid.uuid4())
    return session["sid"]

def load_store_from_db(sid):
    db = get_db()
    if not db or not db.get("movies"): return False
    df = movies_list_to_df(db["movies"])
    df = engineer_features(df)
    with STORE_LOCK:
        STORE[sid] = {"df": df, "fetched": True, "profile": {},
                      "predicted": pd.Series(dtype=float),
                      "model": None, "scaler": None, "feat_cols": None}
    return True

def get_df(sid): return STORE.get(sid, {}).get("df")

def invalidate_store(sid):
    with STORE_LOCK: STORE.pop(sid, None)
    invalidate_db()


# ══════════════════════════════════════════════
#  BUILD RESULT DICT (shared between suggest & top)
# ══════════════════════════════════════════════

def row_to_dict(r):
    yr = r.get("Release Year")
    return {
        "title":      r.get("Title",""),
        "year":       int(yr) if pd.notna(yr) and yr else None,
        "genre":      fmt_genre(r.get("Genre","")),
        "duration":   r.get("Movie Duration",""),
        "imdb":       r.get("IMDB Rating",""),
        "watched":    r.get("Watched",False),
        "my_score":   r.get("My Score",""),
        "actors":     r.get("Actors",""),
        "cast":       r.get("Cast",""),
        "language":   r.get("Language",""),
        "director":   r.get("Director",""),
        "plot":       r.get("Plot",""),
        "poster_url": r.get("Poster URL",""),
        "imdb_url":   r.get("IMDB URL",""),
        "google_url": r.get("Google URL",""),
    }


# ══════════════════════════════════════════════
#  ROUTES — MAIN
# ══════════════════════════════════════════════

@app.route("/")
def index():
    sid = get_sid()
    if r2_storage.movies_db_exists():
        load_store_from_db(sid)
        df = get_df(sid)
        stats = {"total": len(df), "rated": int(df["My Score"].notna().sum())} if df is not None else {}
        return render_template("index.html", has_db=True, stats=stats)
    return render_template("index.html", has_db=False)


@app.route("/suggest", methods=["GET", "POST"])
def suggest():
    sid = get_sid()
    if sid not in STORE:
        if not load_store_from_db(sid): return redirect(url_for("index"))
    store = STORE[sid]; df = store["df"]
    try:
        if not store["profile"]:
            store["profile"] = build_taste_profile(df)
            m, sc, fc = train_predictor(df)
            store["model"] = m; store["scaler"] = sc; store["feat_cols"] = fc
            store["predicted"] = predict_scores(df, m, sc, fc)
    except Exception:
        store["profile"] = {}; store["predicted"] = pd.Series(dtype=float)

    profile = store["profile"]; predicted = store["predicted"]
    def top(d, n=3): return ", ".join(k for k,_ in sorted(d.items(), key=lambda x:-x[1])[:n])
    bias = profile.get("bias", 0)
    # For genres, sort by Bayesian-weighted score (already in "genre" key normalized)
    # but show by raw average for display
    raw_genre = profile.get("raw_genre", {})
    cnt_genre = profile.get("cnt_genre", {})
    bay_genre = profile.get("genre", {})  # normalized Bayesian scores, lowercase keys
    # Sort by Bayesian score so single-movie genres don't dominate
    top_genres_list = sorted(bay_genre.items(), key=lambda x: -x[1])[:3]
    top_genres = ", ".join(
        f"{k.title()} ({cnt_genre.get(k, 0)})" for k, _ in top_genres_list
    )
    profile_summary = {
        "rated":       profile.get("rated_count", 0),
        "genres":      top_genres,
        "eras":        top(profile.get("raw_era", {})),
        "durs":        top(profile.get("raw_dur", {}), 2),
        "bias":        (f"+{bias:.1f}" if bias >= 0 else f"{bias:.1f}") + " vs IMDb",
        "ml":          store["model"] is not None,
        "global_mean": profile.get("global_mean", 0),
    }
    all_genres = extract_genres(df)
    results = []
    if request.method == "POST":
        try:
            res = score_and_filter(df, profile, predicted,
                                   request.form.get("style","Balanced"),
                                   genre_override=request.form.getlist("genre") or None,
                                   era_override=request.form.getlist("era") or None,
                                   dur_override=request.form.getlist("dur") or None,
                                   person_filter=request.form.get("actor","").strip() or None,
                                   watched_filter=request.form.get("watched","Both"))
            for _, r in res.iterrows():
                d = row_to_dict(r)
                d["predicted"] = r.get("Predicted My Score","")
                d["taste"]     = r.get("Taste Match %","")
                d["score"]     = r.get("Composite Score","")
                results.append(d)
        except Exception: pass

    return render_template("suggest.html", profile=profile_summary, year_labels=YEAR_LABELS,
                           dur_labels=DUR_LABELS, styles=list(STYLES.keys()),
                           all_genres=all_genres, results=results, total=len(df),
                           rated=int(df["My Score"].notna().sum()), form=request.form)


@app.route("/top")
def top_movies():
    sid = get_sid()
    if sid not in STORE:
        if not load_store_from_db(sid): return redirect(url_for("index"))
    df = STORE[sid]["df"]
    db = get_db()
    top10_keys = db.get("top10", [])
    top10_rank = {k: i+1 for i, k in enumerate(top10_keys)}
    rated = df[df["My Score"].notna() & df["Watched"]].sort_values("My Score", ascending=False)
    movies = []
    for _, r in rated.iterrows():
        d = row_to_dict(r)
        yr = d["year"]
        k = movie_key(d["title"], yr)
        d["top10_rank"] = top10_rank.get(k)
        d["cast"]       = r.get("Cast", "")
        d["language"]   = r.get("Language", "")
        d["poster_url"] = r.get("Poster URL", "")
        movies.append(d)
    top10  = sorted([m for m in movies if m["top10_rank"]], key=lambda x: x["top10_rank"])
    others = [m for m in movies if not m["top10_rank"]]
    return render_template("top.html", movies=top10+others, total_rated=len(movies))



@app.route("/api/watched_stats")
def watched_stats():
    """Genre and era breakdown for watched/reviewed movies."""
    sid = get_sid()
    if sid not in STORE:
        if not load_store_from_db(sid):
            return jsonify({"genres": [], "eras": []})
    df = STORE[sid]["df"]
    reviewed = df[df["My Score"].notna() & df["Watched"]].copy()
    if reviewed.empty:
        return jsonify({"genres": [], "eras": []})

    # Genre stats
    genre_rows = []
    for col in ["genre1", "genre2"]:
        for _, row in reviewed.iterrows():
            g = str(row.get(col, "")).strip()
            if g and g != "nan":
                genre_rows.append({"genre": g.title(), "score": row["My Score"]})
    gdf = pd.DataFrame(genre_rows)
    genre_stats = []
    if not gdf.empty:
        for g, grp in gdf.groupby("genre"):
            genre_stats.append({
                "name":  g,
                "count": len(grp),
                "avg":   round(grp["score"].mean(), 2)
            })
        genre_stats.sort(key=lambda x: -x["count"])

    # Era stats
    era_stats = []
    for era, grp in reviewed.groupby("Year Range"):
        if era and era != "nan":
            era_stats.append({
                "name":  str(era),
                "count": len(grp),
                "avg":   round(grp["My Score"].mean(), 2)
            })
    # Sort by era label order
    era_order = YEAR_LABELS
    era_stats.sort(key=lambda x: era_order.index(x["name"]) if x["name"] in era_order else 99)

    return jsonify({"genres": genre_stats, "eras": era_stats,
                    "total_reviewed": len(reviewed)})

@app.route("/reset")
def reset():
    sid = get_sid(); invalidate_store(sid)
    return redirect(url_for("index"))


# ══════════════════════════════════════════════
#  IMPORT
# ══════════════════════════════════════════════

@app.route("/import", methods=["GET", "POST"])
def do_import():
    if r2_storage.movies_db_exists(): return redirect(url_for("index"))
    message = None
    if request.method == "POST":
        file = request.files.get("bookmarks_file")
        if not file or file.filename == "":          message = "No file selected."
        elif not file.filename.lower().endswith((".html",".htm")): message = f"'{file.filename}' doesn't look like an HTML file."
        else:
            file_bytes = file.read()
            if b"<" not in file_bytes: message = "File doesn't appear to be valid HTML."
            else:
                sid = get_sid(); raw = parse_bookmarks(file_bytes); df = engineer_features(raw)
                with STORE_LOCK:
                    STORE[sid] = {"df":df,"fetched":False,"profile":{},
                                  "predicted":pd.Series(dtype=float),"model":None,
                                  "scaler":None,"feat_cols":None,"import_mode":True}
                return render_template("fetch.html", total=len(df),
                                       watched=int(df["Watched"].sum()),
                                       rated=int(df["My Score"].notna().sum()),
                                       omdb_configured=bool(OMDB_API_KEY), import_mode=True)
    return render_template("import.html", message=message)


@app.route("/import/finalize")
def import_finalize():
    sid = get_sid()
    with STORE_LOCK: store = STORE.get(sid, {})
    if not store or not store.get("fetched"): return redirect(url_for("do_import"))
    db = {"movies": df_to_movies_list(store["df"]), "version": 1}
    r2_storage.save_movies_db(db); invalidate_db()
    return redirect(url_for("index"))


@app.route("/fetch/start", methods=["POST"])
def fetch_start():
    sid = get_sid()
    if sid not in STORE: return jsonify({"error": "session expired"}), 400
    PROGRESS[sid] = {"done": 0, "total": len(STORE[sid]["df"]), "stats": {}}
    threading.Thread(target=fetch_omdb_background, args=(sid,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/fetch/progress")
def fetch_progress():
    sid  = get_sid()
    prog = PROGRESS.get(sid, {"done": 0, "total": 1, "stats": {}})
    with STORE_LOCK: fetched = STORE.get(sid, {}).get("fetched", False)
    import_mode = STORE.get(sid, {}).get("import_mode", False)
    return jsonify({**prog, "complete": fetched, "import_mode": import_mode})


# ══════════════════════════════════════════════
#  ADMIN
# ══════════════════════════════════════════════

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin_login", next=request.url))
        return f(*args, **kwargs)
    return decorated

@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("password","") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(request.args.get("next") or url_for("add_movie"))
        error = "Incorrect password."
    return render_template("login.html", error=error)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None); return redirect(url_for("index"))


# ══════════════════════════════════════════════
#  ADD / EDIT
# ══════════════════════════════════════════════


@app.route("/admin/debug-add")
@admin_required
def admin_debug_add():
    """Confirms admin session and tests search API."""
    test = None
    if OMDB_API_KEY:
        try:
            r = requests.get("https://www.omdbapi.com/",
                params={"apikey": OMDB_API_KEY, "s": "Godfather", "type": "movie"},
                timeout=8)
            test = r.json()
        except Exception as e:
            test = {"error": str(e)}
    return jsonify({
        "admin_session":  session.get("admin", False),
        "omdb_key_set":   bool(OMDB_API_KEY),
        "omdb_key_chars": len(OMDB_API_KEY),
        "omdb_test":      test,
    })


@app.route("/debug-search")
def debug_search():
    """Open debug endpoint - tests search without auth requirement."""
    q = request.args.get("q", "Godfather")
    results = []
    error = None
    omdb_raw = None
    if OMDB_API_KEY:
        try:
            r = requests.get("https://www.omdbapi.com/",
                params={"apikey": OMDB_API_KEY, "s": q, "type": "movie"},
                timeout=8)
            omdb_raw = r.json()
            if omdb_raw.get("Response") == "True":
                results = omdb_raw.get("Search", [])
        except Exception as e:
            error = str(e)
    return jsonify({
        "admin_in_session": session.get("admin", False),
        "omdb_key_set": bool(OMDB_API_KEY),
        "query": q,
        "omdb_raw": omdb_raw,
        "result_count": len(results),
        "error": error,
    })

@app.route("/debug-all-movies")
def debug_all_movies():
    """Open debug endpoint - returns movie count without auth."""
    sid = get_sid()
    if sid not in STORE:
        loaded = load_store_from_db(sid)
    else:
        loaded = True
    df = get_df(sid)
    return jsonify({
        "admin_in_session": session.get("admin", False),
        "store_loaded": loaded,
        "movie_count": len(df) if df is not None else 0,
        "sample": [{"title": r["Title"], "year": r.get("Release Year")}
                   for _, r in df.head(3).iterrows()] if df is not None else [],
    })

@app.route("/add")
@admin_required
def add_movie():
    sid = get_sid()
    if sid not in STORE: load_store_from_db(sid)
    return render_template("add.html")


@app.route("/api/search_movie")
@admin_required
def api_search_movie():
    q = request.args.get("q","").strip(); year = request.args.get("year","").strip()
    if not q: return jsonify({"results":[], "error":"No query provided"})
    if not OMDB_API_KEY: return jsonify({"results":[], "error":"OMDB_API_KEY not configured in Railway environment variables."})
    results = omdb_search(q, year)
    return jsonify({"results": results})


@app.route("/api/movie_details")
@admin_required
def api_movie_details():
    imdb_id = request.args.get("id","").strip()
    if not imdb_id: return jsonify({"error":"No ID"})
    if not OMDB_API_KEY: return jsonify({"error":"OMDB_API_KEY not configured."})
    data = omdb_details(imdb_id)
    if not data: return jsonify({"error":"Movie not found"})
    actors = data.get("Actors","")
    top6   = ", ".join(a.strip() for a in actors.split(",")[:6])
    title  = data.get("Title",""); year = data.get("Year","")
    return jsonify({
        "title": title, "year": year,
        "genre": data.get("Genre","").replace(", ","/"),
        "duration": data.get("Runtime",""),
        "imdb_id": data.get("imdbID",""),
        "imdb_rating": data.get("imdbRating",""),
        "imdb_url": f"https://www.imdb.com/title/{data.get('imdbID','')}/",
        "google_url": f"https://www.google.com/search?q={quote_plus(title+' '+year+' movie')}",
        "actors": top6, "director": data.get("Director",""),
        "plot": data.get("Plot",""), "poster": data.get("Poster",""),
        "imdb_id_val": data.get("imdbID",""),
        "poster_url": "",  # will be filled by TMDb refresh
        "cast": "",        # will be filled by TMDb refresh
    })


@app.route("/api/save_movie", methods=["POST"])
@admin_required
def api_save_movie():
    try:
        data = request.get_json(); sid = get_sid()
        db = get_db(); movies = db.get("movies",[])[:]
        title_lower = data.get("title","").lower().strip()
        year_val    = data.get("year")
        for existing in movies:
            if (existing.get("title","").lower().strip() == title_lower and
                    str(existing.get("year","")) == str(year_val)):
                return jsonify({"success": False, "error": "Movie already in your list."})
        runtime = data.get("duration",""); dur_fmt = runtime
        m = re.match(r"^(\d+)\s*min", runtime)
        if m:
            mins = int(m.group(1)); h, mn = divmod(mins, 60)
            dur_fmt = f"{h}h {mn}m" if h > 0 else f"{mn}m"
        genre = data.get("genre","").replace(", ","/")
        movies.append({
            "title": data.get("title",""), "year": int(year_val) if year_val else None,
            "genre": genre, "duration": dur_fmt,
            "watched": bool(data.get("watched",False)),
            "my_score": data.get("my_score"),
            "imdb_rating": _safe_float(data.get("imdb_rating")),
            "imdb_id": data.get("imdb_id",""), "imdb_url": data.get("imdb_url",""),
            "google_url": data.get("google_url",""), "actors": data.get("actors",""),
            "cast": data.get("cast",""), "poster_url": data.get("poster_url",""),
            "director": data.get("director",""), "plot": data.get("plot",""), "source":"manual",
        })
        db2 = dict(db); db2["movies"] = movies
        r2_storage.save_movies_db(db2); invalidate_store(sid)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})


@app.route("/api/edit_movie", methods=["POST"])
@admin_required
def api_edit_movie():
    try:
        data = request.get_json(); sid = get_sid()
        db = get_db(); movies = db.get("movies",[])[:]
        # Find by original title+year
        orig_title = data.get("orig_title","").lower().strip()
        orig_year  = str(data.get("orig_year") or "").strip()
        updated = False
        for m in movies:
            m_year  = str(m.get("year") or "").strip()
            m_title = m.get("title","").lower().strip()
            # Match on title (normalized) + year, OR just title if year is missing
            title_match = m_title == orig_title
            year_match  = m_year == orig_year or not orig_year or not m_year
            if title_match and year_match:
                if "my_score"  in data: m["my_score"]  = data["my_score"]
                if "watched"   in data: m["watched"]   = bool(data["watched"])
                if "genre"     in data: m["genre"]     = data["genre"].replace(", ","/")
                if "duration"  in data: m["duration"]  = data["duration"]
                if "title"     in data and data["title"].strip(): m["title"] = data["title"].strip()
                if "year"      in data and data["year"]: m["year"] = int(data["year"]) if str(data["year"]).isdigit() else m["year"]
                # Regenerate google_url if title/year changed
                if "title" in data or "year" in data:
                    t = m.get("title",""); y = str(m.get("year",""))
                    m["google_url"] = f"https://www.google.com/search?q={quote_plus(t+' '+y+' movie')}"
                updated = True; break
        if not updated: return jsonify({"success": False, "error": "Movie not found"})
        db2 = dict(db); db2["movies"] = movies
        r2_storage.save_movies_db(db2); invalidate_store(sid)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})


@app.route("/api/rated_movies")
@admin_required
def api_rated_movies():
    sid = get_sid()
    if sid not in STORE: load_store_from_db(sid)
    df = get_df(sid)
    if df is None: return jsonify({"movies": []})
    rated = df[df["Watched"]].sort_values(["My Score","Title"], ascending=[False,True], na_position="last")
    movies = []
    for _, r in rated.iterrows():
        yr = r.get("Release Year")
        yr_int = int(yr) if pd.notna(yr) and yr else None
        movies.append({"key": movie_key(r.get("Title",""), yr_int),
                       "title": r.get("Title",""), "year": yr_int,
                       "my_score": r.get("My Score"), "genre": r.get("Genre","")})
    return jsonify({"movies": movies})


@app.route("/api/all_movies")
@admin_required
def api_all_movies():
    sid = get_sid()
    if sid not in STORE: load_store_from_db(sid)
    df = get_df(sid)
    if df is None: return jsonify({"movies": []})
    movies = []
    for _, r in df.sort_values("Title").iterrows():
        yr = r.get("Release Year")
        yr_int = int(yr) if pd.notna(yr) and yr else None
        movies.append({
            "title":    r.get("Title",""), "year": yr_int,
            "genre":    r.get("Genre",""), "duration": r.get("Movie Duration",""),
            "watched":  bool(r.get("Watched",False)),
            "my_score": r.get("My Score") if pd.notna(r.get("My Score",None)) else None,
            "imdb":     r.get("IMDB Rating",""),
        })
    return jsonify({"movies": movies})


@app.route("/api/top10", methods=["GET"])
@admin_required
def get_top10():
    db = get_db()
    return jsonify({"top10": db.get("top10",[])})

@app.route("/api/top10", methods=["POST"])
@admin_required
def save_top10():
    try:
        data = request.get_json(); sid = get_sid()
        db = get_db(); db2 = dict(db); db2["top10"] = data.get("top10",[])
        r2_storage.save_movies_db(db2); invalidate_store(sid)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})


# ══════════════════════════════════════════════
#  ADMIN — REFRESH OMDB DATA
# ══════════════════════════════════════════════

REFRESH_STATUS = {"running": False, "done": 0, "total": 0,
                  "from_cache": 0, "from_api": 0, "complete": False, "error": ""}


def _needs_refresh(m):
    r = m.get("imdb_rating")
    return r is None or r == "" or r == 0 or str(r).strip() in ("", "N/A", "0", "0.0")

def _refresh_omdb_background():
    global REFRESH_STATUS
    try:
        db     = r2_storage.load_movies_db()
        movies = db.get("movies", [])
        cache  = _load_cache()

        missing = [m for m in movies if _needs_refresh(m)]

        REFRESH_STATUS.update({
            "running": True, "done": 0, "total": len(missing),
            "from_cache": 0, "from_api": 0, "complete": False, "error": ""
        })

        from_cache = 0
        from_api   = 0

        for i, m in enumerate(missing):
            imdb_id = _extract_imdb_id(m.get("imdb_url", ""))
            data    = None

            # ── Try cache by IMDB id ─────────────────────────────────────────
            if imdb_id and imdb_id in cache:
                cached = cache[imdb_id]
                if cached.get("IMDB Rating") not in (None, "", "N/A"):
                    data = cached
                    from_cache += 1

            # ── Try cache by title key (tt-less) ────────────────────────────
            if data is None:
                title_key = f"title:{m.get('title','').lower()}:{m.get('year','')}"
                if title_key in cache and cache[title_key].get("IMDB Rating") not in (None, "", "N/A"):
                    data = cache[title_key]
                    from_cache += 1

            # ── API: direct title+year lookup (1 call, full details) ─────────
            if data is None and OMDB_API_KEY:
                fetched = omdb_by_title(m.get("title", ""), m.get("year"))
                if fetched and fetched.get("IMDB Rating") not in (None, "", "N/A"):
                    # Store under both the imdb_id key and a title key
                    fid = fetched.get("imdb_id", "")
                    if fid:
                        cache[fid] = fetched
                    cache[f"title:{m.get('title','').lower()}:{m.get('year','')}"] = fetched
                    _save_cache(cache)
                    data = fetched
                    from_api += 1

            # ── Apply to movie record ────────────────────────────────────────
            if data:
                m["imdb_rating"] = data.get("IMDB Rating")
                m["actors"]      = data.get("Actors", "")
                m["director"]    = data.get("Director", "")
                m["plot"]        = data.get("Plot", "")
                if data.get("Language"): m["language"] = data["Language"]
                if not m.get("imdb_url") and data.get("imdb_url"):
                    m["imdb_url"] = data["imdb_url"]
                # Also fill genre/duration if they were empty
                if not m.get("genre") and data.get("genre"):
                    m["genre"] = data["genre"]
                if not m.get("duration") and data.get("duration"):
                    m["duration"] = data["duration"]

            # ── TMDb: cast + poster ──────────────────────────────────────────
            imdb_id_final = _extract_imdb_id(m.get("imdb_url", ""))
            if imdb_id_final and (not m.get("cast") or not m.get("poster_url")):
                tmdb = tmdb_cast_and_poster(imdb_id_final)
                if tmdb.get("cast"):       m["cast"]       = tmdb["cast"]
                if tmdb.get("poster_url"): m["poster_url"] = tmdb["poster_url"]

            REFRESH_STATUS["done"]       = i + 1
            REFRESH_STATUS["from_cache"] = from_cache
            REFRESH_STATUS["from_api"]   = from_api

            # Sync to R2 every 100 movies so progress is saved if it hits the daily limit
            if (i + 1) % 100 == 0:
                db["movies"] = movies
                r2_storage.save_movies_db(db)
                r2_storage.save_cache(cache)
                with STORE_LOCK:
                    STORE.clear()
                invalidate_db()

        # Final save to R2
        db["movies"] = movies
        r2_storage.save_movies_db(db)
        r2_storage.save_cache(cache)

        # Clear ALL in-memory session stores so every user gets fresh data on next load
        with STORE_LOCK:
            STORE.clear()
        invalidate_db()

        REFRESH_STATUS["complete"] = True
        REFRESH_STATUS["running"]  = False

    except Exception as e:
        REFRESH_STATUS["error"]    = str(e)
        REFRESH_STATUS["running"]  = False
        REFRESH_STATUS["complete"] = True


@app.route("/admin/debug")
@admin_required
def admin_debug():
    db     = r2_storage.load_movies_db()
    movies = db.get("movies", [])
    cache  = _load_cache()

    total   = len(movies)
    sample  = movies[:5]

    # Count by imdb_rating type
    counts = {"none": 0, "empty_str": 0, "zero": 0, "has_value": 0, "other": 0}
    for m in movies:
        r = m.get("imdb_rating")
        if r is None:             counts["none"] += 1
        elif r == "":             counts["empty_str"] += 1
        elif r == 0 or r == 0.0: counts["zero"] += 1
        elif _safe_float(str(r)): counts["has_value"] += 1
        else:                     counts["other"] += 1

    has_imdb_url  = sum(1 for m in movies if m.get("imdb_url",""))
    cache_entries = len(cache)
    omdb_set      = bool(OMDB_API_KEY)

    return jsonify({
        "total_movies":    total,
        "omdb_key_set":    omdb_set,
        "cache_entries":   cache_entries,
        "has_imdb_url":    has_imdb_url,
        "imdb_rating_breakdown": counts,
        "sample_movies": [
            {"title": m.get("title"), "year": m.get("year"),
             "imdb_rating": m.get("imdb_rating"),
             "imdb_rating_type": type(m.get("imdb_rating")).__name__,
             "imdb_url": m.get("imdb_url","")[:60]}
            for m in sample
        ]
    })





@app.route("/admin/missing-movies")
@admin_required
def admin_missing_movies():
    """List all movies still missing IMDb data with what OMDb returns for each."""
    db     = r2_storage.load_movies_db()
    movies = db.get("movies", [])
    missing = [m for m in movies if _needs_refresh(m)]

    results = []
    for m in missing:
        title = m.get("title","")
        year  = m.get("year")
        omdb_resp = None
        if OMDB_API_KEY:
            try:
                r = requests.get("https://www.omdbapi.com/",
                    params={"apikey": OMDB_API_KEY, "t": title,
                            "y": str(year) if year else "", "type": "movie"},
                    timeout=6)
                omdb_resp = r.json().get("Response"), r.json().get("Error",""), r.json().get("Title","")
            except Exception as e:
                omdb_resp = ("error", str(e), "")
        results.append({
            "title": title, "year": year,
            "omdb_found": omdb_resp[0] if omdb_resp else "not tested",
            "omdb_error": omdb_resp[1] if omdb_resp else "",
            "omdb_matched_title": omdb_resp[2] if omdb_resp else "",
        })

    return jsonify({"total_missing": len(missing), "movies": results})

@app.route("/admin/omdb-diag")
@admin_required
def admin_omdb_diag():
    """
    Shows exactly which movies are still missing IMDb data,
    tests the API live, and shows raw OMDb responses for the first 5 missing.
    """
    db     = r2_storage.load_movies_db()
    movies = db.get("movies", [])

    missing = [m for m in movies if _needs_refresh(m)]

    # Test API with a known movie
    test_result = None
    if OMDB_API_KEY:
        try:
            r = requests.get("https://www.omdbapi.com/",
                params={"apikey": OMDB_API_KEY, "t": "The Godfather", "y": "1972", "type": "movie"},
                timeout=8)
            test_result = r.json()
        except Exception as e:
            test_result = {"exception": str(e)}

    # Try first 5 missing movies and show raw responses
    samples = []
    for m in missing[:5]:
        title = m.get("title", "")
        year  = m.get("year")
        raw   = None
        if OMDB_API_KEY:
            try:
                r = requests.get("https://www.omdbapi.com/",
                    params={"apikey": OMDB_API_KEY, "t": title,
                            "y": str(year) if year else "", "type": "movie"},
                    timeout=8)
                raw = r.json()
            except Exception as e:
                raw = {"exception": str(e)}
        samples.append({"title": title, "year": year, "omdb_response": raw})

    return jsonify({
        "omdb_key_set":       bool(OMDB_API_KEY),
        "omdb_key_preview":   OMDB_API_KEY[:6] + "..." if OMDB_API_KEY else "",
        "total_missing":      len(missing),
        "api_test_godfather": test_result,
        "first_5_missing":    samples,
    })

@app.route("/admin/test-omdb")
@admin_required
def admin_test_omdb():
    """Test OMDb API with a known movie and return the raw response."""
    if not OMDB_API_KEY:
        return jsonify({"error": "OMDB_API_KEY not set", "key_value": "empty"})
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": "12 Angry Men", "y": "1957",
                    "plot": "short", "type": "movie"},
            timeout=10
        )
        raw = r.json()
        return jsonify({
            "status_code":  r.status_code,
            "omdb_response": raw,
            "key_preview":  OMDB_API_KEY[:6] + "..." if len(OMDB_API_KEY) > 6 else OMDB_API_KEY,
        })
    except Exception as e:
        return jsonify({"exception": str(e), "type": type(e).__name__})



@app.route("/admin/refresh-omdb", methods=["POST"])
@admin_required
def admin_refresh_omdb():
    try:
        if REFRESH_STATUS.get("running"):
            return jsonify({"error": "Refresh already running"}), 400
        REFRESH_STATUS["complete"] = False
        REFRESH_STATUS["error"]    = ""
        threading.Thread(target=_refresh_omdb_background, daemon=True).start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/admin/refresh-omdb/progress")
@admin_required
def admin_refresh_progress():
    return jsonify(REFRESH_STATUS)


# ══════════════════════════════════════════════
#  ADMIN — REFRESH TMDB DATA (cast + poster)
# ══════════════════════════════════════════════

TMDB_REFRESH_STATUS = {"running": False, "done": 0, "total": 0,
                       "filled": 0, "complete": False, "error": ""}


def _tmdb_refresh_background():
    global TMDB_REFRESH_STATUS
    try:
        db     = r2_storage.load_movies_db()
        movies = db.get("movies", [])

        # Only process movies that have an imdb_url (need IMDb ID for TMDb lookup)
        # and are missing cast or poster
        missing = [m for m in movies
                   if _extract_imdb_id(m.get("imdb_url", ""))
                   and (not m.get("cast") or not m.get("poster_url") or not m.get("language"))]

        TMDB_REFRESH_STATUS.update({
            "running": True, "done": 0, "total": len(missing),
            "filled": 0, "complete": False, "error": ""
        })

        filled = 0
        for i, m in enumerate(missing):
            imdb_id = _extract_imdb_id(m.get("imdb_url", ""))
            tmdb    = tmdb_cast_and_poster(imdb_id)
            if tmdb.get("cast"):
                m["cast"]   = tmdb["cast"]
                filled += 1
            if tmdb.get("poster_url"):
                m["poster_url"] = tmdb["poster_url"]
            if tmdb.get("language"):
                m["language"] = tmdb["language"]

            TMDB_REFRESH_STATUS["done"]   = i + 1
            TMDB_REFRESH_STATUS["filled"] = filled

            # Save every 200 movies
            if (i + 1) % 200 == 0:
                db["movies"] = movies
                r2_storage.save_movies_db(db)
                invalidate_db()

        db["movies"] = movies
        r2_storage.save_movies_db(db)
        with STORE_LOCK:
            STORE.clear()
        invalidate_db()

        TMDB_REFRESH_STATUS["complete"] = True
        TMDB_REFRESH_STATUS["running"]  = False

    except Exception as e:
        TMDB_REFRESH_STATUS["error"]    = str(e)
        TMDB_REFRESH_STATUS["running"]  = False
        TMDB_REFRESH_STATUS["complete"] = True


@app.route("/admin/refresh-tmdb", methods=["POST"])
@admin_required
def admin_refresh_tmdb():
    try:
        if TMDB_REFRESH_STATUS.get("running"):
            return jsonify({"error": "TMDb refresh already running"}), 400
        if not TMDB_API_KEY:
            return jsonify({"error": "TMDB_API_KEY is not set in Railway environment variables."}), 400
        TMDB_REFRESH_STATUS["complete"] = False
        TMDB_REFRESH_STATUS["error"]    = ""
        threading.Thread(target=_tmdb_refresh_background, daemon=True).start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/refresh-tmdb/progress")
@admin_required
def admin_tmdb_progress():
    return jsonify(TMDB_REFRESH_STATUS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
