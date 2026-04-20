# JZ's Movie Suggester

Dark space-themed movie recommendation web app.

## Deploy to Railway (step by step)

### 1. Create a GitHub repository
- Go to github.com and create a new repository called `jz-movies`
- Upload all files in this folder to it (drag and drop works in the GitHub UI)

### 2. Set environment variables on Railway
- Go to railway.app and sign up / log in
- Click "New Project" → "Deploy from GitHub repo" → select `jz-movies`
- Once deployed, go to your project → Variables tab → add these:

  | Variable      | Value                        |
  |---------------|------------------------------|
  | OMDB_API_KEY  | your OMDb API key            |
  | SECRET_KEY    | any long random string       |

### 3. Done
Railway auto-detects the Procfile and requirements.txt and builds everything.
Your app will be live at a URL like `jz-movies-production.up.railway.app`

## Run locally (for testing)
```
pip install -r requirements.txt
set OMDB_API_KEY=your_key_here      # Windows
set SECRET_KEY=any-random-string
python app.py
```
Then open http://localhost:5000

## File structure
```
jz_movies/
  app.py              # Flask backend + all Python logic
  requirements.txt    # Python dependencies
  Procfile            # Railway startup command
  templates/
    base.html         # Shared layout + space theme CSS
    index.html        # Upload page
    fetch.html        # OMDb fetch progress page
    suggest.html      # Profile + filters + results
```

## How to use
1. Run your bookmarks HTML through clean_bookmarks.py in Google Colab first
2. Upload the resulting movies_clean.html on the home page
3. Click through the fetch step (ratings are cached after first run)
4. Filter and get suggestions on the suggest page
