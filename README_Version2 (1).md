```markdown
# SENTITMENT

Streamlit app for sentiment analysis and evaluating the impact of discount on profit.

## Fix for ModuleNotFoundError: vaderSentiment

If you saw this error in the Streamlit logs:

    ModuleNotFoundError: No module named 'vaderSentiment.vaderSentiment'

It means the VADER package is not installed in the environment. To fix:

1. Add `vaderSentiment` to your `requirements.txt` (this repository includes it already).
2. Commit and push to GitHub:

```bash
git add requirements.txt app.py
git commit -m "Add vaderSentiment to requirements and make import robust"
git push
```

3. If you're using Streamlit Community Cloud, it will reinstall dependencies and redeploy automatically. If deploying elsewhere, rebuild the environment.

Or install locally:

```bash
pip install vaderSentiment
pip install -r requirements.txt
streamlit run app.py
```

## What I changed
- The app now tries to import VADER using multiple import paths:
  - `from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer`
  - `from vaderSentiment import SentimentIntensityAnalyzer`
  - fallback to `nltk.sentiment.vader.SentimentIntensityAnalyzer` (will attempt to download `vader_lexicon`)
- If none of these are available, the app will display a clear Streamlit error with actionable steps (add `vaderSentiment` to requirements or install it locally).

## Quick start (local)
1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes
- The recommended package is `vaderSentiment` (pip name: vaderSentiment). Installing that package should allow the original import to work.
- If you prefer to use an alternate sentiment library (TextBlob, transformers, etc.), tell me and I can update the app to use it instead.

```