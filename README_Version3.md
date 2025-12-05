```markdown
# SENTITMENT (profitandloss)

Streamlit app for sentiment analysis and evaluating the impact of discount on profit.

This repo's Streamlit main module: app_Version2.py

## Problem you saw
You saw in the logs:

    ModuleNotFoundError: No module named 'statsmodels'

That means the environment where the app runs didn't have `statsmodels` installed. Fixes are below.

## What I changed / added
- app_Version2.py: more robust imports and graceful fallbacks:
  - If `vaderSentiment` isn't available, the app shows clear remediation steps and stops (sentiment requires VADER).
  - If `statsmodels` isn't available, the app falls back to `scikit-learn`'s LinearRegression so you still get coefficient estimates (no p-values).
- requirements.txt: ensure `statsmodels` and `vaderSentiment` (and `nltk`) are listed so Streamlit Cloud installs them.

## How to fix in your repository / redeploy
1. Update requirements.txt (this file includes statsmodels). Commit & push:

```bash
git add requirements.txt app_Version2.py README.md
git commit -m "Add statsmodels to requirements and make app resilient to missing packages"
git push
```

2. On Streamlit Community Cloud the app will reinstall dependencies and redeploy automatically. Check the app logs (Manage app → Logs) for install progress.

3. If `statsmodels` still fails to install on the Cloud environment (for example due to binary wheel issues for a specific Python version), the app will automatically use scikit-learn as a fallback so regression still runs (without p-values). In that case you can:
   - Use the fallback results, or
   - Pin a `statsmodels` version compatible with the environment's Python, for example:
     ```
     statsmodels==0.14.0
     ```
     (Only pin if you confirm compatibility.)

## Local run
1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
streamlit run app_Version2.py
```

## Notes
- If you want to always require statsmodels and fail fast, remove the sklearn fallback and keep the top-level import of statsmodels.
- If the error persists on Streamlit Cloud after adding statsmodels, open the logs and share the install errors; I can help pin a compatible version or adjust the environment.
```