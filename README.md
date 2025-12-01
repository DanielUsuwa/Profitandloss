# Sentiment vs Sales — Profitandloss

This Streamlit app performs sentiment analysis on textual feedback (reviews, tweets, comments) and analyzes the impact of sentiment on sales (units sold, revenue). It computes sentiment scores using NLTK VADER, categorizes sentiment, and shows aggregated stats, plots, correlation, and a simple linear regression.

## Files added
- app.py — Streamlit application
- requirements.txt — Python dependencies
- .gitignore — common ignored files

## Dataset format
Your CSV should contain at least:
- A text column with customer feedback (default column name: `text`)
- A numeric sales column (units sold, revenue) (default column name: `sales`)

The app attempts to parse numbers with commas and common currency symbols (e.g. "$1,200.50"). Example:

text,sales
"This product is amazing!",1200
"Terrible quality",450

If your column names differ, provide their names in the sidebar inputs once you upload your CSV.

## Features
- VADER sentiment compound score per text
- Sentiment label: positive / neutral / negative (VADER thresholds)
- Distribution pie chart for sentiment
- Boxplot of sales by sentiment category
- Scatter of sentiment score vs sales with OLS trendline
- Pearson correlation and linear regression (slope, intercept, R²)
- Download augmented CSV with `sentiment_compound`, `sentiment_label`, and `sales_numeric`

## How to run locally
1. Clone your repository (or place these files in your project folder)
2. Create a virtual environment (recommended)
   - python -m venv .venv
   - source .venv/bin/activate (macOS/Linux) or .venv\Scripts\activate (Windows)
3. Install dependencies:
   - pip install -r requirements.txt
4. Run Streamlit:
   - streamlit run app.py
5. Open the local URL shown by Streamlit (usually http://localhost:8501)

## Deploy to Streamlit Cloud
1. Push your repository to GitHub.
2. On https://share.streamlit.io, sign in and click "New app".
3. Select the repository (`owner/repo`), branch, and `app.py` as the entrypoint.
4. Deploy — Streamlit Cloud will install the dependencies in `requirements.txt`.

Notes:
- Streamlit Cloud has limited compute. VADER is lightweight and recommended for Streamlit hosting.
- For larger transformer models (Hugging Face) prefer deploying to a VM, cloud function, or a paid resource.

## Want transformer-based sentiment?
If you want higher quality (context-aware) sentiment using transformer models (BERT/RoBERTa), I can:
- Add an option to use Hugging Face transformers (requires `transformers`, and likely `torch` or `tensorflow`)
- Provide instructions to host a model on Hugging Face Inference API or use a lighter distil model
- Add caching so predictions reuse results and reduce latency

## Troubleshooting
- If you see NLTK errors on first run, the app attempts to download the VADER lexicon. Ensure the server has internet access.
- If sales values are not being parsed correctly, ensure your CSV uses a standard numeric format or remove stray text from the sales column.

## Next improvements (optional)
- Use a finetuned classification/regression model to predict sales uplift from text.
- Time-series analysis if you include a date column (aggregate sentiment per day vs daily sales).
- Add more visualizations and statistical tests (ANOVA, causality checks).

If you'd like, I can:
- Swap VADER for a transformer-based model
- Add a notebook for model training on your historical data
- Add CI and streamlit config for automatic deploys
