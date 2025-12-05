# SENTITMENT

Streamlit app for sentiment analysis and evaluating the impact of discount on profit.

## What this does
- Lets you upload a CSV/XLSX dataset containing text (e.g., reviews), discount and profit columns.
- Computes VADER sentiment scores (compound) and assigns sentiment labels (positive/neutral/negative).
- Shows sentiment distribution and scatter plots (discount vs profit).
- Runs an OLS regression using `statsmodels`: profit ~ discount + sentiment_compound.
- Allows downloading the processed dataset with sentiment scores.

## Quick start (local)
1. Clone your repository (or add these files to your repo):
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
2. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

> Note: As requested, `statsmodels` is included in `requirements.txt`. If you install packages individually, run:
>
> pip install statsmodels

3. Run the app:

```bash
streamlit run app.py
```

4. Upload your dataset in the app and select the appropriate columns.

## Deployment
- To deploy on Streamlit Community Cloud, push this repo to GitHub and link the repo in Streamlit Cloud. The `requirements.txt` will be used to install dependencies.
- If deploying elsewhere, ensure `streamlit` and `statsmodels` are installed.

## Expected input
- Text column: any text (strings) to analyze sentiment.
- Discount column: numeric or percent strings like `10%`, `10` or `0.1`. The app does basic cleaning (strips `%` and commas).
- Profit column: numeric values.

## Notes & next steps
- The app uses VADER (`vaderSentiment`) which is tuned for social media / short texts. For deeper or domain-specific sentiment, consider fine-tuned transformer models.
- The regression is OLS and is simple; you may want to:
  - Add more controls (category, seasonality, product features).
  - Use log-transformations if profit is skewed.
  - Check heteroskedasticity and other diagnostic tests (statsmodels has tools).
  - Add interaction terms (discount * sentiment_compound) to test if discounts modify the sentiment effect on profit.
- If you want, I can add:
  - Example dataset and unit tests
  - An option to include interaction terms in the regression
  - A cached streaming pipeline for larger datasets

## Contact / Support
If you want customizations (different sentiment model, pre-processing for multi-language text, or automated dashboards), tell me how your dataset looks (sample columns and a few rows) and I will adapt the app accordingly.