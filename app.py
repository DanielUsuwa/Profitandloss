#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import plotly.express as px

# Sentiment
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if needed
nltk_downloaded = False
try:
    _ = SentimentIntensityAnalyzer()
    nltk_downloaded = True
except Exception:
    nltk.download("vader_lexicon")
    nltk_downloaded = True

st.set_page_config(page_title="Sentiment vs Sales — Profitandloss", layout="wide")

st.title("Sentiment Analysis → Impact on Sales")
st.markdown(
    """
Upload a CSV containing text (customer feedback, tweets, reviews) and a numeric sales column (units sold, revenue).
The app computes sentiment scores (VADER), categorizes sentiment, and shows how sentiment relates to sales.
"""
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
- Sentiment engine: NLTK VADER (fast, rule-based)
- Good for short social text, reviews, and social media
- For transformer-based models (BERT, RoBERTa) see README
"""
    )
    st.markdown("### Demo / quick start")
    st.write("You can try the demo dataset below if you don't have a CSV ready.")
    st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV must contain a text column and a numeric sales column.")
use_demo = st.checkbox("Use demo dataset", value=False)

if use_demo and uploaded_file is not None:
    st.warning("Demo selected and file uploaded — demo will be used. Uncheck demo to use your file.")

def generate_demo(n=300, seed=42):
    np.random.seed(seed)
    # Generate synthetic text sentiment and sales
    sentiments = np.random.choice(["positive", "neutral", "negative"], size=n, p=[0.45, 0.25, 0.30])
    base_sales = {"positive": 1200, "neutral": 800, "negative": 500}
    sales = np.array([np.random.normal(loc=base_sales[s], scale=150) for s in sentiments])
    # Create simple text strings
    text_map = {
        "positive": ["love", "great", "excellent", "fantastic", "amazing"],
        "neutral": ["okay", "fine", "average", "so-so", "neutral"],
        "negative": ["bad", "terrible", "poor", "disappointing", "hate"]
    }
    texts = [f"This product is {np.random.choice(text_map[s])}" for s in sentiments]
    df = pd.DataFrame({"text": texts, "sales": np.round(sales).astype(int)})
    return df

if use_demo:
    df = generate_demo()
else:
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

if df is None:
    st.info("Upload a CSV or select 'Use demo dataset' to get started.")
    st.stop()

st.subheader("Preview dataset")
st.dataframe(df.head())

# Ask for column names
st.sidebar.header("Column selection")
text_col = st.sidebar.text_input("Text column name", value="text")
sales_col = st.sidebar.text_input("Sales column name", value="sales")
date_col = st.sidebar.text_input("Optional date column name (YYYY-MM-DD)", value="")

# Validate columns
missing = []
if text_col not in df.columns:
    missing.append(text_col)
if sales_col not in df.columns:
    missing.append(sales_col)

if missing:
    st.error(f"Column(s) not found in uploaded data: {', '.join(missing)}. Please correct column names.")
    st.stop()

# Clean sales column
def parse_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x)
    # Remove currency symbols and commas
    s = re.sub(r"[^\d\.\-eE]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

df["_sales_raw"] = df[sales_col].apply(parse_numeric)
n_missing_sales = df["_sales_raw"].isna().sum()
if n_missing_sales > 0:
    st.warning(f"{n_missing_sales} rows have non-numeric sales after parsing and will be dropped for analysis.")

df_clean = df.dropna(subset=[text_col, "_sales_raw"]).copy()
df_clean = df_clean.reset_index(drop=True)

# Sentiment scoring
sid = SentimentIntensityAnalyzer()

@st.cache_data
def compute_sentiment(series):
    scores = series.apply(lambda t: sid.polarity_scores(str(t))["compound"])
    return scores

with st.spinner("Computing sentiment scores..."):
    df_clean["_sentiment_score"] = compute_sentiment(df_clean[text_col])

# Categorize sentiment
def categorize(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df_clean["_sentiment_label"] = df_clean["_sentiment_score"].apply(categorize)
df_clean["_sales_numeric"] = df_clean["_sales_raw"].astype(float)

st.subheader("Sentiment summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows analyzed", len(df_clean))
with col2:
    st.metric("Avg sentiment (compound)", f"{df_clean['_sentiment_score'].mean():.3f}")
with col3:
    st.metric("Avg sales", f"{df_clean['_sales_numeric'].mean():.2f}")

# Distribution plot
fig_pie = px.pie(df_clean, names="_sentiment_label", title="Sentiment distribution")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Sales by sentiment category")
agg = df_clean.groupby("_sentiment_label")["_sales_numeric"].agg(["count", "mean", "median", "std"]).reset_index()
agg = agg.rename(columns={"count": "n", "mean": "mean_sales", "median": "median_sales", "std": "std_sales"})
st.dataframe(agg.style.format({"mean_sales": "{:.2f}", "median_sales": "{:.2f}", "std_sales": "{:.2f}"}))

fig_box = px.box(df_clean, x="_sentiment_label", y="_sales_numeric", points="all",
                 labels={"_sentiment_label": "Sentiment", "_sales_numeric": "Sales"},
                 title="Sales distribution per sentiment")
st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Scatter: Sentiment score vs Sales")
fig_scatter = px.scatter(df_clean, x="_sentiment_score", y="_sales_numeric", color="_sentiment_label",
                         trendline="ols", labels={"_sentiment_score": "Sentiment (compound)", "_sales_numeric": "Sales"},
                         title="Sentiment score vs Sales with linear fit")
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation and regression
st.subheader("Correlation & simple linear regression")
x = df_clean["_sentiment_score"].values.reshape(-1, 1)
y = df_clean["_sales_numeric"].values

corr_text = "Not enough variance to compute correlation."
try:
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        pearson_r, pval = pearsonr(x.ravel(), y)
        lr = LinearRegression().fit(x, y)
        y_pred = lr.predict(x)
        r2 = r2_score(y, y_pred)
        st.markdown(f"- Pearson r = **{pearson_r:.3f}** (p = {pval:.3g})")
        st.markdown(f"- Linear regression slope = **{lr.coef_[0]:.3f}**, intercept = **{lr.intercept_:.2f}**, R^2 = **{r2:.3f}**")
        # Interpretation
        if abs(pearson_r) < 0.1:
            interpretation = "no clear linear relationship"
        elif abs(pearson_r) < 0.3:
            interpretation = "weak relationship"
        elif abs(pearson_r) < 0.6:
            interpretation = "moderate relationship"
        else:
            interpretation = "strong relationship"
        st.markdown(f"Interpretation: There is a **{interpretation}** between sentiment and sales in this dataset.")
    else:
        st.warning(corr_text)
except Exception as e:
    st.error(f"Could not compute correlation/regression: {e}")

# Raw results and download
st.subheader("Results & export")
st.markdown("You can download the augmented dataset with sentiment scores and labels.")
out_df = df_clean.copy()
out_df = out_df.drop(columns=[col for col in out_df.columns if col.startswith("_") and col not in ["_sentiment_score", "_sentiment_label", "_sales_numeric"]], errors="ignore")
out_df = out_df.rename(columns={"_sentiment_score": "sentiment_compound", "_sentiment_label": "sentiment_label", "_sales_numeric": "sales_numeric"})
csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv, file_name="sentiment_sales_results.csv", mime="text/csv")

st.caption("Built with VADER sentiment (nltk). For transformer-based sentiment, see README.")
