"""
Streamlit app for sentiment analysis and impact of discount on profit.
- Upload CSV/XLSX with text, discount and profit columns.
- Computes VADER sentiment (compound score + label).
- Shows distributions, scatter plots and runs OLS regression (statsmodels).
"""

import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
import plotly.express as px
import io

st.set_page_config(page_title="SENTITMENT — Discount vs Profit Sentiment Analysis", layout="wide")

st.title("SENTITMENT")
st.markdown(
    """
    Analyze sentiment from text and estimate the impact of discount on profit.
    Upload a CSV/XLSX containing:
    - A text column (e.g., reviews, comments)
    - A discount column (numeric or percent strings like '10%')
    - A profit column (numeric)
    """
)

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

def to_numeric_percent(series):
    # Remove percent signs and commas, convert to numeric
    return pd.to_numeric(series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip(), errors='coerce')

@st.cache_data
def compute_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(str(t))['compound'] for t in texts]

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if df.empty:
        st.warning("Uploaded file is empty.")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    text_col = st.selectbox("Select text column for sentiment analysis", options=cols, index=0 if len(cols) > 0 else None)
    discount_col = st.selectbox("Select discount column (numeric or percent)", options=cols, index=1 if len(cols) > 1 else None)
    profit_col = st.selectbox("Select profit column (numeric)", options=cols, index=2 if len(cols) > 2 else None)

    run_button = st.button("Run analysis")

    if run_button:
        df_proc = df.copy()

        # Sentiment
        st.info("Computing sentiment scores (VADER)...")
        df_proc['sentiment_compound'] = compute_sentiment(df_proc[text_col].fillna(''))
        df_proc['sentiment_label'] = df_proc['sentiment_compound'].apply(
            lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
        )

        # Discount and profit numeric conversions
        st.info("Parsing numeric columns (discount, profit)...")
        df_proc[discount_col] = to_numeric_percent(df_proc[discount_col])
        df_proc[profit_col] = pd.to_numeric(df_proc[profit_col], errors='coerce')

        st.success("Processed dataset ready.")
        st.subheader("Processed sample")
        st.dataframe(df_proc[[text_col, discount_col, profit_col, 'sentiment_compound', 'sentiment_label']].head())

        # Visualizations
        st.subheader("Visualizations")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("Sentiment distribution")
            fig_hist = px.histogram(df_proc, x='sentiment_compound', color='sentiment_label', nbins=30,
                                    title="Sentiment (compound) distribution",
                                    labels={'sentiment_compound': 'Compound Sentiment Score'})
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown("Discount vs Profit (colored by sentiment)")
            fig_scatter = px.scatter(df_proc, x=discount_col, y=profit_col, color='sentiment_label',
                                     trendline="ols", title="Discount vs Profit")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("Correlation matrix (discount, profit, sentiment)")
        corr_df = df_proc[[discount_col, profit_col, 'sentiment_compound']].corr()
        st.dataframe(corr_df)

        # Regression using statsmodels
        st.subheader("Regression analysis: profit ~ discount + sentiment")
        st.markdown(
            """
            We'll fit an OLS model using statsmodels: profit ~ discount + sentiment_compound.
            This gives coefficient estimates and summary statistics to evaluate the impact.
            """
        )
        reg_df = df_proc[[discount_col, profit_col, 'sentiment_compound']].dropna()
        if reg_df.shape[0] < 10:
            st.warning("Not enough rows with valid numeric discount and profit and sentiment to run a reliable regression (need >= 10).")
        else:
            X = reg_df[[discount_col, 'sentiment_compound']]
            X = sm.add_constant(X)
            y = reg_df[profit_col]
            model = sm.OLS(y, X).fit()
            st.text(model.summary().as_text())

            # Quick interpretation snippet
            st.markdown("Quick interpretation")
            coef_discount = model.params.get(discount_col, np.nan)
            coef_sent = model.params.get('sentiment_compound', np.nan)
            st.write(f"- Coefficient for discount ({discount_col}): {coef_discount:.4f}")
            st.write(f"- Coefficient for sentiment (compound): {coef_sent:.4f}")
            st.markdown(
                """
                A negative coefficient for discount suggests higher discounts are associated with lower profit, 
                holding sentiment constant. A positive coefficient for sentiment suggests more positive sentiment 
                is associated with higher profit, holding discount constant.
                Check p-values in the regression summary to see if coefficients are statistically significant.
                """
            )

        # Allow download of processed CSV
        st.subheader("Download")
        buf = io.StringIO()
        df_proc.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download processed CSV (with sentiment)", data=buf.getvalue(), file_name="processed_with_sentiment.csv", mime="text/csv")

        st.info("If you want to include interactions (e.g., discount * sentiment) or additional covariates, export the processed CSV and modify the model code accordingly.")
else:
    st.info("Upload a dataset to get started. Example columns: review_text, discount (10% or 0.1 or 10), profit (numeric).")