import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import json
import numpy as np
import feedparser
from datetime import datetime
import time
import requests
from calendar import timegm
from xgboost import XGBRegressor

# ======================================================================================
#  Configuration
# ======================================================================================
st.set_page_config(
    page_title="BTC Strategy Backtest Dashboard",
    page_icon="💼",
    layout="wide",
)

BTC_TICKER = "BTC-USD"
# --- UPDATED: Model config now includes type and file path ---
MODELS_CONFIG = {
    "Linear": {"file": "linear_model.json", "type": "linear"},
    "Ridge": {"file": "ridge_model.json", "type": "linear"},
    "XGBoost": {"file": "xgboost_model.json", "type": "xgboost"}
}
ANNUAL_TRADING_DAYS = 252
NEWS_FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"

# ======================================================================================
#  Model & Data Loading
# ======================================================================================

@st.cache_resource
def load_xgboost_model(file_path):
    """Loads a pre-trained XGBoost model object from a file."""
    model = XGBRegressor()
    model.load_model(file_path)
    return model

@st.cache_data
def load_all_models(models_config):
    """Loads all models (linear and XGBoost) from the config dictionary."""
    loaded_models = {}
    errors = []
    for name, config in models_config.items():
        try:
            if config['type'] == 'linear':
                with open(config['file'], 'r') as f:
                    model_data = json.load(f)
                loaded_models[name] = {
                    "type": "linear",
                    "intercept": model_data['intercept'],
                    "coeffs": pd.Series(model_data['coefficients'])
                }
            elif config['type'] == 'xgboost':
                # Loading the model object itself
                loaded_models[name] = {
                    "type": "xgboost",
                    "model": load_xgboost_model(config['file'])
                }
        except FileNotFoundError:
            errors.append(f"Model file '{config['file']}' for '{name}' not found.")
        except Exception as e:
            errors.append(f"Error loading model '{name}': {e}")
    return loaded_models, errors

@st.cache_data
def fetch_and_prepare_data(num_lags=21): # Increased default lags for XGBoost
    """Fetches data and creates the maximum number of lagged features required."""
    try:
        ticker = yf.Ticker(BTC_TICKER)
        df = ticker.history(period="90d", interval="1d")
        if df.empty: return None, "No data from yfinance."
        
        df = df.reset_index()
        df.rename(columns={'Date': 'time', 'Close': 'price'}, inplace=True)
        df['time'] = pd.to_datetime(df['time']).dt.date
        df['actual_return'] = df['price'].pct_change()
        
        for i in range(1, num_lags + 1):
            df[f'lag_{i}'] = df['actual_return'].shift(i)
        
        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, f"Error fetching data: {e}"

@st.cache_data(ttl=1800)
def fetch_news(feed_url: str):

    if feed_url.rstrip("/") == "https://www.coindesk.com/arc/outboundfeeds/rss":
        feed_url = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
    if feed_url.rstrip("/") == "https://www.coindesk.com/arc/outboundfeeds/rss/":
        feed_url = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"

    try:
        # Use requests to follow redirects and get the final URL/content.
        resp = requests.get(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 (news-bot)"},
            timeout=10,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None, f"Error: Failed to fetch the feed. Final status was {resp.status_code}."

        # Parse the content; give basehref so relative links resolve
        feed = feedparser.parse(resp.content, response_headers=resp.headers, sanitize_html=True)

        entries = getattr(feed, "entries", []) or []
        if not entries:
            return None, "No entries found in the feed."

        keywords = ("bitcoin", "btc")  # lowercase
        articles = []

        def pick_dt(e):
            """Return a datetime for an entry, best-effort."""
            tm = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            if tm and isinstance(tm, time.struct_time):
                # published_parsed is usually UTC
                return datetime.utcfromtimestamp(timegm(tm))
            # Fallback: try string fields
            s = getattr(e, "published", None) or getattr(e, "updated", None)
            if s:
                # Very light parse: try fromisoformat after cleaning Z
                s2 = s.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(s2)
                except Exception:
                    pass
            return None

        for e in entries:
            title = (getattr(e, "title", "") or "").lower()
            summary = (getattr(e, "summary", "") or "").lower()
            tags = " ".join([t.get("term","").lower() for t in getattr(e, "tags", []) or []])

            text = " ".join([title, summary, tags])
            if any(k in text for k in keywords):
                dt = pick_dt(e)
                when = dt.strftime("%d %b %Y, %H:%M UTC") if dt else "—"
                articles.append({
                    "title": getattr(e, "title", "(no title)"),
                    "link": getattr(e, "link", "#"),
                    "published": when,
                })
                if len(articles) >= 5:
                    break

        if not articles:
            return None, "No recent Bitcoin-specific articles found in the feed."

        return articles, None

    except Exception as e:
        return None, f"An unexpected error occurred while fetching the news feed: {e}"
# ======================================================================================
#  Main App UI
# ======================================================================================
st.title("DBA5102 Bitcoin Dashboard")
st.markdown("Analyze and compare the performance of predictive trading models against a buy-and-hold baseline.")

# --- Load initial data and models ---
models, load_errors = load_all_models(MODELS_CONFIG)
if load_errors:
    for error in load_errors: st.error(error)
    # Don't stop completely if only one model fails to load
    if not models: st.stop()

df, data_error = fetch_and_prepare_data()
if data_error:
    st.error(data_error)
    st.stop()

# --- Control Panel ---
with st.container(border=True):
    control_cols = st.columns(2)
    with control_cols[0]:
        selected_models = st.multiselect(
            "Select Models to Backtest",
            options=list(models.keys()),
            default=list(models.keys())
        )
    with control_cols[1]:
        strategy_to_chart = st.radio(
            "Select Strategy View",
            ["Long Only", "Long & Short"],
            horizontal=True,
            help="Choose the strategy to display on the Portfolio Growth chart."
        )

if not selected_models:
    st.warning("Please select at least one model to display.")
    st.stop()

# --- Strategy & Performance Calculation ---
performance_results = {}
feature_columns_14 = [f'lag_{i}' for i in range(1, 15)]
feature_columns_21 = [f'lag_{i}' for i in range(1, 22)]
X_linear = df[feature_columns_14]
X_xgb = df[feature_columns_21]
y_test = df['actual_return']
df['Buy & Hold'] = (1 + y_test).cumprod()

for name in selected_models:
    if name not in models: continue # Skip if a model failed to load
    
    model_config = models[name]
    y_pred = None

    # --- UPDATED: Prediction logic for different model types ---
    if model_config['type'] == 'linear':
        intercept = model_config['intercept']
        coeffs = model_config['coeffs']
        y_pred = X_linear.dot(coeffs) + intercept
    elif model_config['type'] == 'xgboost':
        model_obj = model_config['model']
        y_pred = model_obj.predict(X_xgb)
        y_pred = pd.Series(y_pred, index=df.index)

    if y_pred is not None:
        long_only_returns = np.where(y_pred > 0, y_test, 0)
        long_short_returns = np.sign(y_pred) * y_test
        
        df[f'Long Only_{name}'] = (1 + long_only_returns).cumprod()
        df[f'Long & Short_{name}'] = (1 + long_short_returns).cumprod()

        with np.errstate(divide='ignore', invalid='ignore'):
            long_only_sharpe = (long_only_returns.mean() / long_only_returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
            long_short_sharpe = (long_short_returns.mean() / long_short_returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
        
        performance_results[name] = {
            "Long Only Sharpe": long_only_sharpe,
            "Long & Short Sharpe": long_short_sharpe,
            "Long Only Portfolio": df[f'Long Only_{name}'].iloc[-1],
            "Long & Short Portfolio": df[f'Long & Short_{name}'].iloc[-1]
        }

buy_hold_sharpe = (y_test.mean() / y_test.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
buy_hold_portfolio = df['Buy & Hold'].iloc[-1]

# ======================================================================================
#  Display Area
# ======================================================================================
# The entire display area (charts, metrics, news) remains the same.
# It will dynamically adapt to the new "XGBoost" columns and results.

# --- Chart Display ---
with st.container(border=True):
    st.subheader(f"Portfolio Growth: {strategy_to_chart} Strategy vs. Buy & Hold")
    chart_cols_to_melt = ['Buy & Hold'] + [f'{strategy_to_chart}_{name}' for name in selected_models if name in models]
    df_melted = df.melt(
        id_vars=['time'],
        value_vars=chart_cols_to_melt,
        var_name='Strategy',
        value_name='Portfolio Value'
    ).replace({f'{strategy_to_chart}_{name}': f'{strategy_to_chart} ({name})' for name in selected_models})

    chart = alt.Chart(df_melted).mark_line(point=False, strokeWidth=3).encode(
        x=alt.X('time:T', title='Date'),
        y=alt.Y('Portfolio Value:Q', title='Portfolio Value ($)', scale=alt.Scale(zero=False)),
        color=alt.Color('Strategy:N', title='Legend', scale=alt.Scale(scheme='tableau10')),
        tooltip=[alt.Tooltip('time:T'), alt.Tooltip('Portfolio Value:Q', format='$,.2f')]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- Metrics and News ---
bottom_cols = st.columns([3, 2])
with bottom_cols[0]:
    with st.container(border=True):
        st.subheader("Performance Metrics")
        st.markdown("##### Baseline: Buy & Hold")
        bh_cols = st.columns(2)
        bh_cols[0].metric("Final Portfolio Value", f"${buy_hold_portfolio:,.2f}")
        bh_cols[1].metric("Annualized Sharpe Ratio", f"{buy_hold_sharpe:.2f}")

        for name in selected_models:
            if name in performance_results:
                st.markdown(f"##### Model: {name}")
                model_cols = st.columns(2)
                with model_cols[0]:
                    st.markdown("**Long Only Strategy**")
                    st.metric("Final Portfolio Value", f"${performance_results[name]['Long Only Portfolio']:,.2f}")
                    st.metric("Annualized Sharpe Ratio", f"{performance_results[name]['Long Only Sharpe']:.2f}")
                with model_cols[1]:
                    st.markdown("**Long & Short Strategy**")
                    st.metric("Final Portfolio Value", f"${performance_results[name]['Long & Short Portfolio']:,.2f}")
                    st.metric("Annualized Sharpe Ratio", f"{performance_results[name]['Long & Short Sharpe']:.2f}")

with bottom_cols[1]:
    with st.container(border=True):
        st.subheader("📰 Latest News")
        articles, news_error = fetch_news(NEWS_FEED_URL)
        if news_error: st.error(news_error)
        elif articles:
            for article in articles:
                st.markdown(f"**[{article['title']}]({article['link']})**")
                st.caption(f"Published: {article['published']}")
                st.markdown("---")
        else:
            st.write("No news articles found.")

with st.expander("Show Raw Data"):
    st.dataframe(df.sort_values('time', ascending=False), use_container_width=True)