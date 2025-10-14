import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import json
import numpy as np
import feedparser
from datetime import datetime
import time
from xgboost import XGBRegressor
from openai import OpenAI
import requests
from calendar import timegm

# ======================================================================================
#  Configuration
# ======================================================================================
st.set_page_config(
    page_title="BTC Strategy Backtest Dashboard",
    page_icon="💼",
    layout="wide",
)

# --- App State & Constants ---
BTC_TICKER = "BTC-USD"
MODELS_CONFIG = {
    "Linear": {"file": "linear_model.json", "type": "linear"},
    "Ridge": {"file": "ridge_model.json", "type": "linear"},
    "XGBoost": {"file": "xgboost_model.json", "type": "xgboost"}
}
ANNUAL_TRADING_DAYS = 252
NEWS_FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"

# ======================================================================================
#  Data Fetching & Model Loading Functions
# ======================================================================================

@st.cache_resource
def load_xgboost_model(file_path):
    model = XGBRegressor()
    model.load_model(file_path)
    return model

@st.cache_data
def load_all_models(models_config):
    loaded_models = {}
    errors = []
    for name, config in models_config.items():
        try:
            if config['type'] == 'linear':
                with open(config['file'], 'r') as f:
                    model_data = json.load(f)
                loaded_models[name] = {
                    "type": "linear", "intercept": model_data['intercept'],
                    "coeffs": pd.Series(model_data['coefficients'])
                }
            elif config['type'] == 'xgboost':
                loaded_models[name] = {
                    "type": "xgboost", "model": load_xgboost_model(config['file'])
                }
        except Exception as e:
            errors.append(f"Error loading '{name}': {e}")
    return loaded_models, errors

@st.cache_data
def fetch_and_prepare_data(num_lags=21):
    df = yf.Ticker(BTC_TICKER).history(period="1y", interval="1d")
    if df.empty: return None, "No data from yfinance."
    
    df = df.reset_index()
    df.rename(columns={'Date': 'time', 'Close': 'price'}, inplace=True)
    df['time'] = pd.to_datetime(df['time']).dt.date
    df['actual_return'] = df['price'].pct_change()
    
    for i in range(1, num_lags + 1):
        df[f'lag_{i}'] = df['actual_return'].shift(i)
    
    df.dropna(inplace=True)
    return df, None
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
header_cols = st.columns([8, 1]) # Create a 4:1 ratio for title vs logo
with header_cols[0]:
    st.title("💼 BTC Strategy Backtest Dashboard")
    st.markdown("Analyze and compare predictive trading models against a buy-and-hold baseline.")
with header_cols[1]:
    # Make sure the path to your logo is correct
    st.image("images/logo.png", width=100)

# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Backtest Dashboard", "Our Methods", "About Us", "AI Chatbot"])

# ======================== TAB 1: BACKTEST DASHBOARD ========================
with tab1:
    st.markdown("Analyze and compare predictive trading models against a buy-and-hold baseline.")
    
    models, load_errors = load_all_models(MODELS_CONFIG)
    if load_errors:
        for error in load_errors: st.error(error)

    df_full, data_error = fetch_and_prepare_data()
    if data_error:
        st.error(data_error)
        st.stop()

    with st.container(border=True):
        control_cols = st.columns(2)
        with control_cols[0]:
            selected_models = st.multiselect(
                "Select Models to Backtest", options=list(models.keys()), default=list(models.keys())
            )
        with control_cols[1]:
            strategy_to_chart = st.radio(
                "Select Strategy View", ["Long Only", "Long & Short"], horizontal=True
            )

    if not selected_models:
        st.warning("Please select at least one model to display.")
    else:
        df = df_full.tail(60).copy()
        performance_results = {}
        # ... (Calculation logic from previous script)
        feature_columns_14, feature_columns_21 = [f'lag_{i}' for i in range(1, 15)], [f'lag_{i}' for i in range(1, 22)]
        X_linear, X_xgb = df[feature_columns_14], df[feature_columns_21]
        y_test = df['actual_return']
        df['Buy & Hold'] = (1 + y_test).cumprod()

        for name in selected_models:
            if name not in models: continue
            model_config = models[name]
            y_pred = None
            if model_config['type'] == 'linear':
                y_pred = X_linear.dot(model_config['coeffs']) + model_config['intercept']
            elif model_config['type'] == 'xgboost':
                y_pred = model_config['model'].predict(X_xgb)
                y_pred = pd.Series(y_pred, index=df.index)

            if y_pred is not None:
                long_only_returns = np.where(y_pred > 0, y_test, 0)
                long_short_returns = np.sign(y_pred) * y_test
                df[f'Long Only_{name}'] = (1 + long_only_returns).cumprod()
                df[f'Long & Short_{name}'] = (1 + long_short_returns).cumprod()
                with np.errstate(divide='ignore', invalid='ignore'):
                    performance_results[name] = {
                        "Long Only Sharpe": (np.mean(long_only_returns) / np.std(long_only_returns)) * np.sqrt(ANNUAL_TRADING_DAYS),
                        "Long & Short Sharpe": (np.mean(long_short_returns) / np.std(long_short_returns)) * np.sqrt(ANNUAL_TRADING_DAYS),
                        "Long Only Portfolio": df[f'Long Only_{name}'].iloc[-1],
                        "Long & Short Portfolio": df[f'Long & Short_{name}'].iloc[-1]
                    }
        buy_hold_sharpe = (y_test.mean() / y_test.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
        buy_hold_portfolio = df['Buy & Hold'].iloc[-1]

        with st.container(border=True):
            # ... (Chart display logic)
            st.subheader(f"Portfolio Growth: {strategy_to_chart} Strategy vs. Buy & Hold")
            chart_cols_to_melt = ['Buy & Hold'] + [f'{strategy_to_chart}_{name}' for name in selected_models if name in models]
            df_melted = df.melt(id_vars=['time'],value_vars=chart_cols_to_melt,var_name='Strategy',value_name='Portfolio Value').replace({f'{strategy_to_chart}_{name}': f'{strategy_to_chart} ({name})' for name in selected_models})
            chart = alt.Chart(df_melted).mark_line(point=False, strokeWidth=3).encode(x=alt.X('time:T', title='Date'), y=alt.Y('Portfolio Value:Q', title='Portfolio Value ($)', scale=alt.Scale(zero=False)), color=alt.Color('Strategy:N', title='Legend', scale=alt.Scale(scheme='tableau10')), tooltip=[alt.Tooltip('time:T'), alt.Tooltip('Portfolio Value:Q', format='$,.2f')]).interactive()
            st.altair_chart(chart, use_container_width=True)

        bottom_cols = st.columns([3, 2])
        with bottom_cols[0]:
            with st.container(border=True):
                # ... (Metrics display logic)
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
                # ... (News display logic)
                st.subheader("📰 Latest News")
                articles, news_error = fetch_news(NEWS_FEED_URL)
                if news_error: st.error(news_error)
                elif articles:
                    for article in articles:
                        st.markdown(f"**[{article['title']}]({article['link']})**")
                        st.caption(f"Published: {article['published']}")
                        st.markdown("---", unsafe_allow_html=True)
                else: st.write("No news articles found.")

        with st.expander("Show Raw Data"):
            st.dataframe(df.sort_values('time', ascending=False), use_container_width=True)

# ======================== TAB 2: OUR METHODS ========================
with tab2:
    # --- 1. Define the custom CSS for the hoverable cards ---
    card_style = """
    <style>
        .method-card {
            background-color: #1a1a2e; /* A dark blue-purple background */
            border: 1px solid #2a2a4e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            height: 100%; /* Make cards in the same row have equal height */
        }
        .method-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.4);
            transform: translateY(-5px); /* Lift the card on hover */
            border-color: #00a8ff; /* Highlight with a blue border on hover */
        }
        .method-card h3 {
            color: #e0e0e0;
            margin-bottom: 15px;
        }
        .method-card p, .method-card li {
            color: #b0b0c0;
            font-size: 15px;
            line-height: 1.6;
        }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    # --- 2. Page Header ---
    st.header("🛠️ Our Quantitative Methodology")
    st.markdown("Our investment process is built on a foundation of rigorous quantitative research and disciplined execution. We leverage sophisticated data analysis and machine learning to identify and capitalize on market inefficiencies.")
    st.markdown("---")

    # --- 3. Content for the Cards ---
    # Structuring content this way makes it easier to manage
    methods = [
        {
            "title": "1. Investment Philosophy",
            "content": """
            <p>We believe that market returns, particularly in volatile assets like Bitcoin, exhibit patterns that are discoverable through data. Our philosophy is to develop predictive models that can identify short-term directional movements with a statistical edge.</p>
            """
        },
        {
            "title": "2. Data & Feature Engineering",
            "content": """
            <ul>
                <li><b>Primary Data Sources:</b> Our models are built on daily price data sourced from Yahoo Finance.</li>
                <li><b>Feature Creation:</b> We transform raw price data into a rich feature set based on <b>lagged daily returns</b>, allowing our models to learn from recent price momentum and mean-reversion patterns.</li>
            </ul>
            """
        },
        {
            "title": "3. Predictive Modeling",
            "content": """
            <p>We employ an ensemble of machine learning models to avoid reliance on a single methodology. Our current model suite includes:</p>
            <ul>
                <li><b>Regularized Linear Models (Ridge):</b> Provide a robust, interpretable baseline.</li>
                <li><b>Gradient Boosting Machines (XGBoost):</b> Capture complex, non-linear relationships.</li>
            </ul>
            """
        },
        {
            "title": "4. Strategy & Risk Management",
            "content": """
            <ul>
                <li><b>Signal Generation:</b> Models generate a daily directional forecast.</li>
                <li><b>Execution Logic:</b> We simulate "Long Only" and "Long & Short" strategies.</li>
                <li><b>Performance Evaluation:</b> All strategies are evaluated using the <b>Annualized Sharpe Ratio</b> and total portfolio growth.</li>
            </ul>
            """
        }
    ]

    # --- 4. Display the Cards in a 2x2 Grid ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="method-card"><h3>{methods[0]["title"]}</h3>{methods[0]["content"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="method-card"><h3>{methods[1]["title"]}</h3>{methods[1]["content"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="method-card"><h3>{methods[2]["title"]}</h3>{methods[2]["content"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="method-card"><h3>{methods[3]["title"]}</h3>{methods[3]["content"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Expander for the full Jupyter Notebook ---
    with st.expander("Show Full Technical Notebook"):
        try:
            with open("multimodels.html", "r", encoding="utf-8") as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=1000, scrolling=True)
        except FileNotFoundError:
            st.error("The 'multimodels.html' file was not found. Please ensure it is in the same directory as your Streamlit script.")

    st.write("---")
# ======================== TAB 3: ABOUT US ========================
with tab3:
    st.header("ℹ️ About Our Team")
    st.markdown("We are a team of data scientists and financial analysts dedicated to exploring the intersection of machine learning and cryptocurrency markets.")

    team_members = [
    {
        "name": "Ng Jun Wei", "role": "Lead Data Scientist", "image": "images/junwei.png",
        "description": "Jun Wei specializes in time-series forecasting and developed the XGBoost model used in this dashboard. With a background in quantitative finance, Ng Jun Wei drives our modeling strategy."
    },
    {
        "name": "Tan Hua Swen", "role": "Machine Learning Engineer", "image": "images/swen.png",
        "description": "Swen architected the backtesting engine and the Streamlit application. Her expertise in MLOps ensures our models are robust, scalable, and deployed efficiently."
    },
    {
        "name": "Faris Yusri", "role": "Financial Analyst", "image": "images/faris.png",
        "description": "Faris provides domain expertise on market dynamics and strategy evaluation. He is responsible for interpreting model performance and developing the Sharpe Ratio analytics."
    },
    {
        "name": "Marcus Teo", "role": "Machine Learning Engineer", "image": "images/marcus.png",
        "description": "Marcus architected the backtesting engine and the Streamlit application. His expertise in MLOps ensures our models are robust, scalable, and deployed efficiently."
    },
    {
        "name": "Ng Yu Fei", "role": "Financial Analyst", "image": "images/yufei.png",
        "description": "Yu Fei provides domain expertise on market dynamics and strategy evaluation. He is responsible for interpreting model performance and developing the Sharpe Ratio analytics."
    },
    {
        "name": "Rasyiqah Sahlim", "role": "Lead Data Scientist", "image": "images/rasyiqah.png",
        "description": "Rasyiqah specializes in time-series forecasting and developed the XGBoost model used in this dashboard. With a background in quantitative finance, Rasyiqah drives our modeling strategy."
    }
]

    # Sort the list of members alphabetically by name
    sorted_members = sorted(team_members, key=lambda x: x['name'])

    # --- Create the First Row ---
    st.write("---") # Optional separator
    top_row_cols = st.columns(3)

    # Display the first 3 members
    for i in range(3):
        with top_row_cols[i]:
            member = sorted_members[i]
            st.image(member['image'], caption=member['name'], use_container_width=True)
            st.markdown(f"#### {member['name']}")
            st.write(member['description'])

    # --- Create the Second Row ---
    st.write("---") # Optional separator
    bottom_row_cols = st.columns(3)

    # Display the next 3 members
    for i in range(3):
        with bottom_row_cols[i]:
            member = sorted_members[i + 3]
            st.image(member['image'], caption=member['name'], use_container_width=True)
            st.markdown(f"#### {member['name']}")
            st.markdown(f"**{member['role']}**")
            st.write(member['description'])

# ======================== TAB 4: AI CHATBOT ========================
with tab4:
    st.header("🤖 AI Financial Co-Pilot")
    st.caption("Your personal assistant for market analysis and model interpretation.")

    # --- Two-Column Layout ---
    chat_col, sidebar_col = st.columns([3, 1]) # Main chat area is 3x wider

    with chat_col:
        # --- Chat Interface ---
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            # System prompt to define the AI's persona
            system_prompt = {
                "role": "system",
                "content": "You are an expert financial analyst and data scientist. Your name is 'Co-Pilot'. You specialize in explaining quantitative trading models, backtesting results, and complex financial concepts like Sharpe Ratios in a clear and concise way. You are assisting a user who is looking at a Bitcoin trading strategy dashboard."
            }

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    system_prompt,
                    {"role": "assistant", "content": "Hello! I'm Co-Pilot, your AI financial assistant. How can I help you analyze today's backtest results?"}
                ]

            # Display chat history
            for message in st.session_state.messages:
                if message["role"] != "system": # Don't display the system prompt
                    # Use custom icons for user and assistant
                    avatar = "🧑‍💻" if message["role"] == "user" else "🧠"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])

            # Function to handle sending a prompt
            def send_prompt(prompt):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🧠"):
                    stream = client.chat.completions.create(
                        model="gpt-4o", # Or "gpt-3.5-turbo"
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        stream=True,
                    )
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # User input box at the bottom
            if prompt := st.chat_input("Explain the Sharpe Ratio..."):
                send_prompt(prompt)

        except Exception as e:
            st.error("OpenAI API key not found. Please add it to your `.streamlit/secrets.toml` file.", icon="🚨")

    with sidebar_col:
        # --- Sidebar with Quick Questions ---
        with st.container(border=True):
            st.subheader("Quick Questions")
            st.markdown("Click a button to ask the AI a pre-defined question.")
            
            if st.button("What is a Sharpe Ratio?"):
                send_prompt("What is a Sharpe Ratio and why is it important?")
            
            if st.button("Explain the 'Long & Short' strategy."):
                send_prompt("Explain the 'Long & Short' strategy used in this dashboard.")

            if st.button("What is the latest Bitcoin sentiment?"):
                send_prompt("What is the latest Bitcoin sentiment based on recent news articles?")
            
            if st.button("What are lagged returns?"):
                send_prompt("What are lagged returns and why are they used as features in these models?")

footer_cols = st.columns([3, 1]) # Create a 3:1 ratio for text vs logo
with footer_cols[0]:
    st.markdown("Developed for the NUS MSBA DBA5102 project.")
    st.caption("This dashboard is for educational purposes only and does not constitute financial advice.")
with footer_cols[1]:
    st.image("images/logo.png", width=120)