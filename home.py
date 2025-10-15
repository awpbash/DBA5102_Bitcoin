import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import (
    load_all_models, fetch_and_prepare_data, fetch_news, 
    create_sequences, mc_dropout_predict
)
from config import MODELS_CONFIG, BTC_TICKER, ANNUAL_TRADING_DAYS, NEWS_FEED_URL
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(
    page_title="BTC Backtest Dashboard",
    page_icon="💼",
    layout="wide",
)

# --- Enhanced Interactive Sidebar CSS ---
st.markdown("""
<style>
    /* Main sidebar container */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }

    /* Sidebar navigation links */
    [data-testid="stSidebar"] a {
        padding: 10px 20px; /* NEW: Add padding for a larger click area */
        display: block; /* NEW: Make the entire area clickable */
        color: #b0b0c0;
        border-radius: 8px;
        transition: all 0.3s ease; /* NEW: Animate all properties for a smooth effect */
    }

    /* Active page link */
    [data-testid="stSidebar"] li[aria-selected="true"] a {
        background-color: #00a8ff;
        color: white;
        /* NEW: Add a "glow" effect to the active page */
        box-shadow: 0 0 15px rgba(0, 168, 255, 0.6);
    }

    /* Hover effect for sidebar links */
    [data-testid="stSidebar"] a:hover {
        background-color: #2a2a4e;
        color: #ffffff;
        /* NEW: Make the link slide right and grow slightly on hover */
        transform: translateX(5px) scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
header_cols = st.columns([8, 1])
with header_cols[0]:
    st.title("🏠 BTC Strategy Backtest Dashboard")
    st.markdown("Analyze and compare predictive trading models against a buy-and-hold baseline.")
with header_cols[1]:
    st.image("images/logo.png", width=100)

# --- Main Page Content ---
models, load_errors = load_all_models(MODELS_CONFIG)
if load_errors:
    for error in load_errors: st.error(error)

df_full, data_error = fetch_and_prepare_data(BTC_TICKER)
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
    # --- Calculation and charting logic... ---
    # (This entire block remains the same)
    df = df_full.tail(60).copy()
    performance_results = {}
    feature_columns_14, feature_columns_21 = [f'lag_{i}' for i in range(1, 15)], [f'lag_{i}' for i in range(1, 22)]
    X_linear, X_xgb = df[feature_columns_14], df[feature_columns_21]
    y_test = df['btc_return']
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

        elif model_config['type'] == 'lstm':
            n_steps = 30
            lstm_feature_cols = ['btc_return', 'volatility_14', 'rsi_14']
            start_loc = df_full.index.get_loc(df.index[0])
            required_data_start_loc = start_loc - n_steps
            if required_data_start_loc < 0:
                st.warning(f"Not enough data for LSTM. Skipping.")
                continue
            
            data_for_lstm = df_full.iloc[required_data_start_loc:].copy()
            x_scaler, y_scaler = model_config["x_scaler_obj"], model_config["y_scaler_obj"]
            scaled_features = x_scaler.transform(data_for_lstm[lstm_feature_cols])
            scaled_target = y_scaler.transform(data_for_lstm[['Next_Return']])
            scaled_numpy_array = np.hstack((scaled_features, scaled_target))
            X_test_lstm, _ = create_sequences(scaled_numpy_array, n_steps)

            if X_test_lstm.shape[0] > 0:
                # 1. Convert NumPy array to a TensorFlow Tensor
                X_test_tensor = tf.convert_to_tensor(X_test_lstm, dtype=tf.float32)

                # 2. Call the new, decorated function with the Tensor
                mc_predictions_tensor = mc_dropout_predict(model_config["model"], X_test_tensor)

                # 3. Convert the result back to a NumPy array for calculations
                mc_predictions = mc_predictions_tensor.numpy()

                y_pred_scaled = np.mean(mc_predictions, axis=0)
                y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_pred = pd.Series(y_pred_unscaled, index=df.index)
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
    # save all performance results to a session state variable
    st.session_state['performance_results'] = performance_results
    print(performance_results)
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

# --- Footer ---
footer_cols = st.columns([3, 1])
with footer_cols[0]:
    st.markdown("Developed for the NUS MSBA DBA5102 project.")
    st.caption("This dashboard is for educational purposes only and does not constitute financial advice.")
with footer_cols[1]:
    st.image("images/logo.png", width=120)