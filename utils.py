import streamlit as st
import pandas as pd
import yfinance as yf
import json
import numpy as np
import feedparser
from datetime import datetime
import time
from xgboost import XGBRegressor
import requests
from calendar import timegm
import tensorflow as tf
import joblib

# Note: The 'create_sequences' and 'mc_dropout_predict' functions are now part of this utils file.
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1: break
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def mc_dropout_predict(model, X, samples=100):
    predictions = np.zeros((samples, X.shape[0]))
    for i in range(samples):
        predictions[i, :] = model(X, training=True).numpy().flatten()
    return predictions

@st.cache_resource
def load_model_object(config):
    """Loads a predictive model object based on its type."""
    if config['type'] == 'xgboost':
        model = XGBRegressor()
        model.load_model(config['file'])
        return model
    elif config['type'] == 'lstm':
        return tf.keras.models.load_model(config['file'])
    return None

@st.cache_data
def load_all_models(models_config):
    loaded_models = {}
    errors = []
    for name, config in models_config.items():
        try:
            config_copy = config.copy()
            if config['type'] == 'linear':
                with open(config['file'], 'r') as f:
                    model_data = json.load(f)
                config_copy["intercept"] = model_data['intercept']
                config_copy["coeffs"] = pd.Series(model_data['coefficients'])
            elif config['type'] in ['xgboost', 'lstm']:
                config_copy["model"] = load_model_object(config)
                if config['type'] == 'lstm':
                    config_copy["x_scaler_obj"] = joblib.load(config['x_scaler'])
                    config_copy["y_scaler_obj"] = joblib.load(config['y_scaler'])
            loaded_models[name] = config_copy
        except Exception as e:
            errors.append(f"Error loading model '{name}': {e}")
    return loaded_models, errors

@st.cache_data
def fetch_and_prepare_data(ticker, num_lags=21):
    """Fetches data and creates all required features for all models."""
    df = yf.Ticker(ticker).history(period="1y", interval="1d")
    if df.empty: return None, "No data from yfinance."
    
    df = df.reset_index()
    df.rename(columns={'Date': 'time', 'Close': 'btc_close'}, inplace=True)
    df['time'] = pd.to_datetime(df['time']).dt.date
    df['btc_return'] = df['btc_close'].pct_change()
    df['Next_Return'] = df['btc_return'].shift(-1)
    
    # Features for LSTM model
    df['volatility_14'] = df['btc_return'].rolling(window=14).std()
    delta = df['btc_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Features for Linear/XGBoost models
    for i in range(1, num_lags + 1):
        df[f'lag_{i}'] = df['btc_return'].shift(i)
    
    df.dropna(inplace=True)
    return df, None

@st.cache_data(ttl=1800)
def fetch_news(feed_url: str):
    # This function is moved here unchanged...
    try:
        resp = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200:
            return None, f"Error: Status code {resp.status_code}."
        
        feed = feedparser.parse(resp.content)
        articles = []
        keywords = ("bitcoin", "btc")
        
        for e in feed.entries:
            text = (e.title + e.summary).lower()
            if any(k in text for k in keywords):
                dt = e.get("published_parsed")
                when = datetime.fromtimestamp(time.mktime(dt)).strftime("%d %b %Y") if dt else "—"
                articles.append({"title": e.title, "link": e.link, "published": when})
                if len(articles) >= 5: break
        
        return articles, None
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"