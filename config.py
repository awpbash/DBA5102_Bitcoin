# --- App State & Constants ---
BTC_TICKER = "BTC-USD"
MODELS_CONFIG = {
    "Linear": {"file": "linear_model.json", "type": "linear"},
    "Ridge": {"file": "ridge_model.json", "type": "linear"},
    "XGBoost": {"file": "xgboost_model.json", "type": "xgboost"},
    "LSTM":  {
        "type": "lstm",
        "file": "lstm_model.keras",
        "x_scaler": "lstm_x_scaler.joblib",
        "y_scaler": "lstm_y_scaler.joblib"
        }
}
ANNUAL_TRADING_DAYS = 252
NEWS_FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"