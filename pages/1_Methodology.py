import streamlit as st
import graphviz

# --- 1. Define the custom CSS for the hoverable cards ---
card_style = """
<style>
    .method-card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        height: 100%; /* Ensures cards in the same row have the same height */
        display: flex; /* Use flexbox to align content */
        flex-direction: column; /* Stack content vertically */
    }
    .method-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.4);
        transform: translateY(-5px);
        border-color: #00a8ff;
    }
    .method-card .content {
        flex-grow: 1; /* Allow content to grow and fill space */
    }
    .method-card h3 {
        color: #e0e0e0;
        margin-bottom: 15px;
    }
    .method-card h4 {
        color: #00a8ff; /* A different color for sub-headings */
        margin-top: 15px;
        margin-bottom: 10px;
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
st.header("🛠️ Our Analytical Methodology")
st.markdown("""
This project tackles the challenge of Bitcoin price prediction not by forecasting exact prices, but by modeling **directional returns**.
Our goal is to develop and validate a systematic trading strategy that delivers superior **risk-adjusted performance** compared to a passive "Buy & Hold" approach.
""")
st.markdown("---")


# --- 3. Visual Process Flowchart ---
st.subheader("Systematic Research & Trading Pipeline")
graph = graphviz.Digraph()
graph.attr('node', shape='box', style='rounded,filled', fillcolor='#1a1a2e', fontcolor='#e0e0e0', color='#00a8ff')
graph.attr('edge', color='#b0b0c0', fontcolor='#b0b0c0', fontsize='10')
graph.attr(rankdir='LR', bgcolor='transparent')

graph.node('Data', '1. Data Preparation\n(Log Returns & Stationarity)')
graph.node('Features', '2. Feature Engineering\n(Lagged Return Vectors)')
graph.node('Model', '3. Model Training\n(Diverse Model Ensemble)')
graph.node('Signals', '4. Signal Generation\n(Directional Forecasts)')
graph.node('Backtest', '5. Strategy Backtesting\n(Long/Short Simulation)')
graph.node('Evaluate', '6. Performance Evaluation\n(Sharpe Ratio Analysis)')

graph.edge('Data', 'Features', label=' Time Series ')
graph.edge('Features', 'Model', label=' Feature Matrix ')
graph.edge('Model', 'Signals', label=' Predictions ')
graph.edge('Signals', 'Backtest', label=' Trade Logic ')
graph.edge('Backtest', 'Evaluate', label=' PnL Results ')
graph.edge('Evaluate', 'Model', label=' Refine Models ', style='dashed')
st.graphviz_chart(graph)
st.markdown("---")


# --- 4. Content for the Cards (Rewritten based on your notebook) ---
methods = [
    {
        "title": "🎯 1. Problem Formulation & Data",
        "content": """
        <div class="content">
        <p>Instead of the predicting absolute price levels which is non-trivial, our approach is to forecast the <b>next day's directional movement</b>. This transforms the problem into a more tractable classification or regression task on returns.</p>
        <h4>Core Metric:</h4>
        <p>The primary success metric is the <b>Annualized Sharpe Ratio</b>, which measures return per unit of risk. The goal is to exceed the Sharpe Ratio of a passive Buy & Hold strategy.</p>
        <h4>Data Preparation:</h4>
        <p>We use daily BTC-USD price data from Yahoo Finance. The core variable is the <b>logarithmic return</b>, which ensures the time series is stationary (confirmed via an ADF test) and suitable for modeling.</p>
        </div>
        """
    },
    {
        "title": "🧩 2. Feature Engineering",
        "content": """
        <div class="content">
        <p>Our central hypothesis is that recent past returns contain predictive information about future returns. We construct a feature set to capture these potential auto-correlations.</p>
        <h4>Primary Features:</h4>
        <p>The main features are <b>lagged log returns</b> of Bitcoin itself (e.g., returns from t-1, t-2, ..., t-5). This vector of recent history allows models to identify short-term momentum or mean-reversion patterns.</p>
        <h4>Signal Generation:</h4>
        <p>The model's output (a predicted return) is converted into a clear trading signal: a positive prediction signals a 'long' position, while a negative prediction signals a 'short' or 'neutral' stance.</p>
        </div>
        """
    },
    {
        "title": "⚙️ 3. Diverse Modeling Ensemble",
        "content": """
        <div class="content">
        <p>No single model is universally best. We test a wide array of models, from classical econometrics to complex neural networks, to identify the most effective approach for this specific problem.</p>
        <h4>Models Tested:</h4>
        <ul>
            <li><b>Baselines:</b> Linear and Ridge Regression to capture simple linear relationships.</li>
            <li><b>Econometric:</b> ARMA-GARCH to model time-dependency and volatility clustering.</li>
            <li><b>Machine Learning:</b> XGBoost for non-linear patterns and an Artificial Neural Network (ANN) for deep learning.</li>
        </ul>
        </div>
        """
    },
    {
        "title": "⚖️ 4. Backtesting & Evaluation",
        "content": """
        <div class="content">
        <p>A robust backtesting framework is crucial for validating strategy performance and ensuring results are not due to chance. Our simulation aims for realism.</p>
        <h4>Strategy Simulation:</h4>
        <p>We test two distinct strategies: <b>"Long Only"</b> (cannot short-sell) and <b>"Long & Short"</b>. The backtest systematically applies the model's signals and incorporates a <b>0.1% transaction cost</b> on every trade to simulate real-world fees.</p>
        <h4>Evaluation:</h4>
        <p>The final performance of each model is judged by its ability to generate a superior risk-adjusted return (Sharpe Ratio) and total profit over the entire backtest period.</p>
        </div>
        """
    }
]

# --- 5. Display the Cards in a 2x2 Grid with Fixed Height ---
# Use st.container with a fixed height to ensure all cards are the same size
CARD_HEIGHT = 420

col1, col2 = st.columns(2)
with col1:
    with st.container(height=CARD_HEIGHT, border=False):
        st.markdown(f'<div class="method-card">{methods[0]["title"]}{methods[0]["content"]}</div>', unsafe_allow_html=True)
    with st.container(height=CARD_HEIGHT, border=False):
        st.markdown(f'<div class="method-card">{methods[2]["title"]}{methods[2]["content"]}</div>', unsafe_allow_html=True)

with col2:
    with st.container(height=CARD_HEIGHT, border=False):
        st.markdown(f'<div class="method-card">{methods[1]["title"]}{methods[1]["content"]}</div>', unsafe_allow_html=True)
    with st.container(height=CARD_HEIGHT, border=False):
        st.markdown(f'<div class="method-card">{methods[3]["title"]}{methods[3]["content"]}</div>', unsafe_allow_html=True)

st.markdown("---")

# --- 6. Model Deep Dive & Further Reading ---
st.subheader("A Closer Look at Our Models")
st.markdown("Below is a brief overview of the models deployed in our analysis, along with links to foundational research for those interested in the technical details.")

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.markdown("""
    #### Baseline & Econometric Models
    **Linear/Ridge Regression**
    <p>These models establish a performance baseline by capturing linear relationships between past returns and the next day's return. Ridge regression adds a penalty term to prevent overfitting, which is crucial in noisy financial data.</p>
    *Further Reading: [Hoerl & Kennard (1970), Ridge Regression](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634)*

    **ARMA-GARCH**
    <p>This is a classical time-series model. The ARMA component models the dependency of returns on past values, while the GARCH component models time-varying volatility (i.e., periods of high and low volatility clustering).</p>
    *Further Reading: [Bollerslev (1986), GARCH](https://www.sciencedirect.com/science/article/abs/pii/0304407686900631)*
    """, unsafe_allow_html=True)

with model_col2:
    st.markdown("""
    #### Machine Learning & Deep Learning
    **XGBoost (Extreme Gradient Boosting)**
    <p>XGBoost builds a strong predictive model by sequentially adding simple decision tree models, where each new tree corrects the errors of the previous ones. It excels at capturing complex, non-linear interactions in the feature space.</p>
    *Further Reading: [Chen & Guestrin (2016), XGBoost](https://arxiv.org/abs/1603.02754)*

    **Artificial Neural Network (ANN)**
    <p>Our ANN is a feedforward network that learns hierarchical patterns. By using non-linear activation functions across multiple layers, it can approximate highly complex relationships that other models might miss, making it well-suited for chaotic financial markets.</p>
    *Further Reading: [Rumelhart et al. (1986), Backpropagation](https://www.nature.com/articles/323533a0)*
    """, unsafe_allow_html=True)


# --- 7. Expander for the full Jupyter Notebook ---
with st.expander("Show Full Technical Notebook"):
    try:
        # NOTE: You must first export your .ipynb file to .html
        # In Jupyter: File -> Download as -> HTML (.html)
        with open("multimodels.html", "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=1000, scrolling=True)
    except FileNotFoundError:
        st.error("The 'multimodels.html' file was not found. Please ensure you have exported your notebook to HTML in the same directory.")

