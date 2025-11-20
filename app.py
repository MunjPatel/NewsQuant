import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime, timedelta
from duckduckgo_search import DDGS

# 1. CONFIG
st.set_page_config(page_title="NewsQuant: Multi-Asset Sentiment", page_icon="📰", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .news-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #6b7280;
        transition: transform 0.1s;
    }
    .news-card:hover { transform: scale(1.01); }
    .positive { border-left-color: #22c55e; }
    .negative { border-left-color: #ef4444; }
    
    /* AI Summary Box */
    .ai-summary {
        background-color: #172554; /* Dark Blue */
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        color: #e0f2fe;
    }
    
    a { color: #60a5fa; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 NewsQuant: Event-Driven Sentiment Engine")

TICKER_UNIVERSE = {
    "Mega Cap Tech": {"NVIDIA": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Google": "GOOGL", "Meta": "META", "Tesla": "TSLA", "AMD": "AMD"},
    "Semiconductors": {"Broadcom": "AVGO", "Intel": "INTC", "Qualcomm": "QCOM", "Micron": "MU", "TSMC": "TSM", "Super Micro": "SMCI", "Arm Holdings": "ARM"},
    "Software & AI": {"Palantir": "PLTR", "Snowflake": "SNOW", "Salesforce": "CRM", "Adobe": "ADBE", "CrowdStrike": "CRWD", "Oracle": "ORCL", "C3.ai": "AI"},
    "Crypto & Blockchain": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "Coinbase": "COIN", "MicroStrategy": "MSTR", "Marathon Digital": "MARA"},
    "Finance & Banks": {"JPMorgan": "JPM", "Goldman Sachs": "GS", "Visa": "V", "Mastercard": "MA", "BlackRock": "BLK", "PayPal": "PYPL", "Square": "SQ"},
    "EV & Auto": {"Rivian": "RIVN", "Lucid": "LCID", "Ford": "F", "GM": "GM", "Toyota": "TM", "NIO": "NIO"},
    "Defense & Energy": {"Lockheed Martin": "LMT", "Boeing": "BA", "Exxon Mobil": "XOM", "Chevron": "CVX", "NextEra Energy": "NEE", "Plug Power": "PLUG"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI", "VIX": "^VIX", "Gold": "GLD"}
}

# Flatten for dropdown
flat_tickers = {}
for category, assets in TICKER_UNIVERSE.items():
    for name, symbol in assets.items():
        flat_tickers[f"{name} ({symbol})"] = symbol

# 3. SIDEBAR CONFIGURATION
st.sidebar.header("Configuration")

# API Key
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

# Multi-Select Tickers
selected_labels = st.sidebar.multiselect(
    "Select Assets to Analyze",
    options=list(flat_tickers.keys()),
    default=["Tesla (TSLA)"]
)

# 4. DATA FUNCTIONS
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    """Fetches Price History using yf.download (More Robust)"""
    try:
        # yf.download is often more reliable than Ticker.history for single columns
        df = yf.download(ticker_symbol, period="1mo", interval="1d", progress=False)
        
        # Handle Multi-Index columns if they appear
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker_symbol, axis=1, level=1)
            
        return df
    except Exception as e:
        st.error(f"Price Data Error ({ticker_symbol}): {e}")
        return pd.DataFrame()

def get_news_ddg(ticker_symbol):
    """Fetches News using DuckDuckGo"""
    try:
        # Removing 'USD' for crypto search helps DDG accuracy
        search_term = ticker_symbol.replace("-USD", "")
        results = DDGS().news(keywords=f"{search_term} stock news", max_results=8)
        
        formatted_news = []
        for item in results:
            formatted_news.append({
                "title": item.get('title', 'No Title'),
                "link": item.get('url', '#'),
                "publisher": item.get('source', 'Unknown'),
                "time": item.get('date', datetime.now().isoformat())
            })
        return formatted_news
    except Exception as e:
        st.warning(f"News fetch warning: {e}")
        return []

# 5. AI FUNCTIONS
def analyze_sentiment_and_summary(ticker, news_items, api_key):
    """
    Performs two tasks in one API call to save time:
    1. Classify Sentiment of each headline.
    2. Write a summary inference.
    """
    if not news_items or not api_key:
        return [], "No data for analysis."
    
    client = Groq(api_key=api_key)
    
    headlines_block = "\n".join([f"{i+1}. {item['title']}" for i, item in enumerate(news_items)])
    
    prompt = f"""
    You are a Financial News Analyst. Analyze these headlines for {ticker}:
    
    {headlines_block}
    
    Perform two tasks:
    1. Classify each headline as POSITIVE, NEGATIVE, or NEUTRAL. Return as a JSON list string (e.g., ["POSITIVE", "NEUTRAL"]).
    2. Write a brief "Market Pulse" inference (max 50 words). Why is the stock moving based on these headlines?
    
    Format output exactly like this:
    SENTIMENT_LIST: ["POSITIVE", "NEGATIVE", ...]
    SUMMARY: [Your summary text here]
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = completion.choices[0].message.content
        
        # Manual Parsing for Robustness
        sentiments = []
        summary = "Analysis failed."
        
        import ast
        
        # Extract List
        if "SENTIMENT_LIST:" in content:
            list_part = content.split("SENTIMENT_LIST:")[1].split("SUMMARY:")[0].strip()
            try:
                sentiments = ast.literal_eval(list_part)
            except:
                sentiments = ["NEUTRAL"] * len(news_items)
        
        # Extract Summary
        if "SUMMARY:" in content:
            summary = content.split("SUMMARY:")[1].strip()
            
        return sentiments, summary
            
    except Exception as e:
        return ["NEUTRAL"] * len(news_items), f"AI Error: {e}"

# 6. MAIN APP LOGIC
if not selected_labels:
    st.info("👈 Please select at least one asset from the sidebar.")
else:
    # Create Tabs for each selected ticker
    tabs = st.tabs([label.split(" (")[0] for label in selected_labels])
    
    for i, label in enumerate(selected_labels):
        ticker = flat_tickers[label]
        
        with tabs[i]:
            st.markdown(f"## 🔎 Analysis: {ticker}")
            
            # 1. Fetch Data
            col1, col2 = st.columns([2, 1])
            with col1:
                with st.spinner("Fetching Chart..."):
                    price_df = get_market_data(ticker)
            with col2:
                # News is fast, no spinner needed usually
                news_list = get_news_ddg(ticker)
            
            # 2. Run AI Analysis (If Key Present)
            sentiments = []
            summary = ""
            
            if api_key and news_list:
                with st.spinner("🤖 AI Reading News..."):
                    sentiments, summary = analyze_sentiment_and_summary(ticker, news_list, api_key)
            elif not api_key:
                st.warning("Enter API Key to see AI Inference.")
                sentiments = ["NEUTRAL"] * len(news_list)
            
            # 3. Merge Data
            sentiment_score = 0
            processed_news = []
            limit = min(len(news_list), len(sentiments))
            
            for j in range(limit):
                item = news_list[j]
                sent = sentiments[j]
                item['sentiment'] = sent
                processed_news.append(item)
                if sent == "POSITIVE": sentiment_score += 1
                elif sent == "NEGATIVE": sentiment_score -= 1

            # --- VISUALIZATION ---
            
            # A. AI Inference Box
            if summary:
                st.markdown(f"""
                <div class="ai-summary">
                    <strong>🤖 AI Market Pulse:</strong> {summary}
                </div>
                """, unsafe_allow_html=True)

            # B. Price Chart
            if not price_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df.index, y=price_df['Close'],
                    mode='lines', name='Price',
                    line=dict(color='#60a5fa', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(96, 165, 250, 0.1)'
                ))
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title=None),
                    yaxis=dict(gridcolor='#374151', title=None),
                    title=f"{ticker} - 1 Month Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not load price data. Ticker might be delisted or API limit reached.")

            # C. Sentiment Meter
            st.markdown("#### 🌡️ Media Sentiment")
            m_col1, m_col2 = st.columns([1, 3])
            with m_col1:
                if sentiment_score > 0:
                    st.metric("Net Score", "BULLISH", f"+{sentiment_score}", delta_color="normal")
                elif sentiment_score < 0:
                    st.metric("Net Score", "BEARISH", f"{sentiment_score}", delta_color="inverse")
                else:
                    st.metric("Net Score", "NEUTRAL", "0", delta_color="off")
            
            with m_col2:
                 # Sentiment Bar
                if sentiments:
                    counts = pd.Series(sentiments).value_counts()
                    color_map = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#9ca3af"}
                    
                    fig_bar = go.Figure([go.Bar(
                        x=counts.values, y=counts.index,
                        orientation='h',
                        marker_color=[color_map.get(x, "#9ca3af") for x in counts.index],
                        text=counts.values, textposition='auto'
                    )])
                    fig_bar.update_layout(
                        height=120, margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, visible=False),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # D. News Feed
            st.markdown("#### 🗞️ Latest Headlines")
            for item in processed_news:
                css = "neutral"
                emoji = "⚪"
                if item['sentiment'] == "POSITIVE":
                    css = "positive"; emoji = "🟢"
                elif item['sentiment'] == "NEGATIVE":
                    css = "negative"; emoji = "🔴"
                
                # Parse Date
                try:
                    d = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    date_display = d.strftime("%b %d, %H:%M")
                except:
                    date_display = item['time']

                st.markdown(f"""
                <div class="news-card {css}">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:#9ca3af; font-size:0.8rem;">{item['publisher']} • {date_display}</span>
                        <span style="font-weight:bold; color:{'#4ade80' if css=='positive' else '#f87171' if css=='negative' else '#9ca3af'}">{item['sentiment']}</span>
                    </div>
                    <div style="font-size:1rem;">
                        <a href="{item['link']}" target="_blank" style="color:#f3f4f6; text-decoration:none;">{emoji} {item['title']}</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)