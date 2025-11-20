import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime, timedelta
from duckduckgo_search import DDGS  # <--- NEW RELIABLE NEWS SOURCE

# 1. CONFIG
st.set_page_config(page_title="NewsQuant: Sentiment Engine", page_icon="📰", layout="wide")

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
    a { color: #60a5fa; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 NewsQuant: Event-Driven Sentiment Engine")

# 2. SIDEBAR
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key", type="password")

# 3. DATA FUNCTIONS (Robust Logic)
@st.cache_data(ttl=3600)
def get_market_data(ticker_symbol):
    """Fetches Price History Only"""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1mo", interval="1d")
        return hist
    except Exception:
        return pd.DataFrame()

def get_news_ddg(ticker_symbol):
    """Fetches News using DuckDuckGo (More reliable than yfinance)"""
    try:
        # Search for "{Ticker} stock news"
        results = DDGS().news(keywords=f"{ticker_symbol} stock", max_results=10)
        
        formatted_news = []
        for item in results:
            # DDGS returns: {'date': ..., 'title': ..., 'body': ..., 'url': ..., 'source': ...}
            formatted_news.append({
                "title": item.get('title', 'No Title'),
                "link": item.get('url', '#'),
                "publisher": item.get('source', 'Unknown'),
                "time": item.get('date', datetime.now().isoformat())
            })
        return formatted_news
    except Exception as e:
        st.error(f"News Search Error: {e}")
        return []

# 4. AI SENTIMENT ANALYST
def analyze_sentiment_batch(news_items, api_key):
    if not news_items or not api_key:
        return []
    
    client = Groq(api_key=api_key)
    
    # Simplify prompt to save tokens
    headlines_text = "\n".join([f"{i+1}. {item['title']}" for i, item in enumerate(news_items)])
    
    prompt = f"""
    Classify these financial headlines for {ticker} as POSITIVE, NEGATIVE, or NEUTRAL.
    
    Headlines:
    {headlines_text}
    
    Return ONLY a JSON list of strings, matching the order. Example: ["POSITIVE", "NEUTRAL"]
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = completion.choices[0].message.content
        
        # Robust Parsing
        import ast
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            return ast.literal_eval(content[start:end])
        else:
            return ["NEUTRAL"] * len(news_items)
            
    except Exception as e:
        return ["NEUTRAL"] * len(news_items)

# 5. MAIN EXECUTION
if ticker:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
        
    # Load Data
    with st.spinner(f"Fetching market data for {ticker}..."):
        price_df = get_market_data(ticker)
        
    # Load News (No Spinner needed, it's fast)
    news_list = get_news_ddg(ticker)
    
    if not news_list:
        st.info("No news found. Try a different ticker.")
    else:
        # Analyze
        if api_key:
            with st.spinner("🤖 AI is analyzing sentiment..."):
                sentiments = analyze_sentiment_batch(news_list, api_key)
        else:
            sentiments = ["NEUTRAL"] * len(news_list)
        
        # Calculate Metrics
        sentiment_score = 0
        processed_news = []
        
        # Safety: Ensure lists match length
        limit = min(len(news_list), len(sentiments))
        
        for i in range(limit):
            item = news_list[i]
            sent = sentiments[i]
            item['sentiment'] = sent
            processed_news.append(item)
            
            if sent == "POSITIVE": sentiment_score += 1
            elif sent == "NEGATIVE": sentiment_score -= 1

        # --- VISUALS ---
        
        # 1. Chart
        fig = go.Figure()
        if not price_df.empty:
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df['Close'],
                mode='lines', name='Price',
                line=dict(color='#3b82f6', width=2)
            ))
        fig.update_layout(
            title=f"{ticker} Price Action",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#374151'),
            xaxis=dict(showgrid=False),
            margin=dict(l=10,r=10,t=30,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. Scorecard
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### 🌡️ Score")
            if sentiment_score > 0:
                st.metric("Net Sentiment", "BULLISH", f"+{sentiment_score}", delta_color="normal")
            elif sentiment_score < 0:
                st.metric("Net Sentiment", "BEARISH", f"{sentiment_score}", delta_color="inverse")
            else:
                st.metric("Net Sentiment", "NEUTRAL", "0", delta_color="off")
                
        with col2:
            st.markdown("### 📊 Distribution")
            # Simple color bar
            counts = pd.Series(sentiments).value_counts()
            fig_bar = go.Figure([go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[{"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#9ca3af"}.get(x, "#9ca3af") for x in counts.index],
                text=counts.values, textposition='auto'
            )])
            fig_bar.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, yaxis=dict(showgrid=False, visible=False))
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. News Feed
        st.markdown("### 🗞️ Analyzed Headlines")
        for item in processed_news:
            css = "neutral"
            emoji = "⚪"
            if item['sentiment'] == "POSITIVE":
                css = "positive"; emoji = "🟢"
            elif item['sentiment'] == "NEGATIVE":
                css = "negative"; emoji = "🔴"
            
            # Clean Date
            try:
                date_obj = datetime.fromisoformat(item['time'].replace("Z", "+00:00"))
                date_str = date_obj.strftime("%b %d, %H:%M")
            except:
                date_str = item['time']

            st.markdown(f"""
            <div class="news-card {css}">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#9ca3af; font-size:0.8em;">{item['publisher']} • {date_str}</span>
                    <span style="font-weight:bold; font-size:0.8em; color:{'#4ade80' if css=='positive' else '#f87171' if css=='negative' else '#9ca3af'}">{item['sentiment']}</span>
                </div>
                <div style="font-size:1.1em;">
                    {emoji} <a href="{item['link']}" target="_blank" style="color:#f3f4f6; text-decoration:none;">{item['title']}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Enter a ticker to begin.")