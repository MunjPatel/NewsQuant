import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime, timedelta

# 1. CONFIG
st.set_page_config(page_title="NewsQuant: Sentiment Engine", page_icon="📰", layout="wide")

# Custom CSS for the "Terminal" look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .news-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #6b7280;
    }
    .positive { border-left-color: #22c55e; }
    .negative { border-left-color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 NewsQuant: Event-Driven Sentiment Engine")
st.markdown("### Visualize the impact of News Headlines on Price Action")

# 2. SIDEBAR & INPUTS
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key", type="password")

# 3. DATA FUNCTIONS
@st.cache_data(ttl=3600) # Cache for 1 hour to save API calls
def get_news_and_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    
    # 1. Get Historical Price (1 Month for context)
    hist = stock.history(period="1mo", interval="1d")
    
    # 2. Get News
    raw_news = stock.news
    
    news_data = []
    for item in raw_news:
        # Convert unix timestamp to datetime
        pub_time = datetime.fromtimestamp(item['providerPublishTime'])
        news_data.append({
            "title": item['title'],
            "link": item['link'],
            "publisher": item['publisher'],
            "time": pub_time,
            "timestamp": item['providerPublishTime']
        })
        
    return hist, news_data

# 4. AI SENTIMENT ANALYST
def analyze_sentiment_batch(news_items, api_key):
    if not news_items or not api_key:
        return []
    
    client = Groq(api_key=api_key)
    
    # Prepare a list for the prompt to save tokens (Batch Processing)
    headlines_text = "\n".join([f"{i+1}. {item['title']}" for i, item in enumerate(news_items)])
    
    prompt = f"""
    Analyze the sentiment of these financial news headlines for {ticker}.
    
    Headlines:
    {headlines_text}
    
    Task:
    Classify each headline as POSITIVE, NEGATIVE, or NEUTRAL.
    Return ONLY a JSON array of strings, e.g., ["POSITIVE", "NEUTRAL", "NEGATIVE"].
    Do not explain. Strictly match the order.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        # Parsing the fake-JSON output
        content = completion.choices[0].message.content
        # Simple cleanup to ensure list format
        import ast
        # Find the list bracket in the response
        start = content.find('[')
        end = content.find(']') + 1
        if start != -1 and end != -1:
            sentiment_list = ast.literal_eval(content[start:end])
            return sentiment_list
        else:
            return ["NEUTRAL"] * len(news_items) # Fallback
            
    except Exception as e:
        st.error(f"Sentiment Analysis Failed: {e}")
        return ["NEUTRAL"] * len(news_items)

# 5. MAIN EXECUTION
if ticker and api_key:
    with st.spinner(f"Fetching market data and news for {ticker}..."):
        price_df, news_list = get_news_and_data(ticker)
    
    if not news_list:
        st.warning("No recent news found for this asset.")
    else:
        # Run AI Analysis if button clicked or auto
        with st.spinner("🤖 AI is reading the headlines..."):
            sentiments = analyze_sentiment_batch(news_list, api_key)
        
        # Merge Sentiment into Data
        processed_news = []
        sentiment_score = 0 # Net Score
        
        for i, item in enumerate(news_list):
            sent = sentiments[i] if i < len(sentiments) else "NEUTRAL"
            item['sentiment'] = sent
            processed_news.append(item)
            
            if sent == "POSITIVE": sentiment_score += 1
            elif sent == "NEGATIVE": sentiment_score -= 1

        # --- VISUALIZATION LAYER ---
        
        # 1. Price Chart with News Markers
        fig = go.Figure()
        
        # Price Line
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['Close'],
            mode='lines', name='Price',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # News Markers (We map news time to the closest price time approx)
        # Note: Exact mapping requires aligning timestamps, for visual simplicity we just list them below
        # But advanced candidates map dots. Let's do a simpler "News Feed" below the chart to save complexity.
        
        fig.update_layout(
            title=f"{ticker} Price Action (1 Month)",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#374151'),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. Sentiment Meter
        st.markdown("### 🌡️ News Sentiment Score")
        
        score_col, summary_col = st.columns([1, 3])
        
        with score_col:
            if sentiment_score > 0:
                st.metric("Net Sentiment", "BULLISH", f"+{sentiment_score}", delta_color="normal")
            elif sentiment_score < 0:
                st.metric("Net Sentiment", "BEARISH", f"{sentiment_score}", delta_color="inverse")
            else:
                st.metric("Net Sentiment", "NEUTRAL", "0", delta_color="off")
                
        with summary_col:
            # Quick Bar Chart of Distribution
            counts = pd.Series(sentiments).value_counts()
            color_map = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#9ca3af"}
            
            fig_bar = go.Figure([go.Bar(
                x=counts.index, 
                y=counts.values,
                marker_color=[color_map.get(x, "#9ca3af") for x in counts.index]
            )])
            fig_bar.update_layout(
                height=100, 
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. The News Feed (Rich UI)
        st.markdown("### 🗞️ Analyzed News Feed")
        
        for item in processed_news:
            # Determine CSS class
            css_class = "neutral"
            emoji = "⚪"
            if item['sentiment'] == "POSITIVE": 
                css_class = "positive"
                emoji = "🟢"
            elif item['sentiment'] == "NEGATIVE": 
                css_class = "negative"
                emoji = "🔴"
            
            time_str = item['time'].strftime("%Y-%m-%d %H:%M")
            
            st.markdown(f"""
            <div class="news-card {css_class}">
                <div style="display:flex; justify-content:space-between;">
                    <small style="color:#9ca3af">{item['publisher']} • {time_str}</small>
                    <b style="color:{'#4ade80' if css_class=='positive' else '#f87171' if css_class=='negative' else '#9ca3af'}">{item['sentiment']}</b>
                </div>
                <h4 style="margin: 5px 0;">{emoji} <a href="{item['link']}" target="_blank" style="text-decoration:none; color:#f3f4f6;">{item['title']}</a></h4>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Enter a ticker and API key to scan the news.")