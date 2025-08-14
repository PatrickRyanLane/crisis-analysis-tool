import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import re
from pytrends.request import TrendReq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings

# --- One-time setup for NLTK VADER ---
@st.cache_resource
def get_sentiment_analyzer():
    """Downloads VADER lexicon if not present and returns an analyzer instance."""
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = get_sentiment_analyzer()
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crisis Impact Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("🚨 Reputational Crisis Impact Analysis Tool")
st.markdown("**Analyze the economic impact of reputational crises on stock prices**")

# --- Sidebar Inputs ---
st.sidebar.header("Crisis Analysis Parameters")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="TSLA").upper()
st.sidebar.caption("The company name will be used for Google Trends search.")

TIMEZONE_OPTIONS = [
    "America/New_York",
    "UTC",
    "Europe/London",
    "Asia/Tokyo",
    "Australia/Sydney"
]

user_tz_str = st.sidebar.selectbox("Select Timezone for Input Dates", TIMEZONE_OPTIONS, index=0)
user_timezone = pytz.timezone(user_tz_str)

today = datetime.today()
crisis_start_date = st.sidebar.date_input("Crisis Start Date", value=today - timedelta(days=90))
crisis_end_date = st.sidebar.date_input("Crisis End Date", value=today)

analyze_mitigation = st.sidebar.toggle("Analyze Mitigation Period", value=True, help="Include a specific period for mitigation actions in the analysis and charts.")

if analyze_mitigation:
    mitigation_start_date = st.sidebar.date_input("Mitigation Start Date", value=crisis_end_date)
    mitigation_end_date = st.sidebar.date_input("Mitigation End Date", value=crisis_end_date + timedelta(days=90))
    if mitigation_end_date < mitigation_start_date:
        st.sidebar.error("Mitigation End Date cannot be before Mitigation Start Date.")
else:
    mitigation_start_date = crisis_end_date
    mitigation_end_date = crisis_end_date

# Initialize response_actions in session state
if "response_actions" not in st.session_state:
    st.session_state.response_actions = []
if "trends_keyword_override" not in st.session_state:
    st.session_state.trends_keyword_override = ""
if "last_analysis_params" not in st.session_state:
    st.session_state.last_analysis_params = {}

# --- Helper functions ---
def simplify_company_name(name):
    """Simplifies a company's long name to a more common search term."""
    if not isinstance(name, str):
        return name
    # Remove common corporate suffixes like Inc., Corp., Ltd., etc.
    # This regex looks for an optional comma, whitespace, the suffix, and an optional period at the end of the string.
    name = re.sub(r'[,]?\s*(Inc|Corporation|Corp|Company|Co|Ltd|LLC|PLC)\.?$', '', name, flags=re.IGNORECASE)
    # Remove leading "The "
    name = re.sub(r'^The\s+', '', name, flags=re.IGNORECASE)
    # Trim any leading/trailing whitespace that might be left
    return name.strip()

def add_action_callback():
    """Callback to add a new response action from the text input."""
    desc = st.session_state.get("new_action_desc_input", "").strip()
    if desc:
        st.session_state.response_actions.append({
            "date": st.session_state.new_action_date_input,
            "description": desc
        })
        # Clear the input widget's state for the next entry
        st.session_state.new_action_desc_input = ""
        st.success("New response action added.")


def get_text_positions(dates, labels):
    """
    Generate alternating text positions to prevent overlap
    (Function kept for reference but no visible labels on timeline now)
    """
    if not dates:
        return []
    date_label_pairs = list(zip(dates, labels))
    date_label_pairs.sort(key=lambda x: x[0])
    positions = []
    for i, (date, label) in enumerate(date_label_pairs):
        if i % 4 == 0:
            positions.append('top center')
        elif i % 4 == 1:
            positions.append('bottom center')
        elif i % 4 == 2:
            positions.append('middle left')
        else:
            positions.append('middle right')
    sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
    position_map = {sorted_indices[i]: positions[i] for i in range(len(positions))}
    original_order_positions = [position_map[i] for i in range(len(dates))]
    return original_order_positions


def editable_actions_list():
    st.subheader("🗓️ Editable Response Actions")
    indices_to_delete = []
    should_rerun = False

    # Header for the editable list
    header_cols = st.columns([3, 6, 1])
    header_cols[0].caption("Action Date")
    header_cols[1].caption("Description")
    header_cols[2].caption("Delete")

    for idx, action in enumerate(st.session_state.response_actions):
        cols = st.columns([3, 6, 1])

        new_date = cols[0].date_input(
            label=f"Date {idx + 1}",
            value=action['date'],
            key=f"date_{idx}",
            label_visibility="collapsed"
        )
        new_desc = cols[1].text_input(
            label=f"Description {idx + 1}",
            value=action['description'],
            max_chars=200,
            key=f"desc_{idx}",
            label_visibility="collapsed"
        )
        if cols[2].button("❌", key=f"del_{idx}"):
            indices_to_delete.append(idx)
            should_rerun = True

    # Process deletions before rerunning
    if indices_to_delete:
        for i in sorted(indices_to_delete, reverse=True):
            st.session_state.response_actions.pop(i)

    # Remove deleted items after loop to avoid index conflicts
    if should_rerun:
        st.rerun()


# --- Data Fetching Functions with Caching ---
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data and company info from Yahoo Finance."""
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start=start_date, end=end_date)
    if data.empty:
        return None, None, None

    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')

    try:
        stock_info = ticker_obj.info
        long_name = stock_info.get('longName', ticker)
        company_name = simplify_company_name(long_name)
        shares_outstanding = stock_info.get('sharesOutstanding') # Will be None if not found
    except Exception:
        company_name = ticker
        shares_outstanding = None

    return data, company_name, shares_outstanding

@st.cache_data(ttl=3600) # Cache news for 1 hour
def get_news_with_sentiment(search_query):
    """Fetches recent news, performs sentiment analysis, and caches the result."""
    try:
        news_list = yf.Search(search_query).news
        processed_news = []
        if not news_list:
            return "no_news", []

        for item in news_list:
            # Only process items that have a valid title
            if item and item.get('title'):
                title = item['title']
                score = sia.polarity_scores(title)['compound']
                item['sentiment_score'] = score
                if score >= 0.05: item['sentiment_class'] = 'Positive'
                elif score <= -0.05: item['sentiment_class'] = 'Negative'
                else: item['sentiment_class'] = 'Neutral'
                processed_news.append(item)
        
        if not processed_news:
            return "no_news", []

        return "ok", processed_news
    except Exception as e:
        error_message = f"Could not fetch news. This is often an intermittent issue with the `yfinance` library or network connectivity. Please try again later."
        print(f"yfinance news error for '{search_query}': {e}") # For server-side logging
        return "error", error_message

@st.cache_data
def get_trends_data(keyword, start_date, end_date):
    """Fetches Google Trends data and related queries, then caches it."""
    try:
        pytrends = TrendReq(hl='en-US', tz=360)  # US Central Time
        timeframe = f"{start_date} {end_date}"
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')

        # Fetch interest over time
        trends_df = pytrends.interest_over_time()
        daily_trends = None
        if not trends_df.empty and keyword in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
            # Resample to daily, forward-fill, and align timezone to UTC
            daily_trends = trends_df.resample('D').ffill()
            if daily_trends.index.tz is None:
                daily_trends.index = daily_trends.index.tz_localize('UTC')
            else:
                daily_trends.index = daily_trends.index.tz_convert('UTC')

        # Fetch related queries
        related_queries_dict = pytrends.related_queries()
        related_queries_df = None
        if related_queries_dict and keyword in related_queries_dict:
            top_queries = related_queries_dict[keyword].get('top')
            if top_queries is not None:
                related_queries_df = top_queries

        return daily_trends, related_queries_df
    except Exception as e:
        # Silently fail but we can check for None later
        print(f"Could not fetch Google Trends data for '{keyword}': {e}")
    return None, None


# --- Stock data analysis ---

# Define the current set of parameters for the analysis
current_params = {
    "ticker": ticker,
    "crisis_start_date": crisis_start_date,
    "crisis_end_date": crisis_end_date,
    "analyze_mitigation": analyze_mitigation,
    "mitigation_start_date": mitigation_start_date,
    "mitigation_end_date": mitigation_end_date,
    "user_tz_str": user_tz_str,
    "trends_keyword_override": st.session_state.get("trends_keyword_override", "").strip(),
}

# Determine if a new analysis should be run
should_run_analysis = (
    "analysis_result" not in st.session_state or
    st.session_state.last_analysis_params != current_params
)

if should_run_analysis:
    try:
        # Localize input dates to user timezone
        crisis_start = user_timezone.localize(datetime.combine(crisis_start_date, datetime.min.time()))
        crisis_end = user_timezone.localize(datetime.combine(crisis_end_date, datetime.min.time()))
        start_date_obj = min(crisis_start.date(), mitigation_start_date) - timedelta(days=90)
        end_date_obj = max(crisis_end.date(), mitigation_end_date) + timedelta(days=90)

        data, company_name, shares_outstanding = get_stock_data(ticker, start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d"))
        news_status, news_data = get_news_with_sentiment(company_name)
        if data is None:
            st.error("No data found for this ticker. Please check the symbol.")
            st.stop()

        # Convert dates to UTC for consistent indexing
        mitigation_start = user_timezone.localize(datetime.combine(mitigation_start_date, datetime.min.time()))
        mitigation_end = user_timezone.localize(datetime.combine(mitigation_end_date, datetime.min.time()))
        crisis_start_utc = crisis_start.astimezone(pytz.UTC)
        crisis_end_utc = crisis_end.astimezone(pytz.UTC)
        mitigation_start_utc = mitigation_start.astimezone(pytz.UTC)
        mitigation_end_utc = mitigation_end.astimezone(pytz.UTC)

        pre_crisis_data = data[data.index < crisis_start_utc]
        crisis_data = data[(data.index >= crisis_start_utc) & (data.index <= crisis_end_utc)]
        post_crisis_data = data[data.index > crisis_end_utc]
        mitigation_data = data[(data.index >= mitigation_start_utc) & (data.index <= mitigation_end_utc)]

        if pre_crisis_data.empty or crisis_data.empty:
            st.error("Insufficient data for the selected date range.")
            st.stop()

        pre_crisis_avg = pre_crisis_data['Close'].mean()
        crisis_min = crisis_data['Close'].min()
        crisis_avg = crisis_data['Close'].mean()
        mitigation_avg = mitigation_data['Close'].mean() if not mitigation_data.empty else np.nan
        mitigation_vol = mitigation_data['Close'].std() if not mitigation_data.empty else np.nan

        max_decline = ((crisis_min - pre_crisis_avg) / pre_crisis_avg) * 100
        avg_decline = ((crisis_avg - pre_crisis_avg) / pre_crisis_avg) * 100

        if not post_crisis_data.empty:
            post_crisis_avg = post_crisis_data['Close'].mean()
            recovery_percentage = ((post_crisis_avg - crisis_min) / crisis_min) * 100
            current_postcrisis_price = post_crisis_data['Close'].iloc[-1]
            current_recovery_percentage = ((current_postcrisis_price - crisis_min) / crisis_min) * 100
        else:
            post_crisis_avg = current_postcrisis_price = np.nan
            recovery_percentage = current_recovery_percentage = None

        # Calculate market cap change based on the average price difference
        # This provides a more stable measure of impact than using the single lowest point.
        if shares_outstanding:
            market_cap_change = (crisis_avg - pre_crisis_avg) * shares_outstanding
        else:
            market_cap_change = None

        # Fetch Google Trends data
        trends_keyword_override = st.session_state.get("trends_keyword_override", "")
        trends_search_term = current_params["trends_keyword_override"] if current_params["trends_keyword_override"] else company_name
        trends_data, related_queries = get_trends_data(
            trends_search_term, start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d")
        )

        st.session_state.analysis_result = dict(
            data=data,
            crisis_start_utc=crisis_start_utc,
            crisis_end_utc=crisis_end_utc,
            mitigation_start_utc=mitigation_start_utc,
            mitigation_end_utc=mitigation_end_utc,
            pre_crisis_avg=pre_crisis_avg,
            crisis_min=crisis_min,
            crisis_avg=crisis_avg,
            mitigation_avg=mitigation_avg,
            mitigation_vol=mitigation_vol,
            max_decline=max_decline,
            avg_decline=avg_decline,
            post_crisis_data=post_crisis_data,
            post_crisis_avg=post_crisis_avg,
            current_postcrisis_price=current_postcrisis_price,
            recovery_percentage=recovery_percentage,
            current_recovery_percentage=current_recovery_percentage,
            shares_outstanding=shares_outstanding,
            market_cap_change=market_cap_change,
            trends_data=trends_data,
            related_queries=related_queries,
            company_name=company_name, # The simplified name
            trends_search_term=trends_search_term,
            analyze_mitigation=analyze_mitigation,
            mitigation_start_date_val=mitigation_start_date,
            mitigation_end_date_val=mitigation_end_date,
            news_status=news_status,
            news=news_data
        )

        # Store the parameters of this successful analysis
        st.session_state.last_analysis_params = current_params

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your internet connection and verify the stock ticker symbol.")

# -------- Main Analysis Results and Metrics -----------

if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    data = res['data']
    trends_data = res.get('trends_data')
    smoothed_trends = None
    if trends_data is not None and not trends_data.empty:
        # Smooth the trend data with a 7-day rolling average to make it less noisy and easier to interpret
        keyword = trends_data.columns[0]
        smoothed_trends = trends_data[keyword].rolling(window=7, center=True, min_periods=1).mean()

    # Notify user if trends data failed to load
    if trends_data is None:
        search_term_used = res.get('trends_search_term', ticker)
        st.info(f"Could not retrieve Google Trends data for '{search_term_used}'. The charts will not include the trends overlay.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Pre-Crisis Avg", 
            f"${res['pre_crisis_avg']:.2f}",
            help="The average closing stock price for the 90 days leading up to the crisis start date."
        )
    with col2:
        st.metric(
            "Crisis Minimum", 
            f"${res['crisis_min']:.2f}", 
            delta=f"{res['max_decline']:.1f}%",
            help="The lowest closing price during the crisis period. The delta shows the percentage change from the pre-crisis average. Formula: `((Crisis Minimum - Pre-Crisis Avg) / Pre-Crisis Avg) * 100`"
        )
    with col3:
        st.metric(
            "Crisis Avg", 
            f"${res['crisis_avg']:.2f}", 
            delta=f"{res['avg_decline']:.1f}%",
            help="The average closing price during the crisis period. The delta shows the percentage change from the pre-crisis average. Formula: `((Crisis Avg - Pre-Crisis Avg) / Pre-Crisis Avg) * 100`"
        )
    with col4:
        if not res['post_crisis_data'].empty:
            st.metric(
                "Post-Crisis Recovery Avg", 
                f"{res['recovery_percentage']:.1f}%",
                help="The percentage gain from the crisis low price to the average price in the post-crisis period. Formula: `((Avg Post-Crisis Price - Crisis Low) / Crisis Low) * 100`"
            )
            st.caption(f"Avg post-crisis close: ${res['post_crisis_avg']:.2f}")
        else:
            st.metric("Post-Crisis Recovery", "Not enough data")
    with col5:
        if not res['post_crisis_data'].empty:
            st.metric(
                "Recovery to Current Price", 
                f"{res['current_recovery_percentage']:.1f}%",
                help="The percentage gain from the crisis low price to the most recent closing price. Formula: `((Current Price - Crisis Low) / Crisis Low) * 100`"
            )
            st.caption(f"Current price: ${res['current_postcrisis_price']:.2f}")
            st.caption(f"Difference from crisis min: "
                       f"${res['current_postcrisis_price'] - res['crisis_min']:.2f}")
        else:
            st.metric("Post-Crisis Recovery", "Not enough data")

    impact_col1, impact_col2 = st.columns(2)

    with impact_col1:
        st.subheader("💰 Economic Impact Analysis")
        mc_change = res.get('market_cap_change')
        
        if mc_change is not None:
            if mc_change < 0:
                label = "Est. Market Cap Loss:"
                value_str = f"${abs(mc_change):,.0f}"
            else:
                label = "Est. Market Cap Gain:"
                value_str = f"${mc_change:,.0f}"
            calc_string = f"(\${res['crisis_avg']:.2f} - \${res['pre_crisis_avg']:.2f}) * {res['shares_outstanding']:,} shares"
            calc_caption = f"Calculation: {calc_string}"
        else:
            label = "Est. Market Cap Change:"
            value_str = "`Data not available`"
            calc_caption = "Could not retrieve the number of outstanding shares for this ticker."

        st.markdown(f"**{label}** {value_str}", help="Market Cap Change = (Average Crisis Price - Average Pre-Crisis Price) * Shares Outstanding")
        st.caption(calc_caption)
        st.write(f"**Maximum Stock Price Decline:** {abs(res['max_decline']):.1f}%")
        st.write(f"**Crisis Duration:** {(res['crisis_end_utc'] - res['crisis_start_utc']).days} days")
        if res.get('analyze_mitigation'):
            st.write(f"**Mitigation Period:** {res['mitigation_start_date_val']} to {res['mitigation_end_date_val']} "
                     f"({(res['mitigation_end_utc'] - res['mitigation_start_utc']).days} days)")

    with impact_col2:
        st.subheader("📈 Google Trends Insights")
        search_term = res.get('trends_search_term', 'N/A')
        st.write(f"**Search Term Used:** `{search_term}`")
        st.text_input(
            "Override with new keyword",
            key="trends_keyword_override",
            help="Enter a new term and the analysis will update automatically."
        )

        related_queries = res.get('related_queries')
        if related_queries is not None and not related_queries.empty:
            st.write("**Top 5 Related Search Queries:**")
            display_df = related_queries[['query', 'value']].head(5).copy()
            display_df.rename(columns={'value': 'Relative Popularity (0-100)'}, inplace=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.write("**No related queries found for this term.**")

    # --- Recent News Expander ---
    st.markdown("---")
    with st.expander("📰 Recent News with Sentiment Analysis (to help identify crisis dates)"):
        news_search_term = res.get('company_name', ticker)
        st.caption(f"News search term: `{news_search_term}`")

        news_status = res.get('news_status')
        news_items = res.get('news', []) # This will be a list or an error string

        if news_status == 'ok':
            if news_items:
                sentiment_bg_colors = {'Positive': 'rgba(40, 167, 69, 0.1)', 'Negative': 'rgba(220, 53, 69, 0.1)', 'Neutral': 'rgba(108, 117, 125, 0.1)'}
                for item in news_items:
                    title = item.get('title', 'No Title')
                    link = item.get('link')
                    publisher = item.get('publisher', 'Unknown Publisher')
                    publish_time_unix = item.get('providerPublishTime')

                    if publish_time_unix:
                        publish_time = datetime.fromtimestamp(publish_time_unix).strftime('%Y-%m-%d %H:%M')
                    else:
                        publish_time = "Date not available"

                    sentiment_class = item.get('sentiment_class', 'Neutral')
                    bg_color = sentiment_bg_colors.get(sentiment_class, sentiment_bg_colors['Neutral'])
                    display_title = f"<a href='{link}' target='_blank' style='color: inherit; text-decoration: none; font-weight: bold;'>{title}</a>" if link else f"<span style='font-weight: bold;'>{title}</span>"
                    caption_text = f"Published by {publisher} on {publish_time} | Sentiment: {sentiment_class}"
                    html_block = f"""<div style="background-color: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">{display_title}<div style="font-size: 0.8em; color: #6c757d; margin-top: 4px;">{caption_text}</div></div>"""
                    st.markdown(html_block, unsafe_allow_html=True)
        elif news_status == 'error':
            st.warning(news_items) # Display the error message
        else:
            st.write("No recent news found for this ticker.")

    # --------- Plotting Section: Chart first ---------

    st.markdown("---")
    st.subheader("📈 Stock Price and Google Trends Chart")
    start_y_at_zero = st.toggle("Start Y-Axis at 0", value=False, help="Toggle to see the price scale relative to zero. This can help contextualize the magnitude of price changes.")


    # Prepare response actions dates and labels
    act_dates, act_labels = [], []
    for action in st.session_state.response_actions:
        if action.get("date") and action.get("description"):
            action_dt = datetime.combine(action['date'], datetime.min.time())
            action_dt_aware = user_timezone.localize(action_dt) if action_dt.tzinfo is None else action_dt
            action_dt_utc = action_dt_aware.astimezone(pytz.UTC)
            action_dt_naive = action_dt_utc.replace(tzinfo=None)
            act_dates.append(action_dt_naive)
            label = action['description'][:20] + "..." if len(action['description']) > 20 else action['description']
            act_labels.append(label)

    # Use helper function (retained for completeness) but no labels shown in timeline now
    _ = get_text_positions(act_dates, act_labels)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.01,
        row_titles=["", "Response Actions<br>Timeline"],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Price line trace
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines', name='Stock Price', line=dict(color='blue', width=2)
    ), row=1, col=1, secondary_y=False)

    # Google Trends trace
    if smoothed_trends is not None:
        fig.add_trace(go.Scatter(
            x=smoothed_trends.index, y=smoothed_trends,
            mode='lines', name='Google Trend (7-day avg)',
            line=dict(color='rgba(128, 0, 128, 0.8)', width=1.5),
            fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'
        ), row=1, col=1, secondary_y=True)

    # Crisis & mitigation shaded rectangles
    fig.add_vrect(x0=res['crisis_start_utc'].replace(tzinfo=None), x1=res['crisis_end_utc'].replace(tzinfo=None),
                  fillcolor="red", opacity=0.2, layer="below", line_width=0,
                  annotation_text="Crisis Period", annotation_position="top left", row=1, col=1)
    if res.get('analyze_mitigation'):
        fig.add_vrect(x0=res['mitigation_start_utc'].replace(tzinfo=None), x1=res['mitigation_end_utc'].replace(tzinfo=None),
                      fillcolor="green", opacity=0.13, layer="below", line_width=0,
                      annotation_text="Mitigation Period", annotation_position="top right", row=1, col=1)

    # Lines for averages and min
    fig.add_hline(y=res['pre_crisis_avg'], line_dash="dash", line_color="green",
                  annotation_text="Pre-Crisis Average", row=1, col=1)
    fig.add_hline(y=res['crisis_min'], line_dash="dash", line_color="red",
                  annotation_text="Crisis Low Price", row=1, col=1)

    # Add a dummy trace to ensure the timeline subplot is always drawn,
    # making the central line and x-axis visible even with no actions.
    fig.add_trace(go.Scatter(
        x=[data.index[0]], y=[0], mode='markers', marker=dict(opacity=0),
        hoverinfo='none', showlegend=False
    ), row=2, col=1)

    # Timeline event points with alternating stems for better visualization
    if act_dates:
        # Create alternating y-positions for markers to prevent overlap
        y_positions = [1 if i % 2 == 0 else -1 for i in range(len(act_dates))]

        # Add vertical stems from the center line to the markers
        stem_x, stem_y = [], []
        for date, y_pos in zip(act_dates, y_positions):
            stem_x.extend([date, date, None])
            stem_y.extend([0, y_pos, None])
        
        fig.add_trace(go.Scatter(
            x=stem_x,
            y=stem_y,
            mode='lines',
            line=dict(color='grey', width=1),
            hoverinfo='none',
            showlegend=False
        ), row=2, col=1)

        # Add markers at the end of the stems
        fig.add_trace(go.Scatter(
            x=act_dates,
            y=y_positions,
            mode="markers",
            marker=dict(symbol="circle", size=12, color="#045d1f", line=dict(width=1, color='white')),
            hovertext=[f"<b>{label}</b><br>{date.strftime('%b %d, %Y')}" for label, date in zip(act_labels, act_dates)],
            hoverinfo='text',
            name="Response Actions",
            showlegend=False
        ), row=2, col=1)

    # Add a central horizontal line for the timeline axis
    fig.add_hline(y=0, line_width=2, line_color='grey', row=2, col=1)

    # Conditionally set the y-axis range based on the toggle
    if start_y_at_zero:
        fig.update_yaxes(rangemode='tozero', row=1, col=1, secondary_y=False)

    fig.update_yaxes(showticklabels=False, fixedrange=True, row=2, col=1,
                     range=[-2, 2], showgrid=False, zeroline=False, title=None)
    fig.update_xaxes(title="Date", row=2, col=1, tickformat="%b %d, %Y")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Google Trend (0-100)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_layout(
        height=700,
        title=f"{ticker} Stock Price & Google Trends During Crisis",
        showlegend=True,
        margin=dict(l=20, r=40, t=50, b=20) # Add margin to prevent label overlap
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------- Editable Response Actions List BELOW the chart ---------
    editable_actions_list()

    # --------- Add New Action Section BELOW the editable list ---------
    st.markdown("---")
    st.subheader("Add New Action")
    add_cols = st.columns([1, 3])
    with add_cols[0]:
        st.date_input(
            "Action Date",
            value=mitigation_start_date if mitigation_start_date else datetime.today().date(),
            key="new_action_date_input"
        )
    with add_cols[1]:
        st.text_input(
            "Action Description (press Enter to add)",
            key="new_action_desc_input",
            on_change=add_action_callback,
            placeholder="e.g., 'CEO issues public apology'"
        )

    # -------- Additional Analysis Tables and Summaries ---------
    st.subheader("📈 Timeline Analysis")

    post_crisis_data = res['post_crisis_data']

    timeline_data = pd.DataFrame({
        'Period': [
            'Pre-Crisis (90 days)',
            'Crisis Period',
            'Mitigation Period',
            'Post-Crisis (90 days)'
        ],
        'Average Price': [
            res['pre_crisis_avg'],
            res['crisis_avg'],
            res['mitigation_avg'],
            post_crisis_data['Close'].mean() if not post_crisis_data.empty else np.nan
        ],
        'Volatility': [
            data[data.index < res['crisis_start_utc']]['Close'].std(),
            data[(data.index >= res['crisis_start_utc']) & (data.index <= res['crisis_end_utc'])]['Close'].std(),
            res['mitigation_vol'],
            post_crisis_data['Close'].std() if not post_crisis_data.empty else np.nan
        ]
    }).dropna()

    st.dataframe(timeline_data.round(2), use_container_width=True)
    st.caption("Volatility is measured by the standard deviation of the closing price for each period.")

    if 'Volume' in data.columns:
        st.subheader("📊 Trading Volume Analysis")

        vol_fig = make_subplots(specs=[[{"secondary_y": True}]])

        vol_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume'],
            mode='lines',
            name='Trading Volume',
            line=dict(color='orange', width=1)
        ), secondary_y=False)

        if smoothed_trends is not None:
            vol_fig.add_trace(go.Scatter(
                x=smoothed_trends.index, y=smoothed_trends,
                mode='lines', name='Google Trend (7-day avg)',
                line=dict(color='rgba(128, 0, 128, 0.8)', width=1.5),
                fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'
            ), secondary_y=True)

        vol_fig.add_vrect(
            x0=res['crisis_start_utc'], x1=res['crisis_end_utc'],
            fillcolor="red", opacity=0.18, line_width=0
        )
        if res.get('analyze_mitigation'):
            vol_fig.add_vrect(
                x0=res['mitigation_start_utc'], x1=res['mitigation_end_utc'],
                fillcolor="green", opacity=0.12, line_width=0
            )

        vol_fig.update_layout(
            title_text=f"{ticker} Trading Volume & Google Trends",
            showlegend=True,
            margin=dict(l=20, r=40, t=50, b=20)
        )
        vol_fig.update_xaxes(tickformat="%b %d, %Y")
        vol_fig.update_yaxes(title_text="Trading Volume", secondary_y=False)
        vol_fig.update_yaxes(title_text="Google Trend (0-100)", secondary_y=True, showgrid=False)
        st.plotly_chart(vol_fig, use_container_width=True)

    # Crisis Impact Summary
    st.subheader("🎯 Crisis Impact Summary")

    impact_severity = (
        "High" if abs(res['max_decline']) > 30 else
        "Moderate" if abs(res['max_decline']) > 15 else
        "Low"
    )
    st.write(f"""
    **Crisis Severity:** {impact_severity} Impact

    **Key Findings:**

    - Maximum price decline of {abs(res['max_decline']):.1f}% during the crisis period

    - Stock fell from ${res['pre_crisis_avg']:.2f} average to ${res['crisis_min']:.2f} minimum

    - Crisis lasted {(res['crisis_end_utc'] - res['crisis_start_utc']).days} days

    - Mitigation actions and timeframe shown above

    - Recovery from crisis minimum:

        - Avg post-crisis: {res['recovery_percentage']:.1f}% (${res['post_crisis_avg']:.2f})

        - Current: {res['current_recovery_percentage']:.1f}% (${res['current_postcrisis_price']:.2f})
    """ if not post_crisis_data.empty else """
    Post-crisis recovery metrics not available (not enough post-crisis data).
    """)

else:
    st.info("Enter a stock ticker and adjust the dates in the sidebar to begin your analysis.")

# End of app
