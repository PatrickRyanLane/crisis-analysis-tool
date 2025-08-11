import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from pytrends.request import TrendReq
import warnings

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

crisis_start_date = st.sidebar.date_input("Crisis Start Date", value=datetime(2022, 1, 1))
crisis_end_date = st.sidebar.date_input("Crisis End Date", value=datetime(2022, 6, 30))

mitigation_start_date = st.sidebar.date_input("Mitigation Start Date", value=crisis_end_date)
mitigation_end_date = st.sidebar.date_input("Mitigation End Date", value=crisis_end_date + timedelta(days=90))

if mitigation_end_date < mitigation_start_date:
    st.sidebar.error("Mitigation End Date cannot be before Mitigation Start Date.")

analyze_button_clicked = st.sidebar.button("Analyze Crisis Impact", type="primary")

# Initialize response_actions in session state
if "response_actions" not in st.session_state:
    st.session_state.response_actions = []

# --- Helper functions ---
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
    should_rerun = False  # flag to trigger rerun once

    for idx, action in enumerate(st.session_state.response_actions):
        cols = st.columns([3, 6, 1])

        new_date = cols[0].date_input(
            label=f"Date {idx + 1}",
            value=action['date'],
            key=f"date_{idx}"
        )
        new_desc = cols[1].text_input(
            label=f"Description {idx + 1}",
            value=action['description'],
            max_chars=200,
            key=f"desc_{idx}"
        )
        if cols[2].button("❌", key=f"del_{idx}"):
            indices_to_delete.append(idx)
            should_rerun = True

        if (new_date != action['date']) or (new_desc != action['description']):
            st.session_state.response_actions[idx] = {"date": new_date, "description": new_desc}
            should_rerun = True

    # Remove deleted items after loop to avoid index conflicts
    if indices_to_delete:
        for i in sorted(indices_to_delete, reverse=True):
            st.session_state.response_actions.pop(i)

    if should_rerun:
        st.rerun()


# --- Data Fetching Functions with Caching ---
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data and company info from Yahoo Finance and caches it."""
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if data.empty:
        return None, None, None
    # Ensure timezone is set to UTC for consistency
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')

    # Get company info
    try:
        stock_info = yf.Ticker(ticker).info
        company_name = stock_info.get('longName', ticker)
        shares_outstanding = stock_info.get('sharesOutstanding', 1000000000)
    except Exception:
        company_name = ticker
        shares_outstanding = 1000000000

    return data, company_name, shares_outstanding

@st.cache_data
def get_trends_data(keyword, start_date, end_date):
    """Fetches Google Trends data and caches it."""
    try:
        pytrends = TrendReq(hl='en-US', tz=360)  # US Central Time
        timeframe = f"{start_date} {end_date}"
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()

        if not trends_df.empty and keyword in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
            # Resample to daily, forward-fill, and align timezone to UTC
            daily_trends = trends_df.resample('D').ffill()
            if daily_trends.index.tz is None:
                daily_trends.index = daily_trends.index.tz_localize('UTC')
            else:
                daily_trends.index = daily_trends.index.tz_convert('UTC')
            return daily_trends
    except Exception as e:
        # Silently fail but we can check for None later
        print(f"Could not fetch Google Trends data for '{keyword}': {e}")
    return None


# --- Stock data analysis ---
if "analysis_result" not in st.session_state or analyze_button_clicked:
    try:
        # Localize input dates to user timezone
        crisis_start = user_timezone.localize(datetime.combine(crisis_start_date, datetime.min.time()))
        crisis_end = user_timezone.localize(datetime.combine(crisis_end_date, datetime.min.time()))
        start_date_obj = min(crisis_start.date(), mitigation_start_date) - timedelta(days=90)
        end_date_obj = max(crisis_end.date(), mitigation_end_date) + timedelta(days=90)

        data, company_name, shares_outstanding = get_stock_data(ticker, start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d"))
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

        market_cap_loss = abs(max_decline) / 100 * pre_crisis_avg * shares_outstanding

        # Fetch Google Trends data
        trends_data = get_trends_data(
            company_name, start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d")
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
            market_cap_loss=market_cap_loss,
            trends_data=trends_data,
            company_name=company_name
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your internet connection and verify the stock ticker symbol.")

# -------- Main Analysis Results and Metrics -----------

if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    data = res['data']
    trends_data = res.get('trends_data')

    # Notify user if trends data failed to load
    if trends_data is None:
        st.info(f"Could not retrieve Google Trends data for '{res.get('company_name', ticker)}'. The charts will not include the trends overlay.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Pre-Crisis Avg", f"${res['pre_crisis_avg']:.2f}")
    with col2:
        st.metric("Crisis Minimum", f"${res['crisis_min']:.2f}", delta=f"{res['max_decline']:.1f}%")
    with col3:
        st.metric("Crisis Avg", f"${res['crisis_avg']:.2f}", delta=f"{res['avg_decline']:.1f}%")
    with col4:
        if not res['post_crisis_data'].empty:
            st.metric("Post-Crisis Recovery Avg", f"{res['recovery_percentage']:.1f}%")
            st.caption(f"Avg post-crisis close: ${res['post_crisis_avg']:.2f}")
        else:
            st.metric("Post-Crisis Recovery", "Not enough data")
    with col5:
        if not res['post_crisis_data'].empty:
            st.metric("Recovery to Current Price", f"{res['current_recovery_percentage']:.1f}%")
            st.caption(f"Current price: ${res['current_postcrisis_price']:.2f}")
            st.caption(f"Difference from crisis min: "
                       f"${res['current_postcrisis_price'] - res['crisis_min']:.2f}")
        else:
            st.metric("Post-Crisis Recovery", "Not enough data")

    st.subheader("💰 Economic Impact Analysis")
    st.write(f"**Estimated Market Cap Loss:** ${res['market_cap_loss']:,.0f}")
    st.write(f"**Maximum Stock Price Decline:** {abs(res['max_decline']):.1f}%")
    st.write(f"**Crisis Duration:** {(res['crisis_end_utc'] - res['crisis_start_utc']).days} days")
    st.write(f"**Mitigation Period:** {mitigation_start_date} to {mitigation_end_date} "
             f"({(res['mitigation_end_utc'] - res['mitigation_start_utc']).days} days)")

    # --------- Plotting Section: Chart first ---------

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
        row_titles=["Stock Price Chart", "Response Actions Timeline"],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Price line trace
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines', name='Stock Price', line=dict(color='blue', width=2)
    ), row=1, col=1, secondary_y=False)

    # Google Trends trace
    if trends_data is not None and not trends_data.empty:
        keyword = res.get('company_name', 'Search Trend')
        fig.add_trace(go.Scatter(
            x=trends_data.index, y=trends_data[keyword],
            mode='lines', name='Google Trend Score', line=dict(color='purple', width=1, dash='dot')
        ), row=1, col=1, secondary_y=True)

    # Crisis & mitigation shaded rectangles
    fig.add_vrect(x0=res['crisis_start_utc'].replace(tzinfo=None), x1=res['crisis_end_utc'].replace(tzinfo=None),
                  fillcolor="red", opacity=0.2, layer="below", line_width=0,
                  annotation_text="Crisis Period", annotation_position="top left", row=1, col=1)
    fig.add_vrect(x0=res['mitigation_start_utc'].replace(tzinfo=None), x1=res['mitigation_end_utc'].replace(tzinfo=None),
                  fillcolor="green", opacity=0.13, layer="below", line_width=0,
                  annotation_text="Mitigation Period", annotation_position="top right", row=1, col=1)

    # Lines for averages and min
    fig.add_hline(y=res['pre_crisis_avg'], line_dash="dash", line_color="green",
                  annotation_text="Pre-Crisis Average", row=1, col=1)
    fig.add_hline(y=res['crisis_min'], line_dash="dash", line_color="red",
                  annotation_text="Crisis Minimum", row=1, col=1)

    # Timeline event points - markers only, labels on hover (no text to avoid overlap)
    if act_dates:
        fig.add_trace(go.Scatter(
            x=act_dates,
            y=np.ones(len(act_dates)),
            mode="markers",
            marker=dict(symbol="circle", size=12, color="#045d1f"),
            hovertext=[f"{label}: {date.strftime('%Y-%m-%d')}" for label, date in zip(act_labels, act_dates)],
            hoverinfo='text',
            name="Response Actions",
            showlegend=False
        ), row=2, col=1)

    fig.update_yaxes(showticklabels=False, fixedrange=True, row=2, col=1,
                     range=[0.5, 1.5], showgrid=False, zeroline=False, title=None)
    fig.update_xaxes(title="Date", row=2, col=1)
    fig.update_yaxes(title="Price ($)", row=1, col=1)
    fig.update_layout(
        height=700,
        title=f"{ticker} Stock Price During Crisis & Mitigation",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------- Editable Response Actions List BELOW the chart ---------
    editable_actions_list()

    # --------- Add New Action Form BELOW the editable list ---------
    with st.form("add_action_form", clear_on_submit=True):
        new_action_date = st.date_input(
            "Add New Action Date",
            value=mitigation_start_date if mitigation_start_date else datetime.today().date(),
            key="new_action_date"
        )
        new_action_desc = st.text_input("Add New Action Description", key="new_action_desc")
        submitted = st.form_submit_button("Save New Action")

        if submitted:
            if new_action_desc.strip():
                st.session_state.response_actions.append(
                    {"date": new_action_date, "description": new_action_desc.strip()}
                )
                st.success("New response action added.")
                st.rerun()
            else:
                st.warning("Please enter a description.")

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

    if 'Volume' in data.columns:
        st.subheader("📊 Trading Volume Analysis")

        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume'],
            mode='lines',
            name='Trading Volume',
            line=dict(color='orange', width=1)
        ))
        vol_fig.add_vrect(
            x0=res['crisis_start_utc'], x1=res['crisis_end_utc'],
            fillcolor="red", opacity=0.18, line_width=0
        )
        vol_fig.add_vrect(
            x0=res['mitigation_start_utc'], x1=res['mitigation_end_utc'],
            fillcolor="green", opacity=0.12, line_width=0
        )
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
    st.info("Please perform crisis impact analysis first to load stock data and charts.")

# End of app
