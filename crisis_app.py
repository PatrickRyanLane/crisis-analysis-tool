import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crisis Impact Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("🚨 Reputational Crisis Impact Analysis Tool")
st.markdown("**Analyze the economic impact of reputational crises on stock prices**")

st.sidebar.header("Crisis Analysis Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="TSLA").upper()
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

if "analysis_result" not in st.session_state or st.sidebar.button("Analyze Crisis Impact"):
    try:
        crisis_start = user_timezone.localize(datetime.combine(crisis_start_date, datetime.min.time()))
        crisis_end = user_timezone.localize(datetime.combine(crisis_end_date, datetime.min.time()))
        mitigation_start = user_timezone.localize(datetime.combine(mitigation_start_date, datetime.min.time()))
        mitigation_end = user_timezone.localize(datetime.combine(mitigation_end_date, datetime.min.time()))
        start_date = min(crisis_start, mitigation_start) - timedelta(days=90)
        end_date = max(crisis_end, mitigation_end) + timedelta(days=90)

        data = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if data.empty:
            st.error("No data found for this ticker. Please check the symbol.")
            st.stop()
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')

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

        try:
            stock = yf.Ticker(ticker)
            company_info = stock.info
            shares_outstanding = company_info.get('sharesOutstanding', 1000000000)
            market_cap_loss = abs(max_decline) / 100 * pre_crisis_avg * shares_outstanding
        except Exception:
            shares_outstanding = 1000000000
            market_cap_loss = abs(max_decline) / 100 * pre_crisis_avg * shares_outstanding

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
            market_cap_loss=market_cap_loss
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your internet connection and verify the stock ticker symbol.")

if "response_actions" not in st.session_state:
    st.session_state.response_actions = []

# -------- Main Analysis Results and Metrics -----------
if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    data = res['data']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pre-Crisis Avg", f"${res['pre_crisis_avg']:.2f}")
    with col2:
        st.metric("Crisis Minimum", f"${res['crisis_min']:.2f}", delta=f"{res['max_decline']:.1f}%")
    with col3:
        st.metric("Crisis Avg", f"${res['crisis_avg']:.2f}", delta=f"{res['avg_decline']:.1f}%")
    with col4:
        if not res['post_crisis_data'].empty:
            st.markdown("**Post-Crisis Recovery (Average):**")
            st.metric("Recovery (mean)", f"{res['recovery_percentage']:.1f}%")
            st.caption(f"Avg post-crisis close: ${res['post_crisis_avg']:.2f}")
            st.markdown("---")
            st.markdown("**Current Price Recovery**")
            st.metric("Current recovery", f"{res['current_recovery_percentage']:.1f}%")
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

    # -------------------- Main Chart + Stacked Timeline Subplot -------------------
    # Prepare response actions dates and labels
    act_dates, act_labels = [], []
    for action in st.session_state.response_actions:
        if action.get("date"):
            action_dt = datetime.combine(action['date'], datetime.min.time())
            action_dt_aware = user_timezone.localize(action_dt) if action_dt.tzinfo is None else action_dt
            action_dt_utc = action_dt_aware.astimezone(pytz.UTC)
            action_dt_naive = action_dt_utc.replace(tzinfo=None)
            act_dates.append(action_dt_naive)
            label = action['description'] or "Response"
            act_labels.append(label)

    # Create a subplot figure: row 1 - price, row 2 - timeline dots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.01,
        row_titles=["Stock Price Chart", "Response Actions Timeline"]
    )

    # Main Price Line
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines', name='Stock Price', line=dict(color='blue', width=2)
    ), row=1, col=1)

    # Crisis/Mitigation periods, averages, min lines
    fig.add_vrect(x0=res['crisis_start_utc'].replace(tzinfo=None), x1=res['crisis_end_utc'].replace(tzinfo=None),
                  fillcolor="red", opacity=0.2, layer="below", line_width=0,
                  annotation_text="Crisis Period", annotation_position="top left", row=1, col=1)
    fig.add_vrect(x0=res['mitigation_start_utc'].replace(tzinfo=None), x1=res['mitigation_end_utc'].replace(tzinfo=None),
                  fillcolor="green", opacity=0.13, layer="below", line_width=0,
                  annotation_text="Mitigation Period", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=res['pre_crisis_avg'], line_dash="dash", line_color="green",
                  annotation_text="Pre-Crisis Average", row=1, col=1)
    fig.add_hline(y=res['crisis_min'], line_dash="dash", line_color="red",
                  annotation_text="Crisis Minimum", row=1, col=1)

    # (Old vertical lines omitted)
    # Add timeline event points to row 2
    if act_dates:
        fig.add_trace(go.Scatter(
            x=act_dates,
            y=np.ones(len(act_dates)),
            mode="markers+text",
            marker=dict(symbol="circle", size=16, color="#045d1f"),
            text=act_labels,
            textposition="top center",
            name="Response Actions",
            hovertext=act_labels,
            showlegend=False
        ), row=2, col=1)

    # Timeline track styling
    fig.update_yaxes(showticklabels=False, fixedrange=True, row=2, col=1, range=[0.8, 1.2], showgrid=False, zeroline=False, title=None)
    fig.update_xaxes(title="Date", row=2, col=1)

    # Price chart styling
    fig.update_yaxes(title="Price ($)", row=1, col=1)

    fig.update_layout(
        height=700,
        title=f"{ticker} Stock Price During Crisis & Mitigation",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- Add/Remove Response Actions Editor (Below Chart) ---------
    st.markdown("---")
    st.markdown("### 🛠️ Add or Edit Crisis Response Actions")
    add_col, _ = st.columns([4, 8])
    with add_col:
        if st.button("+ Add Response Action"):
            st.session_state.response_actions.append({'date': None, 'description': ''})

    rem_indices = []
    for i, action in enumerate(st.session_state.response_actions):
        cols = st.columns([2, 7, 1])
        with cols[0]:
            date_val = st.date_input(
                f"Action Date #{i + 1}",
                value=action['date'] if action['date'] else crisis_start_date,
                key=f"action_date_main_{i}",
                label_visibility="collapsed"
            )
        with cols[1]:
            desc_val = st.text_input(
                f"Description #{i + 1}",
                value=action['description'],
                key=f"action_desc_main_{i}",
                label_visibility="collapsed"
            )
        with cols[2]:
            if st.button("❌", key=f"delete_main_{i}"):
                rem_indices.append(i)
        st.session_state.response_actions[i]['date'] = date_val
        st.session_state.response_actions[i]['description'] = desc_val
    for i in reversed(rem_indices):
        st.session_state.response_actions.pop(i)

    # -- Timeline Table --
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
            fillcolor="red", opacity=0.18, line_width=0)
        vol_fig.add_vrect(
            x0=res['mitigation_start_utc'], x1=res['mitigation_end_utc'],
            fillcolor="green", opacity=0.12, line_width=0)
        st.plotly_chart(vol_fig, use_container_width=True)

    if st.session_state.response_actions:
        st.subheader("🛠️ Response Actions Timeline Table")
        response_action_rows = [
            {"Date": str(a['date']), "Description": a['description']}
            for a in st.session_state.response_actions if a['date'] and a['description']
        ]
        if response_action_rows:
            st.table(pd.DataFrame(response_action_rows))

    st.subheader("🎯 Crisis Impact Summary")
    impact_severity = "High" if abs(res['max_decline']) > 30 else \
                      "Moderate" if abs(res['max_decline']) > 15 else \
                      "Low"
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
    st.subheader("📋 How to Use This Tool")
    st.write("""
    1. **Enter a stock ticker** (e.g., TSLA, AAPL, META) in the sidebar.
    2. **Select the timezone** corresponding to your crisis and mitigation date inputs.
    3. **Select crisis, mitigation start and end dates** in any order.
    4. **Click 'Analyze Crisis Impact'** to load and analyze the stock data.
    5. **Use the section below the main chart to add/remove response actions** at any time; the chart updates instantly.

    The app analyzes and visualizes:
    - Crisis and mitigation periods, including economic impact estimates.
    - Response actions as a dot-labeled event track below the main price chart.
    - Post-crisis recovery metrics: both average and latest closing price.
    - All calculations are timezone-robust and clearly visualized.
    """)
    st.subheader("🔍 Example Crisis Events You Can Analyze")
    st.write("""
    - **Tesla (TSLA)**: Twitter acquisition period (Jan-Jun 2022)
    - **Meta (META)**: Cambridge Analytica scandal (Mar-Jul 2018)
    - **Boeing (BA)**: 737 MAX crashes (Mar-Dec 2019)
    """)

st.markdown("---")
st.markdown("**Crisis Impact Analysis Tool** – Add response actions below the chart anytime. Features a timeline event track for clarity. Built with Streamlit, yfinance, and robust timezone support.")
