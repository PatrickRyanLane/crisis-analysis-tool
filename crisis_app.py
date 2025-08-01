import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crisis Impact Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("🚨 Reputational Crisis Impact Analysis Tool")
st.markdown("**Analyze the economic impact of reputational crises on stock prices**")

# Sidebar for inputs
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

# --- Crisis and Mitigation Dates ---
crisis_start_date = st.sidebar.date_input("Crisis Start Date", value=datetime(2022, 1, 1))
crisis_end_date = st.sidebar.date_input("Crisis End Date", value=datetime(2022, 6, 30))

# Mitigation period inputs - no constraints!
mitigation_start_date = st.sidebar.date_input(
    "Mitigation Start Date", value=crisis_end_date
)
mitigation_end_date = st.sidebar.date_input(
    "Mitigation End Date", value=crisis_end_date + timedelta(days=90)
)

if mitigation_end_date < mitigation_start_date:
    st.sidebar.error("Mitigation End Date cannot be before Mitigation Start Date.")

# --- Response Actions (Dynamic Input) ---
st.sidebar.markdown("### Response Actions Taken")
if 'response_actions' not in st.session_state:
    st.session_state.response_actions = []

def add_response_action():
    st.session_state.response_actions.append({'date': None, 'description': ''})

if st.sidebar.button("Add Response Action"):
    add_response_action()

# Render inputs for response actions
to_remove = []
for i, action in enumerate(st.session_state.response_actions):
    col1, col2, col3 = st.sidebar.columns([3, 6, 1])
    with col1:
        date_val = st.date_input(
            f"Action Date #{i+1}",
            value=action['date'] if action['date'] else crisis_start_date,
            key=f"action_date_{i}"
        )
    with col2:
        desc_val = st.text_input(
            "Description",
            value=action['description'],
            key=f"action_desc_{i}"
        )
    with col3:
        if st.button("❌", key=f"delete_{i}"):
            to_remove.append(i)
    st.session_state.response_actions[i]['date'] = date_val
    st.session_state.response_actions[i]['description'] = desc_val
for i in reversed(to_remove):
    st.session_state.response_actions.pop(i)
    st.experimental_rerun()

# --- Button to Run Analysis ---
if st.sidebar.button("Analyze Crisis Impact"):
    try:
        with st.spinner("Fetching stock data..."):
            # Localize all relevant dates
            # crisis and mitigation as aware datetimes in user tz
            crisis_start = user_timezone.localize(datetime.combine(crisis_start_date, datetime.min.time()))
            crisis_end = user_timezone.localize(datetime.combine(crisis_end_date, datetime.min.time()))
            mitigation_start = user_timezone.localize(datetime.combine(mitigation_start_date, datetime.min.time()))
            mitigation_end = user_timezone.localize(datetime.combine(mitigation_end_date, datetime.min.time()))

            # Pull slightly wider range for complete context
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

            # ALL analysis points converted to UTC
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
            crisis_max = crisis_data['Close'].max()
            crisis_avg = crisis_data['Close'].mean()
            mitigation_avg = mitigation_data['Close'].mean() if not mitigation_data.empty else np.nan
            mitigation_vol = mitigation_data['Close'].std() if not mitigation_data.empty else np.nan

            max_decline = ((crisis_min - pre_crisis_avg) / pre_crisis_avg) * 100
            avg_decline = ((crisis_avg - pre_crisis_avg) / pre_crisis_avg) * 100

            # Recovery analysis
            recovery_info = "Not enough post-crisis data"
            if not post_crisis_data.empty:
                post_crisis_avg = post_crisis_data['Close'].mean()
                recovery_percentage = ((post_crisis_avg - crisis_min) / crisis_min) * 100
                recovery_info = f"{recovery_percentage:.1f}% recovery from crisis minimum"

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pre-Crisis Average", f"${pre_crisis_avg:.2f}")
            with col2:
                st.metric("Crisis Minimum", f"${crisis_min:.2f}", delta=f"{max_decline:.1f}%")
            with col3:
                st.metric("Crisis Average", f"${crisis_avg:.2f}", delta=f"{avg_decline:.1f}%")
            with col4:
                st.metric("Post-Crisis Recovery", recovery_info)

            # --- Economic Damage Estimate ---
            try:
                stock = yf.Ticker(ticker)
                company_info = stock.info
                shares_outstanding = company_info.get('sharesOutstanding', 1000000000)
                market_cap_loss = abs(max_decline) / 100 * pre_crisis_avg * shares_outstanding

                st.subheader("💰 Economic Impact Analysis")
                st.write(f"**Estimated Market Cap Loss:** ${market_cap_loss:,.0f}")
                st.write(f"**Maximum Stock Price Decline:** {abs(max_decline):.1f}%")
                st.write(f"**Crisis Duration:** {(crisis_end - crisis_start).days} days")
                st.write(f"**Mitigation Period:** {mitigation_start_date} to {mitigation_end_date} ({(mitigation_end - mitigation_start).days} days)")
            except Exception:
                st.write("**Market cap data unavailable for detailed economic impact calculation**")

            # --- MAIN CHART ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))
            # Crisis period red shading
            fig.add_vrect(
                x0=crisis_start_utc, x1=crisis_end_utc,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Crisis Period", annotation_position="top left"
            )
            # Mitigation period green shading
            fig.add_vrect(
                x0=mitigation_start_utc, x1=mitigation_end_utc,
                fillcolor="green", opacity=0.13,
                layer="below", line_width=0,
                annotation_text="Mitigation Period", annotation_position="top right"
            )
            # Reference lines
            fig.add_hline(y=pre_crisis_avg, line_dash="dash", line_color="green",
                          annotation_text="Pre-Crisis Average")
            fig.add_hline(y=crisis_min, line_dash="dash", line_color="red",
                          annotation_text="Crisis Minimum")

            # Response Action markers/annotations
            for action in st.session_state.response_actions:
                if action['date'] and action['description']:
                    action_dt = user_timezone.localize(datetime.combine(action['date'], datetime.min.time()))
                    action_dt_utc = action_dt.astimezone(pytz.UTC)
                    if data.index.min() <= action_dt_utc <= data.index.max():
                        # Find closest closing price to mark the action
                        closest_idx = data.index.get_indexer([action_dt_utc], method='nearest')[0]
                        action_price = data.iloc[closest_idx]['Close']
                        fig.add_trace(go.Scatter(
                            x=[data.index[closest_idx]],
                            y=[action_price],
                            mode='markers+text',
                            marker=dict(symbol='star', size=13, color='gold'),
                            text=[action['description']],
                            textposition='top center',
                            name=f"Action: {action['description']}"
                        ))

            fig.update_layout(
                title=f"{ticker} Stock Price During Crisis & Mitigation",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Timeline Table
            st.subheader("📈 Timeline Analysis")
            timeline_data = pd.DataFrame({
                'Period': [
                    'Pre-Crisis (90 days)', 
                    'Crisis Period', 
                    'Mitigation Period',
                    'Post-Crisis (90 days)'
                ],
                'Average Price': [
                    pre_crisis_avg,
                    crisis_avg,
                    mitigation_avg,
                    post_crisis_data['Close'].mean() if not post_crisis_data.empty else np.nan
                ],
                'Volatility': [
                    pre_crisis_data['Close'].std(),
                    crisis_data['Close'].std(),
                    mitigation_vol,
                    post_crisis_data['Close'].std() if not post_crisis_data.empty else np.nan
                ]
            }).dropna()
            st.dataframe(timeline_data.round(2), use_container_width=True)

            # Trading Volume Chart (optional)
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
                # Crisis & Mitigation shading
                vol_fig.add_vrect(
                    x0=crisis_start_utc, x1=crisis_end_utc,
                    fillcolor="red", opacity=0.18, line_width=0)
                vol_fig.add_vrect(
                    x0=mitigation_start_utc, x1=mitigation_end_utc,
                    fillcolor="green", opacity=0.12, line_width=0)
                vol_fig.update_layout(
                    title=f"{ticker} Trading Volume During Crisis and Mitigation",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=350
                )
                st.plotly_chart(vol_fig, use_container_width=True)

            # --- Response Actions Table ---
            if st.session_state.response_actions:
                st.subheader("🛠️ Response Actions Timeline")
                # Table format
                response_action_rows = [
                    {
                        "Date": str(a['date']),
                        "Description": a['description']
                    }
                    for a in st.session_state.response_actions if a['date'] and a['description']
                ]
                if response_action_rows:
                    st.table(pd.DataFrame(response_action_rows))

            # --- Summary ---
            st.subheader("🎯 Crisis Impact Summary")
            impact_severity = "High" if abs(max_decline) > 30 else \
                              "Moderate" if abs(max_decline) > 15 else \
                              "Low"
            st.write(f"""
            **Crisis Severity:** {impact_severity} Impact

            **Key Findings:**
            - Maximum price decline of {abs(max_decline):.1f}% during the crisis period
            - Stock fell from ${pre_crisis_avg:.2f} average to ${crisis_min:.2f} minimum
            - Crisis lasted {(crisis_end - crisis_start).days} days
            - Mitigation actions and timeframe shown above
            - {recovery_info}

            **Interpretation:**
            {'This represents a significant reputational crisis with substantial market impact.' if abs(max_decline) > 30 else
             'This shows a moderate crisis impact with noticeable market effects.' if abs(max_decline) > 15 else
             'This indicates a relatively minor crisis impact on stock performance.'}
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your internet connection and verify the stock ticker symbol.")

# --- Instructions If Not Running Analysis ---
else:
    st.subheader("📋 How to Use This Tool")
    st.write("""
    1. **Enter a stock ticker** (e.g., TSLA, AAPL, META) in the sidebar.
    2. **Select the timezone** corresponding to your crisis and mitigation date inputs.
    3. **Select crisis, mitigation start and end dates**, in any order.
    4. **Optionally add multiple "Response Actions"**—each with a date and description.
    5. **Click 'Analyze Crisis Impact'** to generate the full analysis.

    The app analyzes and visualizes:
    - Crisis and mitigation periods, including economic impact estimates.
    - Each distinct response action as a special marker on the timeline.
    - All calculations are timezone-robust and clearly visualized.
    """)

    st.subheader("🔍 Example Crisis Events You Can Analyze")
    st.write("""
    - **Tesla (TSLA)**: Twitter acquisition period (Jan-Jun 2022)
    - **Meta (META)**: Cambridge Analytica scandal (Mar-Jul 2018)
    - **Boeing (BA)**: 737 MAX crashes (Mar-Dec 2019)
    """)

st.markdown("---")
st.markdown("**Crisis Impact Analysis Tool** - Built with Streamlit, yfinance, and robust timezone support.")
