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

# Stock ticker input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="TSLA").upper()

# User timezone selection (optional)
# You can extend this dropdown to support more timezones if you want
TIMEZONE_OPTIONS = [
    "America/New_York",
    "UTC",
    "Europe/London",
    "Asia/Tokyo",
    "Australia/Sydney"
]
user_tz_str = st.sidebar.selectbox("Select Timezone for Input Dates", TIMEZONE_OPTIONS, index=0)
user_timezone = pytz.timezone(user_tz_str)

# Crisis date inputs
crisis_start_date = st.sidebar.date_input(
    "Crisis Start Date", 
    value=datetime(2022, 1, 1)
)
crisis_end_date = st.sidebar.date_input(
    "Crisis End Date", 
    value=datetime(2022, 6, 30)
)

# Convert naive dates to timezone-aware datetimes at midnight in user timezone
crisis_start = user_timezone.localize(datetime.combine(crisis_start_date, datetime.min.time()))
crisis_end = user_timezone.localize(datetime.combine(crisis_end_date, datetime.min.time()))

# Analysis button
if st.sidebar.button("Analyze Crisis Impact"):
    try:
        with st.spinner("Fetching stock data..."):
            # Extend date range for analysis window in user timezone
            start_date = crisis_start - timedelta(days=90)
            end_date = crisis_end + timedelta(days=90)

            # For yfinance, convert to strings in ISO8601 format without tz info (UTC naive is fine)
            # yfinance will handle timezone internally for the market data timestamps
            data = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"), 
                                             end=end_date.strftime("%Y-%m-%d"))
            
            if data.empty:
                st.error("No data found for this ticker. Please check the symbol.")
                st.stop()

            # yfinance data index tz info - convert to UTC for consistent comparisons
            if data.index.tz is None:
                # Make data index timezone-aware as UTC if none
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')

            # Convert crisis input dates to UTC to match data index for comparison
            crisis_start_utc = crisis_start.astimezone(pytz.UTC)
            crisis_end_utc = crisis_end.astimezone(pytz.UTC)

            # Slice data by crisis periods using timezone-aware comparisons
            pre_crisis_data = data[data.index < crisis_start_utc]
            crisis_data = data[(data.index >= crisis_start_utc) & (data.index <= crisis_end_utc)]
            post_crisis_data = data[data.index > crisis_end_utc]

            if pre_crisis_data.empty or crisis_data.empty:
                st.error("Insufficient data for the selected date range.")
                st.stop()

            # Key metrics calculations
            pre_crisis_avg = pre_crisis_data['Close'].mean()
            crisis_min = crisis_data['Close'].min()
            crisis_max = crisis_data['Close'].max()
            crisis_avg = crisis_data['Close'].mean()

            # Impact percentage calculations
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
                st.metric(
                    label="Pre-Crisis Average",
                    value=f"${pre_crisis_avg:.2f}"
                )
            with col2:
                st.metric(
                    label="Crisis Minimum",
                    value=f"${crisis_min:.2f}",
                    delta=f"{max_decline:.1f}%"
                )
            with col3:
                st.metric(
                    label="Crisis Average",
                    value=f"${crisis_avg:.2f}",
                    delta=f"{avg_decline:.1f}%"
                )
            with col4:
                st.metric(
                    label="Post-Crisis Recovery",
                    value=recovery_info
                )

            # Economic damage estimation
            try:
                stock = yf.Ticker(ticker)  # Reinstantiate to access info
                company_info = stock.info
                shares_outstanding = company_info.get('sharesOutstanding', 1000000000)
                market_cap_loss = abs(max_decline) / 100 * pre_crisis_avg * shares_outstanding

                st.subheader("💰 Economic Impact Analysis")
                st.write(f"**Estimated Market Cap Loss:** ${market_cap_loss:,.0f}")
                st.write(f"**Maximum Stock Price Decline:** {abs(max_decline):.1f}%")
                st.write(f"**Crisis Duration:** {(crisis_end - crisis_start).days} days")
            except Exception:
                st.write("**Market cap data unavailable for detailed economic impact calculation**")

            # Plot stock price with crisis period shading
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))
            fig.add_vrect(
                x0=crisis_start_utc, x1=crisis_end_utc,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Crisis Period",
                annotation_position="top left"
            )
            fig.add_hline(y=pre_crisis_avg, line_dash="dash", line_color="green", 
                          annotation_text="Pre-Crisis Average")
            fig.add_hline(y=crisis_min, line_dash="dash", line_color="red", 
                          annotation_text="Crisis Minimum")

            fig.update_layout(
                title=f"{ticker} Stock Price During Crisis Period",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Crisis timeline table
            st.subheader("📈 Crisis Timeline Analysis")
            timeline_data = pd.DataFrame({
                'Period': ['Pre-Crisis (90 days)', 'Crisis Period', 'Post-Crisis (90 days)'],
                'Average Price': [
                    pre_crisis_avg,
                    crisis_avg,
                    post_crisis_data['Close'].mean() if not post_crisis_data.empty else np.nan
                ],
                'Volatility': [
                    pre_crisis_data['Close'].std(),
                    crisis_data['Close'].std(),
                    post_crisis_data['Close'].std() if not post_crisis_data.empty else np.nan
                ]
            })
            timeline_data = timeline_data.dropna()
            st.dataframe(timeline_data.round(2), use_container_width=True)

            # Volume chart, if available
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
                    x0=crisis_start_utc, x1=crisis_end_utc,
                    fillcolor="red", opacity=0.2,
                    layer="below", line_width=0
                )
                vol_fig.update_layout(
                    title=f"{ticker} Trading Volume During Crisis",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=400
                )
                st.plotly_chart(vol_fig, use_container_width=True)

            # Summary
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
            - {recovery_info}

            **Interpretation:**
            {'This represents a significant reputational crisis with substantial market impact.' if abs(max_decline) > 30 else
             'This shows a moderate crisis impact with noticeable market effects.' if abs(max_decline) > 15 else
             'This indicates a relatively minor crisis impact on stock performance.'}
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your internet connection and verify the stock ticker symbol.")

else:
    st.subheader("📋 How to Use This Tool")
    st.write("""
    1. **Enter a stock ticker** (e.g., TSLA, AAPL, META) in the sidebar.
    2. **Select the timezone** corresponding to your crisis date inputs.
    3. **Select crisis start and end dates** using the date pickers.
    4. **Click 'Analyze Crisis Impact'** to generate the analysis.

    This tool will provide:
    - Stock price impact analysis during the crisis period.
    - Estimated economic damage based on market cap.
    - Interactive price and volume charts highlighting the crisis period.
    - Recovery assessment and crisis severity evaluation.
    """)

    st.subheader("🔍 Example Crisis Events You Can Analyze")
    st.write("""
    - **Tesla (TSLA)**: Twitter acquisition period (Jan-Jun 2022)
    - **Meta (META)**: Cambridge Analytica scandal (Mar-Jul 2018)
    - **Wells Fargo (WFC)**: Account fraud scandal (Sep-Dec 2016)
    - **Boeing (BA)**: 737 MAX crashes (Mar-Dec 2019)
    - **Volkswagen (VWAGY)**: Emissions scandal (Sep-Dec 2015)
    """)

st.markdown("---")
st.markdown("**Crisis Impact Analysis Tool** - Built with Streamlit and yfinance")
