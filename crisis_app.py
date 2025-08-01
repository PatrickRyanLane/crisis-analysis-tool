import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crisis Impact Tool",
    page_icon="📊",
    layout="wide"
)

st.title("🚨 Crisis Impact Analysis Tool")
st.markdown("**Analyze the economic impact of reputational crises on stock prices**")

# Sidebar Input
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()

TIMEZONE_OPTIONS = [
    "America/New_York",
    "UTC",
    "Europe/London",
    "Asia/Tokyo",
    "Australia/Sydney"
]

timezone_str = st.sidebar.selectbox("Select Timezone", TIMEZONE_OPTIONS, index=0)
user_tz = pytz.timezone(timezone_str)

crisis_start_date = st.sidebar.date_input("Crisis start date", datetime(2022, 1, 1))
crisis_end_date = st.sidebar.date_input("Crisis end date", datetime(2022, 6, 30))
mitigation_start_date = st.sidebar.date_input("Mitigation start date", crisis_end_date)
mitigation_end_date = st.sidebar.date_input("Mitigation end date", crisis_end_date + timedelta(days=90))

if mitigation_end_date < mitigation_start_date:
    st.sidebar.error("Mitigation end date cannot be before mitigation start date.")

analyze = st.sidebar.button("Analyze")

if analyze or "analysis_result" not in st.session_state:
    # Load and compute all data:
    try:
        c_start = user_tz.localize(datetime.combine(crisis_start_date, datetime.min.time()))
        c_end = user_tz.localize(datetime.combine(crisis_end_date, datetime.min.time()))
        m_start = user_tz.localize(datetime.combine(mitigation_start_date, datetime.min.time()))
        m_end = user_tz.localize(datetime.combine(mitigation_end_date, datetime.min.time()))

        date_start = min(c_start, m_start) - timedelta(days=90)
        date_end = max(c_end, m_end) + timedelta(days=90)

        data = yf.Ticker(ticker).history(start=date_start.strftime("%Y-%m-%d"), end=date_end.strftime("%Y-%m-%d"))
        if data.empty:
            st.error("No data found. Check stock ticker symbol.")
            st.stop()

        # Ensure datetime index timezone aware and UTC:
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")

        c_start_utc = c_start.astimezone(pytz.UTC)
        c_end_utc = c_end.astimezone(pytz.UTC)
        m_start_utc = m_start.astimezone(pytz.UTC)
        m_end_utc = m_end.astimezone(pytz.UTC)

        pre_c = data[data.index < c_start_utc]
        cris = data[(data.index >= c_start_utc) & (data.index <= c_end_utc)]
        post_c = data[data.index > c_end_utc]
        mitig = data[(data.index >= m_start_utc) & (data.index <= m_end_utc)]

        if pre_c.empty or cris.empty:
            st.error("Insufficient historical data.")
            st.stop()

        pre_avg = pre_c["Close"].mean()
        cris_min = cris["Close"].min()
        cris_avg = cris["Close"].mean()
        mitig_avg = mitig["Close"].mean() if not mitig.empty else np.nan
        mitig_vol = mitig["Close"].std() if not mitig.empty else np.nan
        max_drop_pct = ((cris_min - pre_avg) / pre_avg) * 100
        avg_drop_pct = ((cris_avg - pre_avg) / pre_avg) * 100

        if not post_c.empty:
            post_avg = post_c["Close"].mean()
            recov_pct = ((post_avg - cris_min) / cris_min) * 100
            latest_close = post_c["Close"].iloc[-1]
            latest_recov_pct = ((latest_close - cris_min) / cris_min) * 100
        else:
            post_avg = latest_close = np.nan
            recov_pct = latest_recov_pct = None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            shares_out = info.get("sharesOutstanding", 1_000_000_000)
            est_loss = abs(max_drop_pct) * pre_avg * shares_out / 100
        except Exception:
            shares_out = 1_000_000_000
            est_loss = abs(max_drop_pct) * pre_avg * shares_out / 100

        st.session_state["analysis_result"] = dict(
            data=data,
            c_start_utc=c_start_utc,
            c_end_utc=c_end_utc,
            m_start_utc=m_start_utc,
            m_end_utc=m_end_utc,
            pre_avg=pre_avg,
            cris_min=cris_min,
            cris_avg=cris_avg,
            mitig_avg=mitig_avg,
            mitig_vol=mitig_vol,
            max_drop_pct=max_drop_pct,
            avg_drop_pct=avg_drop_pct,
            post_c=post_c,
            post_avg=post_avg,
            latest_close=latest_close,
            recov_pct=recov_pct,
            latest_recov_pct=latest_recov_pct,
            shares_out=shares_out,
            est_loss=est_loss,
        )

    except Exception as e:
        st.error(f"Error during data fetch or computation: {e}")
        st.stop()

if "analysis_result" not in st.session_state:
    st.info("Enter details and click Analyze to load data.")
else:
    res = st.session_state["analysis_result"]
    data = res["data"]

    # Top-level metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-crisis Avg Price", f"${res['pre_avg']:.2f}")
    c2.metric("Crisis Min Price", f"${res['cris_min']:.2f}", delta=f"{res['max_drop_pct']:.1f}%")
    c3.metric("Crisis Avg Price", f"${res['cris_avg']:.2f}", delta=f"{res['avg_drop_pct']:.1f}%")

    with st.expander("Post-crisis Recovery Details", expanded=True):
        if not res["post_c"].empty:
            st.metric("Average Recovery", f"{res['recov_pct']:.1f}%")
            st.caption(f"Avg Price: ${res['post_avg']:.2f}")
            st.metric("Latest Close", f"{res['latest_recov_pct']:.1f}%")
            st.caption(f"Price: ${res['latest_close']:.2f}")
            st.caption(f"Gain since min: ${res['latest_close'] - res['cris_min']:.2f}")
        else:
            st.info("No post-crisis data available.")

    st.subheader("Estimated Economic Impact")
    st.write(f"Estimated Market Cap Loss: **${res['est_loss']:,.0f}**")
    st.write(f"Shares Outstanding: {res['shares_out']:,}")
    st.write(f"Crisis Duration: {(res['c_end_utc'] - res['c_start_utc']).days} days")
    st.write(f"Mitigation Duration: {(res['m_end_utc'] - res['m_start_utc']).days} days")

    # Prepare response events for timeline below chart
    event_dates = []
    event_labels = []
    for event in st.session_state.get("response_actions", []):
        if event.get("date"):
            dt = datetime.combine(event["date"], datetime.min.time())
            dt_aware = user_tz.localize(dt) if dt.tzinfo is None else dt
            dt_utc = dt_aware.astimezone(pytz.UTC).replace(tzinfo=None)
            event_dates.append(dt_utc)
            label = event.get("description") or "Event"
            event_labels.append(label)

    # Calculate vertical offsets to avoid overlapping labels:
    offsets = []
    counts = defaultdict(int)
    for dt in event_dates:
        day_ordinal = dt.toordinal()
        offset = counts[day_ordinal] * 0.3  # 0.3 vertical units per stacked label
        offsets.append(1 + offset)
        counts[day_ordinal] += 1

    # Create subplot: main price + timeline
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.02,
        row_titles=("Price Chart", "Response Timeline")
    )

    # Price time series
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"],
            name="Close Price",
            mode="lines",
            line=dict(color="royalblue", width=2),
        ),
        row=1, col=1,
    )

    # Highlight crisis and mitigation periods
    fig.add_vrect(
        x0=res["c_start_utc"].replace(tzinfo=None),
        x1=res["c_end_utc"].replace(tzinfo=None),
        fillcolor="red", opacity=0.25, layer="below",
        line_width=0, annotation_text="Crisis", annotation_position="top left",
        row=1, col=1,
    )
    fig.add_vrect(
        x0=res["m_start_utc"].replace(tzinfo=None),
        x1=res["m_end_utc"].replace(tzinfo=None),
        fillcolor="green", opacity=0.15, layer="below",
        line_width=0, annotation_text="Mitigation", annotation_position="top right",
        row=1, col=1,
    )

    # Reference lines
    fig.add_hline(
        y=res["pre_avg"], line_dash="dash", line_color="green",
        row=1, col=1, annotation_text="Pre-crisis Avg"
    )
    fig.add_hline(
        y=res["cris_min"], line_dash="dash", line_color="red",
        row=1, col=1, annotation_text="Crisis Min"
    )

    # Add event markers in timeline row
    if event_dates:
        fig.add_trace(
            go.Scatter(
                x=event_dates,
                y=offsets,
                mode="markers+text",
                marker=dict(size=15, symbol="circle", color="darkgreen"),
                text=event_labels,
                textposition="top center",
                hoverinfo="text",
                showlegend=False,
            ),
            row=2, col=1,
        )
        # Clean y-axis in timeline row
        fig.update_yaxes(visible=False, row=2, col=1)
        fig.update_yaxes(range=[0.8, max(offsets) + 0.3], row=2, col=1)

    # Configure axes and layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=50, t=60, b=50),
        title=f"{ticker} Stock with Crisis & Mitigation — Event Timeline Below",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add new response action form to bottom
    st.markdown("---")
    st.markdown("### Add a New Response Action")

    if "pending_action" not in st.session_state:
        st.session_state.pending_action = {"date": mitigation_start_date, "description": ""}

    with st.form("add_response_action"):
        new_date = st.date_input("Date", value=st.session_state.pending_action["date"])
        new_desc = st.text_input("Description", value=st.session_state.pending_action["description"])
        submitted = st.form_submit_button("Add Event")

        if submitted:
            if new_date and new_desc.strip():
                if "response_actions" not in st.session_state:
                    st.session_state["response_actions"] = []
                st.session_state["response_actions"].append({"date": new_date, "description": new_desc.strip()})
                st.session_state.pending_action = {"date": mitigation_start_date, "description": ""}
            else:
                st.warning("Please provide both date and description.")

    # Editable list of existing events with remove option
    st.markdown("### Edit or Remove Existing Events")
    remove_indices = []
    for idx, ev in enumerate(st.session_state.get("response_actions", [])):
        cols = st.columns([2, 7, 1])
        with cols[0]:
            dt = st.date_input(f"Date #{idx + 1}", ev["date"], key=f"date_edit_{idx}")
        with cols[1]:
            desc = st.text_input(f"Description #{idx + 1}", ev["description"], key=f"desc_edit_{idx}")
        with cols[2]:
            if st.button("Remove", key=f"remove_{idx}"):
                remove_indices.append(idx)
        st.session_state["response_actions"][idx]["date"] = dt
        st.session_state["response_actions"][idx]["description"] = desc
    for rm_idx in reversed(remove_indices):
        st.session_state["response_actions"].pop(rm_idx)

    # Show event list table for reference
    if st.session_state.get("response_actions"):
        st.markdown("### All Response Events")
        df_events = pd.DataFrame(st.session_state["response_actions"])
        df_events = df_events.dropna(subset=["date"]).reset_index(drop=True)
        st.dataframe(df_events)

# Show usage tips
else:
    st.info("Fill out parameters and click Analyze to start your analysis.")

st.markdown("---")
st.markdown("### About")
st.markdown("""
This tool estimates the market impact of reputational crises on stock prices.
Add “Response Actions” to visualize steps taken during mitigation and monitor economic impact.
""")
