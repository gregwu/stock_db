import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from scipy.spatial.distance import cosine

# Load environment variables from .env file
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'database'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def get_engine():
    url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

def fetch_tickers_with_latest_date(engine, current_date):
    query = f"""
        SELECT distinct ticker FROM stock_data
        -- WHERE date >  '{current_date}' - interval '5 days'
        -- GROUP BY ticker
    """
    return pd.read_sql(query, engine)['ticker'].tolist()

def fetch_ticker_data(engine, ticker, start_date, end_date):
    query = f"""
        SELECT date, open_price as open, high_price as high, low_price as low, close_price as close, volume
        FROM stock_data
        WHERE ticker = '{ticker}' AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date
    """
    df = pd.read_sql(query, engine, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def normalize_series(series):
    arr = np.array(series)
    if len(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

def main():
    st.title("Cosine Similarity Pattern Scan (DB)")
    engine = get_engine()
    today = datetime.now().date()

    # 1. Select reference ticker and date range (SIDEBAR)
    with st.sidebar:
        st.header("Reference Pattern Selection")
        all_tickers_db = pd.read_sql("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker", engine)['ticker'].tolist()
        # Remove .US for display
        all_tickers = [t[:-3] if t.endswith('.US') else t for t in all_tickers_db]
        default_ticker = 'DEVS'
        default_start = datetime(2025, 2, 1).date()
        default_end = datetime(2025, 5, 7).date()
        ref_ticker = st.selectbox("Select reference ticker", all_tickers, index=all_tickers.index(default_ticker) if default_ticker in all_tickers else 0)
        ref_start = st.date_input("Reference start date", default_start)
        ref_end = st.date_input("Reference end date", default_end)
        scan_button = st.button("Start Scan")


    # File to store scan results, include meta info
    import pickle
    def get_result_file(ticker, start, end):
        return f"similarity_{ticker}_{start}_{end}.pkl"

    result_file = get_result_file(ref_ticker, ref_start, ref_end)

    # Load from file if exists and parameters match
    if not scan_button:
        try:
            with open(result_file, "rb") as f:
                data = pickle.load(f)
                st.session_state['results_df'] = data['results_df']
                st.session_state['tickers_db'] = data['tickers_db']
        except Exception:
            st.session_state['results_df'] = None
            st.session_state['tickers_db'] = None
    if ref_start > ref_end:
        st.error("Start date must be before end date.")
        return
    # Add .US back for DB query
    ref_ticker_db = ref_ticker + '.US' if (ref_ticker + '.US') in all_tickers_db else ref_ticker
    ref_df = fetch_ticker_data(engine, ref_ticker_db, ref_start, ref_end)
    st.write(f"Reference pattern: {ref_ticker} ({ref_start} to {ref_end})")
    # Reference chart: candlestick + BB + volume (no duplicate subplot)
    close = ref_df['close']
    sma = close.rolling(window=20, min_periods=1).mean()
    std = close.rolling(window=20, min_periods=1).std()
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    ref_fig = make_subplots(specs=[[{"secondary_y": True}]])
    ref_fig.add_trace(go.Candlestick(
        x=ref_df.index,
        open=ref_df['open'], high=ref_df['high'], low=ref_df['low'], close=ref_df['close'],
        name='Candlestick',
        increasing_line_color='green', decreasing_line_color='red',
        showlegend=False
    ), secondary_y=False)
    ref_fig.add_trace(go.Scatter(x=ref_df.index, y=upper_bb, mode='lines', name='BB Upper', line=dict(color='rgba(200,0,0,0.4)', dash='dot')),
                      secondary_y=False)
    ref_fig.add_trace(go.Scatter(x=ref_df.index, y=lower_bb, mode='lines', name='BB Lower', line=dict(color='rgba(0,0,200,0.4)', dash='dot')),
                      secondary_y=False)
    ref_fig.add_trace(go.Scatter(x=ref_df.index, y=sma, mode='lines', name='SMA20', line=dict(color='gray', dash='dash')),
                      secondary_y=False)
    if 'volume' in ref_df.columns:
        ref_fig.add_trace(go.Bar(x=ref_df.index, y=ref_df['volume'], name='Volume', marker_color='rgba(100,100,100,0.3)'), secondary_y=True)
    ref_fig.update_layout(
        yaxis=dict(title='Price', rangemode='tozero'),
        yaxis2=dict(title='Volume', rangemode='tozero', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        bargap=0,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(ref_fig, use_container_width=True)

    # 2. Scan all tickers with latest date = today
    st.header("Cosine Similarity Scan")
    if scan_button:
        st.info("Scanning all tickers with up-to-date data...")
        tickers_db = fetch_tickers_with_latest_date(engine, today)
        # Remove .US for display, but keep mapping
        tickers = [t[:-3] if t.endswith('.US') else t for t in tickers_db]
        st.write(f"Found {len(tickers)} tickers to scan.")
        ref_pattern = normalize_series(ref_df['close'])
        results = []
        progress_bar = st.progress(0)
        for idx, ticker_db in enumerate(tickers_db):
            ticker_display = ticker_db[:-3] if ticker_db.endswith('.US') else ticker_db
            if ticker_db == ref_ticker_db:
                progress_bar.progress((idx + 1) / len(tickers_db))
                continue
            scan_start = today - timedelta(days=365)
            scan_end = today
            df = fetch_ticker_data(engine, ticker_db, scan_start, scan_end)
            closes = df['close'].values
            if len(closes) < len(ref_pattern):
                progress_bar.progress((idx + 1) / len(tickers_db))
                continue
            best_sim = -1
            best_start = None
            best_end = None
            for i in range(len(closes) - len(ref_pattern) + 1):
                window = closes[i:i+len(ref_pattern)]
                window_norm = normalize_series(window)
                sim = 1 - cosine(ref_pattern, window_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_start = df.index[i]
                    best_end = df.index[i + len(ref_pattern) - 1]
            latest_close = df['close'].iloc[-1] if not df.empty else None
            results.append({
                'ticker': ticker_display,
                'similarity': best_sim,
                'match_start': best_start,
                'match_end': best_end if best_start is not None else None,
                'latest_close': latest_close
            })
            progress_bar.progress((idx + 1) / len(tickers_db))
        progress_bar.empty()
        results_df = pd.DataFrame(results)
        # Filter: match_end within 6 days of today
        if not results_df.empty:
            results_df = results_df[results_df['match_end'].notnull()]
            results_df = results_df[results_df['match_end'].apply(lambda d: abs((d.date() - today).days) <= 6)]
            results_df = results_df.sort_values('similarity', ascending=False)
        st.session_state['results_df'] = results_df
        st.session_state['tickers_db'] = tickers_db
        # Save to file with meta info
        try:
            with open(result_file, "wb") as f:
                pickle.dump({'results_df': results_df, 'tickers_db': tickers_db}, f)
        except Exception as e:
            st.warning(f"Could not save results to file: {e}")

    # Always show results table if available
    results_df = st.session_state.get('results_df', None)
    tickers_db = st.session_state.get('tickers_db', None)
    if results_df is not None and tickers_db is not None and not results_df.empty:
        st.write("**Matches sorted by similarity (click on a row to view chart):**")
        event = st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="result_table"
        )
        selected_row = None
        if hasattr(event, 'selection') and hasattr(event.selection, 'rows') and event.selection.rows:
            idx = event.selection.rows[0]
            if idx < len(results_df):
                selected_row = results_df.iloc[idx]
        if selected_row is not None:
            ticker_display = selected_row['ticker']
            ticker_db = ticker_display + '.US' if (ticker_display + '.US') in tickers_db else ticker_display
            match_start = selected_row['match_start']
            match_end = selected_row['match_end']
            if pd.notnull(match_start) and pd.notnull(match_end):
                df = fetch_ticker_data(engine, ticker_db, match_start, match_end)
                st.write(f"**{ticker_display}**: {match_start.date()} to {match_end.date()} (match window)")
                # Compute Bollinger Bands
                close = df['close']
                sma = close.rolling(window=20, min_periods=1).mean()
                std = close.rolling(window=20, min_periods=1).std()
                upper_bb = sma + 2 * std
                lower_bb = sma - 2 * std
                import plotly.graph_objs as go
                from plotly.subplots import make_subplots
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name='Candlestick',
                    increasing_line_color='green', decreasing_line_color='red',
                    showlegend=False
                ), secondary_y=False)
                fig.add_trace(go.Scatter(x=df.index, y=upper_bb, mode='lines', name='BB Upper', line=dict(color='rgba(200,0,0,0.4)', dash='dot')),
                              secondary_y=False)
                fig.add_trace(go.Scatter(x=df.index, y=lower_bb, mode='lines', name='BB Lower', line=dict(color='rgba(0,0,200,0.4)', dash='dot')),
                              secondary_y=False)
                fig.add_trace(go.Scatter(x=df.index, y=sma, mode='lines', name='SMA20', line=dict(color='gray', dash='dash')),
                              secondary_y=False)
                if 'volume' in df.columns:
                    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(100,100,100,0.3)'), secondary_y=True)
                fig.update_layout(
                    yaxis=dict(title='Price', rangemode='tozero'),
                    yaxis2=dict(title='Volume', rangemode='tozero', showgrid=False),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    bargap=0,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()