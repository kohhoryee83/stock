import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title='Stock Buy & Sell Put Detector', page_icon='📈', layout='wide')

# --------------------------- Helpers ---------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def technical_snapshot(hist: pd.DataFrame):
    close = hist['Close'].dropna()
    if len(close) < 60:
        raise ValueError('Not enough price history')

    latest = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    rsi14 = rsi(close, 14).iloc[-1]
    macd_line, macd_signal, macd_hist = macd(close)

    # Simple support from 60-day low and recent swing percentile
    window60 = close.tail(60)
    support = float(np.percentile(window60, 10))
    resistance = float(np.percentile(window60, 90))

    return {
        'price': float(latest),
        'ma20': float(ma20),
        'ma50': float(ma50),
        'ma200': float(ma200) if not np.isnan(ma200) else np.nan,
        'rsi': float(rsi14),
        'macd': float(macd_line.iloc[-1]),
        'macd_signal': float(macd_signal.iloc[-1]),
        'macd_hist': float(macd_hist.iloc[-1]),
        'support': support,
        'resistance': resistance,
        '52w_low': float(close.tail(252).min()),
        '52w_high': float(close.tail(252).max()),
    }


def score_stock(tech, info):
    score = 0
    reasons = []

    price = tech['price']
    r = tech['rsi']

    # Technical (50 pts)
    if r < 35:
        score += 18; reasons.append('RSI oversold')
    elif r < 45:
        score += 13; reasons.append('RSI attractive')
    elif r < 55:
        score += 8
    elif r > 70:
        score -= 8; reasons.append('RSI overbought')

    if price <= tech['ma20']:
        score += 8; reasons.append('Below/near MA20')
    if price <= tech['ma50']:
        score += 10; reasons.append('Below/near MA50')
    elif price <= tech['ma50'] * 1.03:
        score += 6

    if not np.isnan(tech['ma200']):
        if price > tech['ma200']:
            score += 5; reasons.append('Above MA200 trend')
        elif price >= tech['ma200'] * 0.95:
            score += 3

    if tech['macd_hist'] > 0:
        score += 7; reasons.append('MACD momentum positive')

    # Fundamentals (40 pts)
    trailing_pe = safe_float(info.get('trailingPE'))
    forward_pe = safe_float(info.get('forwardPE'))
    revenue_growth = safe_float(info.get('revenueGrowth'))
    earnings_growth = safe_float(info.get('earningsGrowth'))
    debt_to_equity = safe_float(info.get('debtToEquity'))
    profit_margins = safe_float(info.get('profitMargins'))

    pe = forward_pe if not np.isnan(forward_pe) else trailing_pe
    if not np.isnan(pe):
        if 0 < pe <= 15:
            score += 12; reasons.append('Low PE')
        elif pe <= 25:
            score += 9
        elif pe <= 40:
            score += 5
        elif pe > 80:
            score -= 4; reasons.append('Very high PE')

    if not np.isnan(revenue_growth):
        if revenue_growth >= 0.20:
            score += 10; reasons.append('Strong revenue growth')
        elif revenue_growth >= 0.08:
            score += 7
        elif revenue_growth > 0:
            score += 4
        else:
            score -= 3

    if not np.isnan(earnings_growth):
        if earnings_growth >= 0.20:
            score += 9; reasons.append('Strong earnings growth')
        elif earnings_growth > 0:
            score += 5
        else:
            score -= 3

    if not np.isnan(profit_margins):
        if profit_margins >= 0.20:
            score += 5
        elif profit_margins > 0:
            score += 3
        else:
            score -= 4; reasons.append('Unprofitable')

    if not np.isnan(debt_to_equity):
        if debt_to_equity < 50:
            score += 4
        elif debt_to_equity > 150:
            score -= 4; reasons.append('High debt')

    # Price vs range (10 pts)
    range_pos = (price - tech['52w_low']) / max(tech['52w_high'] - tech['52w_low'], 1e-9)
    if range_pos <= 0.25:
        score += 10; reasons.append('Near lower 52-week range')
    elif range_pos <= 0.50:
        score += 6
    elif range_pos >= 0.90:
        score -= 4

    return max(0, min(100, int(round(score)))), reasons


def signal_from_score(score):
    if score >= 75:
        return '🟢 BUY ZONE'
    if score >= 60:
        return '🟡 WATCH'
    return '🔴 WAIT'


def option_candidates(ticker_obj, spot, support, min_dte=20, max_dte=50, min_otm_pct=5):
    rows = []
    expiries = ticker_obj.options or []
    today = datetime.utcnow().date()

    for exp in expiries:
        try:
            dte = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days
        except Exception:
            continue
        if dte < min_dte or dte > max_dte:
            continue
        try:
            puts = ticker_obj.option_chain(exp).puts.copy()
        except Exception:
            continue
        if puts.empty:
            continue

        for _, p in puts.iterrows():
            strike = safe_float(p.get('strike'))
            bid = safe_float(p.get('bid'), 0)
            ask = safe_float(p.get('ask'), 0)
            last = safe_float(p.get('lastPrice'), 0)
            iv = safe_float(p.get('impliedVolatility'))
            oi = safe_float(p.get('openInterest'), 0)
            volume = safe_float(p.get('volume'), 0)
            mid = np.nanmean([x for x in [bid, ask] if x > 0]) if (bid > 0 or ask > 0) else last
            if np.isnan(strike) or strike >= spot or mid <= 0:
                continue

            otm_pct = (spot - strike) / spot * 100
            if otm_pct < min_otm_pct:
                continue

            premium_yield = mid / strike * 100
            annualized = premium_yield * 365 / max(dte, 1)
            support_buffer = (support - strike) / spot * 100

            # Sell-put score (0-100)
            score = 0
            if strike <= support:
                score += 25
            elif strike <= support * 1.03:
                score += 18
            elif strike <= support * 1.06:
                score += 10

            if otm_pct >= 12:
                score += 20
            elif otm_pct >= 8:
                score += 15
            else:
                score += 8

            if premium_yield >= 2.5:
                score += 22
            elif premium_yield >= 1.5:
                score += 16
            elif premium_yield >= 1.0:
                score += 10

            if annualized >= 20:
                score += 12
            elif annualized >= 12:
                score += 8

            if oi >= 500:
                score += 10
            elif oi >= 100:
                score += 6

            if not np.isnan(iv):
                if iv >= 0.45:
                    score += 8
                elif iv >= 0.30:
                    score += 5

            if volume >= 50:
                score += 3

            rows.append({
                'Expiry': exp,
                'DTE': dte,
                'Strike': strike,
                'Mid Premium': round(float(mid), 2),
                'OTM %': round(otm_pct, 1),
                'Premium Yield %': round(premium_yield, 2),
                'Annualized %': round(annualized, 1),
                'IV %': round(iv * 100, 1) if not np.isnan(iv) else np.nan,
                'Open Interest': int(oi) if not np.isnan(oi) else 0,
                'Support Buffer %': round(support_buffer, 1),
                'Put Score': min(100, int(round(score))),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['Put Score', 'Premium Yield %'], ascending=False)


@st.cache_data(ttl=900, show_spinner=False)
def analyze_ticker(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period='1y', auto_adjust=True)
    info = t.info or {}
    tech = technical_snapshot(hist)
    score, reasons = score_stock(tech, info)
    return tech, info, score, reasons

# --------------------------- UI ---------------------------
st.title('📈 Stock Buy & Sell Put Detector')
st.caption('Educational screening tool — not financial advice. Live data is fetched from Yahoo Finance through yfinance.')

with st.sidebar:
    st.header('Your Watchlist')
    default_watchlist = 'MU,QCOM,CRWD,NVDA,INTC,IREN,PATH,ROKU,NIO,MRVL'
    watchlist_text = st.text_area('Tickers (comma separated)', default_watchlist, height=130)
    min_buy_score = st.slider('Minimum Buy Score', 0, 100, 60, 5)
    st.divider()
    st.subheader('Sell Put Filters')
    min_dte = st.slider('Minimum DTE', 7, 60, 20)
    max_dte = st.slider('Maximum DTE', 14, 90, 50)
    min_otm = st.slider('Minimum OTM %', 1, 25, 5)
    run = st.button('🔎 Scan Now', type='primary', use_container_width=True)

symbols = [s.strip().upper() for s in watchlist_text.split(',') if s.strip()]

if run or 'results' not in st.session_state:
    results = []
    errors = []
    progress = st.progress(0)
    status = st.empty()
    for i, symbol in enumerate(symbols):
        status.write(f'Analyzing **{symbol}**...')
        try:
            tech, info, score, reasons = analyze_ticker(symbol)
            results.append({
                'Ticker': symbol,
                'Price': round(tech['price'], 2),
                'RSI': round(tech['rsi'], 1),
                'MA20': round(tech['ma20'], 2),
                'MA50': round(tech['ma50'], 2),
                'MA200': round(tech['ma200'], 2) if not np.isnan(tech['ma200']) else np.nan,
                'Support': round(tech['support'], 2),
                'Resistance': round(tech['resistance'], 2),
                'Forward PE': round(safe_float(info.get('forwardPE')), 1) if not np.isnan(safe_float(info.get('forwardPE'))) else np.nan,
                'Revenue Growth %': round(safe_float(info.get('revenueGrowth'))*100, 1) if not np.isnan(safe_float(info.get('revenueGrowth'))) else np.nan,
                'Earnings Growth %': round(safe_float(info.get('earningsGrowth'))*100, 1) if not np.isnan(safe_float(info.get('earningsGrowth'))) else np.nan,
                'Score': score,
                'Signal': signal_from_score(score),
                'Why': ', '.join(reasons[:4]) if reasons else 'Mixed signals',
            })
        except Exception as e:
            errors.append(f'{symbol}: {e}')
        progress.progress((i+1)/max(len(symbols), 1))
    status.empty(); progress.empty()
    st.session_state['results'] = pd.DataFrame(results).sort_values('Score', ascending=False) if results else pd.DataFrame()
    st.session_state['errors'] = errors

results_df = st.session_state.get('results', pd.DataFrame())
errors = st.session_state.get('errors', [])

if not results_df.empty:
    best = results_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric('Top Candidate', best['Ticker'])
    c2.metric('Buy Score', f"{int(best['Score'])}/100")
    c3.metric('Signal', best['Signal'])

    st.subheader('Buy Stock Rankings')
    filtered = results_df[results_df['Score'] >= min_buy_score].copy()
    st.dataframe(filtered if not filtered.empty else results_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader('Stock Detail')
    selected = st.selectbox('Choose a stock', results_df['Ticker'].tolist())
    row = results_df[results_df['Ticker'] == selected].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric('Price', f"${row['Price']:.2f}")
    d2.metric('RSI', f"{row['RSI']:.1f}")
    d3.metric('Support', f"${row['Support']:.2f}")
    d4.metric('Resistance', f"${row['Resistance']:.2f}")

    try:
        chart_hist = yf.Ticker(selected).history(period='6mo', auto_adjust=True)
        chart_df = chart_hist[['Close']].copy()
        chart_df['MA20'] = chart_df['Close'].rolling(20).mean()
        chart_df['MA50'] = chart_df['Close'].rolling(50).mean()
        st.line_chart(chart_df)
    except Exception:
        pass

    st.info(f"**Why the score:** {row['Why']}")

    st.divider()
    st.subheader('Cash-Secured Put Finder')
    st.caption('Finds puts roughly 20–50 DTE by default, preferring strikes near/below support and reasonable premium yield.')
    if st.button(f'Find puts for {selected}', use_container_width=True):
        with st.spinner('Scanning option chain...'):
            try:
                t = yf.Ticker(selected)
                puts = option_candidates(t, row['Price'], row['Support'], min_dte, max_dte, min_otm)
                if puts.empty:
                    st.warning('No put contracts matched your current filters.')
                else:
                    st.dataframe(puts.head(20), use_container_width=True, hide_index=True)
                    top_put = puts.iloc[0]
                    collateral = top_put['Strike'] * 100
                    premium_cash = top_put['Mid Premium'] * 100
                    st.success(
                        f"Top put candidate: {selected} {top_put['Expiry']} ${top_put['Strike']:.2f} put — "
                        f"mid premium ≈ ${premium_cash:,.0f}, collateral ≈ ${collateral:,.0f}, score {int(top_put['Put Score'])}/100."
                    )
            except Exception as e:
                st.error(f'Could not load options: {e}')

if errors:
    with st.expander('Symbols with data issues'):
        for e in errors:
            st.write(e)

st.divider()
st.caption('Scoring is intentionally transparent and rule-based. You can modify the weights in app.py to match your own investing style.')
