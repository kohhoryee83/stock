import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title='Technical & Options Stock Screener', page_icon='📊', layout='wide')

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
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = hist['High'], hist['Low'], hist['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width_pct = (upper - lower) / mid.replace(0, np.nan) * 100
    return mid, upper, lower, width_pct


def crossed_up(a: pd.Series, b: pd.Series) -> bool:
    return len(a.dropna()) >= 2 and len(b.dropna()) >= 2 and a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]


def crossed_down(a: pd.Series, b: pd.Series) -> bool:
    return len(a.dropna()) >= 2 and len(b.dropna()) >= 2 and a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]


def recent_cross(a: pd.Series, b: pd.Series, days: int = 5):
    # Returns ('up'|'down'|None, days_ago)
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None, None
    aa, bb = a.loc[common], b.loc[common]
    start = max(1, len(common) - days)
    for i in range(len(common) - 1, start - 1, -1):
        if aa.iloc[i - 1] <= bb.iloc[i - 1] and aa.iloc[i] > bb.iloc[i]:
            return 'up', len(common) - 1 - i
        if aa.iloc[i - 1] >= bb.iloc[i - 1] and aa.iloc[i] < bb.iloc[i]:
            return 'down', len(common) - 1 - i
    return None, None


def swing_levels(hist: pd.DataFrame, lookback: int = 90):
    h = hist.tail(lookback).copy()
    close = h['Close'].dropna()
    if close.empty:
        return np.nan, np.nan

    # Blend recent swing extrema with robust percentiles to reduce one-day outlier sensitivity.
    lows = h['Low'].rolling(5, center=True).min()
    highs = h['High'].rolling(5, center=True).max()
    swing_lows = h.loc[h['Low'].eq(lows), 'Low'].dropna()
    swing_highs = h.loc[h['High'].eq(highs), 'High'].dropna()

    support_candidates = swing_lows[swing_lows < close.iloc[-1]]
    resistance_candidates = swing_highs[swing_highs > close.iloc[-1]]

    support = support_candidates.tail(6).median() if not support_candidates.empty else np.percentile(close, 10)
    resistance = resistance_candidates.tail(6).median() if not resistance_candidates.empty else np.percentile(close, 90)
    return float(support), float(resistance)


def technical_snapshot(hist: pd.DataFrame):
    if hist.empty or len(hist) < 210:
        raise ValueError('Need at least ~210 trading days of price history')

    close = hist['Close'].dropna()
    ma20_s = close.rolling(20).mean()
    ma50_s = close.rolling(50).mean()
    ma200_s = close.rolling(200).mean()
    rsi_s = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    bb_mid, bb_upper, bb_lower, bb_width = bollinger(close)
    atr_s = atr(hist, 14)

    price = float(close.iloc[-1])
    support, resistance = swing_levels(hist, 90)
    cross_type, cross_days_ago = recent_cross(ma50_s, ma200_s, 7)
    macd_cross_type, macd_cross_days_ago = recent_cross(macd_line, macd_signal, 5)

    # Sideways = compressed volatility + modest 20d trading range + flat MA20.
    recent20 = close.tail(20)
    range20_pct = (recent20.max() - recent20.min()) / max(recent20.mean(), 1e-9) * 100
    ma20_slope_pct = (ma20_s.iloc[-1] / ma20_s.iloc[-6] - 1) * 100 if pd.notna(ma20_s.iloc[-6]) else np.nan
    bb_width_now = float(bb_width.iloc[-1])
    atr_pct = float(atr_s.iloc[-1] / price * 100)
    sideways = bool(range20_pct <= 8 and abs(ma20_slope_pct) <= 2.0 and bb_width_now <= 10)

    return {
        'price': price,
        'ma20': float(ma20_s.iloc[-1]),
        'ma50': float(ma50_s.iloc[-1]),
        'ma200': float(ma200_s.iloc[-1]),
        'rsi': float(rsi_s.iloc[-1]),
        'macd': float(macd_line.iloc[-1]),
        'macd_signal': float(macd_signal.iloc[-1]),
        'macd_hist': float(macd_hist.iloc[-1]),
        'bb_upper': float(bb_upper.iloc[-1]),
        'bb_lower': float(bb_lower.iloc[-1]),
        'bb_width_pct': bb_width_now,
        'atr_pct': atr_pct,
        'range20_pct': float(range20_pct),
        'ma20_slope_pct': float(ma20_slope_pct),
        'sideways': sideways,
        'support': support,
        'resistance': resistance,
        '52w_low': float(close.tail(252).min()),
        '52w_high': float(close.tail(252).max()),
        'cross_type': cross_type,
        'cross_days_ago': cross_days_ago,
        'macd_cross_type': macd_cross_type,
        'macd_cross_days_ago': macd_cross_days_ago,
    }


def build_alerts(tech, rsi_buy=35, rsi_sell=70, near_ma_pct=1.5, near_level_pct=2.0):
    p = tech['price']
    alerts = []

    if tech['cross_type'] == 'up':
        alerts.append(('🟢', 'Golden Cross', f"MA50 crossed above MA200 {tech['cross_days_ago']} trading day(s) ago."))
    elif tech['cross_type'] == 'down':
        alerts.append(('🔴', 'Death Cross', f"MA50 crossed below MA200 {tech['cross_days_ago']} trading day(s) ago."))

    if tech['sideways']:
        alerts.append(('🟡', 'Moving Sideways', f"20-day range {tech['range20_pct']:.1f}%, Bollinger width {tech['bb_width_pct']:.1f}%."))

    if tech['rsi'] <= rsi_buy:
        alerts.append(('🟢', 'RSI Buy Alert', f"RSI {tech['rsi']:.1f} is at/below {rsi_buy}."))
    if tech['rsi'] >= rsi_sell:
        alerts.append(('🔴', 'RSI Sell Alert', f"RSI {tech['rsi']:.1f} is at/above {rsi_sell}."))

    if tech['macd_cross_type'] == 'up':
        alerts.append(('🟢', 'MACD Bullish Cross', f"MACD crossed above signal {tech['macd_cross_days_ago']} trading day(s) ago."))
    elif tech['macd_cross_type'] == 'down':
        alerts.append(('🔴', 'MACD Bearish Cross', f"MACD crossed below signal {tech['macd_cross_days_ago']} trading day(s) ago."))

    for label, ma in [('MA20', tech['ma20']), ('MA50', tech['ma50']), ('MA200', tech['ma200'])]:
        dist = abs(p - ma) / p * 100
        if dist <= near_ma_pct:
            side = 'above' if p >= ma else 'below'
            alerts.append(('🔵', f'Near {label}', f"Price is {dist:.1f}% {side} {label} (${ma:.2f})."))

    if pd.notna(tech['support']):
        support_dist = abs(p - tech['support']) / p * 100
        if support_dist <= near_level_pct:
            alerts.append(('🟢', 'Near Support', f"Price is {support_dist:.1f}% from support near ${tech['support']:.2f}."))
    if pd.notna(tech['resistance']):
        resistance_dist = abs(tech['resistance'] - p) / p * 100
        if resistance_dist <= near_level_pct:
            alerts.append(('🔴', 'Near Resistance', f"Price is {resistance_dist:.1f}% from resistance near ${tech['resistance']:.2f}."))

    if p <= tech['bb_lower']:
        alerts.append(('🟢', 'Lower Bollinger Band', 'Price is at/below the lower Bollinger Band.'))
    if p >= tech['bb_upper']:
        alerts.append(('🔴', 'Upper Bollinger Band', 'Price is at/above the upper Bollinger Band.'))

    return alerts


def trade_bias(tech):
    # Simple technical bias, intentionally rule-based.
    buy = 0
    sell = 0
    if tech['rsi'] < 40: buy += 2
    if tech['rsi'] > 65: sell += 2
    if tech['macd_hist'] > 0: buy += 1
    else: sell += 1
    if tech['price'] > tech['ma50']: buy += 1
    else: sell += 1
    if tech['price'] > tech['ma200']: buy += 1
    else: sell += 1
    if tech['cross_type'] == 'up': buy += 2
    if tech['cross_type'] == 'down': sell += 2
    if tech['price'] <= tech['support'] * 1.03: buy += 2
    if tech['price'] >= tech['resistance'] * 0.97: sell += 2

    if buy >= sell + 3:
        return '🟢 Bullish / Buy-watch'
    if sell >= buy + 3:
        return '🔴 Bearish / Sell-watch'
    return '🟡 Neutral / Range'


def option_candidates(ticker_obj, spot, support, resistance, target_dte=20, dte_tolerance=7, min_otm_pct=3):
    rows = []
    expiries = ticker_obj.options or []
    today = datetime.utcnow().date()

    for exp in expiries:
        try:
            dte = (datetime.strptime(exp, '%Y-%m-%d').date() - today).days
        except Exception:
            continue
        if abs(dte - target_dte) > dte_tolerance:
            continue
        try:
            chain = ticker_obj.option_chain(exp)
        except Exception:
            continue

        for opt_type, df in [('PUT', chain.puts), ('CALL', chain.calls)]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                strike = safe_float(row.get('strike'))
                bid = safe_float(row.get('bid'), 0)
                ask = safe_float(row.get('ask'), 0)
                last = safe_float(row.get('lastPrice'), 0)
                iv = safe_float(row.get('impliedVolatility'))
                oi = safe_float(row.get('openInterest'), 0)
                volume = safe_float(row.get('volume'), 0)
                delta = safe_float(row.get('delta'))
                prices = [x for x in [bid, ask] if x > 0]
                mid = float(np.mean(prices)) if prices else last
                if np.isnan(strike) or mid <= 0:
                    continue

                if opt_type == 'PUT':
                    if strike >= spot:
                        continue
                    otm_pct = (spot - strike) / spot * 100
                    if otm_pct < min_otm_pct:
                        continue
                    level_gap = (support - strike) / spot * 100 if pd.notna(support) else np.nan
                    level_ok = pd.notna(support) and strike <= support * 1.03
                else:
                    if strike <= spot:
                        continue
                    otm_pct = (strike - spot) / spot * 100
                    if otm_pct < min_otm_pct:
                        continue
                    level_gap = (strike - resistance) / spot * 100 if pd.notna(resistance) else np.nan
                    level_ok = pd.notna(resistance) and strike >= resistance * 0.97

                premium_yield = mid / strike * 100
                annualized = premium_yield * 365 / max(dte, 1)

                score = 0
                if level_ok: score += 28
                if otm_pct >= 10: score += 18
                elif otm_pct >= 6: score += 14
                else: score += 8
                if premium_yield >= 2.0: score += 20
                elif premium_yield >= 1.0: score += 13
                elif premium_yield >= 0.5: score += 7
                if oi >= 500: score += 12
                elif oi >= 100: score += 7
                if not np.isnan(iv):
                    if iv >= 0.45: score += 10
                    elif iv >= 0.30: score += 6
                if volume >= 50: score += 5
                if abs(dte - target_dte) <= 2: score += 7

                rows.append({
                    'Type': opt_type,
                    'Expiry': exp,
                    'DTE': dte,
                    'Strike': strike,
                    'Mid Premium': round(mid, 2),
                    'OTM %': round(otm_pct, 1),
                    'Premium Yield %': round(premium_yield, 2),
                    'Annualized %': round(annualized, 1),
                    'IV %': round(iv * 100, 1) if not np.isnan(iv) else np.nan,
                    'Open Interest': int(oi) if not np.isnan(oi) else 0,
                    'Volume': int(volume) if not np.isnan(volume) else 0,
                    'Vs Support/Resistance %': round(level_gap, 1) if not np.isnan(level_gap) else np.nan,
                    'Option Score': min(100, int(round(score))),
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['Option Score', 'Premium Yield %'], ascending=False)


@st.cache_data(ttl=900, show_spinner=False)
def analyze_ticker(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period='2y', auto_adjust=True)
    tech = technical_snapshot(hist)
    return tech


# --------------------------- UI ---------------------------
st.title('📊 Technical & Options Stock Screener')
st.caption('Golden/death cross • sideways detection • RSI/MACD/MA alerts • support/resistance • ~20 DTE sell puts & covered calls')
st.warning('Educational screening tool only — technical signals can fail, and options involve assignment and loss risk.')

with st.sidebar:
    st.header('Watchlist')
    default_watchlist = 'MU,QCOM,CRWD,NVDA,INTC,IREN,PATH,ROKU,NIO,MRVL'
    watchlist_text = st.text_area('Tickers (comma separated)', default_watchlist, height=130)

    st.divider()
    st.subheader('Alert thresholds')
    rsi_buy = st.slider('RSI buy alert ≤', 20, 50, 35)
    rsi_sell = st.slider('RSI sell alert ≥', 55, 85, 70)
    near_ma_pct = st.slider('Near MA threshold (%)', 0.5, 5.0, 1.5, 0.5)
    near_level_pct = st.slider('Near support/resistance (%)', 0.5, 5.0, 2.0, 0.5)

    st.divider()
    st.subheader('Options around 20 DTE')
    target_dte = st.slider('Target DTE', 10, 45, 20)
    dte_tolerance = st.slider('DTE tolerance (+/- days)', 1, 14, 7)
    min_otm = st.slider('Minimum OTM %', 1, 20, 3)

    run = st.button('🔎 Scan Now', type='primary', use_container_width=True)

symbols = [s.strip().upper() for s in watchlist_text.split(',') if s.strip()]

if run or 'technical_results' not in st.session_state:
    results, errors = [], []
    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        status.write(f'Analyzing **{symbol}**...')
        try:
            tech = analyze_ticker(symbol)
            alerts = build_alerts(tech, rsi_buy, rsi_sell, near_ma_pct, near_level_pct)
            cross = '—'
            if tech['cross_type'] == 'up': cross = '🟢 Golden Cross'
            elif tech['cross_type'] == 'down': cross = '🔴 Death Cross'
            sideways = '🟡 Yes' if tech['sideways'] else 'No'

            results.append({
                'Ticker': symbol,
                'Price': round(tech['price'], 2),
                'Bias': trade_bias(tech),
                'Cross': cross,
                'Sideways': sideways,
                'RSI': round(tech['rsi'], 1),
                'MA20': round(tech['ma20'], 2),
                'MA50': round(tech['ma50'], 2),
                'MA200': round(tech['ma200'], 2),
                'MACD Hist': round(tech['macd_hist'], 3),
                'Support': round(tech['support'], 2),
                'Resistance': round(tech['resistance'], 2),
                'ATR %': round(tech['atr_pct'], 1),
                '20d Range %': round(tech['range20_pct'], 1),
                'Alerts': len(alerts),
                'Alert Summary': ' | '.join([f'{a[0]} {a[1]}' for a in alerts]) if alerts else 'None',
            })
        except Exception as e:
            errors.append(f'{symbol}: {e}')
        progress.progress((i + 1) / max(len(symbols), 1))

    status.empty(); progress.empty()
    st.session_state['technical_results'] = pd.DataFrame(results)
    st.session_state['technical_errors'] = errors

results_df = st.session_state.get('technical_results', pd.DataFrame())
errors = st.session_state.get('technical_errors', [])

if not results_df.empty:
    active_alerts = results_df[results_df['Alerts'] > 0]
    golden = results_df[results_df['Cross'].str.contains('Golden', na=False)]
    death = results_df[results_df['Cross'].str.contains('Death', na=False)]
    sideways_df = results_df[results_df['Sideways'].str.contains('Yes', na=False)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Stocks with Alerts', len(active_alerts))
    c2.metric('Recent Golden Cross', len(golden))
    c3.metric('Recent Death Cross', len(death))
    c4.metric('Moving Sideways', len(sideways_df))

    st.subheader('🚨 Active Alerts')
    if active_alerts.empty:
        st.success('No alert conditions currently triggered.')
    else:
        for _, row in active_alerts.iterrows():
            st.info(f"**{row['Ticker']}** — {row['Alert Summary']}")

    st.subheader('Market Scanner')
    st.dataframe(results_df.sort_values(['Alerts', 'Ticker'], ascending=[False, True]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader('Stock Detail')
    selected = st.selectbox('Choose a stock', results_df['Ticker'].tolist())
    row = results_df[results_df['Ticker'] == selected].iloc[0]
    tech = analyze_ticker(selected)
    alerts = build_alerts(tech, rsi_buy, rsi_sell, near_ma_pct, near_level_pct)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric('Price', f"${row['Price']:.2f}")
    d2.metric('RSI', f"{row['RSI']:.1f}")
    d3.metric('Support', f"${row['Support']:.2f}")
    d4.metric('Resistance', f"${row['Resistance']:.2f}")

    st.write(f"**Technical bias:** {row['Bias']}")
    if alerts:
        for icon, title, text in alerts:
            st.write(f"{icon} **{title}:** {text}")
    else:
        st.write('No alert threshold is currently triggered for this stock.')

    try:
        hist = yf.Ticker(selected).history(period='1y', auto_adjust=True)
        chart_df = hist[['Close']].copy()
        chart_df['MA20'] = chart_df['Close'].rolling(20).mean()
        chart_df['MA50'] = chart_df['Close'].rolling(50).mean()
        chart_df['MA200'] = chart_df['Close'].rolling(200).mean()
        st.line_chart(chart_df)
    except Exception:
        pass

    st.divider()
    st.subheader('🧾 Sell Put / Covered Call Screener')
    st.caption('Scans expiries closest to your target DTE. Sell puts are ranked against support; calls are ranked against resistance.')

    if st.button(f'Scan ~{target_dte} DTE options for {selected}', use_container_width=True):
        with st.spinner('Scanning option chain...'):
            try:
                t = yf.Ticker(selected)
                options = option_candidates(t, tech['price'], tech['support'], tech['resistance'], target_dte, dte_tolerance, min_otm)
                if options.empty:
                    st.warning('No contracts matched the current filters.')
                else:
                    put_tab, call_tab = st.tabs(['Sell Puts', 'Covered Calls'])
                    with put_tab:
                        puts = options[options['Type'] == 'PUT']
                        st.dataframe(puts.head(25), use_container_width=True, hide_index=True)
                        if not puts.empty:
                            top = puts.iloc[0]
                            st.success(
                                f"Top put candidate: {selected} {top['Expiry']} ${top['Strike']:.2f} PUT — "
                                f"DTE {int(top['DTE'])}, mid ${top['Mid Premium']:.2f}, OTM {top['OTM %']:.1f}%, score {int(top['Option Score'])}/100."
                            )
                    with call_tab:
                        calls = options[options['Type'] == 'CALL']
                        st.dataframe(calls.head(25), use_container_width=True, hide_index=True)
                        if not calls.empty:
                            top = calls.iloc[0]
                            st.success(
                                f"Top call candidate: {selected} {top['Expiry']} ${top['Strike']:.2f} CALL — "
                                f"DTE {int(top['DTE'])}, mid ${top['Mid Premium']:.2f}, OTM {top['OTM %']:.1f}%, score {int(top['Option Score'])}/100."
                            )
            except Exception as e:
                st.error(f'Could not load options: {e}')

if errors:
    with st.expander('Symbols with data issues'):
        for e in errors:
            st.write(e)

st.divider()
st.caption('Sideways detection uses 20-day range, Bollinger-band width and MA20 slope. Cross alerts detect MA50/MA200 events occurring within the last 7 trading days.')
