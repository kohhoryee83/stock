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


# --------------------------- Backtest helpers ---------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol, period='5y'):
    return yf.Ticker(symbol).history(period=period, auto_adjust=True)


def build_backtest_frame(hist: pd.DataFrame, rsi_buy=35, rsi_sell=70, near_level_pct=2.0):
    df = hist.copy().dropna(subset=['Close', 'High', 'Low'])
    c = df['Close']
    df['MA20'] = c.rolling(20).mean()
    df['MA50'] = c.rolling(50).mean()
    df['MA200'] = c.rolling(200).mean()
    df['RSI'] = rsi(c, 14)
    macd_line, macd_signal, macd_hist = macd(c)
    df['MACD'] = macd_line
    df['MACDSignal'] = macd_signal
    df['MACDHist'] = macd_hist
    bb_mid, bb_upper, bb_lower, bb_width = bollinger(c)
    df['BBWidth'] = bb_width
    df['Range20Pct'] = (c.rolling(20).max() - c.rolling(20).min()) / c.rolling(20).mean() * 100
    df['MA20SlopePct'] = (df['MA20'] / df['MA20'].shift(5) - 1) * 100
    # Historical support/resistance proxies use only prior information to avoid look-ahead bias.
    df['Support60'] = df['Low'].rolling(60).min().shift(1)
    df['Resistance60'] = df['High'].rolling(60).max().shift(1)
    df['NearSupport'] = ((c - df['Support60']).abs() / c * 100 <= near_level_pct)
    df['NearResistance'] = ((df['Resistance60'] - c).abs() / c * 100 <= near_level_pct)
    df['GoldenCross'] = (df['MA50'].shift(1) <= df['MA200'].shift(1)) & (df['MA50'] > df['MA200'])
    df['DeathCross'] = (df['MA50'].shift(1) >= df['MA200'].shift(1)) & (df['MA50'] < df['MA200'])
    df['MACDBullCross'] = (df['MACD'].shift(1) <= df['MACDSignal'].shift(1)) & (df['MACD'] > df['MACDSignal'])
    df['MACDBearCross'] = (df['MACD'].shift(1) >= df['MACDSignal'].shift(1)) & (df['MACD'] < df['MACDSignal'])
    df['Sideways'] = (df['Range20Pct'] <= 8) & (df['MA20SlopePct'].abs() <= 2.0) & (df['BBWidth'] <= 10)
    df['RSIBuy'] = df['RSI'] <= rsi_buy
    df['RSISell'] = df['RSI'] >= rsi_sell
    df['NearMA20'] = (c - df['MA20']).abs() / c * 100 <= 1.5
    df['NearMA50'] = (c - df['MA50']).abs() / c * 100 <= 1.5
    df['NearMA200'] = (c - df['MA200']).abs() / c * 100 <= 1.5
    # Combined thesis proxies; these do NOT reproduce actual option premiums or assignment P&L.
    df['BuyCombo'] = df['RSIBuy'] & df['NearSupport'] & (df['MACDHist'] > 0)
    df['SellPutProxy'] = df['NearSupport'] & (df['RSI'] <= 45) & (c >= df['MA200'])
    df['CoveredCallProxy'] = df['NearResistance'] & (df['RSI'] >= 55)
    for n in [5, 10, 20, 40]:
        df[f'Fwd{n}'] = c.shift(-n) / c - 1
    return df


STRATEGY_MAP = {
    'Golden Cross': ('GoldenCross', 'bullish'),
    'Death Cross': ('DeathCross', 'bearish'),
    'RSI Oversold': ('RSIBuy', 'bullish'),
    'RSI Overbought': ('RSISell', 'bearish'),
    'MACD Bullish Cross': ('MACDBullCross', 'bullish'),
    'MACD Bearish Cross': ('MACDBearCross', 'bearish'),
    'Near Support': ('NearSupport', 'bullish'),
    'Near Resistance': ('NearResistance', 'bearish'),
    'Near MA20': ('NearMA20', 'bullish'),
    'Near MA50': ('NearMA50', 'bullish'),
    'Near MA200': ('NearMA200', 'bullish'),
    'Moving Sideways': ('Sideways', 'neutral'),
    'RSI + Support + Bullish MACD': ('BuyCombo', 'bullish'),
    'Sell Put Setup (stock proxy)': ('SellPutProxy', 'put_proxy'),
    'Covered Call Setup (stock proxy)': ('CoveredCallProxy', 'call_proxy'),
}


def backtest_summary(df: pd.DataFrame, strategy_name: str):
    col, direction = STRATEGY_MAP[strategy_name]
    sig = df[df[col].fillna(False)].copy()
    sig = sig.dropna(subset=['Fwd20'])
    if sig.empty:
        return None, sig
    f20 = sig['Fwd20']
    if direction == 'bullish':
        wins = f20 > 0
        win_label = '20D positive return'
    elif direction == 'bearish':
        wins = f20 < 0
        win_label = '20D negative return'
    elif direction == 'put_proxy':
        # Underlying did not fall more than 5% by ~20 trading days.
        wins = f20 > -0.05
        win_label = '20D return > -5% (proxy only)'
    elif direction == 'call_proxy':
        # Underlying did not rise more than 5% by ~20 trading days.
        wins = f20 < 0.05
        win_label = '20D return < +5% (proxy only)'
    else:
        wins = f20.abs() <= 0.05
        win_label = '20D stayed within ±5%'

    summary = {
        'Signals': int(len(sig)),
        'Win Rate %': round(float(wins.mean() * 100), 1),
        'Win definition': win_label,
        'Avg 5D %': round(float(sig['Fwd5'].mean() * 100), 2),
        'Avg 10D %': round(float(sig['Fwd10'].mean() * 100), 2),
        'Avg 20D %': round(float(sig['Fwd20'].mean() * 100), 2),
        'Avg 40D %': round(float(sig['Fwd40'].mean() * 100), 2),
        'Median 20D %': round(float(sig['Fwd20'].median() * 100), 2),
        'Worst 20D %': round(float(sig['Fwd20'].min() * 100), 2),
        'Best 20D %': round(float(sig['Fwd20'].max() * 100), 2),
    }
    return summary, sig


def persist_watchlist(symbols):
    clean = []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in clean:
            clean.append(s)
    st.session_state['watchlist'] = clean
    # Query parameter survives normal browser refreshes and can be bookmarked/shared.
    st.query_params['watchlist'] = ','.join(clean)


def current_watchlist():
    default = ['MU', 'QCOM', 'CRWD', 'NVDA', 'INTC', 'IREN', 'PATH', 'ROKU', 'NIO', 'MRVL']
    if 'watchlist' not in st.session_state:
        qp = st.query_params.get('watchlist', '')
        if isinstance(qp, list):
            qp = qp[0] if qp else ''
        saved = [s.strip().upper() for s in str(qp).split(',') if s.strip()]
        st.session_state['watchlist'] = saved or default
    return st.session_state['watchlist']


# --------------------------- UI ---------------------------
st.title('📊 Technical & Options Stock Screener V3')
st.caption('Live alerts • persistent watchlist • ~20 DTE option setups • historical backtests')
st.warning('Educational screening tool only — technical signals can fail. Options can be assigned and may cause substantial losses.')

symbols = current_watchlist()

with st.sidebar:
    st.header('Watchlist')
    st.caption('Add/remove tickers here. The list is saved into this page URL so it survives refreshes.')
    new_ticker = st.text_input('Add a ticker', placeholder='e.g. AMD')
    cadd, csave = st.columns(2)
    if cadd.button('➕ Add', use_container_width=True):
        s = new_ticker.strip().upper()
        if not s:
            st.warning('Type a ticker first.')
        elif not all(ch.isalnum() or ch in '.-' for ch in s):
            st.error('Ticker contains unsupported characters.')
        else:
            persist_watchlist(symbols + [s])
            st.session_state.pop('technical_results', None)
            st.rerun()
    if csave.button('💾 Save', use_container_width=True):
        persist_watchlist(symbols)
        st.success('Watchlist saved to the page URL. Bookmark this page to keep it across future visits.')

    if symbols:
        remove_ticker = st.selectbox('Remove a ticker', symbols)
        if st.button('➖ Remove selected', use_container_width=True):
            persist_watchlist([s for s in symbols if s != remove_ticker])
            st.session_state.pop('technical_results', None)
            st.rerun()
    st.code(', '.join(symbols), language=None)

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

    run = st.button('🔎 Scan Watchlist', type='primary', use_container_width=True)


def scan_watchlist(symbols):
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
            bias = trade_bias(tech)
            # Action-oriented classification for quick scanning.
            if tech['sideways'] and tech['price'] <= tech['support'] * 1.05:
                action = '🟡 Sell Put / Range setup'
            elif tech['sideways'] and tech['price'] >= tech['resistance'] * 0.95:
                action = '🟡 Covered Call / Range setup'
            elif ('Bullish' in bias) or (tech['rsi'] <= rsi_buy and tech['price'] <= tech['support'] * 1.05):
                action = '🟢 Buy / Sell Put watch'
            elif ('Bearish' in bias) or (tech['rsi'] >= rsi_sell and tech['price'] >= tech['resistance'] * 0.95):
                action = '🔴 Sell / Covered Call watch'
            else:
                action = '⚪ Wait / Monitor'

            results.append({
                'Ticker': symbol,
                'Price': round(tech['price'], 2),
                'Action': action,
                'Bias': bias,
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
    return pd.DataFrame(results), errors


# Rescan if requested, no cached results, or watchlist changed.
last_scanned = st.session_state.get('last_scanned_symbols', [])
if run or 'technical_results' not in st.session_state or list(symbols) != list(last_scanned):
    df_now, errors_now = scan_watchlist(symbols)
    st.session_state['technical_results'] = df_now
    st.session_state['technical_errors'] = errors_now
    st.session_state['last_scanned_symbols'] = list(symbols)

results_df = st.session_state.get('technical_results', pd.DataFrame())
errors = st.session_state.get('technical_errors', [])

live_tab, options_tab, backtest_tab = st.tabs(['📡 Live Screener', '💰 Options Setup', '🧪 Backtest'])

with live_tab:
    if results_df.empty:
        st.warning('No valid ticker data yet. Add a ticker in the sidebar and scan again.')
    else:
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
                st.info(f"**{row['Ticker']} — {row['Action']}** | {row['Alert Summary']}")

        st.subheader('Market Scanner')
        st.dataframe(results_df.sort_values(['Alerts', 'Ticker'], ascending=[False, True]), use_container_width=True, hide_index=True)

        st.divider()
        selected = st.selectbox('Stock detail', results_df['Ticker'].tolist(), key='live_selected')
        row = results_df[results_df['Ticker'] == selected].iloc[0]
        tech = analyze_ticker(selected)
        alerts = build_alerts(tech, rsi_buy, rsi_sell, near_ma_pct, near_level_pct)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Price', f"${row['Price']:.2f}")
        d2.metric('RSI', f"{row['RSI']:.1f}")
        d3.metric('Support', f"${row['Support']:.2f}")
        d4.metric('Resistance', f"${row['Resistance']:.2f}")
        st.write(f"**Suggested technical action:** {row['Action']}")
        st.write(f"**Technical bias:** {row['Bias']}")
        for icon, title, text in alerts:
            st.write(f"{icon} **{title}:** {text}")
        try:
            hist = load_history(selected, '1y')
            chart_df = hist[['Close']].copy()
            chart_df['MA20'] = chart_df['Close'].rolling(20).mean()
            chart_df['MA50'] = chart_df['Close'].rolling(50).mean()
            chart_df['MA200'] = chart_df['Close'].rolling(200).mean()
            st.line_chart(chart_df)
        except Exception:
            pass

with options_tab:
    st.subheader('💰 ~20 DTE Sell Put / Covered Call Screener')
    st.caption('Sell puts are ranked partly by support; covered calls partly by resistance. Option data comes from Yahoo Finance and can occasionally be incomplete.')
    option_symbol = st.selectbox('Choose stock', symbols, key='option_selected') if symbols else None
    if option_symbol:
        try:
            tech_opt = analyze_ticker(option_symbol)
            o1, o2, o3 = st.columns(3)
            o1.metric('Spot', f"${tech_opt['price']:.2f}")
            o2.metric('Support', f"${tech_opt['support']:.2f}")
            o3.metric('Resistance', f"${tech_opt['resistance']:.2f}")
            if st.button(f'Scan ~{target_dte} DTE options for {option_symbol}', type='primary', use_container_width=True):
                with st.spinner('Scanning option chain...'):
                    t = yf.Ticker(option_symbol)
                    opts = option_candidates(t, tech_opt['price'], tech_opt['support'], tech_opt['resistance'], target_dte, dte_tolerance, min_otm)
                    st.session_state['option_results'] = opts
                    st.session_state['option_results_symbol'] = option_symbol
            opts = st.session_state.get('option_results', pd.DataFrame())
            if st.session_state.get('option_results_symbol') == option_symbol and not opts.empty:
                put_tab, call_tab = st.tabs(['Sell Puts', 'Covered Calls'])
                with put_tab:
                    puts = opts[opts['Type'] == 'PUT']
                    st.dataframe(puts.head(25), use_container_width=True, hide_index=True)
                    if not puts.empty:
                        top = puts.iloc[0]
                        st.success(f"Top put: {option_symbol} {top['Expiry']} ${top['Strike']:.2f} PUT | DTE {int(top['DTE'])} | mid ${top['Mid Premium']:.2f} | OTM {top['OTM %']:.1f}% | score {int(top['Option Score'])}/100")
                with call_tab:
                    calls = opts[opts['Type'] == 'CALL']
                    st.dataframe(calls.head(25), use_container_width=True, hide_index=True)
                    if not calls.empty:
                        top = calls.iloc[0]
                        st.success(f"Top call: {option_symbol} {top['Expiry']} ${top['Strike']:.2f} CALL | DTE {int(top['DTE'])} | mid ${top['Mid Premium']:.2f} | OTM {top['OTM %']:.1f}% | score {int(top['Option Score'])}/100")
            elif st.session_state.get('option_results_symbol') == option_symbol and isinstance(opts, pd.DataFrame) and opts.empty:
                st.info('Press the scan button to load option contracts.')
        except Exception as e:
            st.error(f'Could not load options setup: {e}')

with backtest_tab:
    st.subheader('🧪 Historical Signal Backtest')
    st.caption('Uses adjusted daily stock prices. Option strategies are stock-behaviour proxies only; they do not reconstruct historical option premiums, IV, early assignment or transaction costs.')
    if symbols:
        b1, b2, b3 = st.columns(3)
        bt_symbol = b1.selectbox('Ticker', symbols, key='bt_symbol')
        bt_strategy = b2.selectbox('Strategy', list(STRATEGY_MAP.keys()), key='bt_strategy')
        bt_period = b3.selectbox('History', ['1y', '3y', '5y', '10y'], index=2, key='bt_period')

        if st.button('Run Backtest', type='primary'):
            with st.spinner(f'Backtesting {bt_symbol}...'):
                try:
                    hist_bt = load_history(bt_symbol, bt_period)
                    frame = build_backtest_frame(hist_bt, rsi_buy, rsi_sell, near_level_pct)
                    summary, sig = backtest_summary(frame, bt_strategy)
                    st.session_state['bt_summary'] = summary
                    st.session_state['bt_signals'] = sig
                    st.session_state['bt_key'] = (bt_symbol, bt_strategy, bt_period)
                except Exception as e:
                    st.error(f'Backtest failed: {e}')

        if st.session_state.get('bt_key') == (bt_symbol, bt_strategy, bt_period):
            summary = st.session_state.get('bt_summary')
            sig = st.session_state.get('bt_signals', pd.DataFrame())
            if not summary:
                st.warning('No usable historical signals for this combination.')
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Signals', summary['Signals'])
                m2.metric('20D Win Rate', f"{summary['Win Rate %']:.1f}%")
                m3.metric('Avg 20D Return', f"{summary['Avg 20D %']:+.2f}%")
                m4.metric('Worst 20D Return', f"{summary['Worst 20D %']:+.2f}%")
                st.caption(f"Win definition: {summary['Win definition']}")
                metric_df = pd.DataFrame([summary]).drop(columns=['Win definition'])
                st.dataframe(metric_df, use_container_width=True, hide_index=True)
                show = sig[['Close', 'RSI', 'MA20', 'MA50', 'MA200', 'Fwd5', 'Fwd10', 'Fwd20', 'Fwd40']].tail(100).copy()
                for c in ['Fwd5','Fwd10','Fwd20','Fwd40']:
                    show[c] = (show[c] * 100).round(2)
                st.write('Most recent historical signals')
                st.dataframe(show.sort_index(ascending=False).head(25), use_container_width=True)

        st.divider()
        st.subheader('Backtest All Watchlist')
        all_strategy = st.selectbox('Strategy for all stocks', list(STRATEGY_MAP.keys()), key='all_strategy')
        all_period = st.selectbox('History for all stocks', ['1y', '3y', '5y', '10y'], index=2, key='all_period')
        if st.button('Backtest All Watchlist', use_container_width=True):
            rows = []
            prog = st.progress(0)
            for i, sym in enumerate(symbols):
                try:
                    h = load_history(sym, all_period)
                    f = build_backtest_frame(h, rsi_buy, rsi_sell, near_level_pct)
                    summ, _ = backtest_summary(f, all_strategy)
                    if summ:
                        rows.append({'Ticker': sym, **summ})
                    else:
                        rows.append({'Ticker': sym, 'Signals': 0})
                except Exception as e:
                    rows.append({'Ticker': sym, 'Signals': 0, 'Error': str(e)})
                prog.progress((i + 1) / max(len(symbols), 1))
            prog.empty()
            st.session_state['all_bt'] = pd.DataFrame(rows)
            st.session_state['all_bt_key'] = (all_strategy, all_period, tuple(symbols))
        if st.session_state.get('all_bt_key') == (all_strategy, all_period, tuple(symbols)):
            all_df = st.session_state.get('all_bt', pd.DataFrame())
            if not all_df.empty:
                sort_col = 'Win Rate %' if 'Win Rate %' in all_df.columns else 'Signals'
                st.dataframe(all_df.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

if errors:
    with st.expander('Symbols with data issues'):
        for e in errors:
            st.write(e)

st.divider()
st.caption('Watchlist persistence uses the page URL query parameter. After adding/removing stocks, bookmark the current app URL. Backtests are historical and do not guarantee future results.')
