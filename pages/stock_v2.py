"""
AI 주식 예측 프로그램 (Streamlit)
====================================
[탭1] 상승 예측 랭킹 : 국내/미국 주요 종목을 한 번에 분석해 상승 가능성 순으로 정렬
[탭2] 종목 상세 분석 : 한 종목을 골라 과거 + 예측 곡선 시각화
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import importlib.util
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


# ──────────────────────────────────────────────────────────────
# 종목 사전
# ──────────────────────────────────────────────────────────────
KR_STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "POSCO홀딩스": "005490.KS",
    "셀트리온": "068270.KS",
    "KB금융": "105560.KS",
    "삼성SDI": "006400.KS",
    "에코프로비엠": "247540.KQ",
    "에코프로": "086520.KQ",
    "알테오젠": "196170.KQ",
}

US_STOCKS = {
    "애플 (AAPL)": "AAPL",
    "마이크로소프트 (MSFT)": "MSFT",
    "엔비디아 (NVDA)": "NVDA",
    "알파벳 (GOOGL)": "GOOGL",
    "아마존 (AMZN)": "AMZN",
    "테슬라 (TSLA)": "TSLA",
    "메타 (META)": "META",
    "넷플릭스 (NFLX)": "NFLX",
    "AMD (AMD)": "AMD",
    "팔란티어 (PLTR)": "PLTR",
    "브로드컴 (AVGO)": "AVGO",
    "로켓랩 (RKLB)": "RKLB",
    "블룸에너지 (BE)": "BE",
    "S&P500 ETF (SPY)": "SPY",
    "나스닥100 ETF (QQQ)": "QQQ",
}

BDAYS_PER_MONTH = 21
CDAYS_PER_MONTH = 30.4


# ──────────────────────────────────────────────────────────────
# 데이터 수집 및 지표 계산
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, learn_months: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(learn_months * CDAYS_PER_MONTH))
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ──────────────────────────────────────────────────────────────
# 예측 모델
# ──────────────────────────────────────────────────────────────
def predict_linear(df: pd.DataFrame, periods_bdays: int) -> pd.DataFrame:
    d = df[["Date", "Close"]].dropna().reset_index(drop=True)
    t = np.arange(len(d))
    slope, _ = np.polyfit(t, d["Close"].values, 1)
    resid = d["Close"].values - (slope * t + (d["Close"].values - slope * t).mean())
    std = resid.std()
    last_date = d["Date"].iloc[-1]
    last_close = d["Close"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=periods_bdays)
    steps = np.arange(1, periods_bdays + 1)
    yhat_future = last_close + slope * steps
    hist = pd.DataFrame({"ds": d["Date"], "yhat": d["Close"], "yhat_lower": d["Close"], "yhat_upper": d["Close"]})
    fut = pd.DataFrame({"ds": future_dates, "yhat": yhat_future, "yhat_lower": yhat_future - 1.96 * std, "yhat_upper": yhat_future + 1.96 * std})
    return pd.concat([hist, fut], ignore_index=True)


def predict_lstm(df: pd.DataFrame, periods_bdays: int, lookback: int = 30, epochs: int = 25):
    if not TF_AVAILABLE:
        return None
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    d = df[["Date", "Close"]].dropna().reset_index(drop=True)
    close = d["Close"].values.astype("float32")
    if len(close) < lookback + 40:
        return None
    tf.random.set_seed(42)
    np.random.seed(42)
    cmin, cmax = float(close.min()), float(close.max())
    rng = (cmax - cmin) or 1.0
    scaled = (close - cmin) / rng
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])
    X = np.array(X, dtype="float32").reshape(-1, lookback, 1)
    y = np.array(y, dtype="float32")
    model = Sequential()
    model.add(LSTM(40, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    pred_train = model.predict(X, verbose=0).flatten()
    std = float(((y - pred_train) * rng).std())
    window = scaled[-lookback:].tolist()
    preds = []
    for _ in range(periods_bdays):
        x = np.array(window[-lookback:], dtype="float32").reshape(1, lookback, 1)
        p = float(model(x, training=False).numpy().flatten()[0])
        preds.append(p)
        window.append(p)
    preds = np.array(preds) * rng + cmin
    last_date = d["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=periods_bdays)
    steps = np.arange(1, periods_bdays + 1)
    band = 1.96 * std * np.sqrt(steps)
    hist = pd.DataFrame({"ds": d["Date"], "yhat": d["Close"], "yhat_lower": d["Close"], "yhat_upper": d["Close"]})
    fut = pd.DataFrame({"ds": future_dates, "yhat": preds, "yhat_lower": preds - band, "yhat_upper": preds + band})
    return pd.concat([hist, fut], ignore_index=True)


def quick_forecast_pct(df: pd.DataFrame, horizon_bdays: int):
    d = df["Close"].dropna().values
    if len(d) < 20:
        return None
    t = np.arange(len(d))
    slope, _ = np.polyfit(t, d, 1)
    last = float(d[-1])
    pred = last + slope * horizon_bdays
    return (pred - last) / last * 100, last


# ──────────────────────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI 주식 예측", page_icon="📈", layout="wide")
st.title("📈 AI 주식 예측 프로그램")
st.caption("야후 파이낸스 데이터로 국내·해외 주식의 미래 흐름을 예측합니다.")
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] p { font-size: 0.8rem; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

LEARN_OPTIONS = [6, 12, 24, 36, 48, 60]
PRED_OPTIONS = list(range(1, 13))

def fmt_learn(m): return f"{m // 12}년" if m % 12 == 0 else f"{m}개월"
def fmt_pred(m): return f"{m}개월" if m < 12 else "1년"


# ──────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 종목 상세 분석 설정")

    # ── 1) 드롭다운 종목 선택 ──
    st.markdown("**시장**")
    market = st.radio("시장", ["국내 주식", "해외 주식"], horizontal=True, label_visibility="collapsed")
    stock_dict = KR_STOCKS if market == "국내 주식" else US_STOCKS
    detail_currency = "원" if market == "국내 주식" else "$"

    st.markdown("**종목**")
    stock_name = st.selectbox("종목", list(stock_dict.keys()), label_visibility="collapsed")
    # 드롭다운 선택값을 기본 ticker로 설정
    ticker = stock_dict[stock_name]

    # ── 2) 티커 직접 입력 ──
    if st.checkbox("티커 직접 입력"):
        ticker = st.text_input("야후 파이낸스 티커", value=ticker).strip()

    # ── 3) 종목 검색 (session_state로 ticker 관리) ──
    st.divider()
    st.markdown("#### 🔍 종목 검색")
    st.caption("목록에 없는 종목을 검색해서 선택하세요.")

    search_market = st.radio(
        "검색 시장",
        ["국내", "해외"],
        horizontal=True,
        key="search_market",
    )
    query = st.text_input(
        "종목명 또는 티커",
        placeholder="예: 플러그파워, Plug Power, PLUG",
        label_visibility="collapsed",
        key="search_query",
    )

    if query.strip():
        mkt_filter = "KR" if search_market == "국내" else "US"
        with st.spinner("검색 중…"):
            try:
                quotes = yf.Search(query, max_results=10).quotes
            except Exception:
                quotes = []

        results = []
        for q in quotes:
            tk   = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or tk
            if q.get("quoteType") not in ("EQUITY", "ETF"):
                continue
            if mkt_filter == "KR" and not (tk.endswith(".KS") or tk.endswith(".KQ")):
                continue
            if mkt_filter == "US" and (tk.endswith(".KS") or tk.endswith(".KQ")):
                continue
            results.append({"name": name, "ticker": tk})

        if results:
            for r in results:
                if st.button(
                    f"{r['name']}  `{r['ticker']}`",
                    key=f"srch_{r['ticker']}",
                    use_container_width=True,
                ):
                    st.session_state["search_selected_ticker"] = r["ticker"]
                    st.session_state["search_selected_name"]   = r["name"]
        else:
            st.warning("검색 결과 없음. 티커를 직접 입력해 보세요. (예: PLUG)")

    # 검색으로 선택된 종목이 있으면 ticker 덮어쓰기
    if "search_selected_ticker" in st.session_state:
        ticker = st.session_state["search_selected_ticker"]
        name   = st.session_state["search_selected_name"]
        st.success(f"✅ 선택됨: **{name}** `{ticker}`")
        if st.button("❌ 선택 해제", use_container_width=True):
            del st.session_state["search_selected_ticker"]
            del st.session_state["search_selected_name"]
            st.rerun()


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
st.info(
    "⚠️ **투자 유의 안내** · 예측은 과거 데이터에 기반한 통계적 추정일 뿐 미래 주가를 보장하지 않습니다. "
    "교육·참고용이며 투자 권유가 아닙니다."
)

tab_rank, tab_detail = st.tabs(["🚀 상승 예측 랭킹", "📈 종목 상세 분석"])


# ── 탭1: 랭킹 ────────────────────────────────────────────────
def run_ranking(stocks, horizon_bdays, lm):
    rows = []
    bar = st.progress(0.0)
    items = list(stocks.items())
    for i, (name, tk) in enumerate(items):
        df = load_data(tk, lm)
        if not df.empty:
            res = quick_forecast_pct(df, horizon_bdays)
            if res is not None:
                pct, last = res
                rows.append({"종목": name, "현재가": last, "예측 상승률(%)": pct})
        bar.progress((i + 1) / len(items))
    bar.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("예측 상승률(%)", ascending=False).reset_index(drop=True)


def render_ranking(rdf, currency):
    if rdf.empty:
        st.warning("데이터를 가져오지 못했습니다.")
        return
    dec = 2 if currency == "$" else 0
    top = rdf.iloc[0]
    st.metric(f"🥇 {top['종목']}", f"{top['예측 상승률(%)']:+.1f}%", f"현재가 {top['현재가']:,.{dec}f} {currency}")
    rising = rdf[rdf["예측 상승률(%)"] > 0]
    if not rising.empty:
        items = [f"**{r['종목']}** ({r['예측 상승률(%)']:+.1f}%)" for _, r in rising.head(3).iterrows()]
        st.info(f"🔮 **상승 우세:** {' · '.join(items)}")
    else:
        st.warning("📉 모든 종목이 하락 추세입니다.")
    colors = ["#2E9E6B" if v >= 0 else "#E24B4A" for v in rdf["예측 상승률(%)"]]
    fig = go.Figure(go.Bar(
        x=rdf["예측 상승률(%)"], y=rdf["종목"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in rdf["예측 상승률(%)"]], textposition="outside",
    ))
    fig.update_layout(height=max(320, 34 * len(rdf)), margin=dict(l=10, r=60, t=10, b=10), xaxis_title="예측 상승률(%)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    show = rdf.copy()
    show["현재가"] = show["현재가"].map(lambda x: f"{x:,.{dec}f}")
    show["예측 상승률(%)"] = show["예측 상승률(%)"].map(lambda x: f"{x:+.1f}")
    st.dataframe(show, use_container_width=True, hide_index=True)


with tab_rank:
    st.subheader("🚀 국내·미국 주식 상승 예측")
    rc1, rc2 = st.columns(2)
    learn_r = rc1.select_slider("학습 기간", options=LEARN_OPTIONS, value=24, format_func=fmt_learn, key="learn_rank")
    pred_r  = rc2.select_slider("예측 기간", options=PRED_OPTIONS, value=3, format_func=fmt_pred, key="pred_rank")
    pred_bdays_r = int(pred_r * BDAYS_PER_MONTH)
    st.caption(f"학습 {fmt_learn(learn_r)} 데이터로 향후 {fmt_pred(pred_r)} 상승률을 추세 외삽으로 추정해 정렬합니다.")
    if st.button("📊 상승 예측 실행", type="primary", use_container_width=False, key="rank_run"):
        col_kr, col_us = st.columns(2)
        with col_kr:
            st.markdown("### 🇰🇷 국내 주식")
            with st.spinner("국내 종목 분석 중..."):
                kr = run_ranking(KR_STOCKS, pred_bdays_r, learn_r)
            render_ranking(kr, "원")
        with col_us:
            st.markdown("### 🇺🇸 미국 주식")
            with st.spinner("미국 종목 분석 중..."):
                us = run_ranking(US_STOCKS, pred_bdays_r, learn_r)
            render_ranking(us, "$")
    else:
        st.markdown("👆 **상승 예측 실행** 버튼을 누르면 국내·미국 주요 종목을 한 번에 분석합니다.")


# ── 탭2: 상세 분석 ───────────────────────────────────────────
with tab_detail:
    # 현재 선택된 종목명 표시 (검색 선택 시 이름 반영)
    display_name = st.session_state.get("search_selected_name", stock_name)
    st.subheader(f"📈 {display_name} 상세 분석")

    dc1, dc2 = st.columns(2)
    learn_d = dc1.select_slider("학습 기간", options=LEARN_OPTIONS, value=24, format_func=fmt_learn, key="learn_detail")
    pred_d  = dc2.select_slider("예측 기간", options=PRED_OPTIONS, value=3, format_func=fmt_pred, key="pred_detail")
    pred_bdays_d = int(pred_d * BDAYS_PER_MONTH)

    if st.button("🔮 예측 실행", type="primary", key="detail_run"):
        with st.spinner("데이터를 불러오는 중..."):
            df = load_data(ticker, learn_d)

        if df.empty:
            st.error(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요.")
            st.stop()

        df = add_indicators(df)
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        change = last_close - prev_close
        change_pct = change / prev_close * 100
        last_date = df["Date"].iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("현재가(최근 종가)", f"{last_close:,.2f} {detail_currency}", f"{change:+,.2f} ({change_pct:+.2f}%)")
        c2.metric("학습 데이터", f"{df['Date'].iloc[0]:%Y-%m-%d} ~ {df['Date'].iloc[-1]:%Y-%m-%d}")
        rsi_val = df["RSI"].iloc[-1]
        rsi_state = "과매수" if rsi_val > 70 else ("과매도" if rsi_val < 30 else "중립")
        c3.metric("RSI(14)", f"{rsi_val:.1f}", rsi_state)

        with st.spinner("선형회귀 예측 중..."):
            fc_lin = predict_linear(df, pred_bdays_d)
        lin_fut = fc_lin[fc_lin["ds"] > last_date]

        fc_lstm = None
        if TF_AVAILABLE:
            with st.spinner("LSTM(딥러닝) 예측 중..."):
                fc_lstm = predict_lstm(df, pred_bdays_d)
        lstm_fut = fc_lstm[fc_lstm["ds"] > last_date] if fc_lstm is not None else None

        def summarize(fut):
            p = float(fut["yhat"].iloc[-1])
            return p, (p - last_close) / last_close * 100

        lin_price, lin_pct = summarize(lin_fut)

        st.subheader(f"🎯 {fmt_pred(pred_d)} 후 예측 · 모델 비교")

        def card_html(title, price, pct, accent):
            chg = "#2E9E6B" if pct >= 0 else "#E24B4A"
            return (
                f"<div style='flex:1; min-width:200px; border:1px solid #e6e6e6; border-radius:10px; padding:0.7rem 1rem;'>"
                f"<div style='font-size:0.85rem; font-weight:600; color:{accent};'>{title}</div>"
                f"<div style='font-size:1.3rem; font-weight:600; margin-top:0.2rem;'>{price:,.2f} {detail_currency}</div>"
                f"<div style='font-size:0.9rem; color:{chg};'>{pct:+.2f}%</div></div>"
            )

        cards = [card_html("📏 선형회귀", lin_price, lin_pct, "#C0392B")]
        if lstm_fut is not None:
            lstm_price, lstm_pct = summarize(lstm_fut)
            cards.append(card_html("🧠 LSTM (딥러닝)", lstm_price, lstm_pct, "#6C3FC5"))
        st.markdown(
            "<div style='display:flex; gap:1rem; flex-wrap:wrap; margin:0.3rem 0 0.6rem;'>" + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="실제 종가", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20", line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lin_fut["ds"], y=lin_fut["yhat"], name="선형회귀", line=dict(color="#E24B4A", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([lin_fut["ds"], lin_fut["ds"][::-1]]),
            y=pd.concat([lin_fut["yhat_upper"], lin_fut["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(226,75,74,0.10)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        if lstm_fut is not None:
            fig.add_trace(go.Scatter(x=lstm_fut["ds"], y=lstm_fut["yhat"], name="LSTM", line=dict(color="#7B5BD6", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pd.concat([lstm_fut["ds"], lstm_fut["ds"][::-1]]),
                y=pd.concat([lstm_fut["yhat_upper"], lstm_fut["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(123,91,214,0.10)", line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="거래량", marker_color="rgba(100,100,100,0.4)"), row=2, col=1)
        fig.update_layout(height=680, hovermode="x unified", margin=dict(t=70, l=10, r=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)
