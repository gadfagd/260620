"""
AI 주식 예측 프로그램 (Streamlit)
====================================
[탭1] 상승 예측 랭킹 : 국내/미국 주요 종목을 한 번에 분석해 상승 가능성 순으로 정렬
[탭2] 종목 상세 분석 : 한 종목을 골라 과거 + 예측 곡선 시각화

- 야후 파이낸스(yfinance)에서 시세 수집
- 학습 기간 6개월 ~ 5년 / 예측 기간 1개월 ~ 1년
- Prophet 또는 선형회귀로 예측

실행 방법:
    pip install -r requirements.txt
    streamlit run stock_predictor.py
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


# ──────────────────────────────────────────────────────────────
# 종목 사전 (이름 → 야후 파이낸스 티커)
# 국내: KOSPI ".KS", KOSDAQ ".KQ"
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
    "블룸에너지 (BE)": "BE",
    "스페이스X (SPCX)": "SPCX",
    "S&P500 ETF (SPY)": "SPY",
    "나스닥100 ETF (QQQ)": "QQQ",
}

BDAYS_PER_MONTH = 21       # 한 달 ≈ 영업일 21일
CDAYS_PER_MONTH = 30.4     # 한 달 ≈ 달력일 30.4일


# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, learn_months: int) -> pd.DataFrame:
    """야후 파이낸스에서 일별 시세를 가져온다 (학습 기간: 개월)."""
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


# ──────────────────────────────────────────────────────────────
# 기술적 지표
# ──────────────────────────────────────────────────────────────
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
def predict_prophet(df: pd.DataFrame, periods_days: int) -> pd.DataFrame:
    pdf = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).dropna()
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_range=0.9,          # 최근 추세 변화도 반영
        changepoint_prior_scale=0.1,
    )
    model.fit(pdf)
    future = model.make_future_dataframe(periods=periods_days)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def predict_linear(df: pd.DataFrame, periods_bdays: int) -> pd.DataFrame:
    """선형회귀 추세 외삽. 예측이 '현재가'에서 출발하도록 보정."""
    d = df[["Date", "Close"]].dropna().reset_index(drop=True)
    t = np.arange(len(d))
    slope, _ = np.polyfit(t, d["Close"].values, 1)

    resid = d["Close"].values - (slope * t + (d["Close"].values - slope * t).mean())
    std = resid.std()

    last_date = d["Date"].iloc[-1]
    last_close = d["Close"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=periods_bdays)
    steps = np.arange(1, periods_bdays + 1)
    yhat_future = last_close + slope * steps   # 현재가 앵커링

    # 과거 구간은 실제값으로 채워 자연스럽게 이어지게 함
    hist = pd.DataFrame({
        "ds": d["Date"], "yhat": d["Close"],
        "yhat_lower": d["Close"], "yhat_upper": d["Close"],
    })
    fut = pd.DataFrame({
        "ds": future_dates, "yhat": yhat_future,
        "yhat_lower": yhat_future - 1.96 * std,
        "yhat_upper": yhat_future + 1.96 * std,
    })
    return pd.concat([hist, fut], ignore_index=True)


def quick_forecast_pct(df: pd.DataFrame, horizon_bdays: int):
    """랭킹용 빠른 추세 예측: (예측 상승률%, 현재가) 반환."""
    d = df["Close"].dropna().values
    if len(d) < 20:
        return None
    t = np.arange(len(d))
    slope, _ = np.polyfit(t, d, 1)
    last = float(d[-1])
    pred = last + slope * horizon_bdays   # 현재가에서 추세 연장
    return (pred - last) / last * 100, last


# ──────────────────────────────────────────────────────────────
# UI 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI 주식 예측", page_icon="📈", layout="wide")

st.title("📈 AI 주식 예측 프로그램")
st.caption("야후 파이낸스 데이터로 국내·해외 주식의 미래 흐름을 예측합니다.")

# 탭 라벨 / 지표 폰트 크기 조정
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] p { font-size: 0.8rem; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📊 종목 상세 분석 설정")
    market = st.radio("시장", ["국내 주식", "해외 주식"], horizontal=True)
    stock_dict = KR_STOCKS if market == "국내 주식" else US_STOCKS
    detail_currency = "원" if market == "국내 주식" else "$"
    stock_name = st.selectbox("종목", list(stock_dict.keys()))
    ticker = stock_dict[stock_name]
    if st.checkbox("티커 직접 입력"):
        ticker = st.text_input("야후 파이낸스 티커", value=ticker,
                               help="예) 005930.KS, 247540.KQ, AAPL").strip()

    st.divider()
    st.subheader("⚙️ 설정")

    learn_options = [6, 12, 24, 36, 48, 60]
    learn_months = st.select_slider(
        "학습 기간",
        options=learn_options,
        value=24,
        format_func=lambda m: f"{m // 12}년" if m % 12 == 0 else f"{m}개월",
    )

    pred_months = st.select_slider(
        "예측 기간",
        options=list(range(1, 13)),
        value=3,
        format_func=lambda m: f"{m}개월" if m < 12 else "1년",
    )

    model_options = ["선형회귀 (간단·빠름)"]
    if PROPHET_AVAILABLE:
        model_options.insert(0, "Prophet (추세+계절성)")
    model_choice = st.radio("예측 모델 (상세 분석 탭)", model_options)

pred_bdays = int(pred_months * BDAYS_PER_MONTH)
pred_cdays = int(pred_months * CDAYS_PER_MONTH)

st.info(
    "⚠️ **투자 유의 안내** · 예측은 과거 데이터에 기반한 통계적 추정일 뿐 미래 주가를 보장하지 않습니다. "
    "특히 추세를 그대로 연장하는 방식이라 단기 급등의 되돌림이나 시장 충격은 반영하지 못합니다. "
    "교육·참고용이며 투자 권유가 아닙니다."
)

tab_rank, tab_detail = st.tabs(["🚀 상승 예측 랭킹", "📈 종목 상세 분석"])


# ──────────────────────────────────────────────────────────────
# 탭1: 상승 예측 랭킹
# ──────────────────────────────────────────────────────────────
def run_ranking(stocks: dict, horizon_bdays: int, lm: int) -> pd.DataFrame:
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


def render_ranking(rdf: pd.DataFrame, currency: str):
    if rdf.empty:
        st.warning("데이터를 가져오지 못했습니다.")
        return
    dec = 2 if currency == "$" else 0
    top = rdf.iloc[0]
    st.metric(f"🥇 {top['종목']}", f"{top['예측 상승률(%)']:+.1f}%",
              f"현재가 {top['현재가']:,.{dec}f} {currency}")

    colors = ["#2E9E6B" if v >= 0 else "#E24B4A" for v in rdf["예측 상승률(%)"]]
    fig = go.Figure(go.Bar(
        x=rdf["예측 상승률(%)"], y=rdf["종목"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in rdf["예측 상승률(%)"]], textposition="outside",
    ))
    fig.update_layout(height=max(320, 34 * len(rdf)),
                      margin=dict(l=10, r=60, t=10, b=10),
                      xaxis_title="예측 상승률(%)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    show = rdf.copy()
    show["현재가"] = show["현재가"].map(lambda x: f"{x:,.{dec}f}")
    show["예측 상승률(%)"] = show["예측 상승률(%)"].map(lambda x: f"{x:+.1f}")
    st.dataframe(show, use_container_width=True, hide_index=True)


with tab_rank:
    st.subheader("🚀 국내·미국 주식 상승 예측")
    st.caption(
        f"학습 {learn_months}개월 데이터를 바탕으로 향후 **{pred_months}개월** 상승률을 추세 외삽으로 추정해 정렬합니다. "
        "(여러 종목을 빠르게 비교하기 위한 추세 스크리닝)"
    )
    if st.button("📊 상승 예측 실행", type="primary", use_container_width=True):
        col_kr, col_us = st.columns(2)
        with col_kr:
            st.markdown("### 🇰🇷 국내 주식")
            with st.spinner("국내 종목 분석 중..."):
                kr = run_ranking(KR_STOCKS, pred_bdays, learn_months)
            render_ranking(kr, "원")
        with col_us:
            st.markdown("### 🇺🇸 미국 주식")
            with st.spinner("미국 종목 분석 중..."):
                us = run_ranking(US_STOCKS, pred_bdays, learn_months)
            render_ranking(us, "$")
    else:
        st.markdown("👆 **상승 예측 실행** 버튼을 누르면 국내·미국 주요 종목을 한 번에 분석합니다.")


# ──────────────────────────────────────────────────────────────
# 탭2: 종목 상세 분석
# ──────────────────────────────────────────────────────────────
with tab_detail:
    st.subheader(f"📈 {stock_name} 상세 분석")
    if st.button("🔮 예측 실행", type="primary", key="detail_run"):
        with st.spinner("데이터를 불러오는 중..."):
            df = load_data(ticker, learn_months)

        if df.empty:
            st.error(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요.")
            st.stop()

        df = add_indicators(df)
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        change = last_close - prev_close
        change_pct = change / prev_close * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("현재가(최근 종가)", f"{last_close:,.2f} {detail_currency}",
                  f"{change:+,.2f} ({change_pct:+.2f}%)")
        c2.metric("학습 데이터",
                  f"{df['Date'].iloc[0]:%Y-%m-%d} ~ {df['Date'].iloc[-1]:%Y-%m-%d}")
        rsi_val = df["RSI"].iloc[-1]
        rsi_state = "과매수" if rsi_val > 70 else ("과매도" if rsi_val < 30 else "중립")
        c3.metric("RSI(14)", f"{rsi_val:.1f}", rsi_state)

        with st.spinner("예측 모델을 학습하는 중..."):
            if model_choice.startswith("Prophet"):
                forecast = predict_prophet(df, pred_cdays)
            else:
                forecast = predict_linear(df, pred_bdays)

        future_part = forecast[forecast["ds"] > df["Date"].iloc[-1]]
        pred_price = float(future_part["yhat"].iloc[-1])
        pred_change_pct = (pred_price - last_close) / last_close * 100

        label = f"{pred_months}개월" if pred_months < 12 else "1년"
        st.subheader(f"🎯 {label} 후 예측")

        lower = float(future_part["yhat_lower"].iloc[-1])
        upper = float(future_part["yhat_upper"].iloc[-1])
        chg_color = "#2E9E6B" if pred_change_pct >= 0 else "#E24B4A"
        st.markdown(
            f"""
            <div style="display:flex; gap:2.5rem; flex-wrap:wrap; margin:0.2rem 0 0.6rem;">
              <div>
                <div style="font-size:0.8rem; color:gray;">예측가</div>
                <div style="font-size:1.15rem; font-weight:600;">{pred_price:,.2f} {detail_currency}</div>
                <div style="font-size:0.85rem; color:{chg_color};">{pred_change_pct:+.2f}%</div>
              </div>
              <div>
                <div style="font-size:0.8rem; color:gray;">예측 하한</div>
                <div style="font-size:1.15rem; font-weight:600;">{lower:,.2f} {detail_currency}</div>
              </div>
              <div>
                <div style="font-size:0.8rem; color:gray;">예측 상한</div>
                <div style="font-size:1.15rem; font-weight:600;">{upper:,.2f} {detail_currency}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        trend = "📈 상승" if pred_change_pct > 1 else ("📉 하락" if pred_change_pct < -1 else "➡️ 보합")
        st.markdown(
            f"<div style='font-size:1.05rem; font-weight:600; margin:0.2rem 0 0.6rem;'>"
            f"종합 전망: {trend} (예측 변동률 {pred_change_pct:+.2f}%)</div>",
            unsafe_allow_html=True,
        )

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05,
                            subplot_titles=("주가 & 예측", "거래량"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="실제 종가",
                                 line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20",
                                 line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA60"], name="MA60",
                                 line=dict(color="green", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=future_part["ds"], y=future_part["yhat"], name="예측",
                                 line=dict(color="red", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([future_part["ds"], future_part["ds"][::-1]]),
            y=pd.concat([future_part["yhat_upper"], future_part["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,0,0,0.12)",
            line=dict(color="rgba(255,0,0,0)"), name="예측 구간", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="거래량",
                             marker_color="rgba(100,100,100,0.4)"), row=2, col=1)
        fig.update_layout(height=650, hovermode="x unified",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 예측 데이터 보기"):
            show = future_part.copy()
            show.columns = ["날짜", "예측가", "하한", "상한"]
            show["날짜"] = show["날짜"].dt.strftime("%Y-%m-%d")
            st.dataframe(show.round(2).reset_index(drop=True),
                         use_container_width=True, hide_index=True)
    else:
        st.markdown("👈 왼쪽 사이드바에서 종목을 고르고 **예측 실행** 버튼을 눌러주세요.")
  
