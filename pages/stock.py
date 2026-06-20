"""
AI 주식 예측 프로그램 (Streamlit)
====================================
- 국내(KOSPI/KOSDAQ) / 해외(미국) 주식 선택
- 야후 파이낸스(yfinance)에서 시세 데이터 수집
- Prophet 또는 선형회귀로 미래 가격 예측
- Plotly로 과거 데이터 + 예측 구간 시각화

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

# Prophet은 선택 사항: 설치되어 있으면 사용, 없으면 선형회귀만 제공
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# 1. 종목 사전 (이름 → 야후 파이낸스 티커)
# ──────────────────────────────────────────────────────────────
# 국내 주식: KOSPI는 ".KS", KOSDAQ는 ".KQ" 접미사
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
    "에코프로비엠(코스닥)": "247540.KQ",
    "에코프로(코스닥)": "086520.KQ",
    "알테오젠(코스닥)": "196170.KQ",
}

# 해외(미국) 주식
US_STOCKS = {
    "애플 (AAPL)": "AAPL",
    "마이크로소프트 (MSFT)": "MSFT",
    "엔비디아 (NVDA)": "NVDA",
    "알파벳/구글 (GOOGL)": "GOOGL",
    "아마존 (AMZN)": "AMZN",
    "테슬라 (TSLA)": "TSLA",
    "메타 (META)": "META",
    "넷플릭스 (NFLX)": "NFLX",
    "S&P500 ETF (SPY)": "SPY",
    "나스닥100 ETF (QQQ)": "QQQ",
}


# ──────────────────────────────────────────────────────────────
# 2. 데이터 수집 함수
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period_years: int) -> pd.DataFrame:
    """야후 파이낸스에서 일별 시세를 가져온다."""
    end = datetime.now()
    start = end - timedelta(days=int(period_years * 365))
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance가 MultiIndex 컬럼을 반환하는 경우 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


# ──────────────────────────────────────────────────────────────
# 3. 기술적 지표
# ──────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ──────────────────────────────────────────────────────────────
# 4. 예측 모델
# ──────────────────────────────────────────────────────────────
def predict_prophet(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Prophet으로 예측. ds/yhat/yhat_lower/yhat_upper 반환."""
    pdf = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).dropna()
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,  # 추세 변화 민감도
    )
    model.fit(pdf)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def predict_linear(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """최근 추세를 1차 선형회귀로 외삽한다(간단 모델)."""
    d = df[["Date", "Close"]].dropna().reset_index(drop=True)
    t = np.arange(len(d))
    slope, intercept = np.polyfit(t, d["Close"].values, 1)

    # 잔차 표준편차로 대략적인 신뢰구간 추정
    resid = d["Close"].values - (slope * t + intercept)
    std = resid.std()

    future_t = np.arange(len(d), len(d) + periods)
    last_date = d["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=periods)

    all_t = np.concatenate([t, future_t])
    all_dates = pd.concat([d["Date"], pd.Series(future_dates)], ignore_index=True)
    yhat = slope * all_t + intercept

    return pd.DataFrame({
        "ds": all_dates,
        "yhat": yhat,
        "yhat_lower": yhat - 1.96 * std,
        "yhat_upper": yhat + 1.96 * std,
    })


# ──────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI 주식 예측", page_icon="📈", layout="wide")

st.title("📈 AI 주식 예측 프로그램")
st.caption("야후 파이낸스 데이터를 활용해 국내·해외 주식의 미래 흐름을 예측합니다.")

with st.sidebar:
    st.header("⚙️ 설정")

    market = st.radio("시장 선택", ["국내 주식", "해외 주식"], horizontal=True)

    if market == "국내 주식":
        stock_dict = KR_STOCKS
        currency = "원"
    else:
        stock_dict = US_STOCKS
        currency = "$"

    stock_name = st.selectbox("종목 선택", list(stock_dict.keys()))
    ticker = stock_dict[stock_name]

    # 직접 티커 입력 옵션
    use_custom = st.checkbox("티커 직접 입력")
    if use_custom:
        ticker = st.text_input(
            "야후 파이낸스 티커",
            value=ticker,
            help="예) 국내: 005930.KS / 005930.KQ,  해외: AAPL, TSLA",
        ).strip()

    st.divider()

    learn_years = st.slider("학습 기간 (년)", 1, 10, 3)
    predict_days = st.slider("예측 기간 (영업일)", 5, 120, 30)

    model_options = ["선형회귀 (간단·빠름)"]
    if PROPHET_AVAILABLE:
        model_options.insert(0, "Prophet (추세+계절성)")
    model_choice = st.radio("예측 모델", model_options)

    run = st.button("🔮 예측 실행", type="primary", use_container_width=True)


# 면책 조항 (항상 표시)
st.info(
    "⚠️ **투자 유의 안내** · 이 프로그램의 예측은 과거 데이터에 기반한 통계적 추정일 뿐, "
    "미래 주가를 보장하지 않습니다. 주가는 예측 불가능한 수많은 변수의 영향을 받습니다. "
    "교육·참고용이며 투자 권유나 자문이 아닙니다. 실제 투자 판단과 책임은 본인에게 있습니다."
)


if run:
    with st.spinner("데이터를 불러오는 중..."):
        df = load_data(ticker, learn_years)

    if df.empty:
        st.error(
            f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요.\n\n"
            "예) 국내는 `005930.KS`(코스피)·`247540.KQ`(코스닥), 해외는 `AAPL` 형식입니다."
        )
        st.stop()

    df = add_indicators(df)

    # 현재가 / 등락 요약
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    change = last_close - prev_close
    change_pct = change / prev_close * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("현재가(최근 종가)", f"{last_close:,.2f} {currency}",
              f"{change:+,.2f} ({change_pct:+.2f}%)")
    c2.metric("학습 데이터 기간", f"{df['Date'].iloc[0]:%Y-%m-%d} ~ {df['Date'].iloc[-1]:%Y-%m-%d}")
    rsi_val = df["RSI"].iloc[-1]
    rsi_state = "과매수" if rsi_val > 70 else ("과매도" if rsi_val < 30 else "중립")
    c3.metric("RSI(14)", f"{rsi_val:.1f}", rsi_state)

    # 예측 수행
    with st.spinner("예측 모델을 학습하는 중..."):
        if model_choice.startswith("Prophet"):
            forecast = predict_prophet(df, predict_days)
        else:
            forecast = predict_linear(df, predict_days)

    # 미래 구간만 추출
    future_part = forecast[forecast["ds"] > df["Date"].iloc[-1]]
    pred_price = float(future_part["yhat"].iloc[-1])
    pred_change_pct = (pred_price - last_close) / last_close * 100

    st.subheader(f"🎯 {predict_days}영업일 후 예측")
    p1, p2, p3 = st.columns(3)
    p1.metric("예측가", f"{pred_price:,.2f} {currency}", f"{pred_change_pct:+.2f}%")
    p2.metric("예측 하한", f"{float(future_part['yhat_lower'].iloc[-1]):,.2f} {currency}")
    p3.metric("예측 상한", f"{float(future_part['yhat_upper'].iloc[-1]):,.2f} {currency}")

    trend = "📈 상승" if pred_change_pct > 1 else ("📉 하락" if pred_change_pct < -1 else "➡️ 보합")
    st.markdown(f"### 종합 전망: **{trend}** (예측 변동률 {pred_change_pct:+.2f}%)")

    # ── 차트 ──
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("주가 & 예측", "거래량"),
    )

    # 과거 종가
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], name="실제 종가",
        line=dict(color="#1f77b4", width=1.5)), row=1, col=1)

    # 이동평균선
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20",
                             line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA60"], name="MA60",
                             line=dict(color="green", width=1, dash="dot")), row=1, col=1)

    # 예측선
    fig.add_trace(go.Scatter(
        x=future_part["ds"], y=future_part["yhat"], name="예측",
        line=dict(color="red", width=2)), row=1, col=1)

    # 신뢰구간
    fig.add_trace(go.Scatter(
        x=pd.concat([future_part["ds"], future_part["ds"][::-1]]),
        y=pd.concat([future_part["yhat_upper"], future_part["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,0,0,0.12)",
        line=dict(color="rgba(255,0,0,0)"), name="예측 구간",
        hoverinfo="skip"), row=1, col=1)

    # 거래량
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="거래량",
                         marker_color="rgba(100,100,100,0.4)"), row=2, col=1)

    fig.update_layout(height=650, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # 예측 데이터 표
    with st.expander("📋 예측 데이터 보기"):
        show = future_part.copy()
        show.columns = ["날짜", "예측가", "하한", "상한"]
        show["날짜"] = show["날짜"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            show.round(2).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "💡 예측 모델 한계: 선형회귀는 단순 추세 외삽, Prophet은 추세·계절성 기반입니다. "
        "두 모델 모두 갑작스러운 시장 충격·뉴스·실적 변동은 반영하지 못합니다."
    )
else:
    st.markdown(
        "👈 왼쪽에서 **시장과 종목을 선택**하고 **예측 실행** 버튼을 눌러주세요.\n\n"
        "- **국내 주식**: 삼성전자, SK하이닉스 등 (코스피/코스닥)\n"
        "- **해외 주식**: 애플, 엔비디아, 테슬라 등 (미국)\n"
        "- 원하는 종목이 없으면 **티커 직접 입력**으로 추가할 수 있어요."
    )
