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

import importlib.util
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


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
    "로켓랩 (RKLB)": "RKLB",
    "블룸에너지 (BE)": "BE",
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


def predict_lstm(df: pd.DataFrame, periods_bdays: int,
                 lookback: int = 30, epochs: int = 25):
    """LSTM(딥러닝) 시계열 예측. 데이터 부족 시 None 반환."""
    if not TF_AVAILABLE:
        return None

    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras
