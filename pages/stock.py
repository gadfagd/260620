import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="📊 글로벌 주식 AI 예측 및 비교 대시보드", layout="wide")

st.title("📈 글로벌 주식시장 AI 예측 및 비교 시뮬레이터")
st.info("💡 교내 방화벽 우회 모드: 인터넷 연결 없이도 머신러닝 시뮬레이션이 완벽하게 작동합니다.")

# 탭 구성 (개별 종목 예측 vs 시장 지수 비교)
tab1, tab2 = st.tabs(["🔍 개별 종목 AI 예측", "⚔️ 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교"])

# [오프라인 전용] 초간단 가상 데이터 생성기
def generate_clean_mock_data(ticker, past_years=2):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=past_years * 365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # 평일 기준 날짜 생성
    
    # 종목 고유의 무작위 성 주기
    np.random.seed(seed=abs(hash(ticker)) % 10000)
    days = len(date_range)
    time_index = np.arange(days)
    
    # 2026년 강세장 트렌드를 반영한 기본값 세팅
    if ticker == "005930.KS":    # 삼성전자
        base, trend, noise = 72000, 42.0, 1200
    elif ticker == "000660.KS":  # SK하이닉스
        base, trend, noise = 180000, 110.0, 3500
    elif ticker == "005380.KS":  # 현대차
        base, trend, noise = 240000, 95.0, 4000
    elif ticker == "AAPL":       # 애플
        base, trend, noise = 175, 0.08, 3
    elif ticker == "MSFT":       # 마이크로소프트
        base, trend, noise = 410, 0.18, 5
    elif ticker == "NVDA":       # 엔비디아
        base, trend, noise = 850, 0.85, 15
    elif ticker == "^KS11":      # KOSPI 지수 (폭발적 급등 반영)
        base, trend, noise = 2750, 8.5, 50
    elif ticker == "^GSPC":      # S&P 500 지수
        base, trend, noise = 5100, 3.2, 45
    else:
        base, trend, noise = 100, 0.1, 5
        
    # 데이터 모델링: 선형 추세 + 랜덤 가우시안 노이즈
    prices = base + (time_index * trend) + np.random.normal(0, noise, size=days)
    prices = np.clip(prices, a_min=1, a_max=None)
    
    # 오류 가능성이 없는 깨끗한 단일 데이터프레임 구조 생성
    df = pd.DataFrame({
        'Date': date_range,
        'Price': prices.astype(float)
    })
    return df

# ====================================================================
# [TAB 1] 개별 종목 AI 예측
# ====================================================================
with tab1:
    st.subheader("종목별 미래 주가 트렌드 예측")
    col_layout1, col_layout2 = st.columns([1, 3])
    
    with col_layout1:
        market = st.radio("🌐 시장 선택", ["국내 주식", "해외 주식 (US)"], key="tab1_market")
        
        if market == "국내 주식":
            stock_dict = {"삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "현대차": "005380.KS"}
        else:
            stock_dict = {"애플": "AAPL", "마이크로소프트": "MSFT", "엔비디아": "NVDA"}
            
        selected_stock = st.selectbox("📊 종목 선택", list(stock_dict.keys()), key="tab1_stock")
        ticker = stock_dict[selected_stock]
        
        past_years = st.slider("AI 학습 데이터 기간 (년)", 1, 5, 2, key="tab1_years")
        forecast_days = st.slider("AI 미래 예측 기간 (일)", 5, 60, 30, step=5, key="tab1_days")

    with col_layout2:
        if ticker:
            # 완벽히 정제
