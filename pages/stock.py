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

# [오프라인 전용] 오류 가능성 0%의 안전한 가상 데이터 생성기
def generate_safe_mock_data(stock_name, past_years=2):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=past_years * 365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # 영업일 기준
    
    # 각 종목 글자 자체를 시드로 활용해 항상 고유한 무작위 데이터 생성
    np.random.seed(seed=abs(hash(stock_name)) % 10000)
    days = len(date_range)
    time_index = np.arange(days)
    
    # 선택된 한글 이름에 맞추어 스케일 및 우상향 기조(2026년 상승세) 설정
    if stock_name == "삼성전자":
        base, trend, noise = 72000, 42.0, 1500
    elif stock_name == "SK하이닉스":
        base, trend, noise = 180000, 115.0, 4000
    elif stock_name == "현대차":
        base, trend, noise = 240000, 95.0, 4500
    elif stock_name == "애플":
        base, trend, noise = 175, 0.08, 3
    elif stock_name == "마이크로소프트":
        base, trend, noise = 410, 0.19, 5
    elif stock_name == "엔비디아":
        base, trend, noise = 850, 0.88, 15
    elif stock_name == "KOSPI":
        base, trend, noise = 2750, 8.8, 55
    elif stock_name == "S&P 500":
        base, trend, noise = 5100, 3.4, 45
    else:
        base, trend, noise = 100, 0.2, 5
        
    # 선형 추세선 모델 기반으로 노이즈 데이터 합성
    prices = base + (time_index * trend) + np.random.normal(0, noise, size=days)
    prices = np.clip(prices, a_min=1, a_max=None)
    
    # 데이터프레임 구조 강제 지정
    df = pd.DataFrame()
    df['Date'] = date_range
    df['Price'] = prices.astype(float)
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
            stock_list = ["삼성전자", "SK하이닉스", "현대차"]
            unit = "원"
        else:
            stock_list = ["애플", "마이크로소프트", "엔비디아"]
            unit = "$"
            
        selected_stock = st.selectbox("📊 종목 선택", stock_list, key="tab1_stock")
        
        past_years = st.slider("AI 학습 데이터 기간 (년)", 1, 5, 2, key="tab1_years")
        forecast_days = st.slider("AI 미래 예측 기간 (일)", 5, 60, 30, step=5, key="tab1_days")

    with col_layout2:
        if selected_stock:
            # 안전하게 데이터 생성
            df = generate_safe_mock_data(selected_stock, past_years)
            
            # AI 선형 회귀 연산
            df['Timestamp'] = df['Date'].map(datetime.toordinal)
            X = df[['Timestamp']].values
            y = df['Price'].values
            
            model = LinearRegression().fit(X, y)
            
            # 미래 시점 예측
            future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            future_preds = model.predict(future_X)
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds})
            
            # 시각화 그래프 빌드
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='과거 주가 실적'))
            fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted'], mode='lines', name='AI 향후 예측', line=dict(dash='dash', color='orange')))
            fig.update_layout(title=f"{selected_stock} 주가 트렌드 예측 모델 결과", xaxis_title="날짜", yaxis_title=f"주가 ({unit})", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# [TAB 2] 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교
# ====================================================================
with tab2:
