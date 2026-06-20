
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="📊 글로벌 주식 AI 예측 및 비교 대시보드 (오프라인 실습용)", layout="wide")

st.title("📈 글로벌 주식시장 AI 예측 및 비교 시뮬레이터 (Local Data Mode)")
st.info("💡 교내 방화벽 차단으로 인해 '오프라인 실습 모드(최신 시장 데이터 반영 가상 데이터셋)'로 자동 전환되었습니다.")

# 탭 구성 (개별 종목 예측 vs 시장 지수 비교)
tab1, tab2 = st.tabs(["🔍 개별 종목 AI 예측", "⚔️ 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교"])

# [오프라인용] 가상 데이터 생성기 (최신 상승장 트렌드 반영)
def generate_mock_stock_data(ticker, past_years=2):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=past_years * 365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # 영업일 기준
    
    np.random.seed(seed=abs(hash(ticker)) % 10000) # 종목별로 다른 패턴 생성
    
    # 최근 글로벌 상승세 트렌드를 반영한 가상 베이스 가격 및 우상향 기울기 설정
    if "005930" in ticker or "KS" in ticker: # 국내 주식 (강한 상승장)
        base_price = 70000
        trend = 45.0  # 일일 평균 상승 경향
        noise_level = 1500
    elif "^GSPC" in ticker: # S&P 500 지수 (안정적 우상향)
        base_price = 5000
        trend = 3.5
        noise_level = 60
    elif "^KS11" in ticker: # KOSPI 지수 (폭발적 우상향)
        base_price = 2800
        trend = 8.5
        noise_level = 70
    else: # 해외 우량주 (NVDA 등 폭발적 트렌드)
        base_price = 150
        trend = 0.4
        noise_level = 8
        
    # 랜덤 워크 + 우상향 트렌드로 주가 시뮬레이션
    days = len(date_range)
    time_index = np.arange(days)
    prices = base_price + (time_index * trend) + np.random.normal(0, noise_level, size=days)
    prices = np.clip(prices, a_min=1, a_max=None) # 음수 방지
    
    df = pd.DataFrame({'Date': date_range, 'Price': prices})
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
            # 방화벽 우회용 로컬 데이터 생성
            df = generate_mock_stock_data(ticker, past_years)
            
            if not df.empty:
                # AI 연산 (선형 회귀)
                df['Timestamp'] = df['Date'].map(datetime.toordinal)
                X = df[['Timestamp']].values
                y = df['Price'].values
                model = LinearRegression().fit(X, y)
                
                future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
                future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                future_preds = model.predict(future_X)
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds})
                
                # 차트 그리기
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='과거 주가 실적'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted'], mode='lines', name='AI 향후 예측', line=dict(dash='dash')))
                fig.update_layout(title=f"{selected_stock} ({ticker}) 주가 시뮬레이션 및 AI 트렌드 예측", xaxis_title="날짜", yaxis_title="주가", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# [TAB 2] 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교
# ====================================================================
with tab2:
    st.subheader("⚔️ 한국 KOSPI vs 미국 S&P 500 상승률 및 미래 추세 비교")
    st.write("올해 초(1월) 대비 몇 % 상승했는지를 기준으로 트렌드를 비교합니다.")
    
    compare_days = st.slider("AI 미래 예측 범위 설정 (일)", 10, 90, 30, step=5, key="tab2_days")
    
    try:
        # 방화벽 우회용 인덱스 가상 데이터 생성
        kospi_df = generate_mock_stock_data("^KS11", past_years=1)
        sp500_df = generate_mock_stock_data("^GSPC", past_years=1)
        
        # 2026년 1월 이후 데이터만 필터링하여 매칭
        this_year_start = datetime(datetime.today().year, 1, 1)
        kospi_df = kospi_df[kospi_df['Date'] >= this_year_start].reset_index(drop=True)
        sp500_df = sp500_df[sp500_df['Date'] >= this_year_start].reset_index(drop=True)
        
        if not kospi_df.empty and not sp500_df.empty:
            # 수익률(%) 환산 처리
            base_k = kospi_df['Price'].iloc[0]
            kospi_df['Return'] = ((kospi_df['Price'] - base_k) / base_k) * 100
            
            base_s = sp500_df['Price'].iloc[0]
            sp500_df['Return'] = ((sp500_df['Price'] - base_s) / base_s) * 100
            
            # AI 모델 학습 및 예측 함수
            def train_and_predict_return(df, days):
                df['Timestamp'] = df['Date'].map(datetime.toordinal)
                X = df[['Timestamp']].values
                y = df['Return'].values
                model = LinearRegression().fit(X, y)
                
                f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
                f_X = np.array([d.toordinal() for d in f_dates]).reshape(-1, 1)
                f_preds = model.predict(f_X)
                return model.coef_[0], pd.DataFrame({'Date': f_dates, 'Predicted_Return': f_preds})

            k_slope, k_future = train_and_predict_return(kospi_df, compare_days)
            s_slope, s_future = train_and_predict_return(sp500_df, compare_days)
            
            # 1. 메트릭 출력
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="🇰🇷 한국 KOSPI 올해 누적 수익률 (시뮬레이션)", 
                          value=f"{kospi_df['Return'].iloc[-1]:.2f}%", 
                          delta=f"AI 일일 성장 기울기: {k_slope:.4f}")
            with c2:
                st.metric(label="🇺🇸 미국 S&P 500 올해 누적 수익률 (시뮬레이션)", 
                          value=f"{sp500_df['Return'].iloc[-1]:.2f}%", 
                          delta=f"AI 일일 성장 기울기: {s_slope:.4f}")
            
            # 2. 분석 멘트 판정
            st.markdown("### 🏆 AI의 글로벌 트렌드 분석 판정")
            if k_slope > s_slope:
                st.success(f"현재 머신러닝 학습 결과, **한국 시장(KOSPI)**의 상승 추세선 기울기({k_slope:.3f})가 미국 시장({s_slope:.3f})보다 가파릅니다.")
            else:
                st.info(f"현재 머신러닝 학습 결과, **미국 시장(S&P 500)**의 상승 추세선 기울기({s_slope:.3f})가 한국 시장({k_slope:.3f})보다 가파릅니다.")
                
            # 3. Plotly 종합 차트 시각화
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=kospi_df['Date'], y=kospi_df['Return'], mode='lines', name='KOSPI 실적(%)', line=dict(color='#E53E3E', width=2)))
            fig_compare.add_trace(go.Scatter(x=k_future['Date'], y=k_future['Predicted_Return'], mode='lines', name='KOSPI AI 전망', line=dict(color='#E53E3E', width=2, dash='dot')))
            fig_compare.add_trace(go.Scatter(x=sp500_df['Date'], y=sp500_df['Return'], mode='lines', name='S&P 500 실적(%)', line=dict(color='#3182CE', width=2)))
            fig_compare.add_trace(go.Scatter(x=s_future['Date'], y=s_future['Predicted_Return'], mode='lines', name='S&P 500 AI 전망', line=dict(color='#3182CE', width=2, dash='dot')))
            
            fig_compare.update_layout(
                title="KOSPI vs S&P 500 올해 수익률 추이 및 AI 선형 트렌드 비교 (시뮬레이션 데이터)",
                xaxis_title="날짜", yaxis_title="누적 수익률 (%)", hovermode="x unified"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    except Exception as err:
        st.error(f"데이터 처리 중 오류 발생: {err}")
