import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="📊 글로벌 주식 AI 예측 및 비교 대시보드", layout="wide")

st.title("📈 글로벌 주식시장 AI 예측 및 비교 시뮬레이터")
st.write("국내 시장과 미국 시장의 데이터를 분석하고 AI 트렌드 모델로 미래를 예측합니다.")

# 탭 구성 (개별 종목 예측 vs 시장 지수 비교)
tab1, tab2 = st.tabs(["🔍 개별 종목 AI 예측", "⚔️ 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교"])

# ====================================================================
# [TAB 1] 개별 종목 AI 예측 (기존 기능 유지 및 고도화)
# ====================================================================
with tab1:
    st.subheader("종목별 미래 주가 트렌드 예측")
    
    col_layout1, col_layout2 = st.columns([1, 3])
    
    with col_layout1:
        market = st.radio("🌐 시장 선택", ["국내 주식", "해외 주식 (US)"], key="tab1_market")
        
        if market == "국내 주식":
            stock_dict = {"삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "현대차": "005380.KS", "직접 입력": "CUSTOM"}
        else:
            stock_dict = {"애플": "AAPL", "마이크로소프트": "MSFT", "엔비디아": "NVDA", "직접 입력": "CUSTOM"}
            
        selected_stock = st.selectbox("📊 종목 선택", list(stock_dict.keys()), key="tab1_stock")
        ticker = st.text_input("🔤 티커 직접 입력", "").upper() if stock_dict[selected_stock] == "CUSTOM" else stock_dict[selected_stock]
        
        past_years = st.slider("AI 학습 데이터 기간 (년)", 1, 5, 2, key="tab1_years")
        forecast_days = st.slider("AI 미래 예측 기간 (일)", 5, 60, 30, step=5, key="tab1_days")

    with col_layout2:
        if ticker:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=past_years * 365)
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if not stock_data.empty:
                df = stock_data[['Close']].copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = df.reset_index()
                df.columns = ['Date', 'Close']
                
                # AI 연산 (선형 회귀)
                df['Timestamp'] = df['Date'].map(datetime.toordinal)
                X = df[['Timestamp']].values
                y = df['Close'].values
                model = LinearRegression().fit(X, y)
                
                future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
                future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                future_preds = model.predict(future_X)
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds})
                
                # 차트 그리기
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='과거 주가'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted'], mode='lines', name='AI 예측 트렌드', line=dict(dash='dash')))
                fig.update_layout(title=f"{ticker} 주가 분석 결과", xaxis_title="날짜", yaxis_title="주가", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# [TAB 2] 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교 (새로 추가된 핵심 비교 탭)
# ====================================================================
with tab2:
    st.subheader("⚔️ 한국 KOSPI vs 미국 S&P 500 상승률 및 미래 추세 비교")
    st.write("두 시장의 절대적인 지수 값이 다르므로, **'올해 초(1월) 대비 몇 % 상승했는지'**를 기준으로 AI 트렌드를 비교합니다.")
    
    compare_days = st.slider("AI 미래 예측 범위 설정 (일)", 10, 90, 30, step=5, key="tab2_days")
    
    with st.spinner('글로벌 지수 데이터를 분석하는 중입니다...'):
        # 올해 초부터 현재까지의 데이터 수집
        this_year_start = datetime(datetime.today().year, 1, 1)
        today = datetime.today()
        
        kospi_data = yf.download("^KS11", start=this_year_start, end=today) # 코스피 지수 티커
        sp500_data = yf.download("^GSPC", start=this_year_start, end=today) # S&P 500 지수 티커
        
        if not kospi_data.empty and not sp500_data.empty:
            # 데이터 정제 함수
            def clean_index_data(data):
                df = data[['Close']].copy()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = df.reset_index()
                df.columns = ['Date', 'Price']
                # 올해 첫 거래일 종가를 0% 기준으로 잡고 누적 상승률 계산
                base_price = df['Price'].iloc[0]
                df['Return'] = ((df['Price'] - base_price) / base_price) * 100
                return df

            kospi_df = clean_index_data(kospi_data)
            sp500_df = clean_index_data(sp500_data)
            
            # 각각 AI 선형 회귀 적용 (기울기 비교 목적)
            def train_and_predict(df, days):
                df['Timestamp'] = df['Date'].map(datetime.toordinal)
                X = df[['Timestamp']].values
                y = df['Return'].values
                model = LinearRegression().fit(X, y)
                
                # 미래 날짜 예측
                f_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
                f_X = np.array([d.toordinal() for d in f_dates]).reshape(-1, 1)
                f_preds = model.predict(f_X)
                return model.coef_[0], pd.DataFrame({'Date': f_dates, 'Predicted_Return': f_preds}) # 기울기와 예측 데이터 리턴

            k_slope, k_future = train_and_predict(kospi_df, compare_days)
            s_slope, s_future = train_and_predict(sp500_df, compare_days)
            
            # 대시보드 상단에 현재 성적 요약 표시
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="🇰🇷 한국 KOSPI 올해 누적 수익률", 
                          value=f"{kospi_df['Return'].iloc[-1]:.2f}%", 
                          delta=f"AI 일일 상승 탄력성 속도: {k_slope:.4f}")
            with c2:
                st.metric(label="🇺🇸 미국 S&P 500 올해 누적 수익률", 
                          value=f"{sp500_df['Return'].iloc[-1]:.2f}%", 
                          delta=f"AI 일일 상승 탄력성 속도: {s_slope:.4f}")
            
            # 승자 판정 브리핑
            st.markdown("### 🏆 AI의 글로벌 트렌드 분석 판정")
            if k_slope > s_slope:
                st.success(f"현재 데이터 학습 결과, **한국 시장(KOSPI)**의 상승 기울기(${k_slope:.3f}$)가 미국 시장(${s_slope:.3f}$)보다 가파릅니다. 당분간 국내 주식의 상승 모멘텀이 상대적으로 더 강할 것으로 예측됩니다.")
            else:
                st.info(f"현재 데이터 학습 결과, **미국 시장(S&P 500)**의 상승 기울기(${s_slope:.3f}$)가 한국 시장(${k_slope:.3f}$)보다 안정적으로 가파릅니다. 글로벌 자금의 미국 쏠림 현상이 유지될 확률이 높습니다.")
                
            # 두 시장 비교 시각화 차트
            fig_compare = go.Figure()
            
            # KOSPI 과거 및 미래
            fig_compare.add_trace(go.Scatter(x=kospi_df['Date'], y=kospi_df['Return'], mode='lines', name='KOSPI 실적(%)', line=dict(color='#E53E3E')))
            fig_compare.add_trace(go.Scatter(x=k_future['Date'], y=k_future['Predicted_Return'], mode='lines', name='KOSPI AI 전망', line=dict(color='#E53E3E', dash='dot')))
            
            # S&P 500 과거 및 미래
            fig_compare.add_trace(go.Scatter(x=sp500_df['Date'], y=sp500_df['Return'], mode='lines', name='S&P 500 실적(%)', line=dict(color='#3182CE')))
            fig_compare.add_trace(go.Scatter(x=s_future['Date'], y=s_future['Predicted_Return'], mode='lines', name='S&P 500 AI 전망', line=dict(color='#3182CE', dash='dot')))
            
            fig_compare.update_layout(title="KOSPI vs S&P 500 올해 누적 상승률(%) 및 AI 트렌드 확장선 비교",
                                      xaxis_title="날짜", yaxis_title="누적 수익률 (%)", hovermode="x unified")
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
        else:
            st.error("지수 데이터를 가져오는 데 실패했습니다.")
