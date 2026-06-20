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

# 공통 데이터 정제 함수 (MultiIndex 완벽 파괴 및 단일 1차원 데이터 변환)
def safe_clean_data(raw_data, price_col='Close'):
    if raw_data.empty:
        return pd.DataFrame()
    
    df = raw_data.copy()
    
    # 1. 컬럼이 MultiIndex(다중 구조)라면 상위 레이어를 쳐내고 단일 구조로 변환
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. 인덱스에 Date가 들어가 있다면 컬럼으로 빼내기
    df = df.reset_index()
    
    # 3. 필요한 날짜와 가격 컬럼만 필터링하여 일차원 표 생성
    if 'Date' in df.columns and price_col in df.columns:
        clean_df = df[['Date', price_col]].copy()
        clean_df.columns = ['Date', 'Price']
        clean_df['Price'] = clean_df['Price'].astype(float)
        # 만약 시계열 데이터에 시간대 정보(TZ)가 붙어있으면 순수 날짜형태로 제거
        clean_df['Date'] = pd.to_datetime(clean_df['Date']).dt.tz_localize(None)
        return clean_df
    else:
        return pd.DataFrame()

# ====================================================================
# [TAB 1] 개별 종목 AI 예측
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
            
            try:
                raw_stock = yf.download(ticker, start=start_date, end=end_date)
                df = safe_clean_data(raw_stock, 'Close')
                
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
                    fig.update_layout(title=f"{ticker} 주가 분석 및 AI 트렌드 예측", xaxis_title="날짜", yaxis_title="주가", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("선택한 종목의 주가 데이터를 분석할 수 없습니다. 티커명을 확인해 주세요.")
            except Exception as e:
                st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")

# ====================================================================
# [TAB 2] 국장(KOSPI) vs 미장(S&P 500) AI 트렌드 비교
# ====================================================================
with tab2:
    st.subheader("⚔️ 한국 KOSPI vs 미국 S&P 500 상승률 및 미래 추세 비교")
    st.write("올해 초(1월) 대비 몇 % 상승했는지를 기준으로 트렌드를 비교합니다.")
    
    compare_days = st.slider("AI 미래 예측 범위 설정 (일)", 10, 90, 30, step=5, key="tab2_days")
    
    with st.spinner('야후 파이낸스에서 글로벌 지수 데이터를 가져오는 중입니다...'):
        this_year_start = datetime(datetime.today().year, 1, 1)
        today = datetime.today()
        
        try:
            raw_kospi = yf.download("^KS11", start=this_year_start, end=today)
            raw_sp500 = yf.download("^GSPC", start=this_year_start, end=today)
            
            kospi_df = safe_clean_data(raw_kospi, 'Close')
            sp500_df = safe_clean_data(raw_sp500, 'Close')
            
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
                    st.metric(label="🇰🇷 한국 KOSPI 올해 누적 수익률", 
                              value=f"{kospi_df['Return'].iloc[-1]:.2f}%", 
                              delta=f"AI 일일 성장 기울기: {k_slope:.4f}")
                with c2:
                    st.metric(label="🇺🇸 미국 S&P 500 올해 누적 수익률", 
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
                    title="KOSPI vs S&P 500 올해 수익률 추이 및 AI 선형 트렌드 비교",
                    xaxis_title="날짜", yaxis_title="누적 수익률 (%)", hovermode="x unified"
                )
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.error("지수 데이터를 가져오지 못했습니다.")
        except Exception as err:
            st.error(f"데이터 처리 중 일시적인 오류가 발생했습니다: {err}")
