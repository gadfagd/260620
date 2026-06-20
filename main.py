# -*- coding: utf-8 -*-
"""
🌸 두근두근 지진 안전 탐험대 🌸
-------------------------------------------------
USGS(미국지질조사국) 실시간 지진 데이터 API를 활용하여
관심 지역의 지진 발생 횟수 / 규모 / 통계 기반 위험도를
귀엽고 직관적으로 보여주는 Streamlit 앱입니다.

⚠️ 교육적 안내
지진은 현재 과학기술로 "정확한 예측(언제·어디서·얼마나)"이
불가능합니다. 이 앱은 미래를 예언하는 것이 아니라
"과거 데이터를 보면 이 지역은 지진이 얼마나 자주/세게 일어났는가"를
통계적으로 보여주는 '위험도 참고 자료'입니다.
이 점을 학생들에게 꼭 짚어주세요!

실행 방법:
    pip install streamlit requests pandas plotly pydeck
    streamlit run earthquake_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta

# =========================================================
# 0. 페이지 기본 설정 + 귀여운 CSS 테마
# =========================================================
st.set_page_config(
    page_title="🌸 두근두근 지진 안전 탐험대",
    page_icon="🐰",
    layout="wide",
)

CUTE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&family=Jua&display=swap');

html, body, [class*="css"]  {
    font-family: 'Gaegu', sans-serif;
}

.main {
    background: linear-gradient(180deg, #FFF0F5 0%, #FDF6FF 50%, #F0F8FF 100%);
}

h1, h2, h3 {
    font-family: 'Jua', sans-serif !important;
    color: #FF6FA5 !important;
}

div[data-testid="stMetric"] {
    background: #ffffffcc;
    border: 3px solid #FFC1DA;
    border-radius: 20px;
    padding: 14px;
    box-shadow: 4px 4px 0px #FFD6E8;
    text-align: center;
}

div[data-testid="stMetricLabel"] {
    font-family: 'Jua', sans-serif !important;
    color: #FF8FB3 !important;
}

.cute-box {
    background: #ffffffd9;
    border: 3px dashed #FFB6D9;
    border-radius: 22px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 4px 4px 0px #FFE3EE;
}

.warn-box {
    background: #FFF7E0;
    border: 3px solid #FFD27A;
    border-radius: 22px;
    padding: 16px 20px;
    margin-bottom: 14px;
}

.safe-box {
    background: #E9FFF0;
    border: 3px solid #8CE6A8;
    border-radius: 22px;
    padding: 16px 20px;
}

.danger-box {
    background: #FFE6E6;
    border: 3px solid #FF9C9C;
    border-radius: 22px;
    padding: 16px 20px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFE3F1 0%, #E9F0FF 100%);
}

.stButton>button {
    background-color: #FF8FB3;
    color: white;
    border-radius: 20px;
    border: none;
    font-family: 'Jua', sans-serif;
    padding: 8px 22px;
}

.stButton>button:hover {
    background-color: #FF6FA5;
    color: white;
}
</style>
"""
st.markdown(CUTE_CSS, unsafe_allow_html=True)

# =========================================================
# 1. 여행지(또는 관심 지역) 프리셋 — 학생들이 헷갈리지 않게 좌표 내장
# =========================================================
PRESET_LOCATIONS = {
    "🗼 일본 - 도쿄": (35.6762, 139.6503),
    "🦌 일본 - 오사카": (34.6937, 135.5023),
    "🌊 일본 - 오키나와": (26.2124, 127.6809),
    "♨️ 일본 - 후쿠오카": (33.5904, 130.4017),
    "🗻 일본 - 삿포로": (43.0618, 141.3545),
    "🇰🇷 대한민국 - 서울": (37.5665, 126.9780),
    "🇰🇷 대한민국 - 경주": (35.8562, 129.2247),
    "🇺🇸 미국 - 캘리포니아(LA)": (34.0522, -118.2437),
    "🇮🇩 인도네시아 - 자카르타": (-6.2088, 106.8456),
    "🇵🇭 필리핀 - 마닐라": (14.5995, 120.9842),
    "🇹🇷 튀르키예 - 이스탄불": (41.0082, 28.9784),
    "✏️ 직접 좌표 입력": None,
}

# =========================================================
# 2. USGS API 데이터 불러오기
# =========================================================
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_earthquakes(lat, lon, radius_km, start_date, end_date, min_mag):
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "orderby": "time",
    }
    res = requests.get(USGS_URL, params=params, timeout=20)
    res.raise_for_status()
    data = res.json()

    rows = []
    for feature in data.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]  # [lon, lat, depth]
        rows.append({
            "발생시각": pd.to_datetime(props["time"], unit="ms", utc=True).tz_convert("Asia/Seoul"),
            "규모": props["mag"],
            "위치설명": props["place"],
            "깊이(km)": coords[2],
            "위도": coords[1],
            "경도": coords[0],
            "쓰나미경보": "⚠️ 있음" if props.get("tsunami") == 1 else "없음",
        })
    df = pd.DataFrame(rows)
    return df


# =========================================================
# 3. 통계 기반 "위험도 참고" 점수 계산 (예언이 아닌 통계!)
# =========================================================
def calc_risk_message(df, days):
    if df.empty:
        return "safe", "이 기간엔 기록된 규모 이상 지진이 없었어요! 🌷 비교적 평온한 데이터예요.", 0

    count = len(df)
    avg_mag = df["규모"].mean()
    max_mag = df["규모"].max()
    per_month = count / max(days / 30, 1)

    score = 0
    score += min(per_month * 2, 40)          # 빈도 점수
    score += min(max_mag * 8, 40)            # 최대 규모 점수
    score += min(avg_mag * 4, 20)            # 평균 규모 점수
    score = round(min(score, 100))

    if score < 30:
        level = "safe"
        msg = f"최근 데이터상 지진 활동이 비교적 적은 편이에요 🐰🌸 (참고용 점수: {score}/100)"
    elif score < 65:
        level = "warn"
        msg = f"가끔 지진이 발생하는 지역이에요! 기본 안전수칙은 알아두면 좋아요 🦊⚡ (참고용 점수: {score}/100)"
    else:
        level = "danger"
        msg = f"지진 활동이 비교적 활발한 지역이에요. 여행 시 대피 요령을 꼭 숙지하세요! 🐯🔥 (참고용 점수: {score}/100)"

    return level, msg, score


# =========================================================
# 4. 헤더
# =========================================================
st.markdown("# 🌸 두근두근 지진 안전 탐험대 🐰")
st.markdown(
    "<div class='cute-box'>가고 싶은 여행지를 골라서 <b>실제 지진 데이터</b>를 확인해보고,"
    " 안전하게 떠날 준비를 해볼까요? 🧳✈️</div>",
    unsafe_allow_html=True,
)

# =========================================================
# 5. 사이드바 — 사용자 입력
# =========================================================
with st.sidebar:
    st.markdown("## 🎀 검색 조건 설정")

    loc_name = st.selectbox("📍 궁금한 지역을 골라주세요", list(PRESET_LOCATIONS.keys()))

    if PRESET_LOCATIONS[loc_name] is None:
        lat = st.number_input("위도(latitude)", value=35.68, format="%.4f")
        lon = st.number_input("경도(longitude)", value=139.65, format="%.4f")
    else:
        lat, lon = PRESET_LOCATIONS[loc_name]
        st.caption(f"위도 {lat}, 경도 {lon}")

    radius_km = st.slider("🔍 반경 (km)", min_value=50, max_value=1000, value=300, step=50)

    period = st.radio(
        "🗓️ 조회 기간",
        ["최근 1개월", "최근 3개월", "최근 1년", "최근 5년"],
        index=2,
    )
    period_days = {"최근 1개월": 30, "최근 3개월": 90, "최근 1년": 365, "최근 5년": 365 * 5}[period]

    min_mag = st.slider("📏 최소 규모(Magnitude)", 0.0, 7.0, 2.5, 0.1)

    search = st.button("🔎 지진 데이터 검색하기")

st.markdown(
    "<div class='warn-box'>📢 <b>꼭 알아두세요!</b> 지진은 현재 과학기술로 정확히 "
    "'언제·어디서·얼마나'를 예측할 수 없어요. 아래 결과는 <b>과거 데이터를 바탕으로 한 "
    "통계적 참고자료</b>이지, 미래를 알려주는 점쟁이가 아니에요! 🙅‍♀️🔮</div>",
    unsafe_allow_html=True,
)

# =========================================================
# 6. 데이터 조회 & 표시
# =========================================================
if search or "last_query" in st.session_state:
    st.session_state["last_query"] = True

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=period_days)

    with st.spinner("지진 데이터를 불러오는 중이에요... 🐰💨"):
        try:
            df = fetch_earthquakes(lat, lon, radius_km, start_date, end_date, min_mag)
        except Exception as e:
            st.error(f"데이터를 불러오지 못했어요 😢 ({e})")
            st.stop()

    st.markdown(f"## 📍 {loc_name} 결과 ({period}, 반경 {radius_km}km, 규모 {min_mag} 이상)")

    # ---- 요약 지표 ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 지진 발생 횟수", f"{len(df)}건")
    col2.metric("📏 평균 규모", f"{df['규모'].mean():.2f}" if not df.empty else "0")
    col3.metric("🔥 최대 규모", f"{df['규모'].max():.2f}" if not df.empty else "0")
    col4.metric("🌊 쓰나미경보 발생", f"{(df['쓰나미경보']=='⚠️ 있음').sum() if not df.empty else 0}건")

    # ---- 위험도 참고 메시지 ----
    level, msg, score = calc_risk_message(df, period_days)
    box_class = {"safe": "safe-box", "warn": "warn-box", "danger": "danger-box"}[level]
    st.markdown(f"<div class='{box_class}'><h3 style='margin:0;'>💌 위험도 참고 카드</h3>{msg}</div>",
                unsafe_allow_html=True)
    st.progress(score / 100)

    if not df.empty:
        # ---- 지도 ----
        st.markdown("### 🗺️ 지진 발생 위치 지도")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[경도, 위도]',
            get_radius="규모 * 8000",
            get_fill_color='[255, 111, 165, 160]',
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=4, pitch=0)
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={"text": "{위치설명}\n규모: {규모}"},
        ))

        # ---- 규모 분포 히스토그램 ----
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📊 규모(magnitude) 분포")
            fig_hist = px.histogram(
                df, x="규모", nbins=15,
                color_discrete_sequence=["#FF8FB3"],
            )
            fig_hist.update_layout(
                plot_bgcolor="#FFF6FA", paper_bgcolor="#FFF6FA",
                font_family="Gaegu", bargap=0.1,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown("### 📈 시간에 따른 발생 추이")
            df_trend = df.set_index("발생시각").resample("D").size().reset_index(name="건수")
            fig_trend = px.line(
                df_trend, x="발생시각", y="건수",
                markers=True, color_discrete_sequence=["#A0C4FF"],
            )
            fig_trend.update_layout(
                plot_bgcolor="#F0F8FF", paper_bgcolor="#F0F8FF",
                font_family="Gaegu",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # ---- 상세 표 ----
        with st.expander("📋 지진 상세 기록 보기"):
            show_df = df.sort_values("발생시각", ascending=False).copy()
            show_df["발생시각"] = show_df["발생시각"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                show_df[["발생시각", "위치설명", "규모", "깊이(km)", "쓰나미경보"]],
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("선택한 조건에 맞는 지진 기록이 없어요! 안심하고 여행 준비를 해도 좋겠네요 🌸✈️")

    # ---- 교육용 마무리 안내 ----
    st.markdown(
        "<div class='cute-box'>🐰 <b>안전 Tip!</b> 지진이 잦은 지역으로 여행할 땐 "
        "숙소의 비상구 위치를 미리 확인하고, '몸 낮추고 - 머리 보호하고 - "
        "움직이지 않기(Drop, Cover, Hold on)' 요령을 기억해두세요! 🦺</div>",
        unsafe_allow_html=True,
    )
    st.caption("데이터 출처: USGS Earthquake Hazards Program (실시간 공개 API)")

else:
    st.markdown(
        "<div class='cute-box'>왼쪽 사이드바에서 지역과 기간을 고르고 "
        "<b>'지진 데이터 검색하기'</b> 버튼을 눌러주세요! 🌷</div>",
        unsafe_allow_html=True,
    )
