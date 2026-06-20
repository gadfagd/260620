# ──────────────────────────────────────────────────────────────
# [기존 코드] 231번째 줄 근처
# ──────────────────────────────────────────────────────────────
    st.caption(
        f"학습 {fmt_learn(learn_r)} 데이터로 향후 {fmt_pred(pred_r)} 상승률을 추세 외삽으로 추정해 정렬합니다. "
        "(여러 종목을 빠르게 비교하기 위한 추세 스크리닝)"
    )
    if st.button("📊 상승 예측 실행", type="primary", use_container_width=True, key="rank_run"): # ← 이 부분!
        col_kr, col_us = st.columns(2)


# ──────────────────────────────────────────────────────────────
# [이렇게 수정하세요! (True -> False)]
# ──────────────────────────────────────────────────────────────
    st.caption(
        f"학습 {fmt_learn(learn_r)} 데이터로 향후 {fmt_pred(pred_r)} 상승률을 추세 외삽으로 추정해 정렬합니다. "
        "(여러 종목을 빠르게 비교하기 위한 추세 스크리닝)"
    )
    if st.button("📊 상승 예측 실행", type="primary", use_container_width=False, key="rank_run"): # ← False로 변경
        col_kr, col_us = st.columns(2)
