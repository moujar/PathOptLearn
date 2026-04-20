import streamlit as st

st.set_page_config(
    page_title="Adaptive Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AUTH GUARD ───────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.switch_page("pages/1_Login.py")
else:
    st.switch_page("pages/2_Dashboard.py")
