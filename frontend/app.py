"""AdaptLearn AI — Streamlit main application."""
import streamlit as st

st.set_page_config(
    page_title="AdaptLearn AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────
DEFAULTS = {
    "student_id": "",
    "goal_id": "",
    "kg_id": "",
    "path_id": "",
    "quiz_id": "",
    "goal_data": None,
    "quiz_data": None,
    "knowledge_vector": None,
    "learning_path": None,
    "api_base": "http://localhost:8000/api/v1",
    "api_key": "dev-secret",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 AdaptLearn AI")
    st.caption("Adaptive Learning Path Recommendation")
    st.divider()

    # Student ID
    student_id = st.text_input(
        "Student ID", value=st.session_state.student_id or "student_001", key="sidebar_student_id"
    )
    st.session_state.student_id = student_id

    # API config
    with st.expander("⚙️ API Settings"):
        st.session_state.api_base = st.text_input(
            "API Base URL", value=st.session_state.api_base
        )
        st.session_state.api_key = st.text_input(
            "API Key", value=st.session_state.api_key, type="password"
        )

    st.divider()

    # Progress overview
    st.subheader("Pipeline Progress")
    steps = [
        ("1. Goal", bool(st.session_state.goal_id)),
        ("2. Quiz", bool(st.session_state.quiz_id)),
        ("3. Assessment", bool(st.session_state.knowledge_vector)),
        ("4. Knowledge Graph", bool(st.session_state.kg_id)),
        ("5. Learning Path", bool(st.session_state.path_id)),
    ]
    for label, done in steps:
        icon = "✅" if done else "⏳"
        st.write(f"{icon} {label}")

    st.divider()
    st.caption("Master's Thesis — AdaptLearn AI v1.0")

# ── Main content ──────────────────────────────────────────────────────────
st.title("AdaptLearn AI")
st.subheader("Adaptive Learning Path Recommendation System")

st.markdown(
    """
    Welcome to **AdaptLearn AI** — a research-grade adaptive learning system that:

    1. **Parses** your learning goal into a structured topic tree
    2. **Generates** an adaptive quiz calibrated to your level
    3. **Assesses** your knowledge state using IRT-CAT
    4. **Builds** a dynamic knowledge graph of the domain
    5. **Recommends** an optimal learning path using DRL-PPO ★
    6. **Adapts** the path continuously as you learn ★

    ---
    👈 Use the sidebar to navigate between stages, or use the page navigation below.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🎯 Stage 1-2**\nSet your goal and take the adaptive quiz")
with col2:
    st.info("**📊 Stage 3-4**\nView your knowledge state and knowledge graph")
with col3:
    st.info("**🚀 Stage 5-6**\nGet your personalised path and track progress")

# Show current session state for thesis demo
if any([st.session_state.goal_id, st.session_state.quiz_id]):
    with st.expander("🔍 Session State (Thesis Demo)"):
        st.json(
            {
                "student_id": st.session_state.student_id,
                "goal_id": st.session_state.goal_id,
                "quiz_id": st.session_state.quiz_id,
                "kg_id": st.session_state.kg_id,
                "path_id": st.session_state.path_id,
            }
        )
