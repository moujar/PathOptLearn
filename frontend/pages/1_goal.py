"""Page 1 — Goal Elicitation & Content Harvesting."""
import streamlit as st
import requests

st.set_page_config(page_title="Goal — AdaptLearn AI", layout="wide")
st.title("🎯 Stage 1: Define Your Learning Goal")

API = st.session_state.get("api_base", "http://localhost:8000/api/v1")
HEADERS = {"Authorization": f"Bearer {st.session_state.get('api_key', 'dev-secret')}"}


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API}{endpoint}", json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ── Goal Input ────────────────────────────────────────────────────────────
st.subheader("What do you want to learn?")

examples = [
    "I want to learn Machine Learning from scratch",
    "I want to understand Neural Networks and Deep Learning",
    "I want to learn Python for Data Science",
    "I want to master Natural Language Processing",
]

col1, col2 = st.columns([3, 1])
with col1:
    raw_goal = st.text_area(
        "Enter your learning goal",
        placeholder="e.g. I want to learn Machine Learning from scratch",
        height=100,
    )
with col2:
    st.write("**Quick examples:**")
    for ex in examples:
        if st.button(ex[:40] + "…", key=ex):
            raw_goal = ex

student_id = st.session_state.get("student_id", "student_001")

if st.button("🚀 Parse Goal", type="primary", disabled=not raw_goal):
    with st.spinner("Parsing your goal with GPT-4…"):
        result = api_post("/goal/parse", {"user_id": student_id, "raw_goal": raw_goal})

    if result:
        st.session_state.goal_id = result["goal_id"]
        st.session_state.goal_data = result
        st.success(f"✅ Goal parsed! ID: `{result['goal_id']}`")

# ── Display Goal ──────────────────────────────────────────────────────────
if st.session_state.get("goal_data"):
    goal = st.session_state.goal_data
    st.divider()
    st.subheader(f"📚 Topic: **{goal['topic']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Difficulty", goal["difficulty_hint"].capitalize())
        st.write("**Sub-topics (learning order):**")
        for i, sub in enumerate(goal["sub_topics"], 1):
            st.write(f"  {i}. {sub}")

    with col2:
        st.write("**KG Query Terms:**")
        tags_html = " ".join(f'<span style="background:#1f77b4;color:white;padding:3px 8px;border-radius:12px;margin:2px;font-size:12px">{t}</span>' for t in goal["kg_query_terms"])
        st.markdown(tags_html, unsafe_allow_html=True)

    with st.expander("📄 Raw API Response"):
        st.json(goal)

    st.divider()
    st.subheader("📥 Content Harvesting")
    st.write("Harvest web pages and YouTube videos for each sub-topic.")

    if st.button("🌐 Harvest Content", type="secondary"):
        with st.spinner("Fetching content from Tavily + YouTube… (this may take ~30s)"):
            result = api_post("/goal/harvest", {"goal_id": goal["goal_id"]})

        if result:
            n_docs = result.get("n_docs", 0)
            st.success(f"✅ Harvested {n_docs} documents!")

            docs = result.get("docs", [])
            web_docs = [d for d in docs if d.get("source_type") == "web"]
            yt_docs = [d for d in docs if d.get("source_type") == "youtube"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Web Articles", len(web_docs))
            with col2:
                st.metric("YouTube Videos", len(yt_docs))

            with st.expander("📋 Harvested Documents"):
                for doc in docs[:10]:
                    icon = "🎬" if doc["source_type"] == "youtube" else "📰"
                    st.write(f"{icon} [{doc['title']}]({doc['url']})")

    st.info("➡️ Next: Go to **Stage 2 — Quiz Generation** to generate your adaptive quiz.")
