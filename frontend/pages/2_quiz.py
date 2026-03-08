"""Page 2 — Adaptive Quiz Generation & Administration."""
import streamlit as st
import requests

st.set_page_config(page_title="Quiz — AdaptLearn AI", layout="wide")
st.title("📝 Stage 2: Adaptive Quiz")

API = st.session_state.get("api_base", "http://localhost:8000/api/v1")
HEADERS = {"Authorization": f"Bearer {st.session_state.get('api_key', 'dev-secret')}"}

BLOOM_COLORS = {
    "remember": "#3498db",
    "understand": "#2ecc71",
    "apply": "#f39c12",
    "analyze": "#e74c3c",
}


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API}{endpoint}", json=payload, headers=HEADERS, timeout=90)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


if not st.session_state.get("goal_id"):
    st.warning("⚠️ Please complete Stage 1 (Goal) first.")
    st.stop()

# ── Quiz Generation ───────────────────────────────────────────────────────
st.subheader("Quiz Configuration")
col1, col2 = st.columns(2)
with col1:
    n_questions = st.slider("Total Questions", 4, 20, 12)
with col2:
    st.write("**Bloom's Taxonomy Distribution:**")
    bloom_remember = st.number_input("Remember", 0, 5, 2)
    bloom_understand = st.number_input("Understand", 0, 5, 4)
    bloom_apply = st.number_input("Apply", 0, 5, 4)
    bloom_analyze = st.number_input("Analyze", 0, 5, 2)

if st.button("⚡ Generate Quiz", type="primary"):
    with st.spinner("Generating adaptive quiz with GPT-4 + IRT calibration…"):
        result = api_post(
            "/quiz/generate",
            {
                "goal_id": st.session_state.goal_id,
                "n_questions": n_questions,
                "bloom_distribution": {
                    "remember": bloom_remember,
                    "understand": bloom_understand,
                    "apply": bloom_apply,
                    "analyze": bloom_analyze,
                },
            },
        )

    if result:
        st.session_state.quiz_id = result["quiz_id"]
        st.session_state.quiz_data = result
        st.success(f"✅ Quiz generated! {len(result['items'])} items (estimated {result['estimated_duration_min']} min)")

# ── Quiz Administration ───────────────────────────────────────────────────
if st.session_state.get("quiz_data"):
    quiz = st.session_state.quiz_data
    items = quiz.get("items", [])

    st.divider()
    st.subheader(f"📋 Quiz: `{quiz['quiz_id']}` — {len(items)} items")

    if "quiz_responses" not in st.session_state:
        st.session_state.quiz_responses = {}

    for item in items:
        iid = item["item_id"]
        bloom = item.get("bloom_level", "remember")
        color = BLOOM_COLORS.get(bloom, "#999")

        badge = f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:11px">{bloom.upper()}</span>'
        concept_badge = f'<span style="background:#555;color:white;padding:2px 8px;border-radius:10px;font-size:11px">{item.get("concept","?")}</span>'
        irt_info = f"b={item.get('difficulty_b', 0):.2f}  a={item.get('discrimination_a', 1):.2f}"

        st.markdown(f"{badge} {concept_badge}", unsafe_allow_html=True)
        st.markdown(f"**{item['question']}**")
        st.caption(f"IRT params: {irt_info}  |  Source: {item.get('source_url','')}")

        answer = st.radio(
            "Select answer:",
            options=item.get("options", []),
            key=f"q_{iid}",
            index=None,
        )
        if answer is not None:
            idx = item["options"].index(answer)
            st.session_state.quiz_responses[iid] = {
                "item_id": iid,
                "answer_index": idx,
                "response_time_ms": 3000,
            }
        st.divider()

    answered = len(st.session_state.quiz_responses)
    st.progress(answered / max(len(items), 1), text=f"Answered: {answered}/{len(items)}")

    if st.button("✅ Submit Quiz", type="primary", disabled=answered < len(items)):
        responses_list = list(st.session_state.quiz_responses.values())
        with st.spinner("Running CAT-IRT assessment…"):
            ks_result = api_post(
                "/assessment/run",
                {
                    "student_id": st.session_state.student_id,
                    "quiz_id": quiz["quiz_id"],
                    "responses": responses_list,
                },
            )

        if ks_result:
            st.session_state.knowledge_vector = ks_result
            st.success(f"✅ Assessment complete! Overall ability (θ) = {ks_result.get('theta', 0):.3f}")
            st.info("➡️ Go to **Stage 3 — Assessment** to view your knowledge state.")

    with st.expander("📄 Raw Quiz JSON"):
        st.json(quiz)
