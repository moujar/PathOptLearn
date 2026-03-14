"""LearnFlow AI — Streamlit frontend.

All communication with the FastAPI backend happens via httpx (sync).
Session state persists in st.session_state; no login required.
"""

import httpx
import streamlit as st

API_URL = "http://localhost:8000/api"
TIMEOUT = 120  # seconds — LLM calls can be slow

st.set_page_config(
    page_title="LearnFlow AI",
    page_icon="◈",
    layout="wide",
)

# ── Default session state ───────────────────────────────────────────────────

DEFAULTS: dict = {
    "phase": "landing",
    "session_id": None,
    "topic": "",
    "goal": "",
    "assessment_qs": [],
    "level": "",
    "knowledge_gaps": [],
    "roadmap": [],
    "current_module": 0,
    "module_content": "",
    "quiz_qs": [],
    "quiz_result": None,
    "completed": [],
    "error": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── API helpers ─────────────────────────────────────────────────────────────


def api_post(endpoint: str, payload: dict | None = None) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{API_URL}{endpoint}", json=payload or {})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        st.session_state.error = f"API error {e.response.status_code}: {e.response.text}"
        return {}
    except Exception as e:
        st.session_state.error = str(e)
        return {}


def api_get(endpoint: str) -> dict:
    try:
        with httpx.Client(timeout=60) as client:
            r = client.get(f"{API_URL}{endpoint}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        st.session_state.error = f"API error {e.response.status_code}: {e.response.text}"
        return {}
    except Exception as e:
        st.session_state.error = str(e)
        return {}


# ── Sidebar ─────────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("## ◈ LearnFlow AI")
    st.caption("Adaptive AI-powered learning")
    st.divider()

    if st.session_state.session_id:
        st.markdown(f"**Topic:** {st.session_state.topic}")
        st.markdown(f"**Level:** {st.session_state.level or 'TBD'}")
        st.markdown(f"**Phase:** `{st.session_state.phase}`")
        if st.session_state.roadmap:
            done = len(st.session_state.completed)
            total = len(st.session_state.roadmap)
            st.progress(done / total, text=f"{done}/{total} modules")
        st.divider()
        st.caption("Session ID")
        st.code(st.session_state.session_id, language=None)
        if st.button("New Session", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Dismiss"):
            st.session_state.error = None
            st.rerun()


# ── Phase: Landing ──────────────────────────────────────────────────────────


def show_landing():
    st.markdown("# ◈ LearnFlow AI")
    st.markdown("### Learn anything with AI-powered adaptive learning")
    st.markdown("> No account needed. Just start learning.")
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Machine Learning, SQL, Spanish...",
            key="input_topic",
        )
    with col2:
        goal = st.text_input(
            "Your goal (optional)",
            placeholder="e.g. Build a chatbot",
            key="input_goal",
        )

    if st.button("🚀 Start Learning", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a topic first.")
        else:
            with st.spinner("Crafting your personalised assessment..."):
                data = api_post(
                    "/sessions/create",
                    {"topic": topic.strip(), "goal": goal.strip() or topic.strip()},
                )
            if data:
                st.session_state.session_id = data["session_id"]
                st.session_state.topic = topic.strip()
                st.session_state.goal = goal.strip()
                st.session_state.assessment_qs = data["assessment_questions"]
                st.session_state.phase = "assessment"
                st.rerun()

    st.divider()
    st.markdown("#### Resume an existing session")
    sid = st.text_input("Paste your session ID:", key="resume_sid")
    if st.button("Resume Session"):
        if sid.strip():
            data = api_get(f"/sessions/{sid.strip()}")
            if data:
                st.session_state.session_id = sid.strip()
                st.session_state.topic = data.get("topic", "")
                st.session_state.level = data.get("detected_level", "")
                st.session_state.roadmap = data.get("roadmap", [])
                st.session_state.completed = data.get("completed_modules", [])
                st.session_state.current_module = data.get("current_module_idx", 0)
                st.session_state.phase = data.get("phase", "roadmap")
                st.rerun()
        else:
            st.warning("Please paste a session ID.")


# ── Phase: Assessment ───────────────────────────────────────────────────────


def show_assessment():
    st.markdown(f"## 📝 Level Assessment — *{st.session_state.topic}*")
    st.info(
        "Answer these questions honestly so we can build the perfect roadmap for you. "
        "It's okay not to know — that's the point!"
    )

    questions = st.session_state.assessment_qs
    answers: dict[str, int] = {}

    for i, q in enumerate(questions):
        st.markdown(f"**{i + 1}. {q['question']}**")
        selected = st.radio(
            f"Q{i + 1}",
            options=list(range(len(q["options"]))),
            format_func=lambda x, q=q: q["options"][x],
            key=f"assess_{i}",
            label_visibility="collapsed",
            horizontal=True,
        )
        answers[str(i)] = selected
        st.divider()

    if st.button("Submit Assessment →", type="primary", use_container_width=True):
        with st.spinner("Analyzing your level and building your roadmap... (this may take 30s)"):
            data = api_post(
                f"/sessions/{st.session_state.session_id}/assess",
                {"answers": answers},
            )
        if data:
            st.session_state.level = data["level"]
            st.session_state.knowledge_gaps = data.get("knowledge_gaps", [])
            st.session_state.roadmap = data["roadmap"]
            st.session_state.phase = "roadmap"
            st.rerun()


# ── Phase: Roadmap ──────────────────────────────────────────────────────────


def show_roadmap():
    st.markdown(f"## 🗺️ Your Learning Roadmap — *{st.session_state.topic}*")

    col1, col2, col3 = st.columns(3)
    col1.metric("Level", st.session_state.level or "—")
    col2.metric("Modules", len(st.session_state.roadmap))
    col3.metric("Completed", len(st.session_state.completed))

    if st.session_state.knowledge_gaps:
        st.markdown("**Knowledge gaps identified:**")
        for gap in st.session_state.knowledge_gaps:
            st.markdown(f"- {gap}")

    st.divider()

    completed = st.session_state.completed
    current = st.session_state.current_module

    for i, mod in enumerate(st.session_state.roadmap):
        is_done = i in completed
        is_current = i == current and not is_done
        is_locked = i > current and not is_done

        icon = "✅" if is_done else "👉" if is_current else "🔒"
        status_label = "Done" if is_done else "Active" if is_current else "Locked"

        with st.expander(
            f"{icon} Module {i + 1}: {mod['title']} — *{status_label}*",
            expanded=is_current,
        ):
            st.write(mod.get("description", ""))

            c1, c2, c3 = st.columns(3)
            c1.metric("Difficulty", mod.get("difficulty", "N/A"))
            c2.metric("Est. Time", mod.get("estimatedTime", "~30 min"))
            c3.metric("Status", status_label)

            if mod.get("learningObjectives"):
                st.markdown("**Objectives:**")
                for obj in mod["learningObjectives"]:
                    st.markdown(f"- {obj}")

            if is_current:
                if st.button(
                    f"Start Module {i + 1} →",
                    key=f"start_{i}",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Generating your lesson..."):
                        data = api_post(
                            f"/modules/{st.session_state.session_id}/generate-content"
                        )
                    if data:
                        st.session_state.module_content = data["content"]
                        st.session_state.phase = "learning"
                        st.rerun()


# ── Phase: Learning ─────────────────────────────────────────────────────────


def show_learning():
    idx = st.session_state.current_module
    mod = st.session_state.roadmap[idx]
    total = len(st.session_state.roadmap)

    st.progress(
        (len(st.session_state.completed)) / total,
        text=f"Module {idx + 1} of {total}",
    )
    st.markdown(f"## {mod['title']}")

    if mod.get("resources"):
        with st.expander("📚 Resources for this module"):
            for res in mod["resources"]:
                st.markdown(f"- [{res['title']}]({res['url']}) — *{res['type']}*")

    st.divider()
    st.markdown(st.session_state.module_content)
    st.divider()

    if st.button("🧪 Take Quiz to Continue →", type="primary", use_container_width=True):
        with st.spinner("Preparing your quiz..."):
            data = api_post(
                f"/quizzes/{st.session_state.session_id}/generate-quiz"
            )
        if data:
            st.session_state.quiz_qs = data["quiz_questions"]
            st.session_state.phase = "quiz"
            st.rerun()


# ── Phase: Quiz ─────────────────────────────────────────────────────────────


def show_quiz():
    idx = st.session_state.current_module
    mod = st.session_state.roadmap[idx]

    st.markdown(f"## 🧪 Quiz — *{mod['title']}*")
    st.warning("You need **70%** to pass and move to the next module.")

    answers: dict[str, int] = {}
    for i, q in enumerate(st.session_state.quiz_qs):
        st.markdown(f"**{i + 1}. {q['question']}**")
        selected = st.radio(
            f"Quiz Q{i + 1}",
            options=list(range(len(q["options"]))),
            format_func=lambda x, q=q: q["options"][x],
            key=f"quiz_{i}",
            label_visibility="collapsed",
            horizontal=True,
        )
        answers[str(i)] = selected
        st.divider()

    if st.button("Submit Quiz →", type="primary", use_container_width=True):
        with st.spinner("Evaluating your answers..."):
            data = api_post(
                f"/quizzes/{st.session_state.session_id}/submit-quiz",
                {"answers": answers},
            )
        if data:
            st.session_state.quiz_result = data
            st.session_state.phase = "quiz_result"
            st.rerun()


# ── Phase: Quiz result ──────────────────────────────────────────────────────


def show_quiz_result():
    result = st.session_state.quiz_result
    pct = int(result["score"] * 100)

    if result["passed"]:
        st.balloons()
        st.success(f"🎉 Passed! Score: **{pct}%**")

        # Update local state to reflect progress
        st.session_state.completed = result["completed_modules"]
        st.session_state.current_module = result["current_module_idx"]

        if result["is_course_complete"]:
            st.session_state.phase = "completed"
            st.rerun()

        if st.button("Next Module →", type="primary", use_container_width=True):
            st.session_state.phase = "roadmap"
            st.rerun()
    else:
        st.error(f"❌ Score: **{pct}%** — Need 70%. Review the material and try again.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Review Lesson →", use_container_width=True):
                st.session_state.phase = "learning"
                st.rerun()
        with col2:
            if st.button("Retry Quiz →", type="primary", use_container_width=True):
                # Regenerate quiz (different questions on retry)
                with st.spinner("Generating a new quiz..."):
                    data = api_post(
                        f"/quizzes/{st.session_state.session_id}/generate-quiz"
                    )
                if data:
                    st.session_state.quiz_qs = data["quiz_questions"]
                    st.session_state.phase = "quiz"
                    st.rerun()


# ── Phase: Completed ────────────────────────────────────────────────────────


def show_completed():
    st.balloons()
    st.markdown("# 🏆 Congratulations!")
    st.markdown(
        f"You have completed all modules for **{st.session_state.topic}**!\n\n"
        f"Level achieved: **{st.session_state.level}**"
    )
    st.success(
        f"You finished {len(st.session_state.roadmap)} modules. "
        "Keep building on this foundation!"
    )

    if st.button("Learn Something New →", type="primary", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Phase router ────────────────────────────────────────────────────────────

PHASES = {
    "landing": show_landing,
    "assessment": show_assessment,
    "roadmap": show_roadmap,
    "learning": show_learning,
    "quiz": show_quiz,
    "quiz_result": show_quiz_result,
    "completed": show_completed,
}

PHASES[st.session_state.phase]()
