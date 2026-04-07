import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph.graph import (
    LearnFlowState, save_session,
    node_generate_assessment,
    node_analyze_level,
    node_research_resources,
    node_build_roadmap,
    node_generate_content,
    node_generate_quiz,
    node_evaluate_quiz,
    node_advance,
)

st.set_page_config(page_title="LearnFlow — PathOptLearn", page_icon="🧠", layout="wide")

# ── AUTH GUARD ───────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.switch_page("pages/1_Login.py")

username = st.session_state.get("username", "Student")


def reset_learnflow():
    for key in ["lf_state", "lf_step", "lf_eval"]:
        st.session_state.pop(key, None)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {username}")
    st.markdown("---")
    if st.button("🏠 Dashboard", use_container_width=True):
        st.switch_page("pages/2_Dashboard.py")
    if st.button("🔄 New Topic", use_container_width=True):
        reset_learnflow()
        st.rerun()

    # Show roadmap progress in sidebar when available
    state = st.session_state.get("lf_state")
    if state and state.roadmap:
        st.markdown("---")
        st.markdown("**Your Roadmap**")
        for i, m in enumerate(state.roadmap):
            icon = "✅" if i < state.current_module else ("▶" if i == state.current_module else "○")
            st.markdown(f"{icon} {m.get('title', '')}")


# ── MAIN ─────────────────────────────────────────────────────
st.title("🧠 LearnFlow — Personalised Learning")

step = st.session_state.get("lf_step", "start")

# ── STEP: start ───────────────────────────────────────────────────────────────
if step == "start":
    st.markdown("Answer a quick diagnostic quiz and get a personalised learning roadmap with lessons and quizzes.")
    topic = st.text_input("What do you want to learn?", placeholder="e.g. Machine Learning, Quantum Computing…")

    if st.button("Start Learning 🚀", type="primary", disabled=not topic.strip()):
        with st.spinner("Generating your diagnostic quiz…"):
            state = LearnFlowState(topic=topic.strip())
            save_session(state)
            state = node_generate_assessment(state)
        st.session_state["lf_state"] = state
        st.session_state["lf_step"]  = "assessment"
        st.rerun()


# ── STEP: assessment ──────────────────────────────────────────────────────────
elif step == "assessment":
    state     = st.session_state["lf_state"]
    questions = state.assessment_questions

    st.subheader("📋 Diagnostic Assessment")
    st.caption("Answer all questions so we can personalise your roadmap.")

    answers = [""] * len(questions)

    with st.form("assessment_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}: {q['q']}**")
            if q.get("options"):
                opts   = {o[0]: o for o in q["options"]}
                choice = st.radio(
                    label=f"q{i}",
                    options=list(opts.keys()),
                    format_func=lambda k, o=opts: o[k],
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
                answers[i] = choice
            else:
                answers[i] = st.text_area("Your answer", key=f"q_{i}", label_visibility="collapsed")
            st.markdown("---")

        submitted = st.form_submit_button("Submit Answers →", type="primary")

    if submitted:
        with st.spinner("Analysing your level and building roadmap…"):
            state.assessment_answers = answers
            state = node_analyze_level(state)
            state = node_research_resources(state)
            state = node_build_roadmap(state)
        st.session_state["lf_state"] = state
        st.session_state["lf_step"]  = "roadmap"
        st.rerun()


# ── STEP: roadmap ─────────────────────────────────────────────────────────────
elif step == "roadmap":
    state = st.session_state["lf_state"]

    level_color = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(state.level, "⚪")
    st.subheader(f"Your Level: {level_color} {state.level.capitalize()}")

    st.markdown("**Knowledge gaps identified:**")
    for g in state.gaps:
        st.markdown(f"- {g}")

    st.markdown("---")
    st.subheader("📚 Your Learning Roadmap")
    for m in state.roadmap:
        with st.expander(f"Module {m.get('module')}: {m.get('title')}  ·  ⏱ {m.get('duration')}"):
            st.markdown(f"**Objective:** {m.get('objective')}")
            st.markdown(f"**Concepts:** {', '.join(m.get('concepts', []))}")

    if st.button("Start Module 1 →", type="primary"):
        st.session_state["lf_step"] = "content"
        st.rerun()


# ── STEP: content ─────────────────────────────────────────────────────────────
elif step == "content":
    state = st.session_state["lf_state"]

    if not state.content:
        with st.spinner("Generating lesson content…"):
            state = node_generate_content(state)
        st.session_state["lf_state"] = state

    total = len(state.roadmap)
    idx   = state.current_module

    st.progress((idx) / total, text=f"Module {idx + 1} / {total}")
    module = state.current_module_info()
    st.subheader(f"📖 {module.get('title', '')}")
    st.caption(f"Objective: {module.get('objective', '')}")
    st.markdown("---")
    st.markdown(state.content)
    st.markdown("---")

    if st.button("Take the Quiz 🧠", type="primary"):
        st.session_state["lf_step"] = "quiz"
        st.rerun()


# ── STEP: quiz ────────────────────────────────────────────────────────────────
elif step == "quiz":
    state = st.session_state["lf_state"]

    if not state.quiz:
        with st.spinner("Generating quiz questions…"):
            state = node_generate_quiz(state)
        st.session_state["lf_state"] = state

    questions = state.quiz
    if not questions:
        st.warning("No quiz questions available.")
    else:
        st.subheader("🧠 Module Quiz")

        answers = [""] * len(questions)
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}: {q['q']}**")
                if q.get("options"):
                    opts   = {o[0]: o for o in q["options"]}
                    choice = st.radio(
                        label=f"quiz_q{i}",
                        options=list(opts.keys()),
                        format_func=lambda k, o=opts: o[k],
                        key=f"quiz_{i}",
                        label_visibility="collapsed",
                    )
                    answers[i] = choice
                else:
                    answers[i] = st.text_area("Answer", key=f"quiz_{i}", label_visibility="collapsed")
                st.markdown("---")

            submitted = st.form_submit_button("Submit Quiz →", type="primary")

        if submitted:
            with st.spinner("Evaluating your answers & finding resources…"):
                state.quiz_answers = answers
                eval_result        = node_evaluate_quiz(state)
            st.session_state["lf_state"] = state
            st.session_state["lf_eval"]  = eval_result
            st.session_state["lf_step"]  = "results"
            st.rerun()


# ── STEP: results ─────────────────────────────────────────────────────────────
elif step == "results":
    state = st.session_state["lf_state"]
    ev    = st.session_state.get("lf_eval", {})

    score    = ev.get("score", 0)
    passed   = ev.get("passed", False)
    feedback = ev.get("feedback", [])

    col1, col2 = st.columns([1, 3])
    with col1:
        color = "green" if passed else "red"
        st.markdown(f"<h1 style='color:{color};text-align:center'>{score:.0f}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>/ 100</p>", unsafe_allow_html=True)
    with col2:
        if passed:
            st.success(f"✅ Passed! (attempt {ev.get('attempts', 1)})")
        else:
            st.error(f"❌ Score below 70 — review and try again (attempt {ev.get('attempts', 1)})")

    if feedback:
        st.subheader("📝 Feedback")
        for fb in feedback:
            icon = "✅" if fb.get("correct") else "❌"
            st.markdown(f"{icon} **Q{fb.get('q','?')}:** {fb.get('explanation','')}")

    st.markdown("---")

    videos = ev.get("videos", [])
    if videos:
        st.subheader("🎬 Suggested Videos")
        cols = st.columns(min(len(videos), 2))
        for i, v in enumerate(videos):
            with cols[i % 2]:
                st.image(v["thumb"], use_container_width=True)
                st.markdown(f"**[{v['title']}]({v['url']})**")
                st.caption(f"📺 {v['channel']}  ·  ⏱ {v['duration']}")

    edu = ev.get("edu_resources", [])
    if edu:
        st.subheader("🌐 Educational Resources")
        for r in edu:
            st.markdown(f"- **[{r.get('title', r['url'])}]({r['url']})**  \n  {r.get('snippet','')[:120]}")

    st.markdown("---")

    if passed:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Next Module ➡️", type="primary"):
                with st.spinner("Advancing…"):
                    state = node_advance(state)
                state.content = ""
                state.quiz    = []
                if state.completed:
                    st.session_state["lf_state"] = state
                    st.session_state["lf_step"]  = "done"
                else:
                    st.session_state["lf_state"] = state
                    st.session_state["lf_step"]  = "content"
                st.rerun()
        with col_b:
            if st.button("🏠 New Topic"):
                reset_learnflow()
                st.rerun()
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Retry Lesson", type="primary"):
                state.content = ""
                state.quiz    = []
                st.session_state["lf_state"] = state
                st.session_state["lf_step"]  = "content"
                st.rerun()
        with col_b:
            if st.button("🏠 New Topic"):
                reset_learnflow()
                st.rerun()


# ── STEP: done ────────────────────────────────────────────────────────────────
elif step == "done":
    state = st.session_state.get("lf_state")
    st.balloons()
    st.title("🏆 Learning Complete!")
    if state:
        st.success(f"You completed all {len(state.roadmap)} modules as a **{state.level}** learner. Great job!")
        st.subheader("Modules covered:")
        for m in state.roadmap:
            st.markdown(f"✅ Module {m.get('module')}: {m.get('title')}")

    if st.button("🚀 Start a New Topic", type="primary"):
        reset_learnflow()
        st.rerun()
