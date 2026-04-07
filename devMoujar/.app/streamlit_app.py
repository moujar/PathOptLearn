import streamlit as st
import requests

from app.config import API_BASE

st.set_page_config(
    page_title="DeepSearch + LearnFlow",
    page_icon="🔍",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Run: `uvicorn app.main:app --reload`")
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def reset_learnflow():
    for key in ["lf_session", "lf_step", "lf_questions", "lf_answers",
                "lf_roadmap", "lf_level", "lf_gaps", "lf_content",
                "lf_quiz", "lf_eval"]:
        st.session_state.pop(key, None)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔍 DeepSearch")
page = st.sidebar.radio("Navigate", ["🌐 Quick Search", "🎬 YouTube Search", "📚 LearnFlow"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Ollama · DuckDuckGo · yt-dlp")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Quick Search
# ══════════════════════════════════════════════════════════════════════════════

if page == "🌐 Quick Search":
    st.title("🌐 Quick Web Search")
    st.caption("DuckDuckGo snippets + LLM synthesis — no full page scraping.")

    topic = st.text_input("Enter a topic or question", placeholder="e.g. latest breakthroughs in quantum computing")

    if st.button("Search", type="primary", disabled=not topic.strip()):
        with st.spinner("Searching & synthesising…"):
            data = api("post", "/api/search/quick", json={"topic": topic})
        if data:
            st.subheader("📝 Summary")
            st.markdown(data["summary"])

            st.subheader(f"🔗 Sources ({len(data['results'])})")
            for r in data["results"]:
                with st.expander(r.get("title", r["url"])):
                    st.write(r.get("snippet", ""))
                    st.markdown(f"[Open link]({r['url']})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — YouTube Search
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎬 YouTube Search":
    st.title("🎬 YouTube Search")

    query       = st.text_input("Search YouTube", placeholder="e.g. machine learning explained 2024")
    max_results = st.slider("Max results", 2, 10, 4)

    if st.button("Search YouTube", type="primary", disabled=not query.strip()):
        with st.spinner("Searching YouTube…"):
            data = api("post", "/api/search/youtube",
                       json={"query": query, "max_results": max_results})
        if data and data.get("videos"):
            cols = st.columns(2)
            for i, v in enumerate(data["videos"]):
                with cols[i % 2]:
                    st.image(v["thumb"], use_container_width=True)
                    st.markdown(f"**[{v['title']}]({v['url']})**")
                    st.caption(f"📺 {v['channel']}  ·  ⏱ {v['duration']}  ·  👁 {v['views']:,}" if v.get("views") else f"📺 {v['channel']}  ·  ⏱ {v['duration']}")
                    if v.get("desc"):
                        st.write(v["desc"][:120] + "…")
                    st.markdown("---")
        else:
            st.info("No videos found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LearnFlow
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 LearnFlow":
    st.title("📚 LearnFlow — Personalised Learning")

    step = st.session_state.get("lf_step", "start")

    # ── STEP: start ───────────────────────────────────────────────────────────
    if step == "start":
        st.markdown("Answer a quick diagnostic quiz and get a personalised learning roadmap with lessons and quizzes.")
        topic = st.text_input("What do you want to learn?", placeholder="e.g. Machine Learning, Quantum Computing…")

        if st.button("Start Learning 🚀", type="primary", disabled=not topic.strip()):
            with st.spinner("Generating your diagnostic quiz…"):
                data = api("post", "/api/learnflow/start", json={"topic": topic})
            if data:
                st.session_state["lf_session"]   = data["session_id"]
                st.session_state["lf_questions"]  = data["questions"]
                st.session_state["lf_step"]       = "assessment"
                st.session_state["lf_answers"]    = [""] * len(data["questions"])
                st.rerun()

    # ── STEP: assessment ──────────────────────────────────────────────────────
    elif step == "assessment":
        questions = st.session_state["lf_questions"]
        st.subheader("📋 Diagnostic Assessment")
        st.caption("Answer all questions so we can personalise your roadmap.")

        answers = list(st.session_state.get("lf_answers", [""] * len(questions)))

        with st.form("assessment_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}: {q['q']}**")
                if q.get("options"):
                    opts = {o[0]: o for o in q["options"]}   # "A" → "A) ..."
                    choice = st.radio(
                        label=f"q{i}",
                        options=list(opts.keys()),
                        format_func=lambda k, o=opts: o[k],
                        key=f"q_{i}",
                        label_visibility="collapsed",
                    )
                    answers[i] = choice
                else:
                    answers[i] = st.text_area(f"Your answer", key=f"q_{i}", label_visibility="collapsed")
                st.markdown("---")

            submitted = st.form_submit_button("Submit Answers →", type="primary")

        if submitted:
            with st.spinner("Analysing your level and building roadmap…"):
                data = api("post", "/api/learnflow/assessment",
                           json={"session_id": st.session_state["lf_session"],
                                 "answers": answers})
            if data:
                st.session_state["lf_level"]   = data["level"]
                st.session_state["lf_gaps"]    = data["gaps"]
                st.session_state["lf_roadmap"] = data["roadmap"]
                st.session_state["lf_step"]    = "roadmap"
                st.rerun()

    # ── STEP: roadmap ─────────────────────────────────────────────────────────
    elif step == "roadmap":
        level   = st.session_state["lf_level"]
        gaps    = st.session_state["lf_gaps"]
        roadmap = st.session_state["lf_roadmap"]

        level_color = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(level, "⚪")
        st.subheader(f"Your Level: {level_color} {level.capitalize()}")

        st.markdown("**Knowledge gaps identified:**")
        for g in gaps:
            st.markdown(f"- {g}")

        st.markdown("---")
        st.subheader("📚 Your Learning Roadmap")
        for m in roadmap:
            with st.expander(f"Module {m.get('module')}: {m.get('title')}  ·  ⏱ {m.get('duration')}"):
                st.markdown(f"**Objective:** {m.get('objective')}")
                st.markdown(f"**Concepts:** {', '.join(m.get('concepts', []))}")

        if st.button("Start Module 1 →", type="primary"):
            st.session_state["lf_step"] = "content"
            st.rerun()

    # ── STEP: content ─────────────────────────────────────────────────────────
    elif step == "content":
        if "lf_content" not in st.session_state:
            with st.spinner("Generating lesson content…"):
                data = api("post", "/api/learnflow/content",
                           json={"session_id": st.session_state["lf_session"]})
            if data:
                st.session_state["lf_content"] = data
        else:
            data = st.session_state["lf_content"]

        if data:
            roadmap = st.session_state.get("lf_roadmap", [])
            total   = data["total_modules"]
            idx     = data["module_idx"]

            # Progress bar
            st.progress((idx) / total, text=f"Module {idx + 1} / {total}")
            st.subheader(f"📖 {data['module_title']}")
            st.caption(f"Objective: {data['module_obj']}")
            st.markdown("---")
            st.markdown(data["content"])
            st.markdown("---")

            if st.button("Take the Quiz 🧠", type="primary"):
                st.session_state.pop("lf_content", None)
                st.session_state["lf_step"] = "quiz"
                st.rerun()

    # ── STEP: quiz ────────────────────────────────────────────────────────────
    elif step == "quiz":
        if "lf_quiz" not in st.session_state:
            with st.spinner("Generating quiz questions…"):
                data = api("post", "/api/learnflow/quiz",
                           json={"session_id": st.session_state["lf_session"]})
            if data:
                st.session_state["lf_quiz"] = data["questions"]

        questions = st.session_state.get("lf_quiz", [])
        if not questions:
            st.warning("No quiz questions available.")
        else:
            roadmap = st.session_state.get("lf_roadmap", [])
            st.subheader("🧠 Module Quiz")

            answers = [""] * len(questions)
            with st.form("quiz_form"):
                for i, q in enumerate(questions):
                    st.markdown(f"**Q{i+1}: {q['q']}**")
                    if q.get("options"):
                        opts = {o[0]: o for o in q["options"]}
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
                    data = api("post", "/api/learnflow/evaluate",
                               json={"session_id": st.session_state["lf_session"],
                                     "answers": answers})
                if data:
                    st.session_state["lf_eval"] = data
                    st.session_state["lf_step"] = "results"
                    st.session_state.pop("lf_quiz", None)
                    st.rerun()

    # ── STEP: results ─────────────────────────────────────────────────────────
    elif step == "results":
        ev = st.session_state.get("lf_eval", {})

        score   = ev.get("score", 0)
        passed  = ev.get("passed", False)
        feedback = ev.get("feedback", [])

        # Score display
        col1, col2 = st.columns([1, 3])
        with col1:
            color = "green" if passed else "red"
            st.markdown(f"<h1 style='color:{color};text-align:center'>{score:.0f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center'>/ 100</p>", unsafe_allow_html=True)
        with col2:
            if passed:
                st.success(f"✅ Passed! (attempt {ev.get('attempts', 1)})")
            else:
                st.error(f"❌ Score below 70 — review and try again (attempt {ev.get('attempts', 1)})")

        # Feedback
        if feedback:
            st.subheader("📝 Feedback")
            for fb in feedback:
                icon = "✅" if fb.get("correct") else "❌"
                st.markdown(f"{icon} **Q{fb.get('q','?')}:** {fb.get('explanation','')}")

        st.markdown("---")

        # YouTube suggestions
        videos = ev.get("videos", [])
        if videos:
            st.subheader("🎬 Suggested Videos")
            cols = st.columns(min(len(videos), 2))
            for i, v in enumerate(videos):
                with cols[i % 2]:
                    st.image(v["thumb"], use_container_width=True)
                    st.markdown(f"**[{v['title']}]({v['url']})**")
                    st.caption(f"📺 {v['channel']}  ·  ⏱ {v['duration']}")

        # Educational resources
        edu = ev.get("edu_resources", [])
        if edu:
            st.subheader("🌐 Educational Resources")
            for r in edu:
                st.markdown(f"- **[{r.get('title', r['url'])}]({r['url']})**  \n  {r.get('snippet','')[:120]}")

        st.markdown("---")

        # Navigation buttons
        if passed:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Next Module ➡️", type="primary"):
                    with st.spinner("Advancing…"):
                        adv = api("post", "/api/learnflow/advance",
                                  json={"session_id": st.session_state["lf_session"]})
                    if adv:
                        if adv["completed"]:
                            st.session_state["lf_step"] = "done"
                        else:
                            st.session_state.pop("lf_content", None)
                            st.session_state["lf_step"] = "content"
                        st.rerun()
            with col_b:
                if st.button("🏠 New Topic"):
                    reset_learnflow()
                    st.rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Retry Lesson", type="primary"):
                    st.session_state.pop("lf_content", None)
                    st.session_state["lf_step"] = "content"
                    st.rerun()
            with col_b:
                if st.button("🏠 New Topic"):
                    reset_learnflow()
                    st.rerun()

    # ── STEP: done ────────────────────────────────────────────────────────────
    elif step == "done":
        st.balloons()
        st.title("🏆 Learning Complete!")
        roadmap = st.session_state.get("lf_roadmap", [])
        level   = st.session_state.get("lf_level", "")
        st.success(f"You completed all {len(roadmap)} modules as a **{level}** learner. Great job!")

        st.subheader("Modules covered:")
        for m in roadmap:
            st.markdown(f"✅ Module {m.get('module')}: {m.get('title')}")

        if st.button("🚀 Start a New Topic", type="primary"):
            reset_learnflow()
            st.rerun()
