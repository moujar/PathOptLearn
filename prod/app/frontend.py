"""
PathOptLearn — Streamlit frontend
Full learning flow:
  Register/Login → Topic → DeepSearch → Diagnostic Quiz → Level/Gaps
  → Roadmap → Module lesson → Quiz → (pass → next | fail → resources → retry)
  → Complete
"""
import os
import time

import requests
import streamlit as st
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    _AGRAPH_AVAILABLE = True
except ImportError:
    _AGRAPH_AVAILABLE = False
    # Stubs so the linter doesn't complain — never called at runtime
    agraph = Node = Edge = Config = None  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="PathOptLearn",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def api_post(path: str, json: dict = None, params: dict = None):
    try:
        r = requests.post(f"{API}{path}", json=json, params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def api_stream(path: str, **params):
    """Yield text chunks from a streaming GET endpoint."""
    try:
        with requests.get(f"{API}{path}", params=params, stream=True, timeout=180) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None):
                yield chunk.decode("utf-8", errors="ignore")
    except Exception as e:
        yield f"\n\n[Stream error: {e}]"


# ── Session-state defaults ────────────────────────────────────────────────────

def _defaults():
    defaults = {
        "page":            "auth",       # auth | topic | searching | quiz | gaps | roadmap | lesson | complete
        "student_id":      None,
        "student_name":    "",
        "session_id":      None,
        "topic":           "",
        "level":           "beginner",
        "level_emoji":     "🟢",
        "gaps":            [],
        "roadmap":         {},
        "diag_questions":  [],
        "diag_answers":    {},
        "module_uid":      None,
        "module_title":    "",
        "lesson_content":  "",
        "lesson_sources":  [],
        "lesson_videos":   [],
        "recommended":     [],
        "quiz_questions":  [],
        "quiz_answers":    {},
        "quiz_score":      None,
        "attempts":        0,
        "gap_resources":   {},
        "search_resources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_defaults()


def go(page: str):
    st.session_state["page"] = page
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
        st.title("PathOptLearn")
        st.caption("Adaptive · Personalised · AI-powered")
        st.divider()

        if st.session_state["student_id"]:
            st.success(f"Logged in as **{st.session_state['student_name']}**")
            if st.button("Logout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                _defaults()
                st.rerun()
            st.divider()

        page = st.session_state["page"]
        steps = [
            ("auth",      "1. Account"),
            ("topic",     "2. Choose Topic"),
            ("searching", "3. Research"),
            ("quiz",      "4. Diagnostic Quiz"),
            ("gaps",      "5. Level & Gaps"),
            ("roadmap",   "6. Roadmap"),
            ("lesson",    "7. Learn"),
            ("complete",  "8. Complete"),
        ]
        for key, label in steps:
            icon = "✅" if _step_done(key) else ("▶️" if key == page else "○")
            st.markdown(f"{icon} {label}")

        if st.session_state["topic"]:
            st.divider()
            st.markdown(f"**Topic:** {st.session_state['topic']}")
            if st.session_state["level"]:
                st.markdown(
                    f"**Level:** {st.session_state['level_emoji']} "
                    f"{st.session_state['level'].capitalize()}"
                )

        st.divider()
        if st.session_state.get("student_id"):
            if st.button("📊 My Progress", use_container_width=True):
                st.session_state["_progress_prev"] = st.session_state["page"]
                go("progress")
        if st.button("🕸️ Knowledge Graph", use_container_width=True):
            st.session_state["_graph_prev"] = st.session_state["page"]
            go("graph")


def _step_done(key: str) -> bool:
    order = ["auth", "topic", "searching", "quiz", "gaps", "roadmap", "lesson", "complete"]
    current = st.session_state["page"]
    try:
        return order.index(current) > order.index(key)
    except ValueError:
        return False


# ── Page: auth ────────────────────────────────────────────────────────────────

def page_auth():
    st.title("Welcome to PathOptLearn 🎓")
    st.markdown(
        "An AI-powered adaptive learning platform. "
        "Register or log in to start your personalised learning journey."
    )
    st.divider()

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # Login tab ────────────────────────────────────────────────────────────────
    with tab_login:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
            if not username or not password:
                st.warning("Enter username and password.")
            else:
                # Use existing users endpoint (GET /users) to find by username
                data = api_get("/users")
                if data:
                    match = next(
                        (u for u in data.get("users", []) if u["username"] == username),
                        None,
                    )
                    if match:
                        st.session_state["student_id"]   = match["id"]
                        st.session_state["student_name"] = match["username"]
                        go("topic")
                    else:
                        st.error("Username not found. Please register first.")

    # Register tab ─────────────────────────────────────────────────────────────
    with tab_register:
        st.subheader("Create Account")
        col1, col2 = st.columns(2)
        with col1:
            reg_username = st.text_input("Username", key="reg_username")
            reg_email    = st.text_input("Email",    key="reg_email")
        with col2:
            reg_password  = st.text_input("Password (min 6 chars)", type="password", key="reg_password")
            reg_password2 = st.text_input("Confirm Password",       type="password", key="reg_password2")

        if st.button("Create Account", type="primary", use_container_width=True, key="btn_register"):
            if not all([reg_username, reg_email, reg_password]):
                st.warning("All fields are required.")
            elif reg_password != reg_password2:
                st.error("Passwords do not match.")
            else:
                result = api_post("/students", json={
                    "username": reg_username,
                    "email":    reg_email,
                    "password": reg_password,
                })
                if result:
                    st.success(
                        f"Account created! Your verification code is: "
                        f"**{result['verification_code']}**"
                    )
                    # Auto-verify for demo purposes
                    verify = api_post(
                        f"/students/{result['student_id']}/verify",
                        params={"code": result["verification_code"]},
                    )
                    if verify and verify.get("verified"):
                        st.success("Account verified! You can now log in.")
                        st.session_state["student_id"]   = result["student_id"]
                        st.session_state["student_name"] = reg_username
                        time.sleep(1)
                        go("topic")

    st.divider()
    if st.button("Continue as Guest", use_container_width=True):
        st.session_state["student_id"]   = None
        st.session_state["student_name"] = "Guest"
        go("topic")


# ── Page: topic ───────────────────────────────────────────────────────────────

def page_topic():
    st.title("What do you want to learn? 🔍")
    st.markdown(
        "Enter any subject — from *Machine Learning* to *Quantum Physics* to *History of Rome*."
    )
    st.divider()

    topic = st.text_input(
        "Enter a topic or paste a question",
        placeholder="e.g. Machine Learning, Electromagnetism, Binary Trees …",
        key="topic_input",
        value=st.session_state.get("topic", ""),
    )

    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Start Learning →", type="primary", use_container_width=True)
    with col2:
        skip_search = st.checkbox("Skip deep search (faster, less context)", value=False)

    if start:
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            st.session_state["topic"]       = topic.strip()
            st.session_state["skip_search"] = skip_search
            go("searching")


# ── Page: searching ───────────────────────────────────────────────────────────

def page_searching():
    topic = st.session_state["topic"]
    st.title(f"Researching: {topic} 🌐")
    st.markdown(
        "Gathering educational resources from the web, "
        "Wikipedia, arXiv, YouTube, and the knowledge graph…"
    )

    if st.session_state.get("skip_search"):
        st.info("Skipping deep search — using LLM directly.")
        st.session_state["search_resources"] = []
        go("quiz")
        return

    placeholder = st.empty()
    progress    = st.progress(0, text="Starting search…")

    result = None
    with st.spinner("Running deep search pipeline…"):
        payload = {"topic": topic}
        if st.session_state["student_id"]:
            payload["student_id"] = st.session_state["student_id"]
        result = api_post("/deep-search", json=payload)

    progress.progress(100, text="Done!")

    if result:
        st.session_state["search_resources"] = result.get("resources", [])
        st.success(
            f"Found **{result['total']}** resources "
            f"({'cached' if result.get('cached') else 'fresh'})"
        )
        _render_resources(result.get("resources", [])[:6])
    else:
        st.warning("Search returned no results — continuing with LLM only.")
        st.session_state["search_resources"] = []

    if st.button("Continue to Diagnostic Quiz →", type="primary"):
        go("quiz")


def _render_resources(resources: list):
    if not resources:
        return
    st.subheader("Resources found")
    for r in resources:
        icon = "▶️" if r.get("resource_type") == "youtube" else "🔗"
        with st.expander(f"{icon} {r.get('title', 'Resource')}", expanded=False):
            st.markdown(f"[{r.get('url', '')}]({r.get('url', '')})")
            if r.get("summary"):
                st.markdown(f"**Summary:** {r['summary']}")
            if r.get("channel"):
                st.caption(f"Channel: {r['channel']} · {r.get('duration', '')}")


# ── Page: quiz (diagnostic) ───────────────────────────────────────────────────

def page_quiz():
    topic = st.session_state["topic"]
    st.title(f"Diagnostic Quiz: {topic} 📝")
    st.markdown(
        "Answer these questions honestly — they assess your *current* knowledge "
        "so we can build the right learning path for you."
    )
    st.divider()

    # Generate questions on first visit
    if not st.session_state["diag_questions"]:
        with st.spinner("Generating diagnostic questions…"):
            data = api_get("/assess", topic=topic)
        if not data:
            st.error("Could not generate quiz. Check the API is running.")
            return
        st.session_state["diag_questions"] = data.get("questions", [])
        st.session_state["diag_answers"]   = {}

    questions = st.session_state["diag_questions"]
    if not questions:
        st.error("No questions returned.")
        return

    with st.form("diagnostic_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            level_badge = {"basic": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(
                q.get("level", "basic"), "⚪"
            )
            st.caption(f"{level_badge} {q.get('level', '').capitalize()}")

            options = q.get("options") or []
            if options:
                options_with_idk = options + ["E. I don't know"]
                choice = st.radio(
                    label="Select answer",
                    options=options_with_idk,
                    key=f"diag_q_{i}",
                    label_visibility="collapsed",
                )
                # Extract letter (A/B/C/D/E) from "A. ..." format
                st.session_state["diag_answers"][i] = choice[0] if choice else "E"
            else:
                ans = st.text_input("Your answer", key=f"diag_short_{i}")
                st.session_state["diag_answers"][i] = ans

            st.divider()

        submitted = st.form_submit_button("Submit Answers →", type="primary", use_container_width=True)

    if submitted:
        answers = [
            st.session_state["diag_answers"].get(i, "A")
            for i in range(len(questions))
        ]
        with st.spinner("Analysing your answers…"):
            result = api_post("/assess/evaluate", json={
                "topic":     topic,
                "questions": questions,
                "answers":   answers,
            })
        if result:
            st.session_state["level"]       = result.get("level", "beginner")
            st.session_state["level_emoji"] = result.get("level_emoji", "🟢")
            st.session_state["diag_score"]  = result.get("score", 0)
            st.session_state["diag_feedback"] = result.get("feedback", "")

            # Also run /find-gaps to get structured gap list
            gap_payload = {"topic": topic, "questions": questions, "answers": answers}
            if st.session_state["student_id"]:
                gap_payload["student_id"] = st.session_state["student_id"]
            gap_result = api_post("/find-gaps", json=gap_payload)
            if gap_result:
                st.session_state["gaps"] = [
                    g["concept"] for g in gap_result.get("gaps", [])
                ]

            go("gaps")


# ── Page: gaps ────────────────────────────────────────────────────────────────

def page_gaps():
    topic   = st.session_state["topic"]
    level   = st.session_state["level"]
    emoji   = st.session_state["level_emoji"]
    score   = st.session_state.get("diag_score", 0)
    feedback = st.session_state.get("diag_feedback", "")
    gaps    = st.session_state["gaps"]

    st.title("Your Learning Profile 🧠")

    # Level card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{score}%")
    with col2:
        st.metric("Level", f"{emoji} {level.capitalize()}")
    with col3:
        st.metric("Gaps Found", len(gaps))

    if feedback:
        st.info(feedback)

    st.divider()

    if gaps:
        st.subheader("Knowledge Gaps Identified")
        for g in gaps:
            st.markdown(f"- **{g}**")

        st.subheader("Recommended Resources for Your Gaps")
        with st.spinner("Finding best resources for your gaps…"):
            rec_params = {"gaps": ",".join(gaps[:4]), "limit": 3}
            if st.session_state["student_id"]:
                rec_params["student_id"] = st.session_state["student_id"]
            recs = api_get("/recommender", **rec_params)
        if recs:
            st.session_state["gap_resources"] = recs.get("results", {})
            for gap_name, resources in recs.get("results", {}).items():
                st.markdown(f"**{gap_name}**")
                for r in resources:
                    icon = "▶️" if r.get("resource_type") == "youtube" else "🔗"
                    st.markdown(
                        f"- {icon} [{r.get('title', r.get('url'))}]({r.get('url')}) "
                        f"— *{r.get('reason', '')}*"
                    )
    else:
        st.success("No significant gaps detected — you have a solid foundation!")

    st.divider()
    if st.button("Build My Learning Roadmap →", type="primary", use_container_width=True):
        go("roadmap")


# ── Page: roadmap ─────────────────────────────────────────────────────────────

def page_roadmap():
    topic = st.session_state["topic"]
    level = st.session_state["level"]
    st.title(f"Your Learning Roadmap 🗺️")
    st.markdown(f"**Topic:** {topic} · **Level:** {st.session_state['level_emoji']} {level.capitalize()}")
    st.divider()

    if not st.session_state["roadmap"]:
        with st.spinner("Building your personalised roadmap (may take 30–60 s)…"):
            data = api_get("/roadmap", topic=topic, level=level)
        if not data:
            st.error("Could not generate roadmap.")
            return
        st.session_state["roadmap"] = data
        # Overwrite topic with the cleaned version returned by the backend
        if data.get("topic"):
            st.session_state["topic"] = data["topic"]

        # Start a learning session
        if st.session_state["student_id"]:
            sess = api_post(
                "/session/start",
                params={
                    "topic":       data.get("topic", topic),
                    "level":       level,
                    "level_emoji": st.session_state["level_emoji"],
                    "user_id":     st.session_state["student_id"],
                },
            )
            if sess:
                st.session_state["session_id"] = sess["session_id"]

    roadmap = st.session_state["roadmap"]
    levels  = roadmap.get("levels", [])

    if levels:
        for lvl in levels:
            lname  = lvl.get("level_name", "")
            lemoji = lvl.get("emoji", "")
            st.subheader(f"{lemoji} {lname} Level")
            cols = st.columns(min(len(lvl.get("modules", [])), 3))
            for i, mod in enumerate(lvl.get("modules", [])):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**Module {mod['id']}**")
                        st.markdown(f"*{mod['title']}*")
                        st.caption(mod.get("objective", ""))
                        if mod.get("concepts"):
                            st.caption("Concepts: " + ", ".join(mod["concepts"][:3]))
                        st.caption(f"⏱ {mod.get('duration_minutes', 60)} min")
    else:
        # Flat module list (fallback)
        for mod in roadmap.get("modules", []):
            with st.container(border=True):
                st.markdown(f"**Module {mod['id']}: {mod['title']}**")
                st.caption(mod.get("objective", ""))

    st.divider()
    total = roadmap.get("total_duration", "")
    st.caption(f"Total estimated time: {total}" if total else "")

    # Pre-select first module
    modules = roadmap.get("modules", [])
    if modules:
        first = modules[0]
        st.session_state["module_uid"]   = f"{roadmap.get('topic', topic)}::{first['id']}"
        st.session_state["module_title"] = first["title"]

    if st.button("Start First Module →", type="primary", use_container_width=True):
        st.session_state["lesson_content"] = ""
        st.session_state["quiz_questions"] = []
        st.session_state["quiz_answers"]   = {}
        st.session_state["quiz_score"]     = None
        st.session_state["attempts"]       = 0
        go("lesson")


# ── Page: lesson ──────────────────────────────────────────────────────────────

def page_lesson():
    module_uid = st.session_state["module_uid"]
    module_num = module_uid.split("::")[-1] if "::" in module_uid else "1"
    # Use the topic embedded in module_uid (always the cleaned version)
    topic      = module_uid.rsplit("::", 1)[0] if "::" in module_uid else st.session_state["topic"]
    session_id = st.session_state.get("session_id")

    # ── Subpage: show lesson content ──────────────────────────────────────────
    if not st.session_state["lesson_content"]:
        st.title(f"Loading Module {module_num}… 📖")
        with st.spinner("Generating lesson content (deep search + AI synthesis)…"):
            params = {"topic": topic, "module_id": int(module_num)}
            if session_id:
                params["session_id"] = session_id
            data = api_get("/lesson", **params)
        if not data:
            st.error("Could not load lesson.")
            return
        st.session_state["lesson_content"] = data.get("content", "")
        st.session_state["lesson_sources"] = data.get("sources", [])
        st.session_state["lesson_videos"]  = data.get("videos", [])
        st.session_state["recommended"]    = data.get("recommended", [])
        st.session_state["module_title"]   = data.get("module_title", f"Module {module_num}")
        st.rerun()

    title   = st.session_state["module_title"]
    content = st.session_state["lesson_content"]
    sources = st.session_state["lesson_sources"]
    videos  = st.session_state["lesson_videos"]
    recs    = st.session_state["recommended"]

    st.title(f"Module {module_num}: {title} 📖")

    # Score badge if returning after a failed quiz attempt
    score = st.session_state["quiz_score"]
    if score is not None:
        if score < 70:
            st.warning(
                f"Previous attempt: **{score:.0f}%** (need 70%). "
                f"Review the lesson and try again."
            )
        else:
            st.success(f"Passed with **{score:.0f}%**!")

    # Tabs: Lesson | Resources | Videos
    tab_lesson, tab_resources, tab_videos = st.tabs(["Lesson", "Resources", "Videos"])

    with tab_lesson:
        st.markdown(content)

    with tab_resources:
        if recs:
            st.subheader("Recommended for this module")
            for r in recs:
                if isinstance(r, str):
                    st.markdown(f"- 🔗 {r}")
                    continue
                icon = "▶️" if r.get("type") == "video" else "🔗"
                st.markdown(
                    f"- {icon} [{r.get('title', r.get('url', r))}]({r.get('url', '#')}) "
                    f"— *{r.get('reason', '')}*"
                )
        if sources:
            st.subheader("All sources")
            for s in sources[:8]:
                if isinstance(s, str):
                    st.markdown(f"- {s}")
                    continue
                st.markdown(f"- [{s.get('title', s.get('url', s))}]({s.get('url', '#')})")
                if s.get("snippet"):
                    st.caption(s["snippet"][:150])

    with tab_videos:
        if videos:
            cols = st.columns(2)
            for i, v in enumerate(videos[:6]):
                if isinstance(v, str):
                    st.markdown(f"- ▶️ {v}")
                    continue
                with cols[i % 2]:
                    with st.container(border=True):
                        vid_id = v.get("id") or v.get("video_id", "")
                        if vid_id:
                            st.image(
                                f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
                                use_column_width=True,
                            )
                        st.markdown(f"**[{v.get('title', 'Video')}]({v.get('url', '')})**")
                        st.caption(f"{v.get('channel', '')} · {v.get('duration', '')}")
        else:
            st.info("No videos found for this module.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Take the Quiz →", type="primary", use_container_width=True):
            st.session_state["quiz_questions"] = []
            st.session_state["quiz_answers"]   = {}
            st.session_state["quiz_score"]     = None
            go("quiz_module")
    with col2:
        if st.button("Back to Roadmap", use_container_width=True):
            go("roadmap")


# ── Page: quiz_module ─────────────────────────────────────────────────────────

def page_quiz_module():
    title   = st.session_state["module_title"]
    content = st.session_state["lesson_content"]
    attempt = st.session_state["attempts"] + 1

    st.title(f"Quiz: {title} ✏️")
    st.caption(f"Attempt {attempt} · Passing score: 70%")
    st.divider()

    # Generate quiz questions from lesson content
    if not st.session_state["quiz_questions"]:
        with st.spinner("Generating quiz questions…"):
            data = api_post("/quiz", json={"content": content, "num_questions": 5})
        if not data:
            st.error("Could not generate quiz.")
            return
        st.session_state["quiz_questions"] = data.get("questions", [])
        st.session_state["quiz_answers"]   = {}

    questions = st.session_state["quiz_questions"]
    if not questions:
        st.error("No questions generated.")
        return

    with st.form("module_quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            options = q.get("options") or []
            if options:
                choice = st.radio(
                    label="Select answer",
                    options=options,
                    key=f"mod_q_{i}",
                    label_visibility="collapsed",
                )
                st.session_state["quiz_answers"][i] = choice[0] if choice else "A"
            else:
                ans = st.text_input("Answer", key=f"mod_short_{i}")
                st.session_state["quiz_answers"][i] = ans
            st.divider()

        submitted = st.form_submit_button("Submit Quiz →", type="primary", use_container_width=True)

    if submitted:
        answers = [
            st.session_state["quiz_answers"].get(i, "A")
            for i in range(len(questions))
        ]

        # Score via /find-gaps (it returns score + gaps)
        topic  = st.session_state["topic"]
        gap_payload2 = {"topic": title, "questions": questions, "answers": answers}
        if st.session_state["student_id"]:
            gap_payload2["student_id"] = st.session_state["student_id"]
        result = api_post("/find-gaps", json=gap_payload2)

        if result:
            score = float(result.get("score", 0))
            passed = score >= 70
            st.session_state["quiz_score"] = score
            st.session_state["attempts"]  += 1
            gaps = [g["concept"] for g in result.get("gaps", [])]
            st.session_state["gaps"] = gaps

            if passed:
                # Persist progress and advance
                session_id = st.session_state.get("session_id")
                if session_id:
                    api_post("/next", json={
                        "session_id":           session_id,
                        "completed_module_uid": st.session_state["module_uid"],
                        "quiz_score":           score,
                        "num_quiz_questions":   5,
                    })
                go("quiz_result")
            else:
                go("quiz_result")


# ── Page: quiz_result ─────────────────────────────────────────────────────────

def page_quiz_result():
    score   = st.session_state.get("quiz_score", 0)
    passed  = score >= 70
    gaps    = st.session_state.get("gaps", [])
    topic   = st.session_state["topic"]
    roadmap = st.session_state["roadmap"]

    if passed:
        st.success(f"# Passed! {score:.0f}% ✅")
        st.balloons()
        st.markdown("Great work! You've mastered this module.")

        # Find next module from roadmap
        modules     = roadmap.get("modules", [])
        current_uid = st.session_state["module_uid"]
        try:
            current_id  = int(current_uid.split("::")[-1])
            done_ids    = {current_id}
            next_mod    = next(
                (m for m in modules if m["id"] not in done_ids and m["id"] > current_id),
                None,
            )
        except (ValueError, IndexError):
            next_mod = None

        if next_mod:
            st.info(f"Next up: **Module {next_mod['id']}: {next_mod['title']}**")
            if st.button("Start Next Module →", type="primary", use_container_width=True):
                base_topic = roadmap.get("topic", topic)
                st.session_state["module_uid"]   = f"{base_topic}::{next_mod['id']}"
                st.session_state["module_title"] = next_mod["title"]
                st.session_state["lesson_content"] = ""
                st.session_state["quiz_questions"] = []
                st.session_state["quiz_answers"]   = {}
                st.session_state["quiz_score"]     = None
                st.session_state["attempts"]       = 0
                go("lesson")
        else:
            st.success("🎉 You've completed all modules in this roadmap!")
            if st.button("See Completion Page →", type="primary"):
                go("complete")
    else:
        st.error(f"# {score:.0f}% — Keep going! ❌")
        st.markdown(f"You need **70%** to pass. You got **{score:.0f}%**.")

        if gaps:
            st.subheader("Focus on these concepts before retrying:")
            for g in gaps:
                st.markdown(f"- **{g}**")

            st.subheader("Recommended resources")
            with st.spinner("Finding resources for your gaps…"):
                rec_params = {"gaps": ",".join(gaps[:4]), "limit": 3}
                if st.session_state["student_id"]:
                    rec_params["student_id"] = st.session_state["student_id"]
                recs = api_get("/recommender", **rec_params)
            if recs:
                for gap_name, resources in recs.get("results", {}).items():
                    st.markdown(f"**{gap_name}**")
                    for r in resources:
                        icon = "▶️" if r.get("resource_type") == "youtube" else "🔗"
                        st.markdown(
                            f"- {icon} [{r.get('title', r.get('url'))}]({r.get('url')}) "
                            f"— *{r.get('reason', '')}*"
                        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Review Lesson", use_container_width=True):
                go("lesson")
        with col2:
            if st.button("Retry Quiz →", type="primary", use_container_width=True):
                st.session_state["quiz_questions"] = []
                st.session_state["quiz_answers"]   = {}
                st.session_state["quiz_score"]     = None
                go("quiz_module")


# ── Page: complete ────────────────────────────────────────────────────────────

def page_complete():
    topic = st.session_state["topic"]
    level = st.session_state["level"]
    st.title(f"Course Complete! 🏆")
    st.balloons()

    st.markdown(
        f"""
        **Congratulations, {st.session_state['student_name']}!**

        You have completed the **{topic}** learning path at **{level}** level.
        """
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Learn Another Topic", type="primary", use_container_width=True):
            st.session_state["topic"]           = ""
            st.session_state["roadmap"]         = {}
            st.session_state["diag_questions"]  = []
            st.session_state["diag_answers"]    = {}
            st.session_state["gaps"]            = []
            st.session_state["lesson_content"]  = ""
            st.session_state["quiz_questions"]  = []
            st.session_state["quiz_score"]      = None
            st.session_state["attempts"]        = 0
            go("topic")
    with col2:
        if st.session_state["student_id"]:
            history = api_get(f"/users/{st.session_state['student_id']}/history")
            if history:
                st.subheader("Your Learning History")
                for sess in history.get("history", [])[:5]:
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"**{sess['topic']}**")
                            st.caption(sess["started_at"][:10])
                        with col_b:
                            st.metric("Modules", sess["modules_completed"])
                        with col_c:
                            avg = sess.get("avg_score")
                            st.metric("Avg Score", f"{avg:.0f}%" if avg else "—")


# ── Page: progress ───────────────────────────────────────────────────────────

def page_progress():
    student_id = st.session_state.get("student_id")
    if not student_id:
        st.warning("You must be logged in to view your progress.")
        if st.button("Go to Login"):
            go("auth")
        return

    st.title(f"My Learning Progress 📊")
    st.markdown(f"Logged in as **{st.session_state['student_name']}**")
    st.divider()

    with st.spinner("Loading your progress…"):
        data = api_get(f"/students/{student_id}/progress")
    if not data:
        st.error("Could not load progress data.")
        return

    stats   = data.get("stats", {})
    assessments = data.get("assessments", [])
    gaps    = data.get("gaps", [])
    mastery = data.get("mastery", [])
    progress = data.get("progress", [])

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quizzes Taken",    stats.get("assessments_taken", 0))
    c2.metric("Avg Score",        f"{stats.get('avg_score', 0)}%")
    c3.metric("Best Score",       f"{stats.get('best_score', 0)}%")
    c4.metric("Open Gaps",        stats.get("open_gaps", 0))
    c5.metric("Topics Mastered",  stats.get("topics_mastered", 0))
    st.divider()

    tab_assess, tab_gaps, tab_mastery, tab_modules = st.tabs([
        "📝 Assessments", "🔍 Knowledge Gaps", "🏅 Topic Mastery", "📖 Module Progress"
    ])

    # ── Assessments tab ───────────────────────────────────────────────────────
    with tab_assess:
        if not assessments:
            st.info("No assessments taken yet.")
        for i, a in enumerate(assessments):
            score = a.get("score", 0) or 0
            level = a.get("level", "—")
            emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(level, "⚪")
            date  = (a.get("taken_at") or "")[:10]
            with st.expander(f"Assessment {i+1} — {emoji} {level.capitalize()}  |  Score: {score}%  |  {date}"):
                st.progress(score / 100)
                answers = a.get("answers", [])
                if answers:
                    correct = sum(1 for ans in answers if ans.get("is_correct"))
                    st.caption(f"{correct}/{len(answers)} correct answers")
                    for ans in answers:
                        icon = "✅" if ans.get("is_correct") else "❌"
                        st.markdown(
                            f"{icon} **{ans.get('concept','—')}** — "
                            f"*{ans.get('question','')[:80]}*  \n"
                            f"Your answer: `{ans.get('student_answer','')}` | "
                            f"Correct: `{ans.get('correct_answer','')}`"
                        )

    # ── Gaps tab ──────────────────────────────────────────────────────────────
    with tab_gaps:
        open_gaps   = [g for g in gaps if not g.get("resolved_at")]
        closed_gaps = [g for g in gaps if g.get("resolved_at")]

        if not gaps:
            st.success("No knowledge gaps recorded yet!")
        else:
            if open_gaps:
                st.subheader(f"Open Gaps ({len(open_gaps)})")
                for g in open_gaps:
                    sev = g.get("severity", "medium")
                    sev_colour = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
                    st.markdown(
                        f"{sev_colour} **{g['topic_name']}** — "
                        f"severity: `{sev}` · identified: {(g.get('identified_at') or '')[:10]}"
                    )
            if closed_gaps:
                st.subheader(f"Resolved Gaps ({len(closed_gaps)})")
                for g in closed_gaps:
                    st.markdown(
                        f"✅ **{g['topic_name']}** — "
                        f"resolved: {(g.get('resolved_at') or '')[:10]}"
                    )

    # ── Mastery tab ───────────────────────────────────────────────────────────
    with tab_mastery:
        if not mastery:
            st.info("No topic mastery data yet. Complete some quizzes to build it up.")
        else:
            for m in mastery:
                score = float(m.get("mastery_score", 0))
                pct   = min(int(score * 100), 100)
                label = "Mastered ✅" if score >= 0.7 else ("In progress 🔄" if score >= 0.4 else "Needs work ❌")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{m['topic_name']}**")
                    st.progress(pct / 100)
                with col_b:
                    st.metric("Score", f"{pct}%", label)
                st.caption(
                    f"Attempts: {m.get('attempt_count', 0)} | "
                    f"Passed: {m.get('pass_count', 0)} | "
                    f"Last updated: {(m.get('last_updated') or '')[:10]}"
                )
                st.divider()

    # ── Module progress tab ───────────────────────────────────────────────────
    with tab_modules:
        if not progress:
            st.info("No module progress recorded yet.")
        else:
            for p in progress:
                score  = p.get("score", 0) or 0
                total  = p.get("total", 1) or 1
                passed = bool(p.get("passed"))
                pct    = round(score / total * 100) if total else 0
                title  = p.get("module_title") or p.get("title") or "Module"
                date   = (p.get("timestamp") or "")[:10]
                icon   = "✅" if passed else "❌"
                with st.container(border=True):
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.markdown(f"{icon} **{title}**")
                        st.caption(f"Date: {date} | Attempt: {p.get('attempt_number', 1)}")
                    with col_b:
                        st.metric("Score", f"{pct}%")
                    with col_c:
                        st.metric("Result", "Pass" if passed else "Fail")

    st.divider()
    if st.button("← Back", use_container_width=False):
        go(st.session_state.get("_progress_prev", "topic"))


# ── Page: knowledge graph ────────────────────────────────────────────────────

def page_graph():
    st.title("Knowledge Graph 🕸️")
    st.markdown("Visual map of all topics, modules, concepts and resources stored in Neo4j.")
    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_topic = st.text_input(
            "Filter by topic (leave blank for all)",
            placeholder="e.g. Machine Learning",
        )
    with col2:
        limit = st.slider("Max nodes", 50, 500, 200, step=50)

    if st.button("Load Graph", type="primary"):
        params = {"limit": limit}
        if filter_topic.strip():
            params["topic"] = filter_topic.strip()
        data = api_get("/graph", **params)
        if data:
            st.session_state["_graph_data"] = data

    data = st.session_state.get("_graph_data")
    if not data:
        st.info("Click **Load Graph** to fetch the knowledge graph from Neo4j.")
        return

    stats = data.get("stats", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Topics",    stats.get("topics", 0))
    c2.metric("Modules",   stats.get("modules", 0))
    c3.metric("Concepts",  stats.get("concepts", 0))
    c4.metric("Resources", stats.get("resources", 0))
    c5.metric("Edges",     stats.get("edges", 0))
    st.divider()

    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", [])

    if not nodes_raw:
        st.warning("No nodes found in the graph yet. Run a deep search or build a roadmap first.")
        return

    # ── Colour map per node type ──────────────────────────────────────────────
    colours = {
        "Topic":    "#FF6B6B",
        "Module":   "#4ECDC4",
        "Concept":  "#FFE66D",
        "Resource": "#A8E6CF",
        "LevelGroup": "#C3B1E1",
    }
    sizes = {
        "Topic": 30, "Module": 20, "Concept": 15,
        "Resource": 12, "LevelGroup": 18,
    }

    if _AGRAPH_AVAILABLE:
        ag_nodes = [
            Node(  # type: ignore[operator]
                id=n["id"],
                label=n["label"][:25],
                title=n.get("title", n["label"]),
                color=colours.get(n["type"], "#CCCCCC"),
                size=sizes.get(n["type"], 15),
            )
            for n in nodes_raw
        ]
        ag_edges = [
            Edge(  # type: ignore[operator]
                source=e["source"],
                target=e["target"],
                label=e.get("label", ""),
            )
            for e in edges_raw
        ]
        config = Config(  # type: ignore[operator]
            width="100%",
            height=620,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
        )
        agraph(nodes=ag_nodes, edges=ag_edges, config=config)  # type: ignore[operator]
    else:
        # Fallback: plain table view when agraph not installed
        st.warning("streamlit-agraph not available — showing table view.")
        tab_n, tab_e = st.tabs(["Nodes", "Edges"])
        with tab_n:
            for ntype in ["Topic", "Module", "Concept", "Resource"]:
                subset = [n for n in nodes_raw if n["type"] == ntype]
                if subset:
                    st.subheader(f"{colours.get(ntype,'⚪')} {ntype}s ({len(subset)})")
                    for n in subset:
                        st.markdown(f"- **{n['label']}**")
        with tab_e:
            for e in edges_raw:
                st.markdown(f"- `{e['source']}` —[{e.get('label','')}]→ `{e['target']}`")

    st.divider()
    if st.button("← Back", use_container_width=False):
        go(st.session_state.get("_graph_prev", "topic"))


# ── Router ────────────────────────────────────────────────────────────────────

_sidebar()

page = st.session_state["page"]

if   page == "auth":        page_auth()
elif page == "topic":       page_topic()
elif page == "searching":   page_searching()
elif page == "quiz":        page_quiz()
elif page == "gaps":        page_gaps()
elif page == "roadmap":     page_roadmap()
elif page == "lesson":      page_lesson()
elif page == "quiz_module": page_quiz_module()
elif page == "quiz_result": page_quiz_result()
elif page == "complete":    page_complete()
elif page == "graph":       page_graph()
elif page == "progress":    page_progress()
else:
    go("auth")
