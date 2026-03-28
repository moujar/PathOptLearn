import json
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepTutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

def _init():
    defaults = {
        "page":            "🎓 Learn",
        "step":            0,           # learning flow step
        "topic":           "",
        "session_id":      None,
        "level":           "beginner",
        "level_emoji":     "🟢",
        "roadmap":         None,
        "diag_questions":  [],
        "diag_answers":    {},
        "current_lesson":  None,
        "current_quiz":    [],
        "quiz_answers":    {},
        "completed_uids":  [],
        "last_score":      None,
        # user management
        "user_id":         None,
        "user":            None,        # full user dict from API
        "show_register":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def stream_get(path: str, params: dict = None):
    """Generator that yields text chunks from a streaming GET endpoint."""
    try:
        with requests.get(f"{API}{path}", params=params, stream=True, timeout=300) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"\n\n⚠️ Stream error: {e}"


def score_badge(score: float) -> str:
    if score >= 70:
        return f"🟢 {score:.0f}%"
    if score >= 40:
        return f"🟡 {score:.0f}%"
    return f"🔴 {score:.0f}%"


def level_badge(level: str) -> str:
    return {"beginner": "🟢 Beginner", "intermediate": "🟡 Intermediate",
            "advanced": "🔴 Advanced"}.get(level, level)


def _render_history_table(user_id: int):
    """Render a full learning history table for a user (sessions + per-module rows)."""
    import pandas as pd

    hdata = api_get(f"/users/{user_id}/history")
    if not hdata or not hdata["history"]:
        st.info("No learning history yet.")
        return

    history = hdata["history"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_modules = sum(s["modules_completed"] for s in history)
    all_scores    = [s["avg_score"] for s in history if s["avg_score"] is not None]
    overall_avg   = round(sum(all_scores) / len(all_scores), 1) if all_scores else None

    m1, m2, m3 = st.columns(3)
    m1.metric("Sessions",         hdata["total_sessions"])
    m2.metric("Modules completed", total_modules)
    m3.metric("Average quiz score", f"{overall_avg}%" if overall_avg else "—")
    st.divider()

    # ── Per-session expandable tables ─────────────────────────────────────────
    for sess in history:
        avg  = f"{sess['avg_score']}%" if sess["avg_score"] is not None else "—"
        best = f"{sess['best_score']}%" if sess["best_score"] is not None else "—"
        label = (
            f"{'🎓' if sess['modules_completed'] > 0 else '📝'} "
            f"**{sess['topic']}** · {sess['level_emoji']} {sess['level'].capitalize()} · "
            f"✅ {sess['modules_completed']} modules · avg {avg} · {sess['started_at'][:10]}"
        )
        with st.expander(label, expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Modules done", sess["modules_completed"])
            sc2.metric("Avg score",    avg)
            sc3.metric("Best score",   best)

            if sess["modules"]:
                rows = []
                for m in sess["modules"]:
                    score = m["quiz_score"]
                    badge = ("🟢" if score >= 70 else "🟡" if score >= 40 else "🔴") if score is not None else "—"
                    rows.append({
                        "Step":      int(m["step"]),
                        "Module":    m["module_title"] or m["module_uid"],
                        "Score":     f"{badge} {score:.0f}%" if score is not None else "—",
                        "Completed": m["completed_at"][:16] if m["completed_at"] else "—",
                        "_score_val": score if score is not None else 0,
                    })
                df = pd.DataFrame(rows)
                st.dataframe(
                    df[["Step", "Module", "Score", "Completed"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Step":      st.column_config.NumberColumn("Step", width="small"),
                        "Module":    st.column_config.TextColumn("Module"),
                        "Score":     st.column_config.TextColumn("Quiz score"),
                        "Completed": st.column_config.TextColumn("Completed at"),
                    },
                )
                score_vals = [m["quiz_score"] for m in sess["modules"] if m["quiz_score"] is not None]
                if score_vals:
                    st.markdown("**Score progression**")
                    bar_df = pd.DataFrame({
                        "Module": [m["module_title"] or m["module_uid"]
                                   for m in sess["modules"] if m["quiz_score"] is not None],
                        "Score":  score_vals,
                    })
                    st.bar_chart(bar_df.set_index("Module")["Score"], height=180)
            else:
                st.info("No modules completed in this session yet.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎓 DeepTutor")
    st.caption("AI-powered personalised learning")
    st.divider()

    # ── User login panel ──────────────────────────────────────────────────────
    if st.session_state.user:
        u = st.session_state.user
        st.success(f"👤 **{u['username']}**")
        st.caption(f"{u['email']}  ·  {level_badge(u['level'])}")
        if st.button("Logout", key="logout_btn"):
            st.session_state.user    = None
            st.session_state.user_id = None
            st.session_state.pop("cached_users", None)
            st.rerun()
    else:
        if not st.session_state.show_register:
            st.markdown("**Login**")
            # Cache users list so the sidebar doesn't hit the API on every render
            if "cached_users" not in st.session_state:
                users_data = api_get("/users", {"limit": 100})
                st.session_state.cached_users = users_data["users"] if users_data else []
            users_list = st.session_state.cached_users
            if users_list:
                names = [u["username"] for u in users_list]
                chosen = st.selectbox("Select user", ["— choose —"] + names,
                                      key="login_select", label_visibility="collapsed")
                if st.button("Login →", key="login_btn") and chosen != "— choose —":
                    u = next((x for x in users_list if x["username"] == chosen), None)
                    if u:
                        st.session_state.user    = u
                        st.session_state.user_id = u["id"]
                        st.rerun()
            else:
                st.caption("No users yet.")
            if st.button("Create account", key="goto_register"):
                st.session_state.show_register = True
                st.rerun()
        else:
            st.markdown("**Create account**")
            reg_username  = st.text_input("Username",  key="reg_username")
            reg_email     = st.text_input("Email",     key="reg_email")
            reg_fullname  = st.text_input("Full name (optional)", key="reg_fullname")
            reg_level     = st.selectbox("Level", ["beginner", "intermediate", "advanced"],
                                         key="reg_level")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Register →", key="reg_btn",
                             disabled=not (reg_username.strip() and reg_email.strip())):
                    new_user = api_post("/users", {
                        "username":  reg_username.strip(),
                        "email":     reg_email.strip(),
                        "full_name": reg_fullname.strip() or None,
                        "level":     reg_level,
                    })
                    if new_user:
                        st.session_state.user             = new_user
                        st.session_state.user_id          = new_user["id"]
                        st.session_state.show_register    = False
                        st.session_state.pop("cached_users", None)
                        st.rerun()
            with c2:
                if st.button("Cancel", key="reg_cancel"):
                    st.session_state.show_register = False
                    st.rerun()
    st.divider()

    PAGES = ["🎓 Learn", "🔍 Deep Search", "🌐 Quick Search",
             "📺 YouTube", "🔬 Full Research", "📂 History", "👤 Users"]
    page = st.radio(
        "Navigate",
        PAGES,
        index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
        label_visibility="collapsed",
    )
    st.session_state.page = page
    st.divider()

    # Learning session status
    if st.session_state.session_id:
        st.success(f"**Session #{st.session_state.session_id}**")
        st.write(f"📖 {st.session_state.topic}")
        st.write(level_badge(st.session_state.level))
        completed = len(st.session_state.completed_uids)
        total = len(st.session_state.roadmap["modules"]) if st.session_state.roadmap else "?"
        st.write(f"✅ {completed} / {total} modules done")
        if st.button("🔄 Reset session"):
            for k in ["step", "topic", "session_id", "level", "level_emoji",
                      "roadmap", "diag_questions", "diag_answers",
                      "current_lesson", "current_quiz", "quiz_answers",
                      "completed_uids", "last_score"]:
                st.session_state[k] = _init.__defaults__  # reset via defaults
            st.session_state.step       = 0
            st.session_state.session_id = None
            st.session_state.roadmap    = None
            st.session_state.completed_uids = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🎓 LEARN  (main flow)
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎓 Learn":

    # ── Progress bar ──────────────────────────────────────────────────────────
    steps  = ["Topic", "Assessment", "Results", "Roadmap", "Lesson", "Quiz", "Next"]
    step   = st.session_state.step
    cols   = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            if i < step:
                st.markdown(f"✅ ~~{label}~~")
            elif i == step:
                st.markdown(f"**➡️ {label}**")
            else:
                st.markdown(f"⬜ {label}")
    st.divider()

    # ── STEP 0 — Enter topic ──────────────────────────────────────────────────
    if step == 0:
        st.header("What do you want to learn?")
        topic = st.text_input(
            "Enter a subject",
            placeholder="e.g. Machine Learning, Electromagnetism, Calculus, Organic Chemistry…",
            value=st.session_state.topic,
        )
        if st.button("Start learning →", type="primary", disabled=not topic.strip()):
            st.session_state.topic = topic.strip()
            with st.spinner("Generating diagnostic questions…"):
                data = api_get("/assess", {"topic": topic.strip()})
            if data:
                st.session_state.diag_questions = data["questions"]
                st.session_state.diag_answers   = {}
                st.session_state.step           = 1
                st.rerun()

    # ── STEP 1 — Diagnostic quiz ──────────────────────────────────────────────
    elif step == 1:
        st.header(f"📋 Diagnostic Assessment — {st.session_state.topic}")
        st.caption("Answer all questions so we can personalise your learning path.")

        questions = st.session_state.diag_questions
        with st.form("diagnostic_form"):
            for q in questions:
                lvl_icon = {"basic": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(q["level"], "")
                st.markdown(f"**Q{q['id']}** {lvl_icon} — {q['question']}")
                ans = st.radio(
                    f"q_{q['id']}",
                    options=q["options"],
                    index=None,
                    label_visibility="collapsed",
                    key=f"diag_{q['id']}",
                )
                # Store just the letter
                if ans:
                    st.session_state.diag_answers[q["id"]] = ans[0]
                st.write("")

            submitted = st.form_submit_button("Submit answers →", type="primary")

        if submitted:
            answers = [st.session_state.diag_answers.get(q["id"], "A") for q in questions]
            with st.spinner("Evaluating your level…"):
                result = api_post("/assess/evaluate", {
                    "topic":     st.session_state.topic,
                    "questions": questions,
                    "answers":   answers,
                })
            if result:
                st.session_state.level       = result["level"]
                st.session_state.level_emoji = result["level_emoji"]
                st.session_state._assess_result = result
                st.session_state.step        = 2
                st.rerun()

    # ── STEP 2 — Assessment results + create session ──────────────────────────
    elif step == 2:
        result = getattr(st.session_state, "_assess_result", {})
        st.header("📊 Your Assessment Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Score", f"{result.get('score', 0):.0f} / 100")
        col2.metric("Level", level_badge(result.get("level", "beginner")))
        col3.metric("Topic", st.session_state.topic)

        st.info(f"💬 {result.get('feedback', '')}")
        st.write("")

        if st.button("Build my learning roadmap →", type="primary"):
            with st.spinner("Creating learning session…"):
                uid_param = (f"&user_id={st.session_state.user_id}"
                             if st.session_state.user_id else "")
                sess = api_post(
                    f"/session/start?topic={st.session_state.topic}"
                    f"&level={st.session_state.level}"
                    f"&level_emoji={st.session_state.level_emoji}{uid_param}",
                    {},
                )
            if sess:
                st.session_state.session_id = sess["session_id"]
                # Use the cleaned topic the backend stored in the session
                if sess.get("topic"):
                    st.session_state.topic = sess["topic"]
                st.session_state.step = 3
                st.rerun()

    # ── STEP 3 — Build roadmap ────────────────────────────────────────────────
    elif step == 3:
        st.header(f"🗺️ Learning Roadmap — {st.session_state.topic}")

        if st.session_state.roadmap is None:
            with st.spinner("Building your personalised multi-level roadmap…"):
                data = api_get("/roadmap", {
                    "topic": st.session_state.topic,
                    "level": st.session_state.level,
                })
            if data:
                # Update topic to the cleaned version returned by the API
                if data.get("topic"):
                    st.session_state.topic = data["topic"]
                st.session_state.roadmap = data

        rm = st.session_state.roadmap
        if rm:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Your level",   f"{rm.get('level_emoji','')} {rm.get('level','').capitalize()}")
            col_b.metric("Total modules", rm.get("total_modules", len(rm.get("modules", []))))
            col_c.metric("Total time",    rm.get("total_duration", "—"))

            gaps_str = " · ".join(str(g) for g in rm.get("gaps", []))
            if gaps_str:
                st.caption(f"📌 Topics covered: {gaps_str}")
            st.divider()

            # ── Multi-level display ────────────────────────────────────────────
            levels = rm.get("levels", [])
            if levels:
                # Show each level as a tab
                tab_labels = [f"{lg['emoji']} Level {lg['level_num']}: {lg['level_name']}" for lg in levels]
                tabs = st.tabs(tab_labels)
                for tab, lg in zip(tabs, levels):
                    with tab:
                        mods = lg["modules"]
                        st.caption(f"{len(mods)} modules · {sum(m.get('duration_minutes',60) for m in mods)} min total")
                        for mod in mods:
                            uid = f"{st.session_state.topic}::{mod['id']}"
                            is_done = uid in set(st.session_state.completed_uids)
                            status  = "✅" if is_done else "⬜"
                            prereqs = mod.get("prerequisites", [])
                            prereq_str = f" · requires #{', #'.join(map(str, prereqs))}" if prereqs else ""
                            with st.container(border=True):
                                c1, c2 = st.columns([5, 1])
                                with c1:
                                    st.markdown(f"{status} **{mod['id']}.** {mod['title']}")
                                    st.caption(f"🎯 {mod['objective']}{prereq_str}")
                                    concepts = mod.get("concepts", [])
                                    if concepts:
                                        st.markdown(" ".join(f"`{c}`" for c in concepts))
                                with c2:
                                    st.metric("", f"{mod.get('duration_minutes', 60)} min")
            else:
                # Fallback: flat module list (old format)
                for mod in rm.get("modules", []):
                    prereqs = f" *(requires: {', '.join(map(str, mod['prerequisites']))})*" if mod.get("prerequisites") else ""
                    st.markdown(
                        f"**{mod['id']}.** {mod['title']}{prereqs}  \n"
                        f"_{mod['objective']}_  \n"
                        f"Concepts: `{'`, `'.join(mod.get('concepts',[]))}`  · ⏱ {mod.get('duration_minutes',60)} min"
                    )

            st.divider()
            if st.button("Start first lesson →", type="primary"):
                st.session_state.step = 4
                st.rerun()

    # ── STEP 4 — Lesson ───────────────────────────────────────────────────────
    elif step == 4:
        rm = st.session_state.roadmap
        if not rm:
            st.warning("No roadmap found. Go back to step 3.")
            st.stop()

        # Find next module not yet completed
        completed = set(st.session_state.completed_uids)
        next_mod  = next(
            (m for m in rm["modules"] if f"{st.session_state.topic}::{m['id']}" not in completed),
            None,
        )
        if next_mod is None:
            st.success("🎉 You've completed all modules!")
            st.stop()

        uid = f"{st.session_state.topic}::{next_mod['id']}"
        st.header(f"📚 Lesson {next_mod['id']}: {next_mod['title']}")
        st.caption(f"Objective: {next_mod['objective']}")

        if (st.session_state.current_lesson is None
                or st.session_state.current_lesson.get("module_uid") != uid):
            with st.spinner("Deep-searching and generating your lesson…"):
                data = api_get("/lesson", {
                    "topic":      st.session_state.topic,
                    "module_id":  next_mod["id"],
                    "session_id": st.session_state.session_id,
                })
            if data:
                st.session_state.current_lesson = data
                st.session_state.current_quiz   = []
                st.session_state.quiz_answers   = {}

        lesson = st.session_state.current_lesson
        if lesson:
            # ── 1. Recommended resources (before content) ─────────────────
            recommended = [r for r in lesson.get("recommended", []) if isinstance(r, dict)]
            if recommended:
                st.subheader("🌟 Best Resources for This Lesson")
                st.caption("Curated by AI — start here before diving into the lesson.")
                for i, r in enumerate(recommended):
                    icon = "📺" if r.get("type") == "video" else "📖"
                    with st.container(border=True):
                        col1, col2 = st.columns([6, 1])
                        with col1:
                            st.markdown(f"**{icon} [{r['title']}]({r['url']})**")
                            st.caption(f"💡 {r.get('reason', '')}")
                        with col2:
                            st.link_button("Open →", r["url"])
                st.divider()

            # ── 2. All sources (expandable) ───────────────────────────────
            sources = [s for s in lesson.get("sources", []) if isinstance(s, dict)]
            videos  = [v for v in lesson.get("videos",  []) if isinstance(v, dict)]
            with st.expander(f"📚 All Sources ({len(sources)} articles · {len(videos)} videos)"):
                if sources:
                    st.markdown("**📄 Web Articles**")
                    for s in sources:
                        st.markdown(f"- **[{s['title']}]({s['url']})**")
                        if s.get("snippet"):
                            st.caption(s["snippet"][:120] + "…")
                if videos:
                    st.markdown("**📺 Videos**")
                    cols = st.columns(min(3, len(videos)))
                    for i, v in enumerate(videos):
                        with cols[i % 3]:
                            st.image(v["thumb"], use_container_width=True)
                            st.markdown(f"**[{v['title'][:45]}]({v['url']})**")
                            st.caption(f"{v['channel']} · ⏱ {v['duration']}")
            st.divider()

            # ── 3. Lesson content ──────────────────────────────────────────
            st.subheader("📖 Lesson Content")
            st.markdown(lesson["content"])
            st.divider()

            # ── 4. Further reading footer ──────────────────────────────────
            if videos:
                st.subheader("📺 Watch to Reinforce")
                for v in videos[:3]:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.image(v["thumb"], use_container_width=True)
                    with col2:
                        st.markdown(f"**[{v['title']}]({v['url']})**")
                        st.caption(f"{v['channel']} · ⏱ {v['duration']}")
                st.divider()

            if st.button("I'm ready — take the quiz →", type="primary"):
                st.session_state.step = 5
                st.rerun()

    # ── STEP 5 — Quiz ─────────────────────────────────────────────────────────
    elif step == 5:
        lesson = st.session_state.current_lesson
        if not lesson:
            st.warning("No lesson found. Go back.")
            st.stop()

        st.header(f"✅ Quiz — {lesson.get('module_title', '')}")

        # Generate quiz if not yet done
        if not st.session_state.current_quiz:
            with st.spinner("Generating quiz questions…"):
                data = api_post("/quiz", {
                    "content":       lesson["content"],
                    "num_questions": 5,
                })
            if data:
                st.session_state.current_quiz = data["questions"]
                st.session_state.quiz_answers = {}

        questions = st.session_state.current_quiz
        if not questions:
            st.error("Could not generate quiz. Try again.")
            st.stop()

        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}.** {q['question']}")
                ans = st.radio(
                    f"q_{i}",
                    options=q["options"],
                    index=None,
                    label_visibility="collapsed",
                    key=f"quiz_{i}",
                )
                if ans:
                    st.session_state.quiz_answers[i] = ans[0]
                st.write("")
            submitted = st.form_submit_button("Submit quiz →", type="primary")

        if submitted:
            # Score locally
            correct = sum(
                1 for i, q in enumerate(questions)
                if st.session_state.quiz_answers.get(i, "").upper() == q["answer"].upper()
            )
            score = round(correct / len(questions) * 100, 1)
            st.session_state.last_score = score

            # Show per-question feedback
            st.divider()
            for i, q in enumerate(questions):
                user = st.session_state.quiz_answers.get(i, "?").upper()
                correct_ans = q["answer"].upper()
                ok = user == correct_ans
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **Q{i+1}:** {q['question']}")
                if not ok:
                    st.caption(f"Your answer: {user} · Correct: {correct_ans}")
                st.caption(f"💡 {q.get('explanation','')}")

            st.divider()
            st.metric("Your score", score_badge(score))

            # ── Post-quiz suggestion based on score ───────────────────────────
            PASS_THRESHOLD = 70

            if score >= PASS_THRESHOLD:
                st.success(
                    f"🎉 Great job! You scored {score:.0f}% — you've mastered this module."
                )

                # ── Preview the next module from the roadmap ──────────────────
                rm = st.session_state.roadmap
                current_uid = st.session_state.current_lesson.get("module_uid", "")
                next_mod = None
                if rm:
                    completed = set(st.session_state.completed_uids) | {current_uid}
                    next_mod = next(
                        (m for m in rm["modules"]
                         if f"{st.session_state.topic}::{m['id']}" not in completed),
                        None,
                    )

                if next_mod:
                    st.divider()
                    st.subheader("⏭️ Up next")
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### {next_mod['id']}. {next_mod['title']}")
                            st.caption(f"🎯 {next_mod['objective']}")
                            concepts = next_mod.get("concepts", [])
                            if concepts:
                                st.markdown(
                                    "**Concepts covered:** " +
                                    " · ".join(f"`{c}`" for c in concepts)
                                )
                            prereqs = next_mod.get("prerequisites", [])
                            if prereqs:
                                st.caption(f"Requires: {', '.join(map(str, prereqs))}")
                        with c2:
                            st.metric("Duration", f"{next_mod.get('duration_minutes', '?')} min")
                        if st.button(f"Start: {next_mod['title']} →", type="primary"):
                            st.session_state.step = 6
                            st.rerun()
                else:
                    st.info("🏆 This is the last module — completing it will finish the roadmap!")
                    if st.button("Finish course →", type="primary"):
                        st.session_state.step = 6
                        st.rerun()
            else:
                missed = [
                    q for i, q in enumerate(questions)
                    if st.session_state.quiz_answers.get(i, "").upper() != q["answer"].upper()
                ]
                st.warning(
                    f"📚 You scored {score:.0f}% — a bit more practice will help. "
                    f"Review the lesson below and try again, or continue if you feel ready."
                )

                # Show what to focus on
                if missed:
                    with st.expander("🔍 Topics to review", expanded=True):
                        st.markdown("Focus on these areas before retrying:")
                        for q in missed:
                            st.markdown(f"- **{q['question']}**")
                            st.caption(f"  💡 {q.get('explanation', '')}")

                # Show the lesson content again inline for quick review
                with st.expander("📖 Review the lesson", expanded=False):
                    lesson = st.session_state.current_lesson
                    if lesson:
                        st.markdown(lesson["content"])
                        rec_list = [r for r in lesson.get("recommended", []) if isinstance(r, dict)]
                        if rec_list:
                            st.markdown("**Recommended resources:**")
                            for r in rec_list[:3]:
                                icon = "📺" if r.get("type") == "video" else "📖"
                                st.markdown(f"- {icon} [{r['title']}]({r['url']})")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry this quiz", type="primary"):
                        # Reset quiz answers and regenerate
                        st.session_state.current_quiz = []
                        st.session_state.quiz_answers = {}
                        st.rerun()
                with col2:
                    if st.button("⏭️ Continue anyway →"):
                        st.session_state.step = 6
                        st.rerun()

    # ── STEP 6 — Next (loop engine) ───────────────────────────────────────────
    elif step == 6:
        lesson = st.session_state.current_lesson
        score  = st.session_state.last_score or 0

        st.header("⏭️ Advancing…")
        with st.spinner("Saving progress and loading next lesson…"):
            result = api_post("/next", {
                "session_id":           st.session_state.session_id,
                "completed_module_uid": lesson["module_uid"],
                "quiz_score":           score,
                "num_quiz_questions":   5,
            })

        if result:
            # Track completed
            st.session_state.completed_uids.append(lesson["module_uid"])

            if result.get("completed"):
                st.balloons()
                st.success(result["message"])
                st.metric("Average score across all modules", score_badge(result["avg_score"]))
                st.session_state.step = 4   # will show "all done"
            else:
                nm = result["next_module"]
                # Pre-load next lesson into session state
                st.session_state.current_lesson = {
                    "module_uid":   nm["uid"],
                    "module_title": nm["title"],
                    "objective":    nm["objective"],
                    "concepts":     nm["concepts"],
                    "content":      nm["content"],
                    "lesson_id":    nm["lesson_id"],
                    "sources":      nm.get("sources", []),
                    "videos":       nm.get("videos",  []),
                    "recommended":  nm.get("recommended", []),
                }
                st.session_state.current_quiz = result["quiz"]["questions"]
                st.session_state.quiz_answers = {}

                remaining = result.get("remaining_modules")
                st.success(f"✅ Module completed with {score_badge(score)}")
                remaining_str = f" · {remaining} module(s) remaining" if remaining is not None else ""
                st.info(f"Next: **{nm['title']}**{remaining_str}")
                if st.button(f"Start: {nm['title']} →", type="primary"):
                    st.session_state.step = 4
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🔍 DEEP SEARCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Deep Search":
    st.header("🔍 Deep Search")
    st.caption("Full pipeline: sub-queries → scrape → rank → educational lesson (streaming)")

    topic = st.text_input("Topic", placeholder="e.g. quantum computing, photosynthesis, derivatives…")
    if st.button("Search", type="primary", disabled=not topic.strip()):
        st.divider()
        st.write_stream(stream_get("/deepSearch", {"topic": topic.strip()}))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🌐 QUICK SEARCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌐 Quick Search":
    st.header("🌐 Quick Search")
    st.caption("Fast DuckDuckGo snippet search — no page scraping (streaming)")

    topic = st.text_input("Topic", placeholder="e.g. latest AI models, Newton's laws…")
    if st.button("Search", type="primary", disabled=not topic.strip()):
        st.divider()
        st.write_stream(stream_get("/deepSearchWebsite", {"topic": topic.strip()}))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📺 YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📺 YouTube":
    st.header("📺 YouTube Search")
    st.caption("Search YouTube for educational videos on any subject")

    topic = st.text_input("Search query", placeholder="e.g. machine learning tutorial for beginners")
    if st.button("Search", type="primary", disabled=not topic.strip()):
        data = api_get("/deepSearchYoutube", {"topic": topic.strip()})
        if data:
            st.caption(f"🔎 Query sent to YouTube: `{data.get('query', topic)}`")
            st.write(f"Found **{data['count']}** videos")
            st.divider()
            for v in data["videos"]:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(v["thumb"], use_container_width=True)
                with col2:
                    st.markdown(f"**[{v['title']}]({v['url']})**")
                    st.caption(
                        f"📺 {v['channel']} · ⏱ {v['duration']} · "
                        f"👁 {v['views']:,}" if v["views"] else f"📺 {v['channel']} · ⏱ {v['duration']}"
                    )
                    if v.get("desc"):
                        st.write(v["desc"][:150] + "…")
                st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 🔬 FULL RESEARCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Full Research":
    st.header("🔬 Full Research")
    st.caption("Web summary + YouTube videos in one shot")

    topic = st.text_input("Topic", placeholder="e.g. large language models 2025")
    if st.button("Research", type="primary", disabled=not topic.strip()):
        st.divider()
        videos_shown = False
        full_text    = ""

        with requests.get(f"{API}/fullResearch", params={"topic": topic.strip()},
                          stream=True, timeout=300) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                if not videos_shown and "\n---\n" in chunk:
                    json_part, rest = chunk.split("\n---\n", 1)
                    try:
                        videos = json.loads(json_part)["videos"]
                        st.subheader("📺 Related Videos")
                        for v in videos:
                            st.markdown(f"- **[{v['title']}]({v['url']})** — {v['channel']} ⏱ {v['duration']}")
                        st.divider()
                        st.subheader("📝 Web Summary")
                    except Exception:
                        rest = chunk
                    videos_shown = True
                    full_text += rest
                else:
                    full_text += chunk

        if full_text:
            st.markdown(full_text)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 📂 HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 History":
    st.header("📂 Search History")

    col1, col2 = st.columns([3, 1])
    with col2:
        limit = st.number_input("Max rows", min_value=1, max_value=100, value=20)
    with col1:
        st.write("")

    data = api_get("/showDB", {"limit": limit})
    if data and data["searches"]:
        searches = data["searches"]
        st.write(f"**{len(searches)} searches stored**")
        st.divider()
        for s in searches:
            with st.expander(f"#{s['id']} — {s['topic']} · `{s['search_type']}`"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Web results",   s["web_count"])
                col2.metric("YouTube",       s["yt_count"])
                col3.metric("Reports",       s["report_count"])
                col4.metric("Created",       str(s["created_at"])[:16])

                brow1, brow2 = st.columns(2)
                with brow1:
                    if s["report_count"] > 0 and st.button("📄 View report", key=f"rep_{s['id']}"):
                        rep = api_get(f"/report/{s['id']}")
                        if rep:
                            st.markdown(rep["content"])
                with brow2:
                    if s["yt_count"] > 0 and st.button("📺 View videos", key=f"yt_{s['id']}"):
                        vdata = api_get(f"/videos/{s['id']}")
                        if vdata:
                            for v in vdata["videos"]:
                                st.markdown(f"- [{v['title']}]({v['url']}) — {v['channel']}")
    else:
        st.info("No searches stored yet. Run a search first.")


elif page == "👤 Users":
    st.header("👤 User Management")

    tab_list, tab_create, tab_profile = st.tabs(["All Users", "Create User", "My Profile"])

    # ── Tab: All Users ────────────────────────────────────────────────────────
    with tab_list:
        data = api_get("/users", {"limit": 100})
        if data and data["users"]:
            users = data["users"]
            st.write(f"**{len(users)} users registered**")
            st.divider()
            for u in users:
                is_me = st.session_state.user_id == u["id"]
                label = f"{'⭐ ' if is_me else ''}**{u['username']}** — {u['email']}"
                with st.expander(label):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Level",      level_badge(u["level"]))
                    c2.metric("Full name",  u["full_name"] or "—")
                    c3.metric("Joined",     str(u["created_at"])[:10])

                    # Learning history for this user
                    st.markdown("**Learning History**")
                    _render_history_table(u["id"])

                    # Delete button (only shown to the logged-in user for their own account, or always for admin feel)
                    if st.button("🗑️ Delete user", key=f"del_{u['id']}",
                                 help="Permanently delete this user"):
                        try:
                            r = requests.delete(f"{API}/users/{u['id']}", timeout=30)
                            r.raise_for_status()
                            st.success(f"User {u['username']} deleted.")
                            if st.session_state.user_id == u["id"]:
                                st.session_state.user    = None
                                st.session_state.user_id = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No users yet. Create one in the 'Create User' tab.")

    # ── Tab: Create User ──────────────────────────────────────────────────────
    with tab_create:
        st.subheader("Register a new user")
        with st.form("create_user_form"):
            new_username  = st.text_input("Username *")
            new_email     = st.text_input("Email *")
            new_fullname  = st.text_input("Full name (optional)")
            new_level     = st.selectbox("Starting level", ["beginner", "intermediate", "advanced"])
            submitted = st.form_submit_button("Create user →", type="primary")

        if submitted:
            if not new_username.strip() or not new_email.strip():
                st.error("Username and email are required.")
            else:
                result = api_post("/users", {
                    "username":  new_username.strip(),
                    "email":     new_email.strip(),
                    "full_name": new_fullname.strip() or None,
                    "level":     new_level,
                })
                if result:
                    st.success(f"User **{result['username']}** created (ID: {result['id']}).")
                    if st.button("Login as this user"):
                        st.session_state.user    = result
                        st.session_state.user_id = result["id"]
                        st.rerun()

    # ── Tab: My Profile ───────────────────────────────────────────────────────
    with tab_profile:
        if not st.session_state.user:
            st.info("Login from the sidebar to see your profile.")
        else:
            u = st.session_state.user
            st.subheader(f"Profile — {u['username']}")

            # Edit form
            with st.form("edit_profile_form"):
                upd_fullname = st.text_input("Full name",  value=u.get("full_name") or "")
                upd_email    = st.text_input("Email",      value=u["email"])
                upd_level    = st.selectbox(
                    "Level",
                    ["beginner", "intermediate", "advanced"],
                    index=["beginner", "intermediate", "advanced"].index(u["level"]),
                )
                save = st.form_submit_button("Save changes →", type="primary")

            if save:
                patch = {}
                if upd_fullname.strip() != (u.get("full_name") or ""):
                    patch["full_name"] = upd_fullname.strip() or None
                if upd_email.strip() != u["email"]:
                    patch["email"] = upd_email.strip()
                if upd_level != u["level"]:
                    patch["level"] = upd_level
                if patch:
                    try:
                        r = requests.put(f"{API}/users/{u['id']}", json=patch, timeout=30)
                        r.raise_for_status()
                        updated = r.json()
                        st.session_state.user  = updated
                        st.session_state.user_id = updated["id"]
                        st.success("Profile updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info("Nothing changed.")

            st.divider()
            st.subheader("Learning History")
            _render_history_table(u["id"])
