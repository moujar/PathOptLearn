import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import get_courses, create_course, delete_course, get_course_stats, get_course_progress

st.set_page_config(page_title="Dashboard — PathOptLearn", page_icon="🎓", layout="wide")

# ── AUTH GUARD ───────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.switch_page("pages/1_Login.py")

student_id = st.session_state["student_id"]
username   = st.session_state.get("username", "Student")
is_guest   = student_id == "guest"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {username}")
    st.markdown("---")
    if st.button("🧠 LearnFlow", use_container_width=True, type="primary"):
        st.switch_page("pages/4_LearnFlow.py")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("pages/1_Login.py")

# ── HEADER ───────────────────────────────────────────────────
st.title("📊 My Dashboard")
st.markdown(f"Welcome back, **{username}**!")

col_a, col_b = st.columns(2)
with col_a:
    st.info("**📚 Video Courses** — watch a YouTube video, take an AI-generated quiz, get the next video recommended.")
with col_b:
    st.info("**🧠 LearnFlow** — enter any topic, get a diagnostic quiz, personalized roadmap, and guided modules.")

st.markdown("---")

# ── CREATE NEW COURSE ────────────────────────────────────────
if is_guest:
    st.info("💡 You are browsing as a guest. **LearnFlow** is fully available. [Create an account](pages/1_Login.py) to save course progress.")
else:
    with st.expander("➕ Create New Video Course", expanded=False):
        with st.form("new_course_form"):
            course_name = st.text_input("Course name", placeholder="e.g. Calculus, Machine Learning, Python...")
            submitted   = st.form_submit_button("Create Course", type="primary")
        if submitted:
            if not course_name.strip():
                st.error("Please enter a course name.")
            else:
                course_id = create_course(student_id, course_name)
                st.success(f"Course '{course_name}' created!")
                st.session_state["course_id"]   = course_id
                st.session_state["course_name"] = course_name.strip()
                for key in ["transcript", "transcript_segments", "video_id", "video_title",
                            "questions", "eval_result", "recommendation", "recommended_video_id"]:
                    st.session_state.pop(key, None)
                st.switch_page("pages/3_Learning.py")

# ── COURSES LIST ─────────────────────────────────────────────
courses = [] if is_guest else get_courses(student_id)

if not courses:
    st.info("You have no video courses yet. Create one above, or try **LearnFlow** from the sidebar.")
else:
    st.markdown(f"**{len(courses)} course(s)**")
    st.markdown("")

    for course_id, course_name, created_at in courses:
        stats = get_course_stats(student_id, course_id)

        with st.container():
            col_info, col_stats, col_actions = st.columns([3, 4, 2])

            with col_info:
                st.markdown(f"### 📚 {course_name}")
                st.caption(f"Created: {created_at[:10]}")
                if stats["last_activity"]:
                    st.caption(f"Last activity: {stats['last_activity']}")

            with col_stats:
                if stats["videos_watched"] == 0:
                    st.info("No videos watched yet in this course.")
                else:
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Videos", stats["videos_watched"])
                    with m2:
                        st.metric("Avg Score", f"{stats['avg_score']}%")
                    with m3:
                        st.metric("Pass Rate", f"{stats['pass_rate']}%")

                    st.progress(stats["pass_rate"] / 100)

                    with st.expander("📋 Video history"):
                        rows = get_course_progress(student_id, course_id)
                        for i, (vid_id, title, score, total, passed, timestamp) in enumerate(rows, 1):
                            pct    = int(score / total * 100) if total > 0 else 0
                            status = "✅" if passed else "🔄"
                            st.markdown(
                                f"{status} **{i}. {title[:45]}{'...' if len(title) > 45 else ''}** "
                                f"— {score}/{total} ({pct}%) — {timestamp}"
                            )

            with col_actions:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("▶ Continue", key=f"cont_{course_id}", type="primary", use_container_width=True):
                    st.session_state["course_id"]   = course_id
                    st.session_state["course_name"] = course_name
                    for key in ["transcript", "transcript_segments", "video_id", "video_title",
                                "questions", "eval_result", "recommendation", "recommended_video_id"]:
                        st.session_state.pop(key, None)
                    st.switch_page("pages/3_Learning.py")

                if st.button("🗑 Delete", key=f"del_{course_id}", use_container_width=True):
                    delete_course(course_id, student_id)
                    st.rerun()

            st.markdown("---")
