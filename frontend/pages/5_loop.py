"""Page 5 — Adaptive Learning Loop & Progress Tracking."""
import streamlit as st
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="Loop — AdaptLearn AI", layout="wide")
st.title("🔄 Stage 6: Adaptive Learning Loop")

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


def api_get(endpoint: str) -> dict | None:
    try:
        resp = requests.get(f"{API}{endpoint}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


if not st.session_state.get("learning_path"):
    st.warning("⚠️ Please complete Stage 4-5 (Path Recommendation) first.")
    st.stop()

path = st.session_state.learning_path
student_id = st.session_state.student_id

# ── Session Simulation ────────────────────────────────────────────────────
st.subheader("📚 Log a Study Session")

col1, col2 = st.columns(2)
with col1:
    time_since_last = st.slider(
        "Hours since last session",
        min_value=1, max_value=168, value=24,
        help="Used to compute Ebbinghaus forgetting decay",
    )
with col2:
    st.write("**Current Path:**")
    st.write(f"Algorithm: `{path.get('algorithm', 'DRL-PPO')}`")
    st.write(f"Steps: {len(path.get('steps', []))}")

st.write("**Mark responses for this session:**")
steps = path.get("steps", [])
session_responses: list[dict] = []

for step in steps[:5]:  # Show first 5 steps for simulation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{step['concept']}**")
    with col2:
        correct = st.checkbox("Correct ✓", key=f"correct_{step['step']}", value=True)
    session_responses.append(
        {
            "item_id": f"sim_{step['step']}",
            "concept": step["concept"],
            "answer_index": 0 if correct else 1,
            "response_time_ms": 3500,
            "correct": correct,
        }
    )

if st.button("📤 Submit Session & Re-optimise Path", type="primary"):
    session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    with st.spinner("Applying forgetting curve + updating knowledge state + re-optimising path…"):
        result = api_post(
            "/loop/update",
            {
                "student_id": student_id,
                "path_id": path["path_id"],
                "session_id": session_id,
                "session_responses": session_responses,
                "time_since_last_session_hours": float(time_since_last),
            },
        )

    if result:
        st.session_state.knowledge_vector = result["updated_knowledge_vector"]
        new_path = result.get("new_path_recommendation", path)
        if new_path.get("path_id") != path.get("path_id"):
            st.session_state.learning_path = new_path
            st.session_state.path_id = new_path["path_id"]

        st.success("✅ Session logged and path re-optimised!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Learning Efficiency",
                f"{result.get('session_learning_efficiency', 0):.3f}",
                help="knowledge_gain / time_invested",
            )
        with col2:
            st.metric("Progress", f"{result.get('cumulative_progress_pct', 0):.1f}%")
        with col3:
            st.metric("Sessions Remaining", result.get("estimated_sessions_remaining", "?"))

        # Forgetting applied
        forgetting = result.get("forgetting_applied", {})
        if forgetting:
            st.subheader("🧠 Forgetting Decay Applied (Ebbinghaus)")
            try:
                import pandas as pd
                fdf = pd.DataFrame(
                    [(c, v) for c, v in forgetting.items()],
                    columns=["Concept", "Mastery Lost"],
                )
                st.dataframe(fdf, use_container_width=True)
            except ImportError:
                for c, v in forgetting.items():
                    st.write(f"- {c}: {v:.4f}")

        # Path changes
        changes = result.get("path_changes", [])
        if changes:
            st.subheader("🔄 Path Changes")
            for change in changes:
                icon = "➕" if change.startswith("added") else ("➖" if change.startswith("removed") else "✅")
                st.write(f"{icon} {change}")

st.divider()

# ── History Dashboard ─────────────────────────────────────────────────────
st.subheader("📈 Learning History")

if st.button("🔄 Load History"):
    history_data = api_get(f"/loop/history/{student_id}")
    if history_data:
        sessions = history_data.get("sessions", [])
        if not sessions:
            st.info("No sessions recorded yet.")
        else:
            try:
                import pandas as pd
                import plotly.express as px

                # Mastery over time
                rows = []
                for sess in sessions:
                    ts = sess.get("timestamp", "")
                    eff = sess.get("efficiency_score", 0)
                    after = sess.get("mastery_after", {})
                    avg_mastery = sum(after.values()) / max(len(after), 1) if after else 0
                    rows.append({
                        "Session": ts[:19],
                        "Avg Mastery": round(avg_mastery, 3),
                        "Efficiency": round(eff, 4),
                        "N Responses": sess.get("n_responses", 0),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Mastery over time line chart
                fig = px.line(
                    df, x="Session", y="Avg Mastery",
                    title="Average Concept Mastery Over Sessions",
                    markers=True,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

                # Forgetting curve for a selected concept
                all_concepts = list(sessions[-1].get("mastery_after", {}).keys()) if sessions else []
                if all_concepts:
                    selected_concept = st.selectbox("Forgetting curve for concept:", all_concepts)
                    concept_history = [
                        {
                            "Session": s.get("timestamp", "")[:19],
                            "Mastery Before": s.get("mastery_before", {}).get(selected_concept, 0),
                            "Mastery After": s.get("mastery_after", {}).get(selected_concept, 0),
                        }
                        for s in sessions
                        if selected_concept in s.get("mastery_after", {})
                    ]
                    if concept_history:
                        cdf = pd.DataFrame(concept_history)
                        fig2 = px.line(
                            cdf, x="Session",
                            y=["Mastery Before", "Mastery After"],
                            title=f"Forgetting Curve: {selected_concept}",
                            markers=True,
                        )
                        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                        st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.json(sessions)

# ── Progress gauge ─────────────────────────────────────────────────────────
ks = st.session_state.get("knowledge_vector", {})
concept_mastery = ks.get("concept_mastery", {})
if concept_mastery:
    mastered = sum(1 for v in concept_mastery.values() if v >= 0.85)
    total = len(concept_mastery)
    pct = mastered / max(total, 1)
    st.divider()
    st.subheader("🎯 Overall Goal Progress")
    st.progress(pct, text=f"{mastered}/{total} concepts mastered ({pct:.0%})")
    if pct >= 1.0:
        st.balloons()
        st.success("🎉 Congratulations! You've mastered all concepts in your learning goal!")
