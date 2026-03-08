"""Page 3 — Knowledge State Assessment Visualisation."""
import streamlit as st
import requests
import json

st.set_page_config(page_title="Assessment — AdaptLearn AI", layout="wide")
st.title("📊 Stage 3: Knowledge State Assessment")

API = st.session_state.get("api_base", "http://localhost:8000/api/v1")
HEADERS = {"Authorization": f"Bearer {st.session_state.get('api_key', 'dev-secret')}"}


if not st.session_state.get("knowledge_vector"):
    st.warning("⚠️ Please complete Stage 2 (Quiz) first.")
    st.stop()

ks = st.session_state.knowledge_vector

# ── Overview Metrics ──────────────────────────────────────────────────────
st.subheader("Ability Estimate")
theta = ks.get("theta", 0.0)
ci = ks.get("confidence_interval", [theta - 0.4, theta + 0.4])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Overall θ (theta)", f"{theta:.3f}")
with col2:
    st.metric("95% CI Lower", f"{ci[0]:.3f}")
with col3:
    st.metric("95% CI Upper", f"{ci[1]:.3f}")
with col4:
    level = "Beginner" if theta < -1 else ("Intermediate" if theta < 1 else "Advanced")
    st.metric("Level", level)

st.divider()

# ── Per-Concept Mastery ───────────────────────────────────────────────────
concept_mastery = ks.get("concept_mastery", {})

if concept_mastery:
    st.subheader("Per-Concept Mastery")

    # Sort by mastery ascending
    sorted_concepts = sorted(concept_mastery.items(), key=lambda x: x[1])

    # Bar chart
    try:
        import pandas as pd
        df = pd.DataFrame(sorted_concepts, columns=["Concept", "Mastery"])
        df["Mastery %"] = (df["Mastery"] * 100).round(1)
        df["Status"] = df["Mastery"].apply(
            lambda x: "✅ Mastered" if x >= 0.85 else ("🟡 Developing" if x >= 0.5 else "🔴 Needs Work")
        )
        st.bar_chart(df.set_index("Concept")["Mastery %"])
        st.dataframe(df[["Concept", "Mastery %", "Status"]], use_container_width=True)
    except ImportError:
        for concept, mastery in sorted_concepts:
            pct = mastery * 100
            color = "green" if mastery >= 0.85 else ("orange" if mastery >= 0.5 else "red")
            st.progress(mastery, text=f"{concept}: {pct:.1f}%")

    # Radar chart using plotly
    try:
        import plotly.graph_objects as go
        concepts = list(concept_mastery.keys())
        values = list(concept_mastery.values())
        if len(concepts) >= 3:
            fig = go.Figure(
                data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=concepts + [concepts[0]],
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.3)",
                    line_color="#1f77b4",
                    name="Knowledge State",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Knowledge Radar",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for radar chart visualisation.")

else:
    st.info("No per-concept mastery data available.")

st.divider()

# ── Raw Data ──────────────────────────────────────────────────────────────
with st.expander("📄 Raw Knowledge Vector JSON"):
    st.json(ks)

st.info("➡️ Next: Go to **Stage 4 — Path** to build the knowledge graph and get your learning path.")
