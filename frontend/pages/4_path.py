"""Page 4 — Knowledge Graph + Learning Path Recommendation."""
import streamlit as st
import requests

st.set_page_config(page_title="Path — AdaptLearn AI", layout="wide")
st.title("🚀 Stage 4-5: Knowledge Graph & Learning Path")

API = st.session_state.get("api_base", "http://localhost:8000/api/v1")
HEADERS = {"Authorization": f"Bearer {st.session_state.get('api_key', 'dev-secret')}"}


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API}{endpoint}", json=payload, headers=HEADERS, timeout=120)
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


if not st.session_state.get("knowledge_vector"):
    st.warning("⚠️ Please complete Stage 3 (Assessment) first.")
    st.stop()

# ── Knowledge Graph Build ─────────────────────────────────────────────────
st.subheader("🕸️ Knowledge Graph")

if not st.session_state.get("kg_id"):
    if st.button("🔨 Build Knowledge Graph", type="primary"):
        with st.spinner("Building KG with GraphRAG + GAT embeddings + Neo4j… (may take ~60s)"):
            result = api_post(
                "/kg/build",
                {
                    "goal_id": st.session_state.goal_id,
                    "student_id": st.session_state.student_id,
                },
            )
        if result:
            st.session_state.kg_id = result["kg_id"]
            st.session_state.kg_data = result
            st.success(
                f"✅ KG built! {result['n_nodes']} concepts, {result['n_edges']} relationships"
            )
else:
    st.success(f"KG ID: `{st.session_state.kg_id}` ✅")

# ── KG Visualisation ──────────────────────────────────────────────────────
if st.session_state.get("kg_id"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗺️ Visualise Knowledge Graph"):
            vis_data = api_get(f"/kg/{st.session_state.kg_id}/visualize")
            if vis_data:
                try:
                    from pyvis.network import Network
                    import streamlit.components.v1 as components

                    net = Network(height="500px", width="100%", bgcolor="#1e1e1e", font_color="white")
                    net.set_options("""
                    {
                      "physics": {"enabled": true, "stabilization": {"iterations": 100}},
                      "edges": {"arrows": {"to": {"enabled": true}}},
                      "nodes": {"font": {"size": 14}}
                    }
                    """)

                    for n in vis_data.get("nodes", []):
                        net.add_node(
                            n["id"], label=n["label"], title=n.get("title", ""),
                            color=n.get("color", "#1f77b4"), size=20
                        )
                    for e in vis_data.get("edges", []):
                        net.add_edge(
                            e["from"], e["to"],
                            label=e.get("label", ""),
                            arrows=e.get("arrows", ""),
                            dashes=e.get("dashes", False),
                            width=e.get("width", 1),
                        )

                    net.save_graph("/tmp/kg_vis.html")
                    with open("/tmp/kg_vis.html") as f:
                        html = f.read()
                    components.html(html, height=520)
                except ImportError:
                    st.info("Install pyvis for interactive graph visualisation.")
                    st.json(vis_data)

st.divider()

# ── Path Recommendation ───────────────────────────────────────────────────
st.subheader("🎯 Learning Path Recommendation")

if not st.session_state.get("kg_id"):
    st.warning("Build the knowledge graph first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    algorithm = st.selectbox(
        "Algorithm",
        ["DRL-PPO", "AKT-greedy", "DKVMN-greedy", "BKT-greedy"],
        help="DRL-PPO is the main thesis contribution",
    )
with col2:
    benchmark_mode = st.checkbox("Benchmark Mode (compare all algorithms)")

if st.button("✨ Generate Learning Path", type="primary"):
    with st.spinner(f"Running {algorithm}… "):
        result = api_post(
            "/recommend/path",
            {
                "student_id": st.session_state.student_id,
                "goal_id": st.session_state.goal_id,
                "kg_id": st.session_state.kg_id,
                "algorithm": algorithm,
                "benchmark_mode": benchmark_mode,
            },
        )

    if result:
        path = result.get("primary_path", result) if benchmark_mode else result
        st.session_state.path_id = path["path_id"]
        st.session_state.learning_path = path
        st.session_state.benchmark_data = result.get("benchmark") if benchmark_mode else None
        st.success(f"✅ Path generated! {len(path['steps'])} steps · energy={path['energy_score']:.1f}")

# ── Display Path ──────────────────────────────────────────────────────────
if st.session_state.get("learning_path"):
    path = st.session_state.learning_path
    steps = path.get("steps", [])

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", path["algorithm"])
    with col2:
        st.metric("Steps", len(steps))
    with col3:
        st.metric("Predicted Sessions", path["predicted_completion_sessions"])

    # Step cards
    st.subheader("📋 Your Personalised Learning Path")
    for step in steps:
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])
            with col1:
                st.metric(f"Step {step['step']}", "")
            with col2:
                icon = "🎬" if step["resource_type"] == "video" else "📰"
                st.write(f"**{icon} {step['concept']}**")
                if step.get("url"):
                    st.markdown(f"[🔗 Open Resource]({step['url']})")
            with col3:
                st.metric("Duration", f"{step['duration_min']} min")
                st.metric("Mastery Δ", f"+{step['predicted_mastery_delta']:.2f}")
            st.divider()

    # Benchmark comparison table
    if st.session_state.get("benchmark_data"):
        st.subheader("📊 Benchmark Comparison")
        bench = st.session_state.benchmark_data
        try:
            import pandas as pd
            rows = []
            for algo, data in bench.items():
                m = data.get("metrics", {})
                rows.append({
                    "Algorithm": algo,
                    "AUC": f"{m.get('auc', 0):.4f}",
                    "Accuracy": f"{m.get('accuracy', 0):.4f}",
                    "RMSE": f"{m.get('rmse', 0):.4f}",
                    "Recall@5": f"{m.get('recall_at_k', 0):.4f}",
                    "LES": f"{m.get('les', 0):.4f}",
                    "Steps": len(data.get("path", {}).get("steps", [])),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption("LES = Learning Efficiency Score = knowledge_gain / time_invested")
        except ImportError:
            st.json(bench)

    with st.expander("📄 Raw Path JSON"):
        st.json(path)

    # LLM Explanation
    st.divider()
    st.subheader("💬 Why This Path?")
    exp_type = st.selectbox("Explanation style", ["full", "brief", "motivational"])
    if st.button("🤖 Get Explanation"):
        with st.spinner("Generating RAG-augmented explanation…"):
            try:
                resp = requests.post(
                    f"{API}/explain/path",
                    json={
                        "student_id": st.session_state.student_id,
                        "path_id": path["path_id"],
                        "explanation_type": exp_type,
                    },
                    headers=HEADERS,
                    timeout=60,
                )
                resp.raise_for_status()
                exp = resp.json()
                st.info(exp.get("explanation", ""))
                st.write("**Key Reasons:**")
                for r in exp.get("key_reasons", []):
                    st.write(f"• {r}")
                st.success(f"**Next Action:** {exp.get('next_action', '')}")
            except Exception as exc:
                st.error(f"Explanation error: {exc}")

    st.info("➡️ Go to **Stage 5 — Loop** to track your progress after study sessions.")
