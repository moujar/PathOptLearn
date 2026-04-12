"""
Evaluation Dashboard — PathOptLearn
====================================
Streamlit dashboard for visualising all evaluation results.

Launch
------
  cd evalution
  streamlit run dashboard/eval_dashboard.py -- --db eval_output/results.db

Tabs
----
  1. Overview         — KPI summary cards
  2. Learning Metrics — trajectory plots, gain distribution
  3. LLM Quality      — radar chart, hallucination rate
  4. Benchmark (KT)   — knowledge-tracing table vs. Riiid! / EdNet / ASSISTments
  5. A/B Testing      — box plots, statistical significance
  6. System Perf      — latency per endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EVAL_ROOT))
sys.path.insert(0, str(_EVAL_ROOT / "pipeline"))

from pipeline.experiment_tracker import ExperimentTracker  # noqa: E402

# ── Try importing optional plotting libs ──────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PathOptLearn — Evaluation Dashboard",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_tracker(db_path: str) -> pd.DataFrame:
    """Load all run data from the experiment tracker."""
    try:
        tracker = ExperimentTracker(db_path)
        return tracker.compare_runs()
    except Exception as e:
        return pd.DataFrame()


def load_json(path: str) -> Any:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _metric(df: pd.DataFrame, col: str) -> float | None:
    """Get the latest non-null value of a column from the runs DataFrame."""
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

def render_overview(df: pd.DataFrame, output_dir: str):
    """Tab 1 — high-level KPI cards."""
    st.header("Overview")

    if df.empty:
        st.warning("No experiment runs found. Run `eval_runner.py` first.")
        return

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    def kpi(col, label: str, val, fmt: str = "{:.1%}"):
        if val is not None:
            col.metric(label, fmt.format(val))
        else:
            col.metric(label, "—")

    kpi(col1, "Mean Pass Rate",  _metric(df, "sim.module_success.eventual_pass_rate"))
    kpi(col2, "Mean Avg Score",  _metric(df, "sim.cohort_stats.mean_avg_score"),       "{:.1f}%")
    kpi(col3, "Hallucination Rate", _metric(df, "quality.lesson_quality.mean_hallucination_rate"))
    kpi(col4, "Gap Coverage",    _metric(df, "resource.gap_coverage.coverage_rate"))
    kpi(col5, "Gap F1",          _metric(df, "gap.f1"),                                "{:.3f}")

    st.divider()
    st.subheader("All Experiment Runs")
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Run timeline")
    if "started_at" in df.columns and "sim.cohort_stats.mean_avg_score" in df.columns:
        time_df = df[["name", "started_at",
                       "sim.cohort_stats.mean_avg_score"]].dropna()
        time_df.columns = ["name", "started_at", "mean_score"]
        if PLOTLY and not time_df.empty:
            fig = px.line(time_df, x="started_at", y="mean_score",
                          text="name", markers=True,
                          title="Mean module score over time",
                          labels={"mean_score": "Score (%)", "started_at": "Time"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(time_df.set_index("started_at")[["mean_score"]])


def render_learning(output_dir: str):
    """Tab 2 — learning effectiveness plots."""
    st.header("Learning Metrics")

    sim_path = Path(output_dir) / "sim_results.json"
    data     = load_json(str(sim_path))

    if not data:
        st.info(f"No simulation results found at `{sim_path}`.")
        return

    summaries = data.get("summaries", [])
    if not summaries:
        st.info("No student summaries in results file.")
        return

    # ── Learning trajectories ──────────────────────────────────────────────
    st.subheader("Learning Trajectories (Diagnostic → Module Scores)")

    traj_data: dict[str, list] = {}
    for sim in summaries:
        label  = sim.get("profile", "?")
        scores = [sim.get("diag_score", 0)]
        for mod in sim.get("module_results", []):
            scores.append(mod.get("final_score", 0))
        traj_data[label] = scores

    if PLOTLY and traj_data:
        fig = go.Figure()
        for label, scores in traj_data.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                mode="lines+markers",
                name=label,
            ))
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Mastery threshold (70%)")
        fig.update_layout(
            title="Score trajectory per student",
            xaxis_title="Step (0 = diagnostic)",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        traj_df = pd.DataFrame.from_dict(traj_data, orient="index").T
        st.line_chart(traj_df)

    # ── Per-profile stats ──────────────────────────────────────────────────
    st.subheader("Per-Profile Performance")
    rows = []
    for sim in summaries:
        rows.append({
            "Profile":       sim.get("profile", "?"),
            "Diag Score":    sim.get("diag_score", 0),
            "Avg Score":     sim.get("avg_score", 0),
            "Pass Rate":     f"{sim.get('pass_rate', 0):.0%}",
            "Retries":       sim.get("total_retries", 0),
            "Wall Time (s)": sim.get("wall_time_s", 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Normalised learning gain distribution ──────────────────────────────
    metrics = data.get("metrics", {})
    gain    = metrics.get("learning_gain", {})
    labels  = gain.get("labels", {})
    if labels and PLOTLY:
        st.subheader("Normalised Learning Gain Distribution")
        fig2 = px.bar(
            x=list(labels.keys()),
            y=list(labels.values()),
            labels={"x": "Category", "y": "# Students"},
            title="Hake's Normalised Learning Gain (g)",
            color=list(labels.keys()),
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_llm_quality(output_dir: str):
    """Tab 3 — LLM content quality."""
    st.header("LLM Content Quality")

    qual_path = Path(output_dir) / "quality_results.json"
    data      = load_json(str(qual_path))

    if not data:
        st.info(f"No quality results found at `{qual_path}`.")
        return

    lq = data.get("lesson_quality", {})
    qq = data.get("quiz_quality",   {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lessons Evaluated", lq.get("n_lessons", 0))
    col2.metric("Overall Score",
                f"{lq.get('mean_overall_score', 0):.2f}/5" if lq.get("mean_overall_score") else "—")
    col3.metric("Hallucination Rate",
                f"{lq.get('hallucination_rate', 0):.1%}" if lq.get("hallucination_rate") is not None else "—")
    col4.metric("Readability Grade",
                f"{lq.get('mean_readability_grade', 0):.1f}" if lq.get("mean_readability_grade") else "—")

    # Radar chart
    dims = {
        "Factual Accuracy":       lq.get("mean_factual_accuracy"),
        "Relevance":              lq.get("mean_relevance"),
        "Coherence":              lq.get("mean_coherence"),
        "Level Appropriateness":  lq.get("mean_level_appropriateness"),
        "Example Quality":        lq.get("mean_example_quality"),
    }
    vals = [(k, v) for k, v in dims.items() if v is not None]

    if vals and PLOTLY:
        cats   = [v[0] for v in vals] + [vals[0][0]]
        scores = [v[1] for v in vals] + [vals[0][1]]
        fig = go.Figure(go.Scatterpolar(
            r=scores, theta=cats, fill="toself", name="Lesson Quality"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Lesson Quality Radar",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quiz quality
    st.subheader("Quiz Quality")
    col5, col6 = st.columns(2)
    col5.metric("Quizzes Evaluated", qq.get("n_quizzes", 0))
    col6.metric("Distractor Quality",
                f"{qq.get('mean_distractor_quality_score', 0):.2f}/5"
                if qq.get("mean_distractor_quality_score") else "—")

    # Readability distribution
    rd = lq.get("readability_label_dist", {})
    if rd and PLOTLY:
        fig2 = px.pie(values=list(rd.values()), names=list(rd.keys()),
                      title="Lesson Readability Distribution")
        st.plotly_chart(fig2, use_container_width=True)


def render_benchmark(output_dir: str):
    """Tab 4 — knowledge-tracing benchmark results."""
    st.header("Knowledge Tracing Benchmark")
    st.caption(
        "Results from `run_benchmark.py` comparing PathOptLearn's /find-gaps "
        "against Riiid!, EdNet-KT1, and ASSISTments datasets."
    )

    # Look for any results*.json files in output_dir
    result_files = list(Path(output_dir).glob("benchmark_*.json")) + \
                   list(Path(output_dir).glob("results.json"))

    if not result_files:
        st.info(
            "No benchmark results found. Run:\n\n"
            "```bash\n"
            "python benchmakring/run_benchmark.py \\\n"
            "    --dataset riiid --data /path/to/train.csv \\\n"
            "    --api http://localhost:8000 \\\n"
            "    --output eval_output/benchmark_riiid.json\n"
            "```"
        )
        return

    rows = []
    for f in result_files:
        data = load_json(str(f))
        if isinstance(data, list):
            for res in data:
                for model in res.get("models", []):
                    rows.append({
                        "Dataset":   res.get("dataset", "?"),
                        "Students":  res.get("n_students", 0),
                        "Model":     model.get("model", "?"),
                        "AUC":       model.get("auc"),
                        "Accuracy":  model.get("accuracy"),
                        "RMSE":      model.get("rmse"),
                        "N Samples": model.get("n_samples"),
                    })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if PLOTLY:
            fig = px.bar(df, x="Model", y="AUC", color="Dataset",
                         barmode="group", title="AUC by Model and Dataset",
                         labels={"AUC": "AUC (higher is better)"})
            fig.add_hline(y=0.5, line_dash="dot", annotation_text="Random baseline")
            st.plotly_chart(fig, use_container_width=True)


def render_ab_testing(output_dir: str):
    """Tab 5 — A/B testing results."""
    st.header("A/B Testing")

    ab_path = Path(output_dir) / "ab_results.json"
    data    = load_json(str(ab_path))

    if not data:
        st.info(
            f"No A/B results at `{ab_path}`. Run:\n\n"
            "```bash\n"
            "python benchmakring/ab_testing.py \\\n"
            "    --topic 'Machine Learning' \\\n"
            "    --api http://localhost:8000 \\\n"
            "    --profiles beginner intermediate\n"
            "```"
        )
        return

    st.subheader(f"Topic: {data.get('topic', '?')}")

    # Box-plot data
    raw = data.get("raw_results", {})
    box_rows = []
    for cond, profiles in raw.items():
        for profile, results in profiles.items():
            for r in results:
                box_rows.append({
                    "Condition": cond,
                    "Profile":   profile,
                    "Avg Score": r.get("avg_score", 0),
                    "Pass Rate": r.get("pass_rate", 0),
                })

    if box_rows and PLOTLY:
        df_box = pd.DataFrame(box_rows)
        fig = px.box(df_box, x="Condition", y="Avg Score", color="Profile",
                     title="Average Module Score by Condition",
                     points="all")
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Mastery threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    st.subheader("Statistical Comparisons")
    comp_rows = []
    for pair, comp in data.get("comparison", {}).items():
        mw = comp.get("mann_whitney", {})
        ci_a = comp.get("bootstrap_ci_a", {})
        ci_b = comp.get("bootstrap_ci_b", {})
        comp_rows.append({
            "Comparison":    pair,
            "Cohen's d":     comp.get("cohen_d", 0),
            "p-value":       mw.get("p_value", "—"),
            "Significant":   "✓" if mw.get("significant") else "✗",
            "A mean score":  ci_a.get("mean", 0),
            "B mean score":  ci_b.get("mean", 0),
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


def render_performance(output_dir: str):
    """Tab 6 — system latency and cost."""
    st.header("System Performance")
    st.caption(
        "Latency data is extracted from simulation event logs. "
        "Each event records a wall-clock timestamp."
    )

    # Look for simulation results with wall_time_s
    sim_path = Path(output_dir) / "sim_results.json"
    data     = load_json(str(sim_path))

    if not data:
        st.info("Run simulation evaluation first to populate this tab.")
        return

    rows = []
    for sim in data.get("summaries", []):
        rows.append({
            "Profile":       sim.get("profile", "?"),
            "Modules":       sim.get("n_modules", 0),
            "Wall Time (s)": sim.get("wall_time_s", 0),
            "Avg per Module (s)": round(
                sim.get("wall_time_s", 0) / max(sim.get("n_modules", 1), 1), 1
            ),
            "Retries":       sim.get("total_retries", 0),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if PLOTLY:
            fig = px.bar(df, x="Profile", y="Wall Time (s)",
                         title="Total simulation time by profile",
                         color="Profile")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost Estimation")
    st.info(
        "PathOptLearn runs fully locally with Ollama (llama3.2:1b). "
        "Estimated cost per student session:\n\n"
        "- API calls: ~20–40 requests per full session\n"
        "- LLM calls: ~10–20 Ollama inferences (local, no cost)\n"
        "- DB reads/writes: ~50–100 PostgreSQL + Neo4j queries\n\n"
        "For cloud LLM alternatives, estimate based on ~50K tokens/session."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Parse --db argument passed after `--` in streamlit run
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db",     default="eval_output/results.db")
    parser.add_argument("--output", default="eval_output")
    args, _ = parser.parse_known_args()

    output_dir = args.output

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("PathOptLearn Evaluation")
    st.sidebar.caption(f"DB: `{args.db}`")

    df = load_tracker(args.db)
    if not df.empty:
        run_ids = df["run_id"].tolist()
        selected = st.sidebar.multiselect(
            "Filter runs", run_ids, default=run_ids[-3:] if len(run_ids) > 3 else run_ids
        )
        if selected:
            df = df[df["run_id"].isin(selected)]

    if not PLOTLY:
        st.sidebar.warning("Install `plotly` for interactive charts:\n`pip install plotly`")

    output_dir_input = st.sidebar.text_input("Output directory", value=output_dir)
    if output_dir_input:
        output_dir = output_dir_input

    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Learning Metrics",
        "🤖 LLM Quality",
        "🏆 Benchmark (KT)",
        "🔬 A/B Testing",
        "⚡ System Perf",
    ])

    with tab1:
        render_overview(df, output_dir)
    with tab2:
        render_learning(output_dir)
    with tab3:
        render_llm_quality(output_dir)
    with tab4:
        render_benchmark(output_dir)
    with tab5:
        render_ab_testing(output_dir)
    with tab6:
        render_performance(output_dir)


main()
