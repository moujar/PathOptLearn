# PathOptLearn — Benchmarking & Evaluation Framework

## Overview

This document specifies the complete evaluation strategy for PathOptLearn, an
AI-powered personalised learning system. The framework covers seven evaluation
dimensions, defines measurable metrics, describes benchmark datasets, and
outlines the methodology for academic publication.

**System under test:** PathOptLearn API (`http://localhost:8000`)  
**LLM:** Ollama `llama3.2:1b` (local, no cloud cost)  
**Stores:** PostgreSQL 16 (relational) + Neo4j 5 (knowledge graph)

---

## 1. Evaluation Goals

| Goal | What we want to know |
|------|----------------------|
| **Learning effectiveness** | Does the system help students learn? Do scores improve from diagnostic to final modules? |
| **Personalisation accuracy** | Does the diagnosed level match actual performance? Does module difficulty align with learner level? |
| **LLM reasoning quality** | Are generated lessons factually correct? Do quizzes have good distractors? Is content level-appropriate? |
| **Knowledge gap detection** | Does `/find-gaps` correctly identify which concepts a student is missing? |
| **Resource relevance** | Are recommended resources relevant to the identified gaps? |
| **System performance** | What is per-request latency? What is the cost per learner session? |
| **Comparison vs. baselines** | Is personalisation better than a static curriculum or search-only approach? |

---

## 2. Evaluation Dimensions & Metrics

### 2.1 Learning Effectiveness

| Metric | Definition | Target |
|--------|-----------|--------|
| **Normalised Learning Gain (NLG)** | `g = (post − pre) / (max − pre)` where pre = diagnostic score, post = mean final module score | g ≥ 0.3 (medium gain) |
| **Module Success Rate** | % of modules passed on first attempt | ≥ 60% |
| **Eventual Pass Rate** | % of modules eventually passed (any attempt) | ≥ 80% |
| **Never-Mastered Rate** | % of modules abandoned after exhausting patience | ≤ 10% |
| **Quiz Improvement Rate** | Mean score delta per retry for multi-attempt modules | > 5 points/retry |
| **Time-to-Mastery** | Mean number of attempts to reach 70% | ≤ 2.0 attempts |
| **Completion Rate** | % of students who pass all modules | ≥ 50% |
| **Dropout Rate** | % of students completing < 50% of modules | ≤ 20% |

**Implementation:** `evalution/metrics/learning_metrics.py`  
**Data source:** `llm_student.py` simulation results or PostgreSQL `PROGRESS` + `TOPIC_MASTERY` tables.

### 2.2 Personalisation Quality

| Metric | Definition |
|--------|-----------|
| **Diagnostic–Final Correlation** | Pearson r between diagnostic score and mean module score across students. High r → system correctly identifies learner level. |
| **Mean Uplift** | `avg(final_score) − avg(diag_score)` — does the system produce measurable improvement? |
| **Level–Performance Alignment** | Do beginner-diagnosed students score lower than advanced-diagnosed students on equivalent modules? |

### 2.3 LLM Output Quality

#### Lessons

| Metric | Range | How measured |
|--------|-------|-------------|
| **Factual Accuracy** | 1–5 | Ollama-as-judge |
| **Relevance** | 1–5 | Ollama-as-judge (does lesson address module objective?) |
| **Coherence** | 1–5 | Ollama-as-judge |
| **Level Appropriateness** | 1–5 | Ollama-as-judge |
| **Example Quality** | 1–5 | Ollama-as-judge |
| **Hallucination Rate** | 0–1 | Fraction of lessons flagged with hallucination_risk=True |
| **Readability Grade** | FK Grade Level | Deterministic (Flesch-Kincaid approximation) |

#### Quizzes

| Metric | How measured |
|--------|-------------|
| **Format Validity %** | % of questions with correct MCQ structure (4 options, valid answer key) |
| **Distractor Quality** | 1–5 (Ollama judge) — are wrong options plausible, not trivially obvious? |
| **Answer Distribution Balance** | Are correct answers evenly distributed across A/B/C/D? |

**Implementation:** `evalution/metrics/llm_quality.py`

### 2.4 Knowledge Gap Detection Quality

| Metric | Definition |
|--------|-----------|
| **Precision** | Of predicted gaps, what fraction are true gaps? |
| **Recall** | Of true gaps, what fraction did the system detect? |
| **F1 Score** | Harmonic mean of precision and recall |
| **Detection Rate** | "over" / "under" / "on_target" (predicted set size vs. ground truth) |
| **Severity Calibration** | Point-biserial correlation between predicted severity (high/medium/low) and quiz failure. High r → severity is predictive. |
| **Gap Stability (Jaccard)** | Similarity between two `/find-gaps` calls with identical input. Measures reproducibility. |

**Implementation:** `evalution/metrics/gap_eval.py`  
**Ground truth:** `evalution/ground_truth/benchmark_creator.py`

### 2.5 Resource Recommendation Quality

| Metric | Definition |
|--------|-----------|
| **TF-IDF Relevance Score** | Cosine similarity between resource text and gap concept query |
| **Gap Coverage Rate** | Fraction of gaps with ≥ 1 resource above relevance threshold |
| **Source Diversity Score** | Harmonic mean of domain-diversity and type-diversity |
| **Mean Reciprocal Rank (MRR)** | For ranked resource lists vs. ground-truth URLs |

**Implementation:** `evalution/metrics/resource_eval.py`

### 2.6 Knowledge Tracing (Benchmark)

Evaluated on public educational datasets using the existing `run_benchmark.py`:

| Metric | Description |
|--------|-----------|
| **AUC-ROC** | Area under ROC curve for predicting next-question correctness |
| **Accuracy** | Binary classification accuracy at threshold 0.5 |
| **RMSE** | Root mean square error of predicted probability vs. true label |

Datasets: Riiid! (13M interactions), EdNet-KT1 (95M), ASSISTments 2009–2010.

### 2.7 System Performance

| Metric | Description |
|--------|-----------|
| **API Latency (p50/p95)** | Median and 95th-percentile response time per endpoint |
| **Cost per Session** | Estimated compute cost for one full student session |
| **Tokens per Session** | Total LLM tokens generated per student (local model, no dollar cost) |
| **Wall Time per Student** | End-to-end simulation time (from `llm_student.py`) |

---

## 3. Benchmark Datasets

### 3.1 Curated Topic Benchmark

**15 topics** across 3 domains, defined in `benchmark_creator.py`:

| Domain | Topics |
|--------|--------|
| Computer Science | Machine Learning, Data Structures, Algorithms, Computer Networks, Cybersecurity, OS, Deep Learning, NLP |
| Mathematics | Linear Algebra, Calculus, Statistics & Probability, Discrete Mathematics |
| Applied | Database Systems, Software Engineering, System Design |

**Per topic:**
- 5 MCQ questions per difficulty level (beginner / intermediate / advanced)
- Ground-truth correct answers
- Expected knowledge gaps per level

**Generation:** Questions generated via Ollama (LLM) with static fallbacks for well-known topics. Expert annotation recommended before production use.

### 3.2 Public Knowledge Tracing Datasets

| Dataset | Size | Columns used |
|---------|------|--------------|
| Riiid! (Kaggle 2020) | 13M rows | `user_id`, `question_id`, `answered_correctly`, `timestamp` |
| EdNet-KT1 | 95M rows | `user_id`, `question_id`, `correct`, `elapsed_time` |
| ASSISTments 2009–2010 | ~340K rows | `user_id`, `problem_id`, `correct`, `skill_name`, `hint_count` |

### 3.3 LLM-Judge Ground Truth

For lesson quality evaluation, Ollama acts as judge with structured prompts.
Scores are on a 1–5 scale per dimension. For academic-grade evaluation, a
sample of 50+ lessons should be double-annotated by human evaluators and
compared against Ollama judgements using Cohen's κ.

---

## 4. Benchmarking Methodology

### 4.1 Offline Evaluation (Knowledge Tracing)

```
Dataset → Sample N students → Leave-one-out per student
→ Feed history window to /find-gaps → Predict correctness probability
→ Compare to ground truth with AUC / Accuracy / RMSE
→ Compare against LogReg baseline and random baseline
```

**Script:** `evalution/benchmakring/run_benchmark.py`

### 4.2 Synthetic Learner Simulation

```
5 student profiles × M topics × R repetitions
→ Each student navigates the full PathOptLearn flow via the real API
→ Collect: diag_score, per-module scores, retries, wall time
→ Compute all learning metrics
→ Log to ExperimentTracker (SQLite)
```

**Profiles:** beginner (65% error), intermediate (35%), advanced (10%),
struggling (75%), fast_learner (15%).

**Script:** `evalution/UserSimulation/llm_student.py`  
**Orchestrator:** `evalution/pipeline/eval_runner.py`

### 4.3 A/B Testing

Four conditions run with identical student profiles:

| Condition | Description |
|-----------|-------------|
| **A — PathOptLearn** | Full system: diagnostic → personalised roadmap → gap-based remediation |
| **B — Static** | No diagnostic; fixed module order; no remediation |
| **C — No Remediation** | Full system but patience=1; no gap resources on failure |
| **D — Search Only** | Resources only; no quizzes |

**Statistical tests:**
- Mann-Whitney U (non-parametric, no normality assumption)
- Cohen's d (effect size)
- 95% bootstrap confidence intervals

**Script:** `evalution/benchmakring/ab_testing.py`

### 4.4 Human Evaluation (for Publication)

Recruit 10–30 real users per condition:

1. Pre-test: 10-question quiz on the chosen topic (independently validated)
2. Learning session: full PathOptLearn run (or control condition)
3. Post-test: same 10 questions (randomised order)
4. Measure: NLG, time spent, self-reported engagement (Likert 1–5)
5. Analysis: paired t-test (pre vs. post) + ANCOVA (conditions)

---

## 5. Baselines

| Baseline | Description | Implemented |
|----------|-------------|-------------|
| **Random** | Predict 50% correctness for all students | Yes (run_benchmark.py) |
| **Logistic Regression** | Student-level feature aggregates (avg_correct, n_interactions) | Yes (run_benchmark.py) |
| **Static Curriculum** | Fixed module order, no diagnostic personalisation | Yes (ab_testing.py, Condition B) |
| **Search Only** | Resources without quizzes or gap detection | Yes (ab_testing.py, Condition D) |
| **No Remediation** | PathOptLearn without retry resources | Yes (ab_testing.py, Condition C) |

For academic comparison, consider also adding:
- **BKT** (Bayesian Knowledge Tracing) as KT baseline
- **DKT** (Deep Knowledge Tracing) if deep learning baseline is needed

---

## 6. Automated Evaluation Pipeline

### 6.1 Architecture

```
EvalConfig (topic, profiles, flags)
    │
    ├── EvalRunner.run_all()
    │       ├── run_simulation_eval()     → LLM student simulations
    │       ├── run_llm_quality_eval()    → Lesson + quiz quality
    │       ├── run_resource_eval()       → Resource recommendation quality
    │       └── run_gap_eval()            → Gap detection precision/recall
    │
    └── ExperimentTracker (SQLite)
            ├── Numeric metrics (flat scalars)
            ├── JSON artefacts (full summaries)
            └── Compare / export CSV
```

### 6.2 Running the Pipeline

```bash
# Full evaluation on Machine Learning
cd prod/evalution
python pipeline/eval_runner.py \
    --topic "Machine Learning" \
    --api   http://localhost:8000 \
    --profiles beginner intermediate advanced \
    --output eval_output

# Knowledge-tracing benchmark (requires dataset)
python benchmakring/run_benchmark.py \
    --dataset riiid \
    --data    /path/to/train.csv \
    --api     http://localhost:8000 \
    --sample  1000 \
    --output  eval_output/benchmark_riiid.json

# A/B test
python benchmakring/ab_testing.py \
    --topic "Machine Learning" \
    --api   http://localhost:8000 \
    --profiles beginner intermediate \
    --n-reps 3 \
    --output eval_output/ab_results.json

# Launch evaluation dashboard
streamlit run dashboard/eval_dashboard.py -- \
    --db     eval_output/results.db \
    --output eval_output
```

### 6.3 Experiment Tracking

All runs are stored in SQLite (`eval_output/results.db`).

```python
from pipeline.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("eval_output/results.db")
tracker.print_summary()           # console table
tracker.export_csv("runs.csv")    # CSV export
tracker.export_json("runs.json")  # full JSON with artefacts

# Compare specific runs
df = tracker.compare_runs(["abc12345", "def67890"])
```

### 6.4 Reproducibility

| Practice | How |
|----------|-----|
| Random seeds | `--seed 42` flag in all scripts |
| Config logging | Full `EvalConfig` dict stored in each run record |
| Artefact storage | Full simulation summaries stored as JSON in SQLite |
| Docker environment | All services pinned to specific versions in `docker-compose.yml` |
| Model pinning | `llama3.2:1b` pulled deterministically by `ollama-init` service |

---

## 7. Visualization & Reporting

### 7.1 Evaluation Dashboard

Launch: `streamlit run dashboard/eval_dashboard.py`

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards: pass rate, avg score, hallucination rate, gap coverage, F1 |
| **Learning Metrics** | Trajectory plots per student, NLG distribution, per-profile table |
| **LLM Quality** | Radar chart (5 dimensions), hallucination rate, readability distribution |
| **Benchmark (KT)** | AUC/Accuracy/RMSE comparison table + bar chart vs. baselines |
| **A/B Testing** | Box plots by condition + statistical significance table |
| **System Perf** | Wall-time by profile, cost estimation |

### 7.2 Standard Output Files

| File | Contents |
|------|---------|
| `eval_output/eval_report.json` | Master results dict (all dimensions) |
| `eval_output/report.txt` | Human-readable text report |
| `eval_output/sim_results.json` | Full simulation summaries + learning metrics |
| `eval_output/quality_results.json` | Lesson and quiz quality evaluations |
| `eval_output/resource_results.json` | Resource recommendation evaluation |
| `eval_output/gap_results.json` | Gap detection precision/recall/F1 |
| `eval_output/ab_results.json` | A/B test raw results + statistics |
| `eval_output/results.db` | SQLite experiment tracker |
| `eval_output/comparison.csv` | Flat CSV of all runs and metrics |

---

## 8. Research-Level Evaluation (Academic Publication)

### 8.1 Research Questions

1. **RQ1 — Learning Effectiveness:** Does PathOptLearn produce measurably
   higher normalised learning gain than a static curriculum?
2. **RQ2 — Personalisation Value:** Do students diagnosed as beginner show
   lower error rates and higher retries, consistent with the personalisation
   hypothesis?
3. **RQ3 — Gap Detection Quality:** How accurately does LLM-based gap
   detection (no explicit student model) approximate a knowledge-tracing
   signal compared to BKT/DKT baselines?
4. **RQ4 — Resource Remediation:** Does gap-based resource remediation
   (Condition A vs. C) improve pass rates on failed modules?

### 8.2 Experimental Design for a Paper

**Study design:** 2 × 2 factorial (Personalised × Remediation) + search-only control

```
Conditions:
  A: Personalised=YES, Remediation=YES   ← PathOptLearn
  B: Personalised=NO,  Remediation=NO    ← Static LMS
  C: Personalised=YES, Remediation=NO    ← No retry resources
  D: Personalised=NO,  Remediation=YES   ← Static + gap resources
```

**Power analysis (for human study):**
- Effect size: d = 0.5 (medium, conservative)
- α = 0.05, power = 0.80
- Required N per condition: ~34 participants
- Total: ~136 participants (4 conditions)

**Evaluation protocol:**
1. Stratified random assignment (balance prior knowledge using pre-test)
2. Same topic used across all conditions to control content
3. 45-minute learning session (timed uniformly)
4. Post-test 24–48 hours later (delayed retention test)
5. Primary metric: NLG from pre-test to post-test
6. Secondary metrics: time-on-task, completion rate, self-efficacy (Likert)

**Statistical analysis:**
- Two-way ANOVA (personalisation × remediation interaction)
- Post-hoc Tukey HSD for pairwise comparisons
- Covariate: pre-test score (ANCOVA)
- Report: effect size (η²), 95% CIs, Bonferroni-corrected p-values

### 8.3 Knowledge Tracing Paper Contribution

If the target venue is EDM / LAK / AIED:

**Claim:** An LLM-based zero-shot gap detector (PathOptLearn `/find-gaps`)
achieves AUC > 0.65 on three public KT datasets without explicit student
modelling or historical training data.

**Comparison table:**

| Method | Riiid AUC | EdNet AUC | ASSISTments AUC |
|--------|-----------|-----------|-----------------|
| Random baseline | 0.50 | 0.50 | 0.50 |
| Logistic Regression | ~0.65 | ~0.63 | ~0.67 |
| **PathOptLearn (LLM gaps)** | ~0.68* | ~0.66* | ~0.70* |
| BKT (reference) | ~0.69 | ~0.68 | ~0.72 |
| DKT (reference) | ~0.74 | ~0.73 | ~0.82 |

(*Estimated targets — actual values come from `run_benchmark.py`.)

**Novel contribution:** First evaluation of a zero-shot LLM gap detector
as a knowledge-tracing signal, without student-model pre-training.

### 8.4 Paper Metrics Table Template

```
Table 1: System Evaluation Summary
┌──────────────────────────────────┬────────────┬─────────────┐
│ Dimension                        │ Metric     │ Value       │
├──────────────────────────────────┼────────────┼─────────────┤
│ Learning Effectiveness           │ Mean NLG   │ 0.XX ± 0.XX │
│                                  │ Pass Rate  │ XX%         │
│                                  │ Completion │ XX%         │
├──────────────────────────────────┼────────────┼─────────────┤
│ Gap Detection                    │ Precision  │ 0.XXX       │
│                                  │ Recall     │ 0.XXX       │
│                                  │ F1         │ 0.XXX       │
│                                  │ Calibr. r  │ 0.XXX       │
├──────────────────────────────────┼────────────┼─────────────┤
│ Content Quality (Ollama judge)   │ Factuality │ X.X/5       │
│                                  │ Relevance  │ X.X/5       │
│                                  │ Hallucin.  │ X.X%        │
├──────────────────────────────────┼────────────┼─────────────┤
│ Resource Recommendation          │ Coverage   │ XX%         │
│                                  │ Relevance  │ 0.XXXX      │
├──────────────────────────────────┼────────────┼─────────────┤
│ KT Benchmark (Riiid!)            │ AUC        │ 0.XXXX      │
│                                  │ Accuracy   │ 0.XXXX      │
│                                  │ RMSE       │ 0.XXXX      │
├──────────────────────────────────┼────────────┼─────────────┤
│ A/B vs. Static (Cohen's d)       │ avg_score  │ +X.XX (p<.) │
│ A/B vs. No-Remediation           │ pass_rate  │ +X.XX (p<.) │
└──────────────────────────────────┴────────────┴─────────────┘
```

---

## 9. File Structure Reference

```
evalution/
├── metrics/
│   ├── learning_metrics.py    # NLG, pass rate, completion, TTM, trajectories
│   ├── llm_quality.py         # Lesson + quiz quality (Ollama judge + readability)
│   ├── resource_eval.py       # TF-IDF relevance, diversity, gap coverage
│   └── gap_eval.py            # Precision, recall, F1, severity calibration
│
├── pipeline/
│   ├── experiment_tracker.py  # SQLite-based run tracking and comparison
│   └── eval_runner.py         # Master orchestrator (CLI + Python API)
│
├── ground_truth/
│   └── benchmark_creator.py   # 15-topic curated benchmark + validation
│
├── benchmakring/
│   ├── run_benchmark.py        # Knowledge-tracing benchmark (Riiid/EdNet/ASSISTments)
│   └── ab_testing.py           # A/B test framework (4 conditions, stats)
│
├── UserSimulation/
│   └── llm_student.py          # LLM student simulator (5 profiles)
│
├── dashboard/
│   └── eval_dashboard.py       # Streamlit 6-tab evaluation dashboard
│
├── FRAMEWORK.md                # This document
└── requirements.txt            # pandas, numpy, sklearn, scipy, streamlit, plotly
```

---

## 10. Quick-Start Checklist

- [ ] Start PathOptLearn: `docker-compose up -d` in `prod/`
- [ ] Install eval deps: `pip install -r evalution/requirements.txt`
- [ ] Generate ground-truth benchmark: `python ground_truth/benchmark_creator.py --output benchmark.json`
- [ ] Run full evaluation: `python pipeline/eval_runner.py --topic "Machine Learning"`
- [ ] Run A/B test: `python benchmakring/ab_testing.py --topic "Machine Learning" --n-reps 2`
- [ ] Launch dashboard: `streamlit run dashboard/eval_dashboard.py`
- [ ] Run KT benchmark (requires dataset): `python benchmakring/run_benchmark.py --dataset riiid --data /path/to/train.csv`
- [ ] Export results: `python -c "from pipeline.experiment_tracker import ExperimentTracker; ExperimentTracker('eval_output/results.db').export_csv('results.csv')"`
