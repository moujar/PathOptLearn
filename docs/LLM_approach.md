# LLM Approach for PathOptLearn

This document outlines how to integrate **Large Language Models (LLMs)** into PathOptLearn for **AI Learning Path Generation** toward **optimal learning paths**.

---

## Objective (app / project)

- **Primary**: Build an AI system that **generates optimal learning paths** for a given student and learning target (e.g. master a skill set, pass an exam).
- **Modeling**: Use **learning dynamics** (e.g. knowledge tracing, success prediction) and **content representation** (items, skills, prerequisites) to inform path generation.
- **Personalization**: Adapt paths to **student state** (history, knowledge estimate, pace) and **constraints** (max steps, available items, curriculum).
- **Optimality**: Optimize for **learning gain**, **efficiency** (steps/time to mastery), and **engagement**; respect **prerequisites** and pedagogical order.
- **Deliverable**: An approach (and optionally an app or API) that, given student + target + constraints, outputs an **ordered path** (and optionally explanations).

---

## AI Learning Path Generation — Optimal Path

**Goal**: Use AI (e.g. LLMs + learning dynamics) to **generate learning paths** that are **optimal** for a given student and objective.

### What is a learning path?

A **learning path** is an ordered sequence of learning steps (items, exercises, skills, or concepts) that a student is recommended to follow. Each step can include content, difficulty, and expected outcome.

### What makes a path optimal?

Optimality depends on the **objective** and **constraints**:

| Objective | Meaning | Example metric |
|-----------|--------|----------------|
| **Learning gain** | Maximize mastery or skill growth | Post-path assessment score, skill coverage |
| **Efficiency** | Reach target mastery in minimal time/steps | Steps to mastery, time to 80% correct |
| **Engagement** | Keep student motivated, reduce dropout | Completion rate, time on task, boredom/flow |
| **Personalization** | Match path to prior knowledge and pace | Prerequisite satisfaction, difficulty calibration |
| **Curriculum alignment** | Respect prerequisites and pedagogical order | Prerequisite graph consistency, expert ranking correlation |

In practice, **optimal path** = path that maximizes a combination of these (e.g. learning gain + efficiency under a constraint on path length).

### How AI generates the path

1. **Input**: Student state (history, knowledge estimate, preferences), target (e.g. “master skill set S”), optional constraints (max steps, available items).
2. **Model**: AI (LLM + dynamics or RL/ranking) predicts or scores candidate next steps or full paths.
3. **Output**: An ordered path (sequence of items/skills) and optionally explanations.

Methods include: **next-step loop** (greedy), **beam search** over paths, **ranking** of pre-generated paths, or **policy** (e.g. RL) that selects the next step from state.

---

## 1. Why LLMs for Learning Pathways?

- **Rich representations**: LLMs capture semantic structure of content (skills, concepts, difficulty) from text and metadata.
- **Sequence modeling**: They naturally model sequences (e.g. student interaction logs from Riiid!, EdNet, ASSISTments).
- **Flexible inputs**: Can combine item text, skill tags, correctness, and timestamps in a unified representation.
- **Explainability**: Can generate natural-language explanations of recommended next steps or predicted difficulty.

---

## 2. Possible Roles for the LLM

| Role | Description | Fits PathOptLearn |
|------|-------------|-------------------|
| **Student / sequence encoder** | Encode interaction history (items, correctness, time) into a vector for pathway prediction. | ✓ Core |
| **Item / content encoder** | Encode exercise text, skills, and metadata for similarity and prerequisite reasoning. | ✓ Core |
| **Next-step predictor** | Given history, predict next best item or skill (classification or generation). | ✓ Core |
| **Path generator** | Generate or rank full pathways (ordered sequences of items/skills). | ✓ Core |
| **Explanation generator** | Explain why a given step or path is recommended. | ✓ Optional |

---

## 3. Recommended Approach (Phased)

### Phase 1 — Represent content and history with an LLM

- **Item encoding**: Use a pretrained LLM (e.g. BERT, sentence-transformers, or a small LLaMA-style model) to encode:
  - Exercise/question text
  - Skill tags and metadata
- **Sequence encoding**: Feed chronological interaction sequences (item ids or text + correctness + optional time) into a sequence model (Transformer decoder or encoder–decoder) to get a “student state” embedding.
- **Data**: Start with one of Riiid!, EdNet, or ASSISTments; align items to text/skills where available.

### Phase 2 — Predict next step or pathway segment

- **Next-item / next-skill prediction**: Add a small head on top of the sequence representation (e.g. MLP or linear layer) to predict:
  - Next item id, or
  - Next skill, or
  - Probability of success on candidate items.
- **Training**: Supervised learning on historical sequences (e.g. next item as target, or success at next step).
- **Evaluation**: Accuracy/NDCG for next item, AUC for success prediction, consistency with learning curves.

### Phase 3 — Optimal path generation

- **Objective**: Generate **optimal learning paths** (maximize learning gain, efficiency, engagement; respect prerequisites).
- **Path generation**: Greedy (next best step), beam search (top-k paths), path ranking, or RL policy roll-out.
- **Training for optimality**: RL (reward = learning gain / time to mastery), listwise ranking of paths, or imitation from expert/successful student paths.
- **Evaluation**: Learning gain, steps to mastery, prerequisite satisfaction, curriculum/expert alignment.

---

## 4. Technical Choices (Concrete)

- **Base model**: Start with **encoder-only** (BERT, RoBERTa, or sentence-transformers) or **decoder-only** (GPT-style) for sequences; encoder–decoder if you need explicit “content → decision” structure.
- **Scale**: For TER scope, small models (e.g. `bert-base`, `distilbert`, or 100M–500M parameter LMs) are easier to train and evaluate; document compute and data size.
- **Frameworks**: PyTorch + Hugging Face (`transformers`, `datasets`); optional `trl` for RL or preference-based training later.
- **Data**: Use public benchmarks (Riiid! Answer Correctness, EdNet KT, ASSISTments) so results are comparable.

---

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Little or no text in datasets | Use skill/concept ids + metadata as “tokens”; or enrich items with generated/synthetic descriptions. |
| Cold start for new items | Prefer structure (skill graph, prerequisites) + LLM for semantics; use generic “unknown item” embedding. |
| Compute cost | Use small models, fixed context length, and subset of data for experiments. |
| Evaluation gap | Define metrics that reflect learning (e.g. predicted learning gain, alignment with curricula) not only prediction accuracy. |

---

## 6. Minimal Next Steps

1. **Pick one dataset**: Riiid!, EdNet, or ASSISTments; list available fields (item id, skill, text, correctness, timestamp).
2. **Define input format**: How each interaction and each item is turned into token/id sequences for the LLM.
3. **Implement baseline**: Simple next-step predictor (e.g. logistic regression or small MLP on hand-crafted features) to compare against.
4. **Implement LLM baseline**: One encoder for items, one for sequences; one head for next-item or success prediction; train and report metrics.
5. **Iterate**: Add pathway-level objective (RL or ranking) and optional explanation generation.

---

## Research papers (bibliography) — deep research

### Foundational knowledge tracing

- **Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J.** (2015). *Deep Knowledge Tracing.* NeurIPS 2015. arXiv:1506.05908. [First DKT; RNN for student knowledge over time; ASSISTments data.]
- **Corbett, A. T., & Anderson, J. R.** (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278. [Classic Bayesian KT.]
- **Pardos, Z. A., & Heffernan, N. T.** (2010). Modeling individual difference in the effectiveness of learning for student modeling. *UMAP*. [ASSISTments, student modeling.]

### Datasets (education / KT)

- **Choi, Y., Lee, Y., Cho, J., Baek, J., Kim, B., Cha, Y., … & Kim, S.** (2020). *EdNet: A Large-Scale Hierarchical Dataset in Education.* AIED 2020. Springer. arXiv:1912.03072. DOI: 10.1007/978-3-030-52240-7_13. [131M+ interactions; Santa/Riiid; hierarchical; CC-BY-NC.]
- **Riiid Answer Correctness.** Kaggle competition / EdNet KT1–KT4. [Public benchmark for correctness prediction.]
- **Feng, M., Heffernan, N., & Heffernan, C.** (2009). Using learning decomposition to analyze student fluency development. *AIED*. [ASSISTments platform; see also assistments.org for dataset.]

### Transformer / self-attention for KT

- **Pandey, S., & Karypis, G.** (2019). *A Self-Attentive Model for Knowledge Tracing.* EDM 2019. [SAKT; self-attention over past KCs; handles sparsity; +4.43% AUC.]
- **Choi, Y., et al.** (2020). *SAINT+: Integrating Temporal Features for EdNet Correctness Prediction.* LAK 2021. arXiv:2010.12042. ACM. [Encoder–decoder KT; elapsed/lag time; SOTA on EdNet.]
- **Shin, D., Shim, J., Yu, H., Lee, S., & Kim, B.** (2021). SAINT: Separated Self-Attentive Neural Knowledge Tracing. *ICLR 2021* (or prior SAINT variant). [Separate encoder for exercises, decoder for responses.]

### Sequential recommendation (transfer to path/sequence)

- **Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P.** (2019). *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.* CIKM 2019. arXiv:1904.06690. [Cloze-style; bidirectional; sequential rec.]
- **Kang, W.-C., & McAuley, J.** (2018). *Self-Attentive Sequential Recommendation.* ICDM 2018. IEEE. [SASRec; self-attention for next-item prediction.]
- **Chen, Z., Wang, X., et al.** (2022). Learning to teach: Dynamic composition of knowledge for sequential recommendation. *WWW 2022.* [Curriculum / sequencing for recommendation.]

### Learning path recommendation & personalization

- **Liu, Q., Huang, Z., Yin, Y., Chen, E., Xiong, H., Su, Y., & Hu, G.** (2019). *ETUR: Learning to Route.* Or: Learning path recommendation based on knowledge tracing and reinforcement learning. *IEEE* (e.g. ICBK 2020). [KT + RL for path recommendation.]
- **Tang, C., et al.** (2020). *Learning path personalization and recommendation: A survey of the state-of-the-art.* Expert Systems with Applications / *Computers in Human Behavior* or similar. Elsevier. [Survey: personalization, parameters, evaluation.]
- **Wang, Z., et al.** (2025). *Learning Path Recommendation Enhanced by Knowledge Tracing and Large Language Model.* Electronics (MDPI), 14(22), 4385. [LLM as planner; KT as evaluator; knowledge promotion.]
- **Cascading Deep Q-Networks / multi-behavior.** (2024). Learning path recommendation with multi-behavior user modeling. *Knowledge-Based Systems.* Elsevier. [Multi-behavior; DQN for path.]

### Reinforcement learning for education & ITS

- **Rafferty, A. N., Brunskill, E., Griffiths, T. L., & Shafto, P.** (2016). *Faster teaching via POMDP planning.* Cognitive Science. [Policy for teaching; POMDP.]
- **Rafferty, A. N., LaMar, M. M., & Griffiths, T. L.** (2015). *Inferring learners’ knowledge from their actions.* Cognitive Science. [Student modeling + policy.]
- **Doroudi, S., Brunskill, E., & Aleven, V.** (2017). *Sequence matters.* EDM / LAK. [Order of practice; half-life; bandits.]
- **Integrating RL with Dynamic KT.** (2025). *Integrating Reinforcement Learning with Dynamic Knowledge Tracing for personalized learning path optimization.* Nature Scientific Reports / similar. [RL-DKT; path optimization; dropout, completion time.]
- **IBM EDUPLANNER.** Get a head start: On-demand pedagogical policy selection in intelligent tutoring. [Off-policy; policy selection for ITS.]

### Surveys & reviews

- **Shen, S., Liu, Q., Chen, E., Huang, Z., Huang, W., Yin, Y., … & Su, Y.** (2021). *A survey of knowledge tracing: Models, variants, and applications.* IEEE Transactions on Learning Technologies / arXiv:2105.15106. [KT models, variants, applications.]
- **Learning path personalization and recommendation: A survey.** (2020). Elsevier. [State-of-the-art path personalization.]

### LLMs & transformers (tools / education)

- **Hugging Face.** *Transformers*, *Datasets*. https://huggingface.co/docs/transformers, https://huggingface.co/datasets. [BERT, GPT-style; pipelines.]
- **BERT / RoBERTa for KT.** Various EDM/LAK papers on embedding exercises with BERT for knowledge tracing. [Search: "BERT knowledge tracing" or "transformer KT".]

### Curricula & bandits

- **Clement, B., Roy, D., Oudeyer, P.-Y., & Lopes, M.** (2015). Multi-armed bandits for intelligent tutoring systems. *Journal of Educational Data Mining.* [Bandits for item selection.]
- **Brunskill, E., & Russell, S.** (2016). *Reinforcement learning of pedagogical policies.* Adaptive and Intelligent Educational Systems. [RL for pedagogy.]

---

*For the TER report: use full author lists, DOIs, and URLs (e.g. arXiv, ACM, IEEE Xplore, Springer, Elsevier) from the publisher or Google Scholar.*
