```

You enter a topic
      │
      ▼
  [1] Check topic on Knowledge Graph  OR  [2] Search DuckDuckGo to enrich context
      │
      ▼
Diagnostic Quiz — 8 questions (MCQ + short-answer, mixed difficulty)
      │
      ▼
Submit answers → LLM analyses level + identifies gaps
      │
      ▼
  [1] Fetch resources from KG if they exist  OR  [2] Deep web search (DuckDuckGo + edu sites + YouTube)
      │
      ▼
Store enriched resources in Knowledge Graph
      │
      ▼
Personalised roadmap generated (4–6 modules, progressive)
      │
      ▼
For each module:
  AI-generated lesson  →  5-question quiz  →  score ≥ 70%  →  next module
                                           └→  score < 70%  →  find gaps
                                                                    │
                                                                    ▼
                                                         suggest best resources for gaps
                                                                    │
                                                                    ▼
                                                              review & retry
      │
      ▼
 All modules complete 🏆


```
/CreateUser
/DeepSearch [input[topic] -> output[links, metadata, summaryzation for each link]] put on KG
/GeneaterQuiz [input[topic or topics ] -> output[MCQ]] check KG from db and based to geneater MCQ
/findGaps [input[MCQ + ansswer] -> output[gaps]]
/recommader[input [gaps] -> output[best resources]]



Student answers quiz
        │
   /find-gaps
        │
   [knowledge_gaps table]
   [gap: "gradient descent", severity: "high"]
        │
   /recommender?gaps=gradient descent
        │
        ├── concept_resources (cache hit?) ──► LLM ranks ──► top 3 resources
        │         cache miss ↓
        │   search_web + search_youtube
        │   fetch each page → LLM summary → save
        │
        └── returns [{url, title, summary, reason}] per gap

Student completes a module (score saved to session_progress)
        │
   /recommend/next
        │
        ├── Neo4j: find unlocked modules, score by concepts + weak_matches
        ├── PostgreSQL: load history, derive weak_concepts from low scores
        └── LLM: pick best candidate → {recommended_uid, reason, learning_tip}
