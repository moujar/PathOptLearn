erDiagram
    STUDENTS {
        INTEGER id PK
        TEXT username UK
        TEXT email UK
        TEXT password_hash
        INTEGER verified
        TEXT verification_code
        TEXT code_expiry
        TEXT created_at
    }

    COURSES {
        INTEGER id PK
        INTEGER student_id FK
        TEXT name
        TEXT description
        TEXT playlist_url
        TEXT created_at
    }

    ASSESSMENTS {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT level
        INTEGER score
        TEXT taken_at
    }

    ASSESSMENT_ANSWERS {
        INTEGER id PK
        INTEGER assessment_id FK
        TEXT question
        TEXT concept
        TEXT student_answer
        TEXT correct_answer
        INTEGER is_correct
    }

    KNOWLEDGE_GAPS {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT topic_name
        TEXT severity
        TEXT source
        TEXT identified_at
        TEXT resolved_at
    }

    ROADMAP_MODULES {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        INTEGER module_number
        TEXT title
        TEXT objective
        TEXT concepts_json
        TEXT duration
        TEXT status
    }

    TOPIC_MASTERY {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT topic_name
        REAL mastery_score
        INTEGER attempt_count
        INTEGER pass_count
        TEXT last_updated
    }

    CONCEPT_RESOURCES {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT concept_name
        TEXT resource_type
        TEXT url
        TEXT title
        TEXT channel
        TEXT duration
        INTEGER views
        TEXT summary
        TEXT metadata_json
        TEXT fetched_at
    }

    PROGRESS {
        INTEGER id PK
        INTEGER student_id FK
        INTEGER course_id FK
        INTEGER module_id FK
        TEXT video_id
        TEXT title
        INTEGER score
        INTEGER total
        INTEGER passed
        INTEGER attempt_number
        TEXT timestamp
    }

    STUDENTS ||--o{ COURSES : creates
    STUDENTS ||--o{ ASSESSMENTS : takes
    STUDENTS ||--o{ KNOWLEDGE_GAPS : has
    STUDENTS ||--o{ TOPIC_MASTERY : tracks
    STUDENTS ||--o{ CONCEPT_RESOURCES : researches
    STUDENTS ||--o{ PROGRESS : records
    COURSES ||--o{ ASSESSMENTS : has
    COURSES ||--o{ KNOWLEDGE_GAPS : scoped_to
    COURSES ||--o{ ROADMAP_MODULES : contains
    COURSES ||--o{ TOPIC_MASTERY : scoped_to
    COURSES ||--o{ CONCEPT_RESOURCES : scoped_to
    ASSESSMENTS ||--o{ ASSESSMENT_ANSWERS : has
    ROADMAP_MODULES ||--o{ PROGRESS : linked_to

