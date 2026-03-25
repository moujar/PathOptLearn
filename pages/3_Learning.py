import streamlit as st
import re
import json
import os
import sys
import tempfile
import requests
import yt_dlp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db import log_progress, get_course_progress

os.environ['PATH'] = r'C:\ffmpeg\bin' + os.pathsep + os.environ.get('PATH', '')

st.set_page_config(page_title="Adaptive Learning", page_icon="🎓", layout="wide")

# ── CONFIG ──────────────────────────────────────────────────
API_URL      = "https://dulotic-fumigatory-romona.ngrok-free.dev"
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"

CACHE_DIR = os.path.join(tempfile.gettempdir(), "transcript_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── AUTH GUARD ───────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.switch_page("pages/1_Login.py")

if not st.session_state.get("course_id"):
    st.switch_page("pages/2_Dashboard.py")

student_id  = st.session_state["student_id"]
course_id   = st.session_state["course_id"]
course_name = st.session_state.get("course_name", "My Course")


# ── TRANSCRIPT HELPERS ──────────────────────────────────────
def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([^&]+)',
        r'(?:youtu\.be\/)([^?]+)',
        r'(?:youtube\.com\/embed\/)([^?]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.strip()

def get_video_title(video_id):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False
            )
            return info.get('title', video_id)
    except:
        return video_id

def get_transcript_captions(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'subtitles' in info and 'en' in info['subtitles']:
                sub_url = info['subtitles']['en'][0]['url']
            elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                sub_url = info['automatic_captions']['en'][0]['url']
            else:
                return None
            import urllib.request
            with urllib.request.urlopen(sub_url) as response:
                data = json.loads(response.read())
            transcript = []
            for event in data.get('events', []):
                if 'segs' in event:
                    text = ''.join([seg.get('utf8', '') for seg in event['segs']])
                    if text.strip():
                        transcript.append({
                            'text': text.strip(),
                            'start': event.get('tStartMs', 0) / 1000,
                        })
            return transcript
    except:
        return None

def get_transcript_whisper(video_id):
    cache_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_dir = tempfile.gettempdir()
    output_file = os.path.join(temp_dir, f'{video_id}_audio.m4a')
    st.info("🎵 Downloading audio...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': r'C:\ffmpeg\bin',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not os.path.exists(output_file):
            raise Exception("Audio download failed")
        st.info("🎤 Transcribing with Whisper...")
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(output_file, fp16=False)
        transcript = [
            {'text': seg['text'].strip(), 'start': seg['start']}
            for seg in result['segments']
        ]
        with open(cache_file, 'w') as f:
            json.dump(transcript, f)
        os.remove(output_file)
        return transcript
    except Exception as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise Exception(f"Failed: {str(e)}")


# ── COLAB API HELPERS ───────────────────────────────────────
def call_generate_api(transcript_text, num_questions, temperature):
    endpoint = API_URL.rstrip("/") + "/generate"
    payload  = {
        "transcript":    transcript_text,
        "num_questions": num_questions,
        "temperature":   temperature,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("questions", []), None
    except requests.exceptions.ConnectionError:
        return [], "Cannot reach the API. Is your Colab notebook running?"
    except requests.exceptions.Timeout:
        return [], "Request timed out. Try again."
    except Exception as e:
        return [], f"Error: {str(e)}"

def call_evaluate_api(transcript_text, questions, student_answers, preference):
    endpoint = API_URL.rstrip("/") + "/evaluate"
    payload  = {
        "transcript":      transcript_text,
        "questions":       questions,
        "student_answers": student_answers,
        "preference":      preference,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is your Colab notebook running?"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Try again."
    except Exception as e:
        return None, f"Error: {str(e)}"



# ── GROQ GENERATION & EVALUATION ────────────────────────────
def groq_api_call(prompt, max_tokens=800, temperature=0.7):
    """Single reusable Groq call. Returns response text or None."""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None


def groq_generate_questions(transcript_text, num_questions=5):
    """Generate MCQ questions from transcript using Groq. Returns (questions, error)."""
    chunk_size = 2800
    overlap    = 200
    chunks     = []
    start      = 0
    while start < len(transcript_text):
        chunks.append(transcript_text[start:start + chunk_size])
        start += chunk_size - overlap

    questions_per_chunk = [num_questions // len(chunks)] * len(chunks)
    for i in range(num_questions % len(chunks)):
        questions_per_chunk[i] += 1

    all_questions = []

    for chunk_idx, (chunk, n_q) in enumerate(zip(chunks, questions_per_chunk)):
        for i in range(n_q):
            prompt = f"""You are an educational quiz generator.

Read the following passage and generate ONE multiple-choice question that tests CONCEPTUAL UNDERSTANDING, not just recall.

Passage:
\"\"\"{ chunk.strip()}\"\"\"

Rules:
- The question must require thinking, not just copying text
- Provide exactly 4 choices labeled A, B, C, D
- Only one choice is correct
- Reply ONLY with valid JSON, no explanation:

{{"question": "...", "choices": ["choice A", "choice B", "choice C", "choice D"], "correct": 0}}

The "correct" field is the 0-based index of the correct answer."""

            content = groq_api_call(prompt, max_tokens=300, temperature=0.7)
            if not content:
                continue
            try:
                cleaned = re.sub(r"```json|```", "", content).strip()
                parsed  = json.loads(cleaned)
                if "question" in parsed and "choices" in parsed and "correct" in parsed:
                    if len(parsed["choices"]) == 4:
                        all_questions.append(parsed)
            except:
                continue

    if not all_questions:
        return [], "Groq returned no valid questions. Try again."
    return all_questions, None


def groq_evaluate(transcript_text, questions, student_answers, preference):
    """Evaluate student answers using Groq. Returns (eval_result_dict, error)."""
    score = 0
    wrong = []
    for q, ans in zip(questions, student_answers):
        correct_ans = q["choices"][q["correct"]]
        if ans == correct_ans:
            score += 1
        else:
            wrong.append(f"Q: {q['question']}\n  Student: {ans}\n  Correct: {correct_ans}")

    total  = len(questions)
    passed = score >= total * 0.6
    pct    = int(score / total * 100) if total > 0 else 0
    wrong_text = "\n".join(wrong) if wrong else "None — all correct."

    prompt = f"""You are an educational assessment system.

A student just watched a video and answered a quiz.

Topic summary: {transcript_text[:400]}

Score: {score}/{total} ({pct}%)
Questions answered incorrectly:
{wrong_text}

Student prefers: {preference}

Write:
1. A 2-sentence EVALUATION of the student's performance, mentioning specific gaps if any.
2. A 1-sentence RECOMMENDATION for what to do next, based on their preference for {preference}.

Format exactly like this:
EVALUATION: <your evaluation here>
RECOMMENDATION: <your recommendation here>"""

    content = groq_api_call(prompt, max_tokens=200, temperature=0.5)

    evaluation     = ""
    recommendation = ""
    if content:
        for line in content.split("\n"):
            if line.startswith("EVALUATION:"):
                evaluation = line.replace("EVALUATION:", "").strip()
            elif line.startswith("RECOMMENDATION:"):
                recommendation = line.replace("RECOMMENDATION:", "").strip()

    if not evaluation:
        evaluation = f"The student scored {pct}% on this quiz."
    if not recommendation:
        recommendation = "Ready to advance." if passed else "Consider reviewing this topic."

    return {
        "evaluation":     evaluation,
        "recommendation": recommendation,
        "score":          score,
        "total":          total,
        "passed":         passed,
    }, None


def generate_questions(transcript_text, num_questions, temperature):
    """Router: calls Colab or Groq based on provider setting."""
    if st.session_state.get("provider", "our_models") == "groq":
        return groq_generate_questions(transcript_text, num_questions)
    return call_generate_api(transcript_text, num_questions, temperature)


def evaluate_answers(transcript_text, questions, student_answers, preference):
    """Router: calls Colab or Groq based on provider setting."""
    if st.session_state.get("provider", "our_models") == "groq":
        return groq_evaluate(transcript_text, questions, student_answers, preference)
    return call_evaluate_api(transcript_text, questions, student_answers, preference)

# ── YOUTUBE SEARCH ──────────────────────────────────────────
def search_youtube(query, max_results=10):
    """
    Search YouTube using yt-dlp. Returns list of candidate dicts:
    {video_id, title, description, has_captions, transcript_snippet}
    """
    candidates = []
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(
                f"ytsearch{max_results}:{query}",
                download=False
            )
            for entry in results.get('entries', []):
                if not entry:
                    continue
                video_id = entry.get('id', '')
                if not video_id:
                     continue

    # Skip unavailable videos before even checking captions
                try:
                   with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl_check:
                     check = ydl_check.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                 )
                   if not check:
                    continue
                except:
                    continue  # Video unavailable, skip it

                title       = entry.get('title', '')
                description = entry.get('description', '') or ''

                # Check for captions
                transcript  = get_transcript_captions(video_id)
                has_captions = transcript is not None

                snippet = ""
                if has_captions and transcript:
                    snippet = " ".join([t['text'] for t in transcript])[:1500]
                else:
                    snippet = f"Title: {title}\nDescription: {description[:500]}"

                candidates.append({
                    "video_id":     video_id,
                    "title":        title,
                    "description":  description[:300],
                    "has_captions": has_captions,
                    "snippet":      snippet,
                })
    except Exception as e:
        st.warning(f"YouTube search error: {e}")

    return candidates


# ── GROQ RECOMMENDATION ─────────────────────────────────────
def extract_topic(current_transcript):
    """
    Step 1: One focused Groq call — extract the specific topic of the video in 3-5 words.
    This is the anchor that keeps the recommendation on topic no matter what.
    """
    prompt = f"""What is the specific educational topic of this video transcript?
Answer with ONLY 3-5 words, nothing else. Be specific (e.g. "calculus derivatives introduction" not "math tutorial").

Transcript excerpt:
\"\"\"{current_transcript[:600]}\"\"\""""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 15,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        topic = resp.json()["choices"][0]["message"]["content"].strip().strip('"').lower()
        return topic
    except Exception as e:
        st.warning(f"Topic extraction error: {e}")
        return None


def build_search_query(current_transcript, eval_result, questions, user_answers):
    """
    Step 2: Build the query programmatically — topic is locked, weak points steer difficulty.
    Groq only does one small focused task (topic extraction) instead of generating the full query.
    """
    # Extract the topic anchor first
    topic = extract_topic(current_transcript)
    if not topic:
        # Last resort fallback — first 4 meaningful words of transcript
        words = [w for w in current_transcript.split() if len(w) > 3][:4]
        topic = " ".join(words)

    # Find the specific concepts the student got wrong
    weak_concepts = []
    for q, student_ans in zip(questions, user_answers):
        correct_ans = q['choices'][q['correct']]
        if student_ans != correct_ans:
            # Extract the key concept from the question (first 6 words)
            concept = " ".join(q['question'].split()[:6])
            weak_concepts.append(concept)

    if eval_result["passed"]:
        query = f"{topic} advanced"
    else:
        query = f"{topic} beginner explained"

    # Keep it clean and under 8 words
    query_words = query.split()[:8]
    return " ".join(query_words)


def groq_pick_best_video(candidates, eval_result, preference, topic):
    """
    Send all candidates to Groq and let it pick the best one.
    Returns {video_id, title, reason} or None.
    """
    if not candidates:
        return None

    # Keep snippets short to avoid 400
    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        source = "has transcript" if c["has_captions"] else "metadata only"
        candidates_text += f"{i}. [{source}] {c['title']} — {c['snippet'][:200]}\n"

    # Build title → video_id map so we can fix mismatched IDs from Groq
    id_map = {c['title']: c['video_id'] for c in candidates}

    prompt = f"""Pick the best YouTube video for a student who {'passed' if eval_result['passed'] else 'failed'} a quiz (score {eval_result['score']}/{eval_result['total']}).
The video MUST be about: {topic}
They prefer {preference}. Prefer videos with transcripts.
{'Choose a more advanced follow-up.' if eval_result['passed'] else 'Choose a simpler review video.'}
Reject any video that is not clearly about the topic above.

Candidates:
{candidates_text}

Reply ONLY with JSON: {{"video_id": "...", "title": "...", "reason": "one sentence"}}
Use the exact title from the list above."""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"```json|```", "", content).strip()
        result  = json.loads(content)

        # Fix video_id if Groq returned the right title but wrong ID
        if result.get("title") in id_map:
            result["video_id"] = id_map[result["title"]]

        return result
    except Exception as e:
        st.warning(f"Groq pick error: {e}")
        # Fallback: filter by topic keyword first, then take first captioned
        topic_keywords = topic.lower().split()
        for c in candidates:
            title_lower = c["title"].lower()
            if any(kw in title_lower for kw in topic_keywords) and c["has_captions"]:
                return {"video_id": c["video_id"], "title": c["title"], "reason": "Best on-topic match with captions."}
        for c in candidates:
            if any(kw in c["title"].lower() for kw in topic_keywords):
                return {"video_id": c["video_id"], "title": c["title"], "reason": "Best on-topic match."}
        # Last resort
        if candidates:
            return {"video_id": candidates[0]["video_id"], "title": candidates[0]["title"], "reason": "Best available match."}
        return None


def recommend_next_video(current_transcript, eval_result, preference, questions, user_answers):
    """Full recommendation pipeline."""
    # Step 1: Lock the topic first
    with st.spinner("🔍 Extracting video topic..."):
        topic = extract_topic(current_transcript)
    st.caption(f"Topic: *{topic}*")

    # Step 2: Build search query from topic + weak points
    query = build_search_query(current_transcript, eval_result, questions, user_answers)
    st.caption(f"Search query: *{query}*")

    # Step 3: Search YouTube
    with st.spinner(f"🎬 Searching YouTube..."):
        candidates = search_youtube(query, max_results=10)

    if not candidates:
        return None

    st.caption(f"Found {len(candidates)} candidates, {sum(1 for c in candidates if c['has_captions'])} with captions.")

    # Step 4: Groq picks best — topic is passed to keep it grounded
    with st.spinner("🤖 Groq is picking the best video..."):
        recommendation = groq_pick_best_video(candidates, eval_result, preference, topic)

    return recommendation


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.get('username', 'Student')}")
    st.markdown(f"**Course:** {course_name}")
    st.markdown("---")

    st.markdown("**🤖 AI Provider**")
    provider_label = st.selectbox(
        "Model provider",
        options=["Our Models (Colab)", "Groq (Cloud — Free)"],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state["provider"] = "groq" if provider_label == "Groq (Cloud — Free)" else "our_models"
    if st.session_state["provider"] == "groq":
        st.caption("Using Groq — no Colab needed.")
    else:
        st.caption("Using fine-tuned Mistral via Colab.")
    st.markdown("---")

    rows = get_course_progress(student_id, course_id)
    if rows:
        st.markdown(f"**{len(rows)} video(s) in this course**")
        for i, (vid_id, title, score, total, passed, timestamp) in enumerate(rows, 1):
            pct    = int(score / total * 100) if total > 0 else 0
            status = "✅" if passed else "🔄"
            with st.expander(f"{status} {i}. {title[:30]}..."):
                st.markdown(f"**Score:** {score}/{total} ({pct}%)")
                st.markdown(f"**Time:** {timestamp}")
    else:
        st.info("No videos watched yet.")

    st.markdown("---")
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("pages/2_Dashboard.py")


# ── MAIN UI ─────────────────────────────────────────────────
st.title(f"📚 {course_name}")
st.markdown("Watch a video, take a quiz, get personalized feedback and your next video.")
st.markdown("---")

# ── STEP 1: Transcript ──────────────────────────────────────
st.header("Step 1 — Watch & Extract Transcript")

# If a recommendation was made, pre-fill the input
default_url = ""
if st.session_state.get("recommended_video_id"):
    default_url = f"https://www.youtube.com/watch?v={st.session_state['recommended_video_id']}"

video_input    = st.text_input("YouTube URL or Video ID:", value=default_url, placeholder="WUvTyaaNkzM")
extract_button = st.button("🔍 Get Transcript", type="primary")

if extract_button and video_input:
    try:
        video_id = extract_video_id(video_input)
        st.info(f"📹 Video ID: `{video_id}`")
        with st.spinner("Processing..."):
            transcript = get_transcript_captions(video_id)
            if transcript:
                st.success("✅ Found captions!")
            else:
                st.warning("No captions — using Whisper (1-2 min)...")
                transcript = get_transcript_whisper(video_id)
        full_text = " ".join([t['text'] for t in transcript])
        title     = get_video_title(video_id)

        st.session_state["transcript"]          = full_text
        st.session_state["transcript_segments"] = transcript
        st.session_state["video_id"]            = video_id
        st.session_state["video_title"]         = title
        st.session_state["questions"]           = None
        st.session_state["eval_result"]         = None
        st.session_state["recommendation"]      = None
        st.session_state["recommended_video_id"] = None
    except Exception as e:
        st.error(f"❌ {str(e)}")
elif extract_button:
    st.warning("⚠️ Enter a video URL or ID first.")


if st.session_state.get("transcript"):
    video_id   = st.session_state["video_id"]
    full_text  = st.session_state["transcript"]
    transcript = st.session_state["transcript_segments"]

    st.subheader("📺 Video")
    st.video(f"https://www.youtube.com/watch?v={video_id}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Segments", len(transcript))
    with col2:
        st.metric("Words", len(full_text.split()))

    with st.expander("Full Transcript", expanded=False):
        st.text_area("Transcript", value=full_text, height=250, label_visibility="collapsed")

    st.markdown("---")

    # ── STEP 2: Generate Quiz ───────────────────────────────
    st.header("Step 2 — Generate Quiz")

    col_a, col_b = st.columns(2)
    with col_a:
        num_questions = st.slider("Number of questions", min_value=1, max_value=10, value=5)
    with col_b:
        temperature = st.slider("Creativity", min_value=0.5, max_value=1.2, value=0.8, step=0.05)

    if st.button("🧠 Generate Questions", type="primary"):
        with st.spinner(f"Generating {num_questions} questions..."):
            questions, error = generate_questions(full_text, num_questions, temperature)
        if error:
            st.error(f"❌ {error}")
        elif not questions:
            st.warning("No questions returned. Try a longer transcript.")
        else:
            st.session_state["questions"]      = questions
            st.session_state["eval_result"]    = None
            st.session_state["recommendation"] = None
            st.success(f"✅ Generated {len(questions)} questions!")


# ── STEP 3: Quiz ────────────────────────────────────────────
if st.session_state.get("questions"):
    questions = st.session_state["questions"]

    st.markdown("---")
    st.header("Step 3 — Take the Quiz")

    preference = st.radio(
        "I prefer to learn via:",
        options=["videos", "articles"],
        horizontal=True,
    )

    with st.form("quiz_form"):
        user_answers = []
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            answer = st.radio(
                label=f"Select answer for question {i+1}",
                options=q['choices'],
                key=f"q_{i}",
                label_visibility="collapsed",
            )
            user_answers.append(answer)
            st.markdown("")

        submitted = st.form_submit_button("✅ Submit & Evaluate")

    if submitted:
        # Per-question feedback
        st.subheader("📊 Your Answers")
        for i, (q, user_ans) in enumerate(zip(questions, user_answers)):
            correct_ans = q['choices'][q['correct']]
            if user_ans == correct_ans:
                st.success(f"Q{i+1}: ✅ Correct — {correct_ans}")
            else:
                st.error(f"Q{i+1}: ❌ Wrong. You chose: **{user_ans}** | Correct: **{correct_ans}**")

        # Evaluation
        st.markdown("---")
        st.subheader("🤖 AI Evaluation")
        with st.spinner("Evaluating your performance..."):
            eval_result, error = evaluate_answers(
                full_text, questions, user_answers, preference
            )

        if error:
            st.error(f"❌ Evaluation failed: {error}")
        else:
            st.session_state["eval_result"] = eval_result

            score = eval_result["score"]
            total = eval_result["total"]
            pct   = int(score / total * 100)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score} / {total}")
            with col2:
                st.metric("Percentage", f"{pct}%")
            with col3:
                status = "✅ Ready to advance" if eval_result["passed"] else "🔄 Review recommended"
                st.metric("Status", status)

            if eval_result["evaluation"]:
                st.markdown("**Evaluation:**")
                st.info(eval_result["evaluation"])

            if eval_result["recommendation"]:
                st.markdown("**Recommendation:**")
                if eval_result["passed"]:
                    st.success(eval_result["recommendation"])
                else:
                    st.warning(eval_result["recommendation"])

            # Log to SQLite
            log_progress(
                student_id = student_id,
                course_id  = course_id,
                video_id   = st.session_state["video_id"],
                title      = st.session_state.get("video_title", video_id),
                score      = score,
                total      = total,
                passed     = eval_result["passed"],
            )

            if pct == 100:
                st.balloons()

            # ── STEP 4: Recommend next video ───────────────
            st.markdown("---")
            st.header("Step 4 — Next Video")

            recommendation = recommend_next_video(full_text, eval_result, preference, questions, user_answers)

            if recommendation:
                st.session_state["recommendation"]       = recommendation
                st.session_state["recommended_video_id"] = recommendation["video_id"]

                st.success(f"**Recommended:** {recommendation['title']}")
                st.markdown(f"*{recommendation['reason']}*")
                st.video(f"https://www.youtube.com/watch?v={recommendation['video_id']}")

                if st.button("▶ Load this video", type="primary"):
                    st.session_state["transcript"]          = None
                    st.session_state["transcript_segments"] = None
                    st.session_state["questions"]           = None
                    st.session_state["eval_result"]         = None
                    st.session_state["recommendation"]      = None
                    st.rerun()
            else:
                st.warning("Could not find a suitable next video. Try refreshing.")
