import streamlit as st
import re
import json
import os
import tempfile
import requests
import yt_dlp

os.environ['PATH'] = r'C:\ffmpeg\bin' + os.pathsep + os.environ.get('PATH', '')

st.set_page_config(page_title="Adaptive Learning", page_icon="🎓", layout="wide")

# ── CONFIG ─────────────────────────────────────────────────
API_URL = "https://dulotic-fumigatory-romona.ngrok-free.dev"

CACHE_DIR = os.path.join(tempfile.gettempdir(), "transcript_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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


# ── API HELPERS ─────────────────────────────────────────────
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
    except requests.exceptions.HTTPError as e:
        return [], f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return [], f"Unexpected error: {str(e)}"


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
    except requests.exceptions.HTTPError as e:
        return None, f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ── UI ──────────────────────────────────────────────────────
st.title("🎓 Adaptive Learning System")
st.markdown("Watch a video, take a quiz, get personalized feedback.")
st.markdown("---")

# ── STEP 1: Transcript ──────────────────────────────────────
st.header("Step 1 — Extract Transcript")

video_input    = st.text_input("YouTube URL or Video ID:", placeholder="WUvTyaaNkzM")
extract_button = st.button("🔍 Get Transcript", type="primary")

if extract_button and video_input:
    try:
        video_id = extract_video_id(video_input)
        st.info(f"📹 Video ID: `{video_id}`")
        with st.spinner("Processing..."):
            st.info("⚡ Checking for captions...")
            transcript = get_transcript_captions(video_id)
            if transcript:
                st.success("✅ Found captions!")
            else:
                st.warning("No captions — using Whisper (1-2 min)...")
                transcript = get_transcript_whisper(video_id)
        full_text = " ".join([t['text'] for t in transcript])
        st.session_state["transcript"]          = full_text
        st.session_state["transcript_segments"] = transcript
        st.session_state["video_id"]            = video_id
        st.session_state["questions"]           = None
        st.session_state["eval_result"]         = None
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

    st.subheader("📝 Transcript")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Segments", len(transcript))
    with col2:
        st.metric("Words", len(full_text.split()))

    with st.expander("Full Text", expanded=False):
        st.text_area("Transcript text", value=full_text, height=300, label_visibility="collapsed")

    with st.expander("Timestamped"):
        for t in transcript[:50]:
            m = int(t['start'] // 60)
            s = int(t['start'] % 60)
            st.markdown(f"**[{m:02d}:{s:02d}]** {t['text']}")

    st.download_button("⬇️ Download Transcript", full_text, f"transcript_{video_id}.txt")
    st.markdown("---")

    # ── STEP 2: Generate Quiz ───────────────────────────────
    st.header("Step 2 — Generate Quiz")

    col_a, col_b = st.columns(2)
    with col_a:
        num_questions = st.slider("Number of questions", min_value=1, max_value=10, value=5)
    with col_b:
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.5, max_value=1.2, value=0.8, step=0.05,
        )

    if st.button("🧠 Generate Questions", type="primary"):
        with st.spinner(f"Generating {num_questions} questions..."):
            questions, error = call_generate_api(full_text, num_questions, temperature)
        if error:
            st.error(f"❌ {error}")
        elif not questions:
            st.warning("No questions returned. Try a longer transcript.")
        else:
            st.session_state["questions"]   = questions
            st.session_state["eval_result"] = None
            st.success(f"✅ Generated {len(questions)} questions!")


# ── STEP 3: Take the Quiz ───────────────────────────────────
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
        # Per-question results
        st.subheader("📊 Your Answers")
        for i, (q, user_ans) in enumerate(zip(questions, user_answers)):
            correct_ans = q['choices'][q['correct']]
            if user_ans == correct_ans:
                st.success(f"Q{i+1}: ✅ Correct — {correct_ans}")
            else:
                st.error(f"Q{i+1}: ❌ Wrong. You chose: **{user_ans}** | Correct: **{correct_ans}**")

        # Evaluation model
        st.markdown("---")
        st.subheader("🤖 AI Evaluation")
        with st.spinner("Evaluating your performance..."):
            eval_result, error = call_evaluate_api(
                st.session_state["transcript"],
                questions,
                user_answers,
                preference,
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

            if pct == 100:
                st.balloons()
