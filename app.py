import io
import os
import queue
import re
import time
import wave
from collections import deque

import numpy as np
import streamlit as st
from groq import Groq
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Voice Interview Assistant", page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def frame_to_mono(frame):
    """Convert an av.AudioFrame to a mono float32 numpy array in [-1, 1]."""
    arr = frame.to_ndarray()
    n_ch = max(1, len(frame.layout.channels))
    if frame.format.is_planar:
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
    else:
        mono = arr.reshape(-1, n_ch).mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        mono = mono / float(np.iinfo(arr.dtype).max + 1)
    return mono.astype(np.float32), frame.sample_rate, n_ch, frame.format.name


def resample(x, sr, target=16000):
    if sr == target or len(x) == 0:
        return x
    if sr % target == 0:
        f = sr // target
        n = (len(x) // f) * f
        return x[:n].reshape(-1, f).mean(axis=1).astype(np.float32)
    n = int(len(x) * target / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_wav_bytes(audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class VADEngine:
    """Energy-based voice activity detector with an adaptive noise floor.

    Runs entirely in the Streamlit script thread (frames are pulled from the
    WebRTC receiver), so no locks are needed.
    """

    def __init__(self):
        self.min_level = 0.015
        self.silence_limit = 1.2
        self.min_speech = 0.5
        self.max_utterance = 30.0
        self.preroll_seconds = 0.4
        self.reset_runtime()

    def reset_runtime(self):
        self.frames_total = 0
        self.sample_rate = None
        self.channels = None
        self.fmt = ""
        self.level = 0.0
        self.noise_floor = 0.005
        self.speaking = False
        self.buf = []
        self.preroll = deque()
        self.preroll_dur = 0.0
        self.voiced = 0.0
        self.silence = 0.0
        self.total = 0.0
        self.utterances = deque()
        self.last_diag = 0.0

    def threshold(self):
        return max(self.min_level, self.noise_floor * 2.5)

    def feed(self, frames):
        for frame in frames:
            x, sr, ch, fmt = frame_to_mono(frame)
            if x.size == 0:
                continue
            self.frames_total += 1
            self.sample_rate, self.channels, self.fmt = sr, ch, fmt
            dur = x.size / float(sr)
            rms = float(np.sqrt(np.mean(np.square(x))))
            self.level = 0.6 * self.level + 0.4 * rms
            voiced = rms > self.threshold()

            if not self.speaking:
                if voiced:
                    self.speaking = True
                    self.buf = [a for a, _ in self.preroll] + [x]
                    self.voiced = dur
                    self.silence = 0.0
                    self.total = sum(d for _, d in self.preroll) + dur
                    self.preroll.clear()
                    self.preroll_dur = 0.0
                else:
                    self.noise_floor = max(0.001, 0.995 * self.noise_floor + 0.005 * rms)
                    self.preroll.append((x, dur))
                    self.preroll_dur += dur
                    while self.preroll_dur > self.preroll_seconds and self.preroll:
                        _, d = self.preroll.popleft()
                        self.preroll_dur -= d
            else:
                self.buf.append(x)
                self.total += dur
                if voiced:
                    self.voiced += dur
                    self.silence = 0.0
                else:
                    self.silence += dur
                if self.silence >= self.silence_limit or self.total >= self.max_utterance:
                    self._finish()

    def _finish(self):
        if self.voiced >= self.min_speech and self.buf:
            self.utterances.append((np.concatenate(self.buf), self.sample_rate))
        self.speaking = False
        self.buf = []
        self.voiced = 0.0
        self.silence = 0.0
        self.total = 0.0

    def new_speech_detected(self):
        """True when the user has clearly started a new utterance."""
        return self.speaking and self.voiced >= 0.35


# ----------------------------------------------------------------------------
# Groq helpers
# ----------------------------------------------------------------------------
NOISE_TRANSCRIPTS = {
    "", "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok",
    "thank you for watching", "so", "uh", "um", "hmm",
}


def is_noise(text):
    clean = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return clean in NOISE_TRANSCRIPTS


def transcribe(client, audio, sr, model, language):
    audio16 = resample(audio, sr, 16000)
    wav = to_wav_bytes(audio16, 16000)
    kwargs = dict(
        file=("speech.wav", wav),
        model=model,
        response_format="json",
        temperature=0.0,
    )
    if language.strip():
        kwargs["language"] = language.strip()
    res = client.audio.transcriptions.create(**kwargs)
    return (res.text or "").strip()


LENGTH_RULES = {
    "Short (2-3 sentences)": "Answer in 2-3 sentences.",
    "Medium (4-6 sentences)": "Answer in 4-6 sentences.",
    "Detailed (short paragraphs)": "Answer in up to 2 short paragraphs.",
}


STYLE_RULES = (
    "VOICE AND STYLE\n"
    "- Sound like a real student talking in an interview, not like an AI. Use plain everyday words "
    "and contractions (I'm, I've, it's, didn't).\n"
    "- Start with the substance. Never open with filler like 'Great question', 'Certainly', "
    "'Absolutely' or 'Sure'.\n"
    "- Avoid buzzwords and inflated phrases: leverage, utilize, delve, robust, seamless, cutting-edge, "
    "game-changer, passionate about, 'in today's fast-paced world', 'I am excited to'.\n"
    "- In spoken answers use no bullet points, headings, bold text or em dashes. Mix short and "
    "medium sentences the way people actually talk.\n"
    "- Be specific. Name a real project, tool, or result from the resume instead of making general "
    "claims. If the candidate lacks direct experience, say so honestly and connect it to the closest "
    "thing they have done.\n"
    "- Do not end with a summary line, a question back to the interviewer, or an offer of more help.\n"
    "- Before replying, silently re-read your draft and rewrite any sentence that sounds scripted, "
    "generic, or like a textbook. Output only the final version.\n"
)

CODE_RULES = (
    "WHEN THE INTERVIEWER ASKS FOR CODE\n"
    "If the question asks the candidate to write, show, implement, fix, or explain code or a query "
    "(for example 'write a function to...', 'can you code this', 'show me the logic', 'explain this "
    "code'), ignore the length limit and reply in exactly this order:\n"
    "1. One or two spoken sentences explaining the approach in simple words.\n"
    "2. The complete, correct, runnable code in a single fenced Python code block. "
    "ALWAYS use Python, regardless of the programming language mentioned by the interviewer. "
    "Do not include comments inside the code. Keep the code simple, readable, and use clear variable names.\n"
    "3. A numbered, step-by-step explanation of how the Python code works, worded naturally "
    "as the candidate would explain it aloud.\n"
    "4. One line for time complexity and one line for space complexity, wherever applicable.\n"
    "For every other question, give a normal spoken answer and do not include code unless asked.\n"
)


def build_messages(question, resume_text, history, role, length_rule):
    system = (
        "You are an interview response assistant helping a job candidate during a live interview. "
        "Act as the candidate, not as an interviewer or interview coach. "
        "Write exactly what the candidate should say, in first person, as if answering the interviewer directly.\n\n"
        f"LENGTH FOR NORMAL ANSWERS: {length_rule}\n\n"
        + STYLE_RULES
        + "\nCONTENT RULES\n"
        "- Ground the answer in the resume below (education, skills, internships, projects). "
        "Never invent employers, degrees, numbers, or technologies that are not in the resume. "
        "If the resume has nothing relevant, answer honestly and briefly from general knowledge.\n"
        "- For behavioral questions use a compact situation, action, result flow.\n"
        "- For conceptual technical questions give the correct direct explanation first, then one "
        "short example.\n"
        "- Use earlier questions and answers to resolve follow-ups such as 'why?' or 'tell me more'.\n\n"
        + CODE_RULES
    )
    if role.strip():
        system += f"\nTarget role / company: {role.strip()}\n"
    system += "\n=== RESUME ===\n" + (resume_text.strip()[:12000] or "(No resume provided.)")

    msgs = [{"role": "system", "content": system}]
    for h in history[-8:]:
        msgs.append({"role": "user", "content": h["q"]})
        if h["a"]:
            suffix = " (cut off)" if h.get("interrupted") else ""
            msgs.append({"role": "assistant", "content": h["a"] + suffix})
    msgs.append({"role": "user", "content": question})
    return msgs


def extract_resume_text(upload):
    name = upload.name.lower()
    data = upload.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
ss = st.session_state
if "engine" not in ss:
    ss.engine = VADEngine()
if "history" not in ss:
    ss.history = []
if "resume_text" not in ss:
    ss.resume_text = ""
if "resume_key" not in ss:
    ss.resume_key = None
if "was_playing" not in ss:
    ss.was_playing = False

engine = ss.engine

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
default_key = os.environ.get("GROQ_API_KEY", "")
if not default_key:
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", type="password", value=default_key)
    llm_model = st.text_input("LLM model", value="openai/gpt-oss-120b")
    stt_model = st.text_input("Whisper model", value="whisper-large-v3-turbo")
    language = st.text_input("Spoken language code (blank = auto)", value="en")
    length_choice = st.selectbox("Answer length", list(LENGTH_RULES.keys()), index=0)
    role = st.text_input("Target role / company (optional)")

    upload = st.file_uploader("Resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    if upload is not None:
        key = (upload.name, upload.size)
        if ss.resume_key != key:
            try:
                ss.resume_text = extract_resume_text(upload)
                ss.resume_key = key
            except Exception as e:
                st.error(f"Could not read resume: {e}")
    if ss.resume_text:
        st.caption(f"Resume loaded: {len(ss.resume_text):,} characters")

    st.subheader("Voice detection")
    engine.min_level = st.slider("Minimum speech level", 0.003, 0.100, 0.015, 0.001,
                                 help="Raise it if background noise triggers false questions. "
                                      "Lower it if your voice is not detected.")
    engine.silence_limit = st.slider("Silence that ends a question (seconds)", 0.5, 3.0, 1.2, 0.1)
    engine.min_speech = st.slider("Minimum speech length (seconds)", 0.2, 1.5, 0.5, 0.1)

    if st.button("Clear conversation"):
        ss.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
st.title("🎙️ Voice Interview Assistant")
st.caption("Press START, allow the microphone, then just speak. Answers appear as text.")

ctx = webrtc_streamer(
    key="interview-mic",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=4096,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": False,
        "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
    },
)

status_ph = st.empty()
diag_ph = st.empty()
st.divider()
live_ph = st.container()
history_ph = st.empty()


def render_history():
    with history_ph.container():
        if not ss.history:
            st.write("Your questions and answers will appear here.")
            return
        st.subheader("Conversation")
        for h in reversed(ss.history):
            with st.chat_message("user"):
                st.write(h["q"])
            with st.chat_message("assistant"):
                st.write(h["a"] if h["a"] else "(no answer)")
                if h.get("interrupted"):
                    st.caption("Interrupted by a new question")


playing = bool(ctx.state.playing) and ctx.audio_receiver is not None

if playing and not ss.was_playing:
    engine.reset_runtime()
ss.was_playing = playing

if not playing:
    status_ph.info("Click START above, allow microphone access, and wait a moment.")
    render_history()
    st.stop()

# ----------------------------------------------------------------------------
# Live listening loop
# ----------------------------------------------------------------------------
if not api_key:
    st.warning("Enter your Groq API key in the sidebar to enable transcription and answers. "
               "Microphone diagnostics below still work without it.")
client = Groq(api_key=api_key) if api_key else None

receiver = ctx.audio_receiver
started_at = time.time()
last_status = {"text": None}
live_q = live_ph.empty()
live_a = live_ph.empty()
render_history()


def set_status(kind, text):
    if last_status["text"] != text:
        last_status["text"] = text
        getattr(status_ph, kind)(text)


def show_diag():
    with diag_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Audio frames received", f"{engine.frames_total:,}")
        c2.metric("Sample rate", f"{engine.sample_rate or '-'} Hz")
        c3.metric("Mic level (RMS)", f"{engine.level:.3f}")
        c4.metric("Speech threshold", f"{engine.threshold():.3f}")
        st.progress(float(min(engine.level / 0.2, 1.0)))
        if engine.frames_total == 0 and time.time() - started_at > 6:
            st.error(
                "No audio frames have arrived from the browser yet. Check that the page is opened "
                "on localhost or HTTPS, the browser has microphone permission, the correct input "
                "device is selected, and no other app is locking the microphone."
            )


def tick(timeout=0.1):
    """Pull pending audio frames, run voice detection, refresh diagnostics."""
    try:
        frames = receiver.get_frames(timeout=timeout)
    except queue.Empty:
        frames = []
    if frames:
        engine.feed(frames)
    now = time.time()
    if now - engine.last_diag > 0.3:
        engine.last_diag = now
        show_diag()


def generate_answer(question):
    """Stream an answer. Returns (text, interrupted)."""
    msgs = build_messages(question, ss.resume_text, ss.history, role, LENGTH_RULES[length_choice])
    text = ""
    interrupted = False
    stream = client.chat.completions.create(
        model=llm_model,
        messages=msgs,
        temperature=0.6,
        max_tokens=1500,
        stream=True,
    )
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                text += delta
                live_a.markdown(text + " ▌")
            tick(0.01)
            if engine.new_speech_detected():
                interrupted = True
                break
    finally:
        try:
            stream.close()
        except Exception:
            pass
    live_a.markdown(text if text else "_(stopped)_")
    return text.strip(), interrupted


def handle_utterances(items):
    set_status("warning", "📝 Transcribing...")
    parts = []
    for audio, sr in items:
        try:
            t = transcribe(client, audio, sr, stt_model, language)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            continue
        if not is_noise(t):
            parts.append(t)
    question = " ".join(parts).strip()
    if not question:
        return

    live_q.markdown(f"**Question:** {question}")
    live_a.markdown("_Thinking..._")
    set_status("warning", "✍️ Generating answer... (speak to interrupt)")
    try:
        answer, interrupted = generate_answer(question)
    except Exception as e:
        st.error(f"Answer generation failed: {e}")
        answer, interrupted = "", False

    ss.history.append({"q": question, "a": answer, "interrupted": interrupted})
    render_history()
    if interrupted:
        live_q.markdown("**Interrupted** - listening to your new question...")
        live_a.empty()


while ctx.state.playing:
    tick(0.2)

    if engine.utterances and client is not None:
        items = list(engine.utterances)
        engine.utterances.clear()
        handle_utterances(items)
        continue
    elif engine.utterances:
        engine.utterances.clear()

    if engine.speaking:
        set_status("success", "🗣️ Hearing speech...")
    elif engine.frames_total == 0:
        set_status("info", "⏳ Waiting for the first audio frame from the microphone...")
    else:
        set_status("info", "🎤 Listening... ask your question")
