import streamlit as st
import requests
import speech_recognition as sr
import os
from gtts import gTTS
import io

# --- Page Setup ---
st.set_page_config(
    page_title="SmartHealthChat",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 SmartHealthChat")
st.caption("Your friendly AI health information assistant")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm SmartHealthChat. How are you feeling today?"}
    ]
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# --- SIDEBAR: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Language Selector
    st.info("Select language for Voice Input & Output.")
    voice_lang = st.selectbox(
        "Language:",
        options=["en-IN", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN"],
        format_func=lambda x: {
            "en-IN": "English / Hinglish",
            "hi-IN": "Hindi (हिंदी)",
            "ta-IN": "Tamil (தமிழ்)",
            "te-IN": "Telugu (తెలుగు)",
            "mr-IN": "Marathi (मराठी)",
            "bn-IN": "Bengali (বাংলা)",
            "gu-IN": "Gujarati (ગુજરાતી)",
            "kn-IN": "Kannada (ಕನ್ನಡ)",
            "ml-IN": "Malayalam (മലയാളം)"
        }.get(x, x)
    )

    # 2. AUDIO RESPONSE TOGGLE (New Feature)
    # By default, this is unchecked (False). So audio won't play unless user turns it on.
    enable_audio_response = st.checkbox("🔊 Enable Audio Response")


# --- HELPER: Map Selection to Google TTS Codes ---
TTS_LANG_MAPPING = {
    "en-IN": "en",
    "hi-IN": "hi",
    "ta-IN": "ta",
    "te-IN": "te",
    "mr-IN": "mr",
    "bn-IN": "bn",
    "gu-IN": "gu",
    "kn-IN": "kn",
    "ml-IN": "ml"
}

# --- FUNCTION: Handle Chat (Text or Voice) ---
def handle_chat(user_input):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Get Bot Response from Server
    api_url = "http://localhost:3000/ask"
    answer = "Sorry, I couldn't connect."
    
    try:
        response = requests.get(api_url, params={"q": user_input, "lang": voice_lang})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
        else:
            answer = f"Sorry, error: {response.status_code}"
    except requests.exceptions.RequestException:
        answer = "I'm sorry, I can't reach my brain right now."
    
    # 3. Generate Audio ONLY IF Toggle is ON
    audio_bytes = None
    if enable_audio_response:
        try:
            lang_code = TTS_LANG_MAPPING.get(voice_lang, "en")
            tts = gTTS(text=answer, lang=lang_code, slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_bytes = audio_fp.getvalue()
        except Exception as e:
            print(f"TTS Error: {e}")

    # 4. Save Assistant Message (with or without audio)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "audio": audio_bytes 
    })

# --- DISPLAY CHAT HISTORY ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Play audio only if it exists AND it's the latest message (to prevent overlap)
        if "audio" in message and message["audio"] is not None:
            is_latest = (i == len(st.session_state.messages) - 1)
            # Only autoplay if it's the fresh new message
            st.audio(message["audio"], format="audio/mp3", autoplay=is_latest)

# --- AUDIO INPUT (Bottom) ---
st.write("---") 
audio_value = st.audio_input("🎤 Tap to Speak")

# --- PROCESS VOICE INPUT ---
if audio_value and audio_value != st.session_state.last_audio:
    st.session_state.last_audio = audio_value
    r = sr.Recognizer()
    try:
        with st.spinner("Listening..."):
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                transcribed_text = r.recognize_google(audio_data, language=voice_lang)
                handle_chat(transcribed_text)
                st.rerun()    
    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio.")
    except sr.RequestError:
        st.error("Speech Service Error.")

# --- TEXT INPUT (Bottom) ---
if prompt := st.chat_input("Type your health concern here..."):
    handle_chat(prompt)
    st.rerun()