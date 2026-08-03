import os
import uuid
import asyncio
import tempfile
import pandas as pd
import streamlit as st

from groq import Groq
import edge_tts

from streamlit_mic_recorder import mic_recorder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="PragyanAI Voice Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PragyanAI Multilingual Voice Assistant")

st.caption(
    "Powered by LangChain • Groq • Whisper • Edge TTS • FAISS"
)

# -----------------------------------------------------
# API KEYS
# -----------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

# -----------------------------------------------------
# FILE PATHS
# -----------------------------------------------------

FAQ_FILE = "data/pragyan_faq_prices.xlsx"

# -----------------------------------------------------
# EMBEDDING MODEL
# -----------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = None

# -----------------------------------------------------
# CHAT MEMORY
# -----------------------------------------------------

store = {}

def get_session_history(session_id: str):

    if session_id not in store:

        store[session_id] = ChatMessageHistory()

    return store[session_id]

# -----------------------------------------------------
# LLM
# -----------------------------------------------------

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="llama-3.3-70b-versatile",

    temperature=0.3

)

# -----------------------------------------------------
# SUPPORTED LANGUAGES
# -----------------------------------------------------

LANGUAGES = {
    "English 🇺🇸": {
        "language": "English",
        "voice": "en-US-AriaNeural"
    },
    "ಕನ್ನಡ 🇮🇳": {
        "language": "Kannada",
        "voice": "kn-IN-SapnaNeural"
    },
    "हिन्दी 🇮🇳": {
        "language": "Hindi",
        "voice": "hi-IN-SwaraNeural"
    },
    "தமிழ் 🇮🇳": {
        "language": "Tamil",
        "voice": "ta-IN-PallaviNeural"
    },
    "తెలుగు 🇮🇳": {
        "language": "Telugu",
        "voice": "te-IN-ShrutiNeural"
    },
    "മലയാളം 🇮🇳": {
        "language": "Malayalam",
        "voice": "ml-IN-SobhanaNeural"
    },
    "ગુજરાતી 🇮🇳": {
        "language": "Gujarati",
        "voice": "gu-IN-DhwaniNeural"
    },
    "বাংলা 🇮🇳": {
        "language": "Bengali",
        "voice": "bn-IN-TanishaaNeural"
    },
    "मराठी 🇮🇳": {
        "language": "Marathi",
        "voice": "mr-IN-AarohiNeural"
    },
    "Español 🇪🇸": {
        "language": "Spanish",
        "voice": "es-ES-ElviraNeural"
    },
    "Français 🇫🇷": {
        "language": "French",
        "voice": "fr-FR-DeniseNeural"
    },
    "Deutsch 🇩🇪": {
        "language": "German",
        "voice": "de-DE-KatjaNeural"
    },
    "Italiano 🇮🇹": {
        "language": "Italian",
        "voice": "it-IT-ElsaNeural"
    },
    "Português 🇵🇹": {
        "language": "Portuguese",
        "voice": "pt-PT-RaquelNeural"
    },
    "日本語 🇯🇵": {
        "language": "Japanese",
        "voice": "ja-JP-NanamiNeural"
    },
    "한국어 🇰🇷": {
        "language": "Korean",
        "voice": "ko-KR-SunHiNeural"
    },
    "中文 🇨🇳": {
        "language": "Chinese",
        "voice": "zh-CN-XiaoxiaoNeural"
    }
}

# -----------------------------------------------------
# PERSONAS
# -----------------------------------------------------

SALES_PROMPTS = {

    "PragyanAI Student Counselor":
"""
You are Aarav, Academic & Career Advisor at PragyanAI.

Answer ONLY using the retrieved document context.

Retrieved Context:

{context}

Guide students professionally.

Be encouraging.

Always answer in the selected language.
""",

    "PragyanAI Institutional Advisor":
"""
You are Dr. Kavita.

Help Engineering Colleges understand PragyanAI.

Retrieved Context:

{context}

Always answer in the selected language.
""",

    "PragyanAI Enterprise Lead":
"""
You are Rohan.

Help recruiters and enterprises.

Retrieved Context:

{context}

Always answer in the selected language.
"""

}

# -----------------------------------------------------
# KNOWLEDGE BASE
# -----------------------------------------------------

def load_documents(uploaded_files=None):

    global vectorstore

    docs = []

    if uploaded_files:

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(uploaded_file.getbuffer())

                path = tmp.name

            if suffix.lower() == ".pdf":

                loader = PyPDFLoader(path)

                docs.extend(
                    loader.load()
                )

            elif suffix.lower() in [
                ".xlsx",
                ".xls"
            ]:

                df = pd.read_excel(path)

                for _, row in df.iterrows():

                    docs.append(

                        Document(

                            page_content=" | ".join(
                                [
                                    f"{c}: {v}"
                                    for c, v in row.items()
                                ]
                            ),

                            metadata={
                                "source": uploaded_file.name
                            }

                        )

                    )

    if os.path.exists(FAQ_FILE):

        df = pd.read_excel(FAQ_FILE)

        for _, row in df.iterrows():

            docs.append(

                Document(

                    page_content=" | ".join(
                        [
                            f"{c}: {v}"
                            for c, v in row.items()
                        ]
                    ),

                    metadata={
                        "source": FAQ_FILE
                    }

                )

            )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

load_documents()

# -----------------------------------------------------
# SPEECH TO TEXT (Groq Whisper)
# -----------------------------------------------------

def speech_to_text(audio_path):

    with open(audio_path, "rb") as audio_file:

        transcription = client.audio.transcriptions.create(

            file=(
                os.path.basename(audio_path),
                audio_file.read()
            ),

            model="whisper-large-v3-turbo",

            response_format="json"
        )

    return transcription.text


# -----------------------------------------------------
# TEXT TO SPEECH (Edge TTS)
# -----------------------------------------------------

async def generate_audio(text, voice):

    filename = f"audio_{uuid.uuid4().hex}.mp3"

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(filename)

    return filename


def text_to_speech(text, voice):

    return asyncio.run(
        generate_audio(
            text,
            voice
        )
    )


# -----------------------------------------------------
# CREATE RAG CHAIN
# -----------------------------------------------------

def create_rag_chain(
    persona_name,
    context,
    language
):

    system_prompt = SALES_PROMPTS.get(
        persona_name
    ).format(
        context=context
    )

    system_prompt += f"""

IMPORTANT INSTRUCTIONS

Always answer ONLY in {language}.

Never mix multiple languages.

If user asks in another language,
still answer ONLY in {language}.

Keep responses conversational.

"""

    prompt = ChatPromptTemplate.from_messages(

        [

            (
                "system",
                system_prompt
            ),

            MessagesPlaceholder(
                variable_name="history"
            ),

            (
                "human",
                "{input}"
            )

        ]

    )

    return (

        prompt

        | llm

        | StrOutputParser()

    )


# -----------------------------------------------------
# CHATBOT RESPONSE
# -----------------------------------------------------

def respond(
    question,
    persona,
    language
):

    retriever = vectorstore.as_retriever(

        search_kwargs={
            "k": 4
        }

    )

    docs = retriever.invoke(question)

    context = "\n".join(

        [

            d.page_content

            for d in docs

        ]

    )

    session_id = (

        f"{persona}_{language}"

    )

    chain = RunnableWithMessageHistory(

        create_rag_chain(

            persona,

            context,

            language

        ),

        get_session_history,

        input_messages_key="input",

        history_messages_key="history"

    )

    answer = chain.invoke(

        {

            "input": question

        },

        config={

            "configurable": {

                "session_id": session_id

            }

        }

    )

    return answer

# -----------------------------------------------------
# CLEAR CHAT HISTORY
# -----------------------------------------------------

def clear_chat_history(persona):

    keys_to_remove = [

        key

        for key in store

        if key.startswith(persona)

    ]

    for key in keys_to_remove:

        del store[key]

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

with st.sidebar:

    st.title("🤖 PragyanAI")

    st.markdown("---")

    language_display = st.selectbox(
        "🌍 Select Language",
        list(LANGUAGES.keys())
    )

    selected_language = LANGUAGES[language_display]["language"]

    selected_voice = LANGUAGES[language_display]["voice"]

    st.markdown("---")

    persona = st.selectbox(
        "👤 Select Persona",
        list(SALES_PROMPTS.keys())
    )

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📄 Upload PDF / Excel",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:

        with st.spinner("Updating Knowledge Base..."):

            load_documents(uploaded_files)

        st.success("Knowledge Base Updated")

    st.markdown("---")

    st.success(f"🌍 Language : {selected_language}")

    st.markdown("---")

    if st.button("🗑 Clear Conversation"):

        clear_chat_history(persona)

        st.session_state.messages = []

        st.rerun()


# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# -----------------------------------------------------
# SHOW CHAT HISTORY
# -----------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# -----------------------------------------------------
# VOICE INPUT
# -----------------------------------------------------

st.markdown("## 🎤 Voice Input")

voice = mic_recorder(

    start_prompt="🎤 Start Recording",

    stop_prompt="⏹ Stop Recording",

    just_once=True,

    use_container_width=True,

    key="voice"

)


voice_question = None

if voice:

    audio_bytes = voice["bytes"]

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".wav"

    ) as f:

        f.write(audio_bytes)

        audio_path = f.name

    with st.spinner("🎧 Listening..."):

        voice_question = speech_to_text(audio_path)

    st.success("Speech Recognized")

    st.write(voice_question)


# -----------------------------------------------------
# TEXT INPUT
# -----------------------------------------------------

typed_question = st.chat_input(

    "Type your question..."

)


question = voice_question if voice_question else typed_question


# -----------------------------------------------------
# CHATBOT
# -----------------------------------------------------

if question:

    st.session_state.messages.append(

        {

            "role": "user",

            "content": question

        }

    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            answer = respond(

                question,

                persona,

                selected_language

            )

            st.markdown(answer)

            audio_file = text_to_speech(

                answer,

                selected_voice

            )

            st.audio(audio_file)

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer

        }

    )
