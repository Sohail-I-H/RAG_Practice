import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq


st.set_page_config(
    page_title="PragyanAI Intelligent Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PragyanAI Conversational Sales & FAQ Assistant")
st.markdown(
    "Answers program questions based on the **PragyanAI Presentation & FAQ Sheet**."
)

groq_api_key = st.secrets["GROQ_API_KEY"]

FAQ_FILE = "data/pragyan_faq_prices.xlsx"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = None

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


SALES_PROMPTS = {

    "PragyanAI Student Counselor": """You are Aarav, an Academic & Career Advisor for PragyanAI.

Goal: Guide prospective students to enroll in the 18-Month AI/GenAI Program (6 Month Offline Training + 12 Month Placement Drive).

Strict Rule:
Answer pricing, fee structures, curriculum details, and salary potential ONLY based on the Document Context below.

Retrieved Document Context:
{context}

Behavior Guidelines:

1. Be encouraging, empathetic, and focus on practical builder skill transformation.

2. Highlight:

• 100+ projects

• 48-hour Hackathons

• Risk-shared pricing

• Direct mentorship under Sateesh Ambesange.
""",

    "PragyanAI Institutional / CoE Advisor": """You are Dr. Kavita, Institutional Relations Lead at PragyanAI.

Goal:
Partner with Engineering Colleges.

Strict Rule:

Use only the retrieved document context.

Retrieved Document Context:

{context}

Behavior Guidelines:

1. Industry-oriented tone.

2. Focus on Agentic AI, GenAI and Product Builder mindset.
""",

    "PragyanAI Enterprise AI & Placement Lead": """You are Rohan, Enterprise Placement Lead.

Goal:

Help hiring partners recruit PragyanAI Engineers.

Strict Rule:

Use only retrieved context.

Retrieved Document Context:

{context}

Behavior Guidelines:

1. ROI driven.

2. Mention CrewAI, AutoGen, LangChain, RAG, Multi-Agent Systems whenever available.
"""
}


def load_documents_into_vectorstore(uploaded_files=None):

    global vectorstore

    docs = []

    if uploaded_files:

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

                tmp.write(uploaded_file.getbuffer())

                temp_path = tmp.name

            if suffix.lower() == ".pdf":

                loader = PyPDFLoader(temp_path)

                docs.extend(loader.load())

            elif suffix.lower() in [".xlsx", ".xls"]:

                excel_df = pd.read_excel(temp_path)

                for _, row in excel_df.iterrows():

                    docs.append(

                        Document(

                            page_content=" | ".join(
                                [
                                    f"{col}: {val}"
                                    for col, val in row.items()
                                ]
                            ),

                            metadata={
                                "source": uploaded_file.name
                            }

                        )

                    )

    if os.path.exists(FAQ_FILE):

        excel_df = pd.read_excel(FAQ_FILE)

        for _, row in excel_df.iterrows():

            docs.append(

                Document(

                    page_content=" | ".join(
                        [
                            f"{col}: {val}"
                            for col, val in row.items()
                        ]
                    ),

                    metadata={
                        "source": FAQ_FILE
                    }

                )

            )

    if not docs:

        docs = [

            Document(
                page_content="PragyanAI Program: 6 Months Offline Training + 12 Months Placement Drive."
            ),

            Document(
                page_content="Founding Batch Fee: ₹50,000 Initial Training + ₹50,000 Success Fee."
            )

        ]

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )


load_documents_into_vectorstore()

# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)


# ---------------------------------------------------------------------------
# Create RAG Chain
# ---------------------------------------------------------------------------

def create_rag_chain(
    persona_name: str,
    retrieved_context: str,
    language: str
):

    system_instruction = SALES_PROMPTS.get(
        persona_name,
        SALES_PROMPTS["PragyanAI Student Counselor"]
    ).format(
        context=retrieved_context
    )

    system_instruction += f"""

====================================================

LANGUAGE INSTRUCTION (VERY IMPORTANT)

Always answer ONLY in {language}.

Do NOT mix multiple languages.

If the user asks in another language,
still answer ONLY in {language}.

If the response contains bullet points,
keep everything in {language}.

If the response contains numbers,
only the numbers may remain unchanged.

====================================================

"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ]
    )

    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Respond Function
# ---------------------------------------------------------------------------

def respond(
    message,
    persona_name,
    language
):

    if not message.strip():
        return ""

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(message)

    context_str = "\n".join(
        [
            f"- {doc.page_content}"
            for doc in relevant_docs
        ]
    )

    session_id = (
        f"pragyan_session_"
        f"{persona_name.replace(' ','_')}_"
        f"{language}"
    )

    base_chain = create_rag_chain(
        persona_name,
        context_str,
        language
    )

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = conversational_chain.invoke(
        {
            "input": message
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    return response


# ---------------------------------------------------------------------------
# Clear Chat History
# ---------------------------------------------------------------------------

def clear_chat_history(persona_name):

    session_id = f"pragyan_session_{persona_name.replace(' ', '_')}"

    if session_id in store:
        store[session_id].clear()

  # ---------------------------------------------------------------------------
# Streamlit Sidebar
# ---------------------------------------------------------------------------

LANGUAGES = {
    "English 🇺🇸": ("English", "en-US"),
    "ಕನ್ನಡ 🇮🇳": ("Kannada", "kn-IN"),
    "हिन्दी 🇮🇳": ("Hindi", "hi-IN"),
    "தமிழ் 🇮🇳": ("Tamil", "ta-IN"),
    "తెలుగు 🇮🇳": ("Telugu", "te-IN"),
    "മലയാളം 🇮🇳": ("Malayalam", "ml-IN"),
    "ગુજરાતી 🇮🇳": ("Gujarati", "gu-IN"),
    "বাংলা 🇮🇳": ("Bengali", "bn-IN"),
    "ਪੰਜਾਬੀ 🇮🇳": ("Punjabi", "pa-IN"),
    "मराठी 🇮🇳": ("Marathi", "mr-IN"),
    "اردو": ("Urdu", "ur"),
    "Español": ("Spanish", "es-ES"),
    "Français": ("French", "fr-FR"),
    "Deutsch": ("German", "de-DE"),
    "Italiano": ("Italian", "it-IT"),
    "Português": ("Portuguese", "pt-PT"),
    "Русский": ("Russian", "ru-RU"),
    "日本語": ("Japanese", "ja-JP"),
    "한국어": ("Korean", "ko-KR"),
    "中文": ("Chinese", "zh-CN")
}

with st.sidebar:

    st.title("⚙️ Settings")

    language_display = st.selectbox(
        "🌍 Select Language",
        list(LANGUAGES.keys()),
        index=0
    )

    selected_language = LANGUAGES[language_display][0]
    selected_voice = LANGUAGES[language_display][1]

    st.divider()

    persona = st.selectbox(
        "👤 Select Persona",
        list(SALES_PROMPTS.keys())
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "📄 Upload PDFs / Excel",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:

        with st.spinner("Updating Knowledge Base..."):

            load_documents_into_vectorstore(uploaded_files)

        st.success("✅ Knowledge Base Updated Successfully")

    else:

        st.info("📚 Default PragyanAI Knowledge Base Loaded")

    st.divider()

    st.subheader("🎤 Voice Assistant")

    st.write(f"**Selected Language:** {selected_language}")

    st.caption("Voice input and output will use this language.")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        clear_chat_history(persona)

        st.session_state.messages = []

        st.success("Conversation Cleared")

# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Display Previous Messages
# ---------------------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------

user_prompt = st.chat_input(
    "Ask anything about PragyanAI..."
)


if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = respond(
                user_prompt,
                persona,
                selected_language
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

