import streamlit as st
import tempfile
import time
import requests

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDOC AI", layout="wide")

# ---------------- UI STYLES ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #020617;
    color: white;
}

.hero {
    text-align: center;
    padding: 30px;
}

.robot {
    display:block;
    margin:auto;
    width:120px;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-12px);}
    100% {transform: translateY(0px);}
}

.loader {
    text-align:center;
    animation: blink 1s infinite;
}

@keyframes blink {
    0% {opacity:0.3;}
    50% {opacity:1;}
    100% {opacity:0.3;}
}

.watermark {
    position: fixed;
    bottom: 15px;
    right: 15px;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="robot">
    <h1>ChatDOC AI 🤖 RegiBot</h1>
    <p>Your Smart AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# ✅ Welcome Message
st.info("👋 Welcome to ChatDOC AI! Upload a document and start chatting.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

generate_quiz = st.sidebar.button("🎯 Generate Quiz")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat = []

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- CREATE VECTOR DB (CACHED ✅) ----------------
@st.cache_resource
def load_db(files):
    docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    if not docs:
        return None

    return Chroma.from_documents(
        docs,
        embedding=get_embeddings()
    )

# ---------------- LOAD DB ----------------
db = None
if uploaded_files:
    db = load_db(uploaded_files)

# ---------------- MODEL (DEPLOYMENT SAFE ✅) ----------------
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5}
    )

# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
    """Answer in {language}.

Context:
{context}

Question:
{question}
"""
)

parser = StrOutputParser()

# ---------------- CHAT ----------------
st.markdown("## 💬 Chat with RegiBot")

if db:
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    query = st.chat_input("Ask anything from your document...")

    # ✅ NORMAL CHAT
    if query:
        st.session_state.chat.append(("user", query))

        loader = st.empty()
        loader.markdown('<div class="loader">🤖 RegiBot Thinking...</div>', unsafe_allow_html=True)

        try:
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])

            chain = prompt | llm | parser
            answer = chain.invoke({
                "context": context,
                "question": query,
                "language": language
            })

            loader.empty()
            st.session_state.chat.append(("ai", answer, docs))

        except Exception:
            loader.empty()
            st.error("⚠️ AI failed to respond")

    # ✅ QUIZ
    if generate_quiz and db:
        loader = st.empty()
        loader.markdown('<div class="loader">🎯 Generating Quiz...</div>', unsafe_allow_html=True)

        try:
            docs = retriever.invoke("quiz")
            context = "\n\n".join([d.page_content for d in docs])

            quiz_prompt = PromptTemplate.from_template(
                "Create 5 MCQ questions from this:\n{context}"
            )

            chain = quiz_prompt | llm | parser
            quiz = chain.invoke({"context": context})

            loader.empty()
            st.session_state.chat.append(("ai", quiz, docs))

        except:
            loader.empty()
            st.error("Quiz failed")

    # ✅ DISPLAY CHAT
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            st.chat_message("assistant").write(msg[1])

else:
    st.info("Upload documents to begin")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 ChatDOC AI • RegiBot Assistant")

# ✅ WATERMARK
st.markdown(
    '<div class="watermark">✨ Made by Preethi Regina S D</div>',
    unsafe_allow_html=True
)
