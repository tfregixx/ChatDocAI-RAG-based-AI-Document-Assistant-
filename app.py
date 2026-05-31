import streamlit as st
import tempfile
import requests
import time

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import Ollama, HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDOC AI", layout="wide")

# ---------------- UI + ANIMATIONS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #020617;
    color: white;
}

/* HERO */
.hero {
    text-align: center;
    padding: 40px;
}

.hero h1 {
    font-size: 45px;
    background: linear-gradient(90deg, #3b82f6, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ROBOT */
.robot {
    display:block;
    margin:auto;
    width:160px;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% {transform: translateY(0);}
    50% {transform: translateY(-12px);}
    100% {transform: translateY(0);}
}

/* LOADER */
.loader {
    text-align:center;
    font-size:18px;
    animation: blink 1s infinite;
}
@keyframes blink {
    0% {opacity:0.3;}
    50% {opacity:1;}
    100% {opacity:0.3;}
}

/* WATERMARK */
.watermark {
    position: fixed;
    bottom: 15px;
    right: 15px;
    color: #38bdf8;
    background: rgba(0,0,0,0.5);
    padding: 6px 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    https://cdn-icons-png.flaticon.com/512/4712/4712109.png
    <h1>ChatDOC AI 🤖 RegiBot</h1>
    <p>Your Smart AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ✅ LANGUAGE (Tamil added)
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

# ---------------- PROCESS FILES ----------------
def process_files(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    return Chroma.from_documents(docs, get_embeddings())

# ---------------- LOAD DB ----------------
db = None
if uploaded_files:
    db = process_files(uploaded_files)

# ---------------- MODEL ----------------
def ollama_running():
    try:
        requests.get("http://localhost:11434")
        return True
    except:
        return False

def get_llm():
    if ollama_running():
        st.sidebar.success("✅ Offline Mode")
        return Ollama(model="llama3")
    else:
        return HuggingFaceHub(repo_id="google/flan-t5-base")

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

    query = st.chat_input("Ask your question...")

    # ✅ QUESTION
    if query:
        st.session_state.chat.append(("user", query))

        loader = st.empty()
        loader.markdown('<div class="loader">🤖 RegiBot Thinking...</div>', unsafe_allow_html=True)

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

    # ✅ QUIZ
    if generate_quiz:
        loader = st.empty()
        loader.markdown('<div class="loader">🎯 RegiBot Creating Quiz...</div>', unsafe_allow_html=True)

        docs = retriever.invoke("quiz")
        context = "\n\n".join([d.page_content for d in docs])

        quiz_prompt = PromptTemplate.from_template(
            """Create 5 MCQ questions from this content.

Context:
{context}
"""
        )

        chain = quiz_prompt | llm | parser
        quiz = chain.invoke({"context": context})

        loader.empty()
        st.session_state.chat.append(("ai", quiz, docs))

    # ✅ DISPLAY CHAT (FIXED ✅)
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                text = msg[1]

                shown = ""
                for ch in text:
                    shown += ch
                    placeholder.markdown(shown)
                    time.sleep(0.01)

                # ✅ FIXED SOURCES
                with st.expander("📄 Sources"):
                    for i, doc in enumerate(msg[2]):
                        st.write(f"Source {i+1}:")
                        st.write(doc.page_content[:300] + "...")

else:
    st.info("Upload documents to start")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 ChatDOC AI • RegiBot Assistant")

# ---------------- WATERMARK ----------------
st.markdown(
    '<div class="watermark">✨ Made by Preethi Regina S D</div>',
    unsafe_allow_html=True
)
