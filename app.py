import streamlit as st
import tempfile
import os
import time
import requests

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama, HuggingFaceHub


# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDOC AI", layout="wide")

# ---------------- SIMPLE STYLING ----------------
st.markdown("""
<style>
body { background-color: #020617; color: white; }
.loader { text-align:center; animation: blink 1s infinite; }
@keyframes blink {0%{opacity:0.3;}50%{opacity:1;}100%{opacity:0.3;}}
.watermark {
    position: fixed;
    bottom: 15px;
    right: 15px;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 ChatDOC AI – RegiBot")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Documents")

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

# ---------------- SAFE EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error("Embedding model failed to load")
        st.stop()

# ---------------- PROCESS FILES SAFE ----------------
@st.cache_resource
def process_files(files):
    try:
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
            st.warning("No readable content found in document.")
            return None

        return Chroma.from_documents(
            docs,
            embedding=get_embeddings(),
            persist_directory="vector_db"
        )

    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
        return None

# ---------------- LOAD DB ----------------
db = None
if uploaded_files:
    db = process_files(uploaded_files)

# ---------------- MODEL HANDLING ----------------
def get_llm():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return Ollama(model="llama3")
    except:
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
if db:
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    query = st.chat_input("Ask your question...")

    if query:
        st.session_state.chat.append(("user", query))

        loader = st.empty()
        loader.markdown('<div class="loader">🤖 Thinking...</div>', unsafe_allow_html=True)

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

        except Exception as e:
            loader.empty()
            st.error("AI processing failed.")
            st.session_state.chat.append(("ai", "Error generating response", []))

    if generate_quiz:
        try:
            docs = retriever.invoke("quiz")
            context = "\n\n".join([d.page_content for d in docs])

            quiz_prompt = PromptTemplate.from_template(
                "Create 5 MCQ questions from the following:\n{context}"
            )

            chain = quiz_prompt | llm | parser
            quiz = chain.invoke({"context": context})

            st.session_state.chat.append(("ai", quiz, docs))

        except:
            st.error("Quiz generation failed")

    # ✅ DISPLAY CHAT
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            st.chat_message("assistant").write(msg[1])

else:
    st.info("Upload documents to begin")

# ---------------- WATERMARK ----------------
st.markdown(
    '<div class="watermark">✨ Made by Preethi Regina S D</div>',
    unsafe_allow_html=True
)
