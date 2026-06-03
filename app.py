import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
import time

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ✅ USE FAISS INSTEAD OF CHROMA (MORE STABLE ✅)
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDOC AI", layout="wide")

# ---------------- UI ----------------
st.title("🤖 ChatDOC AI – RegiBot")
st.info("Upload a document and chat with it")

# ---------------- SIDEBAR ----------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

language = st.sidebar.selectbox(
    "Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

generate_quiz = st.sidebar.button("🎯 Generate Quiz")

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LOAD DB (SAFE ✅) ----------------
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

    # ✅ FAISS instead of Chroma
    return FAISS.from_documents(docs, get_embeddings())

# ---------------- DB ----------------
db = None
if uploaded_files:
    db = load_db(uploaded_files)

# ---------------- MODEL ----------------
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5}
    )

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
if "chat" not in st.session_state:
    st.session_state.chat = []

if db:
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    query = st.chat_input("Ask something...")

    if query:
        st.session_state.chat.append(("user", query))
        with st.spinner("🤖 Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])

            chain = prompt | llm | parser
            answer = chain.invoke({
                "context": context,
                "question": query,
                "language": language
            })

        st.session_state.chat.append(("ai", answer, docs))

    if generate_quiz:
        docs = retriever.invoke("quiz")
        context = "\n\n".join([d.page_content for d in docs])

        quiz_prompt = PromptTemplate.from_template(
            "Create 5 MCQ questions from:\n{context}"
        )

        chain = quiz_prompt | llm | parser
        quiz = chain.invoke({"context": context})

        st.session_state.chat.append(("ai", quiz, docs))

    # DISPLAY
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            st.chat_message("assistant").write(msg[1])

else:
    st.info("Upload documents first")

# ---------------- WATERMARK ----------------
st.caption("✨ Made by Preethi Regina S D")
