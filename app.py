import streamlit as st
import tempfile
import time
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDocAI – RegiBot", layout="wide")

# ---------------- UI ----------------
st.markdown("## 🤖 ChatDocAI – GenAI Document Assistant")
st.caption("RAG | LLM | LangChain | RegiBot 🤖")

st.info("📂 Upload documents → 💬 Ask questions → 🤖 Get AI answers")

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

# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LOAD DOCUMENTS ----------------
@st.cache_resource
def load_docs(files):
    texts = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        docs = loader.load()

        for d in docs:
            texts.append(d.page_content)

    return texts

# ---------------- RAG RETRIEVAL ----------------
def get_relevant_docs(query, texts):
    model = get_embeddings()

    query_vec = model.embed_query(query)
    doc_vecs = model.embed_documents(texts)

    scores = cosine_similarity([query_vec], doc_vecs)[0]
    top_idx = scores.argsort()[-5:][::-1]

    return [texts[i] for i in top_idx]

# ---------------- LLM ----------------
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5}
    )

# ---------------- PROMPT ----------------
qa_prompt = PromptTemplate.from_template(
    """You are an AI assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer in {language}.
"""
)

quiz_prompt = PromptTemplate.from_template(
    """Generate 5 multiple-choice questions from the text.

Context:
{context}

Format:
Q1:
A.
B.
C.
D.
Answer:
"""
)

parser = StrOutputParser()

# ---------------- MAIN ----------------
if uploaded_files:
    texts = load_docs(uploaded_files)
    llm = get_llm()

    query = st.chat_input("💬 Ask something from your document...")

    # ✅ CHAT
    if query:
        st.session_state.chat.append(("user", query))

        with st.spinner("🤖 RegiBot is thinking..."):
            docs = get_relevant_docs(query, texts)
            context = "\n\n".join(docs)

            chain = qa_prompt | llm | parser
            answer = chain.invoke({
                "context": context,
                "question": query,
                "language": language
            })

        st.session_state.chat.append(("ai", answer, docs))

    # ✅ QUIZ
    if generate_quiz:
        with st.spinner("🎯 Generating quiz..."):
            docs = get_relevant_docs("quiz", texts)
            context = "\n\n".join(docs)

            chain = quiz_prompt | llm | parser
            quiz = chain.invoke({"context": context})

        st.session_state.chat.append(("ai", quiz, docs))

    # ---------------- DISPLAY ----------------
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            with st.chat_message("assistant"):
                st.write(msg[1])

                # ✅ SHOW SOURCES
                if len(msg) > 2:
                    with st.expander("📄 Source Context"):
                        for i, d in enumerate(msg[2]):
                            st.write(f"Source {i+1}:")
                            st.write(d[:300] + "...")

else:
    st.info("📂 Upload documents to begin")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 ChatDocAI – Built with RAG + LLM + LangChain")
st.caption("✨ Made by Preethi Regina S D")
