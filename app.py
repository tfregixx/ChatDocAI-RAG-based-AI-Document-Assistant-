import streamlit as st
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDocAI – RegiBot", layout="wide")

st.write("✅ App Loaded Successfully")  # Debug indicator

st.title("🤖 ChatDocAI – GenAI Document Assistant")
st.caption("RAG | LLM | LangChain")

st.info("📂 Upload documents → 💬 Ask → 🤖 AI answers")

# ---------------- SIDEBAR ----------------
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

# ---------------- SESSION ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error("⚠️ Embedding model failed")
        st.stop()

# ---------------- LOAD DOCS ----------------
@st.cache_resource
def load_docs(files):
    texts = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                path = tmp.name

            loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
            docs = loader.load()

            for d in docs:
                texts.append(d.page_content)

        return texts

    except Exception as e:
        st.error(f"⚠️ Error loading files: {e}")
        return []

# ---------------- RAG SEARCH ----------------
def get_relevant_docs(query, texts):
    try:
        model = get_embeddings()

        query_vec = model.embed_query(query)
        doc_vecs = model.embed_documents(texts)

        scores = cosine_similarity([query_vec], doc_vecs)[0]
        top_indices = scores.argsort()[-5:][::-1]

        return [texts[i] for i in top_indices]

    except Exception as e:
        st.error("⚠️ Retrieval failed")
        return []

# ---------------- MODEL (SAFE ✅) ----------------
@st.cache_resource
def get_llm():
    try:
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5}
        )
    except Exception as e:
        st.error("⚠️ HuggingFace model failed. Check API token in Secrets.")
        st.stop()

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
    """Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer in {language}.
"""
)

parser = StrOutputParser()

# ---------------- MAIN ----------------
if uploaded_files:
    texts = load_docs(uploaded_files)

    if not texts:
        st.warning("⚠️ No text extracted from documents.")
        st.stop()

    llm = get_llm()
    query = st.chat_input("💬 Ask your question...")

    # ✅ CHAT
    if query:
        st.session_state.chat.append(("user", query))

        with st.spinner("🤖 Thinking..."):
            docs = get_relevant_docs(query, texts)
            context = "\n\n".join(docs)

            try:
                chain = prompt | llm | parser
                answer = chain.invoke({
                    "context": context,
                    "question": query,
                    "language": language
                })
            except Exception:
                answer = "⚠️ Error generating response (check API key or model)"

        st.session_state.chat.append(("ai", answer, docs))

    # ✅ QUIZ
    if generate_quiz:
        with st.spinner("🎯 Generating Quiz..."):
            docs = get_relevant_docs("quiz", texts)
            context = "\n\n".join(docs)

            quiz_prompt = PromptTemplate.from_template(
                "Create 5 MCQ questions from the context:\n{context}"
            )

            try:
                chain = quiz_prompt | llm | parser
                quiz = chain.invoke({"context": context})
            except:
                quiz = "⚠️ Quiz generation failed"

        st.session_state.chat.append(("ai", quiz, docs))

    # ✅ DISPLAY CHAT
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            with st.chat_message("assistant"):
                st.write(msg[1])

                if len(msg) > 2:
                    with st.expander("📄 Source Context"):
                        for d in msg[2]:
                            st.write(d[:300] + "...")

else:
    st.info("📂 Upload documents to begin")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 ChatDocAI – RAG + LLM System")
st.caption("✨ Made by Preethi Regina S D")
