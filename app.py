import streamlit as st
import tempfile
import time
from sklearn.metrics.pairwise import cosine_similarity

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_model():
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

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text("\n\n".join(texts))

# ---------------- SIMILARITY SEARCH ----------------
def get_relevant_docs(query, texts):
    model = get_model()

    query_vec = model.embed_query(query)
    doc_vecs = model.embed_documents(texts)

    scores = cosine_similarity([query_vec], doc_vecs)[0]
    top_idx = scores.argsort()[-5:][::-1]

    return [texts[i] for i in top_idx]

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

# ---------------- MAIN LOGIC ----------------
if uploaded_files:
    texts = load_docs(uploaded_files)
    llm = get_llm()

    query = st.chat_input("Ask your question...")

    if query:
        st.session_state.chat.append(("user", query))

        with st.spinner("🤖 Thinking..."):
            docs = get_relevant_docs(query, texts)
            context = "\n\n".join(docs)

            chain = prompt | llm | parser
            answer = chain.invoke({
                "context": context,
                "question": query,
                "language": language
            })

        st.session_state.chat.append(("ai", answer))

    if generate_quiz:
        with st.spinner("🎯 Generating Quiz..."):
            docs = get_relevant_docs("quiz", texts)
            context = "\n\n".join(docs)

            quiz_prompt = PromptTemplate.from_template(
                "Create 5 MCQ questions from:\n{context}"
            )

            chain = quiz_prompt | llm | parser
            quiz = chain.invoke({"context": context})

        st.session_state.chat.append(("ai", quiz))

    # ---------------- CHAT DISPLAY ----------------
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            st.chat_message("assistant").write(msg[1])

else:
    st.info("Upload documents to begin")

# ---------------- WATERMARK ----------------
st.caption("✨ Made by Preethi Regina S D")
