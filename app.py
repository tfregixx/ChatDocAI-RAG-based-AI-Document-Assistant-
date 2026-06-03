import streamlit as st
import tempfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDOC AI", layout="wide")

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

# ---------------- LOAD TEXT ----------------
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

# ---------------- SIMPLE SEARCH ----------------
def get_relevant_docs(query, texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [query])

    scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_indices = scores.argsort()[-5:][::-1]

    return [texts[i] for i in top_indices]

# ---------------- MODEL ----------------
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

# ---------------- MAIN ----------------
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
        docs = get_relevant_docs("quiz", texts)
        context = "\n\n".join(docs)

        quiz_prompt = PromptTemplate.from_template(
            "Create 5 MCQ questions from:\n{context}"
        )

        chain = quiz_prompt | llm | parser
        quiz = chain.invoke({"context": context})

        st.session_state.chat.append(("ai", quiz))

    # DISPLAY CHAT
    for msg in st.session_state.chat:
        if msg[0] == "user":
            st.chat_message("user").write(msg[1])
        else:
            st.chat_message("assistant").write(msg[1])

else:
    st.info("Upload documents to begin")

st.caption("✨ Made by Preethi Regina S D")
