import streamlit as st
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_core.documents import Document

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ChatDocAI",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 ChatDocAI – AI Document Assistant")
st.caption("RAG + Gemini + FAISS")

st.info(
    "📂 Upload Documents → 💬 Ask Questions → 🤖 Get AI-Powered Answers"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📂 Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT Files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

language = st.sidebar.selectbox(
    "🌐 Language",
    [
        "English",
        "Tamil",
        "Hindi",
        "Spanish",
        "French"
    ]
)

generate_quiz = st.sidebar.button(
    "🎯 Generate Quiz"
)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat = []

# --------------------------------------------------
# LANGUAGE MAP
# --------------------------------------------------

LANG_MAP = {
    "English": "English",
    "Tamil": "Tamil",
    "Hindi": "Hindi",
    "Spanish": "Spanish",
    "French": "French"
}

# --------------------------------------------------
# SESSION
# --------------------------------------------------

if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------------------------
# GEMINI
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="Gemini-2.5-Flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2
    )

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents(files):

    all_docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(
            delete=False
        ) as tmp:

            tmp.write(file.read())
            path = tmp.name

        try:

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(path)

            else:
                loader = TextLoader(
                    path,
                    encoding="utf-8"
                )

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = file.name

            all_docs.extend(docs)

        except Exception as e:

            st.error(
                f"Error loading {file.name}: {e}"
            )

    return all_docs

# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def build_vectorstore(_documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(
        _documents
    )

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vectorstore

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_docs(query, vectorstore):

    return vectorstore.similarity_search_with_score(
        query,
        k=8
    )

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = PromptTemplate.from_template(
"""
You are ChatDocAI.

Answer ONLY using the provided context.

Rules:

1. Use context only.
2. If partial information exists, answer with available information.
3. Translate the answer into {language}.
4. Do not invent facts.
5. If absolutely no information exists,
respond:

"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

parser = StrOutputParser()

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if uploaded_files:

    documents = load_documents(
        uploaded_files
    )

    vectorstore = build_vectorstore(
        documents
    )

    llm = get_llm()

    query = st.chat_input(
        "💬 Ask your question..."
    )

    # ----------------------------------------------
    # CHAT
    # ----------------------------------------------

    if query:

        st.session_state.chat.append(
            ("user", query)
        )

        with st.spinner(
            "🤖 Thinking..."
        ):

            retrieved_docs = retrieve_docs(
                query,
                vectorstore
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc, score in retrieved_docs
                ]
            )

            chain = (
                prompt
                | llm
                | parser
            )

            try:

                answer = chain.invoke(
                    {
                        "context": context,
                        "question": query,
                        "language": LANG_MAP[
                            language
                        ]
                    }
                )

            except Exception as e:

                answer = (
                    f"⚠️ Error: {str(e)}"
                )

        st.session_state.chat.append(
            (
                "ai",
                answer,
                retrieved_docs
            )
        )

    # ----------------------------------------------
    # QUIZ GENERATOR
    # ----------------------------------------------

    if generate_quiz:

        with st.spinner(
            "🎯 Generating Quiz..."
        ):

            docs = retrieve_docs(
                "important concepts",
                vectorstore
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc, score in docs
                ]
            )

            quiz_prompt = PromptTemplate.from_template(
"""
Create 5 multiple-choice questions
from the context.

Context:
{context}

Format:

Question 1

A)
B)
C)
D)

Answer:

Question 2
...
"""
            )

            chain = (
                quiz_prompt
                | llm
                | parser
            )

            try:

                quiz = chain.invoke(
                    {
                        "context": context
                    }
                )

            except Exception as e:

                quiz = (
                    f"⚠️ Quiz Error: {str(e)}"
                )

        st.session_state.chat.append(
            (
                "ai",
                quiz,
                docs
            )
        )

    # ----------------------------------------------
    # DISPLAY CHAT
    # ----------------------------------------------

    for msg in st.session_state.chat:

        if msg[0] == "user":

            st.chat_message(
                "user"
            ).write(msg[1])

        else:

            with st.chat_message(
                "assistant"
            ):

                st.write(msg[1])

                if len(msg) > 2:

                    with st.expander(
                        "📄 Source Context"
                    ):

                        for doc, score in msg[2]:

                            st.markdown(
                                f"**Source:** "
                                f"{doc.metadata.get('source','Unknown')}"
                            )

                            st.markdown(
                                f"**Similarity Score:** "
                                f"{score:.4f}"
                            )

                            st.write(
                                doc.page_content[:500]
                            )

                            st.divider()

# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

else:

    st.info(
        "📂 Upload documents to begin."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "🚀 ChatDocAI – RAG + Gemini + FAISS"
)

st.caption(
    "✨ Made by Preethi Regina S D"
)
