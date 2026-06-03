import streamlit as st
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain.docstore.document import Document

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ChatDocAI – RegiBot",
    layout="wide"
)

st.title("🤖 ChatDocAI – AI Document Assistant")
st.caption("RAG + Gemini + FAISS")

st.info(
    "📂 Upload documents → 💬 Ask Questions → 🤖 AI Answers"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
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
# SESSION
# --------------------------------------------------

if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------------------------
# GEMINI MODEL
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents(files):

    texts = []

    for file in files:

        with tempfile.NamedTemporaryFile(
            delete=False
        ) as tmp:

            tmp.write(file.read())
            path = tmp.name

        if file.name.endswith(".pdf"):

            loader = PyPDFLoader(path)

        else:

            loader = TextLoader(path)

        docs = loader.load()

        texts.extend(
            [doc.page_content for doc in docs]
        )

    return texts

# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def build_vectorstore(texts):

    docs = [
        Document(page_content=text)
        for text in texts
    ]

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_docs(query, vectorstore):

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

    return docs

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = PromptTemplate.from_template(
"""
You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply:

'I could not find this information in the uploaded documents.'

Context:
{context}

Question:
{question}

Answer in {language}.
"""
)

parser = StrOutputParser()

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if uploaded_files:

    texts = load_documents(uploaded_files)

    vectorstore = build_vectorstore(texts)

    llm = get_llm()

    query = st.chat_input(
        "💬 Ask your question..."
    )

    # ------------------------------------------
    # CHAT
    # ------------------------------------------

    if query:

        st.session_state.chat.append(
            ("user", query)
        )

        with st.spinner(
            "🤖 Thinking..."
        ):

            docs = retrieve_docs(
                query,
                vectorstore
            )

            context = "\n\n".join(
                [d.page_content for d in docs]
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
                        "language": language
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
                docs
            )
        )

    # ------------------------------------------
    # QUIZ GENERATOR
    # ------------------------------------------

    if generate_quiz:

        with st.spinner(
            "🎯 Generating Quiz..."
        ):

            docs = retrieve_docs(
                "important concepts",
                vectorstore
            )

            context = "\n\n".join(
                [d.page_content for d in docs]
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

    # ------------------------------------------
    # DISPLAY CHAT
    # ------------------------------------------

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

                        for d in msg[2]:

                            st.write(
                                d.page_content[:500]
                                + "..."
                            )

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
