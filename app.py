import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI

# =======================
# 🔹 LLM
# =======================
llm = ChatOpenAI(
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"],
    model="gpt-3.5-turbo"
)

# =======================
# 🔹 PROCESS DOCUMENT
# =======================
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings)

    return vectordb

# =======================
# 🔹 RETRIEVAL
# =======================
def get_context(vectordb, query):
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

# =======================
# 🔹 MAIN RESPONSE (AGENT SIMULATION)
# =======================
def generate_answer(query, context):
    prompt = f"""
    You are an AI assistant.

    User Query: {query}

    Context:
    {context}

    Instructions:
    - If user asks to summarize → give summary
    - If user asks for quiz/questions → generate questions
    - Otherwise answer clearly using context

    Final Answer:
    """
    return llm.invoke(prompt).content

# =======================
# 🔹 UI
# =======================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.title("🤖 ChatDocAI – RAG System")
st.markdown("🚀 AI document assistant using **LLM + semantic search**")

st.markdown("### 💡 Try asking:")
st.write("- Summarize the document")
st.write("- Generate quiz questions")
st.write("- Explain key concepts")

uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file:
    st.success("✅ File uploaded!")

    vectordb = process_document(uploaded_file)

    query = st.text_input("💬 Ask your question")

    if query:
        with st.spinner("Processing... 🤖"):

            # Step 1: Retrieve
            context = get_context(vectordb, query)

            # Step 2: Generate response
            final_answer = generate_answer(query, context)

        st.subheader("🧠 Final Answer")
        st.markdown(final_answer)

        with st.expander("📄 Retrieved Context"):
            st.write(context)
