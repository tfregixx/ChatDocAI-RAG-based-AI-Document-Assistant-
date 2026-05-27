import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool

# =======================
# 🔹 LOAD LLM (OPENAI)
# =======================
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model="gpt-3.5-turbo"
)

# =======================
# 🔹 PROCESS DOCUMENT
# =======================
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings)

    return vectordb

# =======================
# 🔹 CREATE RAG CHAIN
# =======================
def create_rag(vectordb):
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
    return qa_chain

# =======================
# 🔹 AGENT TOOLS
# =======================
def summarize_tool(text):
    return f"### 📘 Summary\n{text[:300]}..."

def quiz_tool(text):
    return "### 🧠 Quiz\n1. What is the main idea?\n2. Explain key concepts."

tools = [
    Tool(
        name="Summarizer",
        func=summarize_tool,
        description="Use this tool when user asks to summarize content"
    ),
    Tool(
        name="QuizGenerator",
        func=quiz_tool,
        description="Use this tool when user asks to generate quiz or questions"
    )
]

# =======================
# 🔹 CREATE AGENT
# =======================
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

def run_agent(query, context):
    prompt = f"""
    User Query: {query}

    Context:
    {context}

    Decide the steps and generate the final response.
    """
    return agent.run(prompt)

# =======================
# 🔹 STREAMLIT UI
# =======================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.title("🤖 ChatDocAI – RAG + Agentic AI")
st.markdown("🚀 AI-powered document assistant using **RAG + LLM + Agent reasoning**")

st.markdown("### 💡 Try asking:")
st.write("- Summarize the document")
st.write("- Generate quiz questions")
st.write("- Explain key concepts")

uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    vectordb = process_document(uploaded_file)
    qa_chain = create_rag(vectordb)

    query = st.text_input("💬 Ask your question")

    if query:
        with st.spinner("Processing document and generating response... 🤖"):
            # Step 1: RAG
            rag_output = qa_chain.run(query)

            # Step 2: Agent reasoning
            final_answer = run_agent(query, rag_output)

        # ✅ Display results
        st.subheader("🧠 Final Answer")
        st.markdown(final_answer)

        # ✅ Show context (IMPORTANT for recruiters)
        with st.expander("📄 Retrieved Context"):
            st.write(rag_output)
