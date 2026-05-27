import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

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
# 🔹 RETRIEVAL FUNCTION (REPLACING RetrievalQA)
# =======================
def get_context(vectordb, query):
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    return context

# =======================
# 🔹 AGENT TOOLS
# =======================
def summarize_tool(text):
    return f"📘 Summary:\n{text[:300]}..."

def quiz_tool(text):
    return "🧠 Quiz:\n1. What is the main idea?\n2. Explain key concepts."

tools = [
    Tool(
        name="Summarizer",
        func=summarize_tool,
        description="Use to summarize content"
    ),
    Tool(
        name="QuizGenerator",
        func=quiz_tool,
        description="Use to generate quiz questions"
    )
]

# =======================
# 🔹 AGENT
# =======================
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

def run_agent(query, context):
    full_prompt = f"""
    User Query: {query}

    Context:
    {context}

    Perform the task and generate final answer.
    """

    return agent.run(full_prompt)

# =======================
# 🔹 STREAMLIT UI
# =======================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.title("🤖 ChatDocAI – RAG + Agentic AI")
st.markdown("🚀 GenAI system using **LLM + Retrieval + Agent reasoning**")

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

            # ✅ Step 1: Retrieve context (RAG)
            context = get_context(vectordb, query)

            # ✅ Step 2: Agent reasoning
            final_answer = run_agent(query, context)

        st.subheader("🧠 Final Answer")
        st.markdown(final_answer)

        with st.expander("📄 Retrieved Context"):
            st.write(context)
