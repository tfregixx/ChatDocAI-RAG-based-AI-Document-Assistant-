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
# 🔹 RESPONSE GENERATION
# =======================
def generate_answer(query, context):
    prompt = f"""
    You are an intelligent AI assistant.

    User Query: {query}

    Context:
    {context}

    Instructions:
    - If asked to summarize → provide concise summary
    - If asked for quiz → generate questions
    - Else → explain clearly using context

    Final Answer:
    """
    return llm.invoke(prompt).content

# =======================
# 🔹 UI (STARTUP STYLE)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

# ✅ Animation CSS
st.markdown(
    """
    <style>
    .stChatMessage {animation: fadeIn 0.4s ease-in;}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar
st.sidebar.title("🤖 ChatDocAI")
st.sidebar.markdown("AI Document Assistant")
st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 Examples")
st.sidebar.write("• Summarize document")
st.sidebar.write("• Generate quiz")
st.sidebar.write("• Explain concepts")

st.sidebar.markdown("---")
st.sidebar.info("RAG + LLM + Reasoning AI")

# ✅ Title
st.markdown(
    """
    <h1 style='text-align:center;'>🤖 ChatDocAI</h1>
    <p style='text-align:center;color:gray;'>
    AI-powered document assistant
    </p>
    """,
    unsafe_allow_html=True
)

# ✅ Upload Section
st.markdown("### 📂 Upload Document")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ✅ Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Main Logic
if uploaded_file:
    vectordb = process_document(uploaded_file)
    st.success("✅ Document processed!")

    user_input = st.chat_input("Ask something about your document...")

    if user_input:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤖"):

                context = get_context(vectordb, user_input)
                response = generate_answer(user_input, context)

                st.markdown(response)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        # Context display
        with st.expander("📄 Retrieved Context"):
            st.write(context)

