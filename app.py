import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
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
# 🔹 PROCESS DOCUMENT (NO CHROMA)
# =======================
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    return docs  # ✅ return chunks directly (NO vector DB)

# =======================
# 🔹 SIMPLE RETRIEVAL (FAST FIX)
# =======================
def get_context(docs, query):
    # naive retrieval: just join top chunks
    text = "\n".join([doc.page_content for doc in docs[:5]])
    return text

# =======================
# 🔹 RESPONSE GENERATION
# =======================
def generate_answer(query, context):
    prompt = f"""
    You are an intelligent AI assistant.

    User Query: {query}

    Document Context:
    {context}

    Instructions:
    - Summarize if asked
    - Generate quiz if asked
    - Otherwise answer clearly

    Final Answer:
    """

    return llm.invoke(prompt).content

# =======================
# 🔹 UI (STARTUP STYLE)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

# Animation
st.markdown(
    """
    <style>
    .stChatMessage {animation: fadeIn 0.4s ease-in;}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🤖 ChatDocAI")
st.sidebar.markdown("AI Document Assistant")
st.sidebar.markdown("### 💡 Examples")
st.sidebar.write("• Summarize document")
st.sidebar.write("• Generate quiz")
st.sidebar.write("• Explain concepts")

# Title
st.markdown(
    "<h1 style='text-align:center;'>🤖 ChatDocAI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered document assistant</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Upload Document", type="pdf")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if uploaded_file:
    docs = process_document(uploaded_file)
    st.success("✅ Document processed!")

    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤖"):
                context = get_context(docs, user_input)
                response = generate_answer(user_input, context)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
