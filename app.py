import streamlit as st
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =======================
# 🔹 CONFIG
# =======================
HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 HF API CALL
# =======================
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    response = requests.post(MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return "⚠️ Model is loading or API issue. Try again in a few seconds."

    output = response.json()
    return output[0]["generated_text"]

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

    return docs

# =======================
# 🔹 RETRIEVAL
# =======================
def get_context(docs):
    return "\n".join([doc.page_content for doc in docs[:5]])

# =======================
# 🔹 GENERATE ANSWER
# =======================
def generate_answer(query, context):
    context = context[:1200]

    prompt = f"""
    You are an AI assistant.

    Context:
    {context}

    User Query:
    {query}

    Instructions:
    - If summarize → summarize
    - If quiz → create questions
    - Otherwise explain clearly

    Answer:
    """

    return query_huggingface(prompt)

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
st.markdown("<h1 style='text-align:center;'>🤖 ChatDocAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>AI-powered document assistant</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Document", type="pdf")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
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

                context = get_context(docs)
                response = generate_answer(user_input, context)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
