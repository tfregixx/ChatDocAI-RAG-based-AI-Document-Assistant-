import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =======================
# 🔹 CONFIG (HuggingFace)
# =======================
HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 HF API CALL
# =======================
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7
        }
    }

    for _ in range(3):
        try:
            response = requests.post(
                MODEL_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()[0]["generated_text"]

            elif response.status_code == 503:
                time.sleep(5)

        except requests.exceptions.RequestException:
            time.sleep(5)

    return "⚠️ AI model busy. Try again."

# =======================
# 🔹 DOCUMENT PROCESSING
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
# 🔹 SMART RETRIEVAL
# =======================
def get_context(docs, query):
    scored_docs = []

    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query.lower().split())
        scored_docs.append((score, doc.page_content))

    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for _, doc in scored_docs[:3]]

    return "\n".join(top_docs)

# =======================
# 🔹 AGENT TOOLS
# =======================
def summarize_tool(context, language):
    return query_huggingface(f"Summarize in {language}:\n{context}")

def quiz_tool(context, language):
    return query_huggingface(f"Create 3 quiz questions in {language}:\n{context}")

def explain_tool(query, context, language, history):
    return query_huggingface(f"""
    Answer in {language}

    Conversation History:
    {history}

    Context:
    {context}

    Question:
    {query}
    """)

# =======================
# 🔹 MULTI-STEP AGENT
# =======================
def agent_pipeline(query, context, language, history):
    q = query.lower()

    if "summarize" in q:
        return summarize_tool(context, language)

    elif "quiz" in q or "question" in q:
        summary = summarize_tool(context, language)
        return quiz_tool(summary, language)

    else:
        return explain_tool(query, context, language, history)

# =======================
# 🔹 UI
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

# ✅ Animation
st.markdown("""
<style>
.stChatMessage {animation: fadeIn 0.4s ease-in;}
@keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}

.header-box {
    padding: 15px;
    border-radius: 12px;
    background: linear-gradient(90deg, #1f77b4, #9467bd);
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ✅ Header
st.markdown("""
<div class="header-box">
<h2>🤖 ChatDocAI</h2>
<p>Agentic AI Document Assistant (RAG + Multi-Step)</p>
</div>
""", unsafe_allow_html=True)

# ✅ Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Tamil", "Spanish", "French", "Hindi"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 Example Prompts")
st.sidebar.write("• Summarize document")
st.sidebar.write("• Generate quiz")
st.sidebar.write("• Explain concept")

st.sidebar.markdown("---")
st.sidebar.success("✅ Agentic AI Enabled")

# ✅ Upload
st.markdown("### 📂 Upload Document")
uploaded_file = st.file_uploader("", type=["pdf"])

# ✅ Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = ""

# ✅ Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN FLOW
# =======================

if uploaded_file:
    docs = process_document(uploaded_file)
    st.success("✅ Document processed!")

    user_input = st.chat_input("💬 Ask your question...")

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):

                context = get_context(docs, user_input)

                response = agent_pipeline(
                    user_input,
                    context,
                    language,
                    st.session_state.history
                )

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                st.session_state.history += f"\nUser: {user_input}\nAI: {response}"

        with st.expander("📄 Context Used"):
            st.write(context)
