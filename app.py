import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =======================
# 🔹 CONFIG
# =======================
HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 API CALL (AUTO RETRY)
# =======================
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.6
        }
    }

    for _ in range(4):
        try:
            response = requests.post(
                MODEL_URL,
                headers=headers,
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                return response.json()[0]["generated_text"]

            elif response.status_code == 503:
                time.sleep(3)

        except:
            time.sleep(3)

    return "🤖 AI is warming up... please try again."

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
    return splitter.split_documents(documents)

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
# 🔹 TYPING EFFECT
# =======================
def typing_effect(text):
    output = ""
    placeholder = st.empty()

    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(0.01)

# =======================
# 🔹 AGENT TOOLS
# =======================
def summarize_tool(context, lang):
    return query_huggingface(f"Summarize in {lang}:\n{context}")

def quiz_tool(context, lang):
    return query_huggingface(f"Create 3 quiz questions in {lang}:\n{context}")

def explain_tool(query, context, lang, history):
    return query_huggingface(f"""
    Answer in {lang}

    History:
    {history}

    Context:
    {context}

    Question:
    {query}
    """)

# =======================
# 🔹 AGENT PIPELINE
# =======================
def agent_pipeline(query, context, lang, history):
    q = query.lower()

    if "summarize" in q:
        return summarize_tool(context, lang)

    elif "quiz" in q or "question" in q:
        summary = summarize_tool(context, lang)
        return quiz_tool(summary, lang)

    else:
        return explain_tool(query, context, lang, history)

# =======================
# 🔹 UI
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

# CSS
st.markdown("""
<style>
.header-box {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1f77b4, #8e44ad);
    color: white;
    text-align: center;
}

.robot {
    font-size: 40px;
    animation: bounce 2s infinite;
    text-align: center;
}

@keyframes bounce {
    0% {transform: translateY(0);}
    50% {transform: translateY(-10px);}
    100% {transform: translateY(0);}
}

.stChatMessage {
    animation: fadeIn 0.4s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ROBOT
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-box">
<h2>ChatDocAI</h2>
<p>Smart AI assistant that understands and chats with your documents ✨</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
"<p style='text-align:center;color:gray;'>Upload a PDF and start an intelligent conversation 📄💬</p>",
unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

st.sidebar.markdown("### 💡 Try:")
st.sidebar.write("• Summarize")
st.sidebar.write("• Generate quiz")
st.sidebar.write("• Explain content")

st.sidebar.success("✅ Agent AI Ready")

# UPLOAD
uploaded_file = st.file_uploader("📂 Upload your document", type=["pdf"])

# MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = ""

# CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MAIN
if uploaded_file:
    docs = process_document(uploaded_file)
    st.success("✅ Document ready!")

    user_input = st.chat_input("Ask anything about your document...")

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

                typing_effect(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                st.session_state.history += f"\nUser: {user_input}\nAI: {response}"

        with st.expander("📄 Context Used"):
            st.write(context)

