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
# 🔹 API CALL
# =======================
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 120, "temperature": 0.6}
    }

    for _ in range(3):
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

    return "⚡ AI is preparing... please wait."

# =======================
# 🔹 DOC PROCESSING
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
# 🔹 RETRIEVAL
# =======================
def get_context(docs, query):
    scored = []
    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query.lower().split())
        scored.append((score, doc.page_content))

    scored.sort(reverse=True)
    return "\n".join([doc for _, doc in scored[:3]])

# =======================
# 🔹 TYPING EFFECT
# =======================
def typing_effect(text):
    placeholder = st.empty()
    result = ""
    for ch in text:
        result += ch
        placeholder.markdown(result)
        time.sleep(0.004)

# =======================
# 🔹 MULTI-AGENTS 🧠
# =======================

def summarizer_agent(context, lang):
    return query_huggingface(f"Summarize in {lang}: {context}")

def quiz_agent(context, lang):
    return query_huggingface(f"Create quiz questions in {lang}: {context}")

def explainer_agent(query, context, lang, history):
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
# 🔹 AGENT ORCHESTRATOR ✅
# =======================
def multi_agent_system(query, context, lang, history):
    q = query.lower()

    if "summarize" in q:
        return summarizer_agent(context, lang)

    elif "quiz" in q:
        summary = summarizer_agent(context, lang)
        return quiz_agent(summary, lang)

    else:
        return explainer_agent(query, context, lang, history)

# =======================
# 🔹 UI (PARALLAX + GLASS)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

# 🌊 PARALLAX + GLASS CSS
st.markdown("""
<style>

/* 🌊 Animated gradient background */
body {
    background: linear-gradient(270deg, #1e293b, #0f172a, #1e293b);
    background-size: 600% 600%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%}
    50% {background-position: 100% 50%}
    100% {background-position: 0% 50%}
}

/* 🧊 Glass effect */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* 🤖 Robot animation */
.robot {
    font-size: 50px;
    text-align: center;
    animation: float 3s infinite ease-in-out;
}

@keyframes float {
    0% {transform: translateY(0);}
    50% {transform: translateY(-15px);}
    100% {transform: translateY(0);}
}

/* Chat animation */
.stChatMessage {
    animation: fade 0.3s ease-in;
}

@keyframes fade {
    from {opacity: 0;}
    to {opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# 🤖 Robot
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="glass">
<h2 style='text-align:center;'>ChatDocAI</h2>
<p style='text-align:center;'>Talk to your documents with intelligent AI ✨</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

st.sidebar.markdown("---")
st.sidebar.write("💡 Try:")
st.sidebar.write("• Summarize")
st.sidebar.write("• Quiz")
st.sidebar.write("• Explain")

# Upload
st.markdown("""
<div class="glass">
📄 Upload your document and start chatting instantly
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = ""

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MAIN
if uploaded_file:
    docs = process_document(uploaded_file)
    st.success("✅ Document ready")

    query_huggingface("hello")  # warm-up

    user_input = st.chat_input("💬 Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):

                context = get_context(docs, user_input)

                response
