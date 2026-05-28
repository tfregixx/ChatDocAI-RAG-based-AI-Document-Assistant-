import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# ✅ Embeddings (NEW)
from sentence_transformers import SentenceTransformer
import numpy as np

# =======================
# 🔹 CONFIG
# =======================
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")

MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 SESSION CACHE ✅
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =======================
# 🔹 EMBEDDING MODEL ✅
# =======================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# =======================
# 🔹 API CALL (WITH CACHE ✅)
# =======================
def query_huggingface(prompt):

    # ✅ SESSION CACHE HIT
    if prompt in st.session_state.cache:
        return st.session_state.cache[prompt]

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
                result = response.json()[0]["generated_text"]

                # ✅ SAVE TO CACHE
                st.session_state.cache[prompt] = result
                return result

            elif response.status_code == 503:
                time.sleep(2)

        except:
            time.sleep(2)

    return "⚡ AI is preparing... please retry."

# =======================
# 🔹 DOCUMENT PROCESSING
# =======================
@st.cache_resource
def process_document(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]

    # ✅ CREATE EMBEDDINGS
    embeddings = embed_model.encode(texts)

    return texts, embeddings

# =======================
# 🔹 SEMANTIC RETRIEVAL ✅
# =======================
def get_context(texts, embeddings, query):

    query_embedding = embed_model.encode([query])[0]

    # cosine similarity
    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[-3:][::-1]

    return "\n".join([texts[i] for i in top_indices])

# =======================
# 🔹 OFFLINE MODE ✅
# =======================
def offline_answer(context, query):

    return f"""
    📄 Based on document:

    {context[:500]}

    ✅ Answer:
    The document discusses concepts related to your question. Please review the highlighted context.
    """

# =======================
# 🔹 MULTI AGENT ✅
# =======================
def multi_agent(query, context, lang, history, offline):

    if offline:
        return offline_answer(context, query)

    q = query.lower()

    if "summarize" in q:
        return query_huggingface(f"Summarize in {lang}: {context}")

    elif "quiz" in q:
        summary = query_huggingface(f"Summarize in {lang}: {context}")
        return query_huggingface(f"Create quiz in {lang}: {summary}")

    else:
        return query_huggingface(f"""
        Answer in {lang}

        Context:
        {context}

        Question:
        {query}
        """)

# =======================
# 🔹 TYPING EFFECT
# =======================
def typing_effect(text):
    placeholder = st.empty()
    msg = ""
    for ch in text:
        msg += ch
        placeholder.markdown(msg)
        time.sleep(0.002)

# =======================
# 🔹 UI (PARALLAX + GLASS)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(270deg,#1e293b,#020617,#1e293b);
    background-size: 600% 600%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0%{background-position:0 50%}
    50%{background-position:100% 50%}
    100%{background-position:0 50%}
}

.robot {
    font-size: 50px;
    text-align: center;
    animation: float 3s infinite;
}

@keyframes float {
    50%{transform: translateY(-10px);}
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

# UI header
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2 style='text-align:center;'>ChatDocAI</h2>
<p style='text-align:center;'>AI that understands your documents 🧠</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

# ✅ OFFLINE SWITCH
offline_mode = st.sidebar.toggle("⚡ Offline Mode", value=False)

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN
# =======================
if uploaded_file:

    file_bytes = uploaded_file.read()
    texts, embeddings = process_document(file_bytes)

    st.success("✅ Document processed")

    user_input = st.chat_input("Ask anything...")

    if user_input:

        st.session_state.messages.append({"role":"user","content":user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):

                context = get_context(texts, embeddings, user_input)

                response = multi_agent(
                    user_input,
                    context,
                    language,
                    "",
                    offline_mode
                )

                typing_effect(response)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
