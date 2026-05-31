import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =======================
# 🔹 CONFIG
# =======================
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 SESSION CACHE ⚡
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

if "warm" not in st.session_state:
    st.session_state.warm = False

# =======================
# 🔹 FAST API CALL
# =======================
def query_huggingface(prompt):

    # ✅ CACHE HIT → instant
    if prompt in st.session_state.cache:
        return st.session_state.cache[prompt]

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "temperature": 0.5}
    }

    for _ in range(5):  # strong retry
        try:
            response = requests.post(
                MODEL_URL,
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()[0]["generated_text"]

                st.session_state.cache[prompt] = result
                return result

            elif response.status_code == 503:
                time.sleep(2)

        except:
            time.sleep(1)

    return None  # ✅ no error text

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
    return splitter.split_documents(docs)

# =======================
# 🔹 FAST RETRIEVAL
# =======================
def get_context(docs, query):
    query_words = set(query.lower().split())
    scores = []

    for doc in docs:
        words = set(doc.page_content.lower().split())
        score = len(query_words & words)
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)

    return "\n".join([x[1] for x in scores[:3]])

# =======================
# 🔹 OFFLINE FALLBACK
# =======================
def offline_answer(context):
    return f"""
📄 Quick answer based on document:

{context[:300]}

✅ Tip: Try asking more specific questions for better results.
"""

# =======================
# 🔹 AGENT SYSTEM
# =======================
def multi_agent(query, context, lang):

    q = query.lower()

    if "summarize" in q:
        return query_huggingface(f"Summarize in {lang}: {context}")

    elif "quiz" in q:
        summary = query_huggingface(f"Summarize in {lang}: {context}")
        return query_huggingface(f"Create 3 quiz questions in {lang}: {summary}")

    else:
        return query_huggingface(f"""
        Answer in {lang}

        Context:
        {context}

        Question:
        {query}
        """)

# =======================
# 🔹 FAST STREAMING UI
# =======================
def typing_effect(text):
    if not text:
        return
    placeholder = st.empty()
    for i in range(len(text)):
        placeholder.markdown(text[:i+1])
        time.sleep(0.0015)  # ⚡ VERY FAST
# =======================

# =======================
# 🔹 UI
# =======================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>

body {
    background: linear-gradient(270deg,#1e293b,#020617,#1e293b);
    background-size: 600% 600%;
    animation: gradientMove 10s ease infinite;
}

@keyframes gradientMove {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

.robot {
    font-size: 50px;
    text-align: center;
    animation: float 2.5s infinite;
}

@keyframes float {
    50%{transform: translateY(-8px);}
}

</style>
""", unsafe_allow_html=True)

# 🤖 UI header
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2 style='text-align:center;'>ChatDocAI</h2>
<p style='text-align:center;'>Talk to your documents instantly ⚡</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN
# =======================

if uploaded_file:

    docs = process_document(uploaded_file.read())
    st.success("✅ Document ready!")

    # ✅ WARM-UP ONCE (CRITICAL SPEED BOOST)
    if not st.session_state.warm:
        with st.spinner("⚡ Optimizing AI experience..."):
            query_huggingface("Explain AI in one line")
        st.session_state.warm = True

    user_input = st.chat_input("Ask anything...")

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

                response = multi_agent(
                    user_input,
                    context,
                    language
                )

                # ✅ AUTO RECOVERY LOOP
                if not response:
                    for _ in range(2):
                        time.sleep(2)
                        response = multi_agent(
                            user_input,
                            context,
                            language
                        )
                        if response:
                            break

                # ✅ FINAL FALLBACK
                if not response:
                    response = offline_answer(context)

                typing_effect(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
