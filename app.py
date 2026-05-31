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
# 🔹 SESSION CACHE ✅
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =======================
# 🔹 SAFE API CALL (CACHED)
# =======================
def query_huggingface(prompt):

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
                st.session_state.cache[prompt] = result
                return result

            elif response.status_code == 503:
                time.sleep(2)

        except:
            time.sleep(2)

    return "⚡ AI is preparing... please try again."

# =======================
# 🔹 DOCUMENT PROCESSING ✅
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
# 🔹 LIGHTWEIGHT RETRIEVAL ✅
# =======================
def get_context(docs, query):

    query_words = set(query.lower().split())
    scored = []

    for doc in docs:
        content = doc.page_content
        content_words = set(content.lower().split())

        score = len(query_words.intersection(content_words))
        scored.append((score, content))

    scored.sort(reverse=True)
    return "\n".join([doc for _, doc in scored[:3]])

# =======================
# 🔹 OFFLINE MODE ✅
# =======================
def offline_answer(context, query):
    return f"""
📄 Based on your document:

{context[:400]}

✅ Answer:
Relevant information is shown above. You can review it for your question.
"""

# =======================
# 🔹 MULTI-AGENT ✅
# =======================
def multi_agent(query, context, lang, offline):

    if offline:
        return offline_answer(context, query)

    q = query.lower()

    if "summarize" in q:
        return query_huggingface(f"Summarize in {lang}: {context}")

    elif "quiz" in q:
        summary = query_huggingface(f"Summarize in {lang}: {context}")
        return query_huggingface(f"Create quiz questions in {lang}: {summary}")

    else:
        return query_huggingface(f"""
        Answer in {lang}

        Context:
        {context}

        Question:
        {query}
        """)

# =======================
# 🔹 TYPING EFFECT ✅
# =======================
def typing_effect(text):
    placeholder = st.empty()
    output = ""

    for ch in text:
        output += ch
        placeholder.markdown(output)
        time.sleep(0.002)

# =======================
# 🔹 UI (PARALLAX + GLASS)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>

/* 🌊 Animated gradient background */
body {
    background: linear-gradient(270deg,#1e293b,#020617,#1e293b);
    background-size: 600% 600%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

/* 🧊 Glass */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.1);
}

/* 🤖 Robot */
.robot {
    font-size:50px;
    text-align:center;
    animation: float 3s infinite;
}

@keyframes float {
    50%{transform: translateY(-10px);}
}

</style>
""", unsafe_allow_html=True)

# 🤖 Robot
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="glass">
<h2 style='text-align:center;'>ChatDocAI</h2>
<p style='text-align:center;'>AI that chats with your documents intelligently ✨</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "🌐 Language",
    ["English", "Tamil", "Hindi", "Spanish", "French"]
)

offline_mode = st.sidebar.toggle("⚡ Offline Mode", value=False)

st.sidebar.markdown("💡 Try:")
st.sidebar.write("• Summarize")
st.sidebar.write("• Quiz")
st.sidebar.write("• Explain")

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN APP
# =======================

if uploaded_file:

    file_bytes = uploaded_file.read()

    docs = process_document(file_bytes)

    st.success("✅ Document processed successfully")

    # warm-up
    query_huggingface("hello")

    user_input = st.chat_input("Ask anything about your document...")

    if user_input:

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):

                context = get_context(docs, user_input)

                response = multi_agent(
                    user_input,
                    context,
                    language,
                    offline_mode
                )

                # auto retry
                if "preparing" in response.lower():
                    time.sleep(2)
                    response = multi_agent(
                        user_input,
                        context,
                        language,
                        offline_mode
                    )

                typing_effect(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
