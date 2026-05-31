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
# 🔹 SESSION CACHE
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =======================
# 🔹 API CALL (DUAL-LANG READY)
# =======================
def query_hf(prompt, lang):

    cache_key = f"{lang}_{prompt}"
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"Answer ONLY in {lang}:\n{prompt}",
        "parameters": {"max_new_tokens": 100, "temperature": 0.5}
    }

    for _ in range(5):
        try:
            res = requests.post(MODEL_URL, headers=headers, json=payload, timeout=15)

            if res.status_code == 200:
                result = res.json()[0]["generated_text"]
                st.session_state.cache[cache_key] = result
                return result
        except:
            time.sleep(1)

    return None

# =======================
# 🔹 DOCUMENT PROCESSING
# =======================
@st.cache_resource
def process_document(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    docs = PyPDFLoader(file_path).load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    return splitter.split_documents(docs)

# =======================
# 🔹 RETRIEVAL
# =======================
def get_context(docs, query):

    q_words = set(query.lower().split())
    scores = []

    for doc in docs:
        words = set(doc.page_content.lower().split())
        score = len(q_words & words)
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)
    return "\n".join([x[1] for x in scores[:3]])

# =======================
# 🔹 MULTI-AGENT (DUAL LANG)
# =======================
def multi_agent(query, context, lang1, lang2, dual_mode):

    q = query.lower()

    base_prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    # ✅ SINGLE MODE
    if not dual_mode:
        result = query_hf(base_prompt, lang1)
        return result or context[:300]

    # ✅ DUAL LANGUAGE MODE
    res1 = query_hf(base_prompt, lang1)
    res2 = query_hf(base_prompt, lang2)

    if not res1:
        res1 = context[:200]

    if not res2:
        res2 = context[:200]

    return f"""
### 🌐 {lang1}
{res1}

---

### 🌐 {lang2}
{res2}
"""

# =======================
# 🔹 STREAMING EFFECT
# =======================
def typing_effect(text):
    placeholder = st.empty()
    out = ""

    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(0.0015)

# =======================
# 🔹 UI (ADVANCED TECH DESIGN)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>

/* 🌊 Animated gradient */
html, body {
    background: linear-gradient(270deg,#020617,#0f172a,#020617);
    background-size: 600% 600%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

/* 🧊 Glass container */
.glass {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

/* 🤖 Floating robot */
.robot {
    font-size: 52px;
    text-align:center;
    animation: float 2.5s infinite;
}

@keyframes float {
    50%{transform: translateY(-8px);}
}

/* Chat animation */
.stChatMessage {
    animation: fade 0.3s ease-in;
}
@keyframes fade {
    from{opacity:0} to{opacity:1}
}

/* Title glow */
.title {
    text-align:center;
    font-size:30px;
    color:#e2e8f0;
}

.subtitle {
    text-align:center;
    color:#94a3b8;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# 🤖 header
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<div class="title">ChatDocAI</div>
<div class="subtitle">AI-powered multi-language document assistant ⚡</div>
</div>
""", unsafe_allow_html=True)

# =======================
# 🔹 SIDEBAR
# =======================

st.sidebar.title("⚙️ Settings")

# Primary language
lang1 = st.sidebar.selectbox(
    "Primary Language",
    ["English", "Hindi", "Tamil", "Spanish", "French"]
)

# Dual mode toggle
dual_mode = st.sidebar.toggle("🌐 Dual Language Mode")

# Secondary language
lang2 = st.sidebar.selectbox(
    "Secondary Language",
    ["English", "Hindi", "Tamil", "Spanish", "French"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.write("💡 Try:")
st.sidebar.write("• Summarize")
st.sidebar.write("• Quiz")
st.sidebar.write("• Explain")

# =======================
# 🔹 UPLOAD
# =======================

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# =======================
# 🔹 MEMORY
# =======================

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN
# =======================

if uploaded_file:

    docs = process_document(uploaded_file.read())
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

                response = multi_agent(
                    user_input,
                    context,
                    lang1,
                    lang2,
                    dual_mode
                )

                # fallback
                if not response:
                    response = context[:300]

                typing_effect(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
