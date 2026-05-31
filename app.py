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
# 🔹 CACHE
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =======================
# 🔹 STRONG TRANSLATION FUNCTION ✅
# =======================
def query_hf(prompt, lang):

    cache_key = f"{lang}_{prompt}"
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    # ✅ Strong instruction fix
    if lang == "English":
        instruction = "Answer clearly in English."
    else:
        instruction = f"""
        Translate the answer fully into {lang}.
        DO NOT use English at all.
        Use simple and natural {lang}.
        """

    payload = {
        "inputs": f"{instruction}\n{prompt}",
        "parameters": {"max_new_tokens": 120, "temperature": 0.4}
    }

    for _ in range(5):
        try:
            res = requests.post(MODEL_URL, headers=headers, json=payload, timeout=15)

            if res.status_code == 200:
                output = res.json()[0]["generated_text"]

                st.session_state.cache[cache_key] = output
                return output
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
# 🔹 FALLBACK (TRANSLATED ✅)
# =======================
def fallback_answer(context, lang):

    short_context = context[:300]

    fallback_prompt = f"""
    Explain this in {lang}:

    {short_context}
    """

    result = query_hf(fallback_prompt, lang)

    return result if result else short_context

# =======================
# 🔹 MULTI-AGENT (PERFECT LANG)
# =======================
def multi_agent(query, context, lang1, lang2, dual_mode):

    base_prompt = f"""
    Use document content to answer clearly.

    Context:
    {context}

    Question:
    {query}
    """

    # ✅ SINGLE MODE
    if not dual_mode:
        result = query_hf(base_prompt, lang1)
        return result if result else fallback_answer(context, lang1)

    # ✅ DUAL MODE (SAFE)
    res1 = query_hf(base_prompt, lang1)
    res2 = query_hf(base_prompt, lang2)

    if not res1:
        res1 = fallback_answer(context, lang1)

    if not res2:
        res2 = fallback_answer(context, lang2)

    return f"""
### 🌐 {lang1}
{res1}

---

### 🌐 {lang2}
{res2}
"""

# =======================
# 🔹 FAST TYPING
# =======================
def typing_effect(text):
    placeholder = st.empty()
    out = ""

    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(0.0015)

# =======================
# 🔹 UI (PREMIUM)
# =======================

st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(270deg,#020617,#0f172a,#020617);
    background-size: 600% 600%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

.glass {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
}

.robot {
    font-size: 50px;
    text-align:center;
    animation: float 3s infinite;
}

@keyframes float {
    50%{transform: translateY(-10px);}
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2 style='text-align:center;'>ChatDocAI</h2>
<p style='text-align:center;'>Perfect multilingual AI assistant 🌍</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

lang1 = st.sidebar.selectbox(
    "Primary Language",
    ["English", "Hindi", "Tamil", "Spanish", "French"]
)

dual_mode = st.sidebar.toggle("🌐 Dual Language Mode")

lang2 = st.sidebar.selectbox(
    "Secondary Language",
    ["English", "Hindi", "Tamil", "Spanish", "French"],
    index=2
)

st.sidebar.markdown("💡 Try: Summarize / Explain / Quiz")

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =======================
# 🔹 MAIN
# =======================
if uploaded_file:

    docs = process_document(uploaded_file.read())
    st.success("✅ Document ready!")

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
                    lang1,
                    lang2,
                    dual_mode
                )

                # ✅ safety retry
                if not response:
                    time.sleep(2)
                    response = multi_agent(
                        user_input,
                        context,
                        lang1,
                        lang2,
                        dual_mode
                    )

                typing_effect(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
