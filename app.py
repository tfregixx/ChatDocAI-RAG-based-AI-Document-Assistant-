import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =====================
# 🔹 CONFIG
# =====================
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY","")
HF_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =====================
# 🔹 CACHE
# =====================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =====================
# 🔹 OLLAMA (OFFLINE)
# =====================
def query_ollama(prompt, lang):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"Answer ONLY in {lang}.\nDo not copy.\n{prompt}",
                "stream": False
            },
            timeout=30
        )
        if res.status_code == 200:
            return res.json()["response"]
    except:
        return None
    return None

# =====================
# 🔹 ONLINE AI
# =====================
def query_online(prompt, lang):

    cache_key = f"{lang}_{prompt}"

    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"""
Answer ONLY in {lang}.
Do NOT copy the context.
Generate a new answer with reasoning.

{prompt}
""",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.4
        }
    }

    try:
        res = requests.post(HF_URL, headers=headers, json=payload, timeout=15)

        if res.status_code == 200:
            result = res.json()[0]["generated_text"]
            st.session_state.cache[cache_key] = result
            return result
    except:
        return None

    return None

# =====================
# 🔹 HYBRID ROUTER
# =====================
def generate_ai(prompt, lang, mode):

    final_prompt = f"""
You are an intelligent assistant.

Rules:
- Do NOT copy text
- Think step-by-step
- Give structured answer

{prompt}
"""

    if mode == "Offline AI":
        return query_ollama(final_prompt, lang)

    elif mode == "Online AI":
        return query_online(final_prompt, lang)

    elif mode == "Hybrid":
        return query_ollama(final_prompt, lang) or query_online(final_prompt, lang)

    return None

# =====================
# 🔹 DOCUMENT PROCESS
# =====================
@st.cache_resource
def process_document(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    docs = PyPDFLoader(file_path).load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# =====================
# 🔹 RETRIEVAL
# =====================
def get_context(docs, query):
    words = set(query.lower().split())
    scored = []

    for doc in docs:
        content_words = set(doc.page_content.lower().split())
        score = len(words & content_words)
        scored.append((score, doc.page_content))

    scored.sort(reverse=True)
    return "\n".join([x[1] for x in scored[:3]])

# =====================
# 🔹 DIAGRAM
# =====================
def generate_diagram():
    return """
📊 Flow:

User Query → Context Extraction → AI Reasoning → Final Answer
"""

# =====================
# 🔹 MULTI AGENT (FIXED ✅)
# =====================
def multi_agent(query, context, lang1, lang2, dual, mode):

    q = query.lower()

    if "summarize" in q:
        task = "Summarize in 3 simple bullet points."
    elif "quiz" in q:
        task = "Generate 3 quiz questions with answers."
    elif "explain" in q:
        task = "Explain step-by-step clearly."
    else:
        task = "Answer clearly."

    prompt = f"""
TASK:
{task}

CONTEXT:
{context}

QUESTION:
{query}

FORMAT:

✅ Step-by-step Explanation:
1. Explain logically
2. Extract meaning
3. Connect ideas

✅ Final Answer:
Short, clear answer
"""

    # ✅ SINGLE MODE
    if not dual:
        res = generate_ai(prompt, lang1, mode)

        if not res:
            res = "⚠️ Could not generate answer."

        return res + "\n\n" + generate_diagram()

    # ✅ DUAL MODE
    r1 = generate_ai(prompt, lang1, mode)
    r2 = generate_ai(prompt, lang2, mode)

    return f"""
### 🌐 {lang1}
{r1 if r1 else "No response"}

---

### 🌐 {lang2}
{r2 if r2 else "No response"}

{generate_diagram()}
"""

# =====================
# 🔹 STREAM EFFECT
# =====================
def typing_effect(text):
    placeholder = st.empty()
    out = ""
    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(0.001)

# =====================
# 🔹 UI
# =====================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(270deg,#020617,#0f172a,#020617);
    background-size:600% 600%;
    animation: gradient 10s infinite;
}
@keyframes gradient {
    50% {background-position:100% 50%;}
}
.robot {
    font-size:50px;
    text-align:center;
    animation: float 3s infinite;
}
@keyframes float {
    50%{transform:translateY(-10px);}
}
.glass {
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2 style="text-align:center;">ChatDocAI</h2>
<p style="text-align:center;">Final AI Assistant (Fixed ✅)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")

lang1 = st.sidebar.selectbox(
    "Primary Language",
    ["English","Hindi","Tamil","Spanish","French"]
)

dual = st.sidebar.toggle("🌐 Dual Language Mode")

lang2 = st.sidebar.selectbox(
    "Secondary Language",
    ["English","Hindi","Tamil","Spanish","French"],
    index=2
)

mode = st.sidebar.radio(
    "AI Mode",
    ["Hybrid","Offline AI","Online AI"]
)

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MAIN
if uploaded_file:

    docs = process_document(uploaded_file.read())
    st.success("✅ Document ready!")

    user_input = st.chat_input("Ask anything...")

    if user_input:

        st.session_state.messages.append({"role":"user","content":user_input})

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
                    dual,
                    mode
                )

                typing_effect(response)

                st.session_state.messages.append({
                    "role":"assistant",
                    "content":response
                })

        with st.expander("📄 Context Used"):
            st.write(context)
