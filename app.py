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
HF_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =======================
# 🔹 CACHE
# =======================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =======================
# 🔹 OLLAMA (OFFLINE AI)
# =======================
def query_ollama(prompt, lang):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"Answer in {lang}:\n{prompt}",
                "stream": False
            },
            timeout=30
        )
        if res.status_code == 200:
            return res.json()["response"]
    except:
        return None
    return None

# =======================
# 🔹 ONLINE AI
# =======================
def query_online(prompt, lang):

    cache_key = f"{lang}_{prompt}"
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"""
Answer in {lang}.
Do NOT copy context.
Use reasoning.

{prompt}
""",
        "parameters": {"max_new_tokens": 200, "temperature": 0.4}
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

# =======================
# 🔹 HYBRID AI
# =======================
def generate_ai(prompt, lang, mode):

    if mode == "Offline AI":
        return query_ollama(prompt, lang)

    elif mode == "Online AI":
        return query_online(prompt, lang)

    elif mode == "Hybrid":
        return query_ollama(prompt, lang) or query_online(prompt, lang)

    return None

# =======================
# 🔹 DOC PROCESSING
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
    words = set(query.lower().split())
    scores = []

    for doc in docs:
        content_words = set(doc.page_content.lower().split())
        score = len(words & content_words)
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)
    return "\n".join([x[1] for x in scores[:3]])

# =======================
# 🔹 DIAGRAM GENERATOR ✅
# =======================
def generate_diagram(context):
    return f"""
📊 Diagram:

[ Question ]
   ↓
[ Extract Context ]
   ↓
{context[:80]}...
   ↓
[ AI Reasoning ]
   ↓
[ Final Answer ]
"""

# =======================
# 🔹 MULTI AGENT (REASONING)
# =======================
def multi_agent(query, context, lang1, lang2, dual, mode):

    def build_prompt():
        return f"""
You are an AI assistant.

Rules:
- Do NOT copy text
- Think step-by-step
- Explain clearly

FORMAT:

✅ Step-by-step Explanation:
1. Understand question
2. Extract meaning
3. Explain clearly

✅ Final Answer:
Short answer

Context:
{context}

Question:
{query}
"""

    prompt = build_prompt()

    # SINGLE MODE
    if not dual:
        res = generate_ai(prompt, lang1, mode)
        diagram = generate_diagram(context)

        return f"""
{res if res else context[:200]}

{diagram}
"""

    # DUAL MODE
    r1 = generate_ai(prompt, lang1, mode)
    r2 = generate_ai(prompt, lang2, mode)

    diagram = generate_diagram(context)

    return f"""
### 🌐 {lang1}
{r1 if r1 else context[:200]}

---

### 🌐 {lang2}
{r2 if r2 else context[:200]}

{diagram}
"""

# =======================
# 🔹 STREAMING
# =======================
def typing_effect(text):
    placeholder = st.empty()
    out = ""
    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(0.001)

# =======================
# 🔹 UI
# =======================
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

# Header
st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h2 style="text-align:center;">ChatDocAI</h2>
<p style="text-align:center;">Ultimate AI Assistant (Hybrid + Visual + Multilingual)</p>
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

mode = st.sidebar.radio("AI Mode", ["Hybrid","Offline AI","Online AI"])

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# MAIN
if uploaded_file:

    docs = process_document(uploaded_file.read())
    st.success("✅ Document ready!")

    user_input = st.chat_input("Ask anything about your document...")

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
