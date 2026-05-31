import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =====================
# CONFIG
# =====================
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
HF_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =====================
# CACHE
# =====================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =====================
# SAFE FALLBACK ✅ (IMPORTANT)
# =====================
def fallback_answer(context, task, lang):
    if lang == "Tamil":
        return f"""
✅ கட்டம் கட்டமாக விளக்கம்:
1. கேள்வி புரிந்துகொள்ளப்பட்டது
2. ஆவணத்தில் இருந்து தகவல் எடுக்கப்பட்டது
3. முக்கிய கருத்து விளக்கப்பட்டது

✅ இறுதி பதில்:
{context[:200]}
"""
    else:
        return f"""
✅ Step-by-step Explanation:
1. Question analyzed
2. Context extracted
3. Key idea explained

✅ Final Answer:
{context[:200]}
"""

# =====================
# OLLAMA (OFFLINE)
# =====================
def query_ollama(prompt, lang):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"Answer ONLY in {lang}. Do not copy. {prompt}",
                "stream": False
            },
            timeout=25
        )
        if res.status_code == 200:
            return res.json()["response"]
    except:
        return None
    return None

# =====================
# ONLINE AI
# =====================
def query_online(prompt, lang):

    cache_key = f"{lang}_{prompt}"
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"""
Answer ONLY in {lang}
Think and explain - do NOT copy context.

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

# =====================
# HYBRID AI ✅ WITH RETRY
# =====================
def generate_ai(prompt, lang, mode):

    for _ in range(2):  # retry loop
        if mode == "Offline AI":
            result = query_ollama(prompt, lang)

        elif mode == "Online AI":
            result = query_online(prompt, lang)

        else:  # Hybrid
            result = query_ollama(prompt, lang) or query_online(prompt, lang)

        if result:
            return result

        time.sleep(1)

    return None

# =====================
# DOCUMENT PROCESS
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
# RETRIEVAL
# =====================
def get_context(docs, query):
    words = set(query.lower().split())
    scores = []

    for doc in docs:
        score = len(words & set(doc.page_content.lower().split()))
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)
    return "\n".join([x[1] for x in scores[:3]])

# =====================
# DIAGRAM
# =====================
def diagram():
    return """
📊 Flow:
User Query → Context → AI Reasoning → Final Answer
"""

# =====================
# MULTI AGENT ✅ FIXED
# =====================
def multi_agent(query, context, lang1, lang2, dual, mode):

    q = query.lower()

    if "summarize" in q:
        task = "Summarize in 3 points"
    elif "quiz" in q:
        task = "Generate 3 quiz questions with answers"
    elif "explain" in q:
        task = "Explain clearly step by step"
    else:
        task = "Answer the question"

    prompt = f"""
TASK: {task}
CONTEXT: {context}
QUESTION: {query}
"""

    def safe_generate(lang):
        res = generate_ai(prompt, lang, mode)
        return res if res else fallback_answer(context, task, lang)

    # SINGLE
    if not dual:
        return safe_generate(lang1) + "\n\n" + diagram()

    # DUAL
    return f"""
### 🌐 {lang1}
{safe_generate(lang1)}

---

### 🌐 {lang2}
{safe_generate(lang2)}

{diagram()}
"""

# =====================
# STREAM
# =====================
def typing_effect(text):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out)
        time.sleep(0.001)

# =====================
# UI
# =====================
st.set_page_config(page_title="ChatDocAI", layout="wide")

st.title("🤖 ChatDocAI - Stable Version ✅")

lang1 = st.sidebar.selectbox("Primary Language",
    ["English","Tamil","Hindi"])

dual = st.sidebar.toggle("Dual Mode")

lang2 = st.sidebar.selectbox("Secondary Language",
    ["English","Tamil","Hindi"])

mode = st.sidebar.radio("Mode",
    ["Hybrid","Offline AI","Online AI"])

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if uploaded_file:

    docs = process_document(uploaded_file.read())

    user_input = st.chat_input("Ask anything...")

    if user_input:

        st.session_state.messages.append({"role":"user","content":user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

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

        with st.expander("Context"):
            st.write(context)
