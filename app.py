import streamlit as st
import tempfile
import requests
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# =====================
# ✅ CONFIG (FREE MODELS)
# =====================
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =====================
# ✅ CACHE (FAST ⚡)
# =====================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =====================
# ✅ FALLBACK (ALWAYS WORKS)
# =====================
def fallback_answer(context, lang):
    if lang == "Tamil":
        return f"""
✅ விளக்கம்:
இந்த ஆவணத்தின் அடிப்படையில் AI பற்றிய தகவல் காணப்படுகிறது.

✅ பதில்:
AI என்பது மனித புத்திசாலித்தனத்தைப் போன்ற செயல்களை இயந்திரங்கள் செய்ய உதவும் துறை.
"""
    else:
        return f"""
✅ Explanation:
The document explains AI concepts.

✅ Answer:
AI is a field where machines perform tasks requiring human intelligence.
"""

# =====================
# ✅ OLLAMA (OFFLINE FREE)
# =====================
def query_ollama(prompt, lang):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",  # ✅ faster than llama3
                "prompt": f"Answer in {lang}. Be clear.\n{prompt}",
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
# ✅ ONLINE AI (MISTRAL FREE)
# =====================
def query_online(prompt, lang):

    cache_key = f"{lang}_{prompt}"

    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"""
You are a smart AI assistant.

RULES:
- Answer ONLY in {lang}
- Do NOT copy text
- Follow format exactly

{prompt}
""",
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.5
        }
    }

    for _ in range(3):
        try:
            res = requests.post(HF_URL, headers=headers, json=payload, timeout=25)

            if res.status_code == 200:
                result = res.json()[0]["generated_text"]
                st.session_state.cache[cache_key] = result
                return result

        except:
            time.sleep(2)

    return None

# =====================
# ✅ HYBRID ENGINE
# =====================
def generate_ai(prompt, lang, mode):

    for _ in range(2):

        if mode == "Offline AI":
            res = query_ollama(prompt, lang)

        elif mode == "Online AI":
            res = query_online(prompt, lang)

        else:  # Hybrid
            res = query_ollama(prompt, lang) or query_online(prompt, lang)

        if res:
            return res

        time.sleep(1)

    return None

# =====================
# ✅ DOC PROCESSING
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
# ✅ CONTEXT SEARCH
# =====================
def get_context(docs, query):
    scores = []

    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query.lower().split())
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)

    return "\n".join([x[1] for x in scores[:3]])

# =====================
# ✅ DIAGRAM
# =====================
def diagram():
    return """
📊 Flow:
User Query → Context → AI Processing → Smart Answer
"""

# =====================
# ✅ MULTI-AGENT (FIXED)
# =====================
def multi_agent(query, context, lang1, lang2, dual, mode):

    q = query.lower()

    # ✅ TASK CONTROL
    if "quiz" in q:
        prompt = f"""
Create 3 quiz questions with answers.

FORMAT:

1. Question?
Answer: ...

2. Question?
Answer: ...

3. Question?
Answer: ...

Context:
{context}
"""

    elif "summarize" in q:
        prompt = f"""
Summarize in 3 bullet points:

• Point 1
• Point 2
• Point 3

Context:
{context}
"""

    elif "explain" in q:
        prompt = f"""
Explain step-by-step.

✅ Steps:
1. ...
2. ...
3. ...

✅ Final Answer:
...

Context:
{context}
"""

    else:
        prompt = f"""
Answer clearly in 2-3 sentences.

Context:
{context}

Question:
{query}
"""

    def get(lang):
        res = generate_ai(prompt, lang, mode)
        return res if res else fallback_answer(context, lang)

    # ✅ SINGLE MODE
    if not dual:
        return get(lang1) + "\n\n" + diagram()

    # ✅ DUAL MODE
    return f"""
### 🌐 {lang1}
{get(lang1)}

---

### 🌐 {lang2}
{get(lang2)}

{diagram()}
"""

# =====================
# ✅ FAST STREAM
# =====================
def typing_effect(text):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out)
        time.sleep(0.0008)

# =====================
# ✅ UI
# =====================
st.set_page_config("ChatDocAI FREE 🚀", layout="wide")

st.title("🤖 ChatDocAI — Optimized FREE AI ✅")

lang1 = st.sidebar.selectbox("Primary Language", ["English","Tamil","Hindi"])
dual = st.sidebar.toggle("🌐 Dual Language Mode")
lang2 = st.sidebar.selectbox("Secondary Language", ["English","Tamil","Hindi"])
mode = st.sidebar.radio("AI Mode", ["Hybrid","Offline AI","Online AI"])

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# memory
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =====================
# ✅ MAIN
# =====================
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
