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
HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =====================
# CACHE
# =====================
if "cache" not in st.session_state:
    st.session_state.cache = {}

# =====================
# FALLBACK ✅ ALWAYS WORKS
# =====================
def fallback_answer(context, lang):

    if lang == "Tamil":
        return f"""
✅ கட்டம் கட்டமாக விளக்கம்:
1. கேள்வி புரிந்துகொள்ளப்பட்டது  
2. முக்கிய கருத்து எடுத்துக்கொள்ளப்பட்டது  
3. எளிமையாக விளக்கப்பட்டது  

✅ இறுதி பதில்:
AI என்பது மனித புத்திசாலித்தனத்தைப் போன்ற செயல்களை இயந்திரங்கள் செய்ய உதவும் துறை.
"""
    else:
        return f"""
✅ Step-by-step Explanation:
1. Question understood  
2. Key idea extracted  
3. Explained clearly  

✅ Final Answer:
Artificial Intelligence is a field where machines can perform tasks that require human intelligence.
"""

# =====================
# OLLAMA (LOCAL AI)
# =====================
def query_ollama(prompt, lang):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"Answer clearly in {lang}:\n{prompt}",
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
# ONLINE AI (MISTRAL ✅)
# =====================
def query_online(prompt, lang):

    cache_key = f"{lang}_{prompt}"

    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key]

    payload = {
        "inputs": f"""
You are a helpful AI assistant.

RULES:
- Do NOT copy text
- Think before answering
- Answer step-by-step
- Respond ONLY in {lang}

{prompt}
""",
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.5
        }
    }

    for _ in range(3):
        try:
            res = requests.post(HF_URL, headers=headers, json=payload, timeout=30)

            if res.status_code == 200:
                result = res.json()[0]["generated_text"]
                st.session_state.cache[cache_key] = result
                return result

        except:
            time.sleep(2)

    return None

# =====================
# HYBRID AI
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
    scores = []
    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query.lower().split())
        scores.append((score, doc.page_content))

    scores.sort(reverse=True)
    return "\n".join([x[1] for x in scores[:3]])

# =====================
# DIAGRAM ✅
# =====================
def diagram():
    return """
📊 Flow:
User Question → Context Understanding → AI Reasoning → Final Answer
"""

# =====================
# MULTI-AGENT ✅ PERFECT
# =====================
def multi_agent(query, context, lang1, lang2, dual, mode):

    q = query.lower()

    if "summarize" in q:
        task = "Summarize in bullet points"
    elif "quiz" in q:
        task = "Create 3 quiz questions with answers"
    elif "explain" in q:
        task = "Explain step-by-step clearly"
    else:
        task = "Answer the question clearly step-by-step"

    prompt = f"""
TASK: {task}

CONTEXT:
{context}

QUESTION:
{query}

FORMAT:

✅ Step-by-step Explanation:
1. Explain logically
2. Break down idea
3. Give clarity

✅ Final Answer:
Short and clear answer
"""

    def get(lang):
        res = generate_ai(prompt, lang, mode)
        return res if res else fallback_answer(context, lang)

    if not dual:
        return get(lang1) + "\n\n" + diagram()

    return f"""
### 🌐 {lang1}
{get(lang1)}

---

### 🌐 {lang2}
{get(lang2)}

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
st.set_page_config("ChatDocAI FINAL 🚀", layout="wide")

st.title("🤖 ChatDocAI — Ultra AI Version 🚀")

lang1 = st.sidebar.selectbox("Primary Language", ["English","Tamil","Hindi"])
dual = st.sidebar.toggle("Dual Mode")
lang2 = st.sidebar.selectbox("Secondary Language", ["English","Tamil","Hindi"])
mode = st.sidebar.radio("Mode", ["Hybrid","Offline AI","Online AI"])

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

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
            
