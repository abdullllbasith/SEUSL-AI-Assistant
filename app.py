import streamlit as st
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer
import json
import time

# --- Set Page Config ---
st.set_page_config(
    page_title="SEUSL AI",
    page_icon="https://www.seu.ac.lk/images/seu_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Responsive Custom CSS for DeepSeek-like UI, chat-container, and fixed header ---
st.markdown("""
    <style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        padding-bottom: 120px;
        padding-top: 100px !important;
    }
    .fixed-header {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 50%;
        z-index: 10000;
        background: rgba(34, 40, 49, 0.65);
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        border-bottom: 1px solid #393e46;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 130px;
        pointer-events: auto;
        border-bottom-left-radius: 65px;
        border-bottom-right-radius: 65px;
        transition: width 0.3s, height 0.3s;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .header-inner {
        display: flex;
        align-items: center;
        justify-content: center;
        max-width: 900px;
        width: 100%;
        margin: 0 auto;
        padding: 0 16px;
    }
    .logo-title-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .logo-row {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .logo {
        height: 70px;
        width: 60px;
        margin-right: 12px;
        display: block;
        margin-top: 24px;
    }
    .title {
        font-size: 30px !important;
        font-weight: 450 !important;
        color: #FAFAFA !important;
        margin: 0;
        text-align: center !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stMainBlockContainer{
        margin-top:75px;
    }
    .subtitle {
        font-size: 20px;
        font-weight: 500;
        color: #FAFAFA;
        margin: 0;
        text-align: center;
        margin-top: -18px;
        letter-spacing: 1px;
        font-family: 'Inter', sans-serif !important;
        padding-bottom: 25px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Responsive styles */
    @media (max-width: 1200px) {
        .fixed-header {
            width: 80%;
        }
    }
    @media (max-width: 900px) {
        .stApp {
            max-width: 100vw;
            padding-left: 0;
            padding-right: 0;
        }
        .fixed-header {
            width: 98vw;
            left: 50%;
            transform: translateX(-50%);
            border-bottom-left-radius: 30px;
            border-bottom-right-radius: 30px;
            height: 140px !important;
        }
        .header-inner {
            max-width: 100vw;
            padding: 0 8px;
        }
        .logo {
            height: 50px;
            width: 40px;
            margin-top: 12px;
        }
        .title {
            font-size: 22px !important;
        }
        
        .stMainBlockContainer{
            margin-top:55px;
        }
        .subtitle {
            font-size: 15px;
            padding-top: 10px;
        }
    }
    @media (max-width: 600px) {
        .fixed-header {
            width: 100vw;
            left: 50%;
            transform: translateX(-50%);
            border-bottom-left-radius: 18px;
            border-bottom-right-radius: 18px;
            height: 90px;
        }
        .header-inner {
            padding: 0 2vw;
        }
        .logo {
            height: 36px;
            width: 30px;
            margin-right: 10px;
            margin-top: 20px;
        }
        .title {
            font-size: 17px !important;
        }
        .subtitle {
            font-size: 13px;
            margin-top: -8px;
            padding-bottom: 4px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Fixed Header (Permanently Frozen) ---
st.markdown("""
<div class="fixed-header">
    <div class="header-inner">
        <div class="logo-title-wrapper">
            <div class="logo-row">
                <img src="https://www.seu.ac.lk/images/seu_logo.png" class="logo" alt="SEUSL Logo">
                <h1 class="title">South Eastern University of Sri Lanka</h1>
            </div>
            <div class="subtitle">The AI Assistant</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your SEUSL AI assistant. How can I help you today?"}
    ]

if "history" not in st.session_state:
    st.session_state.history = []

# --- Load FAISS index and chunks for RAG ---
AIMLAPI_KEY = "feb75b45a9b646f19798a9cd465ac71c"
MODEL = "google/gemma-3n-e4b-it"

index = faiss.read_index("seusl_index.faiss")
with open("seusl_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search_chunks(query, k=7):
    query_embedding = embedder.encode([query])
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def ask_gemma(question, context):
    url = "https://api.aimlapi.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIMLAPI_KEY}",
        "Content-Type": "application/json"
    }
    messages = st.session_state.history.copy()
    messages.append({
        "role": "user",
        "content": f"You are a helpful assistant for SEUSL. Use the following context to answer the question. The question may be a follow-up to the conversation.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    })
    payload = {
        "model": MODEL,
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Unexpected response: {result}"
    except Exception as e:
        return f"❌ Error parsing response: {e}\n\nRaw response: {response.text}"

def scroll_to_bottom():
    st.markdown("<div id='scroll-anchor'></div>", unsafe_allow_html=True)

# --- Show full conversation using chat_message ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

scroll_to_bottom()

# --- Fixed bottom input bar ---
user_input = st.chat_input("Ask something about SEUSL...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Show loading spinner with 3 dots animation while processing
    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        # Responsive Custom CSS for chat bubble loading dots
        st.markdown("""
            <style>
            .dot-typing {
                position: relative;
                left: 0;
                width: 40px;
                height: 20px;
                margin: 0;
                padding: 0;
                display: inline-block;
                vertical-align: middle;
            }
            .dot-typing span {
                display: inline-block;
                width: 8px;
                height: 8px;
                margin: 0 2px;
                background: #bbb;
                border-radius: 50%;
                opacity: 0.6;
                animation: dot-typing 1s infinite;
            }
            .dot-typing span:nth-child(2) { animation-delay: 0.2s; }
            .dot-typing span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes dot-typing {
                0% { transform: translateY(0); opacity: 0.6; }
                20% { transform: translateY(-4px); opacity: 1; }
                40% { transform: translateY(0); opacity: 0.6; }
            }
            @media (max-width: 600px) {
                .dot-typing {
                    width: 24px;
                    height: 12px;
                }
                .dot-typing span {
                    width: 5px;
                    height: 5px;
                    margin: 0 1px;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        # Show animated loading dots in chat bubble
        loading_html = """
        <div class="dot-typing">
            <span></span><span></span><span></span>
        </div>
        """
        loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
        # Simulate loading while processing
        time.sleep(2)
        # Search RAG chunks
        context = "\n".join(search_chunks(user_input))

        # Ask Gemma model with chat history
        bot_reply = ask_gemma(user_input, context)

        # Show bot reply
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.session_state.history.append({"role": "assistant", "content": bot_reply})

        loading_placeholder.markdown(bot_reply)
        time.sleep(0.2)
        st.rerun()