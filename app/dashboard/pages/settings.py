import streamlit as st
import json
from pathlib import Path
import app.dashboard.components as comp

SETTINGS_FILE = Path("data") / "settings_config.json"

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "openai_api_key": "••••••••••••••••••••",
        "gemini_api_key": "••••••••••••••••••••",
        "ollama_host": "http://localhost:11434",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "system_prompt": "You are a helpful QA assistant. Use the following context items to answer the question:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    }

def save_settings(settings_dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings_dict, indent=2), encoding="utf-8")

def show():
    st.markdown("<h2 style='margin: 0 0 4px 0; font-size: 1.8rem; color: #f8fafc;'>Settings & Model Config</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px;'>Manage LLM credentials, set pipeline prompts, and adjust retrieval hyper-parameters.</p>", unsafe_allow_html=True)

    config = load_settings()

    tab1, tab2 = st.tabs(["🔒 API & Providers", "📝 RAG Prompts & Embedding"])

    with tab1:
        st.markdown("<h3 style='font-size: 1.15rem; color: #f8fafc; margin-bottom: 12px;'>Provider API Keys</h3>", unsafe_allow_html=True)
        
        with st.form("api_keys_form"):
            openai_key = st.text_input("OpenAI API Key", value=config.get("openai_api_key", ""), type="password")
            gemini_key = st.text_input("Gemini API Key", value=config.get("gemini_api_key", ""), type="password")
            ollama_host = st.text_input("Ollama Host URL", value=config.get("ollama_host", "http://localhost:11434"))
            
            st.markdown("<br/>", unsafe_allow_html=True)
            save_api = st.form_submit_button("Save Provider Settings")
            
            if save_api:
                config["openai_api_key"] = openai_key
                config["gemini_api_key"] = gemini_key
                config["ollama_host"] = ollama_host
                save_settings(config)
                st.success("✓ Provider integrations updated.")

        st.markdown("<br/><h3 style='font-size: 1.15rem; color: #f8fafc; margin-bottom: 12px;'>Connection Verification</h3>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Ollama Connection**")
            # Run simple check
            import socket
            try:
                # Parse host
                port = 11434
                if ":" in ollama_host.replace("http://", ""):
                    port = int(ollama_host.split(":")[-1].replace("/", ""))
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                s.close()
                st.markdown(comp.status_badge("SUCCESS") + " CONNECTED", unsafe_allow_html=True)
            except Exception:
                st.markdown(comp.status_badge("DANGER") + " DISCONNECTED", unsafe_allow_html=True)
        with c2:
            st.markdown("**OpenAI Endpoint**")
            if openai_key and openai_key != "••••••••••••••••••••":
                st.markdown(comp.status_badge("SUCCESS") + " ACTIVE KEY", unsafe_allow_html=True)
            else:
                st.markdown(comp.status_badge("WARNING") + " NO KEY SET", unsafe_allow_html=True)
        with c3:
            st.markdown("**Gemini API Link**")
            if gemini_key and gemini_key != "••••••••••••••••••••":
                st.markdown(comp.status_badge("SUCCESS") + " ACTIVE KEY", unsafe_allow_html=True)
            else:
                st.markdown(comp.status_badge("WARNING") + " NO KEY SET", unsafe_allow_html=True)

    with tab2:
        st.markdown("<h3 style='font-size: 1.15rem; color: #f8fafc; margin-bottom: 12px;'>Retrieval Prompt Templates</h3>", unsafe_allow_html=True)
        
        with st.form("rag_prompts_form"):
            emb_model = st.text_input("Embedding Model ID", value=config.get("embedding_model", "BAAI/bge-large-en-v1.5"))
            prompt_tpl = st.text_area("RAG System Prompt Template", value=config.get("system_prompt", ""), height=180)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            save_prompts = st.form_submit_button("Save Pipeline Templates")
            
            if save_prompts:
                config["embedding_model"] = emb_model
                config["system_prompt"] = prompt_tpl
                save_settings(config)
                st.success("✓ Prompt templates saved.")

        st.markdown(
            """
            <div style="background-color: rgba(59, 130, 246, 0.05); border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #94a3b8; margin-top: 16px;">
                💡 <b>Usage:</b> Make sure to include the tags <code>{context}</code> and <code>{question}</code> in your template text. The generator uses these placeholders to inject retrieved FAISS documents and user queries during the evaluation run.
            </div>
            """,
            unsafe_allow_html=True
        )
