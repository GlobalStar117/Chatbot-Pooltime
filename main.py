from config import load_env
from src.rag_chatbot import chat_with_rag
import streamlit as st

load_env()

st.title("💬 PoolTime.se - Assistant")
st.caption("🚀 A Streamlit chatbot for PoolTime.se")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are helpful assistant to assist user generating high quality contents."},
                                    {"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # msg = chat_with_rag(st.session_state.messages)
    msg = chat_with_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)