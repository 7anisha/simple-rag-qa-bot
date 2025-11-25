import streamlit as st
import os
from src.preprocess import generate_documents
from src.build_vectorstore import build_vectorstore
from src.rag_qa import answer

st.title("Simple RAG Q&A — Movie Plots")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("Set OPENAI_API_KEY in environment before running.")

if st.button("Generate docs (chunk dataset)"):
    generate_documents()
    st.success("Documents generated.")

if st.button("Build vectorstore"):
    build_vectorstore(openai_api_key=OPENAI_KEY)
    st.success("Vectorstore built.")

q = st.text_input("Ask a question about movies:")
if st.button("Ask"):
    if not q.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            ans = answer(q, api_key=OPENAI_KEY)
        st.subheader("Answer")
        st.write(ans)
